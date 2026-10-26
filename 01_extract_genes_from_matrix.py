#!/usr/bin/env python3
"""
STEP 01 - Extract focus genes from the raw Macnair counts matrix.
================================================================================
Streams the raw MatrixMarket counts matrix (macnair_counts.mtx.gz) ONCE, keeping
only the rows (genes) of interest, and joins the per-cell counts to the cell
metadata (coldata). Produces a tidy long-format parquet of positive cells for the
focus genes plus a per-gene ground-truth summary JSON.

This is the FIRST stage of the pipeline and is fully self-contained: it depends
only on the three raw files distributed by Macnair et al. (Zenodo 8338963), not on
any previously derived intermediate. It is the same streaming logic used in the
data audit (zero discrepancies vs. the original derived parquet across ~1.47e9
non-zero entries).

Raw inputs (expected in --data-dir):
  macnair_counts.mtx.gz   MatrixMarket coordinate: 33,939 genes x 632,375 cells,
                          1,471,742,161 non-zero entries. After the header the
                          triplet lines are "row col value" (1-indexed);
                          rows = genes, cols = cells.
  macnair_rowdata.txt.gz  gene table (gene_id, ensembl, symbol) - one row per gene,
                          in matrix-row order. Used to resolve gene -> row index
                          (NOT hardcoded), so the script is robust to a different
                          gene order.
  macnair_coldata.txt.gz  cell metadata, comma-separated (despite .txt.gz), one row
                          per cell in matrix-column order. 29 metadata columns
                          including type_broad, type_fine, lesion_type, matter.

Outputs (written to --out-dir):
  focus_gene_counts_long.parquet   tidy long format, positive cells only, with
                                   metadata joined. Columns: gene, cell_idx (0-based),
                                   count, cell_id, sample_id_anon, individual_id_anon,
                                   type_fine, type_broad, lesion_type, matter,
                                   diagnosis, age_at_death, sex.
  focus_gene_raw_summary.json      per-gene ground-truth (n_positive_cells,
                                   total_counts) + matrix dims. Independent check.

Cell-index convention (verified in audit):
  matrix column (1-indexed) == cell_idx (0-indexed) + 1 == coldata row (0-indexed) + 1

Usage:
  python 01_extract_genes_from_matrix.py \
      --data-dir /path/to/raw \
      --out-dir  /path/to/intermediate \
      [--genes EGFR AREG GPR37 PSAP GPR37L1]

Runtime: ~15-20 min single-threaded (streams 1.47e9 lines at ~1.3-1.5M lines/s).
Memory:  low - only the target rows are held; the matrix is never fully loaded.
"""
import argparse
import gzip
import json
import os
import time

import pandas as pd

DEFAULT_GENES = ["EGFR", "AREG", "GPR37", "PSAP", "GPR37L1"]

# Metadata columns carried onto each positive cell (must exist in coldata).
META_COLS = [
    "cell_id", "sample_id_anon", "individual_id_anon",
    "type_fine", "type_broad", "lesion_type", "matter",
    "diagnosis", "age_at_death", "sex",
]


def resolve_gene_rows(rowdata_path, genes):
    """Return {matrix_row_1indexed: symbol} by reading the gene table.

    Row index is 1-based to match MatrixMarket coordinates. We look up the gene
    SYMBOL column rather than hardcoding indices, so a re-released matrix with a
    different gene order still works.
    """
    rd = pd.read_csv(rowdata_path, sep="\t")
    # Identify the symbol column robustly.
    symbol_col = None
    for cand in ["symbol", "Symbol", "gene_symbol", "gene_name", "name"]:
        if cand in rd.columns:
            symbol_col = cand
            break
    if symbol_col is None:
        # Fall back to the last column (rowdata layout: gene_id, ensembl, symbol).
        symbol_col = rd.columns[-1]
    sym = rd[symbol_col].astype(str)

    row_map = {}
    for g in genes:
        hits = sym.index[sym == g].tolist()
        if not hits:
            raise ValueError(f"Gene '{g}' not found in {symbol_col} column of {rowdata_path}")
        if len(hits) > 1:
            print(f"  WARNING: gene '{g}' appears {len(hits)} times; using first (row {hits[0] + 1}).")
        row_map[hits[0] + 1] = g  # +1 -> 1-indexed matrix row
    return row_map, symbol_col


def stream_extract(mtx_path, row_map):
    """Stream the matrix, collecting {gene: {cell_col_1indexed: count}}."""
    result = {g: {} for g in row_map.values()}
    t0 = time.time()
    header_done = False
    dims_line = None
    n_data = 0

    with gzip.open(mtx_path, "rt") as f:
        for line in f:
            if line.startswith("%"):  # comment / MatrixMarket banner
                continue
            if not header_done:
                dims_line = line.strip()  # "nrows ncols nnz"
                header_done = True
                continue
            n_data += 1
            parts = line.split()
            r = int(parts[0])
            if r in row_map:
                result[row_map[r]][int(parts[1])] = int(parts[2])
            if n_data % 100_000_000 == 0:
                el = time.time() - t0
                print(f"  processed {n_data:,} data lines in {el:.0f}s "
                      f"({n_data / el / 1e6:.1f}M lines/s)", flush=True)

    print(f"\nDONE streaming in {time.time() - t0:.0f}s")
    print(f"Dimensions line: {dims_line}")
    print(f"Total data lines processed: {n_data:,}\n")
    return result, dims_line


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Directory with the 3 raw Macnair files.")
    ap.add_argument("--out-dir", required=True, help="Directory for intermediate outputs.")
    ap.add_argument("--genes", nargs="+", default=DEFAULT_GENES,
                    help=f"Gene symbols to extract (default: {' '.join(DEFAULT_GENES)}).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mtx = os.path.join(args.data_dir, "macnair_counts.mtx.gz")
    rowdata = os.path.join(args.data_dir, "macnair_rowdata.txt.gz")
    coldata = os.path.join(args.data_dir, "macnair_coldata.txt.gz")
    for p in (mtx, rowdata, coldata):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required raw input: {p}")

    print("=" * 72)
    print("STEP 01 - raw extraction of focus genes from Macnair matrix")
    print("=" * 72)
    print(f"Genes: {', '.join(args.genes)}")

    # 1. Resolve gene -> matrix row (1-indexed) from the gene table.
    row_map, symbol_col = resolve_gene_rows(rowdata, args.genes)
    print(f"Resolved gene rows from '{symbol_col}' column:")
    for r, g in sorted(row_map.items()):
        print(f"  {g:<8} -> matrix row {r}")

    # 2. Stream the matrix, keeping only target rows.
    result, dims_line = stream_extract(mtx, row_map)

    # 3. Ground-truth summary per gene.
    summary = {}
    for g in args.genes:
        d = result[g]
        summary[g] = {"n_positive_cells": len(d), "total_counts": int(sum(d.values()))}
        print(f"  {g:<8}: {len(d):>8,} positive cells, {sum(d.values()):>11,} total counts")

    # 4. Load coldata, attach a 0-indexed cell_idx (matrix col == cell_idx + 1).
    print("\nLoading coldata and joining metadata ...")
    cd = pd.read_csv(coldata)  # comma-separated despite .txt.gz
    cd["cell_idx"] = range(len(cd))
    keep_meta = ["cell_idx"] + [c for c in META_COLS if c in cd.columns]
    missing_meta = [c for c in META_COLS if c not in cd.columns]
    if missing_meta:
        print(f"  NOTE: coldata is missing expected columns (skipped): {missing_meta}")

    # 5. Build tidy long frame of positive cells.
    frames = []
    for g in args.genes:
        d = result[g]
        if not d:
            continue
        df = pd.DataFrame({
            "gene": g,
            "cell_idx": [c - 1 for c in d.keys()],  # 1-indexed col -> 0-indexed cell_idx
            "count": list(d.values()),
        })
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.merge(cd[keep_meta], on="cell_idx", how="left")

    # 6. Write outputs.
    out_parquet = os.path.join(args.out_dir, "focus_gene_counts_long.parquet")
    out_json = os.path.join(args.out_dir, "focus_gene_raw_summary.json")
    long_df.to_parquet(out_parquet, index=False)
    with open(out_json, "w") as fh:
        json.dump({"dims": dims_line, "genes": args.genes, "summary": summary}, fh, indent=2)

    print(f"\nWrote long-format counts : {out_parquet}  ({len(long_df):,} rows)")
    print(f"Wrote ground-truth summary: {out_json}")
    print("\nSTEP 01 complete.")


if __name__ == "__main__":
    main()
