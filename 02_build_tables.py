#!/usr/bin/env python3
"""
STEP 02 - Build summary tables and figure-input files.
================================================================================
Consumes the tidy long-format parquet from step 01 (positive cells only, metadata
joined) plus the raw coldata (for denominators = ALL cells, including
zero-expressers), and produces the per-cell-type x lesion summary tables and the
exact aggregated inputs the figures consume.

All statistics are computed the same way as in the audited analysis:
  pct_pos          = 100 * (#positive cells) / (#all cells in that cell x lesion)
  mean_count_expr  = mean count among POSITIVE cells only (undefined -> NaN when
                     there are no positive cells; NOT 0)
  mean_count_all   = total_counts / (#all cells)   [dense mean]
Neurons and other composite broad groups are aggregated by SUMMING counts and
denominators, then RE-deriving pct_pos / mean_count_expr from the sums (never by
averaging percentages).

Analysis is restricted to white-matter (matter == 'WM') cells and the 6 WM lesion
states, matching the reviewer-response figures.

Cell-type groupings (broad -> grouped) used for the figures:
  OPCs + COPs                         -> OPCs+COPs
  Oligodendrocytes                    -> Oligodendrocytes
  Astrocytes                          -> Astrocytes
  Microglia                           -> Microglia
  Excitatory neurons + Inhibitory neurons -> Neurons
  Endo + Peri                         -> Endothelial

Outputs (written to --out-dir):
  table_by_lesion.csv          gene x lesion (all cell types pooled)
  table_by_broad_lesion.csv    gene x broad cell type x lesion (ungrouped)
  table_by_fine_lesion.csv     gene x fine cell type x lesion
  fig_input_broad_grouped.csv  gene x grouped broad type x lesion (figure input)
  fig_input_fine_ol.csv        gene x fine OL-lineage type x lesion (figure input)

Usage:
  python 02_build_tables.py --data-dir /path/to/raw \
                            --in-dir  /path/to/intermediate \
                            --out-dir /path/to/intermediate
"""
import argparse
import os

import numpy as np
import pandas as pd

WM_LESIONS = ["WM", "NAWM", "AL", "CAL", "CIL", "RL"]

OL_FINE = ["OPC_1", "OPC_2", "COP", "Oligo_A", "Oligo_B", "Oligo_C",
           "Oligo_D", "Oligo_E", "Oligo_F", "Oligo_G", "Oligo_Astro"]

# broad cell type -> grouped label for figures
BROAD_GROUP = {
    "Astrocytes": "Astrocytes",
    "Microglia": "Microglia",
    "Excitatory neurons": "Neurons",
    "Inhibitory neurons": "Neurons",
    "Endo + Peri": "Endothelial",
    "Oligodendrocytes": "Oligodendrocytes",
    "OPCs + COPs": "OPCs+COPs",
}


def _round(x, n):
    return round(x, n) if pd.notna(x) else x


def build_by_lesion(long_wm, denom_lt, genes):
    rows = []
    for gene in genes:
        gd = long_wm[long_wm["gene"] == gene]
        for lt in WM_LESIONS:
            total = int(denom_lt.get(lt, 0))
            ld = gd[gd["lesion_type"] == lt]
            n = len(ld)
            tot = int(ld["count"].sum())
            rows.append({
                "gene": gene, "lesion_type": lt,
                "total_cells": total, "n_cells_expr": n,
                "pct_pos": _round(100 * n / total, 3) if total else 0,
                "total_counts": tot,
                "mean_count_expr": _round(ld["count"].mean(), 3) if n else np.nan,
                "mean_count_all": _round(tot / total, 4) if total else np.nan,
            })
    return pd.DataFrame(rows)


def build_by_celltype(long_wm, denom, celltype_col, genes, celltypes=None):
    """Generic gene x cell-type x lesion table. denom is a Series indexed by
    (celltype, lesion_type) giving total cells."""
    if celltypes is None:
        celltypes = sorted({ct for (ct, _) in denom.index})
    rows = []
    for gene in genes:
        gd = long_wm[long_wm["gene"] == gene]
        for ct in celltypes:
            for lt in WM_LESIONS:
                total = int(denom.get((ct, lt), 0))
                ld = gd[(gd[celltype_col] == ct) & (gd["lesion_type"] == lt)]
                n = len(ld)
                tot = int(ld["count"].sum())
                rows.append({
                    "gene": gene, celltype_col: ct, "lesion_type": lt,
                    "total_cells": total, "n_cells_expr": n,
                    "pct_pos": _round(100 * n / total, 3) if total else 0,
                    "total_counts": tot,
                    "mean_count_expr": _round(ld["count"].mean(), 3) if n else np.nan,
                    "mean_count_all": _round(tot / total, 4) if total else np.nan,
                })
    return pd.DataFrame(rows)


def build_broad_grouped(long_wm, wm_coldata, genes):
    """Figure input: grouped broad cell types (neurons/endothelial combined by summing)."""
    # invert grouping -> members per group
    members = {}
    for broad, grp in BROAD_GROUP.items():
        members.setdefault(grp, []).append(broad)
    group_order = ["OPCs+COPs", "Oligodendrocytes", "Astrocytes", "Microglia", "Neurons", "Endothelial"]

    rows = []
    for gene in genes:
        gd = long_wm[long_wm["gene"] == gene]
        for grp in group_order:
            mem = members[grp]
            for lt in WM_LESIONS:
                total = int(wm_coldata[wm_coldata["type_broad"].isin(mem)
                                       & (wm_coldata["lesion_type"] == lt)].shape[0])
                ld = gd[gd["type_broad"].isin(mem) & (gd["lesion_type"] == lt)]
                n = len(ld)
                tot = int(ld["count"].sum())
                rows.append({
                    "gene": gene, "cell_type_grouped": grp, "lesion_type": lt,
                    "total_cells": total, "n_cells_expr": n, "total_counts": tot,
                    "pct_pos": _round(100 * n / total, 4) if total else 0,
                    "mean_count_expr": _round(tot / n, 4) if n else np.nan,
                    "mean_count_all": _round(tot / total, 4) if total else np.nan,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Directory with raw coldata.")
    ap.add_argument("--in-dir", required=True, help="Directory with step-01 output.")
    ap.add_argument("--out-dir", required=True, help="Directory for table outputs.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 72)
    print("STEP 02 - build tables and figure inputs")
    print("=" * 72)

    long_df = pd.read_parquet(os.path.join(args.in_dir, "focus_gene_counts_long.parquet"))
    genes = sorted(long_df["gene"].unique().tolist())
    print(f"Genes in long file: {', '.join(genes)}")

    coldata = pd.read_csv(os.path.join(args.data_dir, "macnair_coldata.txt.gz"))
    coldata["cell_idx"] = range(len(coldata))

    # Restrict to WM.
    wm_coldata = coldata[coldata["matter"] == "WM"].copy()
    long_wm = long_df[long_df["matter"] == "WM"].copy()
    print(f"WM cells (denominator pool): {len(wm_coldata):,}")
    print(f"WM positive-cell rows: {len(long_wm):,}")

    # Denominators (ALL cells incl. zero-expressers).
    denom_lt = wm_coldata.groupby("lesion_type").size()
    denom_broad = wm_coldata.groupby(["type_broad", "lesion_type"]).size()
    denom_fine = wm_coldata.groupby(["type_fine", "lesion_type"]).size()

    # Tables.
    t_lesion = build_by_lesion(long_wm, denom_lt, genes)
    t_broad = build_by_celltype(long_wm, denom_broad, "type_broad", genes)
    t_fine = build_by_celltype(long_wm, denom_fine, "type_fine", genes)

    # Figure inputs.
    fig_broad = build_broad_grouped(long_wm, wm_coldata, genes)
    fig_fine_ol = t_fine[t_fine["type_fine"].isin(OL_FINE)].copy()
    # order OL rows by lineage progression
    fig_fine_ol["type_fine"] = pd.Categorical(fig_fine_ol["type_fine"], categories=OL_FINE, ordered=True)
    fig_fine_ol = fig_fine_ol.sort_values(["gene", "type_fine", "lesion_type"])

    # Write.
    def w(df, name):
        p = os.path.join(args.out_dir, name)
        df.to_csv(p, index=False)
        print(f"  wrote {name:<32} ({len(df):>4} rows)")

    print("\nWriting tables and figure inputs:")
    w(t_lesion, "table_by_lesion.csv")
    w(t_broad, "table_by_broad_lesion.csv")
    w(t_fine, "table_by_fine_lesion.csv")
    w(fig_broad, "fig_input_broad_grouped.csv")
    w(fig_fine_ol, "fig_input_fine_ol.csv")

    print("\nSTEP 02 complete.")


if __name__ == "__main__":
    main()
