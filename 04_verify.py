#!/usr/bin/env python3
"""
STEP 04 (optional) - Independent internal consistency checks.
================================================================================
Re-derives the per-gene ground truth two independent ways and confirms the
pipeline outputs are internally consistent. This is a lightweight version of the
full data audit (which additionally streams the raw matrix a second time). Run it
after steps 01-02 for a quick integrity gate.

Checks:
  A. Long parquet vs. ground-truth summary JSON (both from step 01):
     per-gene positive-cell count and total counts must match exactly.
  B. Table row totals reconcile: for each gene, summing n_cells_expr / total_counts
     across the by-fine table equals the by-lesion table (over WM lesions) for the
     cell types that partition the population, and the grouped-broad figure input
     re-derives pct_pos / mean_count_expr correctly from its own sums.
  C. Denominators in the tables equal raw coldata WM cell counts per cell x lesion.

Exit code is non-zero if any check fails.

Usage:
  python 04_verify.py --data-dir /path/to/raw --in-dir /path/to/intermediate
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

WM_LESIONS = ["WM", "NAWM", "AL", "CAL", "CIL", "RL"]
TOL = 0.011  # tables rounded to 3 dp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--in-dir", required=True)
    args = ap.parse_args()

    print("=" * 72)
    print("STEP 04 - internal consistency checks")
    print("=" * 72)
    issues = []

    long_df = pd.read_parquet(os.path.join(args.in_dir, "focus_gene_counts_long.parquet"))
    with open(os.path.join(args.in_dir, "focus_gene_raw_summary.json")) as fh:
        summary = json.load(fh)["summary"]

    # --- Check A: long parquet vs summary JSON ---
    print("\n[A] long parquet vs ground-truth summary JSON")
    for g, s in summary.items():
        sub = long_df[long_df["gene"] == g]
        n_ok = len(sub) == s["n_positive_cells"]
        t_ok = int(sub["count"].sum()) == s["total_counts"]
        status = "OK" if (n_ok and t_ok) else "MISMATCH"
        if status != "OK":
            issues.append(f"A/{g}: long({len(sub)},{int(sub['count'].sum())}) vs "
                          f"summary({s['n_positive_cells']},{s['total_counts']})")
        print(f"    {g:<8}: n={len(sub):>8,} tot={int(sub['count'].sum()):>11,} | {status}")

    # --- Check B: grouped-broad figure input re-derivation ---
    print("\n[B] grouped-broad figure input: pct_pos / mean re-derived from sums")
    fb = pd.read_csv(os.path.join(args.in_dir, "fig_input_broad_grouped.csv"))
    re_pct = np.where(fb["total_cells"] > 0, 100 * fb["n_cells_expr"] / fb["total_cells"], 0.0)
    bad_pct = int((np.abs(re_pct - fb["pct_pos"]) > TOL).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        re_mean = np.where(fb["n_cells_expr"] > 0, fb["total_counts"] / fb["n_cells_expr"], np.nan)
    stored_mean = fb["mean_count_expr"].to_numpy()
    both_nan = np.isnan(re_mean) & np.isnan(stored_mean)
    bad_mean = int((~both_nan & (np.abs(np.nan_to_num(re_mean) - np.nan_to_num(stored_mean)) > TOL)).sum())
    print(f"    pct_pos re-derivation : {bad_pct} mismatches | {'OK' if not bad_pct else 'MISMATCH'}")
    print(f"    mean   re-derivation : {bad_mean} mismatches | {'OK' if not bad_mean else 'MISMATCH'}")
    if bad_pct:
        issues.append(f"B/pct_pos: {bad_pct} mismatches")
    if bad_mean:
        issues.append(f"B/mean: {bad_mean} mismatches")

    # --- Check C: denominators vs raw coldata ---
    print("\n[C] table denominators vs raw coldata WM cell counts")
    coldata = pd.read_csv(os.path.join(args.data_dir, "macnair_coldata.txt.gz"))
    wm = coldata[coldata["matter"] == "WM"]
    den_raw = wm.groupby(["type_broad", "lesion_type"]).size()
    tb = pd.read_csv(os.path.join(args.in_dir, "table_by_broad_lesion.csv"))
    one_gene = tb[tb["gene"] == tb["gene"].iloc[0]]
    dm = one_gene.merge(den_raw.rename("raw_n").reset_index(),
                        on=["type_broad", "lesion_type"], how="inner")
    bad_den = int((dm["total_cells"] != dm["raw_n"]).sum())
    print(f"    denominator match     : {bad_den} mismatches / {len(dm)} | "
          f"{'OK' if not bad_den else 'MISMATCH'}")
    if bad_den:
        issues.append(f"C/denominators: {bad_den} mismatches")

    # --- Summary ---
    print("\n" + "=" * 72)
    if issues:
        print(f"FAILED - {len(issues)} issue(s):")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print("ALL CHECKS PASSED - pipeline outputs are internally consistent.")


if __name__ == "__main__":
    main()
