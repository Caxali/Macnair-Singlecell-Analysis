#!/usr/bin/env python3
"""
STEP 03 (3-gene) - Generate AREG / EGFR / GPR37 heatmap + dotplot figures.
================================================================================
Pipeline-integrated figure script.  Reads the step-02 figure-input files
(fig_input_broad_grouped.csv, fig_input_fine_ol.csv) produced by
02_build_tables.py, plus optional DESeq2 pseudobulk DE result CSVs, and renders:

  1.  % positive-cell heatmap  (3 genes x 2 panels: broad cell types + fine OL lineage)
      with DESeq2 significance asterisks on the broad panels
  2.  Dot plot                  (same layout; size = % positive, colour = mean count)
  3.  Companion DESeq2 stat table (all tests + significance flag)

Oligo_Astro is removed from the fine-OL panels (10 subtypes remain).
PSAP is NOT included — only AREG, EGFR, GPR37.

Usage
-----
    python 03_make_figures_3gene.py \
        --in-dir  ./intermediate \
        --fig-dir ./figures \
        --de-files  table4_pseudobulk_DE_results.csv  table12_pseudobulk_DE_new_genes.csv \
        --out-dir ./results \
        [--genes AREG EGFR GPR37]

If --de-files is omitted the heatmap is drawn without asterisks and no stat
table is written.

Requirements: Python 3 with pandas, numpy, matplotlib, Pillow (for verification).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Font: Liberation Sans (Arial-metric equivalent), editable SVG text ──
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42

# ── Constants ──
WM_LESIONS = ["WM", "NAWM", "AL", "CAL", "CIL", "RL"]
DEFAULT_GENES = ["AREG", "EGFR", "GPR37"]

# Broad cell types (order: OL lineage first, then other glia, then neurons/endo)
BROAD_ORDER = ["OPCs+COPs", "Oligodendrocytes", "Astrocytes",
               "Microglia", "Neurons", "Endothelial"]

# Fine OL lineage — Oligo_Astro REMOVED (10 subtypes)
OL_FINE = ["OPC_1", "OPC_2", "COP",
           "Oligo_A", "Oligo_B", "Oligo_C",
           "Oligo_D", "Oligo_E", "Oligo_F", "Oligo_G"]

LESION_LABELS = {
    "WM":   "WM\n(ctrl)",
    "NAWM": "NAWM",
    "AL":   "AL\n(active)",
    "CAL":  "CAL\n(chronic\nactive)",
    "CIL":  "CIL\n(chronic\ninactive)",
    "RL":   "RL\n(remyelin.)",
}
FINE_LABELS = {
    "OPC_1":    "OPC_1",
    "OPC_2":    "OPC_2",
    "COP":      "COP",
    "Oligo_A":  "Oligo_A\n(immature)",
    "Oligo_B":  "Oligo_B\n(immature)",
    "Oligo_C":  "Oligo_C\n(immature)",
    "Oligo_D":  "Oligo_D\n(mature)",
    "Oligo_E":  "Oligo_E\n(mature)",
    "Oligo_F":  "Oligo_F\n(mature)",
    "Oligo_G":  "Oligo_G\n(mature)",
}

# Per-gene colour ceilings (from audited pipeline — no value clipped)
VMAX_PCT = {   # % positive
    "AREG":  {"broad": 16, "fine": 1},
    "EGFR":  {"broad": 70, "fine": 75},
    "GPR37": {"broad": 70, "fine": 85},
}
VMAX_MEAN = {  # mean count per positive cell
    "AREG":  {"broad": 6,  "fine": 3},
    "EGFR":  {"broad": 8,  "fine": 8},
    "GPR37": {"broad": 4,  "fine": 4},
}
PCT_MAX_DOT = {  # dot-size reference
    "AREG":  {"broad": 16, "fine": 1},
    "EGFR":  {"broad": 70, "fine": 75},
    "GPR37": {"broad": 70, "fine": 85},
}

# DESeq2 cell-type name -> broad table name
DE_CT_MAP = {
    "OPCs + COPs":     "OPCs+COPs",
    "Astrocytes":      "Astrocytes",
    "Microglia":       "Microglia",
    "Oligodendrocytes": "Oligodendrocytes",
}

# WM-referenced comparisons that map to heatmap columns
WM_REF_LESIONS = ["AL", "CAL", "CIL", "RL"]


# ── Data loading ────────────────────────────────────────────────────────────
def load_tables(in_dir):
    """Load step-02 figure-input files, normalising column names."""
    broad_path = os.path.join(in_dir, "fig_input_broad_grouped.csv")
    fine_path  = os.path.join(in_dir, "fig_input_fine_ol.csv")
    for p in (broad_path, fine_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: expected step-02 output not found: {p}")

    broad = pd.read_csv(broad_path)
    fine  = pd.read_csv(fine_path)

    # Normalise fine-table cell-type column name
    if "type_fine" in fine.columns and "cell_type_fine" not in fine.columns:
        fine = fine.rename(columns={"type_fine": "cell_type_fine"})

    # Normalise mean-count column name (step-02 uses mean_count_expr;
    # some pre-computed tables use mean_count for the broad table)
    if "mean_count" in broad.columns and "mean_count_expr" not in broad.columns:
        broad = broad.rename(columns={"mean_count": "mean_count_expr"})

    # Remove Oligo_Astro from fine table if present
    if "Oligo_Astro" in fine["cell_type_fine"].unique():
        fine = fine[fine["cell_type_fine"] != "Oligo_Astro"].copy()

    return broad, fine


def load_deseq2(de_files):
    """Load and concatenate DESeq2 result CSVs.  Returns DataFrame or None."""
    if not de_files:
        return None
    frames = []
    for p in de_files:
        if not os.path.exists(p):
            print(f"  WARNING: DESeq2 file not found, skipping: {p}")
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        return None
    de = pd.concat(frames, ignore_index=True)
    de["cell_type_broad"] = de["cell_type"].map(DE_CT_MAP)
    return de


def build_sig_lookup(de, genes):
    """Build {(gene, broad_cell_type, lesion): padj_global} for WM-referenced
    comparisons (AL_vs_WM, CAL_vs_WM, CIL_vs_WM, RL_vs_WM)."""
    lookup = {}
    for _, row in de.iterrows():
        comp = row["comparison"]
        if "_vs_WM" not in comp:
            continue
        lesion = comp.replace("_vs_WM", "")
        if lesion not in WM_REF_LESIONS:
            continue
        if row["gene"] not in genes:
            continue
        if pd.isna(row.get("padj_global")):
            continue
        key = (row["gene"], row["cell_type_broad"], lesion)
        lookup[key] = row["padj_global"]
    return lookup


# ── Pivot helper ────────────────────────────────────────────────────────────
def pivot_table(df, gene, cell_col, cell_order, value_col):
    sub = df[df["gene"] == gene]
    p = sub.pivot_table(index=cell_col, columns="lesion_type", values=value_col)
    return p.reindex(index=cell_order, columns=WM_LESIONS).values


# ── Heatmap drawing ─────────────────────────────────────────────────────────
def draw_pct_heatmap(ax, mat, row_labels, col_labels, title, vmax,
                     sig_keys=None, sig_lookup=None, cmap="YlOrRd"):
    """% positive heatmap.  NaN -> 'NA' (grey).  Asterisks for DESeq2 sig."""
    disp = np.nan_to_num(mat, nan=0.0)
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "NA", ha="center", va="center",
                        fontsize=6, color="gray")
                continue
            txt = f"{v:.1f}" if v >= 0.1 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if v >= vmax * 0.5 else "black")
            # Asterisk if significant (broad panels only)
            if sig_keys is not None and sig_lookup is not None:
                lesion = WM_LESIONS[j]
                if lesion in WM_REF_LESIONS:
                    key = (sig_keys[0], sig_keys[1][i], lesion)
                    if key in sig_lookup and sig_lookup[key] < 0.05:
                        ax.text(j, i - 0.35, "*", ha="center", va="center",
                                fontsize=8, color="#D55E00", fontweight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cb.set_label("% positive", fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ── Dotplot drawing ─────────────────────────────────────────────────────────
def _nice_legend_levels(pmax):
    if pmax <= 1.2:
        raw = [0.25, 0.5, 1.0]
    elif pmax <= 3:
        raw = [0.5, 1, 2, 3]
    elif pmax <= 6:
        raw = [1, 2, 4, 6]
    elif pmax <= 20:
        raw = [2, 5, 10, 15]
    else:
        raw = [10, 25, 50, 70]
    return [v for v in raw if v <= pmax * 1.01]


def draw_dotplot(ax, df, gene, cell_keys, cell_labels, cell_col,
                 title, pct_max, mean_max):
    """Size = % positive; colour = mean count in positive cells."""
    pct  = pivot_table(df, gene, cell_col, cell_keys, "pct_pos")
    mean = pivot_table(df, gene, cell_col, cell_keys, "mean_count_expr")

    norm = Normalize(vmin=0, vmax=mean_max)
    cmap = plt.cm.YlOrRd
    for i in range(len(cell_keys)):
        for j in range(len(WM_LESIONS)):
            p, m = pct[i, j], mean[i, j]
            if np.isnan(p) or p == 0:
                ax.scatter(j, i, s=8, c="lightgrey", edgecolors="grey",
                           linewidths=0.3, zorder=2)
                continue
            size = max(15, (p / pct_max) * 350)
            color_val = m if not np.isnan(m) else 0
            ax.scatter(j, i, s=size, c=[cmap(norm(color_val))],
                       edgecolors="black", linewidths=0.4, zorder=2)
    ax.set_xticks(range(len(WM_LESIONS)))
    ax.set_xticklabels([LESION_LABELS[l] for l in WM_LESIONS], fontsize=7)
    ax.set_yticks(range(len(cell_keys)))
    ax.set_yticklabels(cell_labels, fontsize=7)
    ax.set_xlim(-0.5, len(WM_LESIONS) - 0.5)
    ax.set_ylim(len(cell_keys) - 0.5, -0.5)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.grid(False)
    # Size legend
    for pl in _nice_legend_levels(pct_max):
        s = min(350, max(15, (pl / pct_max) * 350))
        ax.scatter([], [], s=s, c="lightgrey", edgecolors="black",
                   linewidths=0.4, label=f"{pl:g}%")
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.0), title="% positive",
              fontsize=6, title_fontsize=7, frameon=False, labelspacing=1.2,
              handletextpad=0.8, borderpad=0.4)
    # Colour bar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.08)
    cb.set_label("Mean count\nper pos. cell", fontsize=7)
    cb.ax.tick_params(labelsize=6)


# ── Figure builders ─────────────────────────────────────────────────────────
def make_heatmap(broad, fine, genes, sig_lookup, fig_dir):
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(len(genes), 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.07, right=0.95, top=0.93, bottom=0.04)

    for row, gene in enumerate(genes):
        # Broad panel (left)
        ax_b = fig.add_subplot(gs[row, 0])
        mat_b = pivot_table(broad, gene, "cell_type_grouped",
                            BROAD_ORDER, "pct_pos")
        draw_pct_heatmap(ax_b, mat_b, BROAD_ORDER,
                         [LESION_LABELS[l] for l in WM_LESIONS],
                         f"{gene} — Broad cell types",
                         vmax=VMAX_PCT[gene]["broad"],
                         sig_keys=(gene, BROAD_ORDER),
                         sig_lookup=sig_lookup)

        # Fine OL panel (right)
        ax_f = fig.add_subplot(gs[row, 1])
        mat_f = pivot_table(fine, gene, "cell_type_fine",
                            OL_FINE, "pct_pos")
        draw_pct_heatmap(ax_f, mat_f, [FINE_LABELS[t] for t in OL_FINE],
                         [LESION_LABELS[l] for l in WM_LESIONS],
                         f"{gene} — OL lineage (fine)",
                         vmax=VMAX_PCT[gene]["fine"],
                         sig_keys=None, sig_lookup=None)

    gene_str = ", ".join(genes)
    fig.suptitle(
        f"Macnair et al. — {gene_str} expression by cell type and WM lesion state\n"
        "(% positive cells, snRNA-seq, 632,375 nuclei; "
        "* = DESeq2 padj_global < 0.05 vs WM control)",
        fontsize=11, fontweight="bold", y=0.98)

    stem = "fig_macnair_" + "_".join(g.lower() for g in genes) + "_heatmaps"
    _save(fig, fig_dir, stem)
    return stem


def make_dotplot(broad, fine, genes, fig_dir):
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(len(genes), 2, figure=fig, hspace=0.45, wspace=0.55,
                  left=0.06, right=0.92, top=0.93, bottom=0.04)

    for row, gene in enumerate(genes):
        # Broad panel (left)
        ax_b = fig.add_subplot(gs[row, 0])
        draw_dotplot(ax_b, broad, gene, BROAD_ORDER, BROAD_ORDER,
                     "cell_type_grouped",
                     f"{gene} — Broad cell types",
                     pct_max=PCT_MAX_DOT[gene]["broad"],
                     mean_max=VMAX_MEAN[gene]["broad"])

        # Fine OL panel (right)
        ax_f = fig.add_subplot(gs[row, 1])
        draw_dotplot(ax_f, fine, gene, OL_FINE,
                     [FINE_LABELS[t] for t in OL_FINE],
                     "cell_type_fine",
                     f"{gene} — OL lineage (fine)",
                     pct_max=PCT_MAX_DOT[gene]["fine"],
                     mean_max=VMAX_MEAN[gene]["fine"])

    gene_str = ", ".join(genes)
    fig.suptitle(
        f"Macnair et al. — Dot plots: dot size = % positive, "
        f"dot colour = mean expression\n"
        f"{gene_str} by cell type and WM lesion state "
        f"(snRNA-seq, 632,375 nuclei)",
        fontsize=11, fontweight="bold", y=0.98)

    stem = "fig_macnair_" + "_".join(g.lower() for g in genes) + "_dotplots"
    _save(fig, fig_dir, stem)
    return stem


def make_stat_table(de, genes, out_dir):
    """Companion DESeq2 stat table."""
    de_filt = de[de["gene"].isin(genes)].copy()
    cols = ["gene", "cell_type", "comparison", "test_group", "ref_group",
            "n_test", "n_ref", "log2FC", "pvalue", "padj", "padj_global"]
    # Keep only columns that exist
    cols = [c for c in cols if c in de_filt.columns]
    de_out = de_filt[cols].copy()
    de_out["significant"] = de_out["padj_global"] < 0.05
    de_out = de_out.sort_values(["gene", "cell_type", "comparison"])
    path = os.path.join(out_dir, "macnair_3gene_deseq2_stats.csv")
    de_out.to_csv(path, index=False)
    print(f"  saved macnair_3gene_deseq2_stats.csv ({len(de_out)} rows)")
    sig = de_out[de_out["significant"]]
    print(f"  Significant (padj_global < 0.05): {len(sig)} / {len(de_out)}")
    if len(sig) > 0:
        for _, r in sig.iterrows():
            print(f"    {r['gene']:6s} {r['cell_type']:20s} {r['comparison']:12s} "
                  f"log2FC={r['log2FC']:+.2f} padj_global={r['padj_global']:.4f}")
    return path


# ── Utilities ───────────────────────────────────────────────────────────────
def _save(fig, fig_dir, stem):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"),
                    format=ext, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.svg / .png")


def verify_fig(fig_dir, stem):
    """PIL-based check: dimensions + non-white fraction."""
    try:
        from PIL import Image
    except ImportError:
        print(f"  (Pillow not installed — skipping verification of {stem}.png)")
        return True
    path = os.path.join(fig_dir, f"{stem}.png")
    img = Image.open(path)
    w, h = img.size
    arr = np.array(img.convert("L"))
    non_white = (arr < 250).sum() / arr.size
    print(f"  Verified {stem}.png: {w}x{h}px, non-white fraction={non_white:.3f}")
    return non_white > 0.01


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True,
                    help="Directory with step-02 outputs "
                         "(fig_input_broad_grouped.csv, fig_input_fine_ol.csv).")
    ap.add_argument("--fig-dir", required=True,
                    help="Output directory for figures.")
    ap.add_argument("--de-files", nargs="*", default=None,
                    help="DESeq2 result CSVs (table4 + table12). "
                         "If omitted, no asterisks and no stat table.")
    ap.add_argument("--out-dir", default=".",
                    help="Directory for the companion stat table (default: cwd).")
    ap.add_argument("--genes", nargs="+", default=DEFAULT_GENES,
                    help=f"Genes to plot (default: {' '.join(DEFAULT_GENES)}).")
    args = ap.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 72)
    print(f"MacNair 3-gene figures: {', '.join(args.genes)}")
    print("=" * 72)

    # Load data
    broad, fine = load_tables(args.in_dir)
    # Filter to requested genes
    broad = broad[broad["gene"].isin(args.genes)].copy()
    fine  = fine[fine["gene"].isin(args.genes)].copy()

    print(f"Broad table: {len(broad)} rows, "
          f"genes: {sorted(broad['gene'].unique())}")
    print(f"Fine OL table (Oligo_Astro removed): {len(fine)} rows, "
          f"types: {sorted(fine['cell_type_fine'].unique())}")

    # Load DESeq2
    de = load_deseq2(args.de_files)
    sig_lookup = {}
    if de is not None:
        print(f"DESeq2 results: {len(de)} rows")
        sig_lookup = build_sig_lookup(de, args.genes)
        print(f"Significance lookup entries (WM-referenced): {len(sig_lookup)}")
    else:
        print("No DESeq2 files provided — heatmap will have no asterisks.")

    # Heatmap
    print("\n[1/3] Heatmap figure:")
    stem_h = make_heatmap(broad, fine, args.genes, sig_lookup, args.fig_dir)
    verify_fig(args.fig_dir, stem_h)

    # Dotplot
    print("\n[2/3] Dotplot figure:")
    stem_d = make_dotplot(broad, fine, args.genes, args.fig_dir)
    verify_fig(args.fig_dir, stem_d)

    # Stat table
    print("\n[3/3] Companion stat table:")
    if de is not None:
        make_stat_table(de, args.genes, args.out_dir)
    else:
        print("  Skipped (no DESeq2 files provided).")

    print("\nAll outputs generated.")


if __name__ == "__main__":
    main()
