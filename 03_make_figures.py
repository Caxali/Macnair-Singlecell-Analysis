#!/usr/bin/env python3
"""
STEP 03 - Generate the six figures.
================================================================================
Consumes ONLY the figure-input files produced by step 02
(fig_input_broad_grouped.csv, fig_input_fine_ol.csv) and renders the six
publication figures used in the reviewer response:

  % positive cells (prevalence)          mean count in positive cells (intensity)   dot plots (size=%pos, colour=intensity)
  ------------------------------------   -----------------------------------------   ---------------------------------------
  fig_macnair_egfr_areg_heatmaps         fig_macnair_egfr_areg_mean_heatmaps         fig_macnair_egfr_areg_dotplots
  fig_macnair_gpr37_psap_heatmaps        fig_macnair_gpr37_psap_mean_heatmaps        fig_macnair_gpr37_psap_dotplots

Each figure has 4 panels: {gene1 broad, gene1 fine-OL, gene2 broad, gene2 fine-OL}.
Every figure is saved as both .svg (editable text) and .png (300 dpi).

Design notes (two rendering fixes carried over from the audit):
  * "mean in positive cells" heatmaps show 'NA' (dark grey), not '0', for cells
    with no expressing cells: a per-cell mean over zero expressers is UNDEFINED,
    and positive cells always have count >= 1, so a literal 0 is impossible.
  * dot-plot size legend levels adapt to each panel's pct_max and bubbles are
    capped at the largest data dot, so sparse genes (e.g. AREG) no longer get a
    giant reference circle overflowing the axes.

Usage:
  python 03_make_figures.py --in-dir /path/to/intermediate --fig-dir /path/to/figures
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# User preference: editable-text SVG, Liberation Sans (Arial-metric).
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"

WM_LESIONS = ["WM", "NAWM", "AL", "CAL", "CIL", "RL"]
OL_FINE = ["OPC_1", "OPC_2", "COP", "Oligo_A", "Oligo_B", "Oligo_C",
           "Oligo_D", "Oligo_E", "Oligo_F", "Oligo_G", "Oligo_Astro"]
BROAD_ORDER = ["OPCs+COPs", "Oligodendrocytes", "Astrocytes", "Microglia", "Neurons", "Endothelial"]

LESION_LABELS = {
    "WM": "WM\n(control)", "NAWM": "NAWM", "AL": "AL\n(active)",
    "CAL": "CAL\n(chronic\nactive)", "CIL": "CIL\n(chronic\ninactive)",
    "RL": "RL\n(remyelin.)",
}
FINE_LABELS = {
    "OPC_1": "OPC_1", "OPC_2": "OPC_2", "COP": "COP",
    "Oligo_A": "Oligo_A\n(immature)", "Oligo_B": "Oligo_B\n(immature)",
    "Oligo_C": "Oligo_C\n(immature)", "Oligo_D": "Oligo_D\n(mature)",
    "Oligo_E": "Oligo_E\n(mature)", "Oligo_F": "Oligo_F\n(mature)",
    "Oligo_G": "Oligo_G\n(mature)", "Oligo_Astro": "Oligo_\nAstro",
}

# Per-gene colour ceilings (chosen in audit so no value is clipped).
VMAX_PCT = {  # % positive
    "EGFR": {"broad": 70, "fine": 75}, "AREG": {"broad": 16, "fine": 1},
    "GPR37": {"broad": 70, "fine": 85}, "PSAP": {"broad": 85, "fine": 85},
}
VMAX_MEAN = {  # mean count per positive cell
    "EGFR": {"broad": 8, "fine": 8}, "AREG": {"broad": 6, "fine": 3},
    "GPR37": {"broad": 4, "fine": 4}, "PSAP": {"broad": 12, "fine": 6},
}
PCT_MAX_DOT = {  # dot-size reference
    "EGFR": {"broad": 70, "fine": 75}, "AREG": {"broad": 16, "fine": 1},
    "GPR37": {"broad": 70, "fine": 85}, "PSAP": {"broad": 85, "fine": 85},
}


def pivot(df, gene, celltypes, cell_col, value_col):
    sub = df[df["gene"] == gene]
    p = sub.pivot_table(index=cell_col, columns="lesion_type", values=value_col)
    return p.reindex(index=celltypes, columns=WM_LESIONS).values


def draw_pct_heatmap(ax, mat, row_labels, col_labels, title, vmax, cmap="YlOrRd",
                     cbar_label="% positive cells"):
    """% positive: NaN -> 'NA' (no cells of that type in that lesion)."""
    disp = np.nan_to_num(mat, nan=0.0)
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="gray")
                continue
            txt = f"{v:.1f}" if v >= 0.1 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if v >= vmax * 0.5 else "black")
    _axes_labels(ax, row_labels, col_labels, title)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cb.set_label(cbar_label, fontsize=8); cb.ax.tick_params(labelsize=7)


def draw_mean_heatmap(ax, mat, row_labels, col_labels, title, vmax, cmap="YlOrRd",
                      cbar_label="Mean count\nper pos. cell"):
    """Mean-in-positive: 0/NaN -> 'NA' (undefined; no expressing cells)."""
    disp = np.nan_to_num(mat, nan=0.0)
    im = ax.imshow(disp, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v) or v == 0:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="darkgray")
                continue
            txt = f"{v:.2f}" if v < 1 else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if v >= vmax * 0.5 else "black")
    _axes_labels(ax, row_labels, col_labels, title)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cb.set_label(cbar_label, fontsize=8); cb.ax.tick_params(labelsize=7)


def _axes_labels(ax, row_labels, col_labels, title):
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)


def _nice_legend_levels(pmax):
    """Rounded % levels spanning ~0..pmax (keeps dot-legend bubbles sensible)."""
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


def draw_dotplot(ax, df, gene, cell_keys, cell_labels, cell_col, title, pct_max, mean_max):
    """Size = % positive; colour = mean count in positive cells."""
    pct = pivot(df, gene, cell_keys, cell_col, "pct_pos")
    mean = pivot(df, gene, cell_keys, cell_col, "mean_count_expr")
    norm = Normalize(vmin=0, vmax=mean_max)
    cmap = plt.cm.YlOrRd
    for i in range(len(cell_keys)):
        for j in range(len(WM_LESIONS)):
            p, m = pct[i, j], mean[i, j]
            if np.isnan(p) or p == 0:
                ax.scatter(j, i, s=8, c="lightgrey", edgecolors="grey", linewidths=0.3, zorder=2)
                continue
            size = max(15, (p / pct_max) * 350)
            ax.scatter(j, i, s=size, c=[cmap(norm(m))], edgecolors="black", linewidths=0.4, zorder=2)
    ax.set_xticks(range(len(WM_LESIONS)))
    ax.set_xticklabels([LESION_LABELS[l] for l in WM_LESIONS], fontsize=8)
    ax.set_yticks(range(len(cell_keys)))
    ax.set_yticklabels(cell_labels, fontsize=8)
    ax.set_xlim(-0.5, len(WM_LESIONS) - 0.5)
    ax.set_ylim(len(cell_keys) - 0.5, -0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.grid(False)
    for pl in _nice_legend_levels(pct_max):
        s = min(350, max(15, (pl / pct_max) * 350))
        ax.scatter([], [], s=s, c="lightgrey", edgecolors="black", linewidths=0.4, label=f"{pl:g}%")
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), title="% positive",
              fontsize=7, title_fontsize=8, frameon=False, labelspacing=1.4,
              handletextpad=0.8, borderpad=0.6)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.08)
    cb.set_label("Mean count\nper pos. cell", fontsize=8); cb.ax.tick_params(labelsize=7)


def save(fig, fig_dir, stem):
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(fig_dir, f"{stem}.{ext}"), format=ext, dpi=300)
    plt.close(fig)
    print(f"  saved {stem}.svg / .png")


def _panel_grid(figsize, wspace=0.35, hspace=0.45):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=hspace, wspace=wspace,
                  left=0.08, right=0.95, top=0.92, bottom=0.06)
    return fig, gs


def make_pair_pct(broad, fine, g1, g2, fig_dir):
    fig, gs = _panel_grid((14, 10))
    for row, g in enumerate((g1, g2)):
        ax = fig.add_subplot(gs[row, 0])
        draw_pct_heatmap(ax, pivot(broad, g, BROAD_ORDER, "cell_type_grouped", "pct_pos"),
                         BROAD_ORDER, [LESION_LABELS[l] for l in WM_LESIONS],
                         f"{g} - Broad cell types", vmax=VMAX_PCT[g]["broad"])
        ax = fig.add_subplot(gs[row, 1])
        draw_pct_heatmap(ax, pivot(fine, g, OL_FINE, "cell_type_fine", "pct_pos"),
                         [FINE_LABELS[t] for t in OL_FINE], [LESION_LABELS[l] for l in WM_LESIONS],
                         f"{g} - OL lineage (fine)", vmax=VMAX_PCT[g]["fine"])
    fig.suptitle(f"Macnair et al. - {g1} and {g2} expression by cell type and WM lesion state\n"
                 "(% positive cells, snRNA-seq, 632,375 nuclei)",
                 fontsize=12, fontweight="bold", y=0.98)
    save(fig, fig_dir, f"fig_macnair_{g1.lower()}_{g2.lower()}_heatmaps")


def make_pair_mean(broad, fine, g1, g2, fig_dir):
    fig, gs = _panel_grid((14, 10))
    for row, g in enumerate((g1, g2)):
        ax = fig.add_subplot(gs[row, 0])
        draw_mean_heatmap(ax, pivot(broad, g, BROAD_ORDER, "cell_type_grouped", "mean_count_expr"),
                          BROAD_ORDER, [LESION_LABELS[l] for l in WM_LESIONS],
                          f"{g} - Broad cell types", vmax=VMAX_MEAN[g]["broad"])
        ax = fig.add_subplot(gs[row, 1])
        draw_mean_heatmap(ax, pivot(fine, g, OL_FINE, "cell_type_fine", "mean_count_expr"),
                          [FINE_LABELS[t] for t in OL_FINE], [LESION_LABELS[l] for l in WM_LESIONS],
                          f"{g} - OL lineage (fine)", vmax=VMAX_MEAN[g]["fine"])
    fig.suptitle(f"Macnair et al. - Mean expression in positive cells (intensity among expressers)\n"
                 f"{g1} and {g2} by cell type and WM lesion state",
                 fontsize=12, fontweight="bold", y=0.98)
    save(fig, fig_dir, f"fig_macnair_{g1.lower()}_{g2.lower()}_mean_heatmaps")


def make_pair_dot(broad, fine, g1, g2, fig_dir):
    fig, gs = _panel_grid((16, 10), wspace=0.5, hspace=0.5)
    fig.subplots_adjust(left=0.07, right=0.93)
    for row, g in enumerate((g1, g2)):
        ax = fig.add_subplot(gs[row, 0])
        draw_dotplot(ax, broad, g, BROAD_ORDER, BROAD_ORDER, "cell_type_grouped",
                     f"{g} - Broad cell types", pct_max=PCT_MAX_DOT[g]["broad"],
                     mean_max=VMAX_MEAN[g]["broad"])
        ax = fig.add_subplot(gs[row, 1])
        draw_dotplot(ax, fine, g, OL_FINE, [FINE_LABELS[t] for t in OL_FINE], "cell_type_fine",
                     f"{g} - OL lineage (fine)", pct_max=PCT_MAX_DOT[g]["fine"],
                     mean_max=VMAX_MEAN[g]["fine"])
    fig.suptitle(f"Macnair et al. - Dot plots: dot size = % positive, dot colour = mean expression\n"
                 f"{g1} and {g2} by cell type and WM lesion state",
                 fontsize=12, fontweight="bold", y=0.98)
    save(fig, fig_dir, f"fig_macnair_{g1.lower()}_{g2.lower()}_dotplots")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="Directory with step-02 figure inputs.")
    ap.add_argument("--fig-dir", required=True, help="Output directory for figures.")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    print("=" * 72)
    print("STEP 03 - generate figures")
    print("=" * 72)

    import pandas as pd
    broad = pd.read_csv(os.path.join(args.in_dir, "fig_input_broad_grouped.csv"))
    fine = pd.read_csv(os.path.join(args.in_dir, "fig_input_fine_ol.csv"))
    # step-02 fine file uses 'type_fine'; figures expect 'cell_type_fine'
    if "type_fine" in fine.columns and "cell_type_fine" not in fine.columns:
        fine = fine.rename(columns={"type_fine": "cell_type_fine"})

    print("\n% positive heatmaps:")
    make_pair_pct(broad, fine, "EGFR", "AREG", args.fig_dir)
    make_pair_pct(broad, fine, "GPR37", "PSAP", args.fig_dir)

    print("\nmean-in-positive heatmaps:")
    make_pair_mean(broad, fine, "EGFR", "AREG", args.fig_dir)
    make_pair_mean(broad, fine, "GPR37", "PSAP", args.fig_dir)

    print("\ndot plots:")
    make_pair_dot(broad, fine, "EGFR", "AREG", args.fig_dir)
    make_pair_dot(broad, fine, "GPR37", "PSAP", args.fig_dir)

    print("\nSTEP 03 complete.")


if __name__ == "__main__":
    main()
