#!/usr/bin/env Rscript
#
# 05_deseq2_pseudobulk.R
# ================================================================================
# Pseudobulk differential expression for AREG, EGFR, GPR37 (and optionally other
# genes) in the Macnair et al. MS white-matter snRNA-seq dataset.
#
# Method:
#   1. Read the raw MatrixMarket counts matrix + coldata + rowdata from Zenodo.
#   2. Filter to WM cells with exclude_pseudobulk == FALSE.
#   3. Aggregate raw counts to pseudobulk per donor (individual_id_anon) x
#      cell type (type_broad) x lesion_type.
#   4. For each cell type, run DESeq2 with design ~ lesion_type.
#   5. Extract 9 pairwise contrasts.
#   6. Compute padj (BH within each contrast — equals raw p when 1 gene) and
#      padj_global (BH across all tests for a given gene).
#   7. Handle convergence failures gracefully (skip and log).
#
# Output:
#   A single CSV (or one per gene group) with columns matching the existing
#   table4/table12 format:
#     gene, cell_type, comparison, test_group, ref_group, n_test, n_ref,
#     log2FC, pvalue, padj, mean_test_per_cell, mean_ref_per_cell, padj_global
#
# Usage:
#   Rscript 05_deseq2_pseudobulk.R \
#       --data-dir /path/to/raw \
#       --out-dir  /path/to/results \
#       --genes    AREG EGFR GPR37
#
# Requirements: R >= 4.0 with DESeq2, Matrix, data.table, argparse (or manual parsing).
#
# Runtime: ~5-10 min (pseudobulk matrices are small; the main cost is reading
#          the 4.5 GB .mtx file).
# Memory:  ~16-32 GB (the full sparse matrix is loaded via Matrix::readMM).
#          If memory is tight, pre-extract focus genes with the Python step-01
#          script and adapt this script to read the parquet instead.
#
# IMPORTANT NOTE on padj_global:
#   The original analysis used a custom multiple-testing correction for
#   padj_global that could not be exactly reproduced from the stored values.
#   This script uses standard BH (p.adjust method="BH") per gene across all
#   available cell_type x comparison tests. This produces significance calls
#   that are ~93% concordant with the original for AREG (2 borderline
#   comparisons may cross the 0.05 threshold differently) and 100% concordant
#   for EGFR and GPR37 (0 significant either way).
#   The pre-computed CSVs (table4, table12) are provided as the authoritative
#   source for figure asterisks. Use this script to reproduce the analysis
#   from scratch; use the CSVs for the figures.
# ================================================================================

## ---- Parse arguments (base R, no argparse dependency) ----------------------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- "."
out_dir  <- "."
genes    <- c("AREG", "EGFR", "GPR37")

i <- 1
while (i <= length(args)) {
  if (args[i] == "--data-dir") { data_dir <- args[i+1]; i <- i + 2 }
  else if (args[i] == "--out-dir") { out_dir <- args[i+1]; i <- i + 2 }
  else if (args[i] == "--genes") {
    genes <- c()
    i <- i + 1
    while (i <= length(args) && !startsWith(args[i], "--")) {
      genes <- c(genes, args[i]); i <- i + 1
    }
  } else { i <- i + 1 }
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

cat("================================================================\n")
cat("DESeq2 pseudobulk DE — Macnair MS WM snRNA-seq\n")
cat("================================================================\n")
cat("Data dir:", data_dir, "\n")
cat("Out dir: ", out_dir, "\n")
cat("Genes:   ", paste(genes, collapse = ", "), "\n\n")

## ---- Load packages ---------------------------------------------------------
suppressPackageStartupMessages({
  library(Matrix)
  library(DESeq2)
})

## ---- Load data -------------------------------------------------------------
# File names follow the Zenodo convention (renamed for pipeline use).
mtx_file    <- file.path(data_dir, "macnair_counts.mtx.gz")
coldata_file <- file.path(data_dir, "macnair_coldata.txt.gz")
rowdata_file <- file.path(data_dir, "macnair_rowdata.txt.gz")

for (f in c(mtx_file, coldata_file, rowdata_file)) {
  if (!file.exists(f)) stop("Missing required file: ", f)
}

cat("Reading row data ...\n")
rowdata <- read.csv(rowdata_file, stringsAsFactors = FALSE)
# Columns: gene_id, ensembl, symbol
cat("  Genes in matrix:", nrow(rowdata), "\n")

cat("Reading col data ...\n")
coldata <- read.csv(coldata_file, stringsAsFactors = FALSE)
cat("  Cells in matrix:", nrow(coldata), "\n")

cat("Reading counts matrix (this may take several minutes) ...\n")
counts <- readMM(gzfile(mtx_file))
cat("  Matrix dims:", nrow(counts), "x", ncol(counts), "\n")

# Assign row/col names
rownames(counts) <- rowdata$symbol
colnames(counts) <- as.character(seq_len(ncol(counts)) - 1L)  # 0-indexed cell_idx

## ---- Filter to WM, exclude pseudobulk-flagged cells ------------------------
wm <- coldata$matter == "WM" & !coldata$exclude_pseudobulk
cat("WM cells (exclude_pseudobulk == FALSE):", sum(wm), "\n")

coldata_wm <- coldata[wm, ]
counts_wm  <- counts[, wm]

## ---- Cell types to test ----------------------------------------------------
# These 4 broad cell types have sufficient donors for DESeq2.
cell_types <- c("Astrocytes", "Microglia", "OPCs + COPs", "Oligodendrocytes")

## ---- Contrasts -------------------------------------------------------------
# 9 pairwise comparisons (matching the original analysis).
contrasts <- list(
  list(test = "AL",  ref = "WM",  name = "AL_vs_WM"),
  list(test = "CAL", ref = "WM",  name = "CAL_vs_WM"),
  list(test = "CIL", ref = "WM",  name = "CIL_vs_WM"),
  list(test = "RL",  ref = "WM",  name = "RL_vs_WM"),
  list(test = "AL",  ref = "CAL", name = "AL_vs_CAL"),
  list(test = "AL",  ref = "CIL", name = "AL_vs_CIL"),
  list(test = "CIL", ref = "NAWM", name = "CIL_vs_NAWM"),
  list(test = "RL",  ref = "CAL", name = "RL_vs_CAL"),
  list(test = "RL",  ref = "CIL", name = "RL_vs_CIL")
)

## ---- Aggregate to pseudobulk ----------------------------------------------
# Sum counts per donor x cell_type x lesion_type.
cat("\nAggregating to pseudobulk ...\n")

# Build a grouping factor: donor:celltype:lesion
group_key <- paste(coldata_wm$individual_id_anon,
                   coldata_wm$type_broad,
                   coldata_wm$lesion_type, sep = "||")
group_factor <- factor(group_key)

# Aggregate by summing columns within each group.
# This is memory-efficient for sparse matrices: split cols by group, sum each.
group_levels <- levels(group_factor)
n_groups <- length(group_levels)
cat("  Pseudobulk samples:", n_groups, "\n")

# Parse group labels back into metadata
group_meta <- do.call(rbind, strsplit(group_levels, "\\|\\|"))
colnames(group_meta) <- c("donor", "cell_type", "lesion_type")
group_meta <- as.data.frame(group_meta, stringsAsFactors = FALSE)

# Sum counts per group (focus genes only to save memory)
gene_idx <- which(rownames(counts_wm) %in% genes)
cat("  Focus gene rows:", length(gene_idx), "\n")
counts_focus <- counts_wm[gene_idx, , drop = FALSE]

# Aggregate: for each group, sum the columns
pb_counts <- matrix(0L, nrow = length(gene_idx), ncol = n_groups)
rownames(pb_counts) <- rownames(counts_focus)
for (g in seq_along(group_levels)) {
  cols_in_g <- which(group_factor == group_levels[g])
  if (length(cols_in_g) > 0) {
    pb_counts[, g] <- as.integer(rowSums(counts_focus[, cols_in_g, drop = FALSE]))
  }
}
cat("  Pseudobulk matrix:", nrow(pb_counts), "genes x", ncol(pb_counts), "samples\n")

## ---- Run DESeq2 per cell type ----------------------------------------------
cat("\nRunning DESeq2 per cell type ...\n")

all_results <- list()

for (ct in cell_types) {
  cat("\n  ---", ct, "---\n")

  # Select samples for this cell type
  ct_samples <- which(group_meta$cell_type == ct)
  if (length(ct_samples) < 4) {
    cat("    Skipping: only", length(ct_samples), "samples\n")
    next
  }

  # Subset pseudobulk
  pb_ct <- pb_counts[, ct_samples, drop = FALSE]
  meta_ct <- group_meta[ct_samples, ]

  # Need at least 2 donors per lesion type for a comparison
  lesion_counts <- table(meta_ct$lesion_type)
  cat("    Lesion counts:", paste(names(lesion_counts), lesion_counts, sep = "=", collapse = ", "), "\n")

  # Build DESeq2 object
  dds <- tryCatch({
    DESeqDataSetFromMatrix(
      countData = pb_ct,
      colData   = meta_ct,
      design    = ~ lesion_type
    )
  }, error = function(e) {
    cat("    ERROR creating DESeqDataSet:", conditionMessage(e), "\n")
    return(NULL)
  })
  if (is.null(dds)) next

  # Run DESeq2
  dds <- tryCatch({
    DESeq(dds, quiet = TRUE)
  }, error = function(e) {
    cat("    ERROR running DESeq:", conditionMessage(e), "\n")
    return(NULL)
  })
  if (is.null(dds)) next

  # Extract contrasts
  for (ctr in contrasts) {
    test_les <- ctr$test
    ref_les  <- ctr$ref
    comp_name <- ctr$name

    # Check both lesion types are present
    if (!(test_les %in% levels(dds$lesion_type)) ||
        !(ref_les %in% levels(dds$lesion_type))) {
      next
    }

    n_test <- sum(meta_ct$lesion_type == test_les)
    n_ref  <- sum(meta_ct$lesion_type == ref_les)
    if (n_test < 2 || n_ref < 2) {
      cat("    Skipping", comp_name, ": n_test=", n_test, "n_ref=", n_ref, "\n")
      next
    }

    res <- tryCatch({
      results(dds, contrast = c("lesion_type", test_les, ref_les),
              independentFiltering = FALSE)
    }, error = function(e) {
      cat("    ERROR for", comp_name, ":", conditionMessage(e), "\n")
      return(NULL)
    })
    if (is.null(res)) next

    for (g in genes) {
      if (!(g %in% rownames(res))) next
      r <- res[g, ]

      # Mean per-cell (raw counts / total cells, not just pseudobulk samples)
      test_cells <- coldata_wm$type_broad == ct & coldata_wm$lesion_type == test_les
      ref_cells  <- coldata_wm$type_broad == ct & coldata_wm$lesion_type == ref_les
      mean_test_per_cell <- mean(counts_wm[g, test_cells])
      mean_ref_per_cell  <- mean(counts_wm[g, ref_cells])

      all_results[[length(all_results) + 1]] <- data.frame(
        gene              = g,
        cell_type         = ct,
        comparison        = comp_name,
        test_group        = test_les,
        ref_group         = ref_les,
        n_test            = n_test,
        n_ref             = n_ref,
        log2FC            = r$log2FoldChange,
        pvalue            = r$pvalue,
        padj              = r$padj,   # BH within DESeq2 model (may be NA)
        mean_test_per_cell = mean_test_per_cell,
        mean_ref_per_cell  = mean_ref_per_cell,
        stringsAsFactors  = FALSE
      )
    }
    cat("    ", comp_name, ": OK\n")
  }
}

## ---- Combine and compute padj_global ---------------------------------------
cat("\nCombining results and computing padj_global ...\n")

results_df <- do.call(rbind, all_results)
if (nrow(results_df) == 0) {
  cat("No results generated. Check for errors above.\n")
  quit(save = "no", status = 1)
}

# padj: if NA (single gene in model), use pvalue
results_df$padj <- ifelse(is.na(results_df$padj), results_df$pvalue, results_df$padj)

# padj_global: BH across all tests for each gene
results_df$padj_global <- NA_real_
for (g in unique(results_df$gene)) {
  idx <- results_df$gene == g
  results_df$padj_global[idx] <- p.adjust(results_df$pvalue[idx], method = "BH")
}

# Sort
results_df <- results_df[order(results_df$gene, results_df$cell_type,
                               results_df$comparison), ]

## ---- Write output ----------------------------------------------------------
out_file <- file.path(out_dir, "deseq2_pseudobulk_DE_results.csv")
write.csv(results_df, out_file, row.names = FALSE)
cat("\nWrote", out_file, "(", nrow(results_df), "rows )\n")

# Summary
sig <- results_df[results_df$padj_global < 0.05, ]
cat("Significant (padj_global < 0.05):", nrow(sig), "/", nrow(results_df), "\n")
if (nrow(sig) > 0) {
  for (i in seq_len(nrow(sig))) {
    r <- sig[i, ]
    cat(sprintf("  %-6s %-20s %-12s log2FC=%+.2f padj_global=%.4f\n",
                r$gene, r$cell_type, r$comparison, r$log2FC, r$padj_global))
  }
}

cat("\nDONE.\n")
