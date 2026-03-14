#!/usr/bin/env Rscript
# analysis/R/plot_de.R
# Generates publication-quality figures for the airway DESeq2 analysis.
#
# Usage (from repository root):
#   Rscript analysis/R/plot_de.R

library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(patchwork)
library(scales)
library(pheatmap)
library(RColorBrewer)
library(tibble)

# ── Paths ──────────────────────────────────────────────────────
RESULTS_DIR <- "results"
COUNTS_FILE <- file.path(RESULTS_DIR, "rnaseq/airway/star_salmon/salmon.merged.gene_counts.tsv")
TPM_FILE    <- file.path(RESULTS_DIR, "rnaseq/airway/star_salmon/salmon.merged.gene_tpm.tsv")
VST_FILE    <- file.path(RESULTS_DIR, "differentialabundance/airway/tables/processed_abundance/all.vst.tsv")
DE_FILE     <- file.path(RESULTS_DIR, "differentialabundance/airway/tables/differential/dex_vs_untrt.deseq2.results.tsv")
PLOTS_DIR   <- "analysis/plots"
dir.create(PLOTS_DIR, showWarnings = FALSE, recursive = TRUE)

# ── Plot dimensions ────────────────────────────────────────────
W <- 8
H <- 5
DPI <- 300

# ── Load data ──────────────────────────────────────────────────
counts <- read_tsv(COUNTS_FILE, show_col_types = FALSE)
vst    <- read_tsv(VST_FILE, show_col_types = FALSE)
de     <- read_tsv(DE_FILE, show_col_types = FALSE)

# Gene ID to gene name mapping
gene_map <- counts |> select(gene_id, gene_name)

# Add gene names to DE results
de <- de |>
  left_join(gene_map, by = "gene_id") |>
  mutate(
    sig = case_when(
      is.na(padj) ~ "NS",
      padj >= 0.05 ~ "NS",
      log2FoldChange > 0 ~ "Up",
      log2FoldChange < 0 ~ "Down"
    ) |> factor(levels = c("Up", "Down", "NS"))
  )

# Sample metadata
sample_meta <- tibble(
  sample = c("N61311_untrt", "N61311_dex", "N052611_untrt", "N052611_dex",
             "N080611_untrt", "N080611_dex", "N061011_untrt", "N061011_dex"),
  condition = rep(c("Untreated", "Dexamethasone"), 4),
  cell_line = rep(c("N61311", "N052611", "N080611", "N061011"), each = 2)
)

# ── Custom theme ───────────────────────────────────────────────
COND_FILL <- c(Untreated = "#2171B5", Dexamethasone = "#CB181D")

theme_de <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      strip.background   = element_blank(),
      strip.text         = element_text(face = "bold"),
      legend.position    = "right",
      panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.3),
      plot.title         = element_text(face = "bold", size = base_size + 1),
      plot.subtitle      = element_text(colour = "grey40", size = base_size - 1)
    )
}

# ── Figure 1: PCA ──────────────────────────────────────────────
vst_mat <- vst |>
  column_to_rownames("gene_id") |>
  as.matrix()

# Compute PCA on top 500 most variable genes
rv <- apply(vst_mat, 1, var)
top500 <- names(sort(rv, decreasing = TRUE))[1:500]
pca <- prcomp(t(vst_mat[top500, ]), scale. = TRUE)
pca_df <- as.data.frame(pca$x[, 1:2]) |>

tibble::rownames_to_column("sample") |>
  left_join(sample_meta, by = "sample")

pct_var <- round(100 * summary(pca)$importance[2, 1:2], 1)

p_pca <- ggplot(pca_df, aes(PC1, PC2, colour = condition, shape = cell_line)) +
  geom_point(size = 4) +
  scale_colour_manual(values = COND_FILL) +
  labs(
    title = "PCA of VST-normalised expression",
    subtitle = "Top 500 most variable genes",
    x = paste0("PC1 (", pct_var[1], "% variance)"),
    y = paste0("PC2 (", pct_var[2], "% variance)"),
    colour = "Condition",
    shape = "Cell line"
  ) +
  theme_de()

ggsave(file.path(PLOTS_DIR, "fig1_pca.png"), p_pca,
       width = W, height = H, dpi = DPI, bg = "white")

# ── Figure 2: Volcano plot ────────────────────────────────────
n_up   <- sum(de$sig == "Up",   na.rm = TRUE)
n_down <- sum(de$sig == "Down", na.rm = TRUE)

# Label top genes
top_genes <- de |>
  filter(sig != "NS") |>
  arrange(padj) |>
  slice_head(n = 15)

p_volcano <- ggplot(de |> filter(!is.na(padj)),
                     aes(log2FoldChange, -log10(padj), colour = sig)) +
  geom_point(alpha = 0.5, size = 1) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", colour = "grey50") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", colour = "grey50") +
  ggrepel::geom_text_repel(
    data = top_genes,
    aes(label = gene_name),
    size = 3, max.overlaps = 20, colour = "black"
  ) +
  scale_colour_manual(
    values = c(Up = "#CB181D", Down = "#2171B5", NS = "grey70"),
    labels = c(
      Up   = paste0("Up (", n_up, ")"),
      Down = paste0("Down (", n_down, ")"),
      NS   = "NS"
    )
  ) +
  labs(
    title = "Differential expression: dexamethasone vs untreated",
    subtitle = paste0(n_up + n_down, " significant genes (padj < 0.05)"),
    x = expression(log[2]~fold~change),
    y = expression(-log[10]~adjusted~italic(p)),
    colour = NULL
  ) +
  theme_de() +
  theme(legend.position = c(0.85, 0.85))

ggsave(file.path(PLOTS_DIR, "fig2_volcano.png"), p_volcano,
       width = W, height = H, dpi = DPI, bg = "white")

# ── Figure 3: Heatmap of top DE genes ─────────────────────────
top30 <- de |>
  filter(sig != "NS") |>
  arrange(padj) |>
  slice_head(n = 30)

heatmap_mat <- vst_mat[top30$gene_id, ]
rownames(heatmap_mat) <- top30$gene_name

# Scale rows (z-score)
heatmap_scaled <- t(scale(t(heatmap_mat)))

# Annotation
annot_col <- sample_meta |>
  select(sample, condition) |>
  column_to_rownames("sample")

annot_colors <- list(condition = COND_FILL)

png(file.path(PLOTS_DIR, "fig3_heatmap.png"),
    width = W, height = 8, units = "in", res = DPI)
pheatmap(
  heatmap_scaled,
  annotation_col = annot_col,
  annotation_colors = annot_colors,
  cluster_cols = TRUE,
  cluster_rows = TRUE,
  show_colnames = TRUE,
  fontsize_row = 8,
  fontsize_col = 8,
  main = "Top 30 differentially expressed genes (z-scored VST)"
)
dev.off()

# ── Figure 4: Sample-to-sample distance ───────────────────────
sample_dist <- dist(t(vst_mat[top500, ]))
dist_mat <- as.matrix(sample_dist)

annot_row <- sample_meta |>
  select(sample, condition, cell_line) |>
  column_to_rownames("sample")

annot_colors2 <- list(
  condition = COND_FILL,
  cell_line = c(N61311 = "#66C2A5", N052611 = "#FC8D62",
                N080611 = "#8DA0CB", N061011 = "#E78AC3")
)

png(file.path(PLOTS_DIR, "fig4_sample_distance.png"),
    width = W, height = 6, units = "in", res = DPI)
pheatmap(
  dist_mat,
  clustering_distance_rows = sample_dist,
  clustering_distance_cols = sample_dist,
  annotation_row = annot_row,
  annotation_col = annot_row,
  annotation_colors = annot_colors2,
  show_rownames = TRUE,
  show_colnames = TRUE,
  fontsize = 9,
  main = "Sample-to-sample distance (Euclidean, top 500 genes)"
)
dev.off()

# ── Figure 5: Top DE genes bar plot ───────────────────────────
top10_up <- de |> filter(sig == "Up") |> arrange(padj) |> slice_head(n = 10)
top10_dn <- de |> filter(sig == "Down") |> arrange(padj) |> slice_head(n = 10)
top20 <- bind_rows(top10_up, top10_dn) |>
  mutate(gene_name = factor(gene_name, levels = gene_name[order(log2FoldChange)]))

p_bar <- ggplot(top20, aes(log2FoldChange, gene_name, fill = sig)) +
  geom_col() +
  scale_fill_manual(values = c(Up = "#CB181D", Down = "#2171B5")) +
  geom_vline(xintercept = 0, colour = "grey30") +
  labs(
    title = "Top 10 up- and down-regulated genes",
    subtitle = "By adjusted p-value",
    x = expression(log[2]~fold~change),
    y = NULL,
    fill = NULL
  ) +
  theme_de() +
  theme(legend.position = "none")

ggsave(file.path(PLOTS_DIR, "fig5_top_genes.png"), p_bar,
       width = W, height = 6, dpi = DPI, bg = "white")

# ── Figure 6: Expression boxplots of key genes ────────────────
key_genes <- c("FKBP5", "TSC22D3", "KLF15", "ZBTB16",
               "CXCL12", "KCTD12", "VCAM1", "SOX4")
key_ids <- gene_map |> filter(gene_name %in% key_genes) |> pull(gene_id)

norm_counts <- read_tsv(
  file.path(RESULTS_DIR, "differentialabundance/airway/tables/processed_abundance/all.normalised_counts.tsv"),
  show_col_types = FALSE
)

key_expr <- norm_counts |>
  filter(gene_id %in% key_ids) |>
  left_join(gene_map, by = "gene_id") |>
  pivot_longer(cols = -c(gene_id, gene_name),
               names_to = "sample", values_to = "count") |>
  left_join(sample_meta, by = "sample") |>
  mutate(gene_name = factor(gene_name, levels = key_genes))

p_expr <- ggplot(key_expr, aes(condition, count, fill = condition)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 1.5) +
  facet_wrap(~ gene_name, scales = "free_y", nrow = 2) +
  scale_fill_manual(values = COND_FILL) +
  labs(
    title = "Expression of key differentially expressed genes",
    subtitle = "Normalised counts",
    x = NULL,
    y = "Normalised count",
    fill = NULL
  ) +
  theme_de() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

ggsave(file.path(PLOTS_DIR, "fig6_gene_expression.png"), p_expr,
       width = 10, height = 6, dpi = DPI, bg = "white")

message("All figures saved to ", PLOTS_DIR)
