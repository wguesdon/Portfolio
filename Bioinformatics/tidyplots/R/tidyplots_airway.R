## ============================================================
##  tidyplots tutorial — Airway RNA-seq dataset (GSE52778)
##  Himes et al. 2014: dexamethasone vs untreated in airway
##  smooth muscle cells (4 donors, paired design)
## ============================================================

suppressPackageStartupMessages({
  library(airway)
  library(DESeq2)
  library(tidyplots)
  library(dplyr)
  library(tibble)
  library(ggplot2)
  library(ComplexHeatmap)
  library(circlize)
  library(cli)
})

dir.create("output", showWarnings = FALSE)

## ── 1. Load & prepare data ────────────────────────────────────
cli_inform(c("i" = "Loading airway dataset and running DESeq2 …"))

data(airway)

dds <- DESeqDataSet(airway, design = ~ cell + dex)
dds <- DESeq(dds)
res <- results(dds, contrast = c("dex", "trt", "untrt"), alpha = 0.05)
vsd <- vst(dds, blind = FALSE)

# Tidy results — pre-compute all transformed columns
res_df <- as.data.frame(res) |>
  rownames_to_column("gene_id") |>
  filter(!is.na(padj)) |>
  mutate(
    neg_log10_pvalue = -log10(pvalue),
    neg_log10_padj   = -log10(padj),
    log2_mean        = log2(baseMean + 1),
    significant      = padj < 0.05 & abs(log2FoldChange) > 1,
    direction        = case_when(
      padj < 0.05 & log2FoldChange >  1 ~ "Up",
      padj < 0.05 & log2FoldChange < -1 ~ "Down",
      TRUE                               ~ "NS"
    )
  )

# Tidy count matrix (long format)
counts_long <- assay(vsd) |>
  as.data.frame() |>
  rownames_to_column("gene_id") |>
  tidyr::pivot_longer(-gene_id, names_to = "sample", values_to = "vst") |>
  left_join(
    as.data.frame(colData(dds)) |>
      rownames_to_column("sample") |>
      select(sample, cell, dex),
    by = "sample"
  ) |>
  mutate(condition = ifelse(dex == "trt", "Dexamethasone", "Untreated"))

# One-row-per-sample summary
sample_stats <- counts_long |>
  group_by(sample, condition, cell) |>
  summarise(median_vst = median(vst), .groups = "drop")

cli_inform(c(
  "v" = "DESeq2 complete",
  " " = "Total tested genes              : {nrow(res_df)}",
  " " = "Significant (FDR<0.05, |LFC|>1): {sum(res_df$significant)}",
  " " = "Up-regulated                    : {sum(res_df$direction == 'Up')}",
  " " = "Down-regulated                  : {sum(res_df$direction == 'Down')}"
))


## ── helpers ───────────────────────────────────────────────────
# adjust_size() controls the DATA PANEL in mm.
# save_plot() width/height=NA lets the canvas auto-size around the panel.
# 1200 DPI matches Nature/Cell submission requirements.
save <- function(plot, file, dpi = 1200) {
  save_plot(plot, file, width = NA, height = NA,
            units = "in", view_plot = FALSE, dpi = dpi)
}


## ── 2. PLOT 1: Volcano plot ────────────────────────────────────
cli_inform(c("i" = "Plot 1/10 — volcano plot"))

p_volcano <- res_df |>
  tidyplot(x = log2FoldChange, y = neg_log10_pvalue, color = direction) |>
  add_data_points(alpha = 0.5, size = 1.2) |>
  add_reference_lines(x = c(-1, 1), y = -log10(0.05)) |>
  adjust_colors(c("Up" = "#E63946", "NS" = "#ADB5BD", "Down" = "#457B9D")) |>
  adjust_title("Volcano Plot: Dexamethasone vs Untreated") |>
  adjust_x_axis_title("Log2 Fold Change") |>
  adjust_y_axis_title("-Log10(p-value)") |>
  adjust_legend_title("Direction") |>
  theme_tidyplot() |>
  adjust_legend_position("right") |>
  adjust_size(width = 160, height = 120)

save(p_volcano, "output/01_volcano.png")


## ── 3. PLOT 2: MA plot ─────────────────────────────────────────
cli_inform(c("i" = "Plot 2/10 — MA plot"))

p_ma <- res_df |>
  tidyplot(x = log2_mean, y = log2FoldChange, color = direction) |>
  add_data_points(alpha = 0.4, size = 1) |>
  add_reference_lines(y = c(-1, 0, 1)) |>
  adjust_colors(c("Up" = "#E63946", "NS" = "#ADB5BD", "Down" = "#457B9D")) |>
  adjust_title("MA Plot: Dexamethasone vs Untreated") |>
  adjust_x_axis_title("Log2 Mean Expression") |>
  adjust_y_axis_title("Log2 Fold Change") |>
  adjust_legend_title("Direction") |>
  theme_tidyplot() |>
  adjust_size(width = 160, height = 120)

save(p_ma, "output/02_ma_plot.png")


## ── 4. PLOT 3: Top DEGs — horizontal bar chart ────────────────
cli_inform(c("i" = "Plot 3/10 — top DEGs bar chart"))

top_genes <- res_df |>
  filter(significant) |>
  arrange(padj) |>
  slice_head(n = 20) |>
  mutate(gene_id = forcats::fct_reorder(gene_id, log2FoldChange))

p_bar <- top_genes |>
  tidyplot(x = gene_id, y = log2FoldChange, color = direction) |>
  add_mean_bar() |>
  add_reference_lines(y = 0) |>
  adjust_colors(c("Up" = "#E63946", "Down" = "#457B9D")) |>
  flip_plot() |>
  adjust_title("Top 20 Differentially Expressed Genes") |>
  adjust_x_axis_title("Gene") |>
  adjust_y_axis_title("Log2 Fold Change") |>
  adjust_legend_title("Direction") |>
  theme_tidyplot() |>
  adjust_size(width = 120, height = 160)

save(p_bar, "output/03_top_degs_bar.png")


## ── 5. PLOT 4: Sample median VST expression ───────────────────
cli_inform(c("i" = "Plot 4/10 — sample medians"))

p_samples <- sample_stats |>
  mutate(sample = forcats::fct_reorder(sample, median_vst)) |>
  tidyplot(x = sample, y = median_vst, color = condition) |>
  add_mean_bar() |>
  adjust_colors(c("Dexamethasone" = "#E63946", "Untreated" = "#457B9D")) |>
  flip_plot() |>
  adjust_title("Median VST-Normalised Expression per Sample") |>
  adjust_x_axis_title("Sample") |>
  adjust_y_axis_title("Median VST") |>
  adjust_legend_title("Condition") |>
  theme_tidyplot() |>
  adjust_legend_position("right") |>
  adjust_size(width = 120, height = 100)

save(p_samples, "output/04_sample_medians.png")


## ── 6. PLOT 5: Global expression distribution ─────────────────
cli_inform(c("i" = "Plot 5/10 — expression distribution"))

set.seed(42)
sampled_genes <- sample(unique(counts_long$gene_id), 3000)

p_violin <- counts_long |>
  filter(gene_id %in% sampled_genes) |>
  tidyplot(x = condition, y = vst, color = condition) |>
  add_violin(alpha = 0.6) |>
  add_boxplot(width = 0.15, outlier.shape = NA) |>
  adjust_colors(c("Dexamethasone" = "#E63946", "Untreated" = "#457B9D")) |>
  adjust_title("Global Expression Distribution by Condition") |>
  adjust_x_axis_title("Condition") |>
  adjust_y_axis_title("VST Expression") |>
  theme_tidyplot() |>
  remove_legend() |>
  adjust_size(width = 100, height = 130)

save(p_violin, "output/05_expression_distribution.png")


## ── 7. PLOT 6: Key gene strip plot ────────────────────────────
cli_inform(c("i" = "Plot 6/10 — key gene strip plot"))

key_gene_ids <- res_df |>
  filter(direction == "Up") |>
  arrange(padj) |>
  slice_head(n = 6) |>
  pull(gene_id)

p_strip <- counts_long |>
  filter(gene_id %in% key_gene_ids) |>
  tidyplot(x = condition, y = vst, color = condition) |>
  add_mean_bar(alpha = 0.3) |>
  add_sem_errorbar() |>
  add_data_points_beeswarm(size = 2.5) |>
  adjust_colors(c("Dexamethasone" = "#E63946", "Untreated" = "#457B9D")) |>
  adjust_title("Top Up-regulated Genes: Individual Sample Expression") |>
  adjust_x_axis_title("Condition") |>
  adjust_y_axis_title("VST Expression") |>
  theme_tidyplot() |>
  remove_legend() |>
  adjust_size(width = 60, height = 100) |>
  split_plot(by = "gene_id", ncol = 3)

save(p_strip, "output/06_key_genes_strip.png")


## ── 8. PLOT 7: P-value histogram (QC) ─────────────────────────
cli_inform(c("i" = "Plot 7/10 — p-value histogram"))

p_pval <- res_df |>
  tidyplot(x = pvalue) |>
  add_histogram(bins = 40, fill = "#457B9D", alpha = 0.8) |>
  adjust_title("P-value Distribution (DESeq2 QC)") |>
  adjust_x_axis_title("Raw p-value") |>
  adjust_y_axis_title("Gene Count") |>
  theme_tidyplot() |>
  adjust_size(width = 140, height = 110)

save(p_pval, "output/07_pvalue_histogram.png")


## ── 9. PLOT 8: Significant DEGs scatter ───────────────────────
cli_inform(c("i" = "Plot 8/10 — significant DEGs scatter"))

n_sig <- sum(res_df$significant)

p_sig <- res_df |>
  filter(significant) |>
  tidyplot(x = log2FoldChange, y = neg_log10_padj, color = direction) |>
  add_data_points(size = 2, alpha = 0.7) |>
  adjust_colors(c("Up" = "#E63946", "Down" = "#457B9D")) |>
  adjust_title(paste0("Significant DEGs  (FDR<0.05, |LFC|>1)  n=", n_sig)) |>
  adjust_x_axis_title("Log2 Fold Change") |>
  adjust_y_axis_title("-Log10(Adjusted p-value)") |>
  adjust_legend_title("Direction") |>
  theme_tidyplot() |>
  adjust_legend_position("right") |>
  adjust_size(width = 150, height = 120)

save(p_sig, "output/08_significant_degs.png")


## ── shared heatmap data (used by plots 9 and 10) ──────────────
top50_ids <- res_df |>
  filter(significant) |>
  arrange(padj) |>
  slice_head(n = 50) |>
  pull(gene_id)

# Dex samples first, then Untreated, within each group sorted by cell line
sample_meta <- as.data.frame(colData(dds)) |>
  rownames_to_column("sample") |>
  mutate(condition = ifelse(dex == "trt", "Dexamethasone", "Untreated")) |>
  arrange(desc(dex), cell)

sample_order <- sample_meta$sample
cond_colors  <- c("Dexamethasone" = "#E63946", "Untreated" = "#457B9D")

heatmap_df <- counts_long |>
  filter(gene_id %in% top50_ids) |>
  mutate(
    sample  = factor(sample,  levels = sample_order),
    gene_id = factor(gene_id, levels = top50_ids)
  )


## ── 10. PLOT 9: tidyplots heatmap (simplified) ────────────────
cli_inform(c("i" = "Plot 9/10 — tidyplots heatmap (simplified)"))

p_heat_tidy <- heatmap_df |>
  tidyplot(x = sample, y = gene_id, color = vst) |>
  tidyplots::add_heatmap(scale = "row", rotate_labels = 45) |>
  adjust_colors(colors_diverging_blue2red) |>
  adjust_title("Heatmap: Top 50 DEGs (Dexamethasone vs Untreated)") |>
  adjust_x_axis_title("Sample") |>
  adjust_y_axis_title("Gene") |>
  adjust_legend_title("Row z-score") |>
  theme_tidyplot() |>
  adjust_legend_position("right") |>
  adjust_size(width = 120, height = 220)

save(p_heat_tidy, "output/09_heatmap_tidyplots.png")


## ── 11. PLOT 10: ComplexHeatmap (dendrogram + annotation) ─────
cli_inform(c("i" = "Plot 10/10 — ComplexHeatmap with dendrogram + annotation"))

mat        <- assay(vsd)[top50_ids, sample_order]
mat_scaled <- t(scale(t(mat)))   # row z-score

# Same blue–white–red scale as tidyplots plot above
col_fun <- colorRamp2(c(-2, 0, 2), c("#457B9D", "white", "#E63946"))

top_anno <- HeatmapAnnotation(
  Condition = sample_meta$condition,
  col       = list(Condition = cond_colors),
  border    = TRUE,
  gap       = unit(1, "mm"),
  annotation_name_gp      = gpar(fontsize = 10, fontface = "bold"),
  annotation_legend_param = list(
    Condition = list(
      title     = "Condition",
      title_gp  = gpar(fontsize = 10, fontface = "bold"),
      labels_gp = gpar(fontsize = 9)
    )
  )
)

ht <- Heatmap(
  mat_scaled,
  name                 = "Row z-score",
  col                  = col_fun,
  top_annotation       = top_anno,
  cluster_rows         = TRUE,
  cluster_columns      = FALSE,
  show_row_dend        = TRUE,
  show_column_dend     = FALSE,
  row_dend_side        = "left",
  row_names_side       = "right",
  show_column_names    = TRUE,
  column_names_rot     = 45,
  row_names_gp         = gpar(fontsize = 7),
  column_names_gp      = gpar(fontsize = 9),
  column_title         = "Top 50 DEGs: Dexamethasone vs Untreated",
  column_title_gp      = gpar(fontsize = 12, fontface = "bold"),
  border               = TRUE,
  heatmap_legend_param = list(
    title         = "Row z-score",
    title_gp      = gpar(fontsize = 10, fontface = "bold"),
    labels_gp     = gpar(fontsize = 9),
    legend_height = unit(4, "cm")
  )
)

png("output/10_heatmap_complexheatmap.png",
    width = 10, height = 14, units = "in", res = 1200)
draw(ht, heatmap_legend_side = "right", annotation_legend_side = "right",
     padding = unit(c(5, 5, 5, 5), "mm"))
dev.off()
cli_inform(c("v" = "save_plot: saved to 'output/10_heatmap_complexheatmap.png'"))


cli_inform(c("v" = "All 10 plots saved to output/"))
