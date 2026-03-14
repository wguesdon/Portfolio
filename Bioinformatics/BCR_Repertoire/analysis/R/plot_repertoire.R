#!/usr/bin/env Rscript
# analysis/R/plot_repertoire.R
#
# BCR repertoire visualisation — Galson et al. 2020 (COVID-19, PRJNA638224)
#
# Produces six publication-quality figures saved to analysis/plots/:
#   fig1_funnel.png         — Sequence processing attrition
#   fig2_vgene_family.png   — V gene family usage by condition
#   fig3_vgene_gene.png     — Top V genes (heatmap)
#   fig4_cdr3_length.png    — CDR3 aa-length distribution
#   fig5_isotype.png        — Isotype proportions per sample
#   fig6_clonal.png         — Clonal expansion (clone-size rank plot)
#   fig7_shm.png            — Somatic hypermutation frequency
#
# Usage (inside container):
#   Rscript /data/analysis/R/plot_repertoire.R
#
# Working directory must be the repository root so that relative paths
# (results/, analysis/) resolve correctly.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(RColorBrewer)
})

# ── Constants ─────────────────────────────────────────────────────────────────

PLOTS_DIR    <- "analysis/plots"
RESULTS_DIR  <- "results/airrflow/galson2020"
REPS_DIR     <- file.path(RESULTS_DIR, "repertoire_comparison/repertoires")
FUNNEL_FILE  <- file.path(RESULTS_DIR,
                  "repertoire_comparison/Sequence_numbers_summary",
                  "Table_sequences_assembly.tsv")

# Condition colours (colorblind-safe)
COND_COLS <- c(
  "COVID19_Mild"   = "#2171B5",   # blue
  "COVID19_Severe" = "#CB181D"    # red
)
COND_LABELS <- c(
  "COVID19_Mild"   = "Mild",
  "COVID19_Severe" = "Severe"
)

SAMPLE_COLS <- c(
  "COV_M01" = "#6BAED6",
  "COV_M02" = "#2171B5",
  "COV_M03" = "#08306B",
  "COV_S01" = "#FC9272",
  "COV_S02" = "#CB181D"
)

DPI    <- 300
WIDTH  <- 8   # inches
HEIGHT <- 5   # inches

dir.create(PLOTS_DIR, recursive = TRUE, showWarnings = FALSE)

# ── Helpers ───────────────────────────────────────────────────────────────────

#' Extract primary V gene or family from a v_call string.
#'
#' Args:
#'   x: Character vector of v_call values (e.g. "IGHV3-23*01,IGHV3-23*04").
#'   level: "family" returns "IGHV3"; "gene" returns "IGHV3-23".
#'
#' Returns:
#'   Character vector of the same length as x.
extract_vgene <- function(x, level = c("family", "gene")) {
  level <- match.arg(level)
  primary <- str_extract(x, "^[^,]+")          # first allele
  gene    <- str_replace(primary, "\\*.*", "")  # strip allele
  if (level == "gene") return(gene)
  str_extract(gene, "^IGHV[0-9]+")             # strip sub-family
}

#' Clean theme for publication figures.
theme_bcr <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      strip.background  = element_blank(),
      strip.text        = element_text(face = "bold"),
      legend.position   = "right",
      panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.3),
      plot.title        = element_text(face = "bold", size = base_size + 1),
      plot.subtitle     = element_text(colour = "grey40", size = base_size - 1),
      axis.text         = element_text(colour = "grey20")
    )
}

# ── Load repertoires ──────────────────────────────────────────────────────────

message("Loading clone-pass repertoires …")

rep_files <- list.files(REPS_DIR, pattern = "*__clone-pass.tsv", full.names = TRUE)
if (length(rep_files) == 0) stop("No clone-pass TSV files found in ", REPS_DIR)

reps <- lapply(rep_files, read_tsv,
               col_types = cols(
                 v_call           = col_character(),
                 j_call           = col_character(),
                 junction_length  = col_double(),
                 clone_id         = col_character(),
                 clone_size_count = col_double(),
                 clone_size_freq  = col_double(),
                 mu_freq          = col_double(),
                 isotype          = col_character(),
                 disease_diagnosis= col_character(),
                 subject_id       = col_character(),
                 sample_id        = col_character(),
                 .default         = col_character()
               ),
               show_col_types = FALSE)
db <- bind_rows(reps)

# Derived columns
db <- db |>
  mutate(
    condition   = factor(disease_diagnosis,
                         levels = c("COVID19_Mild", "COVID19_Severe"),
                         labels = c("Mild", "Severe")),
    subject     = factor(subject_id,
                         levels = names(SAMPLE_COLS)),
    v_family    = extract_vgene(v_call, "family"),
    v_gene      = extract_vgene(v_call, "gene"),
    cdr3_aa_len = as.integer(junction_length) %/% 3L,
    isotype_grp = case_when(
      str_detect(isotype, "^IGHG") ~ "IgG",
      str_detect(isotype, "^IGHA") ~ "IgA",
      isotype == "IGHM"            ~ "IgM",
      isotype == "IGHD"            ~ "IgD",
      isotype == "IGHE"            ~ "IgE",
      TRUE                         ~ "Unknown"
    ) |> factor(levels = c("IgM", "IgD", "IgG", "IgA", "IgE", "Unknown"))
  )

n_seqs <- nrow(db)
message(sprintf("  Loaded %d sequences across %d samples", n_seqs, n_distinct(db$subject_id)))

# ── Figure 1: Processing funnel ───────────────────────────────────────────────
message("Figure 1: processing funnel …")

funnel_raw <- read_tsv(FUNNEL_FILE, show_col_types = FALSE)

funnel <- funnel_raw |>
  select(
    subject_id,
    disease_diagnosis,
    raw        = Sequences_R1,
    quality    = Filtered_quality_R1,
    paired     = Paired,
    consensus  = Build_consensus,
    assembled  = Assemble_pairs,
    unique     = Unique,
    igblast    = Igblast
  ) |>
  filter(subject_id != "COV_S03") |>    # excluded: all CDR3s N-masked
  pivot_longer(raw:igblast, names_to = "step", values_to = "count") |>
  mutate(
    step = factor(step, levels = c(
      "raw", "quality", "paired", "consensus",
      "assembled", "unique", "igblast"
    ), labels = c(
      "Raw reads", "Quality filter", "Pair UMIs",
      "Build consensus", "Assemble pairs", "Deduplicate", "IgBLAST"
    )),
    condition = factor(disease_diagnosis,
                       levels = c("COVID19_Mild", "COVID19_Severe"),
                       labels = c("Mild", "Severe")),
    subject_id = factor(subject_id, levels = names(SAMPLE_COLS))
  )

fig1 <- ggplot(funnel, aes(x = step, y = count, colour = condition, group = subject_id)) +
  geom_line(aes(linetype = subject_id), linewidth = 0.7, alpha = 0.85) +
  geom_point(size = 2.5, alpha = 0.9) +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_colour_manual(values = c(Mild = "#2171B5", Severe = "#CB181D"),
                      name = "Severity") +
  scale_linetype_manual(values = c(1, 2, 3, 4, 5), name = "Patient") +
  labs(
    title    = "Sequence processing attrition",
    subtitle = "5 COVID-19 patients (3 mild, 2 severe) · COV_S03 excluded",
    x        = NULL,
    y        = "Sequences (log₁₀ scale)"
  ) +
  theme_bcr() +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.box  = "horizontal"
  )

ggsave(file.path(PLOTS_DIR, "fig1_funnel.png"),
       fig1, width = WIDTH, height = HEIGHT, dpi = DPI)

# ── Figure 2: V gene family usage ────────────────────────────────────────────
message("Figure 2: V gene family usage …")

vfam <- db |>
  filter(!is.na(v_family)) |>
  count(condition, v_family) |>
  group_by(condition) |>
  mutate(freq = n / sum(n)) |>
  ungroup()

fig2 <- ggplot(vfam, aes(x = reorder(v_family, freq), y = freq, fill = condition)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.65, alpha = 0.9) +
  scale_fill_manual(values = c(Mild = "#2171B5", Severe = "#CB181D"),
                    name = "Severity") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  coord_flip() +
  labs(
    title    = "V gene family usage",
    subtitle = "Frequency among productive IGH sequences",
    x        = "V gene family",
    y        = "% of sequences"
  ) +
  theme_bcr() +
  theme(legend.position = "right")

ggsave(file.path(PLOTS_DIR, "fig2_vgene_family.png"),
       fig2, width = WIDTH, height = HEIGHT, dpi = DPI)

# ── Figure 3: Top V genes (dot plot) ─────────────────────────────────────────
message("Figure 3: Top V genes …")

top_genes <- db |>
  filter(!is.na(v_gene)) |>
  count(v_gene) |>
  slice_max(n, n = 15) |>
  pull(v_gene)

vgene_mat <- db |>
  filter(v_gene %in% top_genes, !is.na(v_gene)) |>
  count(subject, condition, v_gene) |>
  group_by(subject) |>
  mutate(freq = n / sum(n)) |>
  ungroup()

fig3 <- ggplot(vgene_mat,
               aes(x = subject, y = reorder(v_gene, freq), size = freq, colour = condition)) +
  geom_point(alpha = 0.8) +
  scale_size_area(max_size = 12, labels = percent_format(accuracy = 1),
                  name = "Freq.") +
  scale_colour_manual(values = c(Mild = "#2171B5", Severe = "#CB181D"),
                      name = "Severity") +
  labs(
    title    = "Top 15 V gene usage per patient",
    subtitle = "Bubble size ∝ frequency within sample",
    x        = "Patient",
    y        = "V gene"
  ) +
  theme_bcr() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(file.path(PLOTS_DIR, "fig3_vgene_gene.png"),
       fig3, width = WIDTH + 1, height = HEIGHT + 1, dpi = DPI)

# ── Figure 4: CDR3 length distribution ───────────────────────────────────────
message("Figure 4: CDR3 length distribution …")

cdr3_range <- db |>
  filter(!is.na(cdr3_aa_len), cdr3_aa_len >= 5, cdr3_aa_len <= 35)

# Per-condition medians for annotation
cdr3_med <- cdr3_range |>
  group_by(condition) |>
  summarise(med = median(cdr3_aa_len), .groups = "drop")

fig4 <- ggplot(cdr3_range, aes(x = cdr3_aa_len, fill = condition, colour = condition)) +
  geom_histogram(binwidth = 1, alpha = 0.55, position = "identity") +
  geom_vline(data = cdr3_med,
             aes(xintercept = med, colour = condition),
             linetype = "dashed", linewidth = 0.8) +
  scale_fill_manual(values  = c(Mild = "#2171B5", Severe = "#CB181D"),
                    name = "Severity") +
  scale_colour_manual(values = c(Mild = "#2171B5", Severe = "#CB181D"),
                      guide = "none") +
  scale_x_continuous(breaks = seq(5, 35, by = 5)) +
  labs(
    title    = "CDR3 length distribution",
    subtitle = "Dashed lines = per-condition median",
    x        = "CDR3 length (amino acids)",
    y        = "Number of sequences"
  ) +
  theme_bcr()

ggsave(file.path(PLOTS_DIR, "fig4_cdr3_length.png"),
       fig4, width = WIDTH, height = HEIGHT, dpi = DPI)

# ── Figure 5: Isotype proportions ────────────────────────────────────────────
message("Figure 5: isotype proportions …")

iso_freq <- db |>
  filter(isotype_grp != "Unknown") |>
  count(subject, condition, isotype_grp) |>
  group_by(subject) |>
  mutate(freq = n / sum(n)) |>
  ungroup()

iso_cols <- c(
  IgM = "#9ECAE1", IgD = "#6BAED6", IgG = "#2171B5",
  IgA = "#FC9272", IgE = "#CB181D"
)

fig5 <- ggplot(iso_freq, aes(x = subject, y = freq, fill = isotype_grp)) +
  geom_col(width = 0.7, colour = "white", linewidth = 0.2) +
  geom_text(aes(label = ifelse(freq > 0.08, percent(freq, accuracy = 1), "")),
            position = position_stack(vjust = 0.5),
            size = 3, colour = "white", fontface = "bold") +
  facet_grid(~ condition, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = iso_cols, name = "Isotype") +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title    = "Isotype distribution per patient",
    subtitle = "Productive IGH sequences with C-region assignment",
    x        = "Patient",
    y        = "% of sequences"
  ) +
  theme_bcr()

ggsave(file.path(PLOTS_DIR, "fig5_isotype.png"),
       fig5, width = WIDTH, height = HEIGHT, dpi = DPI)

# ── Figure 6: Clonal expansion ───────────────────────────────────────────────
message("Figure 6: clonal expansion …")

clone_df <- db |>
  distinct(subject, condition, clone_id, clone_size_count) |>
  filter(!is.na(clone_id)) |>
  group_by(subject, condition) |>
  arrange(desc(clone_size_count)) |>
  mutate(rank = row_number(),
         expanded = clone_size_count > 1) |>
  ungroup()

pct_expanded <- clone_df |>
  group_by(subject, condition) |>
  summarise(
    pct = mean(expanded) * 100,
    .groups = "drop"
  )

fig6a <- ggplot(clone_df, aes(x = rank, y = clone_size_count,
                               colour = condition, group = subject)) +
  geom_step(linewidth = 0.7, alpha = 0.85) +
  scale_colour_manual(values = c(Mild = "#2171B5", Severe = "#CB181D"),
                      name = "Severity") +
  scale_y_continuous(breaks = scales::breaks_extended()) +
  labs(
    title = "Clone size rank plot",
    x     = "Clone rank",
    y     = "Clone size (# sequences)"
  ) +
  theme_bcr()

fig6b <- ggplot(pct_expanded, aes(x = subject, y = pct, fill = condition)) +
  geom_col(width = 0.6, alpha = 0.9) +
  scale_fill_manual(values = c(Mild = "#2171B5", Severe = "#CB181D"),
                    name = "Severity") +
  scale_y_continuous(limits = c(0, 100),
                     labels = function(x) paste0(x, "%")) +
  facet_grid(~ condition, scales = "free_x", space = "free_x") +
  labs(
    title = "% clonally expanded (size ≥ 2)",
    x     = "Patient",
    y     = "% clones"
  ) +
  theme_bcr() +
  theme(legend.position = "none")

fig6 <- fig6a + fig6b + plot_annotation(
  title    = "Clonal expansion",
  subtitle = "COVID-19 IGH repertoires"
)

ggsave(file.path(PLOTS_DIR, "fig6_clonal.png"),
       fig6, width = WIDTH * 1.5, height = HEIGHT, dpi = DPI)

# ── Figure 7: Somatic hypermutation ──────────────────────────────────────────
message("Figure 7: somatic hypermutation …")

shm_df <- db |>
  filter(
    !is.na(mu_freq),
    isotype_grp %in% c("IgM", "IgG", "IgA")
  ) |>
  mutate(mu_pct = as.numeric(mu_freq) * 100)

shm_med <- shm_df |>
  group_by(condition, isotype_grp) |>
  summarise(med = median(mu_pct), .groups = "drop")

fig7 <- ggplot(shm_df, aes(x = condition, y = mu_pct, fill = condition)) +
  geom_boxplot(alpha = 0.6, outlier.shape = NA, width = 0.55) +
  geom_jitter(aes(colour = condition), width = 0.15, size = 1.8, alpha = 0.7) +
  facet_wrap(~ isotype_grp, nrow = 1) +
  scale_fill_manual(values  = c(Mild = "#2171B5", Severe = "#CB181D"),
                    guide   = "none") +
  scale_colour_manual(values = c(Mild = "#2171B5", Severe = "#CB181D"),
                      guide  = "none") +
  labs(
    title    = "Somatic hypermutation frequency",
    subtitle = "Mutation rate in V gene relative to germline",
    x        = "Disease severity",
    y        = "Mutation frequency (%)"
  ) +
  theme_bcr() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))

ggsave(file.path(PLOTS_DIR, "fig7_shm.png"),
       fig7, width = WIDTH, height = HEIGHT, dpi = DPI)

# ── Done ──────────────────────────────────────────────────────────────────────
message("\nAll figures saved to ", PLOTS_DIR, "/")
message("Files written:")
for (f in list.files(PLOTS_DIR, pattern = "\\.png$", full.names = TRUE)) {
  message("  ", f)
}
