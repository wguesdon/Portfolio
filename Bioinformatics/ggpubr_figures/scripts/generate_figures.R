#!/usr/bin/env Rscript
# Generate publication-ready figures using ggpubr + ggprism
# Datasets: palmerpenguins, gapminder, iris
# Color palette: NPG (Nature Publishing Group)

library(ggpubr)
library(ggprism)
library(ggplot2)
library(dplyr)
library(palmerpenguins)
library(gapminder)

outdir <- "output"
dir.create(outdir, showWarnings = FALSE)

penguins_clean <- penguins |> filter(!is.na(bill_length_mm), !is.na(body_mass_g))

# NPG palette colours (from ggsci)
npg_3 <- c("#E64B35", "#4DBBD5", "#00A087")
npg_2 <- c("#E64B35", "#4DBBD5")

# ── 1. Density plot: penguin body mass by species ────────────────────────────
p1 <- ggdensity(penguins_clean, x = "body_mass_g",
                add = "mean", rug = TRUE,
                color = "species", fill = "species",
                palette = npg_3,
                xlab = "Body Mass (g)", ylab = "Density",
                title = "Body Mass Distribution by Penguin Species") +
  theme_prism(base_size = 12) +
  scale_y_continuous(guide = guide_prism_minor())
ggsave(file.path(outdir, "01_density_body_mass.png"), p1, width = 8, height = 5, dpi = 300)

# ── 2. Histogram: bill length by species ─────────────────────────────────────
p2 <- gghistogram(penguins_clean, x = "bill_length_mm",
                  add = "median", rug = TRUE,
                  color = "species", fill = "species",
                  palette = npg_3,
                  xlab = "Bill Length (mm)", ylab = "Count",
                  title = "Bill Length Distribution by Species") +
  theme_prism(base_size = 12) +
  scale_y_continuous(guide = guide_prism_minor())
ggsave(file.path(outdir, "02_histogram_bill_length.png"), p2, width = 8, height = 5, dpi = 300)

# ── 3. Box plot with statistical comparisons ─────────────────────────────────
species_comparisons <- list(c("Adelie", "Chinstrap"),
                            c("Chinstrap", "Gentoo"),
                            c("Adelie", "Gentoo"))

p3 <- ggboxplot(penguins_clean, x = "species", y = "flipper_length_mm",
                color = "species",
                palette = npg_3,
                add = "jitter", shape = "species",
                xlab = "Species", ylab = "Flipper Length (mm)",
                title = "Flipper Length Comparison Across Species") +
  stat_compare_means(comparisons = species_comparisons) +
  stat_compare_means(label.y = 240) +
  theme_prism(base_size = 12) +
  scale_y_continuous(guide = guide_prism_minor()) +
  scale_x_discrete(guide = guide_prism_bracket(width = 0.15))
ggsave(file.path(outdir, "03_boxplot_flipper_length.png"), p3, width = 8, height = 6, dpi = 300)

# ── 4. Violin plot with nested box plot ──────────────────────────────────────
p4 <- ggviolin(penguins_clean, x = "species", y = "bill_depth_mm",
               fill = "species",
               palette = npg_3,
               add = "boxplot", add.params = list(fill = "white"),
               xlab = "Species", ylab = "Bill Depth (mm)",
               title = "Bill Depth Distribution with Statistical Tests") +
  stat_compare_means(comparisons = species_comparisons, label = "p.signif") +
  stat_compare_means(label.y = 25) +
  theme_prism(base_size = 12) +
  scale_y_continuous(guide = guide_prism_minor()) +
  scale_x_discrete(guide = guide_prism_bracket(width = 0.15))
ggsave(file.path(outdir, "04_violin_bill_depth.png"), p4, width = 8, height = 6, dpi = 300)

# ── 5. Scatter plot with correlation ─────────────────────────────────────────
p5 <- ggscatter(penguins_clean, x = "bill_length_mm", y = "body_mass_g",
                color = "species", shape = "species",
                palette = npg_3,
                add = "reg.line", conf.int = TRUE,
                xlab = "Bill Length (mm)", ylab = "Body Mass (g)",
                title = "Bill Length vs Body Mass with Linear Regression") +
  stat_cor(aes(color = species), label.y = c(6200, 6000, 5800)) +
  theme_prism(base_size = 12) +
  scale_x_continuous(guide = guide_prism_minor()) +
  scale_y_continuous(guide = guide_prism_minor())
ggsave(file.path(outdir, "05_scatter_bill_vs_mass.png"), p5, width = 8, height = 6, dpi = 300)

# ── 6. Ordered bar plot: gapminder life expectancy (Americas, 2007) ──────────
americas_2007 <- gapminder |>
  filter(continent == "Americas", year == 2007) |>
  mutate(lifeExp_grp = factor(ifelse(lifeExp < median(lifeExp), "Below Median", "Above Median"),
                              levels = c("Below Median", "Above Median")))

p6 <- ggbarplot(americas_2007, x = "country", y = "lifeExp",
                fill = "lifeExp_grp",
                color = "white",
                palette = npg_2,
                sort.val = "asc",
                sort.by.groups = FALSE,
                x.text.angle = 90,
                xlab = FALSE,
                ylab = "Life Expectancy (years)",
                legend.title = "Group",
                title = "Life Expectancy in the Americas (2007)") +
  theme_prism(base_size = 11) +
  scale_y_continuous(guide = guide_prism_minor()) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
ggsave(file.path(outdir, "06_barplot_lifeexp_americas.png"), p6, width = 10, height = 6, dpi = 300)

# ── 7. Deviation plot: GDP per capita z-scores (Europe, 2007) ────────────────
europe_2007 <- gapminder |>
  filter(continent == "Europe", year == 2007) |>
  mutate(gdp_z = (gdpPercap - mean(gdpPercap)) / sd(gdpPercap),
         gdp_grp = factor(ifelse(gdp_z < 0, "Below Average", "Above Average"),
                          levels = c("Below Average", "Above Average")))

p7 <- ggbarplot(europe_2007, x = "country", y = "gdp_z",
                fill = "gdp_grp",
                color = "white",
                palette = npg_2,
                sort.val = "asc",
                sort.by.groups = FALSE,
                x.text.angle = 90,
                ylab = "GDP per Capita (z-score)",
                xlab = FALSE,
                legend.title = "GDP Group",
                title = "GDP per Capita Deviation in Europe (2007)",
                rotate = TRUE,
                ggtheme = theme_prism(base_size = 10))
ggsave(file.path(outdir, "07_deviation_gdp_europe.png"), p7, width = 8, height = 10, dpi = 300)

# ── 8. Lollipop chart: GDP per capita in Europe & Americas (2007) ─────────────
gdp_2007 <- gapminder |>
  filter((continent == "Europe" | country == "United States"), year == 2007) |>
  mutate(region = ifelse(country == "United States", "USA", "Europe"),
         gdpPercap_k = round(gdpPercap / 1000, 1))

p8 <- ggdotchart(gdp_2007, x = "country", y = "gdpPercap_k",
                 color = "region",
                 palette = npg_2,
                 sorting = "ascending",
                 add = "segments",
                 rotate = TRUE,
                 group = "region",
                 dot.size = 4,
                 label = gdp_2007 |> arrange(gdpPercap_k) |> pull(gdpPercap_k),
                 font.label = list(color = "white", size = 6, vjust = 0.5),
                 xlab = FALSE, ylab = "GDP per Capita (thousands USD)",
                 title = "GDP per Capita: Europe and the USA (2007)",
                 ggtheme = theme_prism(base_size = 9))
ggsave(file.path(outdir, "08_lollipop_asia_lifeexp.png"), p8, width = 8, height = 10, dpi = 300)

# ── 9. Cleveland dot plot: iris sepal length by species ──────────────────────
iris_df <- iris |>
  mutate(name = paste(Species, row_number(), sep = "_")) |>
  group_by(Species) |>
  slice_max(order_by = Sepal.Length, n = 10) |>
  ungroup()

p9 <- ggdotchart(iris_df, x = "name", y = "Sepal.Length",
                 color = "Species",
                 palette = npg_3,
                 sorting = "descending",
                 rotate = TRUE,
                 dot.size = 3,
                 y.text.col = TRUE,
                 xlab = FALSE, ylab = "Sepal Length (cm)",
                 title = "Top 10 Sepal Lengths per Iris Species",
                 ggtheme = theme_prism(base_size = 11)) +
  theme_cleveland()
ggsave(file.path(outdir, "09_cleveland_iris_sepal.png"), p9, width = 8, height = 8, dpi = 300)

# ── 10. Multi-panel: means with error bars (penguin measurements by island) ──
p10a <- ggbarplot(penguins_clean, x = "island", y = "body_mass_g",
                  add = "mean_se", fill = "species",
                  palette = npg_3,
                  position = position_dodge(0.8),
                  ylab = "Body Mass (g)", xlab = "Island") +
  theme_prism(base_size = 11) +
  scale_y_continuous(guide = guide_prism_minor()) +
  scale_x_discrete(guide = guide_prism_bracket(width = 0.15))

p10b <- ggbarplot(penguins_clean, x = "island", y = "flipper_length_mm",
                  add = "mean_se", fill = "species",
                  palette = npg_3,
                  position = position_dodge(0.8),
                  ylab = "Flipper Length (mm)", xlab = "Island") +
  theme_prism(base_size = 11) +
  scale_y_continuous(guide = guide_prism_minor()) +
  scale_x_discrete(guide = guide_prism_bracket(width = 0.15))

p10 <- ggarrange(p10a, p10b, ncol = 2, common.legend = TRUE, legend = "bottom",
                 labels = c("A", "B"))
p10 <- annotate_figure(p10, top = text_grob("Penguin Morphometrics by Island",
                                            face = "bold", size = 14))
ggsave(file.path(outdir, "10_multipanel_island.png"), p10, width = 12, height = 5, dpi = 300)

# ── 11. Faceted box plot ─────────────────────────────────────────────────────
p11 <- ggboxplot(penguins_clean, x = "sex", y = "body_mass_g",
                 color = "sex", palette = npg_2,
                 facet.by = "species",
                 xlab = "Sex", ylab = "Body Mass (g)",
                 title = "Body Mass by Sex Across Penguin Species") +
  stat_compare_means(label = "p.signif") +
  theme_prism(base_size = 12) +
  scale_y_continuous(guide = guide_prism_minor()) +
  scale_x_discrete(guide = guide_prism_bracket(width = 0.2))
ggsave(file.path(outdir, "11_faceted_boxplot_sex.png"), p11, width = 10, height = 5, dpi = 300)

cat("All figures saved to", outdir, "\n")
