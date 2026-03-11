#'──────────────────────────────────────────────────────────────
# Packages ----
#'──────────────────────────────────────────────────────────────
library(ggplot2)
library(ggpubr)      # for stat_compare_means()
library(dplyr)
#'──────────────────────────────────────────────────────────────
# Example data  ----
#'──────────────────────────────────────────────────────────────
set.seed(123)
data <- data.frame(
  condition = rep(c("Serum starved", "Normal culture"), each = 8),
  cell_type = rep(c(rep("Wild-type cells", 4), rep("GPP8 cell line", 4)), 2),
  signal = c(
    rnorm(4,  mean = 40, sd =  8),
    rnorm(4,  mean = 110, sd = 10),
    rnorm(4,  mean = 25, sd =  5),
    rnorm(4,  mean = 35, sd =  7)
  )
)
data$condition <- factor(data$condition,
                         levels = c("Normal culture", "Serum starved"))
data$cell_type <- factor(data$cell_type,
                         levels = c("Wild-type cells", "GPP8 cell line"))

# sample size for annotation
n_df <- data %>% 
  count(condition, cell_type)

#'──────────────────────────────────────────────────────────────
#  Publication-ready boxplot ----
#'──────────────────────────────────────────────────────────────
p <- ggplot(data, aes(x = condition,
                      y = signal,
                      fill = cell_type)) +
  # Boxes
  geom_boxplot(width   = 0.55,
               alpha   = 0.60,
               colour  = "black",
               outlier.shape = NA,
               size    = 0.9,
               position = position_dodge(width = 0.8)) +
  # Raw data
  geom_point(aes(colour = cell_type),
             position = position_jitterdodge(
               jitter.width = 0.20,
               dodge.width  = 0.80),
             size   = 2.5,
             alpha  = 0.9,
             stroke = 0.4) +
  # Median emphasised
  stat_summary(fun = median,
               geom = "point",
               shape = 95,         # A nice horizontal bar
               size  = 7,
               colour = "black",
               position = position_dodge(width = 0.8),
               show.legend = FALSE) +
  # Sample size
  geom_text(data = n_df,
            aes(label = paste0("n = ", n),
                y = -20),            # positioned well below x-axis
            position = position_dodge(width = 0.8),
            vjust = 0.5,
            size = 3.5,
            family = "sans") +
            
  # Multiple comparisons
  stat_compare_means(
    comparisons = list(c("Normal culture", "Serum starved")),
    aes(group = cell_type),
    label = "p.format",
    method = "t.test",
    tip.length = 0.01,
    size = 3.5,
    step.increase = 0.08,
    position = position_dodge(width = 0.8)) +
  # Comparison 2: Between cell types within each condition
  stat_compare_means(
    aes(group = condition),
    comparisons = list(c("Wild-type cells", "GPP8 cell line")),
    label = "p.format", 
    method = "t.test",
    tip.length = 0.01,
    size = 3.5,
    step.increase = 0.12,
    bracket.size = 0.5) +
    
  # A journal-compatible palette
  scale_fill_manual(values  = c("#0072B2", "#D55E00"),   # fill
                    name = NULL) +
  scale_colour_manual(values = c("#005293", "#A54B00"),  # outline for points
                      guide = "none") +
                      
  # Axes - extended y-axis to accommodate multiple comparisons and n labels
  scale_y_continuous(limits = c(-25, 180),
                     breaks = seq(0, 150, 50),
                     expand = expansion(mult = c(0, 0.02))) +
  labs(y = "Signal intensity (a.u.)",
       x = NULL) +
       
  # Theme
  theme_classic(base_size = 14, base_family = "sans") +
  theme(
    legend.position   = c(0.75, 0.95),    # moved legend to avoid overlap
    legend.background = element_blank(),
    legend.box.background = element_rect(fill = "white", colour = NA),
    axis.line         = element_line(size = 1),
    axis.ticks        = element_line(size = 0.8),
    axis.text.x       = element_text(colour = "black"),
    axis.text.y       = element_text(colour = "black"),
    plot.margin       = margin(15, 15, 15, 15)
  ) +
  coord_cartesian(clip = "off")

print(p)

#'──────────────────────────────────────────────────────────────
# Save figure ----
#'──────────────────────────────────────────────────────────────
ggsave("boxplot_example_1.png",
       plot   = p,
       width  = 7,        
       height = 6,       
       dpi    = 600)
  
