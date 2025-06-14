library(ggpubr)
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(123)

data <- data.frame(
  condition = rep(c("Serum starved", "Normal culture"), each = 8),
  cell_type = rep(c(rep("Wild-type cells", 4), rep("GPP8 cell line", 4)), 2),
  signal = c(
    rnorm(4, mean = 40, sd = 8),
    rnorm(4, mean = 110, sd = 10),
    rnorm(4, mean = 25, sd = 5),
    rnorm(4, mean = 35, sd = 7)
  )
)

data$condition <- factor(data$condition, levels = c("Normal culture", "Serum starved"))
data$cell_type <- factor(data$cell_type, levels = c("Wild-type cells", "GPP8 cell line"))

summary_data <- data %>%
  group_by(condition, cell_type) %>%
  summarise(
    mean_signal = mean(signal),
    sd = sd(signal),
    se = sd(signal) / sqrt(n()),
    .groups = 'drop'
  )

p <- ggplot(data, aes(x = condition, y = signal, fill = cell_type)) +
  stat_summary(
    fun = mean,
    geom = "bar",
    position = position_dodge(width = 0.8),
    width = 0.7,
    color = NA
  ) +
  stat_summary(
    fun.data = mean_sdl,
    fun.args = list(mult = 1),
    geom = "errorbar",
    position = position_dodge(width = 0.8),
    width = 0.2,
    size = 0.8
  ) +
  geom_point(
    position = position_jitterdodge(
      jitter.width = 0.15,
      dodge.width = 0.8
    ),
    size = 2.5,
    shape = 16,
    color = "black"
  ) +
  scale_fill_manual(
    values = c("Wild-type cells" = "#0080FF", "GPP8 cell line" = "#FF0000"),
    labels = c("Wild-type cells", "GPP8 cell line")
  ) +
  scale_y_continuous(
    limits = c(0, 150),
    breaks = seq(0, 150, 50),
    expand = c(0, 0)
  ) +
  labs(
    y = "Signal",
    x = NULL,
    fill = NULL
  ) +
  theme_classic(base_family = "sans") +
  theme(
    legend.position = c(0.85, 0.9),
    legend.background = element_blank(),
    legend.key = element_blank(),
    legend.key.size = unit(0.8, "lines"),
    legend.text = element_text(size = 14, family = "sans"),
    legend.spacing.y = unit(0.1, "cm"),
    
    axis.line = element_line(color = "black", size = 1),
    axis.ticks = element_line(color = "black", size = 0.8),
    axis.text = element_text(size = 14, color = "black", face = "plain", family = "sans"),
    axis.title.y = element_text(size = 14, color = "black", face = "plain", family = "sans"),
    
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    plot.background = element_blank(),
    plot.margin = margin(10, 10, 10, 10)
  )

print(p)

p2 <- ggbarplot(
  data,
  x = "condition",
  y = "signal",
  fill = "cell_type",
  color = NA,
  palette = c("#0080FF", "#FF0000"),
  add = c("mean_sd", "jitter"),
  add.params = list(
    size = 0.8,
    width = 0.2,
    jitter = 0.15,
    color = "black"
  ),
  position = position_dodge(0.8),
  width = 0.7,
  legend = "right",
  font.family = "sans"
) +
  scale_y_continuous(
    limits = c(0, 150),
    breaks = seq(0, 150, 50),
    expand = c(0, 0)
  ) +
  labs(y = "Signal") +
  theme(
    legend.position = c(0.85, 0.9),
    legend.title = element_blank(),
    axis.title.x = element_blank(),
    axis.line = element_line(color = "black", size = 1),
    axis.ticks = element_line(color = "black", size = 0.8),
    text = element_text(family = "sans", size = 14)
  )

print("Data summary:")
print(summary_data)

ggsave("barplot.png", plot = p, width = 6, height = 5, dpi = 300)

