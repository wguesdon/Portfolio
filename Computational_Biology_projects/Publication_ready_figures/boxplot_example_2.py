#'──────────────────────────────────────────────────────────────
# Packages ----
#'──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

#'──────────────────────────────────────────────────────────────
# Data generation ----
#'──────────────────────────────────────────────────────────────
np.random.seed(123)

data = pd.DataFrame({
    'condition': ['Serum starved']*8 + ['Normal culture']*8,
    'cell_type': ['Wild-type cells']*4 + ['GPP8 cell line']*4 + ['Wild-type cells']*4 + ['GPP8 cell line']*4,
    'signal': np.concatenate([
        np.random.normal(40, 8, 4),
        np.random.normal(110, 10, 4),
        np.random.normal(25, 5, 4),
        np.random.normal(35, 7, 4)
    ])
})

data['condition'] = pd.Categorical(data['condition'], 
                                   categories=['Normal culture', 'Serum starved'], 
                                   ordered=True)
data['cell_type'] = pd.Categorical(data['cell_type'], 
                                   categories=['Wild-type cells', 'GPP8 cell line'], 
                                   ordered=True)

n_df = data.groupby(['condition', 'cell_type']).size().reset_index(name='n')

#'──────────────────────────────────────────────────────────────
# Plot setup ----
#'──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

colors = {'Wild-type cells': '#0072B2', 'GPP8 cell line': '#D55E00'}
edge_colors = {'Wild-type cells': '#005293', 'GPP8 cell line': '#A54B00'}

positions = np.array([0, 1.5])
width = 0.3
dodge = 0.35

#'──────────────────────────────────────────────────────────────
# Boxplot data assembly ----
#'──────────────────────────────────────────────────────────────
box_data = []
positions_list = []
for i, (pos, condition) in enumerate(zip(positions, ['Normal culture', 'Serum starved'])):
    for j, cell_type in enumerate(['Wild-type cells', 'GPP8 cell line']):
        subset = data[(data['condition'] == condition) & (data['cell_type'] == cell_type)]
        box_data.append(subset['signal'].values)
        positions_list.append(pos + (j - 0.5) * dodge)

#'──────────────────────────────────────────────────────────────
# Boxplot rendering ----
#'──────────────────────────────────────────────────────────────
bp = ax.boxplot(box_data, positions=positions_list, widths=width, 
                patch_artist=True, showfliers=False,
                boxprops=dict(linewidth=0.9),
                whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9),
                medianprops=dict(linewidth=0, color='none'))

for i, patch in enumerate(bp['boxes']):
    cell_type = 'Wild-type cells' if i % 2 == 0 else 'GPP8 cell line'
    patch.set_facecolor(colors[cell_type])
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')

#'──────────────────────────────────────────────────────────────
# Jittered data points ----
#'──────────────────────────────────────────────────────────────
for i, (pos, condition) in enumerate(zip(positions, ['Normal culture', 'Serum starved'])):
    for j, cell_type in enumerate(['Wild-type cells', 'GPP8 cell line']):
        subset = data[(data['condition'] == condition) & (data['cell_type'] == cell_type)]
        x_base = pos + (j - 0.5) * dodge
        x_jitter = np.random.normal(0, 0.04, len(subset))
        ax.scatter(x_base + x_jitter, subset['signal'], 
                  color=edge_colors[cell_type], s=50, alpha=0.9, 
                  edgecolors='black', linewidths=0.4, zorder=3)

#'──────────────────────────────────────────────────────────────
# Median bars ----
#'──────────────────────────────────────────────────────────────
for i, box_d in enumerate(box_data):
    median_val = np.median(box_d)
    x_pos = positions_list[i]
    ax.plot([x_pos - width/2.2, x_pos + width/2.2], [median_val, median_val], 
            'k-', linewidth=2.5, zorder=4)

#'──────────────────────────────────────────────────────────────
# Sample size annotations ----
#'──────────────────────────────────────────────────────────────
for i, (pos, condition) in enumerate(zip(positions, ['Normal culture', 'Serum starved'])):
    for j, cell_type in enumerate(['Wild-type cells', 'GPP8 cell line']):
        n = n_df[(n_df['condition'] == condition) & (n_df['cell_type'] == cell_type)]['n'].values[0]
        x_pos = pos + (j - 0.5) * dodge
        ax.text(x_pos, -15, f'n = {n}', ha='center', va='center', fontsize=9)

#'──────────────────────────────────────────────────────────────
# Statistical annotations ----
#'──────────────────────────────────────────────────────────────
def add_stat_annotation(ax, x1, x2, y, h, text='p < 0.05', tip_length=0.03):
    ax.plot([x1, x1], [y-tip_length*5, y+h], 'k-', linewidth=0.8)
    ax.plot([x1, x2], [y+h, y+h], 'k-', linewidth=0.8)
    ax.plot([x2, x2], [y+h, y-tip_length*5], 'k-', linewidth=0.8)
    ax.text((x1+x2)/2, y+h+2, text, ha='center', va='bottom', fontsize=9)

y_max = 130
y_start1 = y_max + 5

normal_wt = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'Wild-type cells')]['signal']
normal_gpp8 = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'GPP8 cell line')]['signal']
_, p_normal = stats.ttest_ind(normal_wt, normal_gpp8)
p_text_normal = f'p = {p_normal:.3f}' if p_normal >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[0] - 0.5*dodge, positions[0] + 0.5*dodge, y_start1, 3, p_text_normal)

serum_wt = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'Wild-type cells')]['signal']
serum_gpp8 = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'GPP8 cell line')]['signal']
_, p_serum = stats.ttest_ind(serum_wt, serum_gpp8)
p_text_serum = f'p = {p_serum:.3f}' if p_serum >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[1] - 0.5*dodge, positions[1] + 0.5*dodge, y_start1, 3, p_text_serum)

y_start2 = y_start1 + 15

wt_normal = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'Wild-type cells')]['signal']
wt_serum = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'Wild-type cells')]['signal']
_, p_wt = stats.ttest_ind(wt_normal, wt_serum)
p_text_wt = f'p = {p_wt:.3f}' if p_wt >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[0] - 0.5*dodge, positions[1] - 0.5*dodge, y_start2, 4, p_text_wt)

y_start3 = y_start2 + 12
gpp8_normal = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'GPP8 cell line')]['signal']
gpp8_serum = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'GPP8 cell line')]['signal']
_, p_gpp8 = stats.ttest_ind(gpp8_normal, gpp8_serum)
p_text_gpp8 = f'p = {p_gpp8:.3f}' if p_gpp8 >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[0] + 0.5*dodge, positions[1] + 0.5*dodge, y_start3, 4, p_text_gpp8)

#'──────────────────────────────────────────────────────────────
# Plot styling ----
#'──────────────────────────────────────────────────────────────
ax.set_ylim(-25, 180)
ax.set_yticks(range(0, 151, 50))
ax.set_ylabel('Signal intensity (a.u.)', fontsize=12)
ax.set_xticks(positions)
ax.set_xticklabels(['Normal culture', 'Serum starved'], fontsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.tick_params(width=0.8, length=5)
ax.tick_params(axis='both', colors='black')

legend_elements = [mpatches.Patch(facecolor=colors['Wild-type cells'], 
                                 alpha=0.6, edgecolor='black', 
                                 label='Wild-type cells'),
                  mpatches.Patch(facecolor=colors['GPP8 cell line'], 
                                 alpha=0.6, edgecolor='black', 
                                 label='GPP8 cell line')]
ax.legend(handles=legend_elements, loc='center left', 
          bbox_to_anchor=(1.02, 0.5), frameon=False)

ax.set_xlim(-0.7, 2.2)

#'──────────────────────────────────────────────────────────────
# Output ----
#'──────────────────────────────────────────────────────────────
plt.tight_layout()
plt.subplots_adjust(right=0.85)
plt.savefig('boxplot_publication.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

