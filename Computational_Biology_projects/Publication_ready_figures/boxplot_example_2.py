import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(123)

# Create example data
data = pd.DataFrame({
    'condition': ['Serum starved']*8 + ['Normal culture']*8,
    'cell_type': ['Wild-type cells']*4 + ['GPP8 cell line']*4 + ['Wild-type cells']*4 + ['GPP8 cell line']*4,
    'signal': np.concatenate([
        np.random.normal(40, 8, 4),    # Serum starved, Wild-type
        np.random.normal(110, 10, 4),  # Serum starved, GPP8
        np.random.normal(25, 5, 4),    # Normal culture, Wild-type
        np.random.normal(35, 7, 4)     # Normal culture, GPP8
    ])
})

# Convert to categorical with specific order
data['condition'] = pd.Categorical(data['condition'], 
                                   categories=['Normal culture', 'Serum starved'], 
                                   ordered=True)
data['cell_type'] = pd.Categorical(data['cell_type'], 
                                   categories=['Wild-type cells', 'GPP8 cell line'], 
                                   ordered=True)

# Calculate sample sizes
n_df = data.groupby(['condition', 'cell_type']).size().reset_index(name='n')

# Set up the figure
fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

# Define colors
colors = {'Wild-type cells': '#0072B2', 'GPP8 cell line': '#D55E00'}
edge_colors = {'Wild-type cells': '#005293', 'GPP8 cell line': '#A54B00'}

# Create positions for dodge with more spacing
positions = np.array([0, 1.5])  # Increased spacing between conditions
width = 0.3  # Slightly narrower boxes
dodge = 0.35  # Increased dodge for more separation

# Plot boxplots
box_data = []
positions_list = []
for i, (pos, condition) in enumerate(zip(positions, ['Normal culture', 'Serum starved'])):
    for j, cell_type in enumerate(['Wild-type cells', 'GPP8 cell line']):
        subset = data[(data['condition'] == condition) & (data['cell_type'] == cell_type)]
        box_data.append(subset['signal'].values)
        positions_list.append(pos + (j - 0.5) * dodge)

# Create boxplots
bp = ax.boxplot(box_data, positions=positions_list, widths=width, 
                patch_artist=True, showfliers=False,
                boxprops=dict(linewidth=0.9),
                whiskerprops=dict(linewidth=0.9),
                capprops=dict(linewidth=0.9),
                medianprops=dict(linewidth=0, color='none'))  # Hide default median line

# Color the boxes
for i, patch in enumerate(bp['boxes']):
    cell_type = 'Wild-type cells' if i % 2 == 0 else 'GPP8 cell line'
    patch.set_facecolor(colors[cell_type])
    patch.set_alpha(0.6)
    patch.set_edgecolor('black')

# Add jittered points
for i, (pos, condition) in enumerate(zip(positions, ['Normal culture', 'Serum starved'])):
    for j, cell_type in enumerate(['Wild-type cells', 'GPP8 cell line']):
        subset = data[(data['condition'] == condition) & (data['cell_type'] == cell_type)]
        x_base = pos + (j - 0.5) * dodge
        x_jitter = np.random.normal(0, 0.04, len(subset))
        ax.scatter(x_base + x_jitter, subset['signal'], 
                  color=edge_colors[cell_type], s=50, alpha=0.9, 
                  edgecolors='black', linewidths=0.4, zorder=3)

# Add median emphasis (horizontal bars)
for i, box_d in enumerate(box_data):
    median_val = np.median(box_d)
    x_pos = positions_list[i]
    ax.plot([x_pos - width/2.2, x_pos + width/2.2], [median_val, median_val], 
            'k-', linewidth=2.5, zorder=4)

# Add sample size annotations
for i, (pos, condition) in enumerate(zip(positions, ['Normal culture', 'Serum starved'])):
    for j, cell_type in enumerate(['Wild-type cells', 'GPP8 cell line']):
        n = n_df[(n_df['condition'] == condition) & (n_df['cell_type'] == cell_type)]['n'].values[0]
        x_pos = pos + (j - 0.5) * dodge
        ax.text(x_pos, -15, f'n = {n}', ha='center', va='center', fontsize=9)

# Statistical comparisons with better positioning
def add_stat_annotation(ax, x1, x2, y, h, text='p < 0.05', tip_length=0.03):
    """Add statistical comparison brackets with tips"""
    # Draw bracket with small tips
    ax.plot([x1, x1], [y-tip_length*5, y+h], 'k-', linewidth=0.8)
    ax.plot([x1, x2], [y+h, y+h], 'k-', linewidth=0.8)
    ax.plot([x2, x2], [y+h, y-tip_length*5], 'k-', linewidth=0.8)
    ax.text((x1+x2)/2, y+h+2, text, ha='center', va='bottom', fontsize=9)

# Get data ranges for positioning brackets
y_max = 130  # Fixed position for better control

# Comparison 1: Between cell types within each condition (lower brackets)
y_start1 = y_max + 5

# Normal culture comparison
normal_wt = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'Wild-type cells')]['signal']
normal_gpp8 = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'GPP8 cell line')]['signal']
_, p_normal = stats.ttest_ind(normal_wt, normal_gpp8)
p_text_normal = f'p = {p_normal:.3f}' if p_normal >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[0] - 0.5*dodge, positions[0] + 0.5*dodge, y_start1, 3, p_text_normal)

# Serum starved comparison  
serum_wt = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'Wild-type cells')]['signal']
serum_gpp8 = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'GPP8 cell line')]['signal']
_, p_serum = stats.ttest_ind(serum_wt, serum_gpp8)
p_text_serum = f'p = {p_serum:.3f}' if p_serum >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[1] - 0.5*dodge, positions[1] + 0.5*dodge, y_start1, 3, p_text_serum)

# Comparison 2: Between conditions for each cell type (upper brackets)
y_start2 = y_start1 + 15

# Wild-type comparison
wt_normal = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'Wild-type cells')]['signal']
wt_serum = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'Wild-type cells')]['signal']
_, p_wt = stats.ttest_ind(wt_normal, wt_serum)
p_text_wt = f'p = {p_wt:.3f}' if p_wt >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[0] - 0.5*dodge, positions[1] - 0.5*dodge, y_start2, 4, p_text_wt)

# GPP8 comparison (highest bracket)
y_start3 = y_start2 + 12
gpp8_normal = data[(data['condition'] == 'Normal culture') & (data['cell_type'] == 'GPP8 cell line')]['signal']
gpp8_serum = data[(data['condition'] == 'Serum starved') & (data['cell_type'] == 'GPP8 cell line')]['signal']
_, p_gpp8 = stats.ttest_ind(gpp8_normal, gpp8_serum)
p_text_gpp8 = f'p = {p_gpp8:.3f}' if p_gpp8 >= 0.001 else 'p < 0.001'
add_stat_annotation(ax, positions[0] + 0.5*dodge, positions[1] + 0.5*dodge, y_start3, 4, p_text_gpp8)

# Styling
ax.set_ylim(-25, 180)
ax.set_yticks(range(0, 151, 50))
ax.set_ylabel('Signal intensity (a.u.)', fontsize=12)
ax.set_xticks(positions)
ax.set_xticklabels(['Normal culture', 'Serum starved'], fontsize=12)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.tick_params(width=0.8, length=5)

# Make tick labels black
ax.tick_params(axis='both', colors='black')

# Add legend in upper left corner to avoid collision with stats
legend_elements = [mpatches.Patch(facecolor=colors['Wild-type cells'], 
                                 alpha=0.6, edgecolor='black', 
                                 label='Wild-type cells'),
                  mpatches.Patch(facecolor=colors['GPP8 cell line'], 
                                 alpha=0.6, edgecolor='black', 
                                 label='GPP8 cell line')]
ax.legend(handles=legend_elements, loc='upper left', 
          bbox_to_anchor=(0.02, 0.98), frameon=False)

# Set x-axis limits to center the plot
ax.set_xlim(-0.7, 2.2)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('boxplot_publication.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()
