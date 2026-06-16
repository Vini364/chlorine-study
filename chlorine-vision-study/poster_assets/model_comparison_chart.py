import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
models = ['Ridge', 'ElasticNet', 'Lasso', 'Gradient\nBoosting', 'Random\nForest']

cv_r2    = [0.9284, 0.9291, 0.9250, 0.9650, 0.9715]
train_r2 = [0.9555, 0.9570, 0.9538, 0.9996, 0.9985]
hold_r2  = [0.9597, 0.9618, 0.9575, 0.9876, 0.9822]

rmse     = [0.11923, 0.11621, 0.12249, 0.06622, 0.07939]
rmse_colors = ['#2a9d8f', '#6a8fad', '#a8bfc8', '#c1714f', '#a89060']

# ── Layout ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), facecolor='white')
fig.suptitle('Overall Model Comparison', fontsize=22, fontweight='bold', y=1.01)

FONT_TICK  = 18
FONT_LABEL = 20
FONT_TITLE = 18

x = np.arange(len(models))
w = 0.26  # bar width

# ── Chart 1: R² Grouped Bar ────────────────────────────────────────────────────
b1 = ax1.bar(x - w, cv_r2,    width=w, label='CV R²',      color='#c8d4c8', zorder=3)
b2 = ax1.bar(x,     train_r2, width=w, label='Train R²',   color='#1a3a3a', zorder=3)
b3 = ax1.bar(x + w, hold_r2,  width=w, label='Holdout R²', color='#2a9d8f', zorder=3)

ax1.set_title('R² Across Evaluation Stages', fontsize=FONT_TITLE, pad=10)
ax1.set_ylim(0.90, 1.00)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=FONT_TICK)
ax1.tick_params(axis='y', labelsize=FONT_TICK)
ax1.yaxis.grid(True, color='#e0e0e0', zorder=0)
ax1.set_axisbelow(True)
ax1.spines[['top','right','left','bottom']].set_visible(False)
ax1.tick_params(left=False, bottom=False)

legend_patches = [
    mpatches.Patch(color='#c8d4c8', label='CV R²'),
    mpatches.Patch(color='#1a3a3a', label='Train R²'),
    mpatches.Patch(color='#2a9d8f', label='Holdout R²'),
]
ax1.legend(handles=legend_patches, fontsize=FONT_TICK, frameon=False, loc='upper left')

# ── Chart 2: Holdout RMSE ────────────────────────────────────────────────────
bars = ax2.bar(x, rmse, width=0.8, color=rmse_colors, zorder=3)

for bar, val in zip(bars, rmse):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.002,
             f'{val:.3f}',
             ha='center', va='bottom', fontsize=FONT_TICK, color='#222')

ax2.set_title('Holdout RMSE', fontsize=FONT_TITLE, pad=10)
ax2.set_ylim(0, 0.15)
ax2.set_ylabel('mg/L', fontsize=FONT_LABEL)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=FONT_TICK)
ax2.tick_params(axis='y', labelsize=FONT_TICK)
ax2.yaxis.grid(True, color='#e0e0e0', zorder=0)
ax2.set_axisbelow(True)
ax2.spines[['top','right','left','bottom']].set_visible(False)
ax2.tick_params(left=False, bottom=False)

# ── Export ────────────────────────────────────────────────────────────────────
output_path = Path(__file__).with_name('model_comparison.png')

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', transparent=False)
plt.show()
print(f"Saved: {output_path}")
