"""
Quick script to regenerate the plot with corrected labels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Data from the completed experiment
results_dict = {
    'Interval-1 [0-20%]': {
        0.0: {'road': 0.0108, 'std': 0.0075},
        1.0: {'road': 0.0118, 'std': 0.0086},
        3.0: {'road': 0.0151, 'std': 0.0121},
        5.0: {'road': 0.0236, 'std': 0.0205}
    },
    'Interval-2 [20-40%]': {
        0.0: {'road': 0.0114, 'std': 0.0081},
        1.0: {'road': 0.0114, 'std': 0.0080},
        3.0: {'road': 0.0133, 'std': 0.0095},
        5.0: {'road': 0.0166, 'std': 0.0125}
    },
    'Interval-3 [40-60%]': {
        0.0: {'road': 0.0111, 'std': 0.0077},
        1.0: {'road': 0.0110, 'std': 0.0079},
        3.0: {'road': 0.0110, 'std': 0.0077},
        5.0: {'road': 0.0094, 'std': 0.0056}
    },
    'Interval-4 [60-80%]': {
        0.0: {'road': 0.0112, 'std': 0.0078},
        1.0: {'road': 0.0112, 'std': 0.0079},
        3.0: {'road': 0.0124, 'std': 0.0089},
        5.0: {'road': 0.0118, 'std': 0.0081}
    },
    'Interval-5 [80-100%]': {
        0.0: {'road': 0.0111, 'std': 0.0078},
        1.0: {'road': 0.0112, 'std': 0.0078},
        3.0: {'road': 0.0109, 'std': 0.0073},
        5.0: {'road': 0.0107, 'std': 0.0069}
    },
    'Baseline': {
        0.0: {'road': 0.0108, 'std': 0.0078},
        1.0: {'road': 0.0110, 'std': 0.0078},
        3.0: {'road': 0.0110, 'std': 0.0075},
        5.0: {'road': 0.0109, 'std': 0.0076}
    }
}

# Plot with corrected labels
fig, ax = plt.subplots(figsize=(14, 8))

# CORRECTED: Interval-1 = Top (Highest), Interval-5 = Bottom (Lowest)
strategies_config = {
    'Interval-1 [0-20%]': {'color': '#E74C3C', 'linewidth': 3.0, 'label': '[0-20%] Top (Highest Activation)'},
    'Interval-2 [20-40%]': {'color': '#F39C12', 'linewidth': 2.8, 'label': '[20-40%] Upper-Mid'},
    'Interval-3 [40-60%]': {'color': '#3498DB', 'linewidth': 2.8, 'label': '[40-60%] Middle'},
    'Interval-4 [60-80%]': {'color': '#2ECC71', 'linewidth': 2.8, 'label': '[60-80%] Lower-Mid'},
    'Interval-5 [80-100%]': {'color': '#8E44AD', 'linewidth': 3.0, 'label': '[80-100%] Bottom (Lowest Activation)'},
    'Baseline': {'color': '#34495E', 'linewidth': 2.5, 'label': 'Baseline (No Noise)', 'linestyle': '--'},
}

for strategy_name, style in strategies_config.items():
    if strategy_name not in results_dict:
        continue

    strategy_results = results_dict[strategy_name]
    noise_levels = np.array(sorted(strategy_results.keys()))
    road_scores = np.array([strategy_results[noise]['road'] for noise in noise_levels])

    # Create smooth curve
    if len(noise_levels) > 3 and strategy_name != 'Baseline':
        noise_smooth = np.linspace(noise_levels.min(), noise_levels.max(), 300)
        try:
            spl = make_interp_spline(noise_levels, road_scores, k=3)
            road_smooth = spl(noise_smooth)
            ax.plot(noise_smooth, road_smooth,
                   color=style['color'],
                   linewidth=style['linewidth'],
                   label=style['label'],
                   linestyle=style.get('linestyle', '-'))
        except:
            ax.plot(noise_levels, road_scores,
                   color=style['color'],
                   linewidth=style['linewidth'],
                   label=style['label'],
                   linestyle=style.get('linestyle', '-'))
    else:
        ax.plot(noise_levels, road_scores,
               color=style['color'],
               linewidth=style['linewidth'],
               label=style['label'],
               linestyle=style.get('linestyle', '-'))

# Styling
ax.set_xlabel('Noise Strength (σ)', fontsize=16, fontweight='bold')
ax.set_ylabel('ROAD Score (Explanation Quality)', fontsize=16, fontweight='bold')
ax.set_title('Impact of Channel Noise on Explainability: 5-Interval Analysis (CORRECTED)',
            fontsize=17, fontweight='bold', pad=20)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_axisbelow(True)

ax.legend(loc='upper left', fontsize=11, framealpha=0.95,
         edgecolor='gray', ncol=1)

ax.set_xlim([0, 5])
ax.tick_params(axis='both', which='major', labelsize=13)

ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
save_path = './results/channel_noise_sensitivity_5intervals_activation_CORRECTED.png'
plt.savefig(save_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"✅ Corrected plot saved to: {save_path}")
print("\nCorrected interpretation:")
print("  🔴 Red (Top 20%, Highest Activation): ROAD increases most → Most important!")
print("  🟠 Orange (Upper-Mid 20%): Moderate increase")
print("  🔵 Blue (Middle 20%): Slightly decreases → Better than baseline!")
print("  🟢 Green (Lower-Mid 20%): Stable")
print("  🟣 Purple (Bottom 20%, Lowest Activation): Stable → Least important!")
