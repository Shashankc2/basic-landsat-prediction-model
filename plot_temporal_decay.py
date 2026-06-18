import matplotlib.pyplot as plt
import numpy as np

# Set clean, professional plotting style parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14

# 1. Define the Timeline Eras
eras = ['1995-1999', '2000-2004', '2005-2009', '2010-2015']
x_indices = np.arange(len(eras))

# 2. Input your exact empirical findings
real_within_r2 = [0.7611, 0.7453, 0.6378, 0.6357]
placebo_within_r2 = [-0.0699, -0.0337, -0.0289, -0.0165]
real_forward_r2 = [None, 0.6801, 0.6096, 0.5726]
placebo_forward_r2 = [None, -0.0659, -0.0236, -0.0224]

# 3. Create the Canvas
plt.figure(figsize=(10, 6), dpi=300)

# 4. Plot Real Models (Shades of Blue/Teal)
plt.plot(x_indices, real_within_r2, color='#1f77b4', linestyle='-', marker='o', 
         linewidth=2.5, markersize=8, label='Within-Era Baseline (Analysis 1)')

# Mask out the None value for a clean forward-chaining line
plt.plot(x_indices[1:], real_forward_r2[1:], color='#4682b4', linestyle='--', marker='s', 
         linewidth=2.5, markersize=8, label='Forward-Chaining Forecast (Analysis 3)')

# 5. Plot Placebo Models (Shades of Red/Crimson near 0.00)
plt.plot(x_indices, placebo_within_r2, color='#d62728', linestyle=':', marker='x', 
         linewidth=1.8, markersize=8, label='Placebo Within-Era')
plt.plot(x_indices[1:], placebo_forward_r2[1:], color='#e74c3c', linestyle='-.', marker='^', 
         linewidth=1.8, markersize=8, label='Placebo Forward-Chaining')

# 6. Formatting Axes, Labels, and Structural Elements
plt.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5) # Zero baseline line
plt.xticks(x_indices, eras)

# FIX: Tightened boundaries to remove excessive whitespace on the left and right
plt.xlim(-0.15, len(eras) - 0.85)
plt.ylim(-0.15, 1.0) 

plt.title('Temporal Decay Paradox vs. Placebo Baseline', pad=20, weight='bold')
plt.xlabel('Temporal Analytical Window / Era', labelpad=12)
plt.ylabel('Model Performance ($R^2$ Score)', labelpad=12)

# Grid setup for easy value scanning
plt.grid(True, linestyle='--', alpha=0.3, which='both')

# FIX: Relocated legend to the center-right empty space to prevent overlapping data points
plt.legend(loc='center right', frameon=True, facecolor='white', edgecolor='none', shadow=False)

# Clean Layout & Export
plt.tight_layout()
plt.savefig('temporal_decay_chart.png', dpi=300)
print("Graph successfully re-generated and saved as 'temporal_decay_chart.png'")