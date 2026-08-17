import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14

# N = Historical Training Size 
N_train = np.array([2201, 8539, 18691]) 
# Out-of-Sample R2 on Future Era
R2_future = np.array([0.6801, 0.6096, 0.5726])

# Fit an Empirical Log-Linear Scaling Function: R² = a * log(N) + b
log_N = np.log(N_train)
slope, intercept = np.polyfit(log_N, R2_future, 1)

# Generate fine grid for smooth scaling line
N_dense = np.linspace(N_train.min(), N_train.max(), 200)
R2_fit = slope * np.log(N_dense) + intercept

# Create Theoretical Counterfactual (Monotonic Improvement)
# Simulates what standard ML scaling assumes should happen as N grows
R2_standard_ai = 0.6801 + 0.05 * (np.log(N_dense) - np.log(N_train[0]))

# 4. Canvas Setup
plt.figure(figsize=(9, 6), dpi=300)

# Plot Empirical Anti-Scaling Curve
plt.scatter(N_train, R2_future, color='#d62728', s=100, zorder=5, label='Observed Data (Temporal Forward-Chaining)')
plt.plot(N_dense, R2_fit, color='#d62728', linestyle='-', linewidth=2.5, 
         label=f'Empirical Fit ($\Delta R^2 / \Delta \log(N) = {slope:.4f}$)')

# Plot Theoretical Standard AI Scaling Law (Hypothetical)
plt.plot(N_dense, R2_standard_ai, color='#2ca02c', linestyle='--', linewidth=2.0, alpha=0.8,
         label='Standard AI Paradigm (Kaplan / Chinchilla Assumption)')

# 5. Formatting
plt.xscale('log') # Log scale on X-axis is standard for AI scaling papers
plt.xticks(N_train, labels=['2.2K\n(1995-1999)', '8.5K\n(1995-2004)', '18.7K\n(1995-2009)'])
plt.xlabel('Cumulative Training Set Size $N$ (Log Scale)', labelpad=12)
plt.ylabel('Out-of-Sample Predictive $R^2$', labelpad=12)
plt.title('The Anti-Scaling Law in Non-Stationary Environments', pad=18, weight='bold')

plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.savefig('scaling_law_temporal_decay.png', dpi=300)
print(f"Plot saved successfully! Slope alpha: {slope:.4f}")