import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD PRE-CALCULATED DATA
coef_df = pd.read_csv('coefficients_master.csv', index_col=0)

# 2. CALCULATE STABILITY
stds = coef_df.std() 
top_5 = coef_df.abs().mean().sort_values(ascending=False).head(5).index

# 3. PLOT VOLATILITY HISTOGRAM
plt.figure(figsize=(10, 6))
sns.histplot(stds, kde=True, color='lightgray', bins=30)

colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, feature in enumerate(top_5):
    plt.axvline(stds[feature], color=colors[i], linestyle='--', label=f"{feature} (Std: {stds[feature]:.2f})")

plt.title("Stability Comparison: Top 5 vs. All Features")
plt.xlabel("Volatility (Standard Deviation of Coefficient)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig('volatility_histogram.png', dpi=300)
print("Saved: volatility_histogram.png")
plt.show()