import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_true_holdout_evolution_pipeline(X, y, years, windows, alpha=1.0):
    """Diagnostic script that combines:

    1. Intrinsic Feature Space Evolution (PCA on full X to track structural
    mutations)
    2. True out-of-sample holdout evaluation (80/20 train/test split per window)
    """
    print("RUNNING TRUE HOLDOUT EVOLUTION & PERFORMANCE PIPELINE\n")

    performance_metrics = []
    pc1_signatures = []

    for start, end in windows:
        # 1. Isolate the data slice for this 5-year rolling window
        mask = (years >= start) & (years <= end)
        X_w = np.array(X[mask])
        y_w = np.array(y[mask])

        # Ensure we have enough rows to safely run an 80/20 train/test split
        if len(y_w) < 15:
            print(
                f"Window {start}-{end} skipped due to low sample size ({len(y_w)} observations)."
            )
            continue

        current_window_name = f"{start}-{end}"

        # =====================================================================
        # 1. IMPLEMENTING HOLDOUT SPLIT 
        # =====================================================================
        # Hide 20% of the data completely from the model during training
        X_train, X_test, y_train, y_test = train_test_split(
            X_w, y_w, test_size=0.20, random_state=42
        )

        # Standardize using training parameters to strictly avoid data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model exclusively on the 80% training pool
        model_real = Ridge(alpha=alpha)
        model_real.fit(X_train_scaled, y_train)

        # Predict strictly on the HIDDEN 20% test data
        y_pred_test = model_real.predict(X_test_scaled)

        # Calculate out-of-sample performance metrics
        true_holdout_r2 = r2_score(y_test, y_pred_test)
        true_window_mse = mean_squared_error(y_test, y_pred_test)
        true_window_rmse = np.sqrt(true_window_mse)

        # =====================================================================
        # 2. PCA Analysis
        # =====================================================================
        # We calculate PCA on the full window's feature space (X_w) because we
        # want to capture the true, absolute geometry of the physical landscape.
        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X_w)

        pca_evolution = PCA()
        pca_evolution.fit(X_full_scaled)

        pc1_var = pca_evolution.explained_variance_ratio_[0]
        cumulative_top3_var = np.sum(pca_evolution.explained_variance_ratio_[:3])

        # Track the exact directional layout (loadings) of the primary component
        pc1_signatures.append(pca_evolution.components_[0])

        # Log both honest metrics and PCA structural details
        performance_metrics.append(
            {
                "window": current_window_name,
                "True_Holdout_R2": true_holdout_r2,
                "True_Window_MSE": true_window_mse,
                "True_Window_RMSE": true_window_rmse,
                "PC1_Variance_Pct": pc1_var * 100,
                "Top3_Variance_Pct": cumulative_top3_var * 100,
            }
        )

    # =====================================================================
    # PRINTING EXECUTION REPORT
    # =====================================================================
    print("OUT-OF-SAMPLE METRICS & PCA EVOLUTION")
    print("-" * 95)
    print(
        f"{'Time Window':<12} | {'True Holdout R2':<16} | {'True MSE':<12} | {'PC1 Var %':<10} | {'Top 3 Cumul %':<12}"
    )
    print("-" * 95)
    for metric in performance_metrics:
        print(
            f"{metric['window']:<12} | {metric['True_Holdout_R2']:>15.4f} | {metric['True_Window_MSE']:>12.4f} | {metric['PC1_Variance_Pct']:>9.2f}% | {metric['Top3_Variance_Pct']:>11.2f}%"
        )
    print("-" * 95 + "\n")

    # =====================================================================
    # PRINTING THE PCA CROSS-TEMPORAL ALIGNMENT
    # =====================================================================
    if len(pc1_signatures) >= 2:
        # Calculate vector correlation between the first era's PC1 and the last era's PC1
        alignment = np.corrcoef(pc1_signatures[0], pc1_signatures[-1])[0, 1]
        print("-" * 80)
        print(
            f"Vector Correlation between the first and last window's PC1 signature: {alignment:.4f}"
        )
        if alignment > 0.85:
            print(
                "INSIGHT: The physical features (X) are stable! Performance decay is driven by off-image, unseen factors."
            )
        else:
            print(
                "INSIGHT: The features (X) are mutating! The landscape on the ground across Africa is physically changing."
            )
        print("-" * 80)

    return pd.DataFrame(performance_metrics)


# =====================================================================
# SIMULATED INTEGRATED RUN
# =====================================================================
if __name__ == "__main__":
    # Generate mock dataset matching the layout
    np.random.seed(42)
    mock_X = np.random.randn(1000, 384)
    mock_y = 2.5 * mock_X[:, 15] + np.random.randn(1000)
    mock_years = np.random.choice([1995, 2000, 2005, 2010], size=1000)
    target_windows = [(1995, 1999), (2000, 2004), (2005, 2009), (2010, 2015)]

    _ = run_true_holdout_evolution_pipeline(
        mock_X, mock_y, mock_years, target_windows
    )