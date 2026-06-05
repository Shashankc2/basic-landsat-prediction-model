import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def run_unified_robustness_pipeline(X, y, years, windows, alpha=1.0):
    """Executes rolling window analyses, running PCA on real data to test

    multicollinearity, and running a separate placebo shuffle to isolate noise.
    """
    window_names = [f"{start}-{end}" for start, end in windows]

    # Pre-allocate matrices to track weights across time windows
    real_standard_coef_matrix = []
    real_pca_loadings_matrix = []
    placebo_coef_matrix = []

    print("Running Combined Pipeline...")

    for start, end in windows:
        # 1. Filter for the 5-year rolling window
        mask = (years >= start) & (years <= end)
        X_w = np.array(X[mask])
        y_w = np.array(y[mask])

        if len(y_w) < 10:
            continue

        # 2. Standardize features for this window
        scaler = StandardScaler()
        X_w_scaled = scaler.fit_transform(X_w)

        # RUN 1: THE REAL DATA ANALYSIS (Your Baseline)
        model_real = Ridge(alpha=alpha)
        model_real.fit(X_w_scaled, y_w)
        real_standard_coef_matrix.append(model_real.coef_)

        # RUN 2: THE PCA TEST ON REAL DATA (To check Multicollinearity)
        pca = PCA(n_components=0.95)
        X_w_pca = pca.fit_transform(X_w_scaled)

        model_pca = Ridge(alpha=alpha)
        model_pca.fit(X_w_pca, y_w)

        # Convert PCA weights back to 384-dimension space so we can map paths
        pca_loadings_back = np.dot(model_pca.coef_, pca.components_)
        real_pca_loadings_matrix.append(pca_loadings_back)

        # RUN 3: THE PLACEBO SHUFFLE (To isolate algorithmic noise)
        y_w_shuffled = np.random.permutation(y_w)

        model_placebo = Ridge(alpha=alpha)
        model_placebo.fit(X_w_scaled, y_w_shuffled)
        placebo_coef_matrix.append(model_placebo.coef_)

    # Convert lists to numpy arrays for volatility calculations
    real_std_coefs = np.array(real_standard_coef_matrix)
    real_pca_coefs = np.array(real_pca_loadings_matrix)
    fake_shf_coefs = np.array(placebo_coef_matrix)

    # Calculate Volatility (Standard Deviation over time windows)
    vol_real = np.std(real_std_coefs, axis=0)
    vol_pca = np.std(real_pca_coefs, axis=0)
    vol_placebo = np.std(fake_shf_coefs, axis=0)

    # Pull the top 5 most volatile/important features from the real model
    top_5_real_indices = np.argsort(vol_real)[-5:]

    print("\nMASTER COEFFICIENT VOLATILITY ANALYSIS")
    print("-" * 80)
    print(
        f"{'Feature Index':<15} | {'Real Volatility':<18} | {'PCA (Real) Vol':<18} | {'Placebo Vol':<15}"
    )
    print("-" * 80)
    for idx in reversed(top_5_real_indices):
        print(
            f"Feature_{idx:<8} | {vol_real[idx]:<18.4f} | {vol_pca[idx]:<18.4f} | {vol_placebo[idx]:<15.4f}"
        )
    print("-" * 80)

    return {
        "windows": window_names,
        "top_5": top_5_real_indices,
        "real_paths": real_std_coefs,
        "pca_paths": real_pca_coefs,
        "placebo_paths": fake_shf_coefs,
    }


if __name__ == "__main__":
    np.random.seed(42)
    mock_X = np.random.randn(1000, 384)
    mock_y = 3.0 * mock_X[:, 77] - 2.0 * mock_X[:, 279] + np.random.randn(1000)
    mock_years = np.random.choice([1995, 2000, 2005, 2010], size=1000)
    target_windows = [(1995, 1999), (2000, 2004), (2005, 2009), (2010, 2015)]

    outputs = run_unified_robustness_pipeline(
        mock_X, mock_y, mock_years, target_windows
    )