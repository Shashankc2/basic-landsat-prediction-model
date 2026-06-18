import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_comprehensive_temporal_analysis(X, y, years, windows, alpha=1.0):
    """
    Master Pipeline for Cross-Temporal Interpretability and Performance.
    
    Contains:
      1. Within-Window True Holdout Evaluation (Honest R2/MSE tracking)
      2. Intrinsic Feature Space Evolution (PCA Camera & Vector Alignment)
      3. Forward-Chaining Backtest (Cumulative Train Past -> Test Future)
    """
    print("=" * 95)
    print("STARTING MASTER CROSS-TEMPORAL EVALUATION PIPELINE")
    print("=" * 95 + "\n")

    within_window_metrics = []
    pc1_signatures = []

    # =====================================================================
    # ANALYSIS 1 & 2: WITHIN-WINDOW HOLDOUT & PCA GEOMETRY
    # =====================================================================
    print("ANALYSIS 1 & 2: WITHIN-WINDOW HOLDOUT & PCA STRUCTURAL SNAPSHOTS")
    print("-" * 95)

    for start, end in windows:
        mask = (years >= start) & (years <= end)
        X_w = np.array(X[mask])
        y_w = np.array(y[mask])

        if len(y_w) < 15:
            print(f"Window {start}-{end} skipped: low sample size ({len(y_w)} Obs).")
            continue

        current_window_name = f"{start}-{end}"

        # 1. True Holdout Split (Within the window)
        X_train, X_test, y_train, y_test = train_test_split(
            X_w, y_w, test_size=0.20, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_real = Ridge(alpha=alpha)
        model_real.fit(X_train_scaled, y_train)

        y_pred_test = model_real.predict(X_test_scaled)
        true_holdout_r2 = r2_score(y_test, y_pred_test)
        true_window_mse = mean_squared_error(y_test, y_pred_test)

        # 2. Intrinsic PCA Camera (Using full window features to capture terrain geometry)
        scaler_full = StandardScaler()
        X_full_scaled = scaler_full.fit_transform(X_w)

        pca_evolution = PCA()
        pca_evolution.fit(X_full_scaled)

        pc1_var = pca_evolution.explained_variance_ratio_[0]
        cumulative_top3_var = np.sum(pca_evolution.explained_variance_ratio_[:3])

        # Save the 384-dimensional eigenvector signature
        pc1_signatures.append(pca_evolution.components_[0])

        within_window_metrics.append({
            "window": current_window_name,
            "True_Holdout_R2": true_holdout_r2,
            "True_Window_MSE": true_window_mse,
            "PC1_Variance_Pct": pc1_var * 100,
            "Top3_Variance_Pct": cumulative_top3_var * 100
        })

    # Print Report 1 & 2
    df_within = pd.DataFrame(within_window_metrics)
    print(f"{'Time Window':<12} | {'True Holdout R2':<16} | {'True MSE':<12} | {'PC1 Var %':<10} | {'Top 3 Cumul %':<12}")
    print("-" * 95)
    for metric in within_window_metrics:
        print(f"{metric['window']:<12} | {metric['True_Holdout_R2']:>15.4f} | {metric['True_Window_MSE']:>12.4f} | {metric['PC1_Variance_Pct']:>9.2f}% | {metric['Top3_Variance_Pct']:>11.2f}%")
    print("-" * 95)

    # Calculate and print PCA Vector Alignment Mutation
    if len(pc1_signatures) >= 2:
        alignment = np.corrcoef(pc1_signatures[0], pc1_signatures[-1])[0, 1]
        print(f"Vector Correlation between first and last window's PC1: {alignment:.4f}")
        if alignment > 0.85:
            print("INSIGHT: Features (X) are stable over time. Performance drift is off-image.")
        else:
            print("INSIGHT: Features (X) are mutating! The visual landscape geometry has completely rotated.")
    print("\n")


    # =====================================================================
    # ANALYSIS 3: CUMULATIVE FORWARD-CHAINING BACKTEST (Connor's Request)
    # =====================================================================
    print("ANALYSIS 3: CUMULATIVE FORWARD-CHAINING TEMPORAL BACKTEST (T -> T+1)")
    print("-" * 105)

    forward_chaining_results = []

    for i in range(len(windows) - 1):
        # Train on cumulative history up to current point
        train_start = windows[0][0]
        train_end = windows[i][1]
        
        # Test strictly on the next chronological future block
        test_start = windows[i+1][0]
        test_end = windows[i+1][1]

        train_mask = (years >= train_start) & (years <= train_end)
        test_mask = (years >= test_start) & (years <= test_end)

        X_train, y_train = np.array(X[train_mask]), np.array(y[train_mask])
        X_test, y_test = np.array(X[test_mask]), np.array(y[test_mask])

        if len(y_train) < 10 or len(y_test) < 10:
            print(f"Skipping step: Insufficient data to train on {train_start}-{train_end} or test on {test_start}-{test_end}.")
            continue

        # Standardize strictly on training parameters to isolate the future test environment
        scaler_fc = StandardScaler()
        X_train_scaled = scaler_fc.fit_transform(X_train)
        X_test_scaled = scaler_fc.transform(X_test)

        model_fc = Ridge(alpha=alpha)
        model_fc.fit(X_train_scaled, y_train)

        y_pred_future = model_fc.predict(X_test_scaled)
        future_r2 = r2_score(y_test, y_pred_future)
        future_mse = mean_squared_error(y_test, y_pred_future)

        forward_chaining_results.append({
            "Train_Era": f"{train_start}-{train_end}",
            "Test_Future_Era": f"{test_start}-{test_end}",
            "Future_R2": future_r2,
            "Future_MSE": future_mse,
            "Train_N": len(y_train),
            "Test_N": len(y_test)
        })

    # Print Report 3
    df_forward = pd.DataFrame(forward_chaining_results)
    print(f"{'Historical Train Pool':<23} | {'Unseen Future Test':<20} | {'Future R2':<12} | {'Future MSE':<12} | {'Train N':<8} | {'Test N':<8}")
    print("-" * 105)
    for res in forward_chaining_results:
        print(f"{res['Train_Era']:<23} | {res['Test_Future_Era']:<20} | {res['Future_R2']:>12.4f} | {res['Future_MSE']:>12.4f} | {res['Train_N']:>8} | {res['Test_N']:>8}")
    print("-" * 105 + "\n")

    print("=" * 95)
    print("MASTER TEMPORAL PIPELINE COMPLETE")
    print("=" * 95)

    return df_within, df_forward


# =====================================================================
# DATA LOADING ZONE
# =====================================================================
if __name__ == "__main__":
    
    parquet_path = '/Users/shashankchathapuram/Desktop/Basic Prediction Model/master_merged_data.parquet'
    print(f"Reading master dataset from: {parquet_path}...")
    
    df = pd.read_parquet(parquet_path)
    print(f"Successfully loaded dataset. Shape: {df.shape}")

    # Robust index isolation: 
    # Your file has exactly 400 total columns. The final 384 columns are your features.
    # Therefore, the first 16 columns (indexes 0 to 15) contain the metadata.
    
    print("Extracting variables dynamically by position...")
    
    # 1. Timeline Years: The 10th column in your sequence (Index position 9)
    real_years = df.iloc[:, 9].values
    
    # 2. Wealth Index: The 11th column in your sequence (Index position 10)
    real_y = df.iloc[:, 10].values
    
    # 3. Satellite Features: The remaining 384 deep-learning visual feature columns
    real_X = df.iloc[:, 16:].values
    
    print(f"Extracted {real_X.shape[1]} visual satellite features across {real_X.shape[0]} observations.")
    print(f"Target Year range detected: Minimum Year = {real_years.min()}, Maximum Year = {real_years.max()}")

    # Set exact five-year timeline comparison windows
    target_windows = [(1995, 1999), (2000, 2004), (2005, 2009), (2010, 2015)]

    # Execute the master analysis functions
    df_within_window, df_forward_chain = run_comprehensive_temporal_analysis(
        X=real_X, 
        y=real_y, 
        years=real_years, 
        windows=target_windows, 
        alpha=1.0
    )
    
    # Write the calculated outputs to local files
    df_within_window.to_csv("results_pca_and_holdout.csv", index=False)
    df_forward_chain.to_csv("results_forward_chaining.csv", index=False)
    print("Spreadsheets generated: 'results_pca_and_holdout.csv' and 'results_forward_chaining.csv'")