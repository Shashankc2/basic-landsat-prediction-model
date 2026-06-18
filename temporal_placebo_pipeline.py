import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_placebo_temporal_analysis(X, y, years, windows, alpha=1.0):
    print("=" * 95 + "\n")

    print("ANALYSIS 1: PLACEBO WITHIN-WINDOW HOLDOUT")
    print("-" * 95)
    
    # Create a copy of y to manipulate safely
    shuffled_y = np.copy(y)
    
    # CRITICAL PLACEBO STEP: Shuffle target metrics within each isolated era
    # This preserves the data distribution but destroys the true economic relationship
    np.random.seed(42)  # For reproducibility
    for start, end in windows:
        mask = (years >= start) & (years <= end)
        era_indices = np.where(mask)[0]
        if len(era_indices) > 0:
            shuffled_values = shuffled_y[era_indices]
            np.random.shuffle(shuffled_values)
            shuffled_y[era_indices] = shuffled_values

    placebo_within_metrics = []

    for start, end in windows:
        mask = (years >= start) & (years <= end)
        X_w = np.array(X[mask])
        y_w = np.array(shuffled_y[mask])

        if len(y_w) < 15:
            continue

        current_window_name = f"{start}-{end}"
        X_train, X_test, y_train, y_test = train_test_split(
            X_w, y_w, test_size=0.20, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_placebo = Ridge(alpha=alpha)
        model_placebo.fit(X_train_scaled, y_train)

        y_pred_test = model_placebo.predict(X_test_scaled)
        placebo_r2 = r2_score(y_test, y_pred_test)
        placebo_mse = mean_squared_error(y_test, y_pred_test)

        placebo_within_metrics.append({
            "window": current_window_name,
            "Placebo_Holdout_R2": placebo_r2,
            "Placebo_Window_MSE": placebo_mse
        })

    df_placebo_within = pd.DataFrame(placebo_within_metrics)
    print(f"{'Time Window':<12} | {'Placebo Holdout R2':<18} | {'Placebo MSE':<12}")
    print("-" * 95)
    for metric in placebo_within_metrics:
        print(f"{metric['window']:<12} | {metric['Placebo_Holdout_R2']:>17.4f} | {metric['Placebo_Window_MSE']:>12.4f}")
    print("-" * 95 + "\n")


    print("ANALYSIS 3: PLACEBO CUMULATIVE FORWARD-CHAINING BACKTEST")
    print("-" * 105)

    placebo_forward_results = []

    for i in range(len(windows) - 1):
        train_start, train_end = windows[0][0], windows[i][1]
        test_start, test_end = windows[i+1][0], windows[i+1][1]

        train_mask = (years >= train_start) & (years <= train_end)
        test_mask = (years >= test_start) & (years <= test_end)

        X_train, y_train = np.array(X[train_mask]), np.array(shuffled_y[train_mask])
        X_test, y_test = np.array(X[test_mask]), np.array(shuffled_y[test_mask])

        if len(y_train) < 10 or len(y_test) < 10:
            continue

        scaler_fc = StandardScaler()
        X_train_scaled = scaler_fc.fit_transform(X_train)
        X_test_scaled = scaler_fc.transform(X_test)

        model_fc = Ridge(alpha=alpha)
        model_fc.fit(X_train_scaled, y_train)

        y_pred_future = model_fc.predict(X_test_scaled)
        future_r2 = r2_score(y_test, y_pred_future)
        future_mse = mean_squared_error(y_test, y_pred_future)

        placebo_forward_results.append({
            "Train_Era": f"{train_start}-{train_end}",
            "Test_Future_Era": f"{test_start}-{test_end}",
            "Placebo_Future_R2": future_r2,
            "Placebo_Future_MSE": future_mse
        })

    df_placebo_forward = pd.DataFrame(placebo_forward_results)
    print(f"{'Historical Train Pool':<23} | {'Unseen Future Test':<20} | {'Placebo Future R2':<17} | {'Placebo Future MSE':<18}")
    print("-" * 105)
    for res in placebo_forward_results:
        print(f"{res['Train_Era']:<23} | {res['Test_Future_Era']:<20} | {res['Placebo_Future_R2']:>17.4f} | {res['Placebo_Future_MSE']:>18.4f}")
    print("-" * 105 + "\n")

    return df_placebo_within, df_placebo_forward


if __name__ == "__main__":
    parquet_path = '/Users/shashankchathapuram/Desktop/Basic Prediction Model/master_merged_data.parquet'
    df = pd.read_parquet(parquet_path)
    
    real_years = df.iloc[:, 9].values
    real_y = df.iloc[:, 10].values
    real_X = df.iloc[:, 16:].values

    target_windows = [(1995, 1999), (2000, 2004), (2005, 2009), (2010, 2015)]

    df_p_within, df_p_forward = run_placebo_temporal_analysis(
        X=real_X, y=real_y, years=real_years, windows=target_windows, alpha=1.0
    )
    
    df_p_within.to_csv("results_placebo_within.csv", index=False)
    df_p_forward.to_csv("results_placebo_forward.csv", index=False)
    print("Placebo baseline files written successfully.")