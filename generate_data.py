import numpy as np
import pandas as pd

def temp_correct_cond(cond, temp, alpha=0.02):
    return cond / (1 + alpha * (temp - 25.0))

def generate_diffusion_dialysis_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    # 30-minute interval so dataset spans many days
    time = pd.date_range("2026-01-01 08:00:00", periods=n, freq="30min")

    # Simulate repeated fouling-cleaning cycles
    cycle_len = 240  # 240 points x 30 min = 120 hours = 5 days
    cycle_pos = np.arange(n) % cycle_len
    cycle_frac = cycle_pos / cycle_len

    # 7 sensor inputs
    s1_feed_flow = 2.00 + rng.normal(0, 0.02, n)
    s7_water_flow = 2.50 + rng.normal(0, 0.02, n)

    s2_feed_cond = 115 - 1.5 * cycle_frac + rng.normal(0, 0.4, n)
    s3_temp = 25 + 1.2 * cycle_frac + rng.normal(0, 0.05, n)

    # Fouling increases during cycle, then resets after cleaning
    s4_inlet_press = 1.85 + 0.35 * cycle_frac + rng.normal(0, 0.01, n)
    s5_outlet_press = 1.62 - 0.10 * cycle_frac + rng.normal(0, 0.01, n)
    s6_outlet_cond = 72 - 10 * cycle_frac + rng.normal(0, 0.5, n)

    df = pd.DataFrame({
        "time": time,
        "S1_feed_flow_Lmin": np.round(s1_feed_flow, 3),
        "S2_feed_cond_mScm": np.round(s2_feed_cond, 3),
        "S3_temp_C": np.round(s3_temp, 3),
        "S4_inlet_press_bar": np.round(s4_inlet_press, 3),
        "S7_water_flow_Lmin": np.round(s7_water_flow, 3),
        "S5_outlet_press_bar": np.round(s5_outlet_press, 3),
        "S6_outlet_cond_mScm": np.round(s6_outlet_cond, 3),
    })

    # Derived targets
    df["feed_cond_25"] = temp_correct_cond(df["S2_feed_cond_mScm"], df["S3_temp_C"])
    df["outlet_cond_25"] = temp_correct_cond(df["S6_outlet_cond_mScm"], df["S3_temp_C"])

    df["acid_recovery_pct"] = (
        (df["S7_water_flow_Lmin"] * df["outlet_cond_25"]) /
        (df["S1_feed_flow_Lmin"] * df["feed_cond_25"])
    ) * 100

    df["deltaP_bar"] = df["S4_inlet_press_bar"] - df["S5_outlet_press_bar"]

    dp_clean = 0.20
    dp_max = 0.50
    recovery_clean = 82.0
    recovery_min = 60.0

    Pn = (df["deltaP_bar"] - dp_clean) / (dp_max - dp_clean)
    Rn = (recovery_clean - df["acid_recovery_pct"]) / (recovery_clean - recovery_min)

    Pn = Pn.clip(0, 1)
    Rn = Rn.clip(0, 1)

    FI = 0.5 * Pn + 0.5 * Rn
    df["fouling_score"] = (1 + 99 * FI).clip(1, 100)

    # Remaining days to cleaning threshold
    threshold = 80
    days_to_cleaning = []

    times = df["time"].values
    scores = df["fouling_score"].values

    for i in range(len(df)):
        future_idx = np.where(scores[i:] >= threshold)[0]
        if len(future_idx) == 0:
            days_to_cleaning.append(np.nan)
        else:
            j = i + future_idx[0]
            delta_days = (df.loc[j, "time"] - df.loc[i, "time"]).total_seconds() / 86400
            days_to_cleaning.append(delta_days)

    df["days_to_cleaning"] = days_to_cleaning

    # Remove rows that never reach threshold in future window
    df = df.dropna(subset=["days_to_cleaning"]).reset_index(drop=True)

    return df

if __name__ == "__main__":
    df = generate_diffusion_dialysis_data(n=5000, seed=42)
    df.to_csv("diffusion_dialysis_input_data.csv", index=False)
    print("CSV file saved as diffusion_dialysis_input_data.csv")
    print(df.head())
