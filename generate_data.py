import numpy as np
import pandas as pd

def generate_diffusion_dialysis_data(n=100, seed=42):
    rng = np.random.default_rng(seed)

    # Time series: 1-minute interval
    time = pd.date_range("2026-01-01 08:00:00", periods=n, freq="min")

    # 7 sensor inputs
    s1_feed_flow = 2.00 + rng.normal(0, 0.02, n)                      # L/min
    s2_feed_cond = 115 - np.linspace(0, 3, n) + rng.normal(0, 0.4, n) # mS/cm
    s3_temp = 25 + np.linspace(0, 1.5, n) + rng.normal(0, 0.05, n)    # °C
    s4_inlet_press = 1.85 + np.linspace(0, 0.35, n) + rng.normal(0, 0.01, n)  # bar
    s7_water_flow = 2.50 + rng.normal(0, 0.02, n)                     # L/min
    s5_outlet_press = 1.62 - np.linspace(0, 0.10, n) + rng.normal(0, 0.01, n) # bar
    s6_outlet_cond = 72 - np.linspace(0, 10, n) + rng.normal(0, 0.5, n)       # mS/cm

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

    return df

if __name__ == "__main__":
    df = generate_diffusion_dialysis_data(n=100, seed=42)
    df.to_csv("diffusion_dialysis_input_data.csv", index=False)
    print("CSV file saved as diffusion_dialysis_input_data.csv")
    print(df.head())
