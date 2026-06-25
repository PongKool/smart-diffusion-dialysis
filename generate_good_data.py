import numpy as np
import pandas as pd

def generate_good_membrane_data(n=500, seed=123):
    rng = np.random.default_rng(seed)

    # 30-minute intervals
    time = pd.date_range("2026-01-01 08:00:00", periods=n, freq="30min")

    # Healthy membrane: stable and favorable operating conditions
    s1_feed_flow = 2.00 + rng.normal(0, 0.01, n)          # L/min
    s7_water_flow = 2.50 + rng.normal(0, 0.01, n)         # L/min

    s2_feed_cond = 115 + rng.normal(0, 0.2, n)            # mS/cm
    s3_temp = 25.0 + rng.normal(0, 0.03, n)               # °C

    # Low pressure drop: inlet and outlet stay close
    s4_inlet_press = 1.85 + rng.normal(0, 0.005, n)       # bar
    s5_outlet_press = 1.65 + rng.normal(0, 0.005, n)      # bar

    # High outlet conductivity suggests good acid recovery
    s6_outlet_cond = 74 + rng.normal(0, 0.3, n)           # mS/cm

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
    df = generate_good_membrane_data(n=500, seed=123)
    df.to_csv("diffusion_dialysis_input_data_good_membrane.csv", index=False)
    print("CSV file saved as diffusion_dialysis_input_data_good_membrane.csv")
    print(df.head())
