import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Settings for 100 data points
n_points = 100
start_time = datetime(2026, 7, 17, 19, 30, 0)

# Generate synthetic sensor data
data = {
    'time': [start_time + timedelta(minutes=i) for i in range(n_points)],
    'S1_feed_flow_Lmin': np.random.normal(2.0, 0.05, n_points),
    'S2_feed_cond_mScm': np.random.normal(115.0, 0.5, n_points),
    'S3_temp_C': np.random.normal(25.0, 0.1, n_points),
    'S4_inlet_press_bar': np.random.normal(1.85, 0.02, n_points),
    'S7_water_flow_Lmin': np.random.normal(2.5, 0.03, n_points),
    'S5_outlet_press_bar': np.random.normal(1.60, 0.02, n_points),
    'S6_outlet_cond_mScm': np.random.normal(71.0, 0.8, n_points)
}

df = pd.DataFrame(data)
df.to_csv('sensors.csv', index=False)
print("Successfully generated sensors.csv with 100 data points.")
