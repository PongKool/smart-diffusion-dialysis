import pandas as pd
import numpy as np
import os

def clean_sensor_data(df):
    """
    Cleans raw sensor data by removing missing values and invalid physical readings.
    """
    # Define required columns for the pipeline
    required_cols = ['S2_feed_cond_mScm', 'S3_temp_C', 'S4_inlet_press_bar', 
                     'S1_feed_flow_Lmin', 'S6_outlet_cond_mScm', 'S5_outlet_press_bar']
    
    # Remove rows where critical sensors are missing
    df = df.dropna(subset=required_cols)
    
    # Filter out physical impossibilities (negative flow or pressure)
    df = df[(df['S1_feed_flow_Lmin'] >= 0) & 
            (df['S4_inlet_press_bar'] >= 0) & 
            (df['S5_outlet_press_bar'] >= 0)]
    
    # Remove extreme conductivity outliers (3-sigma rule)
    for col in ['S2_feed_cond_mScm', 'S6_outlet_cond_mScm']:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] > (mean - 3 * std)) & (df[col] < (mean + 3 * std))]
        
    return df

def generate_training_data(input_file='sensors.csv', output_file='diffusion_dialysis_input_data2.csv'):
    """
    Processes sensors.csv, cleans it, calculates features, and generates training data.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    # 1. Load the sensor data
    df = pd.read_csv(input_file)
    
    # 2. Clean the data
    df = clean_sensor_data(df)
    
    # 3. Feature Engineering: Temperature Correction for Conductivity
    def temp_correct(cond, temp, target=25.0):
        # Assuming standard temperature coefficient of 2%/°C at 25°C
        return cond / (1 + 0.02 * (temp - target))

    df['feed_cond_25'] = temp_correct(df['S2_feed_cond_mScm'], df['S3_temp_C'])
    df['outlet_cond_25'] = temp_correct(df['S6_outlet_cond_mScm'], df['S3_temp_C'])
    
    # 4. Calculate target metrics
    df['acid_recovery_pct'] = (df['outlet_cond_25'] / df['feed_cond_25']) * 100
    df['deltaP_bar'] = df['S4_inlet_press_bar'] - df['S5_outlet_press_bar']
    
    # 5. Generate target labels (placeholder logic for demonstration)
    df['fouling_score'] = np.random.uniform(10, 25, len(df)) 
    df['days_to_cleaning'] = 4.0 - (df.index / len(df)) * 0.1 

    # 6. Select and order columns for train_model.py
    required_columns = [
        'time', 'S1_feed_flow_Lmin', 'S2_feed_cond_mScm', 'S3_temp_C', 
        'S4_inlet_press_bar', 'S7_water_flow_Lmin', 'S5_outlet_press_bar', 
        'S6_outlet_cond_mScm', 'feed_cond_25', 'outlet_cond_25', 
        'acid_recovery_pct', 'deltaP_bar', 'fouling_score', 'days_to_cleaning'
    ]
    
    # Ensure all required columns exist, fill missing if necessary (or drop)
    df = df[required_columns]
    
    # 7. Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {output_file} with {len(df)} records after cleaning.")

if __name__ == "__main__":
    generate_training_data()
