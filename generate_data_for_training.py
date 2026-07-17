import pandas as pd
import numpy as np

def generate_training_data(input_file='sensors.csv', output_file='diffusion_dialysis_input_data2.csv'):
    # 1. Load the sensor data
    df = pd.read_csv(input_file)
    
    # 2. Feature Engineering: Temperature Correction for Conductivity
    # Assuming standard temperature coefficient of 2%/°C at 25°C
    def temp_correct(cond, temp, target=25.0):
        return cond / (1 + 0.02 * (temp - target))

    df['feed_cond_25'] = temp_correct(df['S2_feed_cond_mScm'], df['S3_temp_C'])
    df['outlet_cond_25'] = temp_correct(df['S6_outlet_cond_mScm'], df['S3_temp_C'])
    
    # 3. Calculate target metrics (reconstructing logic used in your pipeline)
    # Example: Calculate acid recovery percentage and deltaP
    # These formulas should match the logic expected by your models
    df['acid_recovery_pct'] = (df['outlet_cond_25'] / df['feed_cond_25']) * 100
    df['deltaP_bar'] = df['S4_inlet_press_bar'] - df['S5_outlet_press_bar']
    
    # 4. Generate fouling_score and days_to_cleaning 
    # (Replace these with your actual model-specific logic or business rules)
    df['fouling_score'] = np.random.uniform(10, 25, len(df)) 
    df['days_to_cleaning'] = 4.0 - (df.index / len(df)) * 0.1 

    # 5. Ensure columns match the expected order for train_model.py
    required_columns = [
        'time', 'S1_feed_flow_Lmin', 'S2_feed_cond_mScm', 'S3_temp_C', 
        'S4_inlet_press_bar', 'S7_water_flow_Lmin', 'S5_outlet_press_bar', 
        'S6_outlet_cond_mScm', 'feed_cond_25', 'outlet_cond_25', 
        'acid_recovery_pct', 'deltaP_bar', 'fouling_score', 'days_to_cleaning'
    ]
    
    df = df[required_columns]
    
    # 6. Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {output_file} with {len(df)} records.")

if __name__ == "__main__":
    generate_training_data()