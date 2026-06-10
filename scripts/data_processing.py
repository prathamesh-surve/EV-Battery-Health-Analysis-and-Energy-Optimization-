import pandas as pd
import numpy as np
import os
import warnings

def clean_and_engineer_data(input_path, output_path):
    print("🚀 Starting Sprint 2: Data Processing & Feature Engineering Pipeline...")
    
    # Check if the unified raw file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find the master dataset at {input_path}. Ensure Sprint 1 completed successfully.")
        
    # 1. Load the merged dataset (Setting low_memory=False stops the mixed DtypeWarning)
    df = pd.read_csv(input_path, low_memory=False)
    
    # 2. Fix Mixed Types: Convert core sensory columns explicitly to numeric data types
    # Errors='coerce' will cleanly turn any accidental text anomalies into a NaN so we can fill it
    print("🧹 Converting raw sensory inputs into pure numerical values...")
    numeric_features = ['voltage_charger', 'temperature_battery', 'voltage_load', 'current_load']
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 3. Handle Timestamps with the correct underscore formatting
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], format='mixed')
        df = df.sort_values(by='start_time').reset_index(drop=True)
    else:
        raise KeyError("Could not find 'start_time' in the dataset. Please double-check column layout.")
    
    # 4. Handle Missing Values using Linear Interpolation (Best for time-series sensor data)
    print("🔄 Filling missing values using time-series linear interpolation...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    # Backward fill any remaining nulls that interpolation couldn't catch at the very top row
    df[numeric_cols] = df[numeric_cols].bfill()
    
    # 5. Eliminate duplicate entries
    print("🗑️ Removing duplicate entries...")
    df = df.drop_duplicates()
    
    # 6. Feature Engineering: Charge/Discharge Cycle Counting
    # Increment cycle count whenever the 'mode' column switches (e.g., charge -> discharge) [cite: 63, 66]
    print("🔋 Calculating operational battery cycle patterns...")
    if 'mode' in df.columns:
        df['mode_change'] = df['mode'].ne(df['mode'].shift()).astype(int)
        df['cycle_count'] = df['mode_change'].cumsum()
        df = df.drop(columns=['mode_change'])
    else:
        df['cycle_count'] = 1  # Fallback if mode column isn't populated
    
    # 7. Feature Engineering: Moving / Rolling Averages
    # Smooths high-frequency sensor noise to track underlying thermal and electrical stress trends [cite: 84]
    print("📈 Extracting rolling averages for temperature and voltage stress...")
    df['rolling_temp_avg'] = df['temperature_battery'].rolling(window=10, min_periods=1).mean()
    df['rolling_voltage_load'] = df['voltage_load'].rolling(window=10, min_periods=1).mean()
    
    # 8. Feature Engineering: Degradation Proxy Metric (Voltage Variance)
    # As an EV battery degrades, its internal resistance climbs. This causes a widening gap 
    # between Charger Supply Voltage and Internal Load Voltage[cite: 16].
    print("📉 Deriving primary degradation indicators...")
    df['voltage_variance'] = df['voltage_charger'] - df['voltage_load']
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the polished engineering features 
    df.to_csv(output_path, index=False)
    print(f"✅ Sprint 2 complete! Clean dataset with engineered features saved to: {output_path}")
    print(f"Processed Dataset Shape: {df.shape}")

if __name__ == "__main__":
    clean_and_engineer_data(
        input_path="data/raw/unified_raw_battery_data.csv",
        output_path="data/processed/structured_battery_features.csv"
    )