import pandas as pd
import numpy as np
import os
import gc

def clean_and_engineer_data(input_path, output_path, chunk_size=50000):
    print("🚀 Starting Sprint 2: Memory-Safe Pandas Chunk-Processing Pipeline...")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find the master dataset at {input_path}.")
        
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Track execution states across chunks
    is_first_chunk = True
    cumulative_cycles = 0
    last_mode = None
    
    # 1. Read and process the file in bite-sized chunks
    print(f"📥 Streaming dataset in batches of {chunk_size} rows...")
    
    chunks = pd.read_csv(
        input_path, 
        chunksize=chunk_size, 
        low_memory=False
    )
    
    for i, chunk in enumerate(chunks):
        # 2. Fix Mixed Types & Downcast to float32
        numeric_features = ['voltage_charger', 'temperature_battery', 'voltage_load', 'current_load']
        for col in numeric_features:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').astype(np.float32)
        
        # 3. Handle Timestamps
        if 'start_time' in chunk.columns:
            chunk['start_time'] = pd.to_datetime(chunk['start_time'], format='mixed')
        else:
            raise KeyError("Could not find 'start_time' in the dataset.")
            
        # 4. Handle Missing Values via quick forward/backward fill inside the chunk
        chunk[numeric_features] = chunk[numeric_features].interpolate(method='linear').bfill().ffill()
        
        # 5. Feature Engineering: Degradation Proxy Metric
        chunk['voltage_variance'] = chunk['voltage_charger'] - chunk['voltage_load']
        
        # 6. Feature Engineering: Cycle Counting across chunk boundaries
        if 'mode' in chunk.columns:
            # Stitch the mode from the previous chunk to avoid broken cycles
            if last_mode is not None:
                modes = pd.concat([pd.Series([last_mode]), chunk['mode']]).reset_index(drop=True)
                mode_changes = modes.ne(modes.shift()).astype(np.int32).iloc[1:]
            else:
                mode_changes = chunk['mode'].ne(chunk['mode'].shift()).astype(np.int32)
                
            chunk['cycle_count'] = mode_changes.cumsum() + cumulative_cycles
            
            # Save states for the next chunk
            cumulative_cycles = chunk['cycle_count'].iloc[-1]
            last_mode = chunk['mode'].iloc[-1]
        else:
            chunk['cycle_count'] = np.int32(1)
            
        # 7. Feature Engineering: Rolling Averages
        chunk['rolling_temp_avg'] = chunk['temperature_battery'].rolling(window=10, min_periods=1).mean().astype(np.float32)
        chunk['rolling_voltage_load'] = chunk['voltage_load'].rolling(window=10, min_periods=1).mean().astype(np.float32)
        
        # 8. Save/Append Chunk to CSV
        if is_first_chunk:
            chunk.to_csv(output_path, mode='w', index=False)
            is_first_chunk = False
        else:
            chunk.to_csv(output_path, mode='a', header=False, index=False)
            
        print(f"🧩 Processed and saved batch {i + 1}...")
        
        # Explicitly clean up memory allocation
        del chunk
        gc.collect()

    print(f"✅ Sprint 2 complete! Your massive dataset was cleanly processed and saved to: {output_path}")

if __name__ == "__main__":
    clean_and_engineer_data(
        input_path="/opt/airflow/data/raw/unified_raw_battery_data.csv",
        output_path="/opt/airflow/data/processed/structured_battery_features.csv"
    )