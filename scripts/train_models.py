# =========================================================================
# 📦 STANDARD PIPELINE IMPORTS
# =========================================================================
import pandas as pd
import numpy as np
import pickle
import os
import gc  # Garbage Collector to free up memory instantly
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_ml_pipeline():
    print("🔮 Starting Sprint 3: Memory-Optimized Ultra-Streaming Training Phase...")
    
    # =========================================================================
    # 📁 DYNAMIC PATH INITIALIZATION & MOUNT VALIDATION
    # =========================================================================
    processed_data_path = "/opt/airflow/data/processed/structured_battery_features.csv"
    models_dir = "/opt/airflow/models"
    
    if os.path.exists("./data/processed/structured_battery_features.csv"):
        processed_data_path = "./data/processed/structured_battery_features.csv"
        models_dir = "./models"
        print("🏠 Running inside context with local repository directory layout bindings.")
    else:
        print("🐳 Running inside standard Airflow container paths.")

    if not os.path.exists(models_dir):
        print(f"📁 Target directory '{models_dir}' not found on the active layer. Constructing path...")
        os.makedirs(models_dir, exist_ok=True)
        
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(
            f"❌ Missing processed data matrix at: {processed_data_path}. Please execute feature engineering first!"
        )

    # =========================================================================
    # 📥 1. ULTRA-LIGHT LINE STREAMING INGESTION (RAM IMPOSSIBLE TO CRASH)
    # =========================================================================
    print("📥 Streaming large-scale battery features line-by-line to bypass RAM limits...")
    
    features = ['cycle_count', 'temperature_battery', 'rolling_temp_avg', 'rolling_voltage_load']
    target = 'voltage_variance'
    
    X_list = []
    y_list = []
    
    # Open the file as a raw text line stream (uses ~0 MB of RAM)
    with open(processed_data_path, 'r') as f:
        header = f.readline().strip().split(',')
        
        # Get the numeric index numbers of our target features
        idx_features = [header.index(col) for col in features]
        idx_target = header.index(target)
        
        # Read line by line, keeping only every 50th row to keep it incredibly fast and safe
        counter = 0
        for line in f:
            counter += 1
            if counter % 50 != 0:
                continue
                
            row = line.strip().split(',')
            try:
                # Extract columns and convert to floating points immediately
                feat_vals = [float(row[idx]) for idx in idx_features]
                targ_val = float(row[idx_target])
                
                X_list.append(feat_vals)
                y_list.append(targ_val)
            except (ValueError, IndexError):
                continue # Skip any incomplete rows smoothly
                
    # Convert streaming arrays directly into compact float32 NumPy matrices
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"📦 Successfully Streamed and Extracted {len(X)} Data Points safely into Matrix layout!")
    
    del X_list, y_list
    gc.collect()
    
    print("✂️ Splitting data into sequential Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print("⚖️ Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    scaler_out_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_out_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler parameters written safely to: {scaler_out_path}")
        
    # =========================================================================
    # 🌲 2. PRODUCTION MODEL: RANDOM FOREST (OPTIMIZED)
    # =========================================================================
    print("\n🌲 Training Memory-Constrained Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=20, 
        max_depth=8,
        max_samples=0.5,
        random_state=42, 
        n_jobs=2  
    )
    rf_model.fit(X_train_scaled, y_train)
    
    rf_preds = rf_model.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    
    print(f"📉 Random Forest Baseline Metrics -> RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f} | R² Score: {rf_r2:.4f}")
    
    rf_out_path = os.path.join(models_dir, 'random_forest_baseline.pkl')
    with open(rf_out_path, 'wb') as f:
        pickle.dump(rf_model, f)
    print(f"💾 Baseline weights written safely to: {rf_out_path}")
        
    del rf_model, rf_preds
    gc.collect()

    print("\n=========================================================================")
    print(f"✅ SUCCESS: Sprint 3 complete! Analytical deliverables exported to disk.")
    print(f"📂 Location: {os.path.abspath(models_dir)}")
    print("=========================================================================")

if __name__ == "__main__":
    train_ml_pipeline()