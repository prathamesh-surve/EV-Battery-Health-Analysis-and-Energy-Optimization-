import pandas as pd
import numpy as np
import pickle
import os
import gc  # Garbage Collector to free up memory instantly
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_ml_pipeline():
    print("🔮 Starting Sprint 3: Memory-Optimized Model Training Phase...")
    
    processed_data_path = "data/processed/structured_battery_features.csv"
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Missing processed data at {processed_data_path}. Please run Sprint 2 first!")

    # 1. Load your dataset efficiently
    print("📥 Loading large-scale battery features dataset...")
    df = pd.read_csv(processed_data_path)
    print(f"Original Row Count: {len(df)}")
    
    # 2. Downsample for local system memory constraint (E.g., take every 10th row)
    # This maintains chronological time-series variance perfectly while reducing computational overhead
    if len(df) > 500000:
        print("⚡ Downsampling data frequency to optimize local system memory...")
        df = df.iloc[::10].reset_index(drop=True)
        print(f"Optimized Row Count: {len(df)}")
    
    # 3. Cast to float32 (Reduces memory consumption by 50% vs float64)
    features = ['cycle_count', 'temperature_battery', 'rolling_temp_avg', 'rolling_voltage_load']
    target = 'voltage_variance'
    
    for col in features + [target]:
        df[col] = df[col].astype(np.float32)

    X = df[features].values
    y = df[target].values
    
    # Clear memory space immediately
    del df
    gc.collect()
    
    # Sequential Time-series split
    print("✂️ Splitting data into sequential Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale features
    print("⚖️ Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    os.makedirs("models", exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    # ------------------ 1. BASELINE MODEL: RANDOM FOREST ------------------
    print("\n🌲 Training Memory-Constrained Random Forest Regressor...")
    # Reduced n_estimators and limited max_depth/max_samples keeps memory usage low
    rf_model = RandomForestRegressor(
        n_estimators=30, 
        max_depth=12,
        max_samples=0.5,
        random_state=42, 
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate Random Forest
    rf_preds = rf_model.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    
    print(f"📉 Random Forest Metrics -> RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f} | R² Score: {rf_r2:.4f}")
    
    with open('models/random_forest_baseline.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
        
    # Clear memory space immediately
    del rf_model, rf_preds
    gc.collect()

    # ------------------ 2. TIME-SERIES MODEL: LSTM ------------------
    print("\n🧠 Reshaping sequential datasets for LSTM architecture...")
    
    # We will build sequences using a lookback window of 5 consecutive rows
    def create_sequences(X_data, y_data, time_steps=5):
        Xs, ys = [], []
        for i in range(len(X_data) - time_steps):
            Xs.append(X_data[i:(i + time_steps)])
            ys.append(y_data[i + time_steps])
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
        
    X_train_lstm, y_train_lstm = create_sequences(X_train_scaled, y_train, 5)
    X_test_lstm, y_test_lstm = create_sequences(X_test_scaled, y_test, 5)
    
    print(f"LSTM Train Shape: {X_train_lstm.shape} | Test Shape: {X_test_lstm.shape}")
    
    print("🏗️ Constructing LSTM Deep Learning Model...")
    lstm_model = Sequential([
        LSTM(32, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("🔥 Training LSTM network...")
    # Larger batch size speeds up computation and lowers memory spikes
    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=2, batch_size=256, validation_split=0.1, verbose=1)
    
    # Evaluate LSTM
    print("Evaluating LSTM performance...")
    lstm_preds = lstm_model.predict(X_test_lstm).flatten()
    lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_preds))
    lstm_mae = mean_absolute_error(y_test_lstm, lstm_preds)
    lstm_r2 = r2_score(y_test_lstm, lstm_preds)
    
    print(f"📉 LSTM Deep Learning Metrics -> RMSE: {lstm_rmse:.4f} | MAE: {lstm_mae:.4f} | R² Score: {lstm_r2:.4f}")
    
    lstm_model.save('models/lstm_degradation_model.keras')
    print("✅ Sprint 3 complete! Scaler and models exported cleanly to the /models folder.")

if __name__ == "__main__":
    train_ml_pipeline()