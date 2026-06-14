import pandas as pd
import plotly.express as px
import os

def build_dashboard_visuals():
    print("📊 Generating memory-safe analytical dashboards...")
    
    # Check for internal container paths first, fallback to local path structure
    if os.path.exists("/opt/airflow/data/processed/"):
        processed_data_path = "/opt/airflow/data/processed/structured_battery_features.csv"
        output_dir = "/opt/airflow/data/processed"
    else:
        processed_data_path = "data/processed/structured_battery_features.csv"
        output_dir = "data/processed"

    print(f"🔍 Looking for dataset at: {processed_data_path}")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"❌ Processed data file missing at: {processed_data_path}")

    features = ["cycle_count", "voltage_variance", "temperature_battery"]
    
    # Stream and sample data in small chunks to prevent out-of-memory container crashes
    chunk_list = []
    print("📥 Streaming processed data chunks for visualization sampling...")
    for chunk in pd.read_csv(processed_data_path, usecols=features, chunksize=50000):
        sample_chunk = chunk.sample(frac=0.01, random_state=42)
        chunk_list.append(sample_chunk)
        
    df_sampled = pd.concat(chunk_list, ignore_index=True)
    print(f"📦 Sampled {len(df_sampled)} tracking rows safely for rendering.")

    # Render interactive data visualizations using Plotly
    print("🎨 Rendering interactive Plotly data visualizations...")
    fig1 = px.scatter(
        df_sampled,
        x="cycle_count", 
        y="voltage_variance", 
        color="temperature_battery",
        color_continuous_scale="thermal",  # Fixed case-sensitivity issue here
        title="EV Battery Degradation Path: Performance Variance vs. Lifecycle"
    )
    
    html_out = os.path.join(output_dir, "battery_degradation_dashboard.html")
    report_out = os.path.join(output_dir, "model_metrics_report.txt")
    
    print(f"✍️ Writing output assets to: {output_dir}")
    fig1.write_html(html_out)
    
    # Save the text-based metrics report summary
    with open(report_out, "w") as f:
        f.write("=== EV BATTERY INSIGHTS REPORT ===\n")
        f.write(f"Total Operational Cycles Evaluated: {df_sampled['cycle_count'].max()}\n")
        f.write(f"Peak Operational Thermal Threshold: {df_sampled['temperature_battery'].max():.2f} C\n")
        f.write("Status: End-to-End Orchestrated Pipeline execution successful.\n")

    print(f"✅ Dashboard engine successfully completed. Visuals written to: {html_out}")

if __name__ == "__main__":
    build_dashboard_visuals()