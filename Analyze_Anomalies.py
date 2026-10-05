import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def main():
    input_file = "Data/Processed_Data/processed_heat_data_with_anomalies.csv"
    output_dir = "Outputs"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Anomaly dataset not found: {input_file}. Run train_isolation_forest.py first.")

    print("\n[Task 1/3] Loading dataset with anomaly predictions...")
    chunk_size = 1_000_000
    total_rows = sum(1 for _ in open(input_file, encoding='utf-8')) - 1
    num_chunks = (total_rows // chunk_size) + 1

    chunks = []
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size), total=num_chunks, desc="-> Loading Chunks", unit="chunk"):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    print("\n[Task 2/3] Aggregating anomalies by MeterID...")
    # Filter strictly for anomalies (-1)
    print("-> Filtering anomalies...")
    anomalies = df[df['Anomaly'] == -1]

    print("-> Calculating anomaly counts per meter...")
    meter_summary = anomalies.groupby('MeterID').size().reset_index(name='Anomaly_Count')
    
    print("-> Calculating total readings per meter...")
    total_readings = df.groupby('MeterID').size().reset_index(name='Total_Readings')
    
    meter_summary = pd.merge(meter_summary, total_readings, on='MeterID')
    meter_summary['Anomaly_Percentage'] = (meter_summary['Anomaly_Count'] / meter_summary['Total_Readings']) * 100

    # Sort to find worst installations
    meter_summary = meter_summary.sort_values(by='Anomaly_Count', ascending=False).reset_index(drop=True)

    print("\n--- TOP 10 WORST PERFORMING HEAT METERS ---")
    print(meter_summary.head(10).to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    summary_output_path = os.path.join(output_dir, "worst_meters_ranking.csv")
    meter_summary.to_csv(summary_output_path, index=False)
    print(f"\n-> Full meter ranking exported to '{summary_output_path}'")

    print("\n[Task 3/3] Generating diagnostic scatter plot...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Sample data for plotting to avoid overcrowding
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)

    sns.scatterplot(
        data=sample_df,
        x='T_outdoor',
        y='Delta_T_24h',
        hue='Anomaly',
        palette={1: 'blue', -1: 'red'},
        alpha=0.5,
        s=15
    )

    plt.title('Isolation Forest: Outdoor Temperature vs. 24h Rolling Delta T')
    plt.xlabel('Outdoor Temperature (°C)')
    plt.ylabel('24-Hour Rolling Delta T (°C)')
    plt.legend(title='Status', labels=['Normal (1)', 'Anomaly (-1)'])
    
    plot_output_path = os.path.join(output_dir, "anomaly_scatter_plot.png")
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"-> Scatter plot saved to '{plot_output_path}'")
    plt.close()

    print("\n==================================================")
    print("ANALYSIS COMPLETE SUCCESSFULLY!")
    print(f"==================================================\n")


if __name__ == "__main__":
    main()