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

    print("\n[Task 2/3] Aggregating anomalies by MeterID (Ranked by Anomaly %)...")
    
    # Filter strictly for anomalies (-1)
    print("-> Filtering anomalies...")
    anomalies = df[df['Anomaly'] == -1]

    print("-> Calculating anomaly counts per meter...")
    meter_summary = anomalies.groupby('MeterID').size().reset_index(name='Anomaly_Count')
    
    print("-> Calculating total readings per meter...")
    total_readings = df.groupby('MeterID').size().reset_index(name='Total_Readings')
    
    # Merge summary metrics
    meter_summary = pd.merge(meter_summary, total_readings, on='MeterID', how='right').fillna(0)
    meter_summary['Anomaly_Percentage'] = (meter_summary['Anomaly_Count'] / meter_summary['Total_Readings']) * 100

    # Filter out meters with fewer than 1,000 readings to avoid small-sample bias
    min_readings_threshold = 1000
    valid_meters = meter_summary[meter_summary['Total_Readings'] >= min_readings_threshold].copy()

    # Sort strictly by Anomaly Percentage (descending)
    valid_meters = valid_meters.sort_values(by='Anomaly_Percentage', ascending=False).reset_index(drop=True)

    print(f"\n--- TOP 10 WORST PERFORMING HEAT METERS (RANKED BY ANOMALY %) ---")
    print(valid_meters.head(10).to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    summary_output_path = os.path.join(output_dir, "worst_meters_ranked_by_percentage.csv")
    valid_meters.to_csv(summary_output_path, index=False)
    print(f"\n-> Full percentage-ranked meter list exported to '{summary_output_path}'")

    print("\n[Task 3/3] Generating multi-axis diagnostic plots...")
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Take a representative sample for plotting performance
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    palette = {1: 'blue', -1: 'red'}

    # Panel 1: Flow vs. Delta T (Thermodynamic Efficiency)
    sns.scatterplot(
        data=sample_df, x='Flow_24h', y='Delta_T_24h', hue='Anomaly',
        palette=palette, alpha=0.4, s=15, ax=axes[0]
    )
    axes[0].set_title("1. Flow vs. Delta T (Cooling Efficiency)")
    axes[0].set_xlabel("24h Rolling Flow")
    axes[0].set_ylabel("24h Rolling Delta T (°C)")
    axes[0].legend(title='Status', labels=['Normal (1)', 'Anomaly (-1)'])

    # Panel 2: Supply vs. Return Temperature (Sensor & Hardware Faults)
    sns.scatterplot(
        data=sample_df, x='T_supply', y='T_return', hue='Anomaly',
        palette=palette, alpha=0.4, s=15, ax=axes[1]
    )
    # Add 1:1 parity line (where T_return = T_supply, indicating 0 °C Delta T)
    max_temp = max(sample_df['T_supply'].max(), sample_df['T_return'].max())
    axes[1].plot([0, max_temp], [0, max_temp], 'k--', alpha=0.7, label='1:1 Line (Zero Cooling)')
    axes[1].set_title("2. Supply vs. Return Temp (Hardware Faults)")
    axes[1].set_xlabel("Supply Temperature T_supply (°C)")
    axes[1].set_ylabel("Return Temperature T_return (°C)")
    axes[1].legend(title='Status')

    # Panel 3: Outdoor Temp vs. Delta T (Weather Correlation)
    sns.scatterplot(
        data=sample_df, x='T_outdoor', y='Delta_T_24h', hue='Anomaly',
        palette=palette, alpha=0.4, s=15, ax=axes[2]
    )
    axes[2].set_title("3. Outdoor Temp vs. Delta T (Weather Standard)")
    axes[2].set_xlabel("Outdoor Temperature (°C)")
    axes[2].set_ylabel("24h Rolling Delta T (°C)")
    axes[2].legend(title='Status')

    plt.tight_layout()
    plot_output_path = os.path.join(output_dir, "multi_axis_anomaly_comparison.png")
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"-> Multi-axis diagnostic figure saved to '{plot_output_path}'")
    plt.close()

    print("\n==================================================")
    print("ANALYSIS COMPLETE SUCCESSFULLY!")
    print(f"==================================================\n")


if __name__ == "__main__":
    main()