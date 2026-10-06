import os
import pandas as pd
from tqdm import tqdm


def main():
    ranked_meters_file = "Outputs/worst_meters_ranked_by_percentage.csv"
    processed_data_file = "Data/Processed_Data/processed_heat_data.csv"

    if not os.path.exists(ranked_meters_file):
        raise FileNotFoundError(f"Fant ikke rangeringerne: '{ranked_meters_file}'. Kør 'analyze_anomalies.py' først.")

    if not os.path.exists(processed_data_file):
        raise FileNotFoundError(f"Fant ikke det behandlede datasæt: '{processed_data_file}'.")

    print("\n[1/3] Henter top-10 listen over de værste målere...")
    top10_info = pd.read_csv(ranked_meters_file).head(10)
    top10_ids = top10_info['MeterID'].tolist()
    
    print(f"Top 10 MeterIDs: {top10_ids}\n")

    print("[2/3] Filtrerer datasættet i hukommelsessikre chunks...")
    chunk_size = 1_000_000
    total_rows = sum(1 for _ in open(processed_data_file, encoding='utf-8')) - 1
    num_chunks = (total_rows // chunk_size) + 1

    top10_data = []

    for chunk in tqdm(pd.read_csv(processed_data_file, chunksize=chunk_size), total=num_chunks, desc="-> Søger i data", unit="chunk"):
        # Udtræk kun rækker der tilhører de 10 valgte MeterIDs
        filtered = chunk[chunk['MeterID'].isin(top10_ids)]
        if not filtered.empty:
            top10_data.append(filtered)

    df_top10 = pd.concat(top10_data, ignore_index=True)

    print("\n[3/3] Genererer 'describe()' for hver af de 10 værste målere:")
    print("=" * 80)

    columns_to_inspect = ['T_supply', 'T_return', 'Delta_T', 'Flow', 'T_outdoor']

    # Gå igennem målerne i den nøjagtige rækkefølge fra top 10 rangeringen
    for _, row in top10_info.iterrows():
        meter_id = int(row['MeterID'])
        anomaly_pct = row['Anomaly_Percentage']
        total_readings = int(row['Total_Readings'])

        meter_df = df_top10[df_top10['MeterID'] == meter_id]

        print(f"\n>>> MÅLER ID: {meter_id} (Fejlprocent: {anomaly_pct:.2f}% | Målinger: {total_readings:,}) <<<")
        
        if not meter_df.empty:
            stats = meter_df[columns_to_inspect].describe().T[['mean', 'std', 'min', '50%', 'max']]
            stats.columns = ['Gennemsnit', 'Std', 'Min (25%)', 'Median (50%)', 'Max']
            print(stats.to_string())
        else:
            print("Ingen måledata fundet for denne måler.")
            
        print("-" * 80)


if __name__ == "__main__":
    main()