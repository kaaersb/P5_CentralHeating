import os
import glob
import time
import requests
import pandas as pd
from tqdm import tqdm


def load_and_preprocess_zenodo_data(raw_data_dir: str) -> pd.DataFrame:
    """
    Loads all CSV files in raw_data_dir with progress tracking,
    maps Danish column names to standard feature names, and parses 
    timestamps using fast C-accelerated string parsing.
    """
    all_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
    
    # Exclude any weather cache files if they accidentally ended up here
    all_files = [f for f in all_files if "dmi_outdoor_temp_cache" not in f]
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory: {raw_data_dir}")

    df_list = []
    
    column_mapping = {
        'MeterID': 'MeterID',
        'Temperatur 1': 'T_supply',
        'Temperatur 2': 'T_return',
        'Flow 1': 'Flow',
        'RoundedReadTime': 'RoundedReadTime'
    }

    print(f"\n[Task 1/5] Found {len(all_files)} CSV files. Ingesting raw data...")
    
    for file_path in tqdm(all_files, desc="-> Ingesting CSVs", unit="file"):
        temp_df = pd.read_csv(file_path, sep=',', low_memory=False)
        temp_df.columns = temp_df.columns.str.strip()
        
        available_cols = [col for col in column_mapping.keys() if col in temp_df.columns]
        temp_df = temp_df[available_cols].rename(columns=column_mapping)
        
        df_list.append(temp_df)

    print("-> Concatenating dataframes...")
    df = pd.concat(df_list, ignore_index=True)

    print(f"\n[Task 2/5] Parsing dates for {len(df):,} rows using C-accelerated parsing...")
    start_time = time.time()

    df['RoundedReadTime'] = pd.to_datetime(
        df['RoundedReadTime'], 
        format='%d-%m-%Y %H:%M:%S',
        errors='coerce'
    ).dt.tz_localize(None)

    print(f"-> Date parsing completed in {time.time() - start_time:.2f} seconds!")
    return df


def fetch_dmi_outdoor_temperature(
    start_time, 
    end_time, 
    station_id='06060', 
    cache_path="Data/dmi_outdoor_temp_cache.csv"
) -> pd.DataFrame:
    """
    Fetches historical outdoor temperature from DMI's open MetObs API v2.
    Implements local caching outside Raw_Data to avoid API rate limits and file mixing.
    """
    print("\n[Task 3/5] Weather Data Integration...")
    
    if os.path.exists(cache_path):
        print(f"-> Loading cached DMI weather data from '{cache_path}'...")
        dmi_df = pd.read_csv(cache_path)
        dmi_df['RoundedReadTime'] = pd.to_datetime(
            dmi_df['RoundedReadTime'], 
            format='%Y-%m-%d %H:%M:%S'
        )
        return dmi_df

    print("-> No local cache found. Requesting weather data from DMI Open Data API...")
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    url = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"
    params = {
        'stationId': station_id,
        'parameterId': 'temp_dry',
        'datetime': f"{start_str}/{end_str}",
        'limit': 300000
    }

    max_retries = 3
    dmi_df = pd.DataFrame(columns=['RoundedReadTime', 'T_outdoor'])

    for attempt in range(max_retries):
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            records = []
            features = data.get('features', [])
            
            for item in tqdm(features, desc="-> Processing DMI Records", leave=False):
                obs_time = item['properties']['observed']
                temp = item['properties']['value']
                records.append({
                    'RoundedReadTime': pd.to_datetime(obs_time).round('h').tz_localize(None),
                    'T_outdoor': temp
                })

            dmi_df = pd.DataFrame(records)
            dmi_df = dmi_df.drop_duplicates(subset=['RoundedReadTime'])

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            dmi_df.to_csv(cache_path, index=False)
            print(f"-> Successfully downloaded and cached DMI data to '{cache_path}'.")
            return dmi_df

        elif response.status_code == 429:
            time.sleep(5)
            
        else:
            print(f"Warning: DMI API error ({response.status_code}): {response.text}")
            break

    return dmi_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific features for Isolation Forest:
    - Temperature Difference Delta T (T_supply - T_return)
    - 24-hour rolling averages per meter to smooth transient noise
    """
    print("\n[Task 4/5] Feature Engineering...")

    print("-> Computing Delta T (T_supply - T_return)...")
    df['Delta_T'] = df['T_supply'] - df['T_return']

    print("-> Sorting data by MeterID and RoundedReadTime...")
    df = df.sort_values(by=['MeterID', 'RoundedReadTime']).reset_index(drop=True)

    print("-> Computing 24-hour rolling averages per meter (this runs fully optimized)...")
    
    # Standard groupby rolling transform (fast and avoids tqdm/pandas internal compatibility issues)
    df['Delta_T_24h'] = df.groupby('MeterID')['Delta_T'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    df['Flow_24h'] = df.groupby('MeterID')['Flow'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())

    return df


def main():
    raw_data_dir = "Data/Raw_Data"
    output_file = "Data/Processed_Data/processed_heat_data.csv"
    dmi_cache_file = "Data/dmi_outdoor_temp_cache.csv"

    overall_start = time.time()

    # Step 1 & 2: Load data and parse dates
    df_raw = load_and_preprocess_zenodo_data(raw_data_dir)

    # Step 3: DMI Integration
    start_date = df_raw['RoundedReadTime'].min()
    end_date = df_raw['RoundedReadTime'].max()
    print(f"-> Timeframe: {start_date} to {end_date}")

    dmi_df = fetch_dmi_outdoor_temperature(start_date, end_date, cache_path=dmi_cache_file)

    if not dmi_df.empty:
        print("-> Merging DMI outdoor temperature with meter data...")
        df_merged = pd.merge(df_raw, dmi_df, on='RoundedReadTime', how='left')
    else:
        print("-> Warning: DMI data unavailable. Proceeding without outdoor temperature...")
        df_merged = df_raw
        df_merged['T_outdoor'] = float('nan')

    # Step 4: Feature Engineering
    df_engineered = engineer_features(df_merged)

    # Step 5: Clean and Save
    print("\n[Task 5/5] Finalizing Dataset...")
    print("-> Dropping NaN rows...")
    df_final = df_engineered.dropna().reset_index(drop=True)

    print(f"-> Exporting processed dataset to '{output_file}'...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    chunk_size = 500000
    num_chunks = (len(df_final) // chunk_size) + 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(num_chunks), desc="-> Writing Processed CSV", unit="chunk"):
            chunk = df_final.iloc[i * chunk_size : (i + 1) * chunk_size]
            chunk.to_csv(f, header=(i == 0), index=False)

    total_time = time.time() - overall_start
    print(f"\n==================================================")
    print(f"ALL TASKS COMPLETED IN {total_time / 60:.2f} MINUTES!")
    print(f"Output saved to: {output_file}")
    print(f"==================================================\n")
    print(df_final.head())


if __name__ == "__main__":
    main()