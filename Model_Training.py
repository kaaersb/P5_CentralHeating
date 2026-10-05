import os
import time
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from tqdm import tqdm


def main():
    input_file = "Data/Processed_Data/processed_heat_data.csv"
    model_output_file = "Models/isolation_forest_model.pkl"
    output_file = "Data/Processed_Data/processed_heat_data_with_anomalies.csv"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Processed data file not found: {input_file}. Run Data_Cleanup.py first.")

    print("\n[Task 1/4] Loading processed heat meter dataset...")
    chunk_size = 1_000_000
    total_rows = sum(1 for _ in open(input_file, encoding='utf-8')) - 1
    num_chunks = (total_rows // chunk_size) + 1

    chunks = []
    print(f"-> Ingesting {total_rows:,} rows in chunks...")
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunk_size), total=num_chunks, desc="-> Loading Chunks", unit="chunk"):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    feature_cols = ['Delta_T_24h', 'Flow_24h', 'T_outdoor']

    print("\n[Task 2/4] Training Isolation Forest on a robust statistical sample (2,000,000 rows)...")
    # Take a random sample of 2 million rows for training to prevent RAM overflow
    train_sample = df.sample(n=min(2_000, len(df)), random_state=42) if len(df) < 2_000_000 else df.sample(n=2_000_000, random_state=42)
    X_train = train_sample[feature_cols]

    model = IsolationForest(
        contamination=0.05, 
        random_state=42, 
        n_jobs=-1, 
        verbose=1
    )
    
    start_time = time.time()
    model.fit(X_train)
    print(f"-> Model training completed in {time.time() - start_time:.2f} seconds.")

    print("\n[Task 3/4] Predicting anomalies across the full dataset in memory-safe chunks...")
    # Predict in chunks to avoid ArrayMemoryError on the full 82M rows
    prediction_chunks = []
    pred_chunk_size = 2_000_000
    num_pred_chunks = (len(df) // pred_chunk_size) + 1

    for i in tqdm(range(num_pred_chunks), desc="-> Predicting Anomalies", unit="chunk"):
        chunk_data = df.iloc[i * pred_chunk_size : (i + 1) * pred_chunk_size][feature_cols]
        preds = model.predict(chunk_data)
        prediction_chunks.append(preds)

    import numpy as np
    df['Anomaly'] = np.concatenate(prediction_chunks)

    print("\n[Task 4/4] Saving trained model and exported dataset...")
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
    joblib.dump(model, model_output_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    export_chunks = (len(df) // chunk_size) + 1
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(export_chunks), desc="-> Writing Output CSV", unit="chunk"):
            c = df.iloc[i * chunk_size : (i + 1) * chunk_size]
            c.to_csv(f, header=(i == 0), index=False)

    anomaly_count = (df['Anomaly'] == -1).sum()
    normal_count = (df['Anomaly'] == 1).sum()
    print(f"\n==================================================")
    print(f"TRAINING & PREDICTION COMPLETE!")
    print(f"-> Normal points (1): {normal_count:,}")
    print(f"-> Anomalies (-1): {anomaly_count:,} ({anomaly_count / len(df) * 100:.2f}%)")
    print(f"==================================================\n")


if __name__ == "__main__":
    main()