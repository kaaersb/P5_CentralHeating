"""
Stage 1: reduce each raw Zenodo CSV to a compact hourly table.

For every (MeterID, hour) it keeps:
  - mean supply / return temperature and flow
  - the cumulative counters (energy, volume, volume-weighted temperatures),
    which are later differenced per day to get *flow-weighted* daily values
  - how many readings in the hour carried each InfoKoder (meter error code)

One raw file (~780 MB) becomes a ~60 MB pickle, so the whole dataset can be
labelled on a laptop. Run it once; Pseudo_Labeling.py picks up the result.

Usage:
    python Pseudo_Labeling/reduce_raw.py                 # all files in Data/Raw_Data
    python Pseudo_Labeling/reduce_raw.py Data/Raw_Data/7.csv
"""
import os
import sys
import glob
import time

import numpy as np
import pandas as pd

# Paths are relative to the repository root, so the script works from any folder.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "Data/Raw_Data")
OUT_DIR = os.path.join(REPO_ROOT, "Data/Processed_Data/hourly")

RAW_COLUMNS = {
    "MeterID": "MeterID",
    "Energi 1 Varmeenergi": "E_kWh",       # cumulative heat energy
    "Volumen 1": "V_m3",                   # cumulative volume
    "Energi 8": "E8_m3C",                  # cumulative volume x supply temp
    "Energi 9": "E9_m3C",                  # cumulative volume x return temp
    "Temperatur 1": "T_supply",
    "Temperatur 2": "T_return",
    "Flow 1": "Flow",
    "InfoKoder": "InfoKoder",
    "RoundedReadTime": "RoundedReadTime",
}

# InfoKoder texts in the data (Kamstrup MULTICAL 403). Matched on substrings so
# small spelling differences between files do not matter.
INFO_AIR = "luft i flowm"           # "Der er fundet luft i flowmåleren ..."
INFO_REVERSE = "forkert flowretning"  # "Flowmåler V1. forkert flowretning"


def reduce_file(path: str, out_dir: str = OUT_DIR) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, f"{name}.pkl")
    if os.path.exists(out_path):
        print(f"-> {name}: already reduced, skipping")
        return out_path

    t0 = time.time()
    df = pd.read_csv(path, usecols=list(RAW_COLUMNS), low_memory=False)
    df = df.rename(columns=RAW_COLUMNS)
    df["time"] = pd.to_datetime(df["RoundedReadTime"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["time"]).drop(columns="RoundedReadTime")

    info = df["InfoKoder"].fillna("").astype(str).str.lower()
    df["n_air"] = info.str.contains(INFO_AIR, regex=False).astype(np.int16)
    df["n_reverse"] = info.str.contains(INFO_REVERSE, regex=False).astype(np.int16)
    df = df.drop(columns="InfoKoder")

    g = df.groupby(["MeterID", "time"], sort=True)
    hourly = g.agg(
        T_supply=("T_supply", "mean"),
        T_return=("T_return", "mean"),
        Flow=("Flow", "mean"),
        E_kWh=("E_kWh", "max"),
        V_m3=("V_m3", "max"),
        E8_m3C=("E8_m3C", "max"),
        E9_m3C=("E9_m3C", "max"),
        n_air=("n_air", "sum"),
        n_reverse=("n_reverse", "sum"),
    ).reset_index()

    for c in ["T_supply", "T_return", "Flow"]:
        hourly[c] = hourly[c].astype(np.float32)
    hourly["MeterID"] = hourly["MeterID"].astype(np.int32)

    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    hourly.to_pickle(tmp)
    os.replace(tmp, out_path)
    print(f"-> {name}: {len(df):,} readings -> {len(hourly):,} meter-hours "
          f"({hourly.MeterID.nunique()} meters) in {time.time() - t0:.0f}s")
    return out_path


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    files = [f for f in files if "dmi_outdoor_temp_cache" not in f]
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}. "
                                "Put 1.csv ... 25.csv there (see README).")
    pointers = [f for f in files if os.path.getsize(f) < 1000]
    if pointers:
        raise RuntimeError(
            f"{len(pointers)} raw file(s) are Git LFS pointers, not data (e.g. {pointers[0]}).\n"
            "Download the real files with:  git lfs install && git lfs pull")
    for f in files:
        reduce_file(f)


if __name__ == "__main__":
    main()
