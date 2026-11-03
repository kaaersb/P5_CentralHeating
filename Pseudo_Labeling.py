"""
Pseudo-labelling of the Zenodo heat-meter data.

Goes through all meters and gives every meter-day a fault label, so that a
supervised classifier can be trained (Supervised_Training.py).

WHY THE LABELS AND THE FEATURES ARE KEPT APART
----------------------------------------------
If we labelled with rules on ΔT and then trained a model on ΔT, the model
would just re-learn our thresholds and score ~100 % without proving anything.
So:
  * LABELS are built from information the model never sees:
      - flow-weighted daily ΔT, from the meter's cumulative volume x temperature
        counters (Energi 8 / Energi 9), not from the hourly temperatures
      - daily energy and volume, compared to the meter's own energy signature
      - the meter's own error codes (InfoKoder)
  * FEATURES (Pseudo_Labeling/heat_features.py) use only hourly supply/return
    temperature and outdoor temperature, i.e. what the radiator sensor measures.
The model therefore has to learn how a fault *looks in the temperatures*,
which is what we need on the radiator data later.

Some overlap remains (flow-weighted ΔT and hourly ΔT are related), so the rule
baseline in Supervised_Training.py must be reported next to the model.

LABELS (one per meter-day, heating season only)
-----------------------------------------------
  normal            good cooling, consumption as expected, no error codes
  low_cooling       flow-weighted ΔT is low while there is a real heat load:
                    too much water for the heat delivered (wrong settings,
                    bypass, valve/thermostat faults)
  no_heat_draw      heat use collapses on a cold day (heating off, valve
                    closed or stuck, dwelling empty)
  high_consumption  energy far above the meter's own energy signature
                    (window open, setpoint too high, ...)
  meter_fault       the meter itself reports an error (air in the flow meter,
                    reverse flow)
  uncertain         matches no rule clearly -> not used for training
  off_season        outdoor temperature too high for space heating -> not used

Output:
  Data/Processed_Data/meter_days_labeled.pkl     one row per meter-day
  Outputs/pseudo_label_summary.csv               label counts
  Outputs/iforest_vs_pseudolabels.csv            which anomalies the Isolation
                                                 Forest actually flags

Run Pseudo_Labeling/reduce_raw.py first.
"""
import os
import sys
import glob
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(it, **_):
        return it

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pseudo_Labeling"))
from heat_features import installation_day_features, daily_outdoor, HDD_BASE  # noqa: E402

HOURLY_DIR = "Data/Processed_Data/hourly"
DMI_FILE = "Data/dmi_outdoor_temp_cache.csv"
IFOREST_FILE = "Models/isolation_forest_model.pkl"
OUT_FILE = "Data/Processed_Data/meter_days_labeled.pkl"
OUT_DIR = "Outputs"

# ---------------------------------------------------------------------------
# Labelling rules. All thresholds are here so they can be argued for in the
# report and changed in one place.
# ---------------------------------------------------------------------------
RULES = {
    "heating_max_tout": 12.0,     # daily mean outdoor temp (°C) for a heating day
    "min_hours": 20,              # hourly readings needed for a usable day
    "min_volume_m3": 0.05,        # below this, flow-weighted ΔT is unreliable

    # low_cooling: flow-weighted ΔT below this (°C).
    # Leiria et al. (2023) used ΔT < 10 °C to trigger technician visits; we use
    # a slightly wider band and require it on the flow-weighted value.
    "low_cooling_dT": 15.0,
    # ... and only when there is a real heat load (consumption at least this
    # share of expected). Low ΔT on days with almost no heat use is normal for
    # district heating (hot-water circulation dominates) and is not a fault.
    "low_cooling_min_ratio": 0.5,
    # normal requires at least this flow-weighted ΔT (°C); 15-25 is "uncertain"
    "normal_min_dT": 25.0,

    # energy signature: daily kWh ~ a + b * HDD, fitted robustly per meter
    "signature_min_days": 60,
    "signature_min_expected_kwh": 10.0,
    # no_heat_draw: cold day and consumption below this share of expected
    "no_draw_max_tout": 8.0,
    "no_draw_max_ratio": 0.15,
    # high_consumption: daily excess over the signature above this share of the
    # meter's typical consumption on a 0 °C day. Using a fixed reference
    # (instead of a ratio to the expected value) keeps the rule from firing
    # mostly on mild days, where the expected value is small.
    "high_excess_share": 0.5,
    # ... and sustained: the mean excess over day-1..day+1 must also be above
    # this share. Single-day spikes are usually hot water, not a heating fault.
    "high_excess_share_3d": 0.35,
    # normal: consumption within this band of expected
    "normal_ratio_band": (0.6, 1.4),

    # meter_fault: hours in the day with an error code
    "fault_min_hours_reverse": 1,
    "fault_min_hours_air": 6,
}

LABELS = ["normal", "low_cooling", "no_heat_draw", "high_consumption", "meter_fault"]


# ---------------------------------------------------------------------------
def label_inputs(h: pd.DataFrame) -> pd.DataFrame:
    """Daily values from the meter's counters (information the model never sees)."""
    h = h.assign(date=h["time"].dt.floor("D"))
    g = h.groupby("date")
    d = pd.DataFrame({
        "n_readings_h": g["time"].count(),
        "E_max": g["E_kWh"].max(),
        "V_max": g["V_m3"].max(),
        "E8_max": g["E8_m3C"].max(),
        "E9_max": g["E9_m3C"].max(),
        "hours_air": g["n_air"].apply(lambda s: int((s > 0).sum())),
        "hours_reverse": g["n_reverse"].apply(lambda s: int((s > 0).sum())),
    })
    consecutive = d.index.to_series().diff() == pd.Timedelta("1D")
    for src, dst in [("E_max", "dE_kWh"), ("V_max", "dV_m3"), ("E8_max", "dE8"), ("E9_max", "dE9")]:
        diff = d[src].diff().where(consecutive)
        d[dst] = diff.where(diff >= 0)            # counter resets / swaps -> NaN
    ok = d["dV_m3"] >= RULES["min_volume_m3"]
    d["dT_flow_weighted"] = ((d["dE8"] - d["dE9"]) / d["dV_m3"]).where(ok)
    d["Tr_flow_weighted"] = (d["dE9"] / d["dV_m3"]).where(ok)
    return d.drop(columns=["E_max", "V_max", "E8_max", "E9_max", "dE8", "dE9"])


def energy_signature(d: pd.DataFrame):
    """Robust fit of daily kWh ~ a + b*HDD on heating days.
    Returns (expected kWh per day, expected kWh on a 0 °C day)."""
    heat = d[(d["T_outdoor"] < RULES["heating_max_tout"]) & d["dE_kWh"].notna()]
    expected = pd.Series(np.nan, index=d.index)
    if len(heat) < RULES["signature_min_days"]:
        return expected, np.nan
    X = heat[["HDD"]].to_numpy()
    y = heat["dE_kWh"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg = HuberRegressor(max_iter=500).fit(X, y)
        except Exception:
            return expected, np.nan
    if reg.coef_[0] <= 0:      # no heating response: signature meaningless
        return expected, np.nan
    expected[:] = reg.predict(d[["HDD"]].fillna(0).to_numpy())
    return expected, float(reg.predict([[HDD_BASE]])[0])


def assign_labels(d: pd.DataFrame) -> pd.Series:
    R = RULES
    lab = pd.Series("uncertain", index=d.index, dtype=object)

    usable = d["n_readings_h"] >= R["min_hours"]
    heating = d["T_outdoor"] < R["heating_max_tout"]

    exp_ok = d["expected_kwh"] >= R["signature_min_expected_kwh"]
    ratio = d["consumption_ratio"]
    dT = d["dT_flow_weighted"]
    errors = (d["hours_air"] > 0) | (d["hours_reverse"] > 0)

    meter_fault = (d["hours_reverse"] >= R["fault_min_hours_reverse"]) | (d["hours_air"] >= R["fault_min_hours_air"])
    no_draw = exp_ok & (d["T_outdoor"] < R["no_draw_max_tout"]) & (d["dE_kWh"] <= R["no_draw_max_ratio"] * d["expected_kwh"])
    low_cool = (dT < R["low_cooling_dT"]) & exp_ok & (ratio >= R["low_cooling_min_ratio"])
    excess = (d["dE_kWh"] - d["expected_kwh"]) / d["ref_kwh_0C"]
    # mean excess over day-1 .. day+1 (window ending tomorrow, moved back one day)
    excess_3d = (excess.rolling("3D", min_periods=3).mean()
                 .shift(-1, freq="D").reindex(d.index))
    high = exp_ok & (excess > R["high_excess_share"]) & (excess_3d > R["high_excess_share_3d"])
    lo, hi = R["normal_ratio_band"]
    normal = (dT >= R["normal_min_dT"]) & exp_ok & ratio.between(lo, hi) & ~errors

    # Order = priority: a later assignment overwrites an earlier one.
    lab[normal] = "normal"
    lab[high] = "high_consumption"
    lab[low_cool] = "low_cooling"
    lab[no_draw] = "no_heat_draw"
    lab[meter_fault] = "meter_fault"

    # air in the flow meter makes all flow-based labels unreliable
    lab[(d["hours_air"] > 0) & (lab != "meter_fault")] = "uncertain"
    lab[~usable] = "uncertain"
    lab[~heating] = "off_season"
    return lab


# ---------------------------------------------------------------------------
def iforest_daily_share(h: pd.DataFrame, model, tout_hourly: pd.Series) -> pd.Series:
    """Share of hours per day that the existing Isolation Forest flags (-1)."""
    x = pd.DataFrame({
        "Delta_T_24h": (h["T_supply"] - h["T_return"]).rolling(24, min_periods=1).mean().values,
        "Flow_24h": h["Flow"].rolling(24, min_periods=1).mean().values,
        "T_outdoor": tout_hourly.reindex(h["time"]).values,
    })
    ok = x.notna().all(axis=1).values
    flag = np.full(len(h), np.nan)
    if ok.any():
        flag[ok] = (model.predict(x[ok]) == -1).astype(float)
    return pd.Series(flag, index=h["time"].dt.floor("D").values).groupby(level=0).mean()


def process_meter(h, tout_d, tout_h, model):
    h = h.sort_values("time").reset_index(drop=True)
    feats = installation_day_features(h[["time", "T_supply", "T_return"]], tout_d)
    if feats.empty:
        return None
    d = label_inputs(h)
    d = d.join(feats.set_index("date"), how="inner")
    d["expected_kwh"], d["ref_kwh_0C"] = energy_signature(d)
    d["consumption_ratio"] = d["dE_kWh"] / d["expected_kwh"].where(d["expected_kwh"] > 0)
    d["label"] = assign_labels(d)
    if model is not None:
        d["iforest_anomaly_share"] = iforest_daily_share(h, model, tout_h).reindex(d.index)
    d.index.name = "date"
    return d.reset_index()


def meter_file_index(files):
    """Which raw files each meter appears in (a few meters are split across files)."""
    where = {}
    for f in tqdm(files, desc="-> Indexing meters"):
        for m in pd.read_pickle(f)["MeterID"].unique():
            where.setdefault(int(m), []).append(f)
    return where


HOURLY_AGG = {"T_supply": "mean", "T_return": "mean", "Flow": "mean",
              "E_kWh": "max", "V_m3": "max", "E8_m3C": "max", "E9_m3C": "max",
              "n_air": "sum", "n_reverse": "sum"}


def _shrink(r: pd.DataFrame) -> pd.DataFrame:
    for c in r.select_dtypes("float64").columns:
        r[c] = r[c].astype(np.float32)
    return r


def main():
    use_iforest = "--no-iforest" not in sys.argv
    files = sorted(glob.glob(os.path.join(HOURLY_DIR, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No reduced files in {HOURLY_DIR}. Run Pseudo_Labeling/reduce_raw.py first.")

    dmi = pd.read_csv(DMI_FILE)
    dmi["RoundedReadTime"] = pd.to_datetime(dmi["RoundedReadTime"])
    tout_d = daily_outdoor(dmi)
    tout_h = dmi.drop_duplicates("RoundedReadTime").set_index("RoundedReadTime")["T_outdoor"].sort_index()

    model = None
    if use_iforest and os.path.exists(IFOREST_FILE):
        model = joblib.load(IFOREST_FILE)
        model.set_params(verbose=0, n_jobs=1)
        print(f"-> Using existing Isolation Forest from {IFOREST_FILE} for comparison")

    t0 = time.time()
    results = []
    where = meter_file_index(files)
    split_meters = {m for m, fs in where.items() if len(fs) > 1}
    held = []   # rows of meters that are split across files, processed at the end

    def run(h_all, name):
        for meter, h in tqdm(h_all.groupby("MeterID"), desc=f"-> Labelling {name}", leave=False):
            r = process_meter(h, tout_d, tout_h, model)
            if r is not None:
                r.insert(0, "MeterID", meter)
                results.append(_shrink(r))

    # One file at a time keeps memory low (~1-2 GB).
    for f in files:
        h_all = pd.read_pickle(f)
        is_split = h_all["MeterID"].isin(split_meters)
        if is_split.any():
            held.append(h_all[is_split])
        run(h_all[~is_split], os.path.basename(f))
        del h_all
        print(f"-> {os.path.basename(f)}: done ({time.time() - t0:.0f}s elapsed)")

    if held:
        h_all = pd.concat(held, ignore_index=True)
        h_all = h_all.groupby(["MeterID", "time"], sort=True).agg(HOURLY_AGG).reset_index()
        run(h_all, f"{len(split_meters)} meters split across files")
        print(f"-> split meters: done ({time.time() - t0:.0f}s elapsed)")

    df = pd.concat(results, ignore_index=True)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_pickle(OUT_FILE)

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = (df.groupby("label").agg(meter_days=("MeterID", "size"), meters=("MeterID", "nunique"))
                 .assign(share_of_heating_days=lambda s: s.meter_days / s.loc[s.index != "off_season", "meter_days"].sum()))
    summary.loc["off_season", "share_of_heating_days"] = np.nan
    summary.to_csv(os.path.join(OUT_DIR, "pseudo_label_summary.csv"))
    print("\nLabel counts:\n", summary.to_string())

    if "iforest_anomaly_share" in df:
        heat = df[df["label"] != "off_season"].copy()
        heat["iforest_flag_day"] = heat["iforest_anomaly_share"] >= 0.5
        comp = heat.groupby("label").agg(
            meter_days=("MeterID", "size"),
            iforest_mean_anomaly_share=("iforest_anomaly_share", "mean"),
            share_flagged_by_iforest=("iforest_flag_day", "mean"),
        )
        flagged = heat[heat["iforest_flag_day"]]
        comp["share_of_iforest_flags"] = flagged["label"].value_counts(normalize=True)
        comp.to_csv(os.path.join(OUT_DIR, "iforest_vs_pseudolabels.csv"))
        print("\nWhat the Isolation Forest flags (days with >= 50 % anomalous hours):\n", comp.to_string())

    print(f"\nSaved {len(df):,} meter-days for {df.MeterID.nunique()} meters to {OUT_FILE} "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
