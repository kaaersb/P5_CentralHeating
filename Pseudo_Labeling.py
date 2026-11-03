"""
Pseudo-labelling of the Zenodo heat-meter data, following the group's rules.

Goes through all meters and gives every meter-day a fault label, so that a
supervised classifier can be trained (Supervised_Training.py).

BASIC IDEA
----------
Every meter gets its own baseline for energy (E), flow (Q), ΔT (supply minus
return) and return temperature, each as a function of outdoor temperature
(T_out). "Clearly higher/lower than normal" always means compared with the
meter's own baseline at that day's outdoor temperature.

RULES (checked in this order, first match wins)
-----------------------------------------------
1) fejlinstallation   (any outdoor temperature)
     a) ΔT < 0 while there is flow, in >= n hours of the day
     b) Q > 0 but E ≈ 0 over the day
     c) E > 0 but Q = 0 over the day
     d) |E - c·Q·ΔT| / E > 0.3 over the day (energy balance does not add up)
     e) T_out < 12 and median hourly ΔT over the day below a threshold
2) radiator_slukket   T_out < 5, E and Q clearly lower than normal
3) blokeret_radiator  T_out < 12, Q clearly lower than normal
4) radiator_helt_aaben
     A) 8 < T_out < 12, E and Q clearly higher, ΔT lower and return
        temperature higher than normal, OR
     B) T_out < 12, E clearly higher, AND E no longer follows the weather
        (|corr(E, T_out)| < 0.3 over the last n days) AND night use is
        >= 90 % of day use
5) aabent_vindue      T_out < 5, daytime, sudden rise in hourly E compared
                      with the same hour on the previous n days, lasting 1-6
                      hours, after which E returns to normal
6) normal             everything else

Days with too little data to apply the rules are labelled `ingen_data` and
are not used for training.

Every "n" and every "clearly" is a number in RULES below, so it can be argued
for in the report and changed in one place.

WHY THE MODEL CANNOT JUST RE-LEARN THE RULES
--------------------------------------------
The rules use E and Q (energy and flow), plus flow-weighted values from the
meter's counters. The model (Pseudo_Labeling/heat_features.py) only sees
hourly supply, return and outdoor temperature, which is what the radiator
sensor measures. Some rules also use temperatures (1a, 1e, 4A), so the overlap
is not zero; the training script reports a plain ΔT rule next to the model.

Output:
  Data/Processed_Data/meter_days_labeled.pkl     one row per meter-day
  Outputs/pseudo_label_summary.csv               label counts
  Outputs/pseudo_label_rule_hits.csv             how often each sub-rule fires
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
from heat_features import installation_day_features, daily_outdoor  # noqa: E402

HOURLY_DIR = "Data/Processed_Data/hourly"
DMI_FILE = "Data/dmi_outdoor_temp_cache.csv"
IFOREST_FILE = "Models/isolation_forest_model.pkl"
OUT_FILE = "Data/Processed_Data/meter_days_labeled.pkl"
OUT_DIR = "Outputs"

C_WATER = 1.16   # kWh per m³ per K (heat capacity of water at 40-70 °C)

# ---------------------------------------------------------------------------
# All numbers used by the rules. "n" values are counted in hourly readings
# unless the name says days.
# ---------------------------------------------------------------------------
RULES = {
    # data needed before any rule is applied
    "min_hours": 20,                  # hourly readings in the day

    # baselines (per meter, per variable, as a function of T_out)
    "baseline_max_tout": 12.0,        # baselines are fitted on heating days
    "baseline_min_days": 60,
    "baseline_min_expected_kwh": 5.0, # below this the E-baseline is too small to compare

    # 1) fejlinstallation
    "r1_neg_dT_hours": 3,             # a) n hours with ΔT < 0 while flow > 0
    "r1_flow_no_energy_m3": 0.2,      # b) at least this much water ...
    "r1_flow_no_energy_kwh": 0.0,     #    ... but energy at most this
    "r1_energy_no_flow_kwh": 2.0,     # c) at least this much energy with no water
    "r1_balance_max_error": 0.3,      # d) |E - c·Q·ΔT| / E above this
    "r1_balance_min_kwh": 5.0,        #    only on days with at least this energy
    "r1_low_median_dT": 10.0,         # e) median hourly ΔT below this (°C) ...
    "r1_low_median_min_flow_m3": 0.2, #    ... on days with real flow
    "r1_max_tout": 12.0,

    # 2) radiator_slukket
    "r2_max_tout": 5.0,
    "r2_max_ratio": 0.3,              # E and Q below 30 % of baseline

    # 3) blokeret_radiator
    "r3_max_tout": 12.0,
    "r3_max_q_ratio": 0.5,            # Q below 50 % of baseline

    # 4) radiator_helt_aaben
    "r4a_tout": (8.0, 12.0),
    "r4a_min_ratio": 1.3,             # E and Q above 130 % of baseline
    "r4a_max_dT_ratio": 0.85,         # ΔT below 85 % of baseline
    "r4a_min_tr_excess": 2.0,         # return temp at least 2 °C above baseline
    "r4b_max_tout": 12.0,
    "r4b_min_ratio": 1.5,             # E above 150 % of baseline
    "r4b_corr_days": 14,              # n days for the E/T_out correlation
    "r4b_corr_min_days": 10,
    "r4b_corr_min_tout_std": 2.0,     # need some weather variation to judge it
    "r4b_max_abs_corr": 0.3,
    "r4b_night_day_ratio": 0.9,       # night (00-06) use >= 90 % of day (10-16) use
    "r4b_require_all": True,          # True: E high AND weak corr AND flat night.
                                      # False: E high AND (weak corr OR flat night)

    # 5) aabent_vindue
    "r5_max_tout": 5.0,
    "r5_day_hours": (7, 22),          # daytime: 07:00 - 21:59
    "r5_history_days": 7,             # compare with the same hour on n previous days
    "r5_min_history": 4,
    "r5_rise_factor": 2.5,            # hourly E above 250 % of that median ...
    "r5_rise_min_kwh": 1.0,           # ... and at least this much above it
    # The group's rule says 1-6 hours. A 1-hour rise is almost always a shower or
    # bath (hot water goes through the same meter), so we require 2 hours.
    "r5_min_duration_h": 2,           # episode lasts min..max hours, then back to normal
    "r5_max_duration_h": 6,
}

LABELS = ["normal", "fejlinstallation", "radiator_slukket", "blokeret_radiator",
          "radiator_helt_aaben", "aabent_vindue"]


# ---------------------------------------------------------------------------
# Hourly and daily quantities from the meter (the model never sees these)
# ---------------------------------------------------------------------------
def hourly_quantities(h: pd.DataFrame) -> pd.DataFrame:
    """Per-hour energy and volume from the cumulative counters."""
    h = h.copy()
    step_ok = h["time"].diff() == pd.Timedelta("1h")
    for src, dst in [("E_kWh", "E_h"), ("V_m3", "Q_h")]:
        diff = h[src].diff().where(step_ok)
        h[dst] = diff.where(diff >= 0)
    h["dT_h"] = h["T_supply"] - h["T_return"]
    # hourly heat from flow and ΔT: much finer resolution than the whole-kWh counter
    h["EQ_h"] = C_WATER * h["Q_h"] * h["dT_h"]
    h["date"] = h["time"].dt.floor("D")
    h["hour"] = h["time"].dt.hour
    return h


def daily_quantities(h: pd.DataFrame) -> pd.DataFrame:
    R = RULES
    g = h.groupby("date")
    d = pd.DataFrame({
        "n_readings_h": g["time"].count(),
        "E_day": g["E_h"].sum(min_count=R["min_hours"]),
        "Q_day": g["Q_h"].sum(min_count=R["min_hours"]),
        "EQ_day": g["EQ_h"].sum(min_count=R["min_hours"]),
        "median_dT_h": g["dT_h"].median(),
        "neg_dT_hours": (h["dT_h"] < 0).where(h["Q_h"] > 0, False).groupby(h["date"]).sum(),
        "E_night": h[h["hour"] < 6].groupby("date")["E_h"].sum(min_count=5),
        "E_daytime": h[h["hour"].between(10, 15)].groupby("date")["E_h"].sum(min_count=5),
    })
    # flow-weighted daily ΔT and return temperature from the Energi 8/9 counters
    e8 = g["E8_m3C"].max().diff()
    e9 = g["E9_m3C"].max().diff()
    consecutive = d.index.to_series().diff() == pd.Timedelta("1D")
    q_ok = (d["Q_day"] >= 0.05) & consecutive
    d["dT_fw"] = ((e8 - e9) / d["Q_day"]).where(q_ok)
    d["Tr_fw"] = (e9 / d["Q_day"]).where(q_ok)
    d["hours_air"] = g["n_air"].apply(lambda s: int((s > 0).sum()))
    d["hours_reverse"] = g["n_reverse"].apply(lambda s: int((s > 0).sum()))
    return d


def fit_baseline(d: pd.DataFrame, col: str) -> pd.Series:
    """Robust per-meter baseline: col ~ a + b·T_out, fitted on heating days."""
    R = RULES
    fit = d[(d["T_outdoor"] < R["baseline_max_tout"]) & d[col].notna() & d["T_outdoor"].notna()]
    out = pd.Series(np.nan, index=d.index)
    if len(fit) < R["baseline_min_days"]:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reg = HuberRegressor(max_iter=500).fit(fit[["T_outdoor"]].to_numpy(), fit[col].to_numpy())
        except Exception:
            return out
    has_t = d["T_outdoor"].notna()
    out[has_t] = reg.predict(d.loc[has_t, ["T_outdoor"]].to_numpy())
    return out


def open_window_episodes(h: pd.DataFrame, tout_daily: pd.Series) -> pd.Series:
    """Rule 5: number of short daytime E spikes per day."""
    R = RULES
    x = h.set_index("time")["EQ_h"]
    x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="1h"))
    hours = x.index.hour
    # median of the same clock hour on the previous n days
    same_hour = pd.concat([x.shift(24 * k) for k in range(1, R["r5_history_days"] + 1)], axis=1)
    ref = same_hour.median(axis=1).where(same_hour.notna().sum(axis=1) >= R["r5_min_history"])
    spike = (x > R["r5_rise_factor"] * ref) & ((x - ref) > R["r5_rise_min_kwh"])
    lo, hi = R["r5_day_hours"]
    daytime = (hours >= lo) & (hours < hi)
    cold = pd.Series(tout_daily.reindex(x.index.floor("D")).to_numpy() < R["r5_max_tout"], index=x.index)
    spike = spike & daytime & cold

    # runs of consecutive spike hours; keep runs of 1..max hours that are
    # followed by a normal (not spiking, measured) hour
    run_id = (spike != spike.shift()).cumsum()
    episodes = pd.Series(0, index=x.index.floor("D").unique())
    for _, run in spike[spike].groupby(run_id[spike]):
        after = run.index[-1] + pd.Timedelta("1h")
        back_to_normal = after in x.index and pd.notna(x.get(after)) and not spike.get(after, False)
        if R["r5_min_duration_h"] <= len(run) <= R["r5_max_duration_h"] and back_to_normal:
            episodes[run.index[0].floor("D")] += 1
    return episodes


def weather_correlation(d: pd.DataFrame) -> pd.Series:
    """Rule 4B: |corr(E_day, T_out)| over the last n days."""
    R = RULES
    win = f'{R["r4b_corr_days"]}D'
    corr = d["E_day"].rolling(win, min_periods=R["r4b_corr_min_days"]).corr(d["T_outdoor"])
    tstd = d["T_outdoor"].rolling(win, min_periods=R["r4b_corr_min_days"]).std()
    return corr.abs().where(tstd >= R["r4b_corr_min_tout_std"])


# ---------------------------------------------------------------------------
def assign_labels(d: pd.DataFrame):
    R = RULES
    t = d["T_outdoor"]
    E, Q = d["E_day"], d["Q_day"]
    e_ok = d["E_base"] >= R["baseline_min_expected_kwh"]
    rE = (E / d["E_base"]).where(e_ok)
    rQ = (Q / d["Q_base"]).where(d["Q_base"] > 0)
    rdT = (d["dT_fw"] / d["dT_base"]).where(d["dT_base"] > 1)
    tr_excess = d["Tr_fw"] - d["Tr_base"]

    hits = pd.DataFrame(index=d.index)
    # 1) fejlinstallation
    hits["1a_neg_dT"] = d["neg_dT_hours"] >= R["r1_neg_dT_hours"]
    hits["1b_flow_no_energy"] = (Q >= R["r1_flow_no_energy_m3"]) & (E <= R["r1_flow_no_energy_kwh"])
    hits["1c_energy_no_flow"] = (E >= R["r1_energy_no_flow_kwh"]) & (Q == 0)
    bal_err = ((E - d["EQ_day"]).abs() / E).where(E >= R["r1_balance_min_kwh"])
    d["energy_balance_error"] = bal_err
    hits["1d_energy_balance"] = bal_err > R["r1_balance_max_error"]
    hits["1e_low_median_dT"] = ((t < R["r1_max_tout"]) & (Q >= R["r1_low_median_min_flow_m3"])
                                & (d["median_dT_h"] < R["r1_low_median_dT"]))
    # 2) radiator_slukket
    hits["2_radiator_off"] = (t < R["r2_max_tout"]) & (rE < R["r2_max_ratio"]) & (rQ < R["r2_max_ratio"])
    # 3) blokeret_radiator
    hits["3_blocked"] = (t < R["r3_max_tout"]) & (rQ < R["r3_max_q_ratio"])
    # 4) radiator_helt_aaben
    lo, hi = R["r4a_tout"]
    hits["4a_open_mild"] = ((t > lo) & (t < hi) & (rE > R["r4a_min_ratio"]) & (rQ > R["r4a_min_ratio"])
                            & (rdT < R["r4a_max_dT_ratio"]) & (tr_excess > R["r4a_min_tr_excess"]))
    weak_corr = d["abs_corr_E_Tout"] < R["r4b_max_abs_corr"]
    flat_night = d["night_day_ratio"] >= R["r4b_night_day_ratio"]
    pattern = (weak_corr & flat_night) if R["r4b_require_all"] else (weak_corr | flat_night)
    hits["4b_open_constant"] = (t < R["r4b_max_tout"]) & (rE > R["r4b_min_ratio"]) & pattern
    # 5) aabent_vindue
    hits["5_open_window"] = d["window_episodes"] > 0
    hits = hits.fillna(False).astype(bool)

    lab = pd.Series("normal", index=d.index, dtype=object)
    order = [("aabent_vindue", ["5_open_window"]),
             ("radiator_helt_aaben", ["4a_open_mild", "4b_open_constant"]),
             ("blokeret_radiator", ["3_blocked"]),
             ("radiator_slukket", ["2_radiator_off"]),
             ("fejlinstallation", ["1a_neg_dT", "1b_flow_no_energy", "1c_energy_no_flow",
                                   "1d_energy_balance", "1e_low_median_dT"])]
    for name, cols in order:          # reversed priority: rule 1 is written last and wins
        lab[hits[cols].any(axis=1)] = name

    # The rules cannot be applied without a full day of readings and the counters
    enough = (d["n_readings_h"] >= R["min_hours"]) & E.notna() & Q.notna() & t.notna()
    lab[~enough] = "ingen_data"
    return lab, hits


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
    h = hourly_quantities(h.sort_values("time").reset_index(drop=True))
    feats = installation_day_features(h[["time", "T_supply", "T_return"]], tout_d)
    if feats.empty:
        return None
    d = daily_quantities(h).join(feats.set_index("date"), how="inner")
    for col, base in [("E_day", "E_base"), ("Q_day", "Q_base"), ("dT_fw", "dT_base"), ("Tr_fw", "Tr_base")]:
        d[base] = fit_baseline(d, col)
    d["abs_corr_E_Tout"] = weather_correlation(d)
    d["night_day_ratio"] = d["E_night"] / d["E_daytime"].where(d["E_daytime"] > 0)
    d["window_episodes"] = open_window_episodes(h, tout_d).reindex(d.index).fillna(0)
    d["label"], hits = assign_labels(d)
    for c in hits:
        d[f"rule_{c}"] = hits[c]
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
    heating = df["T_outdoor"] < RULES["r1_max_tout"]
    summary = pd.DataFrame({
        "meter_days": df["label"].value_counts(),
        "meters": df.groupby("label")["MeterID"].nunique(),
        "heating_days": df.loc[heating, "label"].value_counts(),
    }).fillna(0).astype(int)
    summary["share_of_heating_days"] = summary["heating_days"] / summary["heating_days"].sum()
    summary.to_csv(os.path.join(OUT_DIR, "pseudo_label_summary.csv"))
    print("\nLabel counts:\n", summary.to_string())

    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    hits = pd.DataFrame({"days_rule_fires": df[rule_cols].sum(),
                         "share_of_heating_days": df.loc[heating, rule_cols].mean()})
    hits.index = [c.removeprefix("rule_") for c in hits.index]
    hits.to_csv(os.path.join(OUT_DIR, "pseudo_label_rule_hits.csv"))
    print("\nHow often each sub-rule fires (before priority is applied):\n", hits.to_string())

    if "iforest_anomaly_share" in df:
        heat = df[heating & (df["label"] != "ingen_data")].copy()
        heat["iforest_flag_day"] = heat["iforest_anomaly_share"] >= 0.5
        comp = heat.groupby("label").agg(
            meter_days=("MeterID", "size"),
            iforest_mean_anomaly_share=("iforest_anomaly_share", "mean"),
            share_flagged_by_iforest=("iforest_flag_day", "mean"),
        )
        flagged = heat[heat["iforest_flag_day"]]
        comp["share_of_iforest_flags"] = flagged["label"].value_counts(normalize=True)
        comp.to_csv(os.path.join(OUT_DIR, "iforest_vs_pseudolabels.csv"))
        print("\nWhat the Isolation Forest flags on heating days "
              "(days with >= 50 % anomalous hours):\n", comp.to_string())

    print(f"\nSaved {len(df):,} meter-days for {df.MeterID.nunique()} meters to {OUT_FILE} "
          f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
