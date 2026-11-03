"""
Test the trained fault classifier on the radiator sensor data from the apartments.

This is the real test: the model was trained on pseudo-labels from district
heating meters, and here it is scored against faults that were staged and
logged by hand on actual radiators.

INPUT FILES
-----------
1. Sensor readings (any sampling rate, e.g. every 30 s), CSV with columns
       timestamp, radiator_id, T_supply, T_return
   timestamp in local Danish time, e.g. 2026-11-20 14:03:30

2. Fault log, CSV with one row per staged fault:
       radiator_id, start, end, fault_type
   fault_type must be a key in FAULT_TO_CLASS below. Times not covered by the
   log are treated as normal operation.

3. Outdoor temperature: fetched from DMI for the period, or pass --outdoor
   with a CSV (RoundedReadTime, T_outdoor), same format as Data/dmi_outdoor_temp_cache.csv.

USAGE
-----
    python Validate_On_Radiator_Data.py --sensors radiator_data.csv --faults fault_log.csv
    python Validate_On_Radiator_Data.py --make-example     # writes example input files

WHAT IS MEASURED
----------------
The model classifies calendar days. A day counts as a fault day when a fault
was active for at least MIN_FAULT_HOURS of it; days with a shorter fault are
left out, because they are neither clearly normal nor clearly faulty.
-> Keep staged faults on for a full day where it is safe (towel over the
   radiator, thermostat on max). A 2-hour fault is diluted in a daily average.

The first REFERENCE_DAYS fault-free days per radiator define its normal
behaviour (all features are relative to this), so start with a calm period.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pseudo_Labeling"))
from heat_features import installation_day_features, to_hourly, daily_outdoor  # noqa: E402

MODEL_FILE = "Models/fault_classifier.pkl"
OUT_DIR = "Outputs"
MIN_FAULT_HOURS = 12
REFERENCE_DAYS = 5
DMI_STATION = "06030"   # Flyvestation Aalborg

# Which of the group's rule classes each staged fault is EXPECTED to look like.
# This is a hypothesis, not a fact, and the meter rules describe a whole
# dwelling while the staged faults affect one radiator. Report where the
# mapping holds and where it does not.
FAULT_TO_CLASS = {
    "normal": "normal",
    "thermostat_off": "radiator_slukket",
    "valve_closed": "radiator_slukket",
    "radiator_covered": "blokeret_radiator",
    "thermostat_blocked": "blokeret_radiator",
    "thermostat_max": "radiator_helt_aaben",
    "window_open": "aabent_vindue",
}


def fetch_dmi(start, end, station=DMI_STATION):
    import requests
    url = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items"
    params = {"stationId": station, "parameterId": "temp_dry",
              "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}", "limit": 300000}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    rec = [{"RoundedReadTime": pd.to_datetime(f["properties"]["observed"]).tz_convert("Europe/Copenhagen")
            .tz_localize(None).round("h"), "T_outdoor": f["properties"]["value"]}
           for f in r.json().get("features", [])]
    return pd.DataFrame(rec).drop_duplicates("RoundedReadTime")


def day_labels(days: pd.DatetimeIndex, log: pd.DataFrame) -> pd.DataFrame:
    """For each day: the fault type active the longest, and for how many hours."""
    out = pd.DataFrame(index=days, data={"fault_type": "normal", "fault_hours": 0.0, "any_fault_hours": 0.0})
    for _, f in log.iterrows():
        for day in days:
            lo, hi = max(f["start"], day), min(f["end"], day + pd.Timedelta("1D"))
            hrs = (hi - lo).total_seconds() / 3600
            if hrs > 0:
                out.loc[day, "any_fault_hours"] += hrs
                if hrs > out.loc[day, "fault_hours"]:
                    out.loc[day, ["fault_type", "fault_hours"]] = [f["fault_type"], hrs]
    return out


def make_example(path_sensors="example_radiator_data.csv", path_faults="example_fault_log.csv"):
    """Synthetic data in the expected format, only to test that the pipeline runs."""
    rng = np.random.default_rng(0)
    t = pd.date_range("2026-11-10", "2026-12-10", freq="30s", inclusive="left")
    rows = []
    faults = [("rad_living", "2026-11-18 00:00", "2026-11-19 00:00", "radiator_covered"),
              ("rad_living", "2026-11-24 00:00", "2026-11-25 12:00", "thermostat_off"),
              ("rad_living", "2026-12-02 08:00", "2026-12-03 08:00", "thermostat_max"),
              ("rad_bedroom", "2026-11-21 00:00", "2026-11-22 00:00", "thermostat_max"),
              ("rad_bedroom", "2026-11-28 00:00", "2026-11-29 00:00", "valve_closed")]
    log = pd.DataFrame(faults, columns=["radiator_id", "start", "end", "fault_type"])
    for rid in ["rad_living", "rad_bedroom"]:
        hour = np.asarray(t.hour + t.minute / 60)
        ts = 45 + 3 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.3, len(t))
        tr = ts - 14 + rng.normal(0, 0.4, len(t))
        for _, f in log[log.radiator_id == rid].iterrows():
            m = np.asarray((t >= pd.Timestamp(f.start)) & (t < pd.Timestamp(f.end)))
            if f.fault_type in ("radiator_covered", "thermostat_max"):
                tr[m] = ts[m] - 4 + rng.normal(0, 0.4, m.sum())
            elif f.fault_type in ("thermostat_off", "valve_closed"):
                ts[m] = 24 + rng.normal(0, 0.3, m.sum())
                tr[m] = 22 + rng.normal(0, 0.3, m.sum())
        rows.append(pd.DataFrame({"timestamp": t, "radiator_id": rid, "T_supply": ts.round(2), "T_return": tr.round(2)}))
    pd.concat(rows).to_csv(path_sensors, index=False)
    log.to_csv(path_faults, index=False)
    print(f"Wrote {path_sensors} and {path_faults}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors")
    ap.add_argument("--faults")
    ap.add_argument("--outdoor", help="CSV with RoundedReadTime, T_outdoor (otherwise fetched from DMI)")
    ap.add_argument("--model", default=MODEL_FILE)
    ap.add_argument("--make-example", action="store_true")
    a = ap.parse_args()

    if a.make_example:
        make_example()
        return
    if not (a.sensors and a.faults):
        ap.error("--sensors and --faults are required")

    bundle = joblib.load(a.model)
    model, features = bundle["model"], bundle["features"]

    s = pd.read_csv(a.sensors, parse_dates=["timestamp"]).rename(columns={"timestamp": "time"})
    log = pd.read_csv(a.faults, parse_dates=["start", "end"])
    unknown = set(log["fault_type"]) - set(FAULT_TO_CLASS)
    if unknown:
        raise ValueError(f"Unknown fault types in log: {unknown}. Add them to FAULT_TO_CLASS.")

    if a.outdoor:
        dmi = pd.read_csv(a.outdoor, parse_dates=["RoundedReadTime"])
    else:
        dmi = fetch_dmi(s["time"].min().tz_localize("Europe/Copenhagen").tz_convert("UTC") - pd.Timedelta("1D"),
                        s["time"].max().tz_localize("Europe/Copenhagen").tz_convert("UTC") + pd.Timedelta("1D"))
    tout = daily_outdoor(dmi)

    results = []
    for rid, g in s.groupby("radiator_id"):
        hourly = to_hourly(g)
        rlog = log[log["radiator_id"] == rid]
        days = pd.DatetimeIndex(sorted(hourly["time"].dt.floor("D").unique()))
        truth = day_labels(days, rlog)
        clean = truth.index[truth["any_fault_hours"] == 0]
        ref = clean[:REFERENCE_DAYS]
        feats = installation_day_features(hourly, tout, reference_days=ref)
        feats = feats.set_index("date").join(truth)
        feats["radiator_id"] = rid
        feats["is_reference_day"] = feats.index.isin(ref)
        results.append(feats)

    df = pd.concat(results).reset_index()
    df["expected_class"] = df["fault_type"].map(FAULT_TO_CLASS)
    df["predicted_class"] = model.predict(df[features])
    proba = model.predict_proba(df[features])
    for i, c in enumerate(model.classes_):
        df[f"p_{c}"] = proba[:, i]

    # Score only clear days, and never the reference days (they define "normal").
    clear = (df["any_fault_hours"] == 0) | (df["fault_hours"] >= MIN_FAULT_HOURS)
    ev = df[clear & ~df["is_reference_day"]]

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "radiator_validation_days.csv"), index=False)

    lines = ["VALIDATION ON RADIATOR DATA (staged faults)\n",
             f"Days scored: {len(ev)}  (excluded: {(~clear).sum()} partial-fault days, "
             f"{df['is_reference_day'].sum()} reference days)\n"]
    if len(ev):
        y, p = ev["expected_class"], ev["predicted_class"]
        labels = [c for c in bundle["train_classes"] if c in set(y) | set(p)]
        lines.append(classification_report(y, p, labels=labels, digits=3, zero_division=0))
        lines.append("Confusion matrix (rows = expected, cols = predicted):")
        lines.append(pd.DataFrame(confusion_matrix(y, p, labels=labels), index=labels, columns=labels).to_string())
        tf, pf = y != "normal", p != "normal"
        rule = ev["rel_dT"] < 0.5
        lines.append("\nFAULT vs NORMAL")
        for name, pred in [("model", pf), ("rule: daily ΔT < 50 % of reference", rule)]:
            lines.append(f"  {name:36s} precision {precision_score(tf, pred, zero_division=0):.3f}  "
                         f"recall {recall_score(tf, pred, zero_division=0):.3f}  F1 {f1_score(tf, pred, zero_division=0):.3f}")
        lines.append("\nPer staged fault type (share of days predicted as each class):")
        lines.append(pd.crosstab(ev["fault_type"], ev["predicted_class"], normalize="index").round(2).to_string())
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT_DIR, "radiator_validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
