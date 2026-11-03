"""
Shared feature code for the supervised fault classifier.

The SAME function is used on
  - the Zenodo heat-meter data (training), and
  - the radiator sensor data from the apartments (testing),
so the model always sees features computed in exactly the same way.

Design rule: features use ONLY what the radiator sensor can measure
(supply temperature, return temperature, outdoor temperature). Flow, energy,
volume and meter error codes are deliberately left out, because
  1. the radiator sensor does not measure them, and
  2. the pseudo-labels are built from them. Keeping them out of the features
     means the model cannot simply re-learn the labelling rules.

All features are dimensionless or relative to the installation's own normal
behaviour (a fixed reference), because a district heating meter runs at
~70/35 °C while a radiator runs at ~45/30 °C. Absolute temperatures would not
transfer between the two.

Unit of classification: one installation-day (calendar day).
"""
import numpy as np
import pandas as pd

ROOM_TEMP = 20.0          # assumed room temperature for the cooling ratio
BASELINE_MIN_DAYS = 3     # reference days needed to define "normal" for an installation
HEATING_MAX_TOUT = 12.0   # daily mean outdoor temp below which it is a heating day
MIN_HOURS_PER_DAY = 18    # days with fewer hourly values are dropped
HDD_BASE = 17.0           # heating degree base temperature

# Features fed to the model. Everything else in the table is for inspection.
FEATURES = [
    "rel_dT",            # today's mean ΔT / reference ΔT
    "rel_dT_p10",        # 10th percentile hourly ΔT / reference ΔT
    "rel_dT_std",        # hourly ΔT std / reference ΔT
    "frac_low_dT",       # share of hours with ΔT < 50 % of reference
    "frac_neg_dT",       # share of hours with ΔT < 0 (return hotter than supply)
    "d_Tr",              # (mean return - reference return) / reference ΔT
    "d_Ts",              # (mean supply - reference supply) / reference ΔT
    "theta",             # (Tr - room) / (Ts - room): 0 = perfect cooling, 1 = none
    "d_theta",           # theta - reference theta
    "rel_dT_change_1d",  # rel_dT minus yesterday's rel_dT
    "d_Tr_7d",           # mean of d_Tr over the last 7 days (slow drift)
    "Tr_max_rise",       # largest hour-to-hour rise in return temp / reference ΔT
    "Ts_range",          # (p90 - p10) of hourly supply temp / reference ΔT
    "Tr_peak_2h",        # (highest 2-hour mean return temp - daily median) / reference ΔT
    "dT_peak_2h",        # (highest 2-hour mean ΔT - daily median ΔT) / reference ΔT
    "night_day_dT",      # (mean ΔT 00-06 - mean ΔT 10-16) / reference ΔT
    "HDD",               # heating degree days, from outdoor temperature
]


def _daily_from_hourly(h: pd.DataFrame) -> pd.DataFrame:
    """h: one installation's hourly data with columns time, T_supply, T_return."""
    h = h.copy()
    h["date"] = h["time"].dt.floor("D")
    h["dT"] = h["T_supply"] - h["T_return"]
    h["Tr_rise"] = h["T_return"].diff().where(h["time"].diff() == pd.Timedelta("1h"))
    # short peaks within the day (an open window lasts hours, not days)
    hh = h.set_index("time")
    h["Tr_2h"] = hh["T_return"].rolling("2h", min_periods=2).mean().to_numpy()
    h["dT_2h"] = hh["dT"].rolling("2h", min_periods=2).mean().to_numpy()
    hour = h["time"].dt.hour

    g = h.groupby("date")
    d = pd.DataFrame({
        "n_hours": g["dT"].count(),
        "Ts_mean": g["T_supply"].mean(),
        "Tr_mean": g["T_return"].mean(),
        "dT_mean": g["dT"].mean(),
        "dT_std": g["dT"].std(),
        "dT_p10": g["dT"].quantile(0.10),
        "Ts_p10": g["T_supply"].quantile(0.10),
        "Ts_p90": g["T_supply"].quantile(0.90),
        "Tr_max_rise_abs": g["Tr_rise"].max(),
        "frac_neg_dT": g["dT"].apply(lambda s: float((s < 0).mean())),
        "Tr_peak_abs": g["Tr_2h"].max() - g["T_return"].median(),
        "dT_peak_abs": g["dT_2h"].max() - g["dT"].median(),
        "night_day_abs": (h["dT"].where(hour < 6).groupby(h["date"]).mean()
                          - h["dT"].where(hour.between(10, 15)).groupby(h["date"]).mean()),
    })
    num = (d["Tr_mean"] - ROOM_TEMP)
    den = (d["Ts_mean"] - ROOM_TEMP)
    d["theta"] = (num / den.where(den > 2)).clip(-1, 2)
    return d, h


def installation_day_features(hourly: pd.DataFrame,
                              tout_daily: pd.Series | None = None,
                              reference_days=None) -> pd.DataFrame:
    """
    hourly: one installation, columns ['time', 'T_supply', 'T_return'],
            one row per hour (use to_hourly() on raw sensor readings first).
    tout_daily: outdoor mean temperature indexed by date (midnight).
    reference_days: which days define this installation's NORMAL behaviour.
        - None (training on heat meters): all heating days of the meter.
          The median is robust as long as the meter is normal most of the time.
        - On radiator data: pass the dates of a known-normal calibration
          period, e.g. the first week before any fault was staged.

    Returns one row per day with FEATURES plus some raw daily values.
    """
    hourly = hourly.sort_values("time")
    d, h = _daily_from_hourly(hourly)
    d = d[d["n_hours"] >= MIN_HOURS_PER_DAY]
    if d.empty:
        return d.reset_index()

    if tout_daily is not None:
        d["T_outdoor"] = tout_daily.reindex(d.index).values
    else:
        d["T_outdoor"] = np.nan
    d["HDD"] = (HDD_BASE - d["T_outdoor"]).clip(lower=0)

    # Fixed reference = the installation's typical behaviour.
    if reference_days is None:
        ref_mask = d["T_outdoor"] < HEATING_MAX_TOUT
        if ref_mask.sum() < BASELINE_MIN_DAYS:
            ref_mask = pd.Series(True, index=d.index)
    else:
        ref_idx = pd.DatetimeIndex(pd.to_datetime(list(reference_days))).floor("D")
        ref_mask = d.index.isin(ref_idx)
    ref = d.loc[ref_mask]
    if len(ref) < BASELINE_MIN_DAYS:
        raise ValueError(f"Need at least {BASELINE_MIN_DAYS} usable reference days, got {len(ref)}")
    r_dT, r_Tr, r_Ts, r_theta = (ref["dT_mean"].median(), ref["Tr_mean"].median(),
                                 ref["Ts_mean"].median(), ref["theta"].median())
    if not np.isfinite(r_dT) or r_dT <= 1.0:   # reference ΔT below 1 °C is not usable as a scale
        return d.iloc[0:0].reset_index()
    scale = r_dT

    d["ref_dT"] = r_dT
    d["rel_dT"] = d["dT_mean"] / scale
    d["rel_dT_p10"] = d["dT_p10"] / scale
    d["rel_dT_std"] = d["dT_std"] / scale
    d["d_Tr"] = (d["Tr_mean"] - r_Tr) / scale
    d["d_Ts"] = (d["Ts_mean"] - r_Ts) / scale
    d["d_theta"] = d["theta"] - r_theta
    d["Tr_max_rise"] = d["Tr_max_rise_abs"] / scale
    d["Ts_range"] = (d["Ts_p90"] - d["Ts_p10"]) / scale
    d["Tr_peak_2h"] = d["Tr_peak_abs"] / scale
    d["dT_peak_2h"] = d["dT_peak_abs"] / scale
    d["night_day_dT"] = d["night_day_abs"] / scale
    d["frac_low_dT"] = (h["dT"] < 0.5 * scale).groupby(h["date"]).mean().reindex(d.index)

    prev = d["rel_dT"].shift(1).where(d.index.to_series().diff() == pd.Timedelta("1D"))
    d["rel_dT_change_1d"] = d["rel_dT"] - prev
    d["d_Tr_7d"] = d["d_Tr"].rolling("7D", min_periods=3).mean()

    d.index.name = "date"
    return d.reset_index()


def to_hourly(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Average sub-hourly sensor readings (e.g. every 30 s) to hourly means."""
    out = (df.set_index(time_col)[["T_supply", "T_return"]]
             .resample("1h").mean()
             .dropna()
             .reset_index())
    return out


def daily_outdoor(dmi: pd.DataFrame) -> pd.Series:
    """DMI hourly cache (RoundedReadTime, T_outdoor) -> daily mean indexed by date."""
    t = pd.to_datetime(dmi["RoundedReadTime"])
    return dmi.groupby(t.dt.floor("D"))["T_outdoor"].mean()
