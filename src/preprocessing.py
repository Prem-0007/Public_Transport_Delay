"""Data preprocessing & wrangling."""
import pandas as pd
from .config import DELAY_THRESHOLD

PEAK_HOURS = {7, 8, 9, 16, 17, 18, 19}


def add_features(df):
    """Feature engineering shared by training and the web app."""
    df = df.copy()
    df["is_peak"] = df["hour"].isin(PEAK_HOURS).astype(int)
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    return df


def clean(df):
    """Returns (clean_df, report). `report` documents every wrangling step."""
    rep = {"rows_raw": len(df), "steps": [], "counts": {}}
    df = df.copy()
    rep["missing_before"] = df.isna().sum()[lambda s: s > 0].to_dict()

    n = len(df); df = df.drop_duplicates(); rep["counts"]["duplicates"] = n - len(df)
    rep["steps"].append(f"Removed {n-len(df)} duplicate rows")

    n = len(df); df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "delay_minutes", "route_id"]); rep["counts"]["rows_dropped_missing_key"] = n - len(df)
    rep["steps"].append("Parsed dates; dropped rows with no date / route / target")

    before_w = df["weather"].copy(); df["weather"] = df["weather"].str.strip().str.title()
    rep["counts"]["label_fixes"] = int((before_w.notna() & (before_w != df["weather"])).sum())
    rep["steps"].append("Standardised text labels (strip spaces, Title Case), e.g. 'RAIN ' -> 'Rain'")

    df["day_of_week"] = df["date"].dt.day_name()
    df["month"] = df["date"].dt.month

    rep["counts"]["imputed"] = {}
    for c in ["temperature", "passenger_load", "traffic_index"]:
        k = int(df[c].isna().sum()); rep["counts"]["imputed"][c] = k
        df[c] = df[c].fillna(df.groupby("transport_mode")[c].transform("median")).fillna(df[c].median())
        rep["steps"].append(f"Imputed {k} missing '{c}' with per-mode median")
    k = int(df["weather"].isna().sum()); rep["counts"]["imputed"]["weather"] = k
    df["weather"] = df["weather"].fillna(df["weather"].mode()[0])
    rep["steps"].append(f"Imputed {k} missing 'weather' with mode")

    q1, q3 = df["delay_minutes"].quantile([.25, .75]); iqr = q3 - q1
    hi = q3 + 3 * iqr
    k = int((df["delay_minutes"] > hi).sum()); rep["counts"]["outliers_capped"] = k; rep["counts"]["cap_value"] = round(float(hi), 1)
    df["delay_minutes"] = df["delay_minutes"].clip(upper=hi)
    rep["steps"].append(f"Capped {k} extreme delay outliers at Q3+3*IQR = {hi:.1f} min (winsorising)")

    df["passenger_load"] = df["passenger_load"].clip(0, 130)
    df["traffic_index"] = df["traffic_index"].clip(0, 100)
    df = add_features(df)
    df["is_delayed"] = (df["delay_minutes"] > DELAY_THRESHOLD).astype(int)
    rep["steps"].append(f"Engineered is_peak, is_weekend and target is_delayed (delay > {DELAY_THRESHOLD} min)")
    rep["rows_clean"] = len(df); rep["missing_after"] = int(df.isna().sum().sum())
    return df.reset_index(drop=True), rep
