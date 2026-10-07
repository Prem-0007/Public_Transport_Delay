"""
Generates a realistic synthetic public-transport dataset (6,000+ records).
It intentionally contains missing values, duplicates and outliers so that the
preprocessing / wrangling stage has real work to do.

TO USE A REAL DATASET: replace data/transport_delays_raw.csv with your own CSV
that has the same column names (see README.md) and run `python train.py`.
"""
import numpy as np
import pandas as pd
from src.config import ROUTES, RAW_CSV

rng = np.random.default_rng(42)
N = 6000

dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
route_ids = rng.choice(list(ROUTES), N)
df = pd.DataFrame({"trip_id": [f"T{100000+i}" for i in range(N)],
                   "date": dates, "route_id": route_ids})
df["transport_mode"] = df.route_id.map(lambda r: ROUTES[r]["mode"])
df["distance_km"] = df.route_id.map(lambda r: ROUTES[r]["distance_km"])
df["num_stops"] = df.route_id.map(lambda r: ROUTES[r]["num_stops"])
df["scheduled_headway_min"] = df.route_id.map(lambda r: ROUTES[r]["headway"])
df["day_of_week"] = df.date.dt.day_name()
df["month"] = df.date.dt.month

peak_hours = np.array([7, 8, 9, 16, 17, 18, 19])
hours = np.where(rng.random(N) < 0.45, rng.choice(peak_hours, N), rng.integers(5, 23, N))
df["hour"] = hours
is_peak = np.isin(hours, peak_hours)
weekend = df.day_of_week.isin(["Saturday", "Sunday"]).values
df["is_holiday"] = (rng.random(N) < 0.04).astype(int)

season_temp = 14 + 12 * np.sin((df.month.values - 4) / 12 * 2 * np.pi)
df["temperature"] = np.round(season_temp + rng.normal(0, 4, N), 1)
cold = df.temperature.values < 4
wprobs = np.where(cold[:, None], [0.35, 0.20, 0.10, 0.30, 0.05], [0.55, 0.28, 0.08, 0.01, 0.08])
df["weather"] = [rng.choice(["Clear", "Rain", "Fog", "Snow", "Storm"], p=p / p.sum()) for p in wprobs]

df["traffic_index"] = np.clip(np.round(35 + 35 * is_peak - 15 * weekend + rng.normal(0, 12, N)), 0, 100)
df["passenger_load"] = np.clip(np.round(40 + 40 * is_peak - 12 * weekend + rng.normal(0, 14, N)), 5, 130)
df["incident"] = (rng.random(N) < 0.06).astype(int)

mode_base = df.transport_mode.map({"Bus": 3.0, "Tram": 2.0, "Metro": 0.8, "Train": 1.8}).values
w_eff = df.weather.map({"Clear": 0, "Rain": 1.5, "Fog": 2.0, "Snow": 4.5, "Storm": 5.5}).values
mode_traffic = df.transport_mode.map({"Bus": 0.07, "Tram": 0.04, "Metro": 0.005, "Train": 0.01}).values

delay = (mode_base + w_eff + mode_traffic * df.traffic_index.values
         + 0.03 * np.maximum(df.passenger_load.values - 70, 0) * 2
         + 6 * is_peak + 14 * df.incident.values
         - 2 * df.is_holiday.values + 0.05 * df.distance_km.values
         + rng.gamma(2.0, 1.2, N) - 10.0)
df["delay_minutes"] = np.round(np.clip(delay, -3, None), 1)

# ---- inject data-quality problems -------------------------------------------
for col, frac in [("temperature", .03), ("passenger_load", .04), ("weather", .02), ("traffic_index", .03)]:
    df.loc[rng.choice(N, int(N * frac), replace=False), col] = np.nan
df.loc[rng.choice(N, 25, replace=False), "delay_minutes"] += rng.uniform(60, 150, 25)  # outliers
df = pd.concat([df, df.sample(60, random_state=1)], ignore_index=True)                # duplicates
df.loc[rng.choice(len(df), 40, replace=False), "weather"] = "RAIN "                    # inconsistent labels
df = df.sample(frac=1, random_state=3).reset_index(drop=True)
df["date"] = df["date"].dt.strftime("%Y-%m-%d")
df.to_csv(RAW_CSV, index=False)
print(f"Saved {len(df)} rows -> {RAW_CSV}")
