"""Exploratory data analysis + visualisations (saved as PNG for the web app)."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .config import CHART_DIR

INK, ACCENT, TEAL = "#1d2733", "#e8a317", "#127c82"
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#c9ced4", "grid.color": "#e6e9ec",
                                    "axes.titleweight": "bold", "axes.titlesize": 13})
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _save(name):
    os.makedirs(CHART_DIR, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(CHART_DIR, name), dpi=110); plt.close()


def make_all(raw, df):
    miss = raw.isna().sum()[lambda s: s > 0].sort_values()
    plt.figure(figsize=(6, 3.5)); miss.plot.barh(color=ACCENT)
    plt.title("Missing values in raw data"); plt.xlabel("count"); _save("missing.png")

    plt.figure(figsize=(6.5, 3.8)); sns.histplot(df.delay_minutes, bins=50, color=TEAL)
    plt.axvline(5, color=ACCENT, ls="--", label="Delayed threshold (5 min)")
    plt.title("Distribution of delay (minutes)"); plt.legend(); _save("delay_dist.png")

    h = df.groupby("hour").delay_minutes.mean()
    plt.figure(figsize=(6.5, 3.8)); plt.plot(h.index, h.values, marker="o", color=INK)
    plt.fill_between(h.index, h.values, alpha=.15, color=TEAL)
    plt.title("Average delay by hour of day"); plt.xlabel("hour"); plt.ylabel("minutes"); _save("by_hour.png")

    plt.figure(figsize=(6.5, 3.8)); sns.barplot(data=df, x="transport_mode", y="delay_minutes",
                                                hue="transport_mode", palette="viridis", legend=False, errorbar=None)
    plt.title("Average delay by transport mode"); plt.xlabel(""); _save("by_mode.png")

    plt.figure(figsize=(6.5, 3.8)); sns.boxplot(data=df, x="weather", y="delay_minutes", hue="weather",
                                                palette="crest", legend=False, showfliers=False)
    plt.title("Delay distribution by weather"); plt.xlabel(""); _save("by_weather.png")

    d = df.groupby("day_of_week").delay_minutes.mean().reindex(DAYS)
    plt.figure(figsize=(6.5, 3.8)); sns.barplot(x=d.index, y=d.values, color=TEAL)
    plt.title("Average delay by day of week"); plt.xticks(rotation=35); plt.ylabel("minutes"); _save("by_day.png")

    r = df.groupby("route_id").delay_minutes.mean().sort_values()
    plt.figure(figsize=(6.5, 4)); r.plot.barh(color=INK)
    plt.title("Average delay by route"); plt.xlabel("minutes"); _save("by_route.png")

    m = df.groupby("month").delay_minutes.mean()
    plt.figure(figsize=(6.5, 3.8)); plt.plot(m.index, m.values, marker="s", color=ACCENT, lw=2)
    plt.title("Monthly trend of average delay"); plt.xlabel("month"); plt.xticks(range(1, 13)); _save("by_month.png")

    s = df.sample(min(1500, len(df)), random_state=0)
    plt.figure(figsize=(6.5, 3.8)); sns.scatterplot(data=s, x="traffic_index", y="delay_minutes",
                                                    hue="transport_mode", alpha=.55, s=18)
    plt.title("Traffic index vs delay"); _save("traffic_scatter.png")

    cols = ["delay_minutes", "hour", "temperature", "passenger_load", "traffic_index",
            "distance_km", "num_stops", "incident", "is_peak"]
    plt.figure(figsize=(7, 5.5)); sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                                             annot_kws={"size": 8})
    plt.title("Correlation heatmap"); _save("corr.png")

    p = df.pivot_table(index="day_of_week", columns="hour", values="delay_minutes", aggfunc="mean").reindex(DAYS)
    plt.figure(figsize=(9, 3.8)); sns.heatmap(p, cmap="YlOrRd", cbar_kws={"label": "min"})
    plt.title("Average delay: day x hour"); plt.xlabel("hour"); plt.ylabel(""); _save("heat_day_hour.png")

    plt.figure(figsize=(5, 3.8)); sns.barplot(data=df, x="incident", y="delay_minutes", hue="incident",
                                              palette=[TEAL, ACCENT], legend=False, errorbar=None)
    plt.xticks([0, 1], ["No incident", "Incident"]); plt.title("Impact of incidents"); plt.xlabel(""); _save("incident.png")


def extra(df):
    r = df.pivot_table(index="route_id", columns="hour", values="delay_minutes", aggfunc="mean")
    plt.figure(figsize=(10, 4)); sns.heatmap(r, cmap="YlOrRd", cbar_kws={"label": "min"}); plt.title("Average delay: route x hour"); plt.xlabel("hour"); plt.ylabel(""); _save("route_hour.png")
    w = df.pivot_table(index="weather", columns="transport_mode", values="delay_minutes", aggfunc="mean")
    plt.figure(figsize=(6, 3.8)); sns.heatmap(w, annot=True, fmt=".1f", cmap="YlOrRd", cbar=False); plt.title("Average delay: weather x mode"); plt.xlabel(""); plt.ylabel(""); _save("weather_mode.png")
    d = df.assign(date=pd.to_datetime(df.date)).groupby("date").delay_minutes.mean().rolling(14, min_periods=3).mean()
    plt.figure(figsize=(6.5, 3.6)); plt.plot(d.index, d.values, color=TEAL); plt.fill_between(d.index, d.values, alpha=.15, color=TEAL); plt.title("14-day rolling average delay"); plt.ylabel("minutes"); _save("rolling.png")
    plt.figure(figsize=(6.5, 3.8)); sns.violinplot(data=df, x="transport_mode", y="delay_minutes", hue="transport_mode", palette="viridis", legend=False, cut=0); plt.title("Delay distribution by mode"); plt.xlabel(""); _save("violin.png")
    b = df.assign(load=pd.cut(df.passenger_load, [0, 40, 60, 80, 100, 140], labels=["<40", "40-60", "60-80", "80-100", ">100"])).groupby("load", observed=True).delay_minutes.mean()
    plt.figure(figsize=(5.5, 3.6)); b.plot.bar(color=INK, rot=0); plt.title("Average delay by passenger load (%)"); plt.ylabel("minutes"); plt.xlabel(""); _save("load.png")
    k = (df.groupby(["transport_mode", "is_peak"]).is_delayed.mean().unstack() * 100); k.columns = ["Off-peak", "Peak"]
    k.plot.bar(figsize=(6, 3.8), color=[TEAL, ACCENT], rot=0); plt.title("% of trips delayed (>5 min): peak vs off-peak"); plt.ylabel("% delayed"); plt.xlabel(""); _save("peak_mode.png")

def summary_stats(df):
    return {
        "records": int(len(df)),
        "avg_delay": round(float(df.delay_minutes.mean()), 2),
        "pct_delayed": round(float(df.is_delayed.mean() * 100), 1),
        "worst_route": df.groupby("route_id").delay_minutes.mean().idxmax(),
        "worst_hour": int(df.groupby("hour").delay_minutes.mean().idxmax()),
        "worst_weather": df.groupby("weather").delay_minutes.mean().idxmax(),
        "routes": int(df.route_id.nunique()),
    }
