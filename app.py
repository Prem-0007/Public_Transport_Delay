"""Flask web app – Public Transport Delay Analysis & Prediction (v2: dataset page, interactive dashboard, explained predictions, batch upload, JSON API)."""
import os, io, json, uuid
import numpy as np, joblib, pandas as pd
from flask import Flask, render_template, request, jsonify, Response
from src.config import MODEL_DIR, CLEAN_CSV, ROUTES
from src.preprocessing import add_features

app = Flask(__name__)
if not os.path.exists(os.path.join(MODEL_DIR, "results.json")):
    raise SystemExit("Models not found. Run `python train.py` first.")
RES = json.load(open(os.path.join(MODEL_DIR, "results.json")))
REGS = joblib.load(os.path.join(MODEL_DIR, "regressors.joblib"))
CLFS = joblib.load(os.path.join(MODEL_DIR, "classifiers.joblib"))
DF = pd.read_csv(CLEAN_CSV)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHER = ["Clear", "Rain", "Fog", "Snow", "Storm"]
DEFAULT = {"route_id": "B12", "hour": 8, "day_of_week": "Monday", "month": 1, "weather": "Clear", "temperature": 15,
           "passenger_load": 60, "traffic_index": 50, "incident": 0, "is_holiday": 0}
MED = {"traffic_index": float(DF.traffic_index.median()), "passenger_load": float(DF.passenger_load.median())}
STORE = {}  # batch results kept in memory for download


def make_row(f):
    rt = ROUTES[f["route_id"]]
    row = pd.DataFrame([{"route_id": f["route_id"], "transport_mode": rt["mode"], "distance_km": rt["distance_km"], "num_stops": rt["num_stops"],
                         "scheduled_headway_min": rt["headway"], "hour": int(f["hour"]), "day_of_week": f["day_of_week"], "month": int(f["month"]),
                         "weather": f["weather"], "temperature": float(f["temperature"]), "passenger_load": float(f["passenger_load"]),
                         "traffic_index": float(f["traffic_index"]), "incident": int(f["incident"]), "is_holiday": int(f["is_holiday"])}])
    return add_features(row)


def run_models(f):
    row = make_row(f)
    reg = {n: round(float(m.predict(row)[0]), 1) for n, m in REGS.items()}
    clf = {n: round(float(m.predict_proba(row)[0][1]) * 100, 1) for n, m in CLFS.items()}
    best = RES["best_regressor"]; pred = reg[best]
    # likely range from the spread of the Random Forest trees
    rf = REGS["Random Forest"]; Z = rf.named_steps["prep"].transform(row)
    tr = np.array([t.predict(Z)[0] for t in rf.named_steps["model"].estimators_]); lo, hi = np.percentile(tr, [10, 90])
    # explanation: change in the best model's prediction when each factor is set to a neutral value
    neutral = {"weather": ("Weather", "Clear"), "traffic_index": ("Traffic", MED["traffic_index"]), "passenger_load": ("Passenger load", MED["passenger_load"]),
               "incident": ("Incident", 0), "hour": ("Rush hour", 11), "is_holiday": ("Holiday", 0)}
    why = []
    for k, (lab, val) in neutral.items():
        g = dict(f); g[k] = val
        delta = pred - float(REGS[best].predict(make_row(g))[0])
        if abs(delta) >= 0.05: why.append({"factor": lab, "delta": round(delta, 1)})
    why.sort(key=lambda d: -abs(d["delta"]))
    level = "ok" if pred <= 3 else "warn" if pred <= 8 else "bad"
    label = {"ok": "Likely on time", "warn": "Minor delay expected", "bad": "Significant delay expected"}[level]
    return {"reg": reg, "clf": clf, "best": best, "pred": round(max(pred, 0), 1), "level": level, "label": label, "mode": ROUTES[f["route_id"]]["mode"],
            "prob": clf[RES["best_classifier"]], "lo": round(max(lo, 0), 1), "hi": round(hi, 1), "why": why[:6], "wmax": max([abs(w["delta"]) for w in why] + [1])}


@app.route("/")
def home(): return render_template("home.html", s=RES["summary"], r=RES, d=RES["dataset"])

@app.route("/dataset")
def dataset():
    prev = DF.head(12).drop(columns=["is_peak", "is_weekend", "is_delayed"]).to_html(classes="tbl", index=False, border=0)
    desc = DF[["distance_km", "hour", "temperature", "traffic_index", "passenger_load", "delay_minutes"]].describe().round(2).T.to_html(classes="tbl", border=0)
    return render_template("dataset.html", d=RES["dataset"], r=RES, preview=prev, desc=desc)

@app.route("/dashboard")
def dashboard(): return render_template("dashboard.html", modes=sorted(DF.transport_mode.unique()), weather=WEATHER, days=DAYS)

@app.route("/api/stats")
def api_stats():
    d = DF
    for col, key in [("transport_mode", "mode"), ("weather", "weather"), ("day_of_week", "day")]:
        v = request.args.get(key)
        if v and v != "All": d = d[d[col] == v]
    p = request.args.get("peak")
    if p in ("0", "1"): d = d[d.is_peak == int(p)]
    if len(d) == 0: return jsonify({"empty": True})
    g = lambda col, order=None: (lambda s: {"labels": [str(i) for i in (s.reindex(order).index if order is not None else s.index)],
                                            "values": [None if pd.isna(x) else round(float(x), 2) for x in (s.reindex(order) if order is not None else s).values]})(d.groupby(col).delay_minutes.mean())
    return jsonify({"empty": False, "trips": int(len(d)), "avg": round(float(d.delay_minutes.mean()), 2), "pct": round(float(d.is_delayed.mean() * 100), 1),
                    "max": round(float(d.delay_minutes.max()), 1), "incident_avg": round(float(d[d.incident == 1].delay_minutes.mean()), 1) if (d.incident == 1).any() else None,
                    "hour": g("hour", list(range(24))), "route": g("route_id"), "weather": g("weather"), "day": g("day_of_week", DAYS),
                    "month": g("month", list(range(1, 13)))})

@app.route("/eda")
def eda(): return render_template("eda.html")

@app.route("/models")
def models(): return render_template("models.html", r=RES)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    f, res = dict(DEFAULT), None
    if request.method == "POST":
        f = {k: request.form.get(k, v) for k, v in DEFAULT.items()}; res = run_models(f)
    return render_template("predict.html", routes=ROUTES, days=DAYS, weather=WEATHER, f=f, res=res)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True, silent=True) or {}
    f = {k: body.get(k, v) for k, v in DEFAULT.items()}
    if f["route_id"] not in ROUTES: return jsonify({"error": f"route_id must be one of {list(ROUTES)}"}), 400
    r = run_models(f); return jsonify({"input": f, "predicted_delay_minutes": r["pred"], "likely_range": [r["lo"], r["hi"]], "probability_delayed_pct": r["prob"],
                                       "verdict": r["label"], "per_model_minutes": r["reg"], "per_model_prob_delayed_pct": r["clf"], "main_factors": r["why"]})

@app.route("/batch", methods=["GET", "POST"])
def batch():
    table, token, err = None, None, None
    if request.method == "POST":
        try:
            up = pd.read_csv(request.files["file"]); out = []
            for _, r in up.iterrows():
                f = {k: (r[k] if k in up.columns and pd.notna(r[k]) else v) for k, v in DEFAULT.items()}
                if f["route_id"] not in ROUTES: f["route_id"] = DEFAULT["route_id"]
                m = run_models(f); out.append({**f, "predicted_delay_min": m["pred"], "prob_delayed_pct": m["prob"], "verdict": m["label"]})
            res = pd.DataFrame(out); token = uuid.uuid4().hex[:8]; STORE[token] = res.to_csv(index=False)
            table = res.head(25).to_html(classes="tbl", index=False, border=0); n = len(res)
            return render_template("batch.html", table=table, token=token, n=n, err=None)
        except Exception as e:
            err = f"Could not read the file: {e}"
    return render_template("batch.html", table=table, token=token, n=0, err=err)

@app.route("/download/<token>")
def download(token):
    return Response(STORE.get(token, ""), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})

@app.route("/sample.csv")
def sample():
    s = DF.sample(8, random_state=3)[list(DEFAULT)].to_csv(index=False)
    return Response(s, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=sample_trips.csv"})

if __name__ == "__main__":
    app.run(debug=False, port=5000)
