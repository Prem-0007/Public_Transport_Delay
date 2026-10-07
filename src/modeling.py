"""Train, tune & compare ML models: 3 regressors (delay minutes) + 3 classifiers (delayed / on-time)."""
import os, time
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from .config import NUM_FEATURES, CAT_FEATURES, MODEL_DIR, CHART_DIR
TEAL, AMBER, INK = "#127c82", "#e8a317", "#1d2733"

def _prep(): return ColumnTransformer([("num", StandardScaler(), NUM_FEATURES), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)])
def _save(n): plt.tight_layout(); plt.savefig(os.path.join(CHART_DIR, n), dpi=110); plt.close()

def run(df):
    os.makedirs(MODEL_DIR, exist_ok=True)
    X = df[NUM_FEATURES + CAT_FEATURES]; out = {"regression": [], "classification": [], "tuning": {}}
    # ===================== REGRESSION =====================
    y = df["delay_minutes"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    out["n_train"], out["n_test"] = len(Xtr), len(Xte)
    lin = Pipeline([("prep", _prep()), ("model", LinearRegression())])
    rf = GridSearchCV(Pipeline([("prep", _prep()), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
                      {"model__n_estimators": [200, 400], "model__max_depth": [12, None], "model__min_samples_leaf": [2, 4]}, cv=3, scoring="r2", n_jobs=1)
    gb = RandomizedSearchCV(Pipeline([("prep", _prep()), ("model", GradientBoostingRegressor(random_state=42))]),
                      {"model__n_estimators": [200, 300, 400], "model__max_depth": [2, 3, 4], "model__learning_rate": [.05, .08, .1], "model__subsample": [.8, 1.0]},
                      n_iter=8, cv=3, scoring="r2", random_state=42, n_jobs=1)
    fitted, preds = {}, {}
    for name, m in [("Linear Regression", lin), ("Random Forest", rf), ("Gradient Boosting", gb)]:
        t = time.time(); m.fit(Xtr, ytr); dt = time.time() - t
        est = getattr(m, "best_estimator_", m)
        if hasattr(m, "best_params_"): out["tuning"][f"reg_{name}"] = {k.replace("model__", ""): v for k, v in m.best_params_.items()}
        p = est.predict(Xte); fitted[name], preds[name] = est, p
        cv = cross_val_score(est, Xtr, ytr, cv=5, scoring="r2")
        out["regression"].append({"model": name, "MAE": round(mean_absolute_error(yte, p), 3), "RMSE": round(float(np.sqrt(mean_squared_error(yte, p))), 3),
                                  "R2": round(r2_score(yte, p), 3), "CV_R2": round(float(cv.mean()), 3), "CV_std": round(float(cv.std()), 3), "train_sec": round(dt, 1)})
    best_reg = max(out["regression"], key=lambda r: r["R2"])["model"]; out["best_regressor"] = best_reg
    joblib.dump(fitted, os.path.join(MODEL_DIR, "regressors.joblib"))
    # ===================== CLASSIFICATION =====================
    yc = df["is_delayed"]
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, yc, test_size=.2, random_state=42, stratify=yc)
    lr = Pipeline([("prep", _prep()), ("model", LogisticRegression(max_iter=1000))])
    rfc = GridSearchCV(Pipeline([("prep", _prep()), ("model", RandomForestClassifier(random_state=42, n_jobs=-1))]),
                       {"model__n_estimators": [200, 400], "model__max_depth": [10, None], "model__min_samples_leaf": [2, 4]}, cv=3, scoring="f1", n_jobs=1)
    gbc = Pipeline([("prep", _prep()), ("model", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42))])
    cfit, cpred, cprob = {}, {}, {}
    for name, m in [("Logistic Regression", lr), ("Random Forest", rfc), ("Gradient Boosting", gbc)]:
        m.fit(Xtr_c, ytr_c); est = getattr(m, "best_estimator_", m)
        if hasattr(m, "best_params_"): out["tuning"][f"clf_{name}"] = {k.replace("model__", ""): v for k, v in m.best_params_.items()}
        p, pr = est.predict(Xte_c), est.predict_proba(Xte_c)[:, 1]; cfit[name], cpred[name], cprob[name] = est, p, pr
        cv = cross_val_score(est, Xtr_c, ytr_c, cv=5, scoring="f1")
        out["classification"].append({"model": name, "Accuracy": round(accuracy_score(yte_c, p), 3), "Precision": round(precision_score(yte_c, p), 3),
                                      "Recall": round(recall_score(yte_c, p), 3), "F1": round(f1_score(yte_c, p), 3),
                                      "ROC_AUC": round(roc_auc_score(yte_c, pr), 3), "CV_F1": round(float(cv.mean()), 3)})
    best_clf = max(out["classification"], key=lambda r: r["F1"])["model"]; out["best_classifier"] = best_clf
    joblib.dump(cfit, os.path.join(MODEL_DIR, "classifiers.joblib"))
    out["confusion_best"] = confusion_matrix(yte_c, cpred[best_clf]).tolist()
    # ===================== EXTRA ANALYSIS =====================
    pb = preds[best_reg]; res = yte.values - pb
    out["residual"] = {"mean": round(float(res.mean()), 3), "std": round(float(res.std()), 3), "within_2min": round(float((np.abs(res) <= 2).mean() * 100), 1),
                       "within_5min": round(float((np.abs(res) <= 5).mean() * 100), 1)}
    modes = df.loc[Xte.index, "transport_mode"]; out["mae_by_mode"] = {m: round(float(np.abs(res[(modes == m).values]).mean()), 3) for m in sorted(modes.unique())}
    pi = permutation_importance(fitted[best_reg], Xte, yte, n_repeats=5, random_state=0, n_jobs=1, scoring="r2")
    pimp = pd.Series(pi.importances_mean, index=X.columns).sort_values(); out["perm_importance"] = {k: round(float(v), 4) for k, v in pimp.sort_values(ascending=False).items()}
    _charts(out, fitted, preds, yte, cpred, cprob, yte_c, res, modes, pimp, best_reg, best_clf)
    return out

def _charts(out, fitted, preds, yte, cpred, cprob, yte_c, res, modes, pimp, best_reg, best_clf):
    r = pd.DataFrame(out["regression"]).set_index("model")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    r[["MAE", "RMSE"]].plot.bar(ax=ax[0], color=[TEAL, AMBER], rot=15); ax[0].set_title("Error (lower is better)")
    r[["R2", "CV_R2"]].plot.bar(ax=ax[1], color=[INK, "#7a8794"], rot=15); ax[1].set_title("R² (higher is better)")
    for a in ax: a.set_xlabel("")
    _save("cmp_reg.png")
    c = pd.DataFrame(out["classification"]).set_index("model")[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
    c.plot.bar(figsize=(8, 3.8), rot=12, colormap="viridis"); plt.ylim(.5, 1.0); plt.xlabel(""); plt.title("Classification metrics (delayed vs on-time)"); plt.legend(fontsize=7, ncol=5); _save("cmp_clf.png")
    p = preds[best_reg]; plt.figure(figsize=(5.5, 4.5)); plt.scatter(yte, p, s=10, alpha=.45, color=TEAL)
    lim = [min(yte.min(), p.min()), max(yte.max(), p.max())]; plt.plot(lim, lim, "--", color=AMBER)
    plt.xlabel("Actual delay (min)"); plt.ylabel("Predicted"); plt.title(f"Actual vs Predicted – {best_reg}"); _save("actual_vs_pred.png")
    plt.figure(figsize=(6, 3.8)); sns.histplot(res, bins=45, color=TEAL, kde=True); plt.axvline(0, color=AMBER, ls="--")
    plt.title(f"Residuals ({best_reg}) – within 2 min: {out['residual']['within_2min']}%"); plt.xlabel("actual − predicted (min)"); _save("residuals.png")
    s = pd.Series(out["mae_by_mode"]).sort_values(); plt.figure(figsize=(5.5, 3.6)); s.plot.barh(color=INK); plt.title("Prediction error (MAE) by transport mode"); plt.xlabel("minutes"); _save("mae_mode.png")
    plt.figure(figsize=(6.5, 4.2)); pimp.tail(12).plot.barh(color=INK); plt.title("Permutation importance (drop in R² when shuffled)"); _save("feat_imp.png")
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for a, n in zip(ax, cpred):
        sns.heatmap(confusion_matrix(yte_c, cpred[n]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=["On-time", "Delayed"], yticklabels=["On-time", "Delayed"])
        a.set_title(n); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    _save("confusion.png")
    plt.figure(figsize=(5.5, 4.5))
    for n, pr in cprob.items(): f, t, _ = roc_curve(yte_c, pr); plt.plot(f, t, label=f"{n} (AUC={roc_auc_score(yte_c, pr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.8); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(loc="lower right", fontsize=8); _save("roc.png")
    out["top_features"] = list(pimp.sort_values(ascending=False).head(6).index)
