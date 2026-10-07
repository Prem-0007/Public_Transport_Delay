"""Runs the complete pipeline: load -> clean -> dataset report -> EDA -> tune/train/compare models -> save artefacts."""
import os, json
import pandas as pd
from src.config import RAW_CSV, CLEAN_CSV, MODEL_DIR
from src import preprocessing, eda, modeling, data_report

if not os.path.exists(RAW_CSV):
    import generate_dataset  # noqa
raw = pd.read_csv(RAW_CSV)
print(f"Loaded raw data: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
df, report = preprocessing.clean(raw)
df.to_csv(CLEAN_CSV, index=False)
print(f"Cleaned data   : {df.shape[0]:,} rows x {df.shape[1]} columns")
for s in report["steps"]: print(" -", s)
ds = data_report.build(raw, df, report)
eda.make_all(raw, df); eda.extra(df)
results = modeling.run(df)
results.update({"preprocessing": report, "summary": eda.summary_stats(df), "dataset": ds})
with open(os.path.join(MODEL_DIR, "results.json"), "w") as f: json.dump(results, f, indent=2, default=str)
print("\nRegression:");    print(pd.DataFrame(results["regression"]).to_string(index=False))
print("\nClassification:"); print(pd.DataFrame(results["classification"]).to_string(index=False))
print("\nTuned parameters:", results["tuning"]); print("\nDone. Now run:  python app.py")
