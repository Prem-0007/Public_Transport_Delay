import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(BASE, "data", "transport_delays_raw.csv")
CLEAN_CSV = os.path.join(BASE, "data", "transport_delays_clean.csv")
MODEL_DIR = os.path.join(BASE, "models")
CHART_DIR = os.path.join(BASE, "static", "charts")
DELAY_THRESHOLD = 5  # minutes: delay > threshold => "Delayed"

ROUTES = {
    "B12": {"mode": "Bus",   "distance_km": 14.5, "num_stops": 32, "headway": 10},
    "B27": {"mode": "Bus",   "distance_km": 9.2,  "num_stops": 24, "headway": 12},
    "B45": {"mode": "Bus",   "distance_km": 18.0, "num_stops": 38, "headway": 15},
    "B61": {"mode": "Bus",   "distance_km": 7.4,  "num_stops": 18, "headway": 8},
    "T3":  {"mode": "Tram",  "distance_km": 11.0, "num_stops": 22, "headway": 8},
    "T7":  {"mode": "Tram",  "distance_km": 8.3,  "num_stops": 17, "headway": 10},
    "T9":  {"mode": "Tram",  "distance_km": 13.6, "num_stops": 26, "headway": 12},
    "M1":  {"mode": "Metro", "distance_km": 21.0, "num_stops": 19, "headway": 4},
    "M2":  {"mode": "Metro", "distance_km": 16.5, "num_stops": 15, "headway": 5},
    "M3":  {"mode": "Metro", "distance_km": 12.2, "num_stops": 12, "headway": 6},
    "R1":  {"mode": "Train", "distance_km": 42.0, "num_stops": 11, "headway": 20},
    "R4":  {"mode": "Train", "distance_km": 35.5, "num_stops": 9,  "headway": 30},
}
NUM_FEATURES = ["hour", "month", "temperature", "passenger_load", "traffic_index", "distance_km",
                "num_stops", "scheduled_headway_min", "incident", "is_holiday", "is_peak", "is_weekend"]
CAT_FEATURES = ["route_id", "transport_mode", "day_of_week", "weather"]
