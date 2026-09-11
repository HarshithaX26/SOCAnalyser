"""ML backend for the SOCAnalyser dashboard.

Trains a real XGBoost binary classifier on the cleaned CIC-IDS2017 dataset
(1 = Attack, 0 = Normal Traffic), computes evaluation metrics from actual
predictions, and exposes SHAP TreeExplainer attributions for real rows the
model flagged as attacks.

Endpoints (all under /api):
  GET  /api/health           -> service + model status
  GET  /api/metrics          -> accuracy / FPR / precision / recall / F1 from real predictions
  GET  /api/attack-samples   -> real test rows predicted as attacks (selected+explainable)
  GET  /api/explain/<id>     -> ranked SHAP contributions for a specific attack-predicted row
  POST /api/predict          -> binary prediction + probability for a supplied feature row

Run:  python ml_backend.py [--host 0.0.0.0] [--port 8000] [--rebuild]

Deployment:
  - The cleaned dataset is auto-downloaded on first training run if it is
    missing (URL comes from the DATASET_URL environment variable). It is never
    downloaded again once it exists locally.
  - Uses the Render PORT environment variable when --port is not supplied.
  - All trained-model artifacts (ml_backend_cache.pkl) are cached on the local
    filesystem and reused for the lifetime of the service.
"""

import argparse
import json
import os
import pickle
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from flask import Flask, request, jsonify, send_file
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cicids2017_cleaned.csv")
HTML_PATH = os.path.join(BASE_DIR, "code.html")
CACHE_PATH = os.path.join(BASE_DIR, "ml_backend_cache.pkl")

LABEL_COL = "Attack Type"
TARGET_COL = "Label"  # 1 = Attack, 0 = Normal Traffic

# Stratified sample caps per class (keeps training fast while covering all classes)
CLASS_CAPS = {
    "Normal Traffic": 120000,
    "DoS": 40000,
    "DDoS": 40000,
    "Port Scanning": 30000,
    "Brute Force": 10000,
    "Web Attacks": 10000,
    "Bots": 10000,
}

RANDOM_STATE = 42
TEST_SIZE = 0.20
PROB_THRESHOLD = 0.5
MAX_SAMPLES = 12

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 7,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# Dataset availability (download-from-DATASET_URL when missing)
# ---------------------------------------------------------------------------
def _download_url(url, dest_path, chunk_size=256 * 1024):
    """Stream `url` to `dest_path` with progress printed to the logs."""
    with urllib.request.urlopen(url) as resp, open(dest_path, "wb") as out:
        total = resp.headers.get("Content-Length")
        downloaded = 0
        chunk = resp.read(chunk_size)
        while chunk:
            out.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"[data]   {downloaded / 1e6:,.1f} MB / {int(total) / 1e6:,.1f} MB",
                      end="\r", flush=True)
            else:
                print(f"[data]   {downloaded / 1e6:,.1f} MB downloaded", end="\r", flush=True)
            chunk = resp.read(chunk_size)
    print()


def ensure_dataset(csv_path=CSV_PATH):
    """Return a path to the cleaned CIC-IDS2017 CSV, downloading it if absent.

    - If the CSV already exists it is used as-is (never re-downloaded).
    - Otherwise the DATASET_URL environment variable is required.
    - Downloads to a temporary ".part" file and only renames it to the final
      name after the transfer completes successfully.
    - Raises a clear error (never a silent fake fallback) on any failure.
    """
    if os.path.exists(csv_path):
        size_mb = os.path.getsize(csv_path) / 1e6
        print(f"[data] Using existing dataset: {csv_path} ({size_mb:,.1f} MB)")
        return csv_path

    dataset_url = os.environ.get("DATASET_URL", "").strip()
    if not dataset_url:
        raise RuntimeError(
            f"Cleaned dataset not found at {csv_path} and the DATASET_URL "
            "environment variable is not set. Configure DATASET_URL in Render's "
            "Environment Variables (or place the CSV next to ml_backend.py)."
        )

    print(f"[data] Dataset '{os.path.basename(csv_path)}' not found locally.")
    print(f"[data] Downloading from DATASET_URL: {dataset_url}")
    tmp_path = csv_path + ".part"
    try:
        _download_url(dataset_url, tmp_path)
        os.replace(tmp_path, csv_path)
        size_mb = os.path.getsize(csv_path) / 1e6
        print(f"[data] Download complete -> {csv_path} ({size_mb:,.1f} MB)")
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(
            f"Dataset download from DATASET_URL failed: {exc}"
        ) from exc
    return csv_path


def load_sampled_dataset(csv_path):
    """Read the cleaned CSV in chunks and return a stratified sample covering all classes."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Expected cleaned CIC-IDS2017 CSV."
        )
    leftover = dict(CLASS_CAPS)
    collected = {}  # label -> list of DataFrames
    chunks = 0
    for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
        chunks += 1
        for label, group in chunk.groupby(LABEL_COL, sort=False):
            cap = leftover.get(label, 0)
            if cap <= 0:
                continue
            take = min(len(group), cap)
            if take < len(group):
                group = group.sample(n=take, random_state=RANDOM_STATE)
            collected.setdefault(label, []).append(group)
            leftover[label] -= take
    if not collected:
        raise ValueError("Dataset contains no usable rows.")
    frames = [g for parts in collected.values() for g in parts]
    df = pd.concat(frames, ignore_index=True)
    return df, chunks


def prepare_xy(df):
    """Build feature matrix + binary target. Ordering is fixed by CSV column order."""
    y = (df[LABEL_COL] != "Normal Traffic").astype(int)
    X = df.drop(columns=[LABEL_COL])
    X = X.apply(pd.to_numeric, errors="coerce").astype("float64")
    if X.isna().any().any():
        X = X.fillna(0.0)
    if np.isinf(X.values).any():
        X = X.replace([np.inf, -np.inf], 0.0)
    return X, y


def compute_metrics(y_test, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "fpr": float(fpr),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "threshold": PROB_THRESHOLD,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_test": int(len(y_test)),
    }


class MLEngine:
    def __init__(self, csv_path, rebuild=False):
        self.csv_path = csv_path
        self.model = None
        self.explainer = None
        self.features = None
        self.metrics = None
        self.attack_samples = []
        self.attack_count = 0
        self.n_train = 0
        self.n_test = 0
        self.cache_used = False
        self.pipeline = None

        if not rebuild and os.path.exists(CACHE_PATH):
            try:
                self._load_cache()
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[ml] Cache load failed ({exc}); retraining.")

        self._train()

    def _load_cache(self):
        with open(CACHE_PATH, "rb") as fh:
            data = pickle.load(fh)
        for key in ("features", "model", "metrics", "attack_samples",
                    "attack_count", "n_train", "n_test"):
            if key not in data:
                raise ValueError(f"cache missing {key}")
        self.features = data["features"]
        self.model = data["model"]
        self.metrics = data["metrics"]
        self.attack_samples = data["attack_samples"]
        self.attack_count = data["attack_count"]
        self.n_train = data["n_train"]
        self.n_test = data["n_test"]
        self.explainer = shap.TreeExplainer(self.model)
        self.cache_used = True
        print(f"[ml] Loaded trained model + SHAP explainer from cache ({self.n_train} train / {self.n_test} test rows).")

    def _save_cache(self):
        data = {
            "features": self.features,
            "model": self.model,
            "metrics": self.metrics,
            "attack_samples": self.attack_samples,
            "attack_count": self.attack_count,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }
        with open(CACHE_PATH, "wb") as fh:
            pickle.dump(data, fh)
        print(f"[ml] Cached trained model -> {CACHE_PATH}")

    def _train(self):
        t0 = time.time()
        print("[ml] Loading cleaned CIC-IDS2017 dataset (stratified sample)...")
        ensure_dataset(self.csv_path)
        df, n_chunks = load_sampled_dataset(self.csv_path)
        print(f"[ml] Sample size: {len(df)} rows (from {n_chunks} chunks).")

        X, y = prepare_xy(df)
        self.features = list(X.columns)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        self.n_train = len(X_tr)
        self.n_test = len(X_te)
        print(f"[ml] Train {self.n_train} / Test {self.n_test} rows, {len(self.features)} features.")

        print("[ml] Training XGBoost binary classifier (Attack vs Normal)...")
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr)
        self.model = model

        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= PROB_THRESHOLD).astype(int)
        self.metrics = compute_metrics(y_te.values, pred, prob)
        self.metrics["m_train_samples"] = self.n_train
        self.metrics["m_features"] = len(self.features)
        self.metrics["m_model"] = "XGBoost (binary:logistic)"

        # Collect sampled test rows predicted as attacks, spanning the confidence range.
        attack_idx = np.where(pred == 1)[0]
        self.attack_count = int(attack_idx.size)
        attack_probs = prob[attack_idx]
        pct_grid = np.linspace(0, 100, MAX_SAMPLES + 2)[1:-1]
        chosen_pos = {int(np.argmin(np.abs(attack_probs - np.percentile(attack_probs, p)))) for p in pct_grid}
        if 0 in chosen_pos:
            chosen_pos.discard(0)
        chosen_pos |= {0, len(attack_idx) - 1}
        chosen = attack_idx[sorted(chosen_pos)]
        chosen = chosen[:MAX_SAMPLES]
        rows = X_te.iloc[chosen]
        probs = prob[chosen]
        self.attack_samples = []
        for i, (row_i, p) in enumerate(zip(rows.itertuples(index=True), probs)):
            values = {
                f: (None if pd.isna(v) else float(v))
                for f, v in zip(self.features, row_i[1:])
            }
            self.attack_samples.append({
                "id": f"atk-{i + 1}",
                "test_index": int(chosen[i]),
                "prob_attack": float(p),
                "label": "Attack",
                "values": values,
            })

        self.explainer = shap.TreeExplainer(model)
        self._save_cache()
        print(f"[ml] Done in {time.time() - t0:.1f}s. Predicted attacks in test set: {self.attack_count}")

    @property
    def ready(self):
        return self.model is not None

    def get_row_dataframe(self, sample):
        values = sample["values"]
        row = pd.DataFrame([{f: values.get(f, 0.0) for f in self.features}])
        return row.astype("float64")

    def shap_for_sample(self, sample):
        row_df = self.get_row_dataframe(sample)
        raw = self.explainer.shap_values(row_df)
        if isinstance(raw, list):
            raw = raw[-1]
        shaps = np.asarray(raw).reshape(-1)
        base = self.explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base)[-1])
        order = np.argsort(-np.abs(shaps))
        top = []
        for i in order[:8]:
            s = float(shaps[i])
            top.append({
                "feature": self.features[i],
                "shap": round(s, 4),
                "direction": "attack" if s >= 0 else "normal",
                "feature_value": round(float(row_df[self.features[i]].iloc[0]), 6),
            })
        return {
            "id": sample["id"],
            "expected_value": round(float(base), 4),
            "prob_attack": sample["prob_attack"],
            "prediction": sample["label"],
            "features_ranked": top,
        }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=BASE_DIR)
ENGINE = None
START_TIME = time.time()


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    if request.method == "OPTIONS":
        resp.status_code = 204
    return resp


def json_error(msg, code=500):
    return jsonify({"ok": False, "error": msg}), code


@app.route("/", methods=["GET"])
def index():
    return send_file(HTML_PATH)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "SOCAnalyser ML Backend",
        "model_ready": bool(ENGINE and ENGINE.ready),
        "cache_used": bool(ENGINE and ENGINE.cache_used),
        "dataset": os.path.basename(CSV_PATH),
        "uptime_s": round(time.time() - START_TIME, 1),
    })


@app.route("/api/metrics", methods=["GET"])
def metrics():
    if not ENGINE or not ENGINE.ready:
        return json_error("Model not trained yet. Check server logs.", 503)
    body = dict(ENGINE.metrics)
    body.update({
        "ok": True,
        "model": "XGBoost",
        "features": ENGINE.features,
        "attack_count": ENGINE.attack_count,
    })
    return jsonify(body)


@app.route("/api/attack-samples", methods=["GET"])
def attack_samples():
    if not ENGINE or not ENGINE.ready:
        return json_error("Model not trained yet. Check server logs.", 503)
    return jsonify({
        "ok": True,
        "total_attacks_predicted": ENGINE.attack_count,
        "samples": [
            {"id": s["id"], "prob_attack": round(s["prob_attack"], 4)}
            for s in ENGINE.attack_samples
        ],
    })


@app.route("/api/explain/<sample_id>", methods=["GET"])
def explain(sample_id):
    if not ENGINE or not ENGINE.ready:
        return json_error("Model not trained yet. Check server logs.", 503)
    sample = next((s for s in ENGINE.attack_samples if s["id"] == sample_id), None)
    if sample is None:
        return json_error(f"Unknown sample_id: {sample_id}", 404)
    try:
        return jsonify({"ok": True, "explanation": ENGINE.shap_for_sample(sample)})
    except Exception as exc:  # noqa: BLE001
        return json_error(f"SHAP calculation failed: {exc}", 500)


@app.route("/api/predict", methods=["POST"])
def predict():
    if not ENGINE or not ENGINE.ready:
        return json_error("Model not trained yet. Check server logs.", 503)
    payload = request.get_json(silent=True) or {}
    features = payload.get("features")
    if not isinstance(features, dict) or not features:
        return json_error("Missing 'features' dict in request body.", 400)
    try:
        row = pd.DataFrame([{f: features.get(f, 0.0) for f in ENGINE.features}])
        row = row.astype("float64").fillna(0.0)
        prob = float(ENGINE.model.predict_proba(row)[0, 1])
        pred = int(prob >= PROB_THRESHOLD)
        return jsonify({
            "ok": True,
            "prediction": pred,
            "label": "Attack" if pred == 1 else "Normal Traffic",
            "probability": prob,
            "threshold": PROB_THRESHOLD,
        })
    except Exception as exc:  # noqa: BLE001
        return json_error(f"Prediction failed: {exc}", 500)


def main():
    parser = argparse.ArgumentParser(description="SOCAnalyser ML backend (XGBoost + SHAP)")
    parser.add_argument("--host",
                        default=os.environ.get("HOST", "0.0.0.0"),
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8000")),
                        help="Bind port (default: Render $PORT or 8000 locally)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force retraining even if a model cache exists")
    args = parser.parse_args()

    global ENGINE
    try:
        ENGINE = MLEngine(CSV_PATH, rebuild=args.rebuild)
    except Exception as exc:  # noqa: BLE001
        print("[fatal] Could not initialise the ML engine:", exc, file=sys.stderr)
        print("[fatal] The service will not start with fabricated model outputs.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("SOCAnalyser ML Backend")
    if ENGINE.ready:
        print(f"  Metrics (from real test predictions): {ENGINE.metrics}")
    print(f"  Attack-predicted test rows available: {ENGINE.attack_count}")
    print(f"  Serving dashboard + API at http://{args.host}:{args.port}/")
    print("=" * 60)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()