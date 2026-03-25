"""
Fraud Detection — Training Pipeline
====================================
Trains an XGBoost binary classifier on financial transaction data,
logs all artifacts to MLflow, and promotes the best model to the
Model Registry with the `champion` alias.

Usage:
    python src/train.py                          # defaults
    TARGET_PRECISION=0.95 python src/train.py    # override threshold
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Configuration (all overridable via env vars)
# ──────────────────────────────────────────────
TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-svc:5000")
EXPERIMENT: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-xgb-prod")
MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "fraud-xgb")
DATA_PATH: str = os.getenv("DATA_PATH", "/data/transactions_1M.csv")
SEED: int = int(os.getenv("SEED", "42"))
TARGET_PRECISION: float = float(os.getenv("TARGET_PRECISION", "0.90"))
SAMPLE_N: int = int(os.getenv("SAMPLE_N", "0"))

FEATURE_COLUMNS: list[str] = [
    "amount", "hour", "dow", "channel", "international", "new_merchant",
    "acct_age_days", "txn_count_24h", "txn_amount_24h", "distance_km",
    "device_change", "ip_risk",
]
TARGET_COLUMN: str = "is_fraud"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the transaction CSV."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Set DATA_PATH or mount the volume correctly."
        )

    df = pd.read_csv(path)

    required = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if 0 < SAMPLE_N < len(df):
        df = df.sample(n=SAMPLE_N, random_state=SEED)
        log.info("Sampled %d rows from %s", SAMPLE_N, path)

    log.info("Loaded %d rows  │  fraud rate: %.2f%%", len(df), df[TARGET_COLUMN].mean() * 100)
    return df


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_prec: float) -> float:
    """
    Select the decision threshold that maximises recall while
    keeping precision >= target_prec.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_thr, best_recall = 0.5, -1.0

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_prec and r > best_recall:
            best_recall = r
            best_thr = float(t)

    return best_thr


def build_xgb_params(scale_pos_weight: float) -> dict:
    """Assemble XGBoost hyperparameters from env vars with sane defaults."""
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": int(os.getenv("MAX_DEPTH", "6")),
        "eta": float(os.getenv("ETA", "0.08")),
        "subsample": float(os.getenv("SUBSAMPLE", "0.9")),
        "colsample_bytree": float(os.getenv("COLSAMPLE", "0.9")),
        "min_child_weight": float(os.getenv("MIN_CHILD_WEIGHT", "1.0")),
        "lambda": float(os.getenv("LAMBDA", "1.0")),
        "alpha": float(os.getenv("ALPHA", "0.0")),
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED,
    }


def save_confusion_matrix(cm: np.ndarray, path: str) -> None:
    """Render and save a confusion-matrix heatmap."""
    labels = ["Legit", "Fraud"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=labels, yticklabels=labels,
        xlabel="Predicted", ylabel="Actual",
        title="Confusion Matrix",
    )
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────
# Training Pipeline
# ──────────────────────────────────────────────
def train() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_dataset(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y,
    )

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(1, n_pos)
    params = build_xgb_params(scale_pos_weight)

    log.info("Training XGBoost  │  %d pos / %d neg  │  spw=%.2f", n_pos, n_neg, scale_pos_weight)

    with mlflow.start_run() as run:
        # ── Log params ──
        mlflow.log_params({
            "data_path": DATA_PATH,
            "rows_used": len(df),
            "target_precision": TARGET_PRECISION,
            "sample_n": SAMPLE_N,
            **params,
        })

        # ── Train ──
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLUMNS)
        num_rounds = int(os.getenv("NUM_BOOST_ROUND", "300"))

        booster = xgb.train(params, dtrain, num_boost_round=num_rounds)

        # ── Evaluate ──
        y_prob = booster.predict(dtest)
        auprc = float(average_precision_score(y_test, y_prob))
        auroc = float(roc_auc_score(y_test, y_prob))

        thr = pick_threshold(y_test, y_prob, TARGET_PRECISION)
        y_hat = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_test, y_hat)
        tn, fp, fn, tp = cm.ravel()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)

        mlflow.log_metrics({
            "auprc": auprc,
            "auroc": auroc,
            "precision_at_thr": float(prec),
            "recall_at_thr": float(rec),
            "threshold": float(thr),
        })

        log.info("AUPRC=%.4f  │  AUROC=%.4f  │  P=%.3f  R=%.3f  @thr=%.3f",
                 auprc, auroc, prec, rec, thr)

        # ── Artifacts ──
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        save_confusion_matrix(cm, str(artifacts_dir / "confusion_matrix.png"))
        mlflow.log_artifact(str(artifacts_dir / "confusion_matrix.png"))

        meta = {
            "features": FEATURE_COLUMNS,
            "rows_used": len(df),
            "class_balance": {"positive": n_pos, "negative": n_neg},
            "threshold": thr,
        }
        meta_path = artifacts_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(meta_path))

        # ── Register model ──
        mlflow.xgboost.log_model(booster, artifact_path="model")

        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)

        log.info("Registered %s v%s as 'champion'  │  run_id=%s",
                 MODEL_NAME, mv.version, run.info.run_id)


if __name__ == "__main__":
    train()
