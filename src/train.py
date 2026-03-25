"""
Fraud Detection — Training Pipeline
====================================
Trains an XGBoost binary classifier on financial transaction data,
logs all artifacts to MLflow, and conditionally promotes the model
to champion only if it outperforms the current champion on AUPRC.

Promotion logic:
    1. Train and evaluate new model
    2. Fetch current champion's AUPRC from MLflow (if one exists)
    3. Compare: new AUPRC > champion AUPRC + MIN_IMPROVEMENT_DELTA?
    4. YES → promote new model to champion
       NO  → keep current champion, log new model as "challenger"

Usage:
    python src/train.py                              # conditional promotion
    FORCE_PROMOTE=1 python src/train.py              # skip comparison, always promote
    TARGET_PRECISION=0.95 python src/train.py        # override threshold
    MIN_DELTA=0.005 python src/train.py              # require 0.5% improvement
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

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

# Promotion controls
PROMOTION_METRIC: str = os.getenv("PROMOTION_METRIC", "auprc")
MIN_IMPROVEMENT_DELTA: float = float(os.getenv("MIN_DELTA", "0.0"))
FORCE_PROMOTE: bool = os.getenv("FORCE_PROMOTE", "0") == "1"

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
# Data structures
# ──────────────────────────────────────────────
@dataclass
class EvalResult:
    """Holds all evaluation metrics for a trained model."""
    auprc: float
    auroc: float
    precision: float
    recall: float
    threshold: float
    confusion_matrix: np.ndarray


# ──────────────────────────────────────────────
# Data loading
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

    log.info("Loaded %d rows  │  fraud rate: %.2f%%",
             len(df), df[TARGET_COLUMN].mean() * 100)
    return df


# ──────────────────────────────────────────────
# Model training helpers
# ──────────────────────────────────────────────
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


def pick_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, target_prec: float,
) -> float:
    """
    Select the decision threshold that maximises recall
    while keeping precision >= target_prec.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_thr, best_recall = 0.5, -1.0

    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_prec and r > best_recall:
            best_recall = r
            best_thr = float(t)

    return best_thr


def evaluate(
    booster: xgb.Booster, dtest: xgb.DMatrix, y_test: np.ndarray,
) -> EvalResult:
    """Run all evaluation metrics on the test set."""
    y_prob = booster.predict(dtest)
    auprc = float(average_precision_score(y_test, y_prob))
    auroc = float(roc_auc_score(y_test, y_prob))

    thr = pick_threshold(y_test, y_prob, TARGET_PRECISION)
    y_hat = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_test, y_hat)
    tn, fp, fn, tp = cm.ravel()

    return EvalResult(
        auprc=auprc,
        auroc=auroc,
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, tp + fn),
        threshold=thr,
        confusion_matrix=cm,
    )


# ──────────────────────────────────────────────
# Champion comparison
# ──────────────────────────────────────────────
def get_champion_metric(client: MlflowClient) -> float | None:
    """
    Fetch the promotion metric from the current champion model.
    Returns None if no champion exists yet.
    """
    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run = client.get_run(champion_version.run_id)
        value = champion_run.data.metrics.get(PROMOTION_METRIC)

        if value is not None:
            log.info(
                "Current champion: v%s  │  %s=%.6f  │  run=%s",
                champion_version.version, PROMOTION_METRIC, value,
                champion_version.run_id,
            )
        return value

    except Exception as e:
        log.info("No existing champion found (%s) — first model will be promoted", e)
        return None


def should_promote(new_metric: float, champion_metric: float | None) -> bool:
    """
    Decide whether the new model deserves the champion alias.

    Rules:
        1. FORCE_PROMOTE=1          → always promote
        2. No current champion      → promote (first model)
        3. new > champion + delta   → promote (measurable improvement)
        4. Otherwise                → keep current champion
    """
    if FORCE_PROMOTE:
        log.info("FORCE_PROMOTE=1 — skipping comparison")
        return True

    if champion_metric is None:
        log.info("No champion exists — promoting first model")
        return True

    improvement = new_metric - champion_metric
    required = MIN_IMPROVEMENT_DELTA

    if improvement > required:
        log.info(
            "New model wins: %.6f > %.6f + %.4f (improvement=+%.6f)",
            new_metric, champion_metric, required, improvement,
        )
        return True

    log.info(
        "New model did NOT beat champion: %.6f <= %.6f + %.4f (delta=%.6f)",
        new_metric, champion_metric, required, improvement,
    )
    return False


def promote_model(
    client: MlflowClient, version: str, promoted: bool,
) -> None:
    """
    Set the appropriate alias based on the promotion decision.
    - Promoted     → alias "champion" (replaces previous)
    - Not promoted → alias "challenger" (available for inspection)
    """
    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "champion", version)
        log.info("Promoted v%s to 'champion'", version)
    else:
        client.set_registered_model_alias(MODEL_NAME, "challenger", version)
        log.info("Tagged v%s as 'challenger' (champion unchanged)", version)


# ──────────────────────────────────────────────
# Artifacts
# ──────────────────────────────────────────────
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


def log_artifacts(eval_result: EvalResult, n_pos: int, n_neg: int) -> None:
    """Save and log all training artifacts to MLflow."""
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Confusion matrix plot
    cm_path = str(artifacts_dir / "confusion_matrix.png")
    save_confusion_matrix(eval_result.confusion_matrix, cm_path)
    mlflow.log_artifact(cm_path)

    # Metadata JSON
    meta = {
        "features": FEATURE_COLUMNS,
        "class_balance": {"positive": n_pos, "negative": n_neg},
        "threshold": eval_result.threshold,
        "promotion_metric": PROMOTION_METRIC,
        "min_improvement_delta": MIN_IMPROVEMENT_DELTA,
    }
    meta_path = artifacts_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    mlflow.log_artifact(str(meta_path))


# ──────────────────────────────────────────────
# Training Pipeline
# ──────────────────────────────────────────────
def train() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    # ── 1. Load data ──────────────────────────
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

    log.info("Training XGBoost  │  %d pos / %d neg  │  spw=%.2f",
             n_pos, n_neg, scale_pos_weight)

    # ── 2. Fetch current champion baseline ────
    champion_metric = get_champion_metric(client)

    # ── 3. Train ──────────────────────────────
    with mlflow.start_run() as run:
        mlflow.log_params({
            "data_path": DATA_PATH,
            "rows_used": len(df),
            "target_precision": TARGET_PRECISION,
            "sample_n": SAMPLE_N,
            "promotion_metric": PROMOTION_METRIC,
            "min_improvement_delta": MIN_IMPROVEMENT_DELTA,
            "force_promote": FORCE_PROMOTE,
            **params,
        })

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_COLUMNS)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_COLUMNS)
        num_rounds = int(os.getenv("NUM_BOOST_ROUND", "300"))

        booster = xgb.train(params, dtrain, num_boost_round=num_rounds)

        # ── 4. Evaluate ──────────────────────────
        result = evaluate(booster, dtest, y_test)

        mlflow.log_metrics({
            "auprc": result.auprc,
            "auroc": result.auroc,
            "precision_at_thr": result.precision,
            "recall_at_thr": result.recall,
            "threshold": result.threshold,
        })

        log.info(
            "AUPRC=%.4f  │  AUROC=%.4f  │  P=%.3f  R=%.3f  @thr=%.3f",
            result.auprc, result.auroc, result.precision, result.recall,
            result.threshold,
        )

        # ── 5. Save artifacts ────────────────────
        log_artifacts(result, n_pos, n_neg)
        mlflow.xgboost.log_model(booster, artifact_path="model")

        # ── 6. Register model ────────────────────
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        # ── 7. Conditional promotion ─────────────
        new_metric = getattr(result, PROMOTION_METRIC)
        promoted = should_promote(new_metric, champion_metric)
        promote_model(client, mv.version, promoted)

        # Tag the run with the promotion decision
        mlflow.set_tag("promoted_to_champion", str(promoted))
        mlflow.set_tag("model_version", mv.version)

        if champion_metric is not None:
            mlflow.log_metric("champion_baseline", champion_metric)
            mlflow.log_metric("improvement_over_champion",
                              new_metric - champion_metric)

        log.info(
            "Registered %s v%s  │  alias=%s  │  run_id=%s",
            MODEL_NAME, mv.version,
            "champion" if promoted else "challenger",
            run.info.run_id,
        )


if __name__ == "__main__":
    train()
