# Fraud Detection — End-to-End MLOps Pipeline

> Real-time fraud scoring for financial transactions, from model training to Kubernetes deployment.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.14-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-ready-326CE5?logo=kubernetes&logoColor=white)

---

## Overview

Production-grade pipeline that trains an **XGBoost** classifier on 1M+ synthetic financial transactions, tracks experiments with **MLflow**, serves predictions through a **FastAPI** REST API, and deploys the full stack on **Kubernetes**.

The system classifies each transaction as fraudulent or legitimate by analysing 12 behavioral and contextual features, achieving **> 0.90 precision** with a calibrated decision threshold.

### Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Training   │────▶│    MLflow     │◀────│   Serving API    │
│   (XGBoost)  │     │   Registry   │     │    (FastAPI)     │
└──────┬───────┘     └──────────────┘     └────────┬─────────┘
       │                                           │
       │  logs params, metrics, artifacts          │  loads champion model
       │  registers model as "champion"            │  returns P(fraud)
       │                                           │
┌──────▼───────┐                          ┌────────▼─────────┐
│  Train Job   │                          │  Serve Deploy    │
│  (K8s Job)   │                          │  (K8s Deployment)│
└──────────────┘                          └──────────────────┘
```

---

## Features

| Feature | Description |
|---|---|
| `amount` | Transaction value (R$) |
| `hour` | Hour of day (0–23) |
| `dow` | Day of week (0=Mon, 6=Sun) |
| `channel` | 0=in-person, 1=online |
| `international` | International transaction (0/1) |
| `new_merchant` | First-time merchant (0/1) |
| `acct_age_days` | Account age in days |
| `txn_count_24h` | Transactions in last 24h |
| `txn_amount_24h` | Total amount in last 24h (R$) |
| `distance_km` | Distance from last transaction (km) |
| `device_change` | Device changed (0/1) |
| `ip_risk` | IP risk score (0.0–1.0) |

---

## Project Structure

```
fraud-detection/
├── src/
│   ├── train.py                # Training pipeline + MLflow logging
│   └── serve.py                # FastAPI inference service
├── docker/
│   ├── Dockerfile.mlflow       # MLflow tracking server
│   ├── Dockerfile.train        # Training job image
│   └── Dockerfile.serve        # API serving image
├── k8s/
│   ├── 00-namespace.yaml       # Namespace isolation
│   ├── 01-mlflow-pvc.yaml      # Storage for MLflow
│   ├── 02-mlflow-deployment.yaml
│   ├── 03-mlflow-service.yaml
│   ├── 04-train-job.yaml       # One-shot training job
│   ├── 05-serve-deployment.yaml # API with health probes
│   ├── 06-serve-service.yaml   # NodePort exposure
│   ├── 07-data-pvc.yaml        # Dataset storage
│   └── 08-upload-pod.yaml      # Data staging helper
├── tests/
│   └── test_api.py             # API integration tests
├── requirements-train.txt
├── requirements-serve.txt
├── Makefile
└── README.md
```

---

## Quick Start

### Local Development

```bash
# 1. Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlartifacts \
              --host 0.0.0.0 --port 5000

# 2. Train
make train

# 3. Serve
make serve

# 4. Test
make test
```

### Kubernetes Deployment

```bash
# Build and push images
make docker-build docker-push

# Deploy full stack
make k8s-deploy

# Port-forward and test
kubectl port-forward svc/fraud-serving-svc 8000:8000 -n fraud-detection
make test
```

---

## API Reference

### `GET /health`

Returns service status and model readiness.

```json
{
  "status": "ok",
  "model_uri": "models:/fraud-xgb@champion",
  "model_loaded": true
}
```

### `POST /predict`

Accepts a batch of transactions and returns fraud probabilities.

**Request:**

```json
{
  "x": [
    [50.0, 10, 1, 0, 0, 0, 800, 1, 50.0, 0.5, 0, 0.05],
    [3500.0, 3, 6, 1, 1, 1, 30, 8, 4200.0, 2800.0, 1, 0.92]
  ]
}
```

**Response:**

```json
{
  "predictions": [0.031204, 0.984712],
  "count": 2
}
```

Probabilities above the calibrated threshold (default ≈ 0.50) indicate a suspicious transaction.

---

## Model Details

| Aspect | Detail |
|---|---|
| **Algorithm** | XGBoost (`binary:logistic`) |
| **Optimization metric** | AUCPR (Area Under PR Curve) |
| **Class imbalance** | Handled via `scale_pos_weight` |
| **Threshold selection** | Maximizes recall at target precision (default 90%) |
| **Experiment tracking** | All params, metrics, and artifacts logged to MLflow |
| **Model registry** | Promoted to `champion` alias after training |

### Configurable Hyperparameters

All tunable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MAX_DEPTH` | 6 | Maximum tree depth |
| `ETA` | 0.08 | Learning rate |
| `SUBSAMPLE` | 0.9 | Row sampling ratio |
| `COLSAMPLE` | 0.9 | Feature sampling ratio |
| `NUM_BOOST_ROUND` | 300 | Number of boosting rounds |
| `TARGET_PRECISION` | 0.90 | Precision target for threshold calibration |

---

## Infrastructure

### Docker

Three purpose-built images, each under 500MB:

- **`mlflow-server`** — MLflow tracking + model registry (SQLite backend)
- **`fraud-train`** — Stateless training job
- **`fraud-serve`** — FastAPI + Uvicorn serving

### Kubernetes

- **Namespace isolation** — all resources under `fraud-detection`
- **Persistent storage** — PVCs for MLflow state and training data
- **Health probes** — readiness and liveness checks on the serving pod
- **Resource limits** — CPU/memory boundaries for predictable scheduling

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | XGBoost 2.1 |
| Experiment Tracking | MLflow 2.14 |
| API Framework | FastAPI 0.112 + Uvicorn |
| Validation | Pydantic 2.7 |
| Containerization | Docker (Python 3.11 slim) |
| Orchestration | Kubernetes |
| Testing | Integration tests with Requests |

---

## License

MIT
