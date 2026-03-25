"""
API Integration Tests
=====================
Run with the serving endpoint available (port-forward or local):

    python tests/test_api.py
    python tests/test_api.py --url http://my-host:8000
"""

import argparse
import sys

import requests

FEATURE_ORDER = [
    "amount", "hour", "dow", "channel", "international", "new_merchant",
    "acct_age_days", "txn_count_24h", "txn_amount_24h", "distance_km",
    "device_change", "ip_risk",
]


def test_health(base: str) -> bool:
    r = requests.get(f"{base}/health", timeout=5)
    data = r.json()
    assert r.status_code == 200
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    print("  ✓ /health")
    return True


def test_predict_array(base: str) -> bool:
    payload = {
        "x": [
            # Low-risk: small amount, business hours, old account
            [50.0, 10, 1, 0, 0, 0, 800, 1, 50.0, 0.5, 0, 0.05],
            # High-risk: large amount, 3am, international, new device, risky IP
            [3500.0, 3, 6, 1, 1, 1, 30, 8, 4200.0, 2800.0, 1, 0.92],
        ]
    }
    r = requests.post(f"{base}/predict", json=payload, timeout=10)
    data = r.json()
    assert r.status_code == 200
    assert data["count"] == 2
    assert data["predictions"][0] < data["predictions"][1], "Low-risk should score lower"
    print(f"  ✓ /predict (array)  → {data['predictions']}")
    return True


def test_predict_dict(base: str) -> bool:
    payload = {
        "x": [
            {
                "amount": 120.0, "hour": 14, "dow": 2, "channel": 0,
                "international": 0, "new_merchant": 0, "acct_age_days": 500,
                "txn_count_24h": 2, "txn_amount_24h": 200.0, "distance_km": 1.0,
                "device_change": 0, "ip_risk": 0.1,
            }
        ]
    }
    r = requests.post(f"{base}/predict", json=payload, timeout=10)
    data = r.json()
    assert r.status_code == 200
    assert data["count"] == 1
    print(f"  ✓ /predict (dict)   → {data['predictions']}")
    return True


def test_predict_bad_input(base: str) -> bool:
    r = requests.post(f"{base}/predict", json={"x": [[1.0, 2.0]]}, timeout=5)
    assert r.status_code == 422
    print("  ✓ /predict (bad input rejected)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"\n{'─' * 50}")
    print(f"  Fraud Detection API — Integration Tests")
    print(f"  Target: {args.url}")
    print(f"{'─' * 50}\n")

    tests = [test_health, test_predict_array, test_predict_dict, test_predict_bad_input]
    passed = 0

    for t in tests:
        try:
            t(args.url)
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")

    print(f"\n  {passed}/{len(tests)} passed\n")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
