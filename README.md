# 📊 Model Monitoring & Drift Detection

![CI](https://github.com/jumma786/mlops-model-monitoring/actions/workflows/monitoring.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-champion-orange)
![SciPy](https://img.shields.io/badge/SciPy-stats-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Part of the [MLOps Portfolio Series](https://github.com/jumma786/mlops-portfolio)** — Project 7 of 10  
> Monitors the XGBoost champion model (Project 5) for data drift and prediction drift using PSI, KS-test, and Chi-squared tests across 3 production windows.

---

## 📂 Project Resources

| Resource | Link |
|---|---|
| 🔍 Drift Detector | [src/monitoring/detector.py](src/monitoring/detector.py) |
| 🎬 Data Simulator | [src/data/simulator.py](src/data/simulator.py) |
| 🚀 Monitor Pipeline | [src/monitor.py](src/monitor.py) |
| 🧪 Tests | [tests/test_monitoring.py](tests/test_monitoring.py) |
| 🤖 CI/CD | [.github/workflows/monitoring.yml](.github/workflows/monitoring.yml) |

---

## 🎯 What This Project Does

Monitors the bank marketing XGBoost model for distribution shift:

1. **Simulates production traffic** — 3 windows with progressive drift
2. **Detects numerical drift** — PSI + KS-test on 9 numerical features
3. **Detects categorical drift** — Chi-squared test on 10 categorical features
4. **Alerts** when significant drift detected (PSI > 0.25)
5. **Saves JSON reports** — one per production window

---

## 📊 Drift Detection Methods

| Method | Features | Threshold |
|---|---|---|
| PSI | Numerical | < 0.10 = none, 0.10-0.25 = moderate, > 0.25 = significant |
| KS-test | Numerical | p-value < 0.05 = drift |
| Chi-squared | Categorical | p-value < 0.05 = drift |

---

## 🔬 Production Windows

| Window | Period | Key Change | Alert |
|---|---|---|---|
| Baseline | Jan-Mar | Training distribution | ❌ No |
| Mild Drift | Apr-Jun | Contact method shift | ⚠️ Moderate |
| Strong Drift | Jul-Sep | Euribor drops 5.1→0.6 | 🚨 Yes |

---

## 🚀 Quick Start

```bash
git clone https://github.com/jumma786/mlops-model-monitoring.git
cd mlops-model-monitoring
pip install -r requirements.txt

# Run tests
make test

# Run monitoring on real data
make monitor

# Run monitoring on synthetic data
make monitor-synthetic
```

---

## 🧪 Test Results

**16/16 tests passing** covering:
- PSI computation (same + different distributions)
- KS-test (same + drifted distributions)
- Chi-squared test (categorical features)
- Drift classification (none/moderate/significant)
- Full feature drift detection pipeline
- Prediction drift detection

---

## 🔗 MLOps Portfolio Series

| # | Project | Repo | Status |
|---|---|---|---|
| 1 | Multi-Model Tournament | [mlops-model-tournament](https://github.com/jumma786/mlops-model-tournament) | ✅ |
| 2 | Scheduled Retraining | [mlops-retraining-pipeline](https://github.com/jumma786/mlops-retraining-pipeline) | ✅ |
| 3 | Feature Engineering | [mlops-feature-pipeline](https://github.com/jumma786/mlops-feature-pipeline) | ✅ |
| 4 | Hyperparameter Tuning | [mlops-hyperparameter-tuning](https://github.com/jumma786/mlops-hyperparameter-tuning) | ✅ |
| 5 | Model Serving | [mlops-model-serving](https://github.com/jumma786/mlops-model-serving) | ✅ |
| 6 | Feature Store | [mlops-feature-store](https://github.com/jumma786/mlops-feature-store) | ✅ |
| **7** | **Model Monitoring** | [mlops-model-monitoring](https://github.com/jumma786/mlops-model-monitoring) | ✅ This repo |
| 8 | A/B Testing | [mlops-ab-testing](https://github.com/jumma786/mlops-ab-testing) | ✅ |
| 9 | Airflow Pipeline | [mlops-airflow-pipeline](https://github.com/jumma786/mlops-airflow-pipeline) | ✅ |
| 10 | Kubernetes Platform | [mlops-k8s-platform](https://github.com/jumma786/mlops-k8s-platform) | ✅ |

---

## 👤 Author

**Jumma Mohammad Teli** — Data Analyst & ML Engineer  
📍 Birmingham, UK  
📧 [jummamohammad477@gmail.com](mailto:jummamohammad477@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/jumma-mohammad) | [GitHub](https://github.com/jumma786)

---

*Project 7 of 10 — MLOps Portfolio Series.*
