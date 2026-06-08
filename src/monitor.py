"""
Model Monitoring Pipeline
==========================
Full monitoring pipeline:
1. Generate production traffic (3 windows with drift)
2. Load reference data (training distribution)
3. Run drift detection on each window
4. Generate HTML monitoring report
5. Alert if significant drift detected

Usage:
    python src/monitor.py
    python src/monitor.py --data-path data/bank-additional-full.csv
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.simulator import generate_all_windows
from src.monitoring.detector import run_monitoring, save_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_reference_data(data_path: str = None, n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Load reference data — training distribution."""
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading real reference data: {data_path}")
        df = pd.read_csv(data_path, sep=";")
        if "duration" in df.columns:
            df = df.drop(columns=["duration"])
        df["y"] = (df["y"] == "yes").astype(int)
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)
    else:
        logger.info("Using synthetic reference data")
        from src.data.simulator import generate_window
        return generate_window(n=n_samples, window=1, random_state=random_state)


def preprocess_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal preprocessing — keep raw values for drift detection."""
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")
    return df


def run_full_monitoring(data_path: str = None, n_samples: int = 2000):
    """Run full monitoring pipeline."""
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Step 1 — Reference data
    logger.info("Step 1: Loading reference data...")
    reference = load_reference_data(data_path=data_path, n_samples=n_samples)
    reference = preprocess_for_comparison(reference)

    # Step 2 — Production traffic
    logger.info("Step 2: Generating production traffic...")
    production = generate_all_windows(n_per_window=n_samples // 3)

    # Step 3 — Run monitoring per window
    logger.info("Step 3: Running drift detection...")
    all_reports = []

    for window in [1, 2, 3]:
        prod_window = production[production["window"] == window].copy()
        prod_window = preprocess_for_comparison(prod_window)
        label = {1:"Baseline", 2:"Mild Drift", 3:"Strong Drift"}[window]

        report = run_monitoring(
            reference=reference,
            production=prod_window,
            window_label=label
        )
        save_report(report, output_dir="reports")
        all_reports.append(report)

    # Step 4 — Summary
    print("\n" + "="*60)
    print("MONITORING SUMMARY — ALL WINDOWS")
    print("="*60)
    print(f"{'Window':<20} {'Drifted':>8} {'Alert':>8}")
    print("-"*60)
    for r in all_reports:
        alert = "YES ⚠️" if r["alert"] else "NO ✅"
        print(f"{r['window_label']:<20} {r['n_features_drifted']:>8} {alert:>8}")
    print("="*60)
    print(f"\nReports saved to: reports/")

    return all_reports


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=2000)
    args = p.parse_args()
    run_full_monitoring(data_path=args.data_path, n_samples=args.n_samples)
