"""
Production Data Simulator
==========================
Simulates live prediction traffic from the Bank Marketing model.
Generates 3 time windows with progressive data drift:

Window 1 — Baseline (Jan-Mar): matches training distribution
Window 2 — Mild drift (Apr-Jun): contact method shift
Window 3 — Strong drift (Jul-Sep): economic shift (euribor drops)

Used to test drift detection without needing a live production system.
"""

import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed"
]

CAT_FEATURES = ["job","marital","education","default","housing","loan",
                "contact","month","day_of_week","poutcome"]

NUM_FEATURES = ["age","campaign","pdays","previous","emp.var.rate",
                "cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]


def generate_window(n: int, window: int, random_state: int = 42) -> pd.DataFrame:
    """Generate one time window of production data with drift."""
    np.random.seed(random_state + window)

    # Economic drift
    if window == 1:
        euribor    = np.random.uniform(3.0, 5.1, n).round(3)
        emp_rate   = np.random.choice([-1.8,-1.7,1.1,1.4], n)
        nr_emp     = np.random.choice([5099.1,5176.3,5195.8,5228.1], n)
        contact    = np.random.choice(["cellular","telephone"], n, p=[0.63,0.37])
        positive_rate = 0.11
    elif window == 2:
        euribor    = np.random.uniform(1.5, 4.0, n).round(3)
        emp_rate   = np.random.choice([-2.9,-3.0,-1.8,1.1], n)
        nr_emp     = np.random.choice([5008.7,5017.5,5099.1,5176.3], n)
        contact    = np.random.choice(["cellular","telephone"], n, p=[0.50,0.50])
        positive_rate = 0.13
    else:
        euribor    = np.random.uniform(0.6, 2.0, n).round(3)
        emp_rate   = np.random.choice([-3.4,-3.0,-2.9], n)
        nr_emp     = np.random.choice([4963.6,5008.7,5017.5], n)
        contact    = np.random.choice(["cellular","telephone"], n, p=[0.30,0.70])
        positive_rate = 0.17

    df = pd.DataFrame({
        "age":            np.random.randint(18, 95, n),
        "job":            np.random.choice(["admin.","blue-collar","management","retired","technician","unknown"], n),
        "marital":        np.random.choice(["divorced","married","single"], n),
        "education":      np.random.choice(["basic.4y","high.school","university.degree","unknown"], n),
        "default":        np.random.choice(["no","yes","unknown"], n, p=[0.79,0.01,0.20]),
        "housing":        np.random.choice(["no","yes"], n),
        "loan":           np.random.choice(["no","yes"], n, p=[0.82,0.18]),
        "contact":        contact,
        "month":          np.random.choice(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], n),
        "day_of_week":    np.random.choice(["mon","tue","wed","thu","fri"], n),
        "campaign":       np.random.randint(1, 15, n),
        "pdays":          np.where(np.random.rand(n) < 0.13, np.random.randint(1,30,n), 999),
        "previous":       np.random.randint(0, 7, n),
        "poutcome":       np.random.choice(["failure","nonexistent","success"], n, p=[0.10,0.86,0.04]),
        "emp.var.rate":   emp_rate,
        "cons.price.idx": np.random.uniform(92.2, 94.8, n).round(3),
        "cons.conf.idx":  np.random.uniform(-50.8, -26.9, n).round(1),
        "euribor3m":      euribor,
        "nr.employed":    nr_emp,
        "y_true":         (np.random.rand(n) < positive_rate).astype(int),
        "window":         window,
        "window_label":   {1:"Baseline",2:"Mild Drift",3:"Strong Drift"}[window],
        "timestamp":      pd.date_range(
            start=f"2022-{'01' if window==1 else '04' if window==2 else '07'}-01",
            periods=n, freq="10min"
        ),
    })

    logger.info(f"Window {window}: {n:,} rows | positive rate: {df['y_true'].mean():.1%} | euribor: {euribor.mean():.2f}")
    return df


def generate_all_windows(n_per_window: int = 1000, output_dir: str = "data") -> pd.DataFrame:
    """Generate all 3 production windows."""
    os.makedirs(output_dir, exist_ok=True)
    dfs = []
    for w in [1, 2, 3]:
        df = generate_window(n=n_per_window, window=w)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    path = os.path.join(output_dir, "production_traffic.parquet")
    full.to_parquet(path, index=False)
    logger.info(f"Saved: {path} ({full.shape[0]:,} rows)")
    return full


if __name__ == "__main__":
    df = generate_all_windows(n_per_window=2000)
    print(f"\nProduction traffic: {df.shape}")
    print(df.groupby("window_label")[["euribor3m","y_true"]].mean().round(3))
