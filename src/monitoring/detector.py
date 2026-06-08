"""
Drift Detection Engine
=======================
Detects data drift and prediction drift between reference and production windows.

Methods:
  - PSI (Population Stability Index) — numerical features
  - Chi-squared test — categorical features
  - KS test (Kolmogorov-Smirnov) — distribution comparison
  - Prediction drift — output distribution shift

PSI interpretation:
  < 0.10 = No drift
  0.10-0.25 = Moderate drift — monitor
  > 0.25 = Significant drift — retrain
"""

import numpy as np
import pandas as pd
from scipy import stats
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PSI thresholds
PSI_LOW      = 0.10
PSI_MODERATE = 0.25

NUM_FEATURES = ["age","campaign","pdays","previous","emp.var.rate",
                "cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]

CAT_FEATURES = ["job","marital","education","default","housing","loan",
                "contact","month","day_of_week","poutcome"]


def compute_psi(reference: pd.Series, production: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index — detects numerical feature drift.
    PSI = sum((P_prod - P_ref) * ln(P_prod / P_ref))
    """
    ref_clean  = reference.dropna()
    prod_clean = production.dropna()

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts  = np.histogram(ref_clean,  bins=breakpoints)[0]
    prod_counts = np.histogram(prod_clean, bins=breakpoints)[0]

    ref_pct  = (ref_counts  + 1e-6) / len(ref_clean)
    prod_pct = (prod_counts + 1e-6) / len(prod_clean)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def compute_ks(reference: pd.Series, production: pd.Series) -> dict:
    """Kolmogorov-Smirnov test for distribution comparison."""
    stat, pvalue = stats.ks_2samp(reference.dropna(), production.dropna())
    return {"ks_statistic": round(float(stat), 4), "p_value": round(float(pvalue), 4)}


def compute_chi2(reference: pd.Series, production: pd.Series) -> dict:
    """Chi-squared test for categorical feature drift."""
    all_cats = set(reference.unique()) | set(production.unique())
    ref_counts  = reference.value_counts().reindex(all_cats, fill_value=0)
    prod_counts = production.value_counts().reindex(all_cats, fill_value=0)

    if ref_counts.sum() == 0 or prod_counts.sum() == 0:
        return {"chi2": 0.0, "p_value": 1.0}

    # Normalize to same scale
    prod_expected = prod_counts.sum() * ref_counts / ref_counts.sum()
    mask = prod_expected > 0
    stat, pvalue = stats.chisquare(
        prod_counts[mask],
        f_exp=prod_expected[mask]
    )
    return {"chi2": round(float(stat), 4), "p_value": round(float(pvalue), 4)}


def classify_drift(psi: float) -> str:
    """Classify drift severity based on PSI."""
    if psi < PSI_LOW:
        return "none"
    elif psi < PSI_MODERATE:
        return "moderate"
    else:
        return "significant"


def detect_feature_drift(reference: pd.DataFrame, production: pd.DataFrame) -> dict:
    """Detect drift for all features."""
    results = {}

    # Numerical features — PSI + KS
    for col in NUM_FEATURES:
        if col not in reference.columns or col not in production.columns:
            continue
        psi  = compute_psi(reference[col], production[col])
        ks   = compute_ks(reference[col], production[col])
        results[col] = {
            "type":      "numerical",
            "psi":       psi,
            "drift":     classify_drift(psi),
            "ks_stat":   ks["ks_statistic"],
            "p_value":   ks["p_value"],
            "ref_mean":  round(float(reference[col].mean()), 4),
            "prod_mean": round(float(production[col].mean()), 4),
        }

    # Categorical features — Chi-squared
    for col in CAT_FEATURES:
        if col not in reference.columns or col not in production.columns:
            continue
        chi2 = compute_chi2(reference[col], production[col])
        results[col] = {
            "type":      "categorical",
            "chi2":      chi2["chi2"],
            "p_value":   chi2["p_value"],
            "drift":     "significant" if chi2["p_value"] < 0.05 else "none",
        }

    return results


def detect_prediction_drift(ref_preds: pd.Series, prod_preds: pd.Series) -> dict:
    """Detect drift in model predictions."""
    psi = compute_psi(ref_preds, prod_preds)
    ks  = compute_ks(ref_preds, prod_preds)
    return {
        "psi":           psi,
        "drift":         classify_drift(psi),
        "ks_stat":       ks["ks_statistic"],
        "p_value":       ks["p_value"],
        "ref_positive_rate":  round(float((ref_preds >= 0.5).mean()), 4),
        "prod_positive_rate": round(float((prod_preds >= 0.5).mean()), 4),
    }


def run_monitoring(reference: pd.DataFrame, production: pd.DataFrame,
                   ref_preds: pd.Series = None, prod_preds: pd.Series = None,
                   window_label: str = "production") -> dict:
    """Full monitoring run — feature drift + prediction drift."""
    logger.info(f"Running monitoring: reference={len(reference):,} | production={len(production):,}")

    feature_drift = detect_feature_drift(reference, production)

    # Count drift severity
    drifted_features = [f for f, r in feature_drift.items() if r.get("drift") in ["moderate","significant"]]
    significant_features = [f for f, r in feature_drift.items() if r.get("drift") == "significant"]

    report = {
        "window_label":          window_label,
        "n_reference":           len(reference),
        "n_production":          len(production),
        "n_features_monitored":  len(feature_drift),
        "n_features_drifted":    len(drifted_features),
        "n_features_significant": len(significant_features),
        "drifted_features":      drifted_features,
        "significant_features":  significant_features,
        "alert":                 len(significant_features) > 0,
        "feature_drift":         feature_drift,
    }

    if ref_preds is not None and prod_preds is not None:
        report["prediction_drift"] = detect_prediction_drift(ref_preds, prod_preds)

    # Summary
    print(f"\n{'='*55}")
    print(f"DRIFT REPORT — {window_label}")
    print(f"{'='*55}")
    print(f"  Features monitored:    {len(feature_drift)}")
    print(f"  Features drifted:      {len(drifted_features)}")
    print(f"  Significant drift:     {len(significant_features)}")
    print(f"  ALERT:                 {'YES — RETRAIN' if report['alert'] else 'NO'}")
    if significant_features:
        print(f"  Drifted features:      {significant_features}")
    print(f"{'='*55}")

    return report


def save_report(report: dict, output_dir: str = "reports"):
    """Save monitoring report as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    label = report["window_label"].replace(" ", "_").lower()
    path = os.path.join(output_dir, f"drift_report_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved: {path}")
    return path
