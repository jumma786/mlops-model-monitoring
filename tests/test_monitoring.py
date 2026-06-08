"""
Test suite for mlops-model-monitoring.
Run: pytest tests/ -v --cov=src
"""

import pytest
import numpy as np
import pandas as pd
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.detector import (
    compute_psi, compute_ks, compute_chi2,
    classify_drift, detect_feature_drift, detect_prediction_drift
)
from src.data.simulator import generate_window


@pytest.fixture
def reference():
    return generate_window(n=500, window=1, random_state=42)


@pytest.fixture
def production_same():
    return generate_window(n=500, window=1, random_state=99)


@pytest.fixture
def production_drifted():
    return generate_window(n=500, window=3, random_state=42)


class TestPSI:
    def test_psi_same_distribution(self, reference):
        s = reference["euribor3m"]
        psi = compute_psi(s, s)
        assert psi < 0.10

    def test_psi_different_distribution(self, reference, production_drifted):
        psi = compute_psi(reference["euribor3m"], production_drifted["euribor3m"])
        assert psi > 0.10

    def test_psi_non_negative(self, reference, production_same):
        psi = compute_psi(reference["age"], production_same["age"])
        assert psi >= 0.0

    def test_psi_returns_float(self, reference, production_same):
        psi = compute_psi(reference["age"], production_same["age"])
        assert isinstance(psi, float)


class TestKS:
    def test_ks_same_distribution(self, reference):
        s = reference["euribor3m"]
        result = compute_ks(s, s)
        assert result["p_value"] > 0.05

    def test_ks_keys(self, reference, production_same):
        result = compute_ks(reference["age"], production_same["age"])
        assert "ks_statistic" in result
        assert "p_value" in result

    def test_ks_drifted(self, reference, production_drifted):
        result = compute_ks(reference["euribor3m"], production_drifted["euribor3m"])
        assert result["ks_statistic"] > 0


class TestChi2:
    def test_chi2_same_distribution(self, reference):
        result = compute_chi2(reference["contact"], reference["contact"])
        assert result["p_value"] > 0.05

    def test_chi2_drifted(self, reference, production_drifted):
        result = compute_chi2(reference["contact"], production_drifted["contact"])
        assert result["p_value"] < 0.05

    def test_chi2_keys(self, reference, production_same):
        result = compute_chi2(reference["job"], production_same["job"])
        assert "chi2" in result
        assert "p_value" in result


class TestClassifyDrift:
    def test_no_drift(self):
        assert classify_drift(0.05) == "none"

    def test_moderate_drift(self):
        assert classify_drift(0.15) == "moderate"

    def test_significant_drift(self):
        assert classify_drift(0.30) == "significant"


class TestDriftDetection:
    def test_feature_drift_returns_dict(self, reference, production_same):
        result = detect_feature_drift(reference, production_same)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_significant_drift_detected(self, reference, production_drifted):
        result = detect_feature_drift(reference, production_drifted)
        significant = [f for f, r in result.items() if r.get("drift") == "significant"]
        assert len(significant) > 0

    def test_prediction_drift_keys(self, reference, production_drifted):
        ref_preds  = pd.Series(np.random.uniform(0, 1, len(reference)))
        prod_preds = pd.Series(np.random.uniform(0.3, 1, len(production_drifted)))
        result = detect_prediction_drift(ref_preds, prod_preds)
        assert "psi" in result
        assert "drift" in result
        assert "ref_positive_rate" in result
        assert "prod_positive_rate" in result
