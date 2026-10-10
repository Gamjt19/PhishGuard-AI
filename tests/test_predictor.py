"""Tests for predictor inference, probability calibration warnings, and empty/missing model handling."""

from pathlib import Path
import pytest

from src.predictor import PhishGuardPredictor, PredictionResult


def test_prediction_output_structure():
    """Verify PredictionResult dataclass contains all required fields."""
    predictor = PhishGuardPredictor()
    res = predictor.predict(
        subject="Action Required: Review Document",
        body="Please find the design document for next week's sprint."
    )

    assert isinstance(res, PredictionResult)
    assert res.predicted_class in ["Likely phishing", "Likely legitimate", "Review recommended"]
    assert res.risk_level in ["Low", "Medium", "High", "Critical"]
    assert 0.0 <= res.risk_score <= 100.0
    assert isinstance(res.indicators, list)
    assert len(res.recommended_action) > 0
    assert "uncalibrated" in res.probability_calibration_note.lower()


def test_empty_input_handling():
    """Verify empty or whitespace-only inputs do not crash and return safe message."""
    predictor = PhishGuardPredictor()
    res = predictor.predict(subject="", body="   ")

    assert res.predicted_class == "Review recommended"
    assert res.risk_score == 0.0
    assert "Empty input" in res.risk_summary
    assert res.model_probability_phishing is None


def test_missing_model_graceful_degradation(tmp_path):
    """Verify predictor functions with fallback heuristics when model artifact is missing."""
    fake_model_path = tmp_path / "non_existent_model.joblib"
    fake_meta_path = tmp_path / "non_existent_meta.json"

    predictor = PhishGuardPredictor(
        model_path=str(fake_model_path),
        metadata_path=str(fake_meta_path)
    )

    assert predictor.pipeline is None

    # Predict with heuristics only
    res = predictor.predict(
        subject="URGENT: Bank account suspended!",
        body="Verify your password immediately at http://192.168.1.1/login within 24 hours."
    )

    assert res.model_available is False
    assert res.model_probability_phishing is None
    # Should still catch the threat via heuristics
    assert res.predicted_class == "Likely phishing"
    assert len(res.indicators) > 0
    assert 0.0 <= res.risk_score <= 100.0


def test_composite_risk_score_edge_cases():
    """Verify composite risk score remains strictly bounded in [0.0, 100.0] across edge cases."""
    predictor = PhishGuardPredictor()

    # 1. Benign text with no indicators
    benign_res = predictor.predict(
        subject="Team lunch tomorrow",
        body="Let's grab lunch at 12:30 at the local cafeteria."
    )
    assert 0.0 <= benign_res.risk_score <= 100.0
    assert benign_res.risk_level in ["Low", "Medium"]
    assert benign_res.decision_breakdown["heuristic_indicator_count"] == 0

    # 2. Extreme lure with multiple high and medium severity indicators
    extreme_lure = (
        "URGENT: Your account is suspended immediately within 24 hours! "
        "Enter your password and verify your credentials at http://192.168.1.1/login. "
        "Wire transfer $5000 to avoid termination: http://login.microsoft365.verify-portal.ru/auth"
    )
    extreme_res = predictor.predict(
        subject="URGENT ACTION REQUIRED",
        body=extreme_lure
    )
    assert 0.0 <= extreme_res.risk_score <= 100.0
    assert extreme_res.risk_level == "Critical"
    assert extreme_res.decision_breakdown["model_score_points"] <= 50.0
    assert extreme_res.decision_breakdown["heuristic_score_points"] <= 50.0
    assert extreme_res.decision_breakdown["composite_risk_score"] <= 100.0
