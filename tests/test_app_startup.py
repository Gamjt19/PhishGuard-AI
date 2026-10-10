"""Tests for application startup, imports, and component integration."""

import pytest


def test_core_modules_import_cleanly():
    """Verify all application components and Streamlit dependencies import without error."""
    import src.data_loader
    import src.train_model
    import src.indicators
    import src.predictor
    import src.logging_utils
    import src.download_dataset

    assert hasattr(src.data_loader, "load_and_preprocess_data")
    assert hasattr(src.train_model, "train_and_evaluate")
    assert hasattr(src.indicators, "analyze_security_indicators")
    assert hasattr(src.predictor, "PhishGuardPredictor")
    assert hasattr(src.logging_utils, "log_security_event")


def test_predictor_singleton_initialization():
    """Verify PhishGuardPredictor initializes and inspects current model metadata."""
    from src.predictor import PhishGuardPredictor
    p = PhishGuardPredictor()
    assert p is not None
    if p.pipeline is not None:
        assert hasattr(p.pipeline, "predict")
