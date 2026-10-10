"""Tests for privacy-preserving structured security event logging."""

import json
from pathlib import Path
import pytest

from src.logging_utils import log_security_event, read_security_event_statistics, SECURITY_EVENTS_FILE


def test_safe_logging_never_stores_raw_body_or_passwords(tmp_path):
    """Verify that confidential bodies, credentials, and passwords are never logged."""
    sensitive_body = "My secret password is SuperSecretP@ssw0rd! and my SSN is 000-11-2222."
    sensitive_subject = "Urgent: Direct deposit to account 987654321"

    event = log_security_event(
        event_type="email_analyzed",
        subject=sensitive_subject,
        body_length=len(sensitive_body),
        has_subject=True,
        predicted_label="Likely phishing",
        risk_level="High",
        model_score=0.91,
        indicator_count=2,
        indicator_categories=["Credential Harvesting", "Financial Coercion"]
    )

    # 1. Returned event dictionary does not contain sensitive text
    event_str = json.dumps(event)
    assert sensitive_body not in event_str
    assert "SuperSecretP@ssw0rd!" not in event_str
    assert "987654321" not in event_str

    # 2. Check the actual file on disk
    if SECURITY_EVENTS_FILE.is_file():
        file_contents = SECURITY_EVENTS_FILE.read_text(encoding="utf-8")
        assert "SuperSecretP@ssw0rd!" not in file_contents
        assert sensitive_body not in file_contents


def test_read_security_event_statistics():
    """Verify reading statistics aggregates event counts safely."""
    stats = read_security_event_statistics()
    assert "total_analyses" in stats
    assert "phishing_count" in stats
    assert "legitimate_count" in stats
    assert "risk_distribution" in stats
    assert isinstance(stats["recent_events"], list)
