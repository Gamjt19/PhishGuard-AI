"""Tests for deterministic explainable security heuristics."""

import pytest
from src.indicators import analyze_security_indicators, extract_urls


def test_credential_indicator_detection():
    """Verify detection of credential harvesting keywords."""
    indicators = analyze_security_indicators(
        subject="Important security update",
        body="Please confirm your password and enter your credentials to complete account verification."
    )
    categories = [ind.category for ind in indicators]
    assert "Credential Harvesting" in categories


def test_urgency_indicator_detection():
    """Verify detection of manufactured urgency and coercive deadlines."""
    indicators = analyze_security_indicators(
        subject="Final Warning: Account Locked",
        body="Your access will be terminated within 24 hours immediately unless you act."
    )
    categories = [ind.category for ind in indicators]
    assert "Psychological Coercion" in categories


def test_financial_indicator_detection():
    """Verify detection of wire transfers and fake overdue invoices."""
    indicators = analyze_security_indicators(
        subject="Overdue Invoice Penalty",
        body="Send a wire transfer or gift card immediately to settle the overdue penalty."
    )
    categories = [ind.category for ind in indicators]
    assert "Financial Coercion" in categories


def test_raw_ip_url_indicator_detection():
    """Verify detection of hyperlinks with raw numerical IP address hostnames."""
    indicators = analyze_security_indicators(
        subject="Login update",
        body="Visit our portal at http://192.168.1.100/login to access records."
    )
    categories = [ind.category for ind in indicators]
    assert "Suspicious URL" in categories


def test_brand_spoof_subdomain_detection():
    """Verify detection of deceptive brand names embedded in subdomains of foreign root domains."""
    indicators = analyze_security_indicators(
        subject="Service alert",
        body="Verify your Microsoft account here: http://login.microsoft365.verify-portal.ru/auth"
    )
    categories = [ind.category for ind in indicators]
    assert "Brand Impersonation" in categories


def test_benign_text_has_no_indicators():
    """Verify that normal legitimate business communication triggers zero indicators."""
    indicators = analyze_security_indicators(
        subject="Lunch plans tomorrow",
        body="Hi Sarah, are you free for lunch at the cafeteria around 12:30?"
    )
    assert len(indicators) == 0


def test_extract_urls_is_static_and_safe():
    """Verify URL extraction extracts links without making any network requests."""
    text = "Check http://example.com/test and https://sub.domain.org/path?id=123"
    urls = extract_urls(text)
    assert len(urls) == 2
    assert "http://example.com/test" in urls
    assert "https://sub.domain.org/path?id=123" in urls
