"""Tests for dataset loading, validation, leakage detection, and splitting."""

import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import (
    load_and_preprocess_data,
    DataValidationError,
    compute_text_hash,
    check_near_duplicates,
    deduplicate_near_texts
)

FIXTURE_PATH = "data/fixtures/emails_fixture.csv"


def test_load_and_preprocess_fixture_success():
    """Verify fixture loads correctly with stratified splits and metadata."""
    X_tr, X_te, y_tr, y_te, meta = load_and_preprocess_data(
        csv_path=FIXTURE_PATH,
        test_size=0.3,
        random_state=42
    )

    assert len(X_tr) > 0
    assert len(X_te) > 0
    assert len(X_tr) + len(X_te) <= meta["cleaned_rows"]
    assert meta["train_size"] == len(X_tr)
    assert meta["test_size"] == len(X_te)
    assert "leakage_analysis" in meta
    assert meta["train_class_distribution"]["phishing_1"] > 0
    assert meta["train_class_distribution"]["legitimate_0"] > 0
    assert meta["leakage_analysis"]["near_duplicate_leakage_count"] == 0


def test_data_validation_missing_columns(tmp_path):
    """Verify DataValidationError when required columns are absent."""
    bad_csv = tmp_path / "bad_columns.csv"
    bad_csv.write_text("random_col_1,random_col_2\nfoo,bar\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="missing required"):
        load_and_preprocess_data(str(bad_csv))


def test_data_validation_unmapped_labels(tmp_path):
    """Verify DataValidationError when labels cannot be mapped."""
    bad_label_csv = tmp_path / "bad_labels.csv"
    bad_label_csv.write_text("body,label\nSome normal email text here,unknown_alien_class\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="unrecognized label"):
        load_and_preprocess_data(str(bad_label_csv))


def test_empty_rows_dropped(tmp_path):
    """Verify empty or ultra-short rows are removed."""
    dirty_csv = tmp_path / "dirty.csv"
    dirty_csv.write_text(
        "body,label\n"
        "Valid text sample number one here,0\n"
        "   ,0\n"
        "ab,0\n"
        "Valid phishing text sample number two here,1\n"
        "Valid legitimate text sample three,0\n"
        "Valid phishing text sample four,1\n"
        "Valid legitimate sample five,0\n"
        "Valid phishing sample six,1\n"
        "Valid legitimate sample seven,0\n"
        "Valid phishing sample eight,1\n"
        "Valid legitimate sample nine,0\n"
        "Valid phishing sample ten,1\n",
        encoding="utf-8"
    )

    X_tr, X_te, y_tr, y_te, meta = load_and_preprocess_data(str(dirty_csv), test_size=0.2)
    assert meta["dropped_empty_rows"] >= 2


def test_compute_text_hash_normalization():
    """Verify whitespace and casing normalization in duplicate hashing."""
    h1 = compute_text_hash("Hello World! Check this out.")
    h2 = compute_text_hash("  hello   world!   check   this out.  ")
    assert h1 == h2


def test_near_duplicate_deduplication_and_leakage():
    """Verify near-duplicate deduplication clusters similar texts and eliminates cross-split leakage."""
    raw_df = pd.DataFrame({
        "text": [
            "The Stanford Canvas team posted five new messages on your Canvas dashboard that requires immediate attention.",
            "Canvas The Stanford Canvas team posted five new messages on your Canvas dashboard that requires immediate attention.",
            "Quarterly engineering sprint sync meeting tomorrow at 2 PM.",
            "Sprint sync meeting tomorrow at 2 PM for engineering team."
        ],
        "label": [1, 1, 0, 0]
    })
    deduped_df, dropped, clusters = deduplicate_near_texts(raw_df, threshold=0.80)
    assert dropped >= 1
    assert len(deduped_df) < len(raw_df)


def test_check_near_duplicates_detection():
    """Verify check_near_duplicates flags pairs when they cross train and test."""
    train_texts = ["urgent verify your password at bank portal now", "regular team meeting tomorrow"]
    test_texts = ["urgent verify your password at bank portal now", "completely unique text"]

    res = check_near_duplicates(train_texts, test_texts)
    assert res["exact_leakage_count"] == 1
    assert "limitation_note" in res
