"""Data loading, validation, leakage detection, and splitting module for PhishGuard AI."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Set, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("phishguard.data_loader")

DEFAULT_LABEL_MAPPING = {
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
    "legitimate": 0,
    "safe": 0,
    "ham": 0,
    "safe email": 0,
    "phishing": 1,
    "phishing email": 1,
    "phish": 1,
    "malicious": 1,
}


class DataValidationError(Exception):
    """Raised when dataset fails schema or integrity validation."""
    pass


def compute_text_hash(text: str) -> str:
    """Compute normalized SHA-256 hash of email text for exact duplicate checking."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def get_word_ngrams(s: str, max_words: int = 150) -> Set[str]:
    """Extract character-cleaned word 3-grams over prefix for lexical overlap calculation."""
    words = [w for w in s.lower().split() if w.isalnum()][:max_words]
    if len(words) < 3:
        return {" ".join(words)}
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def deduplicate_near_texts(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.80,
    max_prefix_words: int = 150
) -> Tuple[pd.DataFrame, int, List[Dict[str, Any]]]:
    """
    Cluster and deduplicate near-duplicate email texts prior to train/test splitting.
    
    Prevents repeated campaign templates (e.g. slight variations of the same phishing
    lure) from crossing the training/test boundary and inflating held-out evaluation metrics.
    
    Returns:
        (deduplicated_df, dropped_count, cluster_records)
    """
    texts = df[text_col].astype(str).tolist()
    n = len(texts)
    ngrams = [get_word_ngrams(t, max_words=max_prefix_words) for t in texts]

    dropped_indices: Set[int] = set()
    clusters: List[Dict[str, Any]] = []

    for i in range(n):
        if i in dropped_indices:
            continue
        for j in range(i + 1, n):
            if j in dropped_indices:
                continue
            if not ngrams[i] or not ngrams[j]:
                continue
            intersection = len(ngrams[i].intersection(ngrams[j]))
            union = len(ngrams[i].union(ngrams[j]))
            if union > 0:
                sim = intersection / union
                if sim >= threshold:
                    dropped_indices.add(j)
                    clusters.append({
                        "canonical_index": i,
                        "dropped_duplicate_index": j,
                        "similarity": round(sim, 4),
                        "canonical_preview": texts[i][:120].strip(),
                        "dropped_preview": texts[j][:120].strip()
                    })

    deduped_df = df.drop(index=list(dropped_indices)).reset_index(drop=True)
    logger.info(
        "Near-duplicate deduplication: dropped %d redundant samples (threshold=%.2f). %d remain.",
        len(dropped_indices), threshold, len(deduped_df)
    )
    return deduped_df, len(dropped_indices), clusters


def check_near_duplicates(
    train_texts: list[str],
    test_texts: list[str],
    threshold: float = 0.80
) -> Dict[str, Any]:
    """
    Verify whether any exact or near-duplicate leakage exists between train and test splits.
    """
    train_hashes = {compute_text_hash(t) for t in train_texts}
    exact_leakage_count = sum(1 for t in test_texts if compute_text_hash(t) in train_hashes)

    train_ngrams = [get_word_ngrams(t) for t in train_texts]
    test_ngrams = [get_word_ngrams(t) for t in test_texts]

    near_leakage_count = 0
    sampled_comparisons = 0
    leaked_pairs: List[Dict[str, Any]] = []

    for te_idx, te_set in enumerate(test_ngrams):
        if not te_set:
            continue
        for tr_idx, tr_set in enumerate(train_ngrams):
            sampled_comparisons += 1
            if not tr_set:
                continue
            intersection = len(te_set.intersection(tr_set))
            union = len(te_set.union(tr_set))
            if union > 0:
                sim = intersection / union
                if sim >= threshold:
                    near_leakage_count += 1
                    leaked_pairs.append({
                        "test_idx": te_idx,
                        "train_idx": tr_idx,
                        "similarity": round(sim, 4),
                        "test_preview": test_texts[te_idx][:100],
                        "train_preview": train_texts[tr_idx][:100]
                    })
                    break

    return {
        "exact_leakage_count": exact_leakage_count,
        "near_duplicate_leakage_count": near_leakage_count,
        "sampled_comparisons": sampled_comparisons,
        "threshold": threshold,
        "method": "word 3-gram Jaccard similarity (150 words prefix)",
        "leaked_pairs": leaked_pairs,
        "limitation_note": (
            "Detects lexical n-gram overlap between train and test sets. "
            "Semantic rephrasing or shared boilerplate disclaimers may still occur."
        )
    }


def load_and_preprocess_data(
    csv_path: str,
    subject_col: str = "subject",
    body_col: str = "body",
    text_col: Optional[str] = None,
    label_col: str = "label",
    label_mapping: Optional[Dict[Any, int]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    deduplicate_exact: bool = True,
    deduplicate_near: bool = True,
    near_duplicate_threshold: float = 0.80
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Load, validate, clean, deduplicate, and split an email dataset for training and evaluation.

    Ensures that exact and near-duplicate email campaign variants are deduplicated
    prior to train/test partitioning, preventing cross-split leakage.

    Returns:
        (X_train, X_test, y_train, y_test, metadata)
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    logger.info("Loaded CSV from %s with %d rows and columns: %s", csv_path, initial_rows, list(df.columns))

    mapping = label_mapping or DEFAULT_LABEL_MAPPING

    # 1. Validate required columns
    if text_col and text_col in df.columns:
        combined_text = df[text_col].astype(str)
    elif body_col in df.columns:
        if subject_col in df.columns:
            combined_text = (
                df[subject_col].fillna("").astype(str).str.strip() + " " +
                df[body_col].fillna("").astype(str).str.strip()
            )
        else:
            combined_text = df[body_col].fillna("").astype(str)
    else:
        raise DataValidationError(
            f"Dataset missing required text columns. Expected '{text_col}' or '{body_col}', "
            f"found columns: {list(df.columns)}"
        )

    if label_col not in df.columns:
        raise DataValidationError(
            f"Dataset missing required label column '{label_col}'. Found: {list(df.columns)}"
        )

    # 2. Map and validate labels
    normalized_labels = df[label_col].astype(str).str.strip().str.lower()
    mapped_labels = normalized_labels.map(lambda v: mapping.get(v, mapping.get(int(v) if v.isdigit() else None, None)))
    
    unmapped_mask = mapped_labels.isna()
    if unmapped_mask.any():
        bad_values = df[label_col][unmapped_mask].unique()
        raise DataValidationError(
            f"Found unrecognized label values: {bad_values}. Allowed mapped values: {list(mapping.keys())}"
        )

    df_cleaned = pd.DataFrame({
        "text": combined_text.str.strip(),
        "label": mapped_labels.astype(int)
    })

    # 3. Filter empty or whitespace-only texts
    non_empty_mask = df_cleaned["text"].str.len() > 3
    dropped_empty = int((~non_empty_mask).sum())
    df_cleaned = df_cleaned[non_empty_mask].reset_index(drop=True)

    # 4. Handle exact duplicates
    dropped_exact = 0
    if deduplicate_exact:
        before_exact = len(df_cleaned)
        df_cleaned["hash"] = df_cleaned["text"].apply(compute_text_hash)
        df_cleaned = df_cleaned.drop_duplicates(subset=["hash"]).reset_index(drop=True)
        df_cleaned = df_cleaned.drop(columns=["hash"])
        dropped_exact = before_exact - len(df_cleaned)

    # 5. Handle near-duplicates (Clustering and deduplication before splitting)
    dropped_near = 0
    near_clusters: List[Dict[str, Any]] = []
    if deduplicate_near:
        df_cleaned, dropped_near, near_clusters = deduplicate_near_texts(
            df_cleaned,
            text_col="text",
            threshold=near_duplicate_threshold
        )

    final_rows = len(df_cleaned)
    if final_rows < 10:
        raise DataValidationError(f"Insufficient usable samples after cleaning: only {final_rows} rows remaining.")

    class_counts = df_cleaned["label"].value_counts().to_dict()
    if len(class_counts) < 2:
        raise DataValidationError(f"Dataset must contain both classes (0 and 1). Found only: {class_counts}")

    # 6. Stratified train/test split on unique, deduplicated corpus
    X = df_cleaned["text"]
    y = df_cleaned["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 7. Verify leakage between train and test splits
    leakage_stats = check_near_duplicates(
        X_train.tolist(),
        X_test.tolist(),
        threshold=near_duplicate_threshold
    )

    metadata = {
        "source_path": str(csv_path),
        "initial_rows": initial_rows,
        "cleaned_rows": final_rows,
        "dropped_empty_rows": dropped_empty,
        "dropped_exact_duplicates": dropped_exact,
        "dropped_near_duplicates": dropped_near,
        "near_duplicate_clusters": near_clusters,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_class_distribution": {
            "legitimate_0": int((y_train == 0).sum()),
            "phishing_1": int((y_train == 1).sum()),
        },
        "test_class_distribution": {
            "legitimate_0": int((y_test == 0).sum()),
            "phishing_1": int((y_test == 1).sum()),
        },
        "leakage_analysis": leakage_stats
    }

    logger.info(
        "Split dataset: %d train (%s), %d test (%s) | Exact leakage=%d, Near leakage=%d",
        len(X_train), metadata["train_class_distribution"],
        len(X_test), metadata["test_class_distribution"],
        leakage_stats["exact_leakage_count"], leakage_stats["near_duplicate_leakage_count"]
    )

    return X_train, X_test, y_train, y_test, metadata
