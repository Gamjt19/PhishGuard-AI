"""Dataset downloader and provenance tracker for PhishGuard AI.

Downloads and prepares genuine phishing vs legitimate email research datasets.
Primary source:
  Shahriar, S., Mukherjee, A., & Gnawali, O. (2022).
  Improving Phishing Detection Via Psychological Trait Scoring.
  Proceedings of the IADIS International Conference Web Based Communities 2022 (pp. 131-139).
  - Genuine phishing emails collected from university IT security archives.
  - Genuine legitimate business emails from the Enron email corpus.
  - 326 total verified emails (163 legitimate [0], 163 phishing [1]).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import urllib.request
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phishguard.downloader")

DATASET_URL = "https://raw.githubusercontent.com/sadat1971/Phishing_Email/main/Data/curated_set.csv"

DATASET_PROVENANCE: Dict[str, Any] = {
    "dataset_name": "Shahriar University Phishing & Enron Legitimate Corpus",
    "primary_paper": "Shahriar, S., Mukherjee, A., & Gnawali, O. (2022). Improving Phishing Detection Via Psychological Trait Scoring. IADIS MCCSIS 2022.",
    "source_url": DATASET_URL,
    "license": "Open Academic Research / CC-BY Compatible",
    "description": (
        "Curated benchmark combining genuine phishing emails collected from university security reports "
        "and legitimate enterprise emails from the Enron corpus. Strictly separates phishing from legitimate "
        "(NOT generic spam/commercial marketing)."
    ),
    "label_definition": {
        "0": "Legitimate email (Enron Corpus)",
        "1": "Phishing email (University Security Archives)"
    },
    "original_columns": ["text", "source", "is_phishing"]
}


def download_benchmark_dataset(
    dest_path: str = "data/phishing_dataset.csv",
    provenance_path: str = "data/dataset_provenance.json",
    timeout_sec: int = 20
) -> Path:
    """Download the benchmark dataset and standardize columns."""
    dest = Path(dest_path)
    prov_file = Path(provenance_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prov_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading academic dataset from %s", DATASET_URL)
    import requests
    resp = requests.get(
        DATASET_URL,
        headers={"User-Agent": "PhishGuardAI-Research/1.0"},
        timeout=timeout_sec
    )
    resp.raise_for_status()
    raw_csv = resp.text

    from io import StringIO
    df = pd.read_csv(StringIO(raw_csv))
    logger.info("Downloaded raw dataset with shape %s", df.shape)

    # Standardize to subject, body, label
    # The source text begins with 'Subject: ...' or full body
    subjects = []
    bodies = []
    for text in df["text"].astype(str):
        if text.lower().startswith("subject:"):
            parts = text.split("\n", 1)
            subj = parts[0][len("subject:"):].strip()
            body = parts[1].strip() if len(parts) > 1 else ""
        else:
            subj = ""
            body = text.strip()
        subjects.append(subj)
        bodies.append(body)

    standardized_df = pd.DataFrame({
        "subject": subjects,
        "body": bodies,
        "label": df["is_phishing"].astype(int)
    })

    # Drop any null or empty rows
    standardized_df = standardized_df.dropna(subset=["body"])
    standardized_df = standardized_df[standardized_df["body"].str.strip() != ""]

    standardized_df.to_csv(dest, index=False)
    logger.info("Saved %d standardized emails to %s", len(standardized_df), dest)

    provenance_record = dict(DATASET_PROVENANCE)
    provenance_record["total_rows"] = int(len(standardized_df))
    provenance_record["class_distribution"] = {
        "legitimate_0": int((standardized_df["label"] == 0).sum()),
        "phishing_1": int((standardized_df["label"] == 1).sum()),
    }
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(provenance_record, f, indent=2)
    logger.info("Saved dataset provenance metadata to %s", prov_file)

    return dest


if __name__ == "__main__":
    download_benchmark_dataset()
