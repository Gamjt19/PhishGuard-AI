"""Safe structured security logging for PhishGuard AI.

Ensures security events are audited without recording sensitive email bodies,
credentials, tokens, or personal identifiers.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("phishguard.security_logger")

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SECURITY_EVENTS_FILE = LOGS_DIR / "security_events.jsonl"


def compute_anonymized_hash(text: str) -> str:
    """Compute truncated SHA-256 fingerprint for event correlation without storing text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def log_security_event(
    event_type: str,
    subject: str = "",
    body_length: int = 0,
    has_subject: bool = False,
    predicted_label: Optional[str] = None,
    risk_level: Optional[str] = None,
    model_score: Optional[float] = None,
    indicator_count: int = 0,
    indicator_categories: Optional[List[str]] = None,
    client_session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Append an anonymized security audit event to the log file.

    SAFETY GUARANTEE:
    - Never stores raw email body, recipient names, passwords, or tokens.
    - Stores only length, metadata presence, anonymized hash prefix, and analytical results.
    """
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "session_id": client_session_id or "local_session",
        "input_metadata": {
            "has_subject": has_subject,
            "body_length_chars": body_length,
            "subject_fingerprint": compute_anonymized_hash(subject) if subject else None,
        },
        "analysis_result": {
            "predicted_label": predicted_label,
            "risk_level": risk_level,
            "model_probability": round(model_score, 4) if model_score is not None else None,
            "indicator_count": indicator_count,
            "indicator_categories": indicator_categories or [],
        }
    }

    try:
        with open(SECURITY_EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error("Failed to write to security event log: %s", e)

    return event


def read_security_event_statistics() -> Dict[str, Any]:
    """
    Read safely recorded security events from disk to compute live dashboard insights.
    Returns aggregated counts and distributions.
    """
    if not SECURITY_EVENTS_FILE.is_file():
        return {
            "total_analyses": 0,
            "phishing_count": 0,
            "legitimate_count": 0,
            "review_count": 0,
            "risk_distribution": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            "indicator_category_counts": {},
            "recent_events": []
        }

    total = 0
    phish = 0
    legit = 0
    review = 0
    risk_dist = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    indicator_cat_counts: Dict[str, int] = {}
    recent_events = []

    try:
        with open(SECURITY_EVENTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue

                if ev.get("event_type") != "email_analyzed":
                    continue

                total += 1
                res = ev.get("analysis_result", {})
                lbl = res.get("predicted_label")
                if lbl == "Likely phishing":
                    phish += 1
                elif lbl == "Likely legitimate":
                    legit += 1
                elif lbl == "Review recommended":
                    review += 1

                rk = res.get("risk_level")
                if rk in risk_dist:
                    risk_dist[rk] += 1

                for cat in res.get("indicator_categories", []):
                    indicator_cat_counts[cat] = indicator_cat_counts.get(cat, 0) + 1

                recent_events.append({
                    "timestamp": ev.get("timestamp_utc", "")[:19].replace("T", " "),
                    "label": lbl,
                    "risk": rk,
                    "indicators": res.get("indicator_count", 0),
                    "fingerprint": ev.get("input_metadata", {}).get("subject_fingerprint", "N/A")
                })
    except Exception as e:
        logger.warning("Error reading security events: %s", e)

    return {
        "total_analyses": total,
        "phishing_count": phish,
        "legitimate_count": legit,
        "review_count": review,
        "risk_distribution": risk_dist,
        "indicator_category_counts": indicator_cat_counts,
        "recent_events": recent_events[-10:][::-1]  # Most recent 10 in reverse order
    }
