"""Inference and Decision Engine for PhishGuard AI.

Combines the trained machine-learning pipeline with transparent cybersecurity heuristics
to produce explainable risk assessments.
"""

from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib

from src.indicators import analyze_security_indicators, SecurityIndicator
from src.logging_utils import log_security_event

logger = logging.getLogger("phishguard.predictor")

DEFAULT_MODEL_PATH = "models/phishguard_model.joblib"
DEFAULT_METADATA_PATH = "models/model_metadata.json"


@dataclass
class PredictionResult:
    predicted_class: str          # "Likely phishing", "Likely legitimate", "Review recommended"
    model_available: bool
    model_probability_phishing: Optional[float]
    model_probability_legitimate: Optional[float]
    probability_calibration_note: str
    risk_level: str               # "Low", "Medium", "High", "Critical"
    risk_score: float             # 0.0 to 100.0
    risk_summary: str
    indicators: List[Dict[str, Any]]
    recommended_action: str
    decision_breakdown: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhishGuardPredictor:
    """Predictor and decision engine that integrates ML and deterministic security heuristics."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.load_model()

    def load_model(self) -> bool:
        """Load the fitted pipeline and metadata from disk if available."""
        if self.model_path.is_file():
            try:
                self.pipeline = joblib.load(self.model_path)
                logger.info("Successfully loaded model from %s", self.model_path)
            except Exception as e:
                logger.error("Error loading model file %s: %s", self.model_path, e)
                self.pipeline = None

        if self.metadata_path.is_file():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.warning("Error loading metadata file: %s", e)
                self.metadata = {}

        return self.pipeline is not None

    def predict(
        self,
        subject: str,
        body: str,
        session_id: Optional[str] = None
    ) -> PredictionResult:
        """
        Analyze an email given its subject and body.

        Handles empty inputs, missing models, and uncalibrated probabilities safely.
        """
        subject_clean = (subject or "").strip()
        body_clean = (body or "").strip()

        # 1. Input validation
        if not subject_clean and not body_clean:
            return PredictionResult(
                predicted_class="Review recommended",
                model_available=self.pipeline is not None,
                model_probability_phishing=None,
                model_probability_legitimate=None,
                probability_calibration_note="No text provided to evaluate.",
                risk_level="Low",
                risk_score=0.0,
                risk_summary="Empty input: Please paste an email subject or body.",
                indicators=[],
                recommended_action="Enter email text above to perform analysis.",
                decision_breakdown={
                    "status": "empty_input",
                    "model_contribution": "None",
                    "heuristic_contribution": "None"
                }
            )

        # 2. Extract deterministic cybersecurity indicators
        indicators = analyze_security_indicators(subject_clean, body_clean)
        indicator_dicts = [ind.to_dict() for ind in indicators]

        # Calculate heuristic weight (0 to 60 points)
        heuristic_score = 0.0
        for ind in indicators:
            if ind.severity == "high":
                heuristic_score += 25.0
            elif ind.severity == "medium":
                heuristic_score += 15.0
            elif ind.severity == "low":
                heuristic_score += 5.0
        heuristic_score = min(heuristic_score, 60.0)

        # 3. Model inference (if model is available)
        combined_text = f"{subject_clean} {body_clean}".strip()
        model_prob_phish: Optional[float] = None
        model_prob_legit: Optional[float] = None
        model_class: Optional[int] = None
        model_score_contrib = 0.0

        if self.pipeline is not None:
            try:
                # Predict probability using fitted Logistic Regression
                probs = self.pipeline.predict_proba([combined_text])[0]
                model_prob_legit = float(round(probs[0], 4))
                model_prob_phish = float(round(probs[1], 4))
                model_class = int(self.pipeline.predict([combined_text])[0])
                model_score_contrib = model_prob_phish * 50.0  # Up to 50 points
            except Exception as e:
                logger.error("Inference failure: %s", e)

        # 4. Synthesize composite risk assessment
        # Composite score is bounded [0.0, 100.0], combining ML confidence and detected indicators
        if self.pipeline is not None and model_prob_phish is not None:
            # 50 points from ML probability (0.0 to 50.0) + 50 points from heuristic indicators (0.0 to 50.0)
            heuristic_contrib = heuristic_score * (50.0 / 60.0)
            composite_score = max(0.0, min(100.0, round(model_score_contrib + heuristic_contrib, 1)))
        else:
            # Fallback when model is offline: heuristics scale to full 0.0-100.0 range
            heuristic_contrib = heuristic_score * (100.0 / 60.0)
            composite_score = max(0.0, min(100.0, round(heuristic_contrib, 1)))

        if composite_score >= 70.0:
            risk_level = "Critical"
        elif composite_score >= 45.0:
            risk_level = "High"
        elif composite_score >= 25.0:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # 5. Determine clear predicted class
        # Clear separation between ML prediction and decision logic
        has_high_indicator = any(ind.severity == "high" for ind in indicators)
        has_medium_indicator = any(ind.severity == "medium" for ind in indicators)

        if self.pipeline is None:
            if len(indicators) >= 2 or has_high_indicator:
                predicted_class = "Likely phishing"
            elif len(indicators) == 1:
                predicted_class = "Review recommended"
            else:
                predicted_class = "Review recommended"
        else:
            # Model available
            if model_prob_phish is not None and model_prob_phish >= 0.60:
                predicted_class = "Likely phishing"
            elif model_prob_phish is not None and model_prob_phish >= 0.50 and (has_high_indicator or len(indicators) >= 2):
                predicted_class = "Likely phishing"
            elif model_prob_phish is not None and model_prob_phish <= 0.40 and not has_high_indicator:
                predicted_class = "Likely legitimate"
            elif model_prob_phish is not None and model_prob_phish <= 0.40 and has_high_indicator:
                # Contradiction: model thinks legit, but strong phishing heuristic detected
                predicted_class = "Review recommended"
            else:
                # Borderline model confidence
                if has_high_indicator:
                    predicted_class = "Likely phishing"
                else:
                    predicted_class = "Review recommended"

        # 6. Recommendation and Next Steps
        if predicted_class == "Likely phishing":
            rec_action = (
                "Do NOT click any hyperlinks, download attachments, or reply with credentials. "
                "Report this message immediately to your organization's IT Security / SecOps team."
            )
            risk_summary = (
                f"Elevated phishing risk ({composite_score}/100). Model indicates high phishing probability "
                f"corroborated by {len(indicators)} detected security indicator(s)."
            )
        elif predicted_class == "Review recommended":
            rec_action = (
                "Exercise caution. Manually verify the sender through an out-of-band channel (e.g. phone call or internal chat) "
                "before taking any requested actions or clicking links."
            )
            risk_summary = (
                f"Ambiguous or mixed signals ({composite_score}/100). The email presents conflicting patterns or borderline "
                f"confidence requiring human security analyst review."
            )
        else:
            rec_action = (
                "Standard business correspondence indicated. As a routine best practice, confirm sender address matches expected domain."
            )
            risk_summary = (
                f"Low risk ({composite_score}/100). Email matches legitimate linguistic patterns and contains no acute credential harvesting indicators."
            )

        calibration_note = (
            "Model estimates are uncalibrated logistic sigmoid scores representing statistical confidence relative to "
            "the training dataset, not absolute cryptographic certainty."
        )

        decision_breakdown = {
            "model_status": "active" if self.pipeline is not None else "unavailable",
            "model_raw_probability_phish": model_prob_phish,
            "model_score_points": round(model_score_contrib, 1) if model_prob_phish is not None else 0.0,
            "heuristic_indicator_count": len(indicators),
            "heuristic_score_points": round(heuristic_contrib, 1),
            "composite_risk_score": composite_score,
            "risk_level": risk_level,
            "disclaimer": "Security indicators are static heuristic patterns and not proof of malicious intent on their own."
        }

        # 7. Safe security logging (no raw body or password stored)
        log_security_event(
            event_type="email_analyzed",
            subject=subject_clean,
            body_length=len(body_clean),
            has_subject=bool(subject_clean),
            predicted_label=predicted_class,
            risk_level=risk_level,
            model_score=model_prob_phish,
            indicator_count=len(indicators),
            indicator_categories=[ind.category for ind in indicators],
            client_session_id=session_id
        )

        return PredictionResult(
            predicted_class=predicted_class,
            model_available=self.pipeline is not None,
            model_probability_phishing=model_prob_phish,
            model_probability_legitimate=model_prob_legit,
            probability_calibration_note=calibration_note,
            risk_level=risk_level,
            risk_score=composite_score,
            risk_summary=risk_summary,
            indicators=indicator_dicts,
            recommended_action=rec_action,
            decision_breakdown=decision_breakdown
        )
