"""PhishGuard AI — Explainable Cybersecurity Email Phishing Detection System.

Competition-ready Streamlit application implementing an explainable ML pipeline,
deterministic security heuristics, real evaluation telemetry, and privacy-preserving audit logs.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st

from src.predictor import PhishGuardPredictor, PredictionResult
from src.logging_utils import read_security_event_statistics

# ---------------------------------------------------------
# Page Configuration & Cybersecurity Design Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="PhishGuard AI | Explainable Phishing Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling: Restrained Dark Navy Dashboard
CUSTOM_CSS = """
<style>
    /* Dark Navy Palette */
    :root {
        --bg-main: #070d1e;
        --bg-card: #0d1a36;
        --bg-card-hover: #12244a;
        --border-subtle: #1c325c;
        --text-primary: #f0f4fc;
        --text-secondary: #9cb1d6;
        --cyan-accent: #00b4d8;
        --green-safe: #10b981;
        --red-danger: #ef4444;
        --amber-warning: #f59e0b;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Header styling */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 18px 24px;
        background: linear-gradient(135deg, #0b1736 0%, #081126 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .brand-title {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .brand-subtitle {
        font-size: 14px;
        color: var(--cyan-accent);
        margin: 0;
    }
    .brand-badge {
        margin-left: auto;
        background: rgba(0, 180, 216, 0.12);
        border: 1px solid rgba(0, 180, 216, 0.3);
        color: var(--cyan-accent);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Stat Cards */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .stat-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-secondary);
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: #ffffff;
    }
    .stat-detail {
        font-size: 11px;
        color: var(--cyan-accent);
        margin-top: 4px;
    }

    /* Result Banner Cards - Structured Visual Hierarchy (No Concatenation) */
    .banner-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(13, 26, 54, 0.95) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 6px solid var(--red-danger);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .banner-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(13, 26, 54, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-left: 6px solid var(--green-safe);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .banner-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(13, 26, 54, 0.95) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-left: 6px solid var(--amber-warning);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .verdict-header-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .verdict-title {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.3px;
        margin-bottom: 12px;
        line-height: 1.2;
    }
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .pill-risk-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
    }
    .pill-risk-high {
        background: rgba(249, 115, 22, 0.2);
        color: #fdba74;
        border: 1px solid #f97316;
    }
    .pill-risk-medium {
        background: rgba(245, 158, 11, 0.2);
        color: #fde68a;
        border: 1px solid #f59e0b;
    }
    .pill-risk-low {
        background: rgba(16, 185, 129, 0.2);
        color: #a7f3d0;
        border: 1px solid #10b981;
    }
    .pill-score {
        background: rgba(0, 180, 216, 0.15);
        color: #7dd3fc;
        border: 1px solid rgba(0, 180, 216, 0.4);
    }

    /* Indicator item card */
    .indicator-card {
        background: #09142e;
        border: 1px solid #1f3663;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .indicator-tag-high {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .indicator-tag-medium {
        background: rgba(245, 158, 11, 0.2);
        color: #fde68a;
        border: 1px solid #f59e0b;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .indicator-tag-low {
        background: rgba(0, 180, 216, 0.2);
        color: #bae6fd;
        border: 1px solid #00b4d8;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed #334155;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 12px;
        color: #94a3b8;
        margin-top: 14px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Application State & Predictor Initialization
# ---------------------------------------------------------
@st.cache_resource
def get_predictor() -> PhishGuardPredictor:
    """Initialize singleton predictor instance."""
    return PhishGuardPredictor()


predictor = get_predictor()


# ---------------------------------------------------------
# Harmless Demonstration Scenarios (Populates Input Only)
# ---------------------------------------------------------
DEMO_SCENARIOS = {
    "Select a pre-configured scenario...": {
        "subject": "",
        "body": "",
        "desc": "Select a scenario to populate the input fields. You must click 'Analyze Email' to run inference."
    },
    "Phishing: Urgent M365 Password Reset": {
        "subject": "URGENT: Microsoft 365 Account Deactivation Notice",
        "body": (
            "Dear User,\n\n"
            "We detected unusual sign-in activity on your corporate account. Your mailbox access will be "
            "terminated within 24 hours unless you verify your identity.\n\n"
            "Please click the security portal link below to update your password immediately:\n"
            "http://login.microsoft365.verify-security.ru/auth\n\n"
            "Failure to act will result in account suspension without delay.\n\n"
            "IT Helpdesk Support"
        ),
        "desc": "Urgent coercive tone, credential harvest request, deceptive brand spoof subdomain on high-risk TLD."
    },
    "Phishing: Fake Overdue Invoice & Wire Penalty": {
        "subject": "Overdue Invoice Notice - Overdue Legal Penalty",
        "body": (
            "Attention Accounts Payable,\n\n"
            "Your vendor invoice #49204 is severely past due. Please wire transfer $4,850.00 immediately "
            "to account 993-201-94 to avoid immediate legal action and corporate credit cancellation.\n\n"
            "Review payment instructions here: http://192.168.1.100/invoice_details.pdf\n\n"
            "Accounts Receivable Department"
        ),
        "desc": "Financial coercion, wire demand, raw IP address hyperlink."
    },
    "Legitimate: Team Sprint Planning": {
        "subject": "Sprint 14 Planning & Architecture Discussion",
        "body": (
            "Hi team,\n\n"
            "Please review the backlog tickets in Jira before our sprint planning meeting this Thursday at 2:00 PM. "
            "We will finalize milestones for the customer dashboard and assign initial story points.\n\n"
            "Let me know if you have agenda items you would like to add.\n\n"
            "Best regards,\nAlex Jordan\nEngineering Lead"
        ),
        "desc": "Normal enterprise internal correspondence, routine meeting invite, no credential or urgency flags."
    },
    "Ambiguous: Routine IT Network Maintenance": {
        "subject": "Scheduled IT Maintenance Window - Saturday 02:00 UTC",
        "body": (
            "All Staff,\n\n"
            "Corporate network infrastructure will undergo scheduled maintenance this Saturday between 02:00 and 04:00 UTC. "
            "External VPN connectivity may experience intermittent latency. No action is required on your part.\n\n"
            "If you experience persistent connectivity issues after Monday morning, submit a ticket to the internal helpdesk.\n\n"
            "IT Operations"
        ),
        "desc": "Contains IT terminology and maintenance references, but lacks coercive links or credential demands."
    }
}


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ System Status")
    
    # Model Status
    if predictor.pipeline is not None:
        st.success("● ML Model: Online (Fitted Pipeline)")
    else:
        st.error("● ML Model: Unavailable (Joblib not found)")
        st.info("Run `python -m src.train_model` to train the pipeline.")

    # Metadata & Dataset Status
    meta = predictor.metadata
    if meta:
        st.markdown("**Dataset Provenance:**")
        prov_name = meta.get("dataset_provenance", {}).get("dataset_name", "Curated Phishing Corpus")
        st.caption(f"📂 {prov_name}")
        train_cnt = meta.get("dataset_split", {}).get("train_size", "N/A")
        test_cnt = meta.get("dataset_split", {}).get("test_size", "N/A")
        st.caption(f"📊 Samples: {train_cnt} train | {test_cnt} held-out test")
    else:
        st.caption("No offline metadata loaded.")

    st.markdown("---")
    st.markdown("### ⚙️ Architecture & Model")
    st.markdown("""
    - **Classifier**: Logistic Regression (L2, Balanced)
    - **Vectorizer**: TF-IDF (1-2 N-grams, Stopwords)
    - **Engine**: Rule-based Security Indicators
    - **Decision**: Multi-Factor Hybrid Risk Model
    """)

    st.markdown("---")
    st.markdown("### 🔒 Privacy Architecture")
    st.markdown("""
    - **Zero External Telemetry**: Runs 100% offline.
    - **Safe Audit Logging**: No email bodies or credentials recorded.
    - **Passive Inspection**: Submitted URLs are never fetched or probed.
    """)


# ---------------------------------------------------------
# App Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <div style="font-size: 34px;">🛡️</div>
    <div>
        <h1 class="brand-title">PhishGuard AI</h1>
        <p class="brand-subtitle">Explainable Phishing Email Defense & Security Decision Support</p>
    </div>
    <div class="brand-badge">Competition Prototype v1.0</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_analyzer, tab_overview, tab_insights, tab_evaluation, tab_scenarios = st.tabs([
    "📥 Email Analyzer",
    "📊 System Overview",
    "📈 Security Insights",
    "🔬 Model Evaluation",
    "🎯 Demo Scenarios"
])


# =========================================================
# TAB 1: EMAIL ANALYZER
# =========================================================
with tab_analyzer:
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### 📧 Input Email Content")
        st.caption("Paste subject and raw body text below for explainable ML and heuristic threat analysis.")

        # Preset filler from session state if populated from Scenarios tab
        default_sub = st.session_state.get("input_subject", "")
        default_body = st.session_state.get("input_body", "")

        subject_input = st.text_input(
            "Email Subject (Optional)",
            value=default_sub,
            placeholder="e.g., URGENT: Verify your account access",
            help="Email subject line often contains coercive psychological framing."
        )

        body_input = st.text_area(
            "Email Body (Required for full analysis)",
            value=default_body,
            height=280,
            placeholder="Paste raw email body text here...\n\nExample:\nDear user, your mailbox is full. Click http://login-portal.biz to verify credentials within 24 hours.",
            help="The message body is evaluated through TF-IDF feature extraction and static indicator heuristics."
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_clicked = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🧹 Clear Input", use_container_width=True):
                st.session_state["input_subject"] = ""
                st.session_state["input_body"] = ""
                st.rerun()

        st.markdown("""
        <div class="disclaimer-box">
            🛡️ <b>Security Safeguard:</b> PhishGuard AI does not fetch or open links contained in submitted text.
            Analysis is performed locally on your device without transmitting contents to external cloud services.
        </div>
        """, unsafe_allow_html=True)

    with col_results:
        st.markdown("### 🛡️ Threat Assessment")

        if analyze_clicked:
            if not subject_input.strip() and not body_input.strip():
                st.warning("⚠️ Please enter an email subject or body to analyze.")
            else:
                with st.spinner("Analyzing message with ML pipeline and security heuristics..."):
                    result: PredictionResult = predictor.predict(
                        subject=subject_input,
                        body=body_input,
                        session_id=st.session_state.get("session_id", "web_session")
                    )

                # 1. Prediction Banner with clean visual hierarchy (No Concatenation)
                risk_pill_class = f"pill-risk-{result.risk_level.lower()}"
                if result.predicted_class == "Likely phishing":
                    st.markdown(f"""
                    <div class="banner-danger">
                        <div class="verdict-header-label" style="color:#fca5a5;">
                            Security Analysis Verdict
                        </div>
                        <div class="verdict-title" style="color:#ef4444;">
                            🚨 Likely Phishing
                        </div>
                        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:12px;">
                            <span class="pill-badge {risk_pill_class}">RISK LEVEL: {result.risk_level.upper()}</span>
                            <span class="pill-badge pill-score">COMPOSITE SCORE: {result.risk_score:.1f} / 100</span>
                        </div>
                        <div style="font-size:13px; color:#fecaca; line-height:1.5; border-top:1px solid rgba(239,68,68,0.25); padding-top:10px;">
                            {result.risk_summary}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                elif result.predicted_class == "Likely legitimate":
                    st.markdown(f"""
                    <div class="banner-safe">
                        <div class="verdict-header-label" style="color:#a7f3d0;">
                            Security Analysis Verdict
                        </div>
                        <div class="verdict-title" style="color:#10b981;">
                            ✅ Likely Legitimate
                        </div>
                        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:12px;">
                            <span class="pill-badge {risk_pill_class}">RISK LEVEL: {result.risk_level.upper()}</span>
                            <span class="pill-badge pill-score">COMPOSITE SCORE: {result.risk_score:.1f} / 100</span>
                        </div>
                        <div style="font-size:13px; color:#a7f3d0; line-height:1.5; border-top:1px solid rgba(16,185,129,0.25); padding-top:10px;">
                            {result.risk_summary}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="banner-warning">
                        <div class="verdict-header-label" style="color:#fde68a;">
                            Security Analysis Verdict
                        </div>
                        <div class="verdict-title" style="color:#f59e0b;">
                            ⚠️ Review Recommended
                        </div>
                        <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:12px;">
                            <span class="pill-badge {risk_pill_class}">RISK LEVEL: {result.risk_level.upper()}</span>
                            <span class="pill-badge pill-score">COMPOSITE SCORE: {result.risk_score:.1f} / 100</span>
                        </div>
                        <div style="font-size:13px; color:#fde68a; line-height:1.5; border-top:1px solid rgba(245,158,11,0.25); padding-top:10px;">
                            {result.risk_summary}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. Distinct Model Probability & Composite Score Breakdown
                st.markdown("#### 🤖 Decision Telemetry")
                col_tele_1, col_tele_2 = st.columns(2)
                with col_tele_1:
                    p_phish_text = f"{result.model_probability_phishing*100:.1f}%" if (result.model_available and result.model_probability_phishing is not None) else "N/A"
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Model Phishing Probability</div>
                        <div class="stat-value" style="color:#00b4d8;">{p_phish_text}</div>
                        <div class="stat-detail">Statistical confidence (TF-IDF + LR)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_tele_2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-label">Composite Risk Rating</div>
                        <div class="stat-value">{result.risk_score:.1f} <span style="font-size:14px; color:#9cb1d6;">/ 100</span></div>
                        <div class="stat-detail">{result.risk_level} risk tier (ML + heuristics)</div>
                    </div>
                    """, unsafe_allow_html=True)

                if result.model_available and result.model_probability_phishing is not None:
                    p_phish = result.model_probability_phishing
                    p_legit = result.model_probability_legitimate or (1.0 - p_phish)
                    st.progress(p_phish, text=f"Model Phishing Score: {p_phish*100:.1f}%  |  Legitimate Score: {p_legit*100:.1f}%")
                    st.caption(f"ℹ️ **Calibration Notice:** {result.probability_calibration_note}")
                else:
                    st.warning("⚠️ ML Model pipeline is offline. Threat level computed via fallback heuristic rule engine.")

                # 3. Transparent Security Indicators
                st.markdown(f"#### 🔎 Detected Security Indicators ({len(result.indicators)})")
                if not result.indicators:
                    st.info("No overt phishing heuristics detected (no credential demands, IP links, or urgency keywords).")
                else:
                    for ind in result.indicators:
                        sev_badge = f"indicator-tag-{ind['severity']}"
                        st.markdown(f"""
                        <div class="indicator-card">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <strong style="color:#ffffff;">{ind['name']}</strong>
                                <span class="{sev_badge}">{ind['severity'].upper()} SEVERITY</span>
                            </div>
                            <div style="font-size:12px; color:#9cb1d6; margin-bottom:6px;">{ind['description']}</div>
                            <div style="font-size:11px; color:#00b4d8; background:rgba(0,180,216,0.08); padding:4px 8px; border-radius:4px; margin-bottom:6px;">
                                <b>Matched Evidence:</b> {', '.join(ind['matched_patterns'])}
                            </div>
                            <div style="font-size:11px; color:#64748b; font-style:italic;">
                                💡 <b>Heuristic Rationale:</b> {ind['heuristic_rationale']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # 4. Actionable Next Step
                st.markdown("#### 🎯 Recommended Action")
                st.info(f"**Security Next Step:** {result.recommended_action}")

        else:
            st.info("👈 Enter an email subject and body on the left and click **'Analyze Email'** to view the explainable threat assessment.")


# =========================================================
# TAB 2: SYSTEM OVERVIEW
# =========================================================
with tab_overview:
    st.markdown("### 📊 Project Status & Evaluation Telemetry")
    st.caption("Transparent reporting of model status, training provenance, and offline validation metrics.")

    meta = predictor.metadata
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    if meta and "evaluation_metrics" in meta:
        ev = meta["evaluation_metrics"]
        split = meta.get("dataset_split", {})
        prov = meta.get("dataset_provenance", {})

        with col_m1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Model Accuracy</div>
                <div class="stat-value">{ev.get('accuracy', 0)*100:.1f}%</div>
                <div class="stat-detail">Held-out test split</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Precision (Phishing)</div>
                <div class="stat-value">{ev.get('precision_phishing', 0)*100:.1f}%</div>
                <div class="stat-detail">True positives / predicted phish</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Recall (Phishing)</div>
                <div class="stat-value">{ev.get('recall_phishing', 0)*100:.1f}%</div>
                <div class="stat-detail">Caught / actual phish</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">F1-Score (Phishing)</div>
                <div class="stat-value">{ev.get('f1_phishing', 0)*100:.1f}%</div>
                <div class="stat-detail">Harmonic mean precision & recall</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_ov_left, col_ov_right = st.columns([1, 1], gap="large")

        with col_ov_left:
            st.markdown("#### 📂 Offline Dataset Provenance")
            st.markdown(f"**Dataset Name:** {prov.get('dataset_name', 'Curated Academic Corpus')}")
            st.markdown(f"**Primary Paper:** *{prov.get('primary_paper', 'Shahriar et al. (IADIS 2022)')}*")
            st.markdown(f"**License:** `{prov.get('license', 'Open Research / CC-BY')}`")
            st.markdown(f"**Source URL:** [{prov.get('source_url', 'GitHub Archive')}]({prov.get('source_url', '#')})")
            st.markdown(f"**Description:** {prov.get('description', '')}")
            
            st.markdown("**Label Definitions:**")
            st.write(prov.get("label_definition", {"0": "Legitimate", "1": "Phishing"}))

        with col_ov_right:
            st.markdown("#### ⚖️ Offline Split & Leakage Verification")
            st.markdown(f"- **Total Cleaned Samples:** {split.get('cleaned_rows', 'N/A')}")
            st.markdown(f"- **Training Samples:** {split.get('train_size', 'N/A')} (Stratified 80%)")
            st.markdown(f"- **Held-Out Test Samples:** {split.get('test_size', 'N/A')} (Stratified 20%)")
            st.markdown(f"- **Exact Duplicates Dropped:** {split.get('dropped_exact_duplicates', 0)}")
            st.markdown(f"- **Near-Duplicate Variants Clustered & Dropped:** {split.get('dropped_near_duplicates', 0)}")
            
            leakage = split.get("leakage_analysis", {})
            st.markdown(f"- **Exact Train-Test Leakage Overlap:** `{leakage.get('exact_leakage_count', 0)}` emails")
            st.markdown(f"- **Near-Duplicate Overlap (Jaccard >= 0.80):** `{leakage.get('near_duplicate_leakage_count', 0)}` pairs (Eliminated)")
            st.caption(f"Method: {leakage.get('method', 'Word n-gram similarity')}")
            st.caption(f"Note: {leakage.get('limitation_note', '')}")

    else:
        st.warning("No model metadata found. Please run model training using `python -m src.train_model`.")


# =========================================================
# TAB 3: SECURITY INSIGHTS
# =========================================================
with tab_insights:
    st.markdown("### 📈 Live Security Audit & Session Telemetry")
    st.caption("Live aggregate analytics computed from safely recorded JSONL audit logs. No raw bodies or credentials are saved.")

    stats = read_security_event_statistics()
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Total Emails Analyzed</div>
            <div class="stat-value">{stats['total_analyses']}</div>
            <div class="stat-detail">Safely audited session events</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Predicted Likely Phishing</div>
            <div class="stat-value" style="color:#ef4444;">{stats['phishing_count']}</div>
            <div class="stat-detail">Classified as likely phishing (decision support)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Predicted Likely Legitimate</div>
            <div class="stat-value" style="color:#10b981;">{stats['legitimate_count']}</div>
            <div class="stat-detail">Classified as routine communication</div>
        </div>
        """, unsafe_allow_html=True)

    with col_s4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">Review Recommended</div>
            <div class="stat-value" style="color:#f59e0b;">{stats['review_count']}</div>
            <div class="stat-detail">Flagged for manual analyst verification</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col_chart1, col_chart2 = st.columns([1, 1], gap="large")

    with col_chart1:
        st.markdown("#### 📊 Risk Level Distribution")
        risk_df = pd.DataFrame(
            list(stats["risk_distribution"].items()),
            columns=["Risk Level", "Count"]
        )
        if stats["total_analyses"] > 0:
            st.bar_chart(risk_df.set_index("Risk Level"), color="#00b4d8")
        else:
            st.info("No analysis events recorded yet in current audit log. Analyze emails in Tab 1 to generate live statistics.")

    with col_chart2:
        st.markdown("#### 🔎 Common Threat Indicator Categories")
        ind_counts = stats["indicator_category_counts"]
        if ind_counts:
            ind_df = pd.DataFrame(
                list(ind_counts.items()),
                columns=["Category", "Detections"]
            ).sort_values("Detections", ascending=False)
            st.dataframe(ind_df, use_container_width=True, hide_index=True)
        else:
            st.info("No threat indicator triggers recorded yet.")

    st.markdown("#### 📜 Safe Security Audit Trail (Recent 10 Events)")
    if stats["recent_events"]:
        audit_df = pd.DataFrame(stats["recent_events"])
        st.dataframe(audit_df, use_container_width=True, hide_index=True)
        st.caption("🔒 All logged events store only timestamp, length, hash fingerprint, and classifications. Raw email text is never logged.")
    else:
        st.caption("No events in audit log.")


# =========================================================
# TAB 4: MODEL EVALUATION
# =========================================================
with tab_evaluation:
    st.markdown("### 🔬 Rigorous Model Evaluation")
    st.caption("Real offline test set evaluation metrics. Not simulated or hardcoded.")

    meta = predictor.metadata
    if meta and "evaluation_metrics" in meta:
        ev = meta["evaluation_metrics"]
        cm_data = ev["confusion_matrix"]
        raw_cm = cm_data.get("raw_matrix", [[0, 0], [0, 0]])

        col_ev1, col_ev2 = st.columns([1, 1], gap="large")

        with col_ev1:
            st.markdown("#### 🧮 Confusion Matrix (Held-out Test Split)")
            cm_display = pd.DataFrame(
                raw_cm,
                index=["Actual Legitimate (0)", "Actual Phishing (1)"],
                columns=["Predicted Legitimate (0)", "Predicted Phishing (1)"]
            )
            st.table(cm_display)

            tn = cm_data.get("true_negatives_legit_correct", 0)
            fp = cm_data.get("false_positives_legit_as_phish", 0)
            fn = cm_data.get("false_negatives_phish_missed", 0)
            tp = cm_data.get("true_positives_phish_caught", 0)

            st.markdown(f"""
            - **True Negatives (Legitimate Correct):** `{tn}`
            - **False Positives (Legitimate Flagged as Phish):** `{fp}`
            - **False Negatives (Phishing Missed):** `{fn}`
            - **True Positives (Phishing Caught):** `{tp}`
            """)

        with col_ev2:
            st.markdown("#### 📋 Detailed Per-Class Classification Report")
            per_class = ev.get("per_class_report", {})
            if "Legitimate" in per_class and "Phishing" in per_class:
                report_rows = [
                    {
                        "Class": "Legitimate (0)",
                        "Precision": f"{per_class['Legitimate']['precision']*100:.1f}%",
                        "Recall": f"{per_class['Legitimate']['recall']*100:.1f}%",
                        "F1-Score": f"{per_class['Legitimate']['f1-score']*100:.1f}%",
                        "Support": int(per_class['Legitimate']['support'])
                    },
                    {
                        "Class": "Phishing (1)",
                        "Precision": f"{per_class['Phishing']['precision']*100:.1f}%",
                        "Recall": f"{per_class['Phishing']['recall']*100:.1f}%",
                        "F1-Score": f"{per_class['Phishing']['f1-score']*100:.1f}%",
                        "Support": int(per_class['Phishing']['support'])
                    }
                ]
                st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

            st.markdown(f"**Overall Accuracy:** `{ev.get('accuracy', 0)*100:.2f}%`")
            st.markdown(f"**Macro F1-Score:** `{ev.get('f1_macro', 0)*100:.2f}%`")

        st.markdown("---")
        st.markdown("#### 🧠 Explainability: Top TF-IDF Weighted Features")
        expl = meta.get("explainability", {})
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown("**Top Indicators Driving Phishing Classification (+)**")
            phish_feats = expl.get("top_phishing_indicators_tfidf", [])[:10]
            if phish_feats:
                st.dataframe(pd.DataFrame(phish_feats), use_container_width=True, hide_index=True)

        with col_exp2:
            st.markdown("**Top Indicators Driving Legitimate Classification (-)**")
            legit_feats = expl.get("top_legitimate_indicators_tfidf", [])[:10]
            if legit_feats:
                st.dataframe(pd.DataFrame(legit_feats), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### ⚠️ Limitations, Ethics, and False Positives/Negatives Analysis")
        limitations = meta.get("limitations_and_ethics", {}).get("known_limitations", [])
        for lim in limitations:
            st.markdown(f"- ⚠️ {lim}")

    else:
        st.warning("Model evaluation metadata unavailable. Run `python -m src.train_model` to generate evaluation results.")


# =========================================================
# TAB 5: DEMO SCENARIOS
# =========================================================
with tab_scenarios:
    st.markdown("### 🎯 Harmless Demonstration Scenarios")
    st.caption("Select a scenario to populate the input fields in the Email Analyzer tab. All scenarios run real model inference when analyzed.")

    scenario_name = st.selectbox(
        "Choose a Competition Demonstration Scenario:",
        list(DEMO_SCENARIOS.keys())
    )

    selected = DEMO_SCENARIOS[scenario_name]
    st.markdown(f"**Scenario Profile:** {selected['desc']}")

    if scenario_name != "Select a pre-configured scenario...":
        col_sc1, col_sc2 = st.columns([1, 1])
        with col_sc1:
            st.markdown(f"**Subject:** `{selected['subject']}`")
            st.text_area("Body Preview", value=selected["body"], height=200, disabled=True)
        with col_sc2:
            st.markdown("#### 🚀 Load Into Analyzer")
            st.write(
                "Clicking below will copy this scenario into the Email Analyzer tab. "
                "You can then inspect or modify the text and click **Analyze Email**."
            )
            if st.button("📥 Load Scenario Into Analyzer", type="primary"):
                st.session_state["input_subject"] = selected["subject"]
                st.session_state["input_body"] = selected["body"]
                st.success("Scenario loaded! Switch to the **'📥 Email Analyzer'** tab to run inference.")
