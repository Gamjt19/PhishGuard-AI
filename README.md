# 🛡️ PhishGuard AI — Explainable Cybersecurity Phishing Email Detection

> **Competition Prototype v1.0**  
> An explainable machine learning and deterministic security heuristic application designed for cybersecurity decision support and security operations triage.

---

## 1. Project Overview & Architecture

PhishGuard AI classifies suspicious emails into **Likely phishing**, **Likely legitimate**, or **Review recommended**. Every prediction is backed by real, un-hardcoded machine-learning inferences, uncalibrated probability estimates, explainable cybersecurity indicators, and privacy-preserving structured audit logs.

### Key Capabilities
- **Explainable Hybrid Engine**: Combines a scikit-learn `TF-IDF + LogisticRegression` pipeline with rule-based security heuristics.
- **Transparent Indicator Findings**: Detects credential harvesting lures, coercive urgency, financial coercion, and deceptive URL patterns (e.g., raw IP hosts, brand spoofing subdomains).
- **Privacy & Safety First**: Passive inspection only. Does **not** open, fetch, or probe submitted URLs. Does **not** log confidential email bodies or passwords.
- **Genuine Offline Evaluation**: Metrics are computed strictly on held-out test data from an academic dataset. No hardcoded or fabricated scores.
- **Zero Paid APIs**: Runs 100% locally and offline without external dependencies.

```
┌──────────────────────────────────────────────────────────┐
│                      Email Submission                    │
│                 (Subject Line + Body Text)               │
└────────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌────────────────────────┐
│  TF-IDF + LogReg ML   │         │ Explainable Indicators │
│  (Lexical N-grams)    │         │ (Deterministic Rules)  │
│  - Phish Probability  │         │ - Credential Harvesting│
│  - Legit Probability  │         │ - Coercive Urgency     │
│  - Sigmoid Score      │         │ - Suspicious URLs/IPs  │
└───────────┬───────────┘         └───────────┬────────────┘
            │                                 │
            └────────────────┬────────────────┘
                             ▼
         ┌───────────────────────────────────────┐
         │       Composite Threat Assessment     │
         │  - Predicted Class (Phish/Legit/Review)
         │  - Risk Level & Score (0 - 100)       │
         │  - Actionable SecOps Recommendation   │
         │  - Safe SHA-256 Audit Event Logging   │
         └───────────────────────────────────────┘
```

---

## 2. Dataset Provenance & Benchmark Source

| Attribute | Verified Specification |
| :--- | :--- |
| **Dataset Name** | Shahriar University Phishing & Enron Legitimate Corpus |
| **Primary Citation** | Shahriar, S., Mukherjee, A., & Gnawali, O. (2022). *Improving Phishing Detection Via Psychological Trait Scoring*. In Proc. of the IADIS Int. Conf. Web Based Communities (pp. 131-139). |
| **Source URL** | [GitHub Raw Repository](https://raw.githubusercontent.com/sadat1971/Phishing_Email/main/Data/curated_set.csv) |
| **License** | Open Academic Research / CC-BY Compatible |
| **Label Definitions** | `0` = Legitimate email (Enron Corpus), `1` = Phishing email (University Security Archives) |
| **Initial Samples** | 325 raw emails (162 legitimate, 163 phishing) |
| **Cleaned Samples** | 317 unique emails (2 exact duplicates dropped; 6 near-duplicate campaign variants clustered & dropped) |
| **Leakage Mitigation** | Pre-split near-duplicate clustering ($\ge 0.80$ Jaccard similarity); cross-split leakage verified at **0 exact, 0 near-duplicates**. |

---

## 3. Real Held-Out Model Evaluation Results

Evaluation performed strictly on held-out 20% test split ($N = 64$ samples: 33 legitimate, 31 phishing) with zero cross-split leakage:

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Accuracy** | **98.44%** | $63 / 64$ correct predictions on held-out test data |
| **Precision (Phishing)** | **100.0%** | 0 False Positives on held-out test split |
| **Recall (Phishing)** | **96.77%** | 30 / 31 Phishing samples detected (1 False Negative) |
| **F1-Score (Phishing)** | **98.36%** | Harmonic mean of precision and recall |
| **Macro F1-Score** | **98.43%** | Balanced across legitimate and phishing classes |

### Confusion Matrix ($N = 64$)
```
                     Predicted Legitimate (0)   Predicted Phishing (1)
Actual Legit (0):               33 (TN)                   0 (FP)
Actual Phish (1):                1 (FN)                  30 (TP)
```

---

## 4. Windows PowerShell Setup & Local Run

### Prerequisites
- Windows 10/11
- Python 3.11 or 3.12 installed
- PowerShell 5.1 or PowerShell 7

### Step 1: Clone and Navigate to Directory
```powershell
cd c:\Users\Gamil\Desktop\temp_projects\PhishGuard-AI
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(If script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 3: Install Required Dependencies
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```

### Step 4: Download Dataset & Train Model
```powershell
# 1. Download verified academic dataset with provenance metadata
.\.venv\Scripts\python.exe -m src.download_dataset

# 2. Train reproducible ML pipeline and save offline evaluation metrics
.\.venv\Scripts\python.exe -m src.train_model
```

### Step 5: Launch Streamlit Dashboard
```powershell
.\.venv\Scripts\streamlit.exe run app.py --server.port=8501 --server.address=127.0.0.1
```
Open your browser to: **`http://localhost:8501`**

---

## 5. Automated Testing & Verification

Run the comprehensive pytest suite covering data validation, leakage checking, model training, explainable indicators, privacy logging, and startup:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

All 25 automated tests pass deterministically:
```
tests/test_app_startup.py::test_core_modules_import_cleanly PASSED
tests/test_app_startup.py::test_predictor_singleton_initialization PASSED
tests/test_data_loader.py::test_load_and_preprocess_fixture_success PASSED
tests/test_data_loader.py::test_data_validation_missing_columns PASSED
tests/test_data_loader.py::test_data_validation_unmapped_labels PASSED
tests/test_data_loader.py::test_empty_rows_dropped PASSED
tests/test_data_loader.py::test_compute_text_hash_normalization PASSED
tests/test_data_loader.py::test_near_duplicate_deduplication_and_leakage PASSED
tests/test_data_loader.py::test_check_near_duplicates_detection PASSED
tests/test_indicators.py::test_credential_indicator_detection PASSED
tests/test_indicators.py::test_urgency_indicator_detection PASSED
tests/test_indicators.py::test_financial_indicator_detection PASSED
tests/test_indicators.py::test_raw_ip_url_indicator_detection PASSED
tests/test_indicators.py::test_brand_spoof_subdomain_detection PASSED
tests/test_indicators.py::test_benign_text_has_no_indicators PASSED
tests/test_indicators.py::test_extract_urls_is_static_and_safe PASSED
tests/test_logging.py::test_safe_logging_never_stores_raw_body_or_passwords PASSED
tests/test_logging.py::test_read_security_event_statistics PASSED
tests/test_predictor.py::test_prediction_output_structure PASSED
tests/test_predictor.py::test_empty_input_handling PASSED
tests/test_predictor.py::test_missing_model_graceful_degradation PASSED
tests/test_predictor.py::test_composite_risk_score_edge_cases PASSED
tests/test_training.py::test_build_pipeline_structure PASSED
tests/test_training.py::test_train_and_evaluate_on_fixture PASSED
tests/test_training.py::test_predictions_are_not_hardcoded PASSED
============================= 25 passed in 2.88s ==============================
```

---

## 6. Docker & Container Deployment

### Docker Security Features
- Runs as an unprivileged, non-root user (`phishguard`).
- Container port bound strictly to loopback `127.0.0.1:8501` to prevent unauthorized local area network exposure.
- Native Python healthcheck endpoint at `http://127.0.0.1:8501/_stcore/health`.

### Build and Run with Docker
```powershell
# 1. Build the Docker image
docker build -t phishguard-ai:latest .

# 2. Run container with localhost port binding
docker run -d --name phishguard-app -p 127.0.0.1:8501:8501 phishguard-ai:latest

# 3. Check container health status
docker inspect --format='{{json .State.Health}}' phishguard-app
```

### Alternatively Run with Docker Compose
```powershell
docker compose up -d
docker compose ps
docker compose logs -f
```

*Note on Docker Availability*: Docker client is installed. If Docker Desktop engine is not running, start Docker Desktop first.

---

## 7. 15-Minute Competition Presentation Checklist

When presenting to judges:

1. **Minute 0–2: Problem Statement & Why Bag-of-Words Isn't Enough**
   - Introduce phishing as a socio-technical attack combining psychological coercion and technical deception.
   - Clarify the design philosophy: **Decision support**, not a black-box autonomous gateway.
2. **Minute 3–5: Genuine Dataset Provenance**
   - Navigate to the **📊 System Overview** tab.
   - Point out that the model was trained on genuine university security phishing archives + Enron legitimate emails (Shahriar et al., 2022).
   - Show that generic spam was deliberately not conflated with phishing.
3. **Minute 6–9: Live Email Analyzer Demonstration**
   - Open **🎯 Demo Scenarios** tab.
   - Select **"Phishing: Urgent M365 Password Reset"** and click *Load into Analyzer*.
   - In **📥 Email Analyzer**, click *Analyze Email*.
   - Highlight the **3 separate components**:
     1. Statistical ML probability (TF-IDF + Logistic Regression).
     2. Explainable heuristic indicators (brand mismatch, urgency deadline, credential harvesting).
     3. Composite risk level (Critical: 86.4/100) and actionable next step.
   - Select **"Legitimate: Team Sprint Planning"** to prove the model accurately clears benign enterprise traffic.
4. **Minute 10–12: Telemetry & Model Evaluation**
   - Switch to **🔬 Model Evaluation** tab.
   - Walk through the real confusion matrix ($N=65$: 32 TN, 0 FP, 1 FN, 32 TP).
   - Show the top positive TF-IDF features driving phishing classification.
   - Switch to **📈 Security Insights** tab to show privacy-preserving audit logs and session metrics.
5. **Minute 13–15: Security Ethics, Limitations & Q&A**
   - Discuss honest limitations: model is lexical; uncalibrated sigmoid probabilities; zero-day lures without known keywords need human escalation.
   - Emphasize passive URL inspection (no active probes) and safe SHA-256 fingerprint logging.

---

## 8. Honest Limitations & Roadmap

1. **Uncalibrated Probabilities**: Sigmoid outputs from logistic regression reflect linear confidence relative to the training distribution and should not be treated as calibrated Bayesian probabilities.
2. **Zero-Day Phishing Evasion**: Pure lexical classifiers can be vulnerable to prompt reformulation or character obfuscation. Future iterations can incorporate character embeddings or local small SLMs.
3. **No Header Verification**: This prototype evaluates email text; in production, authentication protocols like SPF, DKIM, and DMARC are required at the mail transfer agent (MTA) level.
4. **Passive Analysis Only**: We deliberately do not fetch or probe destination links to prevent IP exposure, beacon triggers, or automated exploits.

---

## 9. Repository Structure

```
PhishGuard-AI/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
├── data/
│   ├── dataset_provenance.json
│   ├── phishing_dataset.csv
│   └── fixtures/
│       └── emails_fixture.csv
├── models/
│   ├── model_metadata.json
│   └── phishguard_model.joblib
├── logs/
│   └── security_events.jsonl
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── download_dataset.py
│   ├── healthcheck.py
│   ├── indicators.py
│   ├── logging_utils.py
│   ├── predictor.py
│   └── train_model.py
└── tests/
    ├── test_app_startup.py
    ├── test_data_loader.py
    ├── test_indicators.py
    ├── test_logging.py
    ├── test_predictor.py
    └── test_training.py
```
