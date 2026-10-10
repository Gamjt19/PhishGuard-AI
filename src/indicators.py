"""Explainable cybersecurity indicators module for PhishGuard AI.

Implements rule-based, deterministic heuristics to detect common social engineering
and phishing patterns in email text and URLs.

SAFETY NOTE:
- Strictly static text analysis.
- DOES NOT fetch, resolve, or open any submitted URLs or network hosts.
- DOES NOT claim that a URL is malicious merely because it is long or uses HTTPS.
- Plainly disclaims that all indicators are heuristics, not proof of maliciousness.
"""

from dataclasses import dataclass, asdict
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse


@dataclass
class SecurityIndicator:
    category: str
    indicator_id: str
    name: str
    severity: str  # "high", "medium", "low", "info"
    matched_patterns: List[str]
    description: str
    heuristic_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Regex patterns for heuristic indicator detection
CREDENTIAL_PATTERNS = [
    r"\b(?:enter|verify|update|confirm|reset|provide)\s+(?:your\s+)?(?:password|passcode|pin|credentials|login\s+details|secret\s+key)\b",
    r"\b(?:account|mailbox|profile|login)\s+(?:verification|validation|reactivation|confirmation)\b",
    r"\b(?:2fa|two-factor|security\s+code|one-time\s+passcode|otp)\b",
    r"\b(?:social\s+security\s+number|ssn|identity\s+theft|national\s+id)\b",
    r"\b(?:confirm|verify)\s+your\s+identity\b",
]

URGENCY_PATTERNS = [
    r"\b(?:within\s+(?:24|12|48|72)\s+hours|immediately|right\s+now|without\s+delay|at\s+once)\b",
    r"\b(?:account\s+(?:suspended|locked|terminated|deactivated|closed|frozen))\b",
    r"\b(?:immediate\s+action\s+required|urgent\s+notice|action\s+required|final\s+warning)\b",
    r"\b(?:avoid\s+(?:suspension|termination|penalty|legal\s+action|cancellation))\b",
    r"\b(?:unauthorized\s+(?:login|access|activity|transaction))\b",
]

FINANCIAL_PATTERNS = [
    r"\b(?:wire\s+transfer|direct\s+deposit|bank\s+routing|credit\s+card\s+number|debit\s+card\s+pin)\b",
    r"\b(?:gift\s+card|bitcoin|crypto|usdt|ethereum|western\s+union|moneygram)\b",
    r"\b(?:unpaid\s+invoice|overdue\s+payment|past\s+due|refund\s+pending|claim\s+your\s+prize|won\s+\$?\d+)\b",
]

ATTACHMENT_PATTERNS = [
    r"\b(?:attached|see\s+attached|view\s+attachment|download\s+(?:the\s+)?attached|open\s+the\s+attachment)\b",
    r"\b(?:invoice\.(?:exe|scr|vbs|zip|iso|pdf|xlsm)|remittance\.(?:pdf|zip)|payment_receipt\.(?:zip|html))\b",
]

URL_REGEX = re.compile(
    r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
)

# Known legitimate brand keywords often targeted by homograph or typo squatted subdomains
COMMON_TARGET_BRANDS = [
    "paypal", "microsoft", "office365", "google", "apple", "amazon",
    "chase", "wellsfargo", "bankofamerica", "netflix", "dropbox", "docusign"
]

SUSPICIOUS_TLDS = {
    "xyz", "top", "work", "click", "cc", "ru", "cn", "biz", "country", "stream"
}


def extract_urls(text: str) -> List[str]:
    """Safely extract URL strings from text without performing any network requests."""
    matches = URL_REGEX.findall(text)
    urls = []
    for match in matches:
        raw_url = match[0] if isinstance(match, tuple) else match
        if not raw_url.startswith(("http://", "https://")):
            raw_url = "http://" + raw_url
        urls.append(raw_url)
    return list(dict.fromkeys(urls))  # Deduplicate while preserving order


def analyze_security_indicators(subject: str, body: str) -> List[SecurityIndicator]:
    """
    Scan email subject and body for explainable cybersecurity indicators.

    Returns a structured list of SecurityIndicator objects.
    """
    combined_text = f"{subject}\n{body}".strip()
    indicators: List[SecurityIndicator] = []

    # 1. Credential & Authentication Harvesting Heuristic
    cred_matches = []
    for pat in CREDENTIAL_PATTERNS:
        found = re.findall(pat, combined_text, flags=re.IGNORECASE)
        cred_matches.extend(found)
    if cred_matches:
        indicators.append(SecurityIndicator(
            category="Credential Harvesting",
            indicator_id="IND_CREDENTIALS",
            name="Request for Passwords or Identity Verification",
            severity="high",
            matched_patterns=list(set(cred_matches))[:4],
            description="The text prompts the recipient to enter, verify, or reset sensitive credentials or identity info.",
            heuristic_rationale=(
                "Phishing campaigns frequently impersonate IT helpdesks, banks, or services to coerce users "
                "into entering credentials into credential-harvesting landing pages."
            )
        ))

    # 2. Urgency and Coercive Pressure Heuristic
    urgency_matches = []
    for pat in URGENCY_PATTERNS:
        found = re.findall(pat, combined_text, flags=re.IGNORECASE)
        urgency_matches.extend(found)
    if urgency_matches:
        indicators.append(SecurityIndicator(
            category="Psychological Coercion",
            indicator_id="IND_URGENCY",
            name="Artificial Urgency or Coercive Pressure",
            severity="medium",
            matched_patterns=list(set(urgency_matches))[:4],
            description="Detected manufactured time pressure (e.g., 'within 24 hours') or threat of negative consequence.",
            heuristic_rationale=(
                "Attackers use artificial urgency to induce panic and bypass the victim's critical scrutiny before "
                "they can verify the message through an independent channel."
            )
        ))

    # 3. Financial Coercion & Prize / Wire Requests
    fin_matches = []
    for pat in FINANCIAL_PATTERNS:
        found = re.findall(pat, combined_text, flags=re.IGNORECASE)
        fin_matches.extend(found)
    if fin_matches:
        indicators.append(SecurityIndicator(
            category="Financial Coercion",
            indicator_id="IND_FINANCIAL",
            name="Unusual Payment, Wire, or Prize Reference",
            severity="medium",
            matched_patterns=list(set(fin_matches))[:4],
            description="Contains explicit demands for wire transfers, gift cards, overdue penalties, or unsolicited prizes.",
            heuristic_rationale=(
                "Scammers commonly leverage fake invoices, overdue penalties, or lottery winnings to manipulate "
                "victims into authorizing payments or surrendering financial information."
            )
        ))

    # 4. Attachment / Malicious Payload References
    att_matches = []
    for pat in ATTACHMENT_PATTERNS:
        found = re.findall(pat, combined_text, flags=re.IGNORECASE)
        att_matches.extend(found)
    if att_matches:
        indicators.append(SecurityIndicator(
            category="Payload Delivery",
            indicator_id="IND_ATTACHMENT",
            name="Explicit Reference to External Attachments",
            severity="low",
            matched_patterns=list(set(att_matches))[:4],
            description="The message directs the recipient to open or execute an attached document or archive.",
            heuristic_rationale=(
                "Email attachments are a frequent vector for malware droppers (e.g., macro-enabled Office docs, scripts, or weaponized PDFs)."
            )
        ))

    # 5. URL Heuristics (Static Lexical Inspection Only - No Network Calls)
    urls = extract_urls(combined_text)
    if urls:
        ip_urls = []
        brand_spoof_urls = []
        suspicious_tld_urls = []

        for u in urls:
            try:
                parsed = urlparse(u)
                hostname = (parsed.hostname or "").lower()
                
                # Check for raw IP address hostname (e.g., http://192.168.1.100/login)
                if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
                    ip_urls.append(hostname)

                # Check for Brand Keyword in Subdomain rather than Root Domain
                # e.g., login.microsoft365.verify-security.ru -> brand 'microsoft' used as lure
                parts = hostname.split(".")
                root_domain = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
                for brand in COMMON_TARGET_BRANDS:
                    if brand in hostname and brand not in root_domain:
                        brand_spoof_urls.append(f"{hostname} (brand keyword '{brand}' found outside root domain '{root_domain}')")
                        break

                # Check for high-abuse TLDs
                if len(parts) >= 2 and parts[-1] in SUSPICIOUS_TLDS:
                    suspicious_tld_urls.append(f"{hostname} (TLD .{parts[-1]})")
            except Exception:
                continue

        if ip_urls:
            indicators.append(SecurityIndicator(
                category="Suspicious URL",
                indicator_id="IND_URL_IP",
                name="Direct IP Address Used in Hyperlink Hostname",
                severity="high",
                matched_patterns=ip_urls[:3],
                description="Hyperlink uses a numerical IP address instead of a registered domain name.",
                heuristic_rationale=(
                    "Legitimate organizations almost universally employ branded domain names. Direct IP links "
                    "are commonly used to evade domain reputation filters."
                )
            ))

        if brand_spoof_urls:
            indicators.append(SecurityIndicator(
                category="Brand Impersonation",
                indicator_id="IND_URL_BRAND_MISMATCH",
                name="Misleading Subdomain / Brand Spoofing in URL",
                severity="high",
                matched_patterns=brand_spoof_urls[:3],
                description="A recognizable corporate brand keyword appears in a subdomain or host path, while the actual registered root domain is different.",
                heuristic_rationale=(
                    "Deceptive subdomains exploit visual similarity to make users believe they are navigating to a legitimate corporate portal."
                )
            ))

        if suspicious_tld_urls and not (ip_urls or brand_spoof_urls):
            indicators.append(SecurityIndicator(
                category="URL Characteristics",
                indicator_id="IND_URL_TLD",
                name="URL with Frequently Abused Top-Level Domain",
                severity="low",
                matched_patterns=suspicious_tld_urls[:3],
                description="The URL uses a generic TLD historically associated with low-cost spam registrations.",
                heuristic_rationale=(
                    "Certain TLDs offer free or bulk registrations commonly leveraged in disposable phishing campaigns. "
                    "Note: This is a weak heuristic and alone does not establish maliciousness."
                )
            ))

    return indicators
