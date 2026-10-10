"""Lightweight health check script for PhishGuard AI container and process monitoring."""

import sys
import urllib.request

HEALTH_URL = "http://127.0.0.1:8501/_stcore/health"


def check_health(url: str = HEALTH_URL, timeout_sec: int = 5) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PhishGuard-HealthCheck/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status == 200:
                print("HEALTHCHECK OK: Streamlit application is responding (HTTP 200).")
                return True
            else:
                print(f"HEALTHCHECK WARN: Unexpected status {resp.status}")
                return False
    except Exception as e:
        print(f"HEALTHCHECK FAILED: {e}")
        return False


if __name__ == "__main__":
    is_healthy = check_health()
    sys.exit(0 if is_healthy else 1)
