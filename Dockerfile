# PhishGuard AI - Secure Docker Container Specification
FROM python:3.12-slim

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Create unprivileged security user
RUN groupadd -r phishguard && useradd -r -g phishguard -d /app -s /sbin/nologin phishguard

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and configuration
COPY . .

# Ensure app directories exist and configure non-root permissions
RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R phishguard:phishguard /app

# Switch to non-root user
USER phishguard

# Container internal network exposure
EXPOSE 8501

# Health check using lightweight native python healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD python -m src.healthcheck || exit 1

# Launch Streamlit bound to container interface
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
