# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System-Pakete fuer numpy/xgboost/matplotlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencies separat -> Layer-Cache
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2) Anwendung
COPY app.py .
COPY .streamlit/ ./.streamlit/
COPY data/ ./data/
COPY ["pipeline", "./pipeline"]

# 3) Beim Start: Random-Forest-Pipeline ausfuehren (falls noch nicht
#    gelaufen) und dann Streamlit starten. Die EDA-Skripte (01-11) sind
#    optional und werden hier nicht automatisch ausgefuehrt - die
#    erzeugten Charts liegen bereits im Repo.
WORKDIR /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=3).status==200 else 1)"

ENTRYPOINT ["/entrypoint.sh"]
