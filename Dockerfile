FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/cryptobot_v3

COPY requirements.txt .

# CPU-only torch — avoids 2+ GB GPU download
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/
COPY dashboard/ ./dashboard/
COPY config_spot.yaml config_futures.yaml ./

RUN mkdir -p data logs
