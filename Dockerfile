# Multi-stage Dockerfile for DubYou Enterprise
# Production-ready with optimizations

# Stage 1: Base image with system dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 dubyou && \
    mkdir -p /app /data/audio /data/embeddings /var/log/dubyou && \
    chown -R dubyou:dubyou /app /data /var/log/dubyou

# Stage 2: Python dependencies
FROM base AS dependencies

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 cache purge

# Stage 3: Application
FROM base AS application

WORKDIR /app

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=dubyou:dubyou . .

# Create necessary directories
RUN mkdir -p models/speaker_encoder models/whisper models/nllb models/tts

# Switch to non-root user
USER dubyou

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["python3", "-m", "uvicorn", "main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
