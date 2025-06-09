FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    espeak-ng \
    espeak-ng-data \
    git \
    libsndfile1 \
    curl \
    ffmpeg \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/

# Install UV using the installer script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    mv /root/.local/bin/uvx /usr/local/bin/

# Set working directory
WORKDIR /app

# Clone Kokoro-FastAPI repository
RUN git clone https://github.com/remsky/Kokoro-FastAPI.git kokoro-fastapi

# Set working directory to the cloned repo
WORKDIR /app/kokoro-fastapi

# Set environment variables for phonemizer
ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data

# Install dependencies with GPU extras using UV
RUN uv venv --python 3.10 && \
    uv sync --extra gpu

# Download the model
RUN /app/kokoro-fastapi/.venv/bin/python docker/scripts/download_model.py --output api/src/models/v1_0

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/kokoro-fastapi:/app/kokoro-fastapi/api \
    PATH="/app/kokoro-fastapi/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    USE_GPU=true \
    DEVICE=gpu

# Install runpod
RUN /app/kokoro-fastapi/.venv/bin/pip install runpod

# Copy handler to app root
WORKDIR /app
COPY handler.py ./

# Run the handler
CMD ["python", "-u", "/app/handler.py"]