FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set non-interactive frontend
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    espeak-ng \
    espeak-ng-data \
    git \
    libsndfile1 \
    libsndfile1-dev \
    libffi-dev \
    curl \
    ffmpeg \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/

# Set working directory
WORKDIR /app

# Clone Kokoro-FastAPI repository
RUN git clone --depth 1 https://github.com/remsky/Kokoro-FastAPI.git kokoro-fastapi

# Create virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools && \
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt

# Install any missing Kokoro-FastAPI dependencies
RUN cd kokoro-fastapi && \
    pip install phonemizer loguru pydantic-settings || true

# Download the model
RUN cd kokoro-fastapi && \
    python docker/scripts/download_model.py --output api/src/models/v1_0

# Set environment variables
ENV PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/kokoro-fastapi:/app/kokoro-fastapi/api:/app \
    USE_GPU=true \
    DEVICE=gpu \
    CUDA_VISIBLE_DEVICES=0

# Copy handler and test script
COPY handler.py .
COPY test_imports.py .

# Run the handler
CMD ["python", "-u", "/app/handler.py"]