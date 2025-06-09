# Base image from RunPod with Python
FROM runpod/base:0.4.0-cuda11.8.0

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget \
    espeak-ng \
    libespeak-ng1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlinks
RUN ln -s /usr/bin/python3 /usr/bin/python

# Clone the repository
WORKDIR /app
RUN git clone https://github.com/remsky/Kokoro-FastAPI.git .

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Create virtual environment and install dependencies
RUN uv venv
RUN . .venv/bin/activate && uv sync

# Download the model files
RUN . .venv/bin/activate && python docker/scripts/download_model.py --output api/src/models/v1_0

# Copy RunPod handler
COPY handler.py /app/handler.py

# Install RunPod Python SDK
RUN . .venv/bin/activate && pip install runpod

# Set environment variables
ENV PYTHONPATH=/app
ENV DEVICE=cuda
ENV HOST=0.0.0.0
ENV PORT=8880
ENV PATH="/app/.venv/bin:${PATH}"

# RunPod handler as entrypoint
CMD ["/app/.venv/bin/python", "-u", "handler.py"]