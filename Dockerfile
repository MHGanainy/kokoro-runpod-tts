# Base image from RunPod
FROM runpod/base:0.4.0-cuda11.8.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    espeak-ng \
    libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository
WORKDIR /app
RUN git clone https://github.com/remsky/Kokoro-FastAPI.git .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install uv

# Install the project dependencies using UV
RUN uv sync

# Download the model files
RUN python docker/scripts/download_model.py --output api/src/models/v1_0

# Copy RunPod handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONPATH=/app
ENV DEVICE=cuda
ENV HOST=0.0.0.0
ENV PORT=8000

# RunPod handler as entrypoint
CMD ["python", "-u", "handler.py"]