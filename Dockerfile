FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# GPU optimization environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with GPU optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Optional: Pre-download models during build for faster cold starts
# This adds ~2GB to image but eliminates model download latency
RUN python -c "from kokoro import KPipeline; print('Pre-loading models...'); KPipeline(lang_code='a'); KPipeline(lang_code='b'); print('Models pre-loaded')" || echo "Model pre-loading failed, will load at runtime"

# Warm up GPU and PyTorch
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Create non-root user for security (optional)
# RUN groupadd -r kokoro && useradd -r -g kokoro kokoro
# USER kokoro

# Expose WebSocket port
EXPOSE 8000

# Health check to ensure GPU is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || exit 1

# Run the handler with optimizations
CMD ["python", "-u", "handler.py"]