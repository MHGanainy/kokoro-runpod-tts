FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Optimized environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=2
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Install system dependencies with optimizations
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy optimized handler
COPY handler.py .

# Pre-download and optimize models
RUN python -c "from kokoro import KPipeline; import torch; torch.set_float32_matmul_precision('medium'); print('Pre-loading optimized models...'); KPipeline(lang_code='a'); KPipeline(lang_code='b'); print('Optimized models pre-loaded')" || echo "Model pre-loading failed, will load at runtime"

# Warm up GPU with optimizations
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); torch.backends.cudnn.benchmark = True; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Create optimized user (optional)
# RUN groupadd -r kokoro && useradd -r -g kokoro kokoro
# USER kokoro

# Expose WebSocket port
EXPOSE 8000

# Optimized health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=2 \
    CMD python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" || exit 1

# Run with optimizations
CMD ["python", "-u", "handler.py"]