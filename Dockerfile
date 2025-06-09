FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Pre-download models during build (optional - adds ~2GB to image)
# RUN python -c "from kokoro import KPipeline; KPipeline(lang_code='a'); KPipeline(lang_code='b')"

# Expose WebSocket port
EXPOSE 8000

# Run the handler
CMD ["python", "-u", "handler.py"]






