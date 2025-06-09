FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone Kokoro repo
RUN git clone https://github.com/remsky/Kokoro-FastAPI.git kokoro

# Install Python packages
RUN pip install --no-cache-dir \
    runpod \
    requests \
    soundfile \
    numpy \
    scipy

# Download model and a voice file
RUN mkdir -p /models/kokoro/voices && \
    wget -O /models/kokoro/kokoro-v1_0.pt https://huggingface.co/remsky/kokoro-v1_0/resolve/main/kokoro-v1_0.pt && \
    wget -O /models/kokoro/voices/af_bella.pt https://huggingface.co/remsky/kokoro-v1_0/resolve/main/voices/af_bella.pt && \
    wget -O /models/kokoro/voices/af_sky.pt https://huggingface.co/remsky/kokoro-v1_0/resolve/main/voices/af_sky.pt

# Copy the minimal handler
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]