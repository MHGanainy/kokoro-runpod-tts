x#!/usr/bin/env python3
"""
Test script to debug Kokoro-FastAPI imports
Run this inside the Docker container to see what's available
"""
import os
import sys

print("=== Environment Info ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

print("\n=== Directory Structure ===")
# Check if kokoro-fastapi exists
if os.path.exists('/app/kokoro-fastapi'):
    print("Found /app/kokoro-fastapi")
    for item in os.listdir('/app/kokoro-fastapi'):
        print(f"  - {item}")
        if item == 'api' and os.path.isdir(f'/app/kokoro-fastapi/{item}'):
            for subitem in os.listdir(f'/app/kokoro-fastapi/{item}'):
                print(f"    - {subitem}")
                if subitem == 'src' and os.path.isdir(f'/app/kokoro-fastapi/{item}/{subitem}'):
                    for subsubitem in os.listdir(f'/app/kokoro-fastapi/{item}/{subitem}'):
                        print(f"      - {subsubitem}")

print("\n=== Attempting Imports ===")

# Add paths
sys.path.insert(0, '/app/kokoro-fastapi')
sys.path.insert(0, '/app/kokoro-fastapi/api')

# Try various import patterns
import_attempts = [
    "from src.core.tts import TTSService",
    "from api.src.core.tts import TTSService",
    "from core.tts import TTSService",
    "from kokoro_fastapi.api.src.core.tts import TTSService",
]

for attempt in import_attempts:
    try:
        exec(attempt)
        print(f"✓ Success: {attempt}")
        break
    except ImportError as e:
        print(f"✗ Failed: {attempt}")
        print(f"  Error: {e}")

print("\n=== Checking for __init__.py files ===")
init_paths = [
    '/app/kokoro-fastapi/__init__.py',
    '/app/kokoro-fastapi/api/__init__.py',
    '/app/kokoro-fastapi/api/src/__init__.py',
    '/app/kokoro-fastapi/api/src/core/__init__.py',
]

for path in init_paths:
    if os.path.exists(path):
        print(f"✓ Found: {path}")
    else:
        print(f"✗ Missing: {path}")

print("\n=== Package Installation Check ===")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("✗ PyTorch not installed")

try:
    import runpod
    print(f"✓ Runpod installed")
except ImportError:
    print("✗ Runpod not installed")

try:
    import phonemizer
    print(f"✓ Phonemizer installed")
except ImportError:
    print("✗ Phonemizer not installed")