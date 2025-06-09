import runpod
import torch
import base64
import io
import soundfile as sf
import sys
import os

# Add Kokoro to path
sys.path.append('/app/kokoro/api/src')

# Import what we need from Kokoro
from models import Models

# Global model
model = None

def init_model():
    """Initialize the model once"""
    global model
    if model is None:
        print("Loading Kokoro model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model = Models(
            device=device,
            model_path="/models/kokoro/kokoro-v1_0.pt"
        )
        print("Model loaded!")
    return model

def handler(job):
    """
    Simple handler that converts text to speech
    
    Input:
    {
        "input": {
            "text": "Hello world",
            "voice": "af_bella"  # optional, defaults to af_bella
        }
    }
    
    Output:
    {
        "audio_base64": "...",  # WAV audio encoded in base64
    }
    """
    try:
        # Get inputs
        text = job['input'].get('text', '')
        voice = job['input'].get('voice', 'af_bella')
        
        if not text:
            return {"error": "No text provided"}
        
        # Initialize model
        model = init_model()
        
        # Load voice
        voice_path = f"/models/kokoro/voices/{voice}.pt"
        if not os.path.exists(voice_path):
            voice_path = "/models/kokoro/voices/af_bella.pt"  # fallback
        
        # Generate speech
        print(f"Generating: {text[:50]}...")
        with torch.no_grad():
            audio, _ = model.synthesize(
                text=text,
                voicepack=voice_path,
                speed=1.0
            )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, 24000, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        
        # Return base64 encoded audio
        return {
            "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
            "sample_rate": 24000,
            "format": "wav"
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Initialize on startup
init_model()

# Start RunPod serverless
runpod.serverless.start({"handler": handler})