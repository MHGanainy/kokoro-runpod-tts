"""
Runpod Serverless Handler for Kokoro TTS
"""
import runpod
import io
import base64
import sys
import os
import json
from typing import Dict, Any, Optional

# Setup paths
KOKORO_PATH = '/app/kokoro-fastapi'
API_PATH = os.path.join(KOKORO_PATH, 'api')

# Add paths to sys.path
for path in [KOKORO_PATH, API_PATH]:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added to sys.path: {path}")

# Debug info
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

try:
    # Try different import approaches
    try:
        from src.core.tts import TTSService
        from src.core.models import ModelManager
        from src.core.config import settings
        from src.routers.v1.audio.speech import SpeechRequest, ResponseFormat
        print("Successfully imported from src.*")
    except ImportError as e:
        print(f"Failed to import from src.*: {e}")
        # Try alternative import
        from api.src.core.tts import TTSService
        from api.src.core.models import ModelManager
        from api.src.core.config import settings
        from api.src.routers.v1.audio.speech import SpeechRequest, ResponseFormat
        print("Successfully imported from api.src.*")
except ImportError as e:
    print(f"Import error: {e}")
    print("Directory structure:")
    for root, dirs, files in os.walk('/app'):
        level = root.replace('/app', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Limit files shown
            if file.endswith('.py'):
                print(f"{subindent}{file}")
    raise

# Initialize services globally
model_manager = None
tts_service = None


def init_services():
    """Initialize the TTS services"""
    global model_manager, tts_service
    
    print("Initializing Kokoro TTS services...")
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        
        # Initialize TTS service
        tts_service = TTSService(model_manager)
        
        print("Services initialized successfully!")
    except Exception as e:
        print(f"Error initializing services: {e}")
        import traceback
        traceback.print_exc()
        raise


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runpod handler function
    
    Expected input format:
    {
        "input": {
            "text": "Hello world!",
            "voice": "af_bella",  # or voice combinations like "af_sky+af_bella"
            "response_format": "mp3",  # mp3, wav, flac, pcm, etc.
            "speed": 1.0,  # optional, default 1.0
        }
    }
    """
    try:
        # Get input from event
        job_input = event.get("input", {})
        
        # Extract parameters
        text = job_input.get("text", job_input.get("input", ""))
        voice = job_input.get("voice", "af_bella")
        response_format = job_input.get("response_format", "mp3")
        speed = job_input.get("speed", 1.0)
        
        if not text:
            return {
                "error": "No text provided"
            }
        
        print(f"Processing TTS request: text='{text[:50]}...', voice={voice}, format={response_format}")
        
        # Create speech request
        speech_request = SpeechRequest(
            input=text,
            voice=voice,
            response_format=ResponseFormat(response_format),
            speed=speed
        )
        
        # Generate audio
        audio_generator = tts_service.generate_speech_streaming(
            text=speech_request.input,
            voice=speech_request.voice,
            speed=speech_request.speed,
            response_format=speech_request.response_format
        )
        
        # Collect all audio chunks
        audio_chunks = []
        for chunk in audio_generator:
            audio_chunks.append(chunk)
        
        # Combine chunks
        audio_data = b''.join(audio_chunks)
        
        # Encode audio data as base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        print(f"Generated audio: {len(audio_data)} bytes")
        
        return {
            "audio_base64": audio_base64,
            "format": response_format,
            "voice": voice,
            "text_length": len(text),
            "audio_size_bytes": len(audio_data)
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    # Initialize services on startup
    try:
        init_services()
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        sys.exit(1)
    
    # Start the Runpod serverless handler
    print("Starting Runpod serverless handler...")
    runpod.serverless.start({
        "handler": handler
    })