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

# Add the api directory to Python path
sys.path.append('/app')
sys.path.append('/app/api')

# Import FastAPI app components
from api.src.core.tts import TTSService
from api.src.core.models import ModelManager
from api.src.core.config import settings
from api.src.routers.v1.audio.speech import SpeechRequest, ResponseFormat

# Initialize services globally
model_manager = None
tts_service = None


def init_services():
    """Initialize the TTS services"""
    global model_manager, tts_service
    
    print("Initializing Kokoro TTS services...")
    
    # Initialize model manager
    model_manager = ModelManager()
    
    # Initialize TTS service
    tts_service = TTSService(model_manager)
    
    print("Services initialized successfully!")


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
            "stream": false  # for now, we'll return complete audio
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
            "error": str(e)
        }


if __name__ == "__main__":
    # Initialize services on startup
    init_services()
    
    # Start the Runpod serverless handler
    runpod.serverless.start({
        "handler": handler
    })