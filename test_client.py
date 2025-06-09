"""
Test client for Kokoro TTS on Runpod Serverless
Includes both direct API testing and OpenAI-compatible testing
"""
import os
import requests
import base64
import io
import time
from typing import Optional
import soundfile as sf
import numpy as np

# For OpenAI compatibility testing
from openai import OpenAI


class RunpodKokoroClient:
    """Client for testing Kokoro TTS on Runpod"""
    
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        
    def generate_speech(self, 
                       text: str, 
                       voice: str = "af_bella",
                       response_format: str = "mp3",
                       speed: float = 1.0) -> bytes:
        """Generate speech using Runpod API"""
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "text": text,
                "voice": voice,
                "response_format": response_format,
                "speed": speed
            }
        }
        
        # Submit job
        print(f"Submitting job to Runpod...")
        response = requests.post(
            f"{self.base_url}/runsync",
            json=payload,
            headers=headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to submit job: {response.text}")
        
        result = response.json()
        
        if result.get("status") == "COMPLETED":
            # Extract audio data
            output = result.get("output", {})
            audio_base64 = output.get("audio_base64")
            
            if not audio_base64:
                raise Exception("No audio data in response")
            
            # Decode base64 audio
            audio_data = base64.b64decode(audio_base64)
            
            print(f"Received audio: {len(audio_data)} bytes, format: {output.get('format')}")
            return audio_data
        else:
            raise Exception(f"Job failed: {result}")
    
    def save_audio(self, audio_data: bytes, filename: str):
        """Save audio data to file"""
        with open(filename, "wb") as f:
            f.write(audio_data)
        print(f"Audio saved to {filename}")


class OpenAICompatibleClient:
    """OpenAI-compatible client that connects to a proxy server"""
    
    def __init__(self, proxy_url: str = "http://localhost:8880"):
        """Initialize client pointing to local proxy server"""
        self.client = OpenAI(
            base_url=f"{proxy_url}/v1",
            api_key="not-needed"  # Proxy handles Runpod auth
        )
    
    def stream_to_file(self, text: str, voice: str = "af_bella", filename: str = "output.mp3"):
        """Stream audio to file using OpenAI client"""
        print(f"Streaming audio for: '{text}'")
        
        with self.client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice=voice,
            input=text
        ) as response:
            response.stream_to_file(filename)
        
        print(f"Audio streamed to {filename}")
    
    def stream_to_speakers(self, text: str, voice: str = "af_bella"):
        """Stream audio directly to speakers"""
        try:
            import pyaudio
        except ImportError:
            print("PyAudio not installed. Install with: pip install pyaudio")
            return
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )
        
        print(f"Streaming to speakers: '{text}'")
        
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model="kokoro",
                voice=voice,
                response_format="pcm",
                input=text
            ) as response:
                for chunk in response.iter_bytes(chunk_size=1024):
                    stream.write(chunk)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


def test_direct_api():
    """Test direct Runpod API"""
    print("=== Testing Direct Runpod API ===")
    
    # Get credentials from environment
    endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
    api_key = os.getenv("RUNPOD_API_KEY")
    
    if not endpoint_id or not api_key:
        print("Please set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables")
        return
    
    # Create client
    client = RunpodKokoroClient(endpoint_id, api_key)
    
    # Test cases
    test_cases = [
        {
            "text": "Hello world! This is a test of Kokoro TTS on Runpod.",
            "voice": "af_bella",
            "filename": "test_bella.mp3"
        },
        {
            "text": "Testing voice mixing with multiple speakers.",
            "voice": "af_sky+af_bella",
            "filename": "test_mixed.mp3"
        },
        {
            "text": "This is a WAV format test.",
            "voice": "af_bella",
            "response_format": "wav",
            "filename": "test_bella.wav"
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['text']}")
        try:
            audio_data = client.generate_speech(
                text=test["text"],
                voice=test["voice"],
                response_format=test.get("response_format", "mp3")
            )
            client.save_audio(audio_data, test["filename"])
        except Exception as e:
            print(f"Error: {e}")


def test_openai_compatible():
    """Test OpenAI-compatible interface"""
    print("\n=== Testing OpenAI-Compatible Interface ===")
    print("Note: This requires a local proxy server running on port 8880")
    print("The proxy server should forward requests to your Runpod endpoint")
    
    client = OpenAICompatibleClient()
    
    # Test streaming to file
    try:
        client.stream_to_file(
            "Hello from OpenAI-compatible client! This audio is being streamed.",
            voice="af_bella",
            filename="openai_test.mp3"
        )
    except Exception as e:
        print(f"Error streaming to file: {e}")
    
    # Test streaming to speakers (optional)
    try:
        response = input("\nStream to speakers? (y/n): ")
        if response.lower() == 'y':
            client.stream_to_speakers(
                "This audio is being streamed directly to your speakers!",
                voice="af_bella"
            )
    except Exception as e:
        print(f"Error streaming to speakers: {e}")


def create_proxy_server():
    """Create a simple proxy server script for OpenAI compatibility"""
    proxy_code = '''"""
Simple proxy server to make Runpod endpoint OpenAI-compatible
Run this locally to use OpenAI client with Runpod
"""
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
import requests
import os
import base64
import io

app = FastAPI()

RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    """Proxy OpenAI speech requests to Runpod"""
    
    # Parse request
    data = await request.json()
    
    # Map OpenAI parameters to Runpod format
    runpod_payload = {
        "input": {
            "text": data.get("input"),
            "voice": data.get("voice", "af_bella"),
            "response_format": data.get("response_format", "mp3"),
            "speed": data.get("speed", 1.0)
        }
    }
    
    # Call Runpod
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync",
        json=runpod_payload,
        headers=headers
    )
    
    if response.status_code != 200:
        return Response(content=response.text, status_code=response.status_code)
    
    result = response.json()
    
    if result.get("status") == "COMPLETED":
        # Get audio data
        audio_base64 = result["output"]["audio_base64"]
        audio_data = base64.b64decode(audio_base64)
        
        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=f"audio/{data.get('response_format', 'mp3')}"
        )
    else:
        return Response(content=str(result), status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)
'''
    
    with open("proxy_server.py", "w") as f:
        f.write(proxy_code)
    
    print("\nCreated proxy_server.py for OpenAI compatibility")
    print("To use OpenAI client with Runpod:")
    print("1. Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY environment variables")
    print("2. Run: python proxy_server.py")
    print("3. Use OpenAI client pointing to http://localhost:8880")


if __name__ == "__main__":
    print("Kokoro TTS Runpod Test Client")
    print("=============================")
    
    # Create proxy server script
    create_proxy_server()
    
    # Test direct API
    test_direct_api()
    
    # Test OpenAI-compatible interface
    print("\n" + "="*50)
    test_openai_compatible()