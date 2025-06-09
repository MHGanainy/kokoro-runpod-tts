import runpod
import subprocess
import time
import requests
import json
import base64
from typing import Dict, Any, Generator
import os
import sys

# Add the venv to path
sys.path.insert(0, '/app/.venv/lib/python3.10/site-packages')

# Start the Kokoro FastAPI server
def start_kokoro_server():
    """Start the Kokoro FastAPI server in the background"""
    env = os.environ.copy()
    env['DEVICE'] = 'cuda'
    env['HOST'] = '0.0.0.0'
    env['PORT'] = '8880'
    env['PATH'] = '/app/.venv/bin:' + env.get('PATH', '')
    
    # Start the server using the GPU start script with the virtual environment
    cmd = ['/bin/bash', '-c', 'source /app/.venv/bin/activate && cd /app && ./start-gpu.sh']
    subprocess.Popen(cmd, env=env)
    
    # Wait for the server to be ready
    max_retries = 60  # Increased for model loading time
    for i in range(max_retries):
        try:
            response = requests.get('http://localhost:8880/health', timeout=2)
            if response.status_code == 200:
                print("Kokoro server is ready!")
                return True
        except:
            pass
        if i % 10 == 0:
            print(f"Waiting for server to start... ({i}/{max_retries})")
        time.sleep(2)
    
    raise Exception("Failed to start Kokoro server")

def handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    RunPod handler for Kokoro TTS
    
    Expected input format:
    {
        "input": {
            "text": "Hello world!",
            "voice": "af_bella",  # or voice combinations like "af_bella+af_sky"
            "model": "kokoro",
            "response_format": "mp3",  # mp3, wav, opus, flac, pcm
            "speed": 1.0,
            "stream": true,  # Enable streaming
            "include_timestamps": false  # Optional: include word-level timestamps
        }
    }
    """
    
    try:
        job_input = job['input']
        
        # Extract parameters
        text = job_input.get('text', '')
        voice = job_input.get('voice', 'af_bella')
        model = job_input.get('model', 'kokoro')
        response_format = job_input.get('response_format', 'mp3')
        speed = job_input.get('speed', 1.0)
        stream = job_input.get('stream', True)
        include_timestamps = job_input.get('include_timestamps', False)
        
        if not text:
            yield {"error": "No text provided"}
            return
        
        # Prepare request payload
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "stream": stream
        }
        
        # Choose endpoint based on whether timestamps are needed
        if include_timestamps:
            endpoint = "http://localhost:8880/dev/captioned_speech"
        else:
            endpoint = "http://localhost:8880/v1/audio/speech"
        
        # Make request to Kokoro API
        response = requests.post(
            endpoint,
            json=payload,
            stream=stream
        )
        
        if response.status_code != 200:
            yield {"error": f"API error: {response.text}"}
            return
        
        if stream:
            # Handle streaming response
            if include_timestamps:
                # Stream with timestamps
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        chunk_data = json.loads(line)
                        yield {
                            "audio_chunk": chunk_data["audio"],  # Base64 encoded
                            "timestamps": chunk_data.get("timestamps", []),
                            "is_final": False
                        }
            else:
                # Stream raw audio chunks
                chunk_buffer = b""
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        chunk_buffer += chunk
                        # Send chunks in base64 for easier transport
                        yield {
                            "audio_chunk": base64.b64encode(chunk).decode('utf-8'),
                            "is_final": False
                        }
            
            # Final message
            yield {"is_final": True}
        else:
            # Non-streaming response
            if include_timestamps:
                result = response.json()
                yield {
                    "audio": result["audio"],  # Base64 encoded
                    "timestamps": result.get("timestamps", []),
                    "is_final": True
                }
            else:
                # Return entire audio as base64
                audio_data = response.content
                yield {
                    "audio": base64.b64encode(audio_data).decode('utf-8'),
                    "response_format": response_format,
                    "is_final": True
                }
                
    except Exception as e:
        yield {"error": str(e)}

# Initialize server on cold start
print("Starting Kokoro server...")
start_kokoro_server()

# RunPod handler
runpod.serverless.start({"handler": handler})