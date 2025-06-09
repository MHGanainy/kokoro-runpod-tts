import requests
import base64
import json
import time

# Your RunPod configuration
RUNPOD_ENDPOINT_ID = "your-endpoint-id"  # Replace with your endpoint ID
RUNPOD_API_KEY = "your-api-key"  # Replace with your API key

def text_to_speech(text, voice="af_bella"):
    """Convert text to speech using RunPod endpoint"""
    
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "text": text,
            "voice": voice
        }
    }
    
    print(f"Sending request for: '{text}'")
    start_time = time.time()
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    result = response.json()
    elapsed = time.time() - start_time
    print(f"Response received in {elapsed:.2f}s")
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return None
    
    # Get the audio data
    if "output" in result and "audio_base64" in result["output"]:
        audio_base64 = result["output"]["audio_base64"]
        audio_bytes = base64.b64decode(audio_base64)
        return audio_bytes
    else:
        print("No audio in response")
        return None

def main():
    """Test the RunPod endpoint"""
    
    print("Testing Kokoro TTS on RunPod")
    print("=" * 50)
    
    # Test 1: Simple text
    print("\nTest 1: Simple text")
    audio = text_to_speech("Hello world! This is a test.")
    if audio:
        with open("test1.wav", "wb") as f:
            f.write(audio)
        print("✓ Saved to test1.wav")
    
    # Test 2: Different voice
    print("\nTest 2: Different voice")
    audio = text_to_speech("Testing with Sky voice.", voice="af_sky")
    if audio:
        with open("test2.wav", "wb") as f:
            f.write(audio)
        print("✓ Saved to test2.wav")
    
    # Test 3: Longer text
    print("\nTest 3: Longer text")
    long_text = """
    The sun was setting over the mountains, painting the sky in brilliant 
    shades of orange and purple. It was a beautiful sight to behold.
    """
    audio = text_to_speech(long_text)
    if audio:
        with open("test3.wav", "wb") as f:
            f.write(audio)
        print("✓ Saved to test3.wav")
    
    print("\n" + "=" * 50)
    print("Testing complete!")

if __name__ == "__main__":
    main()