#!/usr/bin/env python3
"""
Test client for Kokoro TTS deployed on RunPod using OpenAI client library
"""

import os
import time
from openai import OpenAI
import pyaudio
import wave
import io

# Configuration
RUNPOD_ENDPOINT_URL = os.getenv("RUNPOD_ENDPOINT_URL", "http://localhost:8000")  # Your proxy URL
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "not-needed")  # Not needed for local proxy

# Initialize OpenAI client pointing to your RunPod endpoint
client = OpenAI(
    base_url=f"{RUNPOD_ENDPOINT_URL}/v1",
    api_key=RUNPOD_API_KEY
)

def test_streaming_to_file():
    """Test 1: Stream audio to file"""
    print("\n=== Test 1: Streaming to File ===")
    
    text = """Hello world! This is a test of the Kokoro text-to-speech system 
    deployed on RunPod. We're testing streaming functionality with a longer text 
    to ensure smooth audio generation."""
    
    try:
        start_time = time.time()
        
        # Stream to file using OpenAI client
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_bella",
            input=text,
            response_format="mp3"
        ) as response:
            response.stream_to_file("test_output_streaming.mp3")
        
        elapsed = time.time() - start_time
        print(f"✓ Successfully streamed to file in {elapsed:.2f} seconds")
        print(f"  Output: test_output_streaming.mp3")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def test_multiple_voices():
    """Test 2: Test different voices and combinations"""
    print("\n=== Test 2: Multiple Voices ===")
    
    test_cases = [
        ("af_bella", "Hello, I'm Bella!"),
        ("af_sky", "Hi there, I'm Sky!"),
        ("af_bella+af_sky", "This is a 50/50 mix of Bella and Sky."),
        ("af_bella(2)+af_sky(1)", "This is 67% Bella and 33% Sky."),
        ("am_adam", "Hello, I'm Adam with a male voice."),
    ]
    
    for voice, text in test_cases:
        try:
            print(f"\nTesting voice: {voice}")
            start_time = time.time()
            
            with client.audio.speech.with_streaming_response.create(
                model="kokoro",
                voice=voice,
                input=text,
                response_format="mp3"
            ) as response:
                filename = f"test_voice_{voice.replace('+', '_').replace('(', '').replace(')', '')}.mp3"
                response.stream_to_file(filename)
            
            elapsed = time.time() - start_time
            print(f"  ✓ Generated in {elapsed:.2f}s -> {filename}")
            
        except Exception as e:
            print(f"  ✗ Error with voice {voice}: {e}")

def test_different_formats():
    """Test 3: Different audio formats"""
    print("\n=== Test 3: Audio Formats ===")
    
    formats = ["mp3", "wav", "opus", "flac"]
    text = "Testing different audio formats."
    
    for fmt in formats:
        try:
            print(f"\nTesting format: {fmt}")
            start_time = time.time()
            
            with client.audio.speech.with_streaming_response.create(
                model="kokoro",
                voice="af_nova",
                input=text,
                response_format=fmt
            ) as response:
                filename = f"test_format.{fmt}"
                response.stream_to_file(filename)
            
            elapsed = time.time() - start_time
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"  ✓ Generated in {elapsed:.2f}s")
            print(f"    File: {filename} ({file_size:.1f} KB)")
            
        except Exception as e:
            print(f"  ✗ Error with format {fmt}: {e}")

def test_streaming_to_speakers():
    """Test 4: Stream directly to speakers"""
    print("\n=== Test 4: Streaming to Speakers ===")
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open audio stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True
        )
        
        print("Streaming audio to speakers...")
        start_time = time.time()
        first_chunk_time = None
        
        # Stream with PCM format for direct playback
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_stella",
            input="Hello! This audio is being streamed directly to your speakers. Pretty cool, right?",
            response_format="pcm"
        ) as response:
            chunk_count = 0
            for chunk in response.iter_bytes(chunk_size=1024):
                if chunk:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    stream.write(chunk)
                    chunk_count += 1
        
        elapsed = time.time() - start_time
        print(f"✓ Streamed {chunk_count} chunks in {elapsed:.2f}s")
        print(f"  First chunk latency: {first_chunk_time*1000:.0f}ms")
        
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except ImportError:
        print("✗ PyAudio not installed. Install with: pip install pyaudio")
    except Exception as e:
        print(f"✗ Error: {e}")

def test_long_text_streaming():
    """Test 5: Long text with chunk timing"""
    print("\n=== Test 5: Long Text Streaming Performance ===")
    
    long_text = """
    The sun was setting over the horizon, painting the sky in brilliant shades of 
    orange and purple. Sarah stood at the edge of the cliff, watching the waves 
    crash against the rocks below. She had come here to think, to find clarity in 
    the chaos of her life. The wind whipped through her hair as she closed her eyes 
    and took a deep breath. Tomorrow would be a new day, a fresh start. She had made 
    her decision, and there was no turning back now. With one last look at the sunset, 
    she turned and walked back towards her car, ready to face whatever came next.
    """
    
    try:
        print("Streaming long text and measuring chunk timings...")
        start_time = time.time()
        first_chunk_time = None
        chunk_times = []
        total_bytes = 0
        
        with client.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="bf_emma",
            input=long_text,
            response_format="mp3",
            speed=1.0
        ) as response:
            chunk_count = 0
            last_chunk_time = start_time
            
            with open("test_long_streaming.mp3", "wb") as f:
                for chunk in response.iter_bytes(chunk_size=4096):
                    if chunk:
                        current_time = time.time()
                        
                        if first_chunk_time is None:
                            first_chunk_time = current_time - start_time
                        
                        chunk_time = current_time - last_chunk_time
                        chunk_times.append(chunk_time)
                        last_chunk_time = current_time
                        
                        f.write(chunk)
                        chunk_count += 1
                        total_bytes += len(chunk)
                        
                        # Print progress
                        if chunk_count % 10 == 0:
                            print(f"  Received chunk {chunk_count}: {len(chunk)} bytes")
        
        elapsed = time.time() - start_time
        avg_chunk_time = sum(chunk_times[1:]) / len(chunk_times[1:]) if len(chunk_times) > 1 else 0
        
        print(f"\n✓ Streaming completed in {elapsed:.2f}s")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total size: {total_bytes/1024:.1f} KB")
        print(f"  First chunk latency: {first_chunk_time*1000:.0f}ms")
        print(f"  Average chunk interval: {avg_chunk_time*1000:.0f}ms")
        print(f"  Words in text: {len(long_text.split())}")
        print(f"  Output: test_long_streaming.mp3")
        
    except Exception as e:
        print(f"✗ Error: {e}")

def test_speed_variations():
    """Test 6: Different speech speeds"""
    print("\n=== Test 6: Speech Speed Variations ===")
    
    text = "This sentence will be spoken at different speeds."
    speeds = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    for speed in speeds:
        try:
            print(f"\nTesting speed: {speed}x")
            start_time = time.time()
            
            # Note: speed parameter might need to be passed differently
            # depending on your RunPod handler implementation
            response = client.audio.speech.create(
                model="kokoro",
                voice="am_michael",
                input=text,
                response_format="mp3",
                speed=speed  # This might need adjustment based on API
            )
            
            filename = f"test_speed_{speed}x.mp3"
            response.stream_to_file(filename)
            
            elapsed = time.time() - start_time
            print(f"  ✓ Generated in {elapsed:.2f}s -> {filename}")
            
        except Exception as e:
            print(f"  ✗ Error with speed {speed}: {e}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Kokoro TTS on RunPod - OpenAI Client Test Suite")
    print("=" * 60)
    print(f"Endpoint: {RUNPOD_ENDPOINT_URL}")
    print(f"API Key: {'Set' if RUNPOD_API_KEY != 'not-needed' else 'Not needed'}")
    
    # Check connectivity
    print("\nChecking connection...")
    try:
        # This might fail if /v1/audio/voices isn't implemented in your proxy
        response = client.audio.speech.create(
            model="kokoro",
            voice="af_bella",
            input="Test",
            response_format="mp3"
        )
        print("✓ Connection successful!")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure your RunPod endpoint or proxy is running.")
        return
    
    # Run tests
    tests = [
        test_streaming_to_file,
        test_multiple_voices,
        test_different_formats,
        test_streaming_to_speakers,
        test_long_text_streaming,
        # test_speed_variations,  # Uncomment if speed parameter is supported
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
        
        # Small delay between tests
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Set these environment variables or modify directly
    # export RUNPOD_ENDPOINT_URL=https://your-proxy-url.com
    # export RUNPOD_API_KEY=your-api-key
    
    main()