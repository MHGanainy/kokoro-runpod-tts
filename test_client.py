#!/usr/bin/env python3
"""
Test client for Kokoro TTS on RunPod
Tests both WebSocket streaming (ElevenLabs-compatible) and RunPod REST endpoints
"""

import asyncio
import json
import base64
import time
import argparse
import sys
from typing import Optional, Dict, Any
import numpy as np

try:
    import websockets
    import aiohttp
except ImportError:
    print("Please install required packages:")
    print("pip install websockets aiohttp")
    sys.exit(1)

# Optional: for audio playback
try:
    import sounddevice as sd
    AUDIO_PLAYBACK = True
except ImportError:
    AUDIO_PLAYBACK = False
    print("Warning: sounddevice not installed. Audio playback disabled.")
    print("Install with: pip install sounddevice")

class RunPodKokoroClient:
    def __init__(self, endpoint_id: str, api_key: str):
        """
        Initialize RunPod client
        
        Args:
            endpoint_id: Your RunPod endpoint ID (e.g., "p9x3xtammij2jv")
            api_key: Your RunPod API key
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        
        # RunPod URLs
        self.rest_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        # WebSocket URL uses the pod's direct URL
        self.ws_base_url = None  # Will be set after getting pod URL
        
    async def get_pod_url(self) -> Optional[str]:
        """Get the direct pod URL for WebSocket connections"""
        print("\n=== Getting Pod URL ===")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get endpoint health
                async with session.get(f"{self.rest_url}/health", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"Endpoint status: {json.dumps(data, indent=2)}")
                        
                        # For serverless endpoints, we need to use the REST API
                        # For pod endpoints, we might get a direct URL
                        if "workers" in data and data["workers"]:
                            # This is a pod endpoint with workers
                            worker = data["workers"][0]
                            if "id" in worker:
                                # Construct pod URL (this format may vary)
                                pod_url = f"https://{worker['id']}-8000.proxy.runpod.net"
                                self.ws_base_url = pod_url.replace("https://", "wss://")
                                print(f"Pod WebSocket URL: {self.ws_base_url}")
                                return pod_url
                        
                        # For serverless, we'll use a different approach
                        print("Serverless endpoint detected - WebSocket may not be directly accessible")
                        return None
                    else:
                        print(f"Failed to get endpoint health: {response.status}")
                        return None
                        
        except Exception as e:
            print(f"Error getting pod URL: {e}")
            # Try to construct URL from endpoint ID pattern
            # RunPod pod URLs often follow this pattern
            if "-" in self.endpoint_id:
                pod_id = self.endpoint_id.split("-")[0]
                pod_url = f"https://{pod_id}-8000.proxy.runpod.net"
                self.ws_base_url = pod_url.replace("https://", "wss://")
                print(f"Attempting with constructed URL: {self.ws_base_url}")
                return pod_url
            return None
    
    async def test_rest_api(
        self,
        text: str = "This is a test of the Kokoro text to speech system on RunPod.",
        voice_id: str = "af_bella",
        operation: str = "run"  # "run" for sync, "run_async" for async
    ):
        """Test RunPod REST API endpoint"""
        print(f"\n=== Testing RunPod REST API ({operation}) ===")
        print(f"Endpoint: {self.endpoint_id}")
        print(f"Text: {text}")
        print(f"Voice: {voice_id}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": {
                "text": text,
                "voice_id": voice_id,
                "speed": 1.0,
                "output_format": "pcm_16000"
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                # For async operation
                if operation == "run_async":
                    # Submit job
                    async with session.post(f"{self.rest_url}/run", json=payload, headers=headers) as response:
                        if response.status != 200:
                            print(f"Failed to submit job: {response.status}")
                            return False
                        
                        job_data = await response.json()
                        job_id = job_data.get("id")
                        print(f"Job submitted: {job_id}")
                    
                    # Poll for results
                    while True:
                        async with session.get(f"{self.rest_url}/status/{job_id}", headers=headers) as response:
                            status_data = await response.json()
                            status = status_data.get("status")
                            print(f"Job status: {status}")
                            
                            if status == "COMPLETED":
                                data = status_data.get("output", {})
                                break
                            elif status in ["FAILED", "CANCELLED"]:
                                print(f"Job failed: {status_data}")
                                return False
                            
                            await asyncio.sleep(0.5)
                
                # For sync operation
                else:
                    async with session.post(f"{self.rest_url}/{operation}", json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"Request failed: {response.status}")
                            print(f"Error: {error_text}")
                            return False
                        
                        result = await response.json()
                        
                        # Check if we got a job ID (async response)
                        if "id" in result and "status" in result:
                            print(f"Job submitted: {result['id']} (status: {result['status']})")
                            job_id = result["id"]
                            
                            # Poll for results
                            print("Polling for results...")
                            poll_count = 0
                            while True:
                                poll_count += 1
                                await asyncio.sleep(0.5)
                                
                                async with session.get(f"{self.rest_url}/status/{job_id}", headers=headers) as status_response:
                                    status_data = await status_response.json()
                                    status = status_data.get("status")
                                    
                                    if poll_count % 4 == 1:  # Print every 2 seconds
                                        print(f"Status: {status} ({poll_count * 0.5:.1f}s elapsed)")
                                    
                                    if status == "COMPLETED":
                                        data = status_data.get("output", {})
                                        break
                                    elif status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                                        print(f"Job failed: {status}")
                                        if "error" in status_data:
                                            print(f"Error: {status_data['error']}")
                                        return False
                                    elif poll_count > 180:  # 90 second timeout
                                        print("Timeout waiting for job completion")
                                        return False
                        
                        # Handle direct response (if not async)
                        elif "output" in result:
                            data = result["output"]
                        else:
                            data = result
                
                elapsed = time.time() - start_time
                print(f"Total time: {elapsed*1000:.1f}ms")
                
                # Process response
                if not data:
                    print("No data in response")
                    return False
                
                if "error" in data:
                    print(f"Error in response: {data['error']}")
                    return False
                
                if "audio" in data:
                    audio_bytes = base64.b64decode(data["audio"])
                    print(f"Audio size: {len(audio_bytes)} bytes")
                    print(f"Processing time: {data.get('processing_time', 0)*1000:.1f}ms")
                    print(f"Characters processed: {data.get('characters', 0)}")
                    
                    # Calculate metrics
                    audio_duration = len(audio_bytes) / 2 / 16000  # 16-bit audio at 16kHz
                    print(f"Audio duration: {audio_duration:.2f}s")
                    print(f"Real-time factor: {elapsed/audio_duration:.2f}x")
                    
                    # Play audio if available
                    if AUDIO_PLAYBACK:
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        print(f"\nPlaying audio...")
                        sd.play(audio_array, samplerate=16000)
                        sd.wait()
                    
                    return True
                else:
                    print(f"No audio in response: {data}")
                    return False
                    
        except Exception as e:
            print(f"REST API error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def test_websocket_streaming(
        self, 
        text: str = "Hello world, this is a streaming test of the Kokoro text to speech system.",
        voice_id: str = "af_bella",
        play_audio: bool = True
    ):
        """Test WebSocket streaming (if available)"""
        print(f"\n=== Testing WebSocket Streaming ===")
        
        # Get pod URL if not set
        if not self.ws_base_url:
            pod_url = await self.get_pod_url()
            if not pod_url:
                print("WebSocket URL not available - this might be a serverless endpoint")
                print("Use REST API for serverless endpoints")
                return False
        
        print(f"Text: {text}")
        print(f"Voice: {voice_id}")
        
        # Connect to WebSocket
        uri = f"{self.ws_base_url}/v1/text-to-speech/{voice_id}/stream-input"
        print(f"Connecting to: {uri}")
        
        audio_chunks = []
        metadata = {}
        
        try:
            # Add timeout for connection
            async with websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                print("Connected to WebSocket")
                
                # Send initial space (ElevenLabs compatibility)
                init_msg = json.dumps({"text": " "})
                await websocket.send(init_msg)
                print("Sent initialization message")
                
                # Send actual text
                message = json.dumps({
                    "text": text,
                    "voice_settings": {
                        "speed": 1.0
                    },
                    "context_id": "test-context-123"
                })
                
                await websocket.send(message)
                print("Sent text message")
                
                # Receive audio chunks
                chunk_count = 0
                start_time = time.time()
                first_chunk_time = None
                
                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        data = json.loads(response)
                        
                        if "audio" in data:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - start_time
                                print(f"First chunk received in: {first_chunk_time*1000:.1f}ms")
                            
                            # Decode audio
                            audio_bytes = base64.b64decode(data["audio"])
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                            audio_chunks.append(audio_array)
                            chunk_count += 1
                            print(f"Received chunk {chunk_count} ({len(audio_bytes)} bytes)")
                        
                        if "alignment" in data:
                            print(f"Received alignment data: {len(data['alignment']['chars'])} characters")
                        
                        if data.get("isFinal"):
                            metadata = data.get("metadata", {})
                            print("Received final message")
                            break
                            
                    except asyncio.TimeoutError:
                        print("Timeout waiting for response")
                        break
                
                # Close connection
                close_msg = json.dumps({"close_socket": True})
                await websocket.send(close_msg)
                
            total_time = time.time() - start_time
            print(f"\nStreaming completed:")
            print(f"  Total chunks: {chunk_count}")
            print(f"  Total time: {total_time*1000:.1f}ms")
            print(f"  First chunk latency: {first_chunk_time*1000:.1f}ms" if first_chunk_time else "  No chunks received")
            
            if metadata:
                print(f"  Server metadata: {json.dumps(metadata, indent=4)}")
            
            # Combine and play audio
            if audio_chunks and play_audio and AUDIO_PLAYBACK:
                full_audio = np.concatenate(audio_chunks)
                print(f"\nPlaying audio ({len(full_audio)/16000:.2f} seconds)...")
                sd.play(full_audio, samplerate=16000)
                sd.wait()
            
            return len(audio_chunks) > 0
            
        except websockets.exceptions.InvalidStatusCode as e:
            print(f"WebSocket connection failed: {e}")
            print("This might be a serverless endpoint - use REST API instead")
            return False
        except Exception as e:
            print(f"WebSocket error: {e}")
            return False
    
    async def test_health(self):
        """Test endpoint health"""
        print("\n=== Testing Endpoint Health ===")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test RunPod health endpoint
                async with session.get(f"{self.rest_url}/health", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"RunPod Health: {json.dumps(data, indent=2)}")
                        
                        # Also try to get metrics if available
                        if self.ws_base_url:
                            metrics_url = self.ws_base_url.replace("wss://", "https://") + "/metrics"
                            try:
                                async with session.get(metrics_url) as metrics_response:
                                    if metrics_response.status == 200:
                                        metrics = await metrics_response.json()
                                        print(f"\nServer Metrics: {json.dumps(metrics, indent=2)}")
                            except:
                                pass
                        
                        return True
                    else:
                        print(f"Health check failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"Health check error: {e}")
            return False
    
    async def benchmark(self, num_requests: int = 10):
        """Run a benchmark test"""
        print(f"\n=== Running Benchmark ({num_requests} requests) ===")
        
        texts = [
            "Short test.",
            "This is a medium length sentence for testing.",
            "This is a much longer sentence that contains more words and should take a bit more time to process and generate audio for.",
        ]
        
        latencies = []
        processing_times = []
        success_count = 0
        
        for i in range(num_requests):
            text = texts[i % len(texts)]
            voice = ["af_bella", "am_adam", "bf_emma", "bm_george"][i % 4]
            
            print(f"\nRequest {i+1}/{num_requests}: {len(text)} chars, voice={voice}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "input": {
                    "text": text,
                    "voice_id": voice,
                    "speed": 1.0,
                    "output_format": "pcm_16000"
                }
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    start_time = time.time()
                    async with session.post(f"{self.rest_url}/run", json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            latency = time.time() - start_time
                            
                            data = result.get("output", result)
                            if "audio" in data:
                                latencies.append(latency)
                                processing_times.append(data.get("processing_time", 0))
                                success_count += 1
                                print(f"  Success: {latency*1000:.1f}ms total, {data.get('processing_time', 0)*1000:.1f}ms processing")
                            else:
                                print(f"  Failed: No audio in response")
                        else:
                            print(f"  Failed: HTTP {response.status}")
                            
            except Exception as e:
                print(f"  Failed: {e}")
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Calculate statistics
        if latencies:
            print(f"\n=== Benchmark Results ===")
            print(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
            print(f"Average latency: {np.mean(latencies)*1000:.1f}ms")
            print(f"Min latency: {np.min(latencies)*1000:.1f}ms")
            print(f"Max latency: {np.max(latencies)*1000:.1f}ms")
            print(f"P95 latency: {np.percentile(latencies, 95)*1000:.1f}ms")
            
            if processing_times:
                print(f"\nProcessing times:")
                print(f"Average: {np.mean(processing_times)*1000:.1f}ms")
                print(f"Min: {np.min(processing_times)*1000:.1f}ms")
                print(f"Max: {np.max(processing_times)*1000:.1f}ms")
        
        return success_count > 0

async def main():
    parser = argparse.ArgumentParser(description="Test Kokoro TTS on RunPod")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID (e.g., p9x3xtammij2jv)")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--text", default="Hello world, this is a test of the Kokoro text to speech system on RunPod.", help="Text to synthesize")
    parser.add_argument("--voice", default="af_bella", help="Voice ID (af_bella, am_adam, bf_emma, bm_george)")
    parser.add_argument("--test", choices=["all", "health", "rest", "rest-async", "websocket", "benchmark"], default="all", help="Test to run")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio playback")
    parser.add_argument("--benchmark-requests", type=int, default=10, help="Number of requests for benchmark")
    
    args = parser.parse_args()
    
    # Create client
    client = RunPodKokoroClient(args.endpoint_id, args.api_key)
    
    # Run tests
    results = {}
    
    if args.test in ["all", "health"]:
        results["health"] = await client.test_health()
    
    if args.test in ["all", "rest"]:
        results["rest"] = await client.test_rest_api(
            text=args.text,
            voice_id=args.voice,
            operation="run"
        )
    
    if args.test == "rest-async":
        results["rest-async"] = await client.test_rest_api(
            text=args.text,
            voice_id=args.voice,
            operation="run_async"
        )
    
    if args.test in ["all", "websocket"]:
        results["websocket"] = await client.test_websocket_streaming(
            text=args.text,
            voice_id=args.voice,
            play_audio=not args.no_audio
        )
    
    if args.test == "benchmark":
        results["benchmark"] = await client.benchmark(args.benchmark_requests)
    
    # Summary
    if results:
        print("\n=== Test Summary ===")
        for test_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        # RunPod-specific tips
        print("\n=== RunPod Tips ===")
        print("- Serverless endpoints only support REST API, not WebSocket streaming")
        print("- Use 'run' for synchronous requests (faster for small texts)")
        print("- Use 'run_async' for large texts or when you need job tracking")
        print("- Cold starts may add 5-30s to the first request")
        print("- Check RunPod dashboard for detailed metrics and logs")
        
        return 0 if all_passed else 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)