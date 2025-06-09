#!/usr/bin/env python3
"""
Test client for Kokoro TTS RunPod instance
Tests both RunPod REST/streaming API and ElevenLabs-compatible WebSocket
"""

import asyncio
import aiohttp
import websockets
import json
import base64
import time
import argparse
from typing import Optional, Dict, Any, List
import numpy as np

# Optional: for audio playback/saving
try:
    import sounddevice as sd
    AUDIO_PLAYBACK = True
except ImportError:
    AUDIO_PLAYBACK = False
    print("Warning: sounddevice not installed. Audio playback disabled.")

try:
    import soundfile as sf
    AUDIO_SAVE = True
except ImportError:
    AUDIO_SAVE = False
    print("Warning: soundfile not installed. Audio saving disabled.")


class KokoroRunPodTester:
    def __init__(self, endpoint_id: str, api_key: str, pod_url: Optional[str] = None):
        """
        Initialize tester for Kokoro TTS on RunPod
        
        Args:
            endpoint_id: RunPod endpoint ID
            api_key: RunPod API key
            pod_url: Direct pod URL for WebSocket (optional, will try to auto-detect)
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.rest_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.pod_url = pod_url
        self.ws_url = None
        
    async def detect_pod_url(self) -> Optional[str]:
        """Try to detect the pod URL for WebSocket connections"""
        if self.pod_url:
            return self.pod_url
            
        print("\n=== Detecting Pod URL ===")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Try to get pod information
                async with session.get(f"{self.rest_url}/health", headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"Endpoint health: {json.dumps(data, indent=2)}")
                        
                        # For pod deployments, construct the WebSocket URL
                        # RunPod pod URLs typically follow this pattern
                        if "-" in self.endpoint_id:
                            pod_id = self.endpoint_id.split("-")[0]
                            pod_url = f"https://{pod_id}-8000.proxy.runpod.net"
                            print(f"Detected pod URL: {pod_url}")
                            self.pod_url = pod_url
                            self.ws_url = pod_url.replace("https://", "wss://")
                            return pod_url
                        
        except Exception as e:
            print(f"Could not detect pod URL: {e}")
            
        return None
        
    async def test_health(self) -> bool:
        """Test the health endpoint"""
        print("\n=== Testing Health Endpoint ===")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test RunPod health
                async with session.get(f"{self.rest_url}/health", headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        print(f"RunPod endpoint health: {json.dumps(data, indent=2)}")
                        
                        # Test pod health if URL available
                        if self.pod_url:
                            async with session.get(f"{self.pod_url}/") as pod_response:
                                if pod_response.status == 200:
                                    pod_data = await pod_response.json()
                                    print(f"\nPod server health: {json.dumps(pod_data, indent=2)}")
                                    
                        return True
                    else:
                        print(f"Health check failed: {response.status}")
                        return False
                        
        except Exception as e:
            print(f"Health check error: {e}")
            return False
            
    async def test_rest_api(
        self,
        text: str = "Hello! This is a test of the Kokoro text to speech system on RunPod.",
        voice_id: str = "af_bella",
        streaming: bool = False
    ) -> Dict[str, Any]:
        """Test the RunPod REST API"""
        print(f"\n=== Testing REST API (streaming={streaming}) ===")
        print(f"Text: {text}")
        print(f"Voice: {voice_id}")
        
        payload = {
            "input": {
                "text": text,
                "voice_id": voice_id,
                "speed": 1.0,
                "output_format": "pcm_16000",
                "streaming": streaming
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                # Submit job
                async with session.post(f"{self.rest_url}/run", json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Request failed ({response.status}): {error_text}")
                        return {}
                        
                    result = await response.json()
                    job_id = result.get("id")
                    print(f"Job submitted: {job_id}")
                    
                # Handle async job
                if job_id and "status" in result:
                    # For streaming, use the stream endpoint
                    if streaming:
                        return await self._handle_streaming(session, job_id, start_time)
                    else:
                        return await self._handle_polling(session, job_id, start_time)
                        
        except Exception as e:
            print(f"REST API error: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    async def _handle_streaming(self, session: aiohttp.ClientSession, job_id: str, start_time: float) -> Dict[str, Any]:
        """Handle streaming response from RunPod"""
        print("Streaming audio chunks...")
        
        stream_url = f"{self.rest_url}/stream/{job_id}"
        audio_chunks = []
        chunk_count = 0
        first_chunk_time = None
        final_result = None
        
        while True:
            async with session.get(stream_url, headers=self.headers) as response:
                if response.status != 200:
                    print(f"Stream request failed: {response.status}")
                    break
                    
                stream_data = await response.json()
                
                if stream_data["status"] == "COMPLETED":
                    print("Streaming completed!")
                    if "output" in stream_data:
                        final_result = stream_data["output"]
                    break
                    
                elif stream_data["status"] == "FAILED":
                    error = stream_data.get("error", "Unknown error")
                    print(f"Job failed: {error}")
                    break
                    
                elif stream_data["status"] == "IN_PROGRESS" and "stream" in stream_data:
                    for output in stream_data["stream"]:
                        if "audio_chunk" in output:
                            if first_chunk_time is None:
                                first_chunk_time = output.get("timestamp", time.time() - start_time)
                                print(f"First chunk at: {first_chunk_time:.3f}s")
                                
                            chunk_count += 1
                            audio_bytes = base64.b64decode(output["audio_chunk"])
                            audio_chunks.append(np.frombuffer(audio_bytes, dtype=np.int16))
                            
                            print(f"  Chunk {chunk_count}: {len(audio_bytes)} bytes")
                            
            await asyncio.sleep(0.1)
            
        total_time = time.time() - start_time
        
        return {
            "audio_chunks": audio_chunks,
            "chunk_count": chunk_count,
            "first_chunk_time": first_chunk_time,
            "total_time": total_time,
            "final_result": final_result
        }
        
    async def _handle_polling(self, session: aiohttp.ClientSession, job_id: str, start_time: float) -> Dict[str, Any]:
        """Handle standard polling response from RunPod"""
        print("Polling for results...")
        
        while True:
            await asyncio.sleep(0.5)
            
            async with session.get(f"{self.rest_url}/status/{job_id}", headers=self.headers) as response:
                if response.status != 200:
                    print(f"Status check failed: {response.status}")
                    break
                    
                status_data = await response.json()
                status = status_data.get("status")
                
                if status == "COMPLETED":
                    output = status_data.get("output", {})
                    total_time = time.time() - start_time
                    
                    if "audio" in output:
                        audio_bytes = base64.b64decode(output["audio"])
                        print(f"\nCompleted in {total_time:.2f}s")
                        print(f"Audio size: {len(audio_bytes)} bytes")
                        print(f"Processing time: {output.get('processing_time', 0):.3f}s")
                        
                        return {
                            "audio": audio_bytes,
                            "total_time": total_time,
                            "output": output
                        }
                    break
                    
                elif status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                    error = status_data.get("error", "Unknown error")
                    print(f"Job {status}: {error}")
                    break
                    
        return {}
        
    async def test_websocket(
        self,
        text: str = "Hello! This is a test of the ElevenLabs-compatible WebSocket interface.",
        voice_id: str = "af_bella"
    ) -> bool:
        """Test the ElevenLabs-compatible WebSocket interface"""
        print(f"\n=== Testing ElevenLabs-Compatible WebSocket ===")
        
        if not self.ws_url:
            print("WebSocket URL not available. Trying to detect...")
            await self.detect_pod_url()
            if not self.ws_url:
                print("Could not determine WebSocket URL")
                return False
                
        print(f"WebSocket URL: {self.ws_url}")
        print(f"Text: {text}")
        print(f"Voice: {voice_id}")
        
        # Build WebSocket URL
        ws_endpoint = f"{self.ws_url}/v1/text-to-speech/{voice_id}/multi-stream-input"
        context_id = f"test-{int(time.time())}"
        
        try:
            async with websockets.connect(ws_endpoint) as websocket:
                print("Connected to WebSocket")
                
                # Send initial space to create context
                init_msg = {
                    "text": " ",
                    "context_id": context_id,
                    "voice_settings": {"speed": 1.0}
                }
                await websocket.send(json.dumps(init_msg))
                print("Sent context initialization")
                
                # Send actual text
                text_msg = {
                    "text": text,
                    "context_id": context_id
                }
                await websocket.send(json.dumps(text_msg))
                print("Sent text message")
                
                # Receive responses
                audio_chunks = []
                chunk_count = 0
                start_time = time.time()
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        if "audio" in data:
                            chunk_count += 1
                            audio_bytes = base64.b64decode(data["audio"])
                            audio_chunks.append(np.frombuffer(audio_bytes, dtype=np.int16))
                            print(f"Received audio chunk {chunk_count}: {len(audio_bytes)} bytes")
                            
                        elif "alignment" in data:
                            print(f"Received alignment data: {len(data['alignment']['chars'])} chars")
                            
                        elif data.get("isFinal"):
                            metadata = data.get("metadata", {})
                            print(f"\nCompleted:")
                            print(f"  Total chunks: {metadata.get('chunks', chunk_count)}")
                            print(f"  Generation time: {metadata.get('generation_time_ms', 0)}ms")
                            print(f"  Audio duration: {metadata.get('audio_duration_ms', 0)}ms")
                            break
                            
                    except asyncio.TimeoutError:
                        print("Timeout waiting for response")
                        break
                        
                # Close connection
                await websocket.send(json.dumps({"close_socket": True}))
                
                total_time = time.time() - start_time
                print(f"\nWebSocket test completed in {total_time:.2f}s")
                
                # Play audio if available
                if audio_chunks and AUDIO_PLAYBACK:
                    full_audio = np.concatenate(audio_chunks)
                    print(f"\nPlaying audio ({len(full_audio)/16000:.2f}s)...")
                    sd.play(full_audio, samplerate=16000)
                    sd.wait()
                    
                return len(audio_chunks) > 0
                
        except Exception as e:
            print(f"WebSocket error: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    async def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("KOKORO TTS RUNPOD TEST SUITE")
        print("=" * 60)
        
        results = {
            "health": False,
            "rest_standard": False,
            "rest_streaming": False,
            "websocket": False
        }
        
        # Test health
        results["health"] = await self.test_health()
        
        # Test REST API (standard)
        rest_result = await self.test_rest_api(streaming=False)
        if rest_result and "audio" in rest_result:
            results["rest_standard"] = True
            
            # Play audio
            if AUDIO_PLAYBACK:
                audio_array = np.frombuffer(rest_result["audio"], dtype=np.int16)
                print(f"\nPlaying audio ({len(audio_array)/16000:.2f}s)...")
                sd.play(audio_array, samplerate=16000)
                sd.wait()
                
        # Test REST API (streaming)
        stream_result = await self.test_rest_api(streaming=True)
        if stream_result and "audio_chunks" in stream_result and stream_result["audio_chunks"]:
            results["rest_streaming"] = True
            
            # Play combined audio
            if AUDIO_PLAYBACK:
                full_audio = np.concatenate(stream_result["audio_chunks"])
                print(f"\nPlaying streamed audio ({len(full_audio)/16000:.2f}s)...")
                sd.play(full_audio, samplerate=16000)
                sd.wait()
                
        # Test WebSocket
        results["websocket"] = await self.test_websocket()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{test_name.ljust(20)}: {status}")
            
        all_passed = all(results.values())
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        return all_passed


async def main():
    parser = argparse.ArgumentParser(description="Test Kokoro TTS on RunPod")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--pod-url", help="Direct pod URL (e.g., https://xxxxx-8000.proxy.runpod.net)")
    parser.add_argument("--test", choices=["all", "health", "rest", "streaming", "websocket"], default="all")
    parser.add_argument("--text", default="Hello! This is a test of the Kokoro text to speech system.", help="Text to synthesize")
    parser.add_argument("--voice", default="af_bella", help="Voice ID")
    
    args = parser.parse_args()
    
    # Create tester
    tester = KokoroRunPodTester(args.endpoint_id, args.api_key, args.pod_url)
    
    # Run tests
    if args.test == "all":
        success = await tester.run_all_tests()
    elif args.test == "health":
        success = await tester.test_health()
    elif args.test == "rest":
        result = await tester.test_rest_api(args.text, args.voice, streaming=False)
        success = bool(result)
    elif args.test == "streaming":
        result = await tester.test_rest_api(args.text, args.voice, streaming=True)
        success = bool(result)
    elif args.test == "websocket":
        success = await tester.test_websocket(args.text, args.voice)
        
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))