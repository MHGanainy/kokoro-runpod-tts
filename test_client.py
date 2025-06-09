#!/usr/bin/env python3
"""
Test Client for Kokoro TTS RunPod WebSocket-Style Handler
Tests all input formats and streaming capabilities
"""

import asyncio
import aiohttp
import json
import base64
import time
import argparse
from typing import Dict, Any, List, Optional
import numpy as np

# Optional audio playback
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

class KokoroWebSocketTestClient:
    """Test client for Kokoro WebSocket-style TTS on RunPod"""
    
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Track results
        self.audio_chunks = []
        self.events = []
        
    async def test_health(self) -> bool:
        """Test health check endpoint"""
        print("\n=== Testing Health Check ===")
        
        payload = {
            "input": {
                "health_check": True
            }
        }
        
        try:
            success = await self._send_request(payload, "Health Check")
            return success
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    async def test_single_tts(self, text: str = "Hello! This is a test of the Kokoro TTS system.", voice_id: str = "af_bella") -> bool:
        """Test single TTS request"""
        print(f"\n=== Testing Single TTS ===")
        print(f"Text: {text}")
        print(f"Voice: {voice_id}")
        
        payload = {
            "input": {
                "text": text,
                "voice_id": voice_id,
                "voice_settings": {
                    "speed": 1.0
                }
            }
        }
        
        return await self._send_request(payload, "Single TTS")
    
    async def test_websocket_message(self, text: str = "Hello from WebSocket-style interface!", voice_id: str = "af_bella") -> bool:
        """Test WebSocket-style message"""
        print(f"\n=== Testing WebSocket Message ===")
        print(f"Text: {text}")
        print(f"Voice: {voice_id}")
        
        payload = {
            "input": {
                "websocket_message": {
                    "text": text,
                    "context_id": f"test-ws-{int(time.time())}",
                    "voice_settings": {
                        "speed": 1.1
                    }
                },
                "voice_id": voice_id
            }
        }
        
        return await self._send_request(payload, "WebSocket Message")
    
    async def test_websocket_initialization(self, voice_id: str = "af_bella") -> bool:
        """Test WebSocket context initialization"""
        print(f"\n=== Testing WebSocket Initialization ===")
        
        context_id = f"test-init-{int(time.time())}"
        
        # Test initialization message (space)
        payload = {
            "input": {
                "websocket_message": {
                    "text": " ",
                    "context_id": context_id,
                    "voice_settings": {"speed": 1.0}
                },
                "voice_id": voice_id
            }
        }
        
        return await self._send_request(payload, "WebSocket Initialization")
    
    async def test_conversation(self, messages: List[str] = None, voice_id: str = "af_bella") -> bool:
        """Test multi-message conversation"""
        if messages is None:
            messages = [
                "Welcome to our advanced AI assistant.",
                "I can help you with various tasks today.",
                "What would you like to know about our services?"
            ]
        
        print(f"\n=== Testing Conversation ===")
        print(f"Messages: {len(messages)}")
        for i, msg in enumerate(messages):
            print(f"  {i+1}: {msg}")
        print(f"Voice: {voice_id}")
        
        payload = {
            "input": {
                "messages": messages,
                "voice_id": voice_id,
                "context_id": f"test-conv-{int(time.time())}",
                "voice_settings": {
                    "speed": 0.9
                }
            }
        }
        
        return await self._send_request(payload, "Conversation")
    
    async def test_websocket_controls(self, voice_id: str = "af_bella") -> bool:
        """Test WebSocket control messages"""
        print(f"\n=== Testing WebSocket Controls ===")
        
        context_id = f"test-ctrl-{int(time.time())}"
        
        # Test close context
        payload = {
            "input": {
                "websocket_message": {
                    "close_context": True,
                    "context_id": context_id
                },
                "voice_id": voice_id
            }
        }
        
        success1 = await self._send_request(payload, "Close Context", clear_audio=False)
        
        # Test close socket
        payload = {
            "input": {
                "websocket_message": {
                    "close_socket": True,
                    "context_id": context_id
                },
                "voice_id": voice_id
            }
        }
        
        success2 = await self._send_request(payload, "Close Socket", clear_audio=False)
        
        return success1 and success2
    
    async def _send_request(self, payload: Dict[str, Any], test_name: str, clear_audio: bool = True) -> bool:
        """Send request and handle streaming response"""
        if clear_audio:
            self.audio_chunks = []
        self.events = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Submit job
                async with session.post(self.run_url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"❌ {test_name} failed: {response.status} - {error_text}")
                        return False
                    
                    result = await response.json()
                    job_id = result.get("id")
                    
                    if not job_id:
                        print(f"❌ {test_name} failed: No job ID returned")
                        return False
                    
                    print(f"📤 Job submitted: {job_id}")
                    
                    # Stream results
                    success = await self._handle_streaming(session, job_id, test_name)
                    
                    if success and self.audio_chunks and AUDIO_PLAYBACK:
                        await self._play_audio()
                    
                    return success
                    
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _handle_streaming(self, session: aiohttp.ClientSession, job_id: str, test_name: str) -> bool:
        """Handle streaming response"""
        stream_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/stream/{job_id}"
        poll_count = 0
        audio_chunk_count = 0
        start_time = time.time()
        
        while poll_count < 120:  # Max 60 seconds
            poll_count += 1
            
            async with session.get(stream_url, headers=self.headers) as response:
                if response.status != 200:
                    print(f"❌ Stream failed: {response.status}")
                    return False
                
                stream_data = await response.json()
                status = stream_data.get("status")
                
                if status == "COMPLETED":
                    elapsed = time.time() - start_time
                    print(f"✅ {test_name} completed in {elapsed:.2f}s")
                    print(f"   📊 Total events: {len(self.events)}")
                    print(f"   🎵 Audio chunks: {audio_chunk_count}")
                    
                    # Show event summary
                    event_types = {}
                    for event in self.events:
                        event_type = event.get("type", "audio" if "audio" in event else "unknown")
                        event_types[event_type] = event_types.get(event_type, 0) + 1
                    
                    print(f"   📋 Event breakdown: {dict(event_types)}")
                    return True
                
                elif status == "FAILED":
                    error = stream_data.get("error", "Unknown error")
                    print(f"❌ {test_name} failed: {error}")
                    return False
                
                elif status == "IN_PROGRESS":
                    if "stream" in stream_data and stream_data["stream"]:
                        for item in stream_data["stream"]:
                            # Handle RunPod output wrapper
                            if "output" in item:
                                output = item["output"]
                            else:
                                output = item
                            
                            self.events.append(output)
                            
                            # Process different event types
                            await self._process_event(output, test_name)
                            
                            # Count audio chunks
                            if "audio" in output:
                                audio_chunk_count += 1
                                if audio_chunk_count <= 3:  # Show first few chunks
                                    print(f"   🎵 Audio chunk {audio_chunk_count}")
            
            await asyncio.sleep(0.1)
        
        print(f"❌ {test_name} timed out")
        return False
    
    async def _process_event(self, event: Dict[str, Any], test_name: str):
        """Process individual streaming events"""
        if "type" in event:
            event_type = event["type"]
            if event_type == "connection_established":
                print(f"   🔌 Connection established: {event.get('contextId')}")
            elif event_type == "generation_start":
                print(f"   🚀 Generation started: {event.get('character_count')} chars")
            elif event_type == "first_chunk":
                latency = event.get("latency_ms", 0)
                print(f"   ⚡ First chunk: {latency}ms latency")
            elif event_type == "message_start":
                idx = event.get("message_index", "?")
                print(f"   💬 Message {idx} started")
            elif event_type == "message_complete":
                idx = event.get("message_index", "?")
                print(f"   ✅ Message {idx} completed")
            elif event_type == "context_initialized":
                print(f"   🆔 Context initialized: {event.get('contextId')}")
            elif event_type == "context_closed":
                print(f"   🔒 Context closed: {event.get('contextId')}")
            elif event_type == "socket_closed":
                print(f"   🔌 Socket closed: {event.get('contextId')}")
            elif event_type == "error":
                print(f"   ❌ Error: {event.get('error')}")
        
        # Handle audio chunks
        elif "audio" in event:
            try:
                audio_data = base64.b64decode(event["audio"])
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                self.audio_chunks.append(audio_array)
            except Exception as e:
                print(f"   ⚠️ Audio decode error: {e}")
        
        # Handle alignment data
        elif "alignment" in event:
            alignment = event["alignment"]
            char_count = len(alignment.get("chars", []))
            print(f"   📍 Alignment: {char_count} characters")
        
        # Handle final completion
        elif event.get("isFinal"):
            metadata = event.get("metadata", {})
            duration = metadata.get("audio_duration_ms", 0)
            rtf = metadata.get("real_time_factor", 0)
            print(f"   🏁 Final: {duration}ms audio, {rtf:.2f}x RTF")
    
    async def _play_audio(self):
        """Play combined audio"""
        if not self.audio_chunks:
            return
        
        try:
            # Kokoro outputs at 24kHz - use correct sample rate
            sample_rate = 24000  # Kokoro's native sample rate
            
            combined_audio = np.concatenate(self.audio_chunks)
            duration = len(combined_audio) / sample_rate
            
            print(f"   🔊 Playing audio: {duration:.2f}s at {sample_rate}Hz")
            sd.play(combined_audio, samplerate=sample_rate)
            sd.wait()
            
        except Exception as e:
            print(f"   ⚠️ Audio playback error: {e}")
    
    def save_audio(self, filename: str = "test_output.wav"):
        """Save combined audio to file"""
        if not AUDIO_SAVE or not self.audio_chunks:
            print("Cannot save audio: no chunks or soundfile not available")
            return
        
        try:
            combined_audio = np.concatenate(self.audio_chunks)
            # Convert to float32 for saving (normalize from int16)
            audio_float = combined_audio.astype(np.float32) / 32767.0
            # Save at Kokoro's native 24kHz sample rate
            sf.write(filename, audio_float, 24000)
            print(f"   💾 Audio saved to: {filename}")
        except Exception as e:
            print(f"   ⚠️ Audio save error: {e}")
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("=" * 60)
        print("KOKORO TTS RUNPOD WEBSOCKET-STYLE TEST SUITE")
        print("=" * 60)
        
        results = {}
        
        # Test health
        results["health"] = await self.test_health()
        
        # Test single TTS
        results["single_tts"] = await self.test_single_tts()
        
        # Test WebSocket initialization
        results["ws_init"] = await self.test_websocket_initialization()
        
        # Test WebSocket message
        results["ws_message"] = await self.test_websocket_message()
        
        # Test conversation
        results["conversation"] = await self.test_conversation()
        
        # Test WebSocket controls
        results["ws_controls"] = await self.test_websocket_controls()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.ljust(20)}: {status}")
            if result:
                passed += 1
        
        print(f"\nResults: {passed}/{total} tests passed")
        
        if self.audio_chunks:
            self.save_audio("final_test_output.wav")
        
        return passed == total

async def main():
    parser = argparse.ArgumentParser(description="Test Kokoro TTS WebSocket-Style Handler")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument(
        "--test", 
        choices=["all", "health", "single", "websocket", "conversation", "controls"],
        default="all",
        help="Test to run"
    )
    parser.add_argument("--text", default="Hello! This is a test message.", help="Text to synthesize")
    parser.add_argument("--voice", default="af_bella", help="Voice ID")
    
    args = parser.parse_args()
    
    client = KokoroWebSocketTestClient(args.endpoint_id, args.api_key)
    
    if args.test == "all":
        success = await client.run_all_tests()
    elif args.test == "health":
        success = await client.test_health()
    elif args.test == "single":
        success = await client.test_single_tts(args.text, args.voice)
    elif args.test == "websocket":
        success = await client.test_websocket_message(args.text, args.voice)
    elif args.test == "conversation":
        messages = [
            "Welcome to our service.",
            "This is a test conversation.",
            "Thank you for trying our TTS system."
        ]
        success = await client.test_conversation(messages, args.voice)
    elif args.test == "controls":
        success = await client.test_websocket_controls(args.voice)
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))