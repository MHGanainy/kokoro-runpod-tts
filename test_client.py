#!/usr/bin/env python3
"""
Real Text Audio Test Client - Test with realistic content and audio playback
"""

import asyncio
import aiohttp
import json
import time
import argparse
import base64
import numpy as np
from typing import List, Dict, Any

# Audio playback
try:
    import sounddevice as sd
    AUDIO_PLAYBACK = True
    print("🔊 Audio playback enabled")
except ImportError:
    AUDIO_PLAYBACK = False
    print("⚠️  Audio playback disabled (install sounddevice)")

try:
    import soundfile as sf
    AUDIO_SAVE = True
    print("💾 Audio saving enabled")
except ImportError:
    AUDIO_SAVE = False
    print("⚠️  Audio saving disabled (install soundfile)")

class RealTextTester:
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.audio_chunks = []
        
    async def test_real_scenarios(self):
        """Test with realistic TTS scenarios"""
        print("🎭 REAL-WORLD TTS PERFORMANCE TEST")
        print("=" * 60)
        
        scenarios = [
            {
                "name": "📰 News Headline",
                "text": "Breaking news: Scientists have made a groundbreaking discovery in renewable energy technology that could revolutionize how we power our cities.",
                "voice": "af_bella",
                "speed": 1.0
            },
            {
                "name": "🏢 Business Presentation",
                "text": "Good morning everyone. Today's quarterly results show a significant increase in customer satisfaction, with our net promoter score rising to 8.7 out of 10.",
                "voice": "am_adam", 
                "speed": 0.9
            },
            {
                "name": "📚 Educational Content",
                "text": "Artificial intelligence is transforming industries worldwide. Machine learning algorithms can now process vast amounts of data to identify patterns humans might miss.",
                "voice": "af_sarah",
                "speed": 1.1
            },
            {
                "name": "🎧 Podcast Intro",
                "text": "Welcome back to Tech Talk Tuesday! I'm your host, and today we're diving deep into the fascinating world of voice synthesis technology.",
                "voice": "am_michael",
                "speed": 1.0
            },
            {
                "name": "📖 Audiobook Sample",
                "text": "It was the best of times, it was the worst of times. The year 2025 brought unprecedented changes to how we communicate with machines.",
                "voice": "bf_emma",
                "speed": 0.8
            }
        ]
        
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n🎯 Test {i}/5: {scenario['name']}")
            print(f"📝 Text: {scenario['text']}")
            print(f"🎤 Voice: {scenario['voice']} (Speed: {scenario['speed']}x)")
            
            result = await self._test_scenario(scenario)
            results.append(result)
            
            if result and AUDIO_PLAYBACK:
                print("🔊 Playing generated audio...")
                self._play_audio()
                
            if result and AUDIO_SAVE:
                filename = f"test_{i}_{scenario['voice']}.wav"
                self._save_audio(filename)
                
            # Clear audio for next test
            self.audio_chunks = []
            
            # Brief pause between tests
            await asyncio.sleep(1)
        
        # Show summary
        self._show_performance_summary(results)
        
    async def _test_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single TTS scenario"""
        payload = {
            "input": {
                "text": scenario["text"],
                "voice_id": scenario["voice"],
                "voice_settings": {
                    "speed": scenario["speed"]
                }
            }
        }
        
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            
            try:
                # Submit job
                async with session.post(self.run_url, json=payload, headers=self.headers) as response:
                    if response.status != 200:
                        print(f"❌ Request failed: {response.status}")
                        return None
                    
                    result = await response.json()
                    job_id = result.get("id")
                    
                    submit_time = time.perf_counter() - start_time
                    print(f"📤 Submitted in: {submit_time:.3f}s")
                    
                    # Stream results
                    metrics = await self._process_stream(session, job_id, start_time)
                    
                    return {
                        "scenario": scenario["name"],
                        "text_length": len(scenario["text"]),
                        "voice": scenario["voice"],
                        "speed": scenario["speed"],
                        **metrics
                    }
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                return None
    
    async def _process_stream(self, session: aiohttp.ClientSession, job_id: str, start_time: float) -> Dict[str, Any]:
        """Process streaming response and collect metrics"""
        stream_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/stream/{job_id}"
        
        metrics = {
            "first_chunk_latency_ms": None,
            "total_time_s": None,
            "audio_duration_ms": None,
            "real_time_factor": None,
            "chunks_received": 0,
            "gpu_used": False
        }
        
        while True:
            async with session.get(stream_url, headers=self.headers) as response:
                stream_data = await response.json()
                
                if stream_data["status"] == "COMPLETED":
                    metrics["total_time_s"] = time.perf_counter() - start_time
                    print(f"✅ Completed in: {metrics['total_time_s']:.3f}s")
                    break
                
                elif stream_data["status"] == "FAILED":
                    print(f"❌ Failed: {stream_data.get('error')}")
                    break
                
                elif stream_data["status"] == "IN_PROGRESS":
                    if "stream" in stream_data and stream_data["stream"]:
                        for item in stream_data["stream"]:
                            output = item.get("output", item)
                            
                            # Process events
                            if output.get("type") == "first_chunk":
                                metrics["first_chunk_latency_ms"] = output.get("latency_ms")
                                print(f"⚡ First chunk: {metrics['first_chunk_latency_ms']}ms")
                            
                            # Process audio
                            elif "audio" in output:
                                metrics["chunks_received"] += 1
                                try:
                                    audio_data = base64.b64decode(output["audio"])
                                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                                    self.audio_chunks.append(audio_array)
                                    
                                    if metrics["chunks_received"] <= 3:
                                        print(f"🎵 Audio chunk {metrics['chunks_received']}: {len(audio_data)} bytes")
                                    elif metrics["chunks_received"] == 4:
                                        print(f"🎵 ... (+{metrics['chunks_received']-3} more chunks)")
                                        
                                except Exception as e:
                                    print(f"⚠️  Audio decode error: {e}")
                            
                            # Process completion
                            elif output.get("isFinal"):
                                metadata = output.get("metadata", {})
                                metrics["audio_duration_ms"] = metadata.get("audio_duration_ms")
                                metrics["real_time_factor"] = metadata.get("real_time_factor")
                                metrics["gpu_used"] = metadata.get("gpu_used", False)
                                
                                print(f"🏁 Audio: {metrics['audio_duration_ms']}ms, RTF: {metrics['real_time_factor']:.3f}")
            
            await asyncio.sleep(0.05)  # Faster polling for real-time feel
        
        return metrics
    
    def _play_audio(self):
        """Play combined audio chunks"""
        if not AUDIO_PLAYBACK or not self.audio_chunks:
            return
        
        try:
            combined_audio = np.concatenate(self.audio_chunks)
            duration = len(combined_audio) / 24000
            print(f"🔊 Playing {duration:.2f}s of audio...")
            
            sd.play(combined_audio, samplerate=24000)
            sd.wait()  # Wait for playback to complete
            
        except Exception as e:
            print(f"⚠️  Playback error: {e}")
    
    def _save_audio(self, filename: str):
        """Save audio to file"""
        if not AUDIO_SAVE or not self.audio_chunks:
            return
        
        try:
            combined_audio = np.concatenate(self.audio_chunks)
            audio_float = combined_audio.astype(np.float32) / 32767.0
            sf.write(filename, audio_float, 24000)
            print(f"💾 Saved: {filename}")
        except Exception as e:
            print(f"⚠️  Save error: {e}")
    
    def _show_performance_summary(self, results: List[Dict[str, Any]]):
        """Show comprehensive performance summary"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            print("❌ No successful tests to analyze")
            return
        
        # Calculate averages
        avg_first_chunk = np.mean([r["first_chunk_latency_ms"] for r in valid_results if r["first_chunk_latency_ms"]])
        avg_total_time = np.mean([r["total_time_s"] for r in valid_results if r["total_time_s"]])
        avg_rtf = np.mean([r["real_time_factor"] for r in valid_results if r["real_time_factor"]])
        total_chunks = sum([r["chunks_received"] for r in valid_results])
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"   Average First Chunk Latency: {avg_first_chunk:.1f}ms")
        print(f"   Average Total Time: {avg_total_time:.2f}s")
        print(f"   Average Real-time Factor: {avg_rtf:.3f}x")
        print(f"   Total Audio Chunks: {total_chunks}")
        print(f"   GPU Used: {'✅' if any(r['gpu_used'] for r in valid_results) else '❌'}")
        
        print(f"\n📋 DETAILED RESULTS:")
        for r in valid_results:
            print(f"   {r['scenario']:<25} | {r['first_chunk_latency_ms']:>3.0f}ms | {r['real_time_factor']:>5.3f}x RTF | {r['voice']}")
        
        # Performance rating
        if avg_first_chunk < 100:
            rating = "🚀 EXCELLENT"
        elif avg_first_chunk < 200:
            rating = "✅ VERY GOOD"
        elif avg_first_chunk < 300:
            rating = "👍 GOOD"
        else:
            rating = "⚠️  NEEDS IMPROVEMENT"
        
        print(f"\n🏆 PERFORMANCE RATING: {rating}")
        print(f"   (Based on {avg_first_chunk:.0f}ms average first chunk latency)")

    async def test_conversation_flow(self):
        """Test realistic conversation scenario"""
        print("\n" + "=" * 60)
        print("💬 CONVERSATION FLOW TEST")
        print("=" * 60)
        
        conversation = [
            "Hello! Welcome to our AI assistant demo.",
            "I can help you with various tasks today.",
            "What would you like to know about our voice technology?",
            "Our system processes speech in real-time with minimal delay.",
            "Thank you for trying our advanced text-to-speech service!"
        ]
        
        payload = {
            "input": {
                "messages": conversation,
                "voice_id": "af_bella",
                "voice_settings": {"speed": 1.0}
            }
        }
        
        print(f"🗣️  Testing {len(conversation)} message conversation...")
        
        async with aiohttp.ClientSession() as session:
            start_time = time.perf_counter()
            
            async with session.post(self.run_url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    print(f"❌ Conversation test failed: {response.status}")
                    return
                
                result = await response.json()
                job_id = result.get("id")
                
                # Process conversation stream
                stream_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/stream/{job_id}"
                message_latencies = []
                
                while True:
                    async with session.get(stream_url, headers=self.headers) as stream_response:
                        stream_data = await stream_response.json()
                        
                        if stream_data["status"] == "COMPLETED":
                            total_time = time.perf_counter() - start_time
                            print(f"✅ Conversation completed in: {total_time:.2f}s")
                            break
                        
                        elif stream_data["status"] == "IN_PROGRESS":
                            if "stream" in stream_data and stream_data["stream"]:
                                for item in stream_data["stream"]:
                                    output = item.get("output", item)
                                    
                                    if output.get("type") == "first_chunk":
                                        msg_idx = output.get("message_index", "?")
                                        latency = output.get("latency_ms")
                                        message_latencies.append(latency)
                                        print(f"   💬 Message {msg_idx}: {latency}ms first chunk")
                                    
                                    elif "audio" in output:
                                        try:
                                            audio_data = base64.b64decode(output["audio"])
                                            audio_array = np.frombuffer(audio_data, dtype=np.int16)
                                            self.audio_chunks.append(audio_array)
                                        except:
                                            pass
                        
                        await asyncio.sleep(0.05)
        
        # Play the full conversation
        if AUDIO_PLAYBACK and self.audio_chunks:
            print("🔊 Playing full conversation...")
            self._play_audio()
        
        if AUDIO_SAVE and self.audio_chunks:
            self._save_audio("conversation_test.wav")
        
        # Show conversation metrics
        if message_latencies:
            avg_latency = np.mean(message_latencies)
            print(f"\n📊 CONVERSATION METRICS:")
            print(f"   Average message latency: {avg_latency:.1f}ms")
            print(f"   Latency consistency: {np.std(message_latencies):.1f}ms std dev")

async def main():
    parser = argparse.ArgumentParser(description="Real Text Audio Test for Kokoro TTS")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--test", choices=["scenarios", "conversation", "all"], default="all")
    
    args = parser.parse_args()
    
    tester = RealTextTester(args.endpoint_id, args.api_key)
    
    if args.test in ["scenarios", "all"]:
        await tester.test_real_scenarios()
    
    if args.test in ["conversation", "all"]:
        await tester.test_conversation_flow()
    
    print(f"\n🎉 Testing complete!")

if __name__ == "__main__":
    asyncio.run(main())