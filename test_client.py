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

# #!/usr/bin/env python3
# """
# Kokoro TTS Stress Test Client - High-load performance testing
# """

# import asyncio
# import aiohttp
# import json
# import time
# import argparse
# import base64
# import numpy as np
# from typing import List, Dict, Any, Optional
# import random
# import threading
# from dataclasses import dataclass
# from collections import defaultdict

# # Audio libraries (optional for stress testing)
# try:
#     import sounddevice as sd
#     import soundfile as sf
#     AUDIO_SUPPORT = True
# except ImportError:
#     AUDIO_SUPPORT = False

# @dataclass
# class StressMetrics:
#     total_requests: int = 0
#     successful_requests: int = 0
#     failed_requests: int = 0
#     total_audio_chunks: int = 0
#     total_audio_duration_ms: float = 0
#     total_processing_time_s: float = 0
#     first_chunk_latencies: List[float] = None
#     real_time_factors: List[float] = None
#     errors: List[str] = None
    
#     def __post_init__(self):
#         if self.first_chunk_latencies is None:
#             self.first_chunk_latencies = []
#         if self.real_time_factors is None:
#             self.real_time_factors = []
#         if self.errors is None:
#             self.errors = []

# class StressTester:
#     def __init__(self, endpoint_id: str, api_key: str):
#         self.endpoint_id = endpoint_id
#         self.api_key = api_key
#         self.run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
#         self.headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json"
#         }
#         self.metrics = StressMetrics()
#         self.active_requests = 0
#         self.peak_concurrent = 0
#         self.start_time = None
        
#         # Stress test content pools
#         self.short_texts = [
#             "Hello world, this is a quick test.",
#             "Testing TTS performance under load.",
#             "Short burst of audio generation.",
#             "Quick synthesis check.",
#             "Brief message for stress testing."
#         ]
        
#         self.medium_texts = [
#             "This is a medium-length text designed to test sustained GPU utilization during TTS inference. The goal is to create realistic workloads that mirror production usage patterns.",
#             "Artificial intelligence and machine learning are transforming how we interact with technology. Voice synthesis has become increasingly important for accessibility and user experience.",
#             "Performance testing is crucial for understanding system capabilities under various load conditions. This text provides a balanced workload for evaluation purposes.",
#             "Natural language processing has advanced significantly in recent years. Modern TTS systems can generate human-like speech with remarkable quality and efficiency.",
#             "Stress testing helps identify bottlenecks and optimization opportunities. By applying controlled load, we can measure system performance characteristics accurately."
#         ]
        
#         self.long_texts = [
#             """
#             This is an extended passage designed to create sustained GPU utilization for comprehensive stress testing. 
#             The Kokoro TTS system processes complex neural network operations that require significant computational resources. 
#             During inference, the model performs multiple matrix multiplications, attention mechanisms, and tensor transformations 
#             that fully utilize GPU compute units. This longer text ensures that each request maintains GPU activity for several 
#             seconds, allowing us to observe sustained performance under load. The goal is to validate that the system can 
#             handle concurrent requests while maintaining consistent audio quality and reasonable response times. Performance 
#             metrics collected during this test will help optimize the deployment for production workloads.
#             """,
#             """
#             Comprehensive performance evaluation requires testing with varied content lengths and complexity. This particular 
#             text segment is designed to challenge the TTS system with longer sequences that require extended processing time. 
#             The neural architecture must maintain coherent prosody and pronunciation across the entire passage while efficiently 
#             utilizing available GPU memory and compute resources. Stress testing with extended content helps identify potential 
#             memory leaks, performance degradation over time, and the system's ability to handle sustained workloads. These 
#             insights are crucial for ensuring reliable operation in production environments where the system may process 
#             continuous streams of text-to-speech requests from multiple users simultaneously.
#             """,
#             """
#             Advanced text-to-speech systems like Kokoro leverage sophisticated neural architectures to generate high-quality 
#             audio output. The inference process involves multiple stages including text preprocessing, phonetic analysis, 
#             prosody prediction, and audio synthesis. Each stage requires careful optimization to achieve optimal performance 
#             while maintaining audio quality. During stress testing, we monitor GPU utilization, memory consumption, response 
#             latency, and throughput to ensure the system meets performance requirements. This comprehensive evaluation helps 
#             identify optimal configurations for different deployment scenarios and user requirements. The results inform 
#             scaling decisions and infrastructure planning for production deployments.
#             """
#         ]
        
#         self.voices = [
#             "af_bella", "af_sarah", "af_jessica", "af_heart",
#             "am_adam", "am_michael", "am_cooper", "am_jackson",
#             "bf_emma", "bf_isabella", "bm_george", "bm_william"
#         ]

#     async def run_stress_test(self, 
#                              duration_seconds: int = 60,
#                              max_concurrent: int = 10,
#                              ramp_up_seconds: int = 10,
#                              test_type: str = "mixed"):
#         """Main stress test orchestrator"""
#         print("🔥 KOKORO TTS STRESS TEST")
#         print("=" * 70)
#         print(f"Duration: {duration_seconds}s | Max Concurrent: {max_concurrent} | Ramp-up: {ramp_up_seconds}s")
#         print(f"Test Type: {test_type}")
#         print("=" * 70)
        
#         self.start_time = time.perf_counter()
        
#         # Start monitoring thread
#         monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
#         monitor_thread.start()
        
#         if test_type == "burst":
#             await self._burst_test(max_concurrent)
#         elif test_type == "sustained":
#             await self._sustained_load_test(duration_seconds, max_concurrent, ramp_up_seconds)
#         elif test_type == "spike":
#             await self._spike_test(duration_seconds, max_concurrent)
#         elif test_type == "mixed":
#             await self._mixed_workload_test(duration_seconds, max_concurrent, ramp_up_seconds)
#         else:
#             await self._custom_stress_test(duration_seconds, max_concurrent)
        
#         # Wait for remaining requests
#         print("\n⏳ Waiting for remaining requests to complete...")
#         while self.active_requests > 0:
#             await asyncio.sleep(0.5)
        
#         # Generate final report
#         self._generate_stress_report()

#     async def _burst_test(self, concurrent_requests: int):
#         """Launch many requests simultaneously"""
#         print(f"\n💥 BURST TEST - {concurrent_requests} simultaneous requests")
        
#         tasks = []
#         for i in range(concurrent_requests):
#             text = random.choice(self.medium_texts)
#             voice = random.choice(self.voices)
#             speed = round(random.uniform(0.8, 1.2), 1)
            
#             task = asyncio.create_task(
#                 self._single_request(f"burst_{i}", text, voice, speed)
#             )
#             tasks.append(task)
        
#         await asyncio.gather(*tasks, return_exceptions=True)

#     async def _sustained_load_test(self, duration: int, max_concurrent: int, ramp_up: int):
#         """Gradually increase load to target level and maintain"""
#         print(f"\n📈 SUSTAINED LOAD TEST")
        
#         end_time = time.perf_counter() + duration
#         ramp_end = time.perf_counter() + ramp_up
        
#         request_id = 0
        
#         while time.perf_counter() < end_time:
#             current_time = time.perf_counter()
            
#             # Calculate target concurrent requests (ramp up)
#             if current_time < ramp_end:
#                 progress = (current_time - self.start_time) / ramp_up
#                 target_concurrent = int(max_concurrent * progress)
#             else:
#                 target_concurrent = max_concurrent
            
#             # Launch requests to reach target
#             while self.active_requests < target_concurrent and current_time < end_time:
#                 request_id += 1
#                 text = self._select_text_by_weight()
#                 voice = random.choice(self.voices)
#                 speed = round(random.uniform(0.8, 1.2), 1)
                
#                 asyncio.create_task(
#                     self._single_request(f"sustained_{request_id}", text, voice, speed)
#                 )
                
#                 await asyncio.sleep(0.1)  # Small delay between launches
#                 current_time = time.perf_counter()
            
#             await asyncio.sleep(0.5)

#     async def _spike_test(self, duration: int, max_concurrent: int):
#         """Random spikes in load"""
#         print(f"\n⚡ SPIKE TEST - Random load spikes")
        
#         end_time = time.perf_counter() + duration
#         request_id = 0
        
#         while time.perf_counter() < end_time:
#             # Random spike intensity
#             spike_size = random.randint(1, max_concurrent)
#             spike_duration = random.uniform(1, 5)
            
#             print(f"📊 Spike: {spike_size} requests for {spike_duration:.1f}s")
            
#             # Launch spike
#             for i in range(spike_size):
#                 request_id += 1
#                 text = self._select_text_by_weight()
#                 voice = random.choice(self.voices)
#                 speed = round(random.uniform(0.7, 1.3), 1)
                
#                 asyncio.create_task(
#                     self._single_request(f"spike_{request_id}", text, voice, speed)
#                 )
            
#             await asyncio.sleep(spike_duration)
            
#             # Random cooldown
#             cooldown = random.uniform(0.5, 3.0)
#             await asyncio.sleep(cooldown)

#     async def _mixed_workload_test(self, duration: int, max_concurrent: int, ramp_up: int):
#         """Mixed workload with different patterns"""
#         print(f"\n🎭 MIXED WORKLOAD TEST")
        
#         # Start with sustained load
#         await asyncio.create_task(
#             self._sustained_load_test(duration // 3, max_concurrent // 2, ramp_up)
#         )
        
#         # Add burst patterns
#         await asyncio.create_task(
#             self._burst_test(max_concurrent)
#         )
        
#         # Finish with spikes
#         await asyncio.create_task(
#             self._spike_test(duration // 3, max_concurrent)
#         )

#     async def _custom_stress_test(self, duration: int, max_concurrent: int):
#         """Custom stress pattern"""
#         print(f"\n🔧 CUSTOM STRESS TEST")
#         await self._sustained_load_test(duration, max_concurrent, 5)

#     def _select_text_by_weight(self) -> str:
#         """Select text with weighted probability"""
#         rand = random.random()
#         if rand < 0.3:  # 30% short
#             return random.choice(self.short_texts)
#         elif rand < 0.8:  # 50% medium  
#             return random.choice(self.medium_texts)
#         else:  # 20% long
#             return random.choice(self.long_texts)

#     async def _single_request(self, request_id: str, text: str, voice: str, speed: float):
#         """Execute single TTS request with metrics collection"""
#         self.active_requests += 1
#         self.peak_concurrent = max(self.peak_concurrent, self.active_requests)
#         self.metrics.total_requests += 1
        
#         payload = {
#             "input": {
#                 "text": text,
#                 "voice_id": voice,
#                 "voice_settings": {"speed": speed}
#             }
#         }
        
#         request_start = time.perf_counter()
        
#         try:
#             async with aiohttp.ClientSession() as session:
#                 # Submit request
#                 async with session.post(self.run_url, json=payload, headers=self.headers) as response:
#                     if response.status != 200:
#                         self.metrics.failed_requests += 1
#                         self.metrics.errors.append(f"{request_id}: HTTP {response.status}")
#                         return
                    
#                     result = await response.json()
#                     job_id = result.get("id")
                    
#                     # Process streaming response
#                     await self._process_request_stream(session, job_id, request_id, request_start)
                    
#         except Exception as e:
#             self.metrics.failed_requests += 1
#             self.metrics.errors.append(f"{request_id}: {str(e)}")
#         finally:
#             self.active_requests -= 1

#     async def _process_request_stream(self, session: aiohttp.ClientSession, 
#                                     job_id: str, request_id: str, request_start: float):
#         """Process streaming response and collect metrics"""
#         stream_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/stream/{job_id}"
        
#         chunks_received = 0
#         first_chunk_time = None
        
#         while True:
#             async with session.get(stream_url, headers=self.headers) as response:
#                 stream_data = await response.json()
                
#                 if stream_data["status"] == "COMPLETED":
#                     self.metrics.successful_requests += 1
#                     self.metrics.total_processing_time_s += time.perf_counter() - request_start
#                     break
                
#                 elif stream_data["status"] == "FAILED":
#                     self.metrics.failed_requests += 1
#                     error_msg = stream_data.get("error", "Unknown error")
#                     self.metrics.errors.append(f"{request_id}: {error_msg}")
#                     break
                
#                 elif stream_data["status"] == "IN_PROGRESS":
#                     if "stream" in stream_data and stream_data["stream"]:
#                         for item in stream_data["stream"]:
#                             output = item.get("output", item)
                            
#                             # First chunk timing
#                             if output.get("type") == "first_chunk" and first_chunk_time is None:
#                                 first_chunk_time = output.get("latency_ms")
#                                 if first_chunk_time:
#                                     self.metrics.first_chunk_latencies.append(first_chunk_time)
                            
#                             # Audio chunks
#                             elif "audio" in output:
#                                 chunks_received += 1
#                                 self.metrics.total_audio_chunks += 1
                            
#                             # Final metrics
#                             elif output.get("isFinal"):
#                                 metadata = output.get("metadata", {})
                                
#                                 audio_duration = metadata.get("audio_duration_ms")
#                                 if audio_duration:
#                                     self.metrics.total_audio_duration_ms += audio_duration
                                
#                                 rtf = metadata.get("real_time_factor")
#                                 if rtf:
#                                     self.metrics.real_time_factors.append(rtf)
            
#             await asyncio.sleep(0.05)

#     def _monitor_performance(self):
#         """Background performance monitoring"""
#         while True:
#             elapsed = time.perf_counter() - self.start_time
#             rps = self.metrics.total_requests / elapsed if elapsed > 0 else 0
#             success_rate = (self.metrics.successful_requests / self.metrics.total_requests * 100) if self.metrics.total_requests > 0 else 0
            
#             print(f"\r📊 Active: {self.active_requests:2d} | Total: {self.metrics.total_requests:4d} | RPS: {rps:5.1f} | Success: {success_rate:5.1f}% | Elapsed: {elapsed:6.1f}s", end="", flush=True)
            
#             time.sleep(1.0)

#     def _generate_stress_report(self):
#         """Generate comprehensive stress test report"""
#         elapsed = time.perf_counter() - self.start_time
        
#         print("\n\n" + "=" * 70)
#         print("📊 STRESS TEST RESULTS")
#         print("=" * 70)
        
#         # Request statistics
#         print(f"\n🎯 REQUEST STATISTICS:")
#         print(f"   Total Requests: {self.metrics.total_requests}")
#         print(f"   Successful: {self.metrics.successful_requests}")
#         print(f"   Failed: {self.metrics.failed_requests}")
#         print(f"   Success Rate: {self.metrics.successful_requests/self.metrics.total_requests*100:.1f}%")
#         print(f"   Peak Concurrent: {self.peak_concurrent}")
        
#         # Performance metrics
#         if self.metrics.successful_requests > 0:
#             avg_processing_time = self.metrics.total_processing_time_s / self.metrics.successful_requests
#             requests_per_second = self.metrics.total_requests / elapsed
            
#             print(f"\n⚡ PERFORMANCE METRICS:")
#             print(f"   Test Duration: {elapsed:.1f}s")
#             print(f"   Requests per Second: {requests_per_second:.2f}")
#             print(f"   Average Processing Time: {avg_processing_time:.3f}s")
#             print(f"   Total Audio Generated: {self.metrics.total_audio_duration_ms/1000:.1f}s")
#             print(f"   Total Audio Chunks: {self.metrics.total_audio_chunks}")
        
#         # Latency analysis
#         if self.metrics.first_chunk_latencies:
#             latencies = np.array(self.metrics.first_chunk_latencies)
#             print(f"\n🚀 LATENCY ANALYSIS:")
#             print(f"   Average First Chunk: {np.mean(latencies):.1f}ms")
#             print(f"   Median First Chunk: {np.median(latencies):.1f}ms")
#             print(f"   P95 First Chunk: {np.percentile(latencies, 95):.1f}ms")
#             print(f"   P99 First Chunk: {np.percentile(latencies, 99):.1f}ms")
#             print(f"   Min/Max: {np.min(latencies):.1f}ms / {np.max(latencies):.1f}ms")
        
#         # Real-time factor analysis
#         if self.metrics.real_time_factors:
#             rtfs = np.array(self.metrics.real_time_factors)
#             print(f"\n⏱️  REAL-TIME FACTOR ANALYSIS:")
#             print(f"   Average RTF: {np.mean(rtfs):.3f}x")
#             print(f"   Median RTF: {np.median(rtfs):.3f}x")
#             print(f"   Best RTF: {np.min(rtfs):.3f}x")
#             print(f"   Worst RTF: {np.max(rtfs):.3f}x")
        
#         # Error analysis
#         if self.metrics.errors:
#             print(f"\n❌ ERRORS ({len(self.metrics.errors)}):")
#             error_counts = defaultdict(int)
#             for error in self.metrics.errors[:10]:  # Show first 10
#                 error_type = error.split(":")[1] if ":" in error else error
#                 error_counts[error_type] += 1
            
#             for error_type, count in error_counts.items():
#                 print(f"   {error_type}: {count}")
            
#             if len(self.metrics.errors) > 10:
#                 print(f"   ... and {len(self.metrics.errors) - 10} more")
        
#         # Performance rating
#         if self.metrics.first_chunk_latencies:
#             avg_latency = np.mean(self.metrics.first_chunk_latencies)
#             success_rate = self.metrics.successful_requests / self.metrics.total_requests * 100
            
#             if avg_latency < 150 and success_rate > 95:
#                 rating = "🚀 EXCELLENT"
#             elif avg_latency < 300 and success_rate > 90:
#                 rating = "✅ VERY GOOD" 
#             elif avg_latency < 500 and success_rate > 85:
#                 rating = "👍 GOOD"
#             else:
#                 rating = "⚠️  NEEDS OPTIMIZATION"
            
#             print(f"\n🏆 OVERALL RATING: {rating}")
#             print(f"   Based on {avg_latency:.0f}ms avg latency and {success_rate:.1f}% success rate")

# async def main():
#     parser = argparse.ArgumentParser(description="Kokoro TTS Stress Test")
#     parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
#     parser.add_argument("--api-key", required=True, help="RunPod API key")
#     parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
#     parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent requests")
#     parser.add_argument("--ramp-up", type=int, default=10, help="Ramp-up time in seconds")
#     parser.add_argument("--test-type", choices=["burst", "sustained", "spike", "mixed", "custom"], 
#                        default="mixed", help="Type of stress test")
    
#     args = parser.parse_args()
    
#     tester = StressTester(args.endpoint_id, args.api_key)
    
#     await tester.run_stress_test(
#         duration_seconds=args.duration,
#         max_concurrent=args.max_concurrent, 
#         ramp_up_seconds=args.ramp_up,
#         test_type=args.test_type
#     )

# if __name__ == "__main__":
#     asyncio.run(main())