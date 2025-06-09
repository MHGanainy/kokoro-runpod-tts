#!/usr/bin/env python3
"""
Debug Test Client - Check GPU and Performance Issues
"""

import asyncio
import aiohttp
import json
import time
import argparse

class DebugClient:
    def __init__(self, endpoint_id: str, api_key: str):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.run_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def debug_health_check(self):
        """Detailed health check to see GPU status"""
        print("=== DETAILED HEALTH CHECK ===")
        
        payload = {"input": {"health_check": True}}
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            async with session.post(self.run_url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    print(f"❌ Health check failed: {response.status}")
                    return
                
                result = await response.json()
                job_id = result.get("id")
                print(f"📤 Health check job: {job_id}")
                
                # Get results
                stream_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/stream/{job_id}"
                
                while True:
                    async with session.get(stream_url, headers=self.headers) as stream_response:
                        stream_data = await stream_response.json()
                        
                        if stream_data["status"] == "COMPLETED":
                            if "output" in stream_data:
                                health_data = stream_data["output"]
                                
                                # Look for health data in the output
                                if isinstance(health_data, list):
                                    for item in health_data:
                                        if isinstance(item, dict) and "status" in item:
                                            health_data = item
                                            break
                                
                                print(f"\n🔍 HEALTH CHECK RESULTS:")
                                print(f"📋 Full response: {json.dumps(health_data, indent=2)}")
                                
                                # Extract key information
                                if isinstance(health_data, dict):
                                    print(f"\n📊 KEY METRICS:")
                                    print(f"   Status: {health_data.get('status', 'unknown')}")
                                    print(f"   Models: {health_data.get('models_loaded', 'unknown')}")
                                    print(f"   Load time: {health_data.get('load_time', 'unknown')}")
                                    print(f"   Mode: {health_data.get('mode', 'unknown')}")
                                    print(f"   Device: {health_data.get('device', 'unknown')}")
                                    
                                    # GPU-specific info
                                    if 'gpu_available' in health_data:
                                        print(f"\n🔥 GPU STATUS:")
                                        print(f"   GPU Available: {health_data.get('gpu_available', 'unknown')}")
                                        print(f"   GPU Name: {health_data.get('gpu_name', 'unknown')}")
                                        print(f"   GPU Memory Total: {health_data.get('gpu_memory_total', 'unknown')}")
                                        print(f"   GPU Memory Used: {health_data.get('gpu_memory_allocated', 'unknown')}")
                                        print(f"   GPU Memory Cached: {health_data.get('gpu_memory_cached', 'unknown')}")
                                    else:
                                        print(f"\n⚠️  NO GPU INFO FOUND - This suggests GPU optimization isn't working")
                            break
                        
                        elif stream_data["status"] == "FAILED":
                            print(f"❌ Health check failed: {stream_data.get('error')}")
                            break
                        
                        await asyncio.sleep(0.1)
                
                elapsed = time.time() - start_time
                print(f"\n⏱️  Health check took: {elapsed:.2f}s")

    async def debug_simple_tts(self):
        """Test simple TTS with detailed timing"""
        print("\n=== SIMPLE TTS DEBUG ===")
        
        text = "Hello GPU test"
        payload = {
            "input": {
                "text": text,
                "voice_id": "af_bella"
            }
        }
        
        async with aiohttp.ClientSession() as session:
            submit_time = time.time()
            
            async with session.post(self.run_url, json=payload, headers=self.headers) as response:
                if response.status != 200:
                    print(f"❌ TTS failed: {response.status}")
                    return
                
                result = await response.json()
                job_id = result.get("id")
                
                submit_elapsed = time.time() - submit_time
                print(f"📤 Job submitted in: {submit_elapsed:.3f}s")
                print(f"🆔 Job ID: {job_id}")
                
                # Stream results
                stream_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/stream/{job_id}"
                first_response_time = None
                first_chunk_time = None
                
                while True:
                    poll_start = time.time()
                    
                    async with session.get(stream_url, headers=self.headers) as stream_response:
                        stream_data = await stream_response.json()
                        
                        if first_response_time is None:
                            first_response_time = time.time() - submit_time
                            print(f"📡 First response in: {first_response_time:.3f}s")
                        
                        if stream_data["status"] == "COMPLETED":
                            total_time = time.time() - submit_time
                            print(f"✅ Completed in: {total_time:.3f}s")
                            break
                        
                        elif stream_data["status"] == "IN_PROGRESS":
                            if "stream" in stream_data and stream_data["stream"]:
                                for item in stream_data["stream"]:
                                    output = item.get("output", item)
                                    
                                    if "type" in output:
                                        event_type = output["type"]
                                        timestamp = output.get("timestamp", 0)
                                        
                                        if event_type == "first_chunk" and first_chunk_time is None:
                                            first_chunk_time = timestamp
                                            latency = output.get("latency_ms", 0)
                                            print(f"⚡ First chunk: {latency}ms latency")
                                        
                                        print(f"   📝 Event: {event_type} at {timestamp:.3f}s")
                                    
                                    elif "audio" in output:
                                        print(f"   🎵 Audio chunk received")
                                    
                                    elif output.get("isFinal"):
                                        metadata = output.get("metadata", {})
                                        rtf = metadata.get("real_time_factor", "unknown")
                                        gpu_used = metadata.get("gpu_used", "unknown")
                                        print(f"   🏁 Final - RTF: {rtf}, GPU: {gpu_used}")
                        
                        elif stream_data["status"] == "FAILED":
                            print(f"❌ Failed: {stream_data.get('error')}")
                            break
                    
                    await asyncio.sleep(0.1)

async def main():
    parser = argparse.ArgumentParser(description="Debug Kokoro TTS Performance")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    
    args = parser.parse_args()
    
    client = DebugClient(args.endpoint_id, args.api_key)
    
    # Run debug tests
    await client.debug_health_check()
    await client.debug_simple_tts()

if __name__ == "__main__":
    asyncio.run(main())