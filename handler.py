#!/usr/bin/env python3
"""
Ultra-Optimized Kokoro TTS Handler - Absolute Minimum Latency
Eliminates all unnecessary overhead for sub-100ms first chunk latency
"""

import runpod
import base64
import time
import numpy as np
import json
from typing import Dict, Generator, Any, Optional
import logging

# Minimal logging for maximum performance
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# GPU setup
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Aggressive GPU optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)  # Minimal CPU threads for GPU workloads
    DEVICE = torch.device("cuda:0")
else:
    torch.set_num_threads(2)
    DEVICE = torch.device("cpu")

# Import Kokoro
try:
    from kokoro import KPipeline
except ImportError:
    logger.error("Kokoro not installed")
    raise

# Global state
PIPELINES = {}
LOAD_START_TIME = time.perf_counter()

def initialize_pipelines():
    """Ultra-fast pipeline initialization"""
    global PIPELINES
    
    languages = {'a': 'American English', 'b': 'British English'}
    
    for lang_code, lang_name in languages.items():
        try:
            start = time.perf_counter()
            pipeline = KPipeline(lang_code=lang_code)
            
            # Minimal warm-up - just one inference
            list(pipeline("hi", voice='af_bella'))
            
            # GPU memory optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            PIPELINES[lang_code] = pipeline
            logger.warning(f"Loaded {lang_name} in {time.perf_counter() - start:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load {lang_name}: {e}")
    
    total_time = time.perf_counter() - LOAD_START_TIME
    logger.warning(f"All models loaded in {total_time:.2f}s")

# Initialize immediately
initialize_pipelines()

def create_minimal_alignment(text: str, audio_duration_ms: float) -> dict:
    """Ultra-fast alignment calculation"""
    if not text:
        return {"chars": [], "charStartTimesMs": [], "charsDurationsMs": []}
    
    char_count = len(text)
    char_duration = audio_duration_ms / char_count
    
    return {
        "chars": list(text),
        "charStartTimesMs": [int(i * char_duration) for i in range(char_count)],
        "charsDurationsMs": [int(char_duration)] * char_count
    }

def ultra_fast_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Ultra-optimized handler with minimal overhead"""
    job_start = time.perf_counter()
    
    try:
        job_input = job["input"]
        
        # Health check fast path
        if job_input.get("health_check"):
            yield {
                "status": "healthy",
                "models_loaded": list(PIPELINES.keys()),
                "mode": "ultra_optimized",
                "device": str(DEVICE),
                "gpu_available": torch.cuda.is_available()
            }
            return
        
        # Direct TTS fast path
        if "text" in job_input:
            yield from handle_direct_tts(job_input, job_start)
        elif "websocket_message" in job_input:
            yield from handle_websocket_message(job_input, job_start)
        elif "messages" in job_input:
            yield from handle_conversation(job_input, job_start)
        else:
            yield {"error": "Invalid input"}
            
    except Exception as e:
        yield {"error": str(e)}

def handle_direct_tts(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Fastest possible single TTS"""
    text = job_input.get("text", "")
    if not text:
        yield {"error": "No text"}
        return
    
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    context_id = f"f{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    
    # Skip start event for maximum speed
    # yield {"type": "generation_start", "contextId": context_id}
    
    yield from generate_ultra_fast_audio(text, voice_id, voice_settings, context_id, job_start)

def handle_websocket_message(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Optimized WebSocket message"""
    ws_msg = job_input.get("websocket_message", {})
    text = ws_msg.get("text", "").strip()
    context_id = ws_msg.get("context_id") or f"ws{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    voice_settings = ws_msg.get("voice_settings", {})
    voice_id = job_input.get("voice_id", "af_bella")
    
    # Fast control message handling
    if ws_msg.get("close_socket"):
        yield {"type": "socket_closed", "contextId": context_id}
        return
    if ws_msg.get("close_context"):
        yield {"type": "context_closed", "contextId": context_id}
        return
    if text == " ":
        yield {"type": "context_initialized", "contextId": context_id}
        return
    if not text:
        yield {"type": "empty_message", "contextId": context_id}
        return
    
    # Skip start event for speed
    yield from generate_ultra_fast_audio(text, voice_id, voice_settings, context_id, job_start)

def handle_conversation(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Optimized conversation"""
    messages = job_input.get("messages", [])
    if not messages:
        return
        
    context_id = f"cv{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    
    yield {"type": "connection_established", "contextId": context_id}
    
    for i, message in enumerate(messages):
        text = message if isinstance(message, str) else message.get("text", "").strip()
        if not text:
            continue
        
        # Minimal events
        yield {"type": "message_start", "contextId": context_id, "message_index": i}
        yield from generate_ultra_fast_audio(text, voice_id, voice_settings, context_id, job_start, i)
        yield {"type": "message_complete", "contextId": context_id, "message_index": i}

def generate_ultra_fast_audio(
    text: str,
    voice_id: str,
    voice_settings: Dict[str, Any],
    context_id: str,
    job_start: float,
    message_index: Optional[int] = None
) -> Generator[Dict[str, Any], None, float]:
    """Absolute minimum latency audio generation"""
    
    # Fast pipeline selection
    lang_code = voice_id[0] if voice_id and voice_id[0] in PIPELINES else 'a'
    pipeline = PIPELINES.get(lang_code, PIPELINES['a'])
    speed = voice_settings.get("speed", 1.0)
    
    audio_chunks = []
    chunk_count = 0
    first_chunk_time = None
    generation_start = time.perf_counter()
    
    try:
        # Ultra-fast generation loop
        for graphemes, phonemes, audio in pipeline(text, voice=voice_id, speed=speed):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - job_start
                
                # Optional: Comment out for absolute minimum latency
                yield {
                    "type": "first_chunk",
                    "contextId": context_id,
                    "latency_ms": int(first_chunk_time * 1000)
                }
            
            # Ultra-fast audio processing
            if hasattr(audio, 'detach'):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio
            
            # Single-step conversion
            audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
            audio_chunks.append(audio_np)
            chunk_count += 1
            
            # Minimal chunk data
            chunk_data = {
                "audio": base64.b64encode(audio_bytes).decode('utf-8'),
                "contextId": context_id,
                "isFinal": False
            }
            
            if message_index is not None:
                chunk_data["message_index"] = message_index
                
            yield chunk_data
        
        # Final processing - minimal overhead
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / 24000.0
            generation_time = time.perf_counter() - generation_start
            
            # Optional alignment (comment out for speed)
            alignment = create_minimal_alignment(text, audio_duration * 1000)
            yield {"alignment": alignment, "contextId": context_id}
            
            # Minimal completion
            yield {
                "isFinal": True,
                "contextId": context_id,
                "metadata": {
                    "total_chunks": chunk_count,
                    "audio_duration_ms": int(audio_duration * 1000),
                    "generation_time_ms": int(generation_time * 1000),
                    "real_time_factor": generation_time / audio_duration if audio_duration > 0 else 0,
                    "gpu_used": torch.cuda.is_available()
                }
            }
            
            return audio_duration
        
        return 0.0
            
    except Exception as e:
        yield {"error": str(e), "contextId": context_id}
        return 0.0
    finally:
        # Quick GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Start with minimal overhead
runpod.serverless.start({
    "handler": ultra_fast_handler,
    "return_aggregate_stream": True
})