#!/usr/bin/env python3
"""
Kokoro Library Optimized Handler - Maximum Performance
Uses Kokoro's built-in optimizations for sub-50ms latency
"""

import runpod
import base64
import time
import numpy as np
import json
from typing import Dict, Generator, Any, Optional
import logging
import os

# Minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# GPU optimization
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    DEVICE = "cuda"
else:
    torch.set_num_threads(2)
    DEVICE = "cpu"

# Import Kokoro with optimizations
try:
    from kokoro import KModel, KPipeline
except ImportError:
    logger.error("Kokoro not installed")
    raise

# Global optimized state
SHARED_MODEL = None
PIPELINES = {}
LOAD_START_TIME = time.perf_counter()

def initialize_optimized_kokoro():
    """Initialize Kokoro with maximum optimization"""
    global SHARED_MODEL, PIPELINES
    
    try:
        # Single shared model for all languages (Kokoro's recommended approach)
        logger.warning("Loading shared Kokoro model...")
        start = time.perf_counter()
        
        # Use Kokoro's automatic device selection with explicit GPU preference
        SHARED_MODEL = KModel(device=DEVICE).eval()
        
        # GPU memory optimization
        if torch.cuda.is_available():
            SHARED_MODEL = SHARED_MODEL.to(DEVICE)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model_time = time.perf_counter() - start
        logger.warning(f"Shared model loaded in {model_time:.2f}s")
        
        # Initialize pipelines for supported languages
        languages = {'a': 'American English', 'b': 'British English'}
        
        for lang_code, lang_name in languages.items():
            try:
                pipeline_start = time.perf_counter()
                
                # Create pipeline with shared model (Kokoro's optimization)
                pipeline = KPipeline(
                    lang_code=lang_code,
                    model=SHARED_MODEL,  # Reuse the same model
                    device=DEVICE
                )
                
                # Pre-load common voices for this language
                common_voices = get_common_voices(lang_code)
                for voice in common_voices:
                    try:
                        pipeline.load_voice(voice)  # This caches the voice
                    except:
                        pass
                
                PIPELINES[lang_code] = pipeline
                pipeline_time = time.perf_counter() - pipeline_start
                logger.warning(f"Pipeline {lang_name} ready in {pipeline_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to initialize {lang_name}: {e}")
        
        # Final GPU optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        total_time = time.perf_counter() - LOAD_START_TIME
        logger.warning(f"Kokoro optimization complete in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Kokoro initialization failed: {e}")
        raise

def get_common_voices(lang_code: str) -> list:
    """Get list of common voices for pre-loading"""
    common_voices = {
        'a': ['af_bella', 'af_sarah', 'am_adam', 'am_michael'],
        'b': ['bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis']
    }
    return common_voices.get(lang_code, ['af_bella'])

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
    """Ultra-optimized handler using Kokoro's built-in optimizations"""
    job_start = time.perf_counter()
    
    try:
        job_input = job["input"]
        
        # Health check
        if job_input.get("health_check"):
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_available": True,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                    "device": DEVICE
                }
            else:
                gpu_info = {"gpu_available": False, "device": DEVICE}
            
            yield {
                "status": "healthy",
                "models_loaded": list(PIPELINES.keys()),
                "mode": "kokoro_optimized",
                "shared_model": SHARED_MODEL is not None,
                **gpu_info
            }
            return
        
        # Route to appropriate handler
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
    """Direct TTS using Kokoro pipeline"""
    text = job_input.get("text", "")
    if not text:
        yield {"error": "No text"}
        return
    
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    context_id = f"d{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    
    yield from generate_with_kokoro(text, voice_id, voice_settings, context_id, job_start)

def handle_websocket_message(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """WebSocket message using Kokoro"""
    ws_msg = job_input.get("websocket_message", {})
    text = ws_msg.get("text", "").strip()
    context_id = ws_msg.get("context_id") or f"ws{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    voice_settings = ws_msg.get("voice_settings", {})
    voice_id = job_input.get("voice_id", "af_bella")
    
    # Fast control messages
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
    
    yield from generate_with_kokoro(text, voice_id, voice_settings, context_id, job_start)

def handle_conversation(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Conversation using Kokoro"""
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
        
        yield {"type": "message_start", "contextId": context_id, "message_index": i}
        yield from generate_with_kokoro(text, voice_id, voice_settings, context_id, job_start, i)
        yield {"type": "message_complete", "contextId": context_id, "message_index": i}

def generate_with_kokoro(
    text: str,
    voice_id: str,
    voice_settings: Dict[str, Any],
    context_id: str,
    job_start: float,
    message_index: Optional[int] = None
) -> Generator[Dict[str, Any], None, float]:
    """Generate audio using optimized Kokoro pipeline"""
    
    # Select pipeline based on voice
    lang_code = voice_id[0] if voice_id and voice_id[0] in PIPELINES else 'a'
    pipeline = PIPELINES.get(lang_code, PIPELINES['a'])
    
    speed = voice_settings.get("speed", 1.0)
    
    chunk_count = 0
    first_chunk_time = None
    generation_start = time.perf_counter()
    audio_chunks = []
    
    try:
        # Use Kokoro's optimized generation with the shared model
        for result in pipeline(text, voice=voice_id, speed=speed):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - job_start
                
                # Send first chunk timing
                yield {
                    "type": "first_chunk",
                    "contextId": context_id,
                    "latency_ms": int(first_chunk_time * 1000)
                }
            
            # Extract audio from Kokoro result
            if result.audio is not None:
                chunk_count += 1
                
                # Convert to PCM16 bytes
                audio_np = result.audio.numpy()
                audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
                audio_chunks.append(audio_np)
                
                # Send audio chunk
                chunk_data = {
                    "audio": base64.b64encode(audio_bytes).decode('utf-8'),
                    "contextId": context_id,
                    "isFinal": False,
                    "chunk_number": chunk_count,
                    "sample_rate": 24000
                }
                
                if message_index is not None:
                    chunk_data["message_index"] = message_index
                    
                yield chunk_data
        
        # Final processing
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / 24000.0
            generation_time = time.perf_counter() - generation_start
            
            # Send alignment
            alignment = create_minimal_alignment(text, audio_duration * 1000)
            yield {"alignment": alignment, "contextId": context_id}
            
            # Send completion
            yield {
                "isFinal": True,
                "contextId": context_id,
                "metadata": {
                    "total_chunks": chunk_count,
                    "audio_duration_ms": int(audio_duration * 1000),
                    "generation_time_ms": int(generation_time * 1000),
                    "real_time_factor": generation_time / audio_duration if audio_duration > 0 else 0,
                    "gpu_used": torch.cuda.is_available(),
                    "kokoro_optimized": True
                }
            }
            
            return audio_duration
        
        return 0.0
            
    except Exception as e:
        yield {"error": str(e), "contextId": context_id}
        return 0.0
    finally:
        # Kokoro handles GPU memory management internally, but we can still clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Initialize Kokoro on import
initialize_optimized_kokoro()

# Start RunPod with minimal overhead
runpod.serverless.start({
    "handler": ultra_fast_handler,
    "return_aggregate_stream": True
})