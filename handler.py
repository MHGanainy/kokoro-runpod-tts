#!/usr/bin/env python3
"""
Fixed Safe Optimized GPU Handler - Corrected voice caching issue
"""

import runpod
import base64
import time
import numpy as np
import json
from typing import Dict, Generator, Any, Optional
import logging
import os
import threading

# Minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Force GPU usage with safe optimizations
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Safe GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(2)
    
    # Force all operations to GPU
    torch.cuda.set_device(0)
    DEVICE = torch.device("cuda:0")
    
    # Test GPU computation
    test_tensor = torch.randn(1000, 1000, device=DEVICE)
    _ = torch.mm(test_tensor, test_tensor)
    torch.cuda.synchronize()
    print(f"✅ GPU computation test passed")
    
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA not available, using CPU")

# Import Kokoro
try:
    from kokoro import KModel, KPipeline
except ImportError:
    logger.error("Kokoro not installed")
    raise

# Global state with voice caching
SHARED_MODEL = None
PIPELINES = {}
LOAD_START_TIME = time.perf_counter()

# Voice cache for performance
VOICE_CACHE = {}
VOICE_CACHE_LOCK = threading.Lock()

def safe_force_gpu_usage():
    """Safe GPU optimization without risky dtype changes"""
    global SHARED_MODEL, PIPELINES
    
    try:
        logger.warning("Initializing safe optimized Kokoro...")
        start = time.perf_counter()
        
        # Create model with safe optimizations
        SHARED_MODEL = KModel().eval()
        
        if torch.cuda.is_available():
            print(f"Moving model to GPU: {DEVICE}")
            SHARED_MODEL = SHARED_MODEL.to(DEVICE)
            
            # Verify model is on GPU
            for name, param in SHARED_MODEL.named_parameters():
                if not param.is_cuda:
                    param.data = param.data.to(DEVICE)
            
            # Safe compilation (if available)
            try:
                SHARED_MODEL = torch.compile(SHARED_MODEL, mode="reduce-overhead")
                print("✅ Model compiled for faster inference")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}, using standard model")
            
            # Force GPU memory allocation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Test inference with safe dtypes
            print("Testing safe GPU inference...")
            dummy_input = torch.LongTensor([[0, 1, 2, 0]]).to(DEVICE)
            dummy_ref = torch.randn(1, 256).to(DEVICE)
            
            with torch.no_grad():
                _ = SHARED_MODEL.forward_with_tokens(dummy_input, dummy_ref, 1.0)
            
            torch.cuda.synchronize()
            print(f"✅ Safe GPU inference test completed")
        
        model_time = time.perf_counter() - start
        logger.warning(f"Safe optimized model loaded in {model_time:.2f}s")
        
        # Initialize pipelines with safe optimizations
        languages = {'a': 'American English', 'b': 'British English'}
        
        for lang_code, lang_name in languages.items():
            try:
                pipeline_start = time.perf_counter()
                
                pipeline = KPipeline(
                    lang_code=lang_code,
                    model=SHARED_MODEL,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Pre-load and cache common voices
                common_voices = get_common_voices(lang_code)
                for voice in common_voices:
                    try:
                        voice_tensor = pipeline.load_voice(voice)
                        if torch.cuda.is_available():
                            voice_tensor = voice_tensor.to(DEVICE)
                            pipeline.voices[voice] = voice_tensor
                            
                            # Global cache
                            with VOICE_CACHE_LOCK:
                                VOICE_CACHE[voice] = voice_tensor
                                
                    except Exception as e:
                        print(f"⚠️ Voice {voice} loading failed: {e}")
                
                PIPELINES[lang_code] = pipeline
                pipeline_time = time.perf_counter() - pipeline_start
                logger.warning(f"Safe pipeline {lang_name} ready in {pipeline_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to initialize {lang_name}: {e}")
        
        # Memory status
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 Safe GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        total_time = time.perf_counter() - LOAD_START_TIME
        logger.warning(f"Complete safe initialization in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Safe GPU initialization failed: {e}")
        raise

def get_common_voices(lang_code: str) -> list:
    """Get common voices for pre-loading"""
    return {
        'a': ['af_bella', 'af_sarah', 'am_adam', 'af_jessica'],
        'b': ['bf_emma', 'bm_george', 'bf_isabella']
    }.get(lang_code, ['af_bella'])

def get_cached_voice(voice_id: str, pipeline) -> torch.FloatTensor:
    """Get voice from cache or load it efficiently"""
    with VOICE_CACHE_LOCK:
        if voice_id in VOICE_CACHE:
            return VOICE_CACHE[voice_id]
    
    # Load voice if not cached
    voice_tensor = pipeline.load_voice(voice_id)
    if torch.cuda.is_available():
        voice_tensor = voice_tensor.to(DEVICE)
    
    # Cache it
    with VOICE_CACHE_LOCK:
        VOICE_CACHE[voice_id] = voice_tensor
    
    return voice_tensor

def create_fast_alignment(text: str, audio_duration_ms: float) -> dict:
    """Fast alignment calculation"""
    if not text or audio_duration_ms <= 0:
        return {"chars": [], "charStartTimesMs": [], "charsDurationsMs": []}
    
    char_count = len(text)
    if char_count == 0:
        return {"chars": [], "charStartTimesMs": [], "charsDurationsMs": []}
    
    char_duration = int(audio_duration_ms // char_count)
    
    return {
        "chars": list(text),
        "charStartTimesMs": [i * char_duration for i in range(char_count)],
        "charsDurationsMs": [char_duration] * char_count
    }

def safe_optimized_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Safe optimized handler"""
    job_start = time.perf_counter()
    
    try:
        job_input = job["input"]
        
        # Health check with optimization info
        if job_input.get("health_check"):
            gpu_info = {"gpu_available": False}
            
            if torch.cuda.is_available():
                gpu_info = {
                    "gpu_available": True,
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
                    "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                    "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                    "device": str(DEVICE),
                    "model_on_gpu": next(SHARED_MODEL.parameters()).is_cuda if SHARED_MODEL else False,
                    "optimizations": "Voice Cache, Model Compilation, Fast Alignment"
                }
            
            yield {
                "status": "healthy",
                "models_loaded": list(PIPELINES.keys()),
                "mode": "safe_optimized",
                "shared_model": SHARED_MODEL is not None,
                "voice_cache_size": len(VOICE_CACHE),
                **gpu_info
            }
            return
        
        # Route requests
        if "text" in job_input:
            yield from handle_safe_tts(job_input, job_start)
        elif "websocket_message" in job_input:
            yield from handle_safe_websocket(job_input, job_start)
        elif "messages" in job_input:
            yield from handle_safe_conversation(job_input, job_start)
        else:
            yield {"error": "Invalid input"}
            
    except Exception as e:
        yield {"error": str(e)}

def handle_safe_tts(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Safe TTS handling"""
    text = job_input.get("text", "")
    if not text:
        yield {"error": "No text"}
        return
    
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    context_id = f"safe{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    
    yield from generate_safe_audio(text, voice_id, voice_settings, context_id, job_start)

def handle_safe_websocket(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Safe WebSocket handling"""
    ws_msg = job_input.get("websocket_message", {})
    text = ws_msg.get("text", "").strip()
    context_id = ws_msg.get("context_id") or f"ws{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    voice_settings = ws_msg.get("voice_settings", {})
    voice_id = job_input.get("voice_id", "af_bella")
    
    # Control messages
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
    
    yield from generate_safe_audio(text, voice_id, voice_settings, context_id, job_start)

def handle_safe_conversation(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Safe conversation handling"""
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
        yield from generate_safe_audio(text, voice_id, voice_settings, context_id, job_start, i)
        yield {"type": "message_complete", "contextId": context_id, "message_index": i}

def generate_safe_audio(
    text: str,
    voice_id: str,
    voice_settings: Dict[str, Any],
    context_id: str,
    job_start: float,
    message_index: Optional[int] = None
) -> Generator[Dict[str, Any], None, float]:
    """Generate audio with safe optimizations - FIXED VERSION"""
    
    # Select pipeline
    lang_code = voice_id[0] if voice_id and voice_id[0] in PIPELINES else 'a'
    pipeline = PIPELINES.get(lang_code, PIPELINES['a'])
    speed = voice_settings.get("speed", 1.0)
    
    chunk_count = 0
    first_chunk_time = None
    generation_start = time.perf_counter()
    audio_chunks = []
    
    # Monitor GPU before generation
    gpu_memory_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    
    try:
        # FIXED: Use voice_id string directly, not cached tensor
        # The pipeline expects a voice name string, not a tensor
        # The caching should happen inside the pipeline
        
        # Check if voice is already cached in the pipeline
        if voice_id not in pipeline.voices:
            # Load voice if not in pipeline cache
            print(f"Loading voice {voice_id} into pipeline cache...")
            voice_tensor = pipeline.load_voice(voice_id)
            # Pipeline already handles moving to device
        
        # Use autocast for performance while maintaining compatibility
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()) if torch.cuda.is_available() else torch.no_grad():
            # Generate with optimizations - use voice_id string, not tensor
            for result in pipeline(text, voice=voice_id, speed=speed):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - job_start
                    
                    yield {
                        "type": "first_chunk",
                        "contextId": context_id,
                        "latency_ms": int(first_chunk_time * 1000),
                        "gpu_memory_before": f"{gpu_memory_before:.2f}GB",
                        "safe_optimized": True
                    }
                
                if result.audio is not None:
                    chunk_count += 1
                    
                    # Efficient audio conversion
                    if hasattr(result.audio, 'detach'):
                        audio_np = result.audio.detach().cpu().numpy()
                    else:
                        audio_np = result.audio.numpy()
                    
                    audio_bytes = (audio_np * 32767).astype(np.int16).tobytes()
                    audio_chunks.append(audio_np)
                    
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
        
        # Final metrics
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / 24000.0
            generation_time = time.perf_counter() - generation_start
            
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            yield {"alignment": create_fast_alignment(text, audio_duration * 1000), "contextId": context_id}
            
            yield {
                "isFinal": True,
                "contextId": context_id,
                "metadata": {
                    "total_chunks": chunk_count,
                    "audio_duration_ms": int(audio_duration * 1000),
                    "generation_time_ms": int(generation_time * 1000),
                    "real_time_factor": generation_time / audio_duration if audio_duration > 0 else 0,
                    "gpu_used": torch.cuda.is_available(),
                    "gpu_memory_before": f"{gpu_memory_before:.2f}GB",
                    "gpu_memory_after": f"{gpu_memory_after:.2f}GB",
                    "safe_optimizations": "Voice Cache, Autocast, Fast Alignment",
                    "model_device": str(next(SHARED_MODEL.parameters()).device) if SHARED_MODEL else "unknown"
                }
            }
            
            return audio_duration
        else:
            # No audio generated - this is the problem!
            yield {
                "error": f"No audio generated for text: '{text}' with voice: '{voice_id}'",
                "contextId": context_id,
                "debug": {
                    "text_length": len(text),
                    "voice_id": voice_id,
                    "pipeline_lang": lang_code,
                    "generation_time_ms": int((time.perf_counter() - generation_start) * 1000)
                }
            }
            return 0.0
        
    except Exception as e:
        yield {
            "error": f"Audio generation failed: {str(e)}",
            "contextId": context_id,
            "debug": {
                "text": text,
                "voice_id": voice_id,
                "exception_type": type(e).__name__
            }
        }
        return 0.0
    finally:
        # Smart memory cleanup
        if torch.cuda.is_available() and len(audio_chunks) > 3:
            torch.cuda.empty_cache()

# Initialize with safe optimizations
safe_force_gpu_usage()

# Start RunPod with safe optimized handler
runpod.serverless.start({
    "handler": safe_optimized_handler,
    "return_aggregate_stream": True
})