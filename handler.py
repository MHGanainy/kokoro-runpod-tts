#!/usr/bin/env python3
"""
Enhanced Optimized Handler - Audio generation + Performance optimizations
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

# Minimal logging - Minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Enhanced GPU optimizations
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Aggressive GPU optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)  # Reduced for better GPU focus
    
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

# Global state with enhanced caching
SHARED_MODEL = None
PIPELINES = {}
LOAD_START_TIME = time.perf_counter()

# Enhanced voice pre-loading strategy
PRELOADED_VOICES = set()
VOICE_PRELOAD_LOCK = threading.Lock()

def enhanced_force_gpu_usage():
    """Enhanced GPU optimization with proper voice preloading"""
    global SHARED_MODEL, PIPELINES
    
    try:
        logger.warning("Initializing enhanced optimized Kokoro...")
        start = time.perf_counter()
        
        # Create model with enhanced optimizations
        SHARED_MODEL = KModel().eval()
        
        if torch.cuda.is_available():
            print(f"Moving model to GPU: {DEVICE}")
            SHARED_MODEL = SHARED_MODEL.to(DEVICE)
            
            # Verify model is on GPU
            for name, param in SHARED_MODEL.named_parameters():
                if not param.is_cuda:
                    param.data = param.data.to(DEVICE)
            
            # Enhanced compilation
            try:
                SHARED_MODEL = torch.compile(SHARED_MODEL, mode="max-autotune")
                print("✅ Model compiled with max-autotune")
            except Exception as e:
                try:
                    SHARED_MODEL = torch.compile(SHARED_MODEL, mode="reduce-overhead")
                    print("✅ Model compiled with reduce-overhead")
                except Exception as e2:
                    print(f"⚠️ Model compilation failed: {e2}, using standard model")
            
            # Enhanced GPU memory management
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Test inference
            print("Testing enhanced GPU inference...")
            dummy_input = torch.LongTensor([[0, 1, 2, 0]]).to(DEVICE)
            dummy_ref = torch.randn(1, 256).to(DEVICE)
            
            with torch.no_grad():
                _ = SHARED_MODEL.forward_with_tokens(dummy_input, dummy_ref, 1.0)
            
            torch.cuda.synchronize()
            print(f"✅ Enhanced GPU inference test completed")
        
        model_time = time.perf_counter() - start
        logger.warning(f"Enhanced model loaded in {model_time:.2f}s")
        
        # Initialize pipelines with enhanced voice preloading
        languages = {'a': 'American English', 'b': 'British English'}
        
        for lang_code, lang_name in languages.items():
            try:
                pipeline_start = time.perf_counter()
                
                pipeline = KPipeline(
                    lang_code=lang_code,
                    model=SHARED_MODEL,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Aggressively pre-load ALL common voices
                all_voices = get_all_voices(lang_code)
                for voice in all_voices:
                    try:
                        print(f"Pre-loading voice: {voice}")
                        voice_tensor = pipeline.load_voice(voice)
                        # Voice is automatically cached in pipeline.voices
                        with VOICE_PRELOAD_LOCK:
                            PRELOADED_VOICES.add(voice)
                        print(f"✅ Pre-loaded {voice}")
                                
                    except Exception as e:
                        print(f"⚠️ Voice {voice} pre-loading failed: {e}")
                
                PIPELINES[lang_code] = pipeline
                pipeline_time = time.perf_counter() - pipeline_start
                logger.warning(f"Enhanced pipeline {lang_name} ready in {pipeline_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to initialize {lang_name}: {e}")
        
        # Memory status
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 Enhanced GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        total_time = time.perf_counter() - LOAD_START_TIME
        logger.warning(f"Complete enhanced initialization in {total_time:.2f}s")
        logger.warning(f"Pre-loaded voices: {len(PRELOADED_VOICES)}")
        
    except Exception as e:
        logger.error(f"Enhanced GPU initialization failed: {e}")
        raise

def get_all_voices(lang_code: str) -> list:
    """Get ALL voices for aggressive pre-loading"""
    return {
        'a': ['af_bella', 'af_sarah', 'am_adam', 'af_jessica', 'af_heart', 'am_michael', 'am_cooper', 'am_jackson'],
        'b': ['bf_emma', 'bm_george', 'bf_isabella', 'bm_william']
    }.get(lang_code, ['af_bella'])

def create_minimal_alignment(text: str, audio_duration_ms: float) -> dict:
    """Ultra-fast alignment calculation"""
    if not text:
        return {"chars": [], "charStartTimesMs": [], "charsDurationsMs": []}
    
    char_count = len(text)
    char_duration = audio_duration_ms / char_count if char_count > 0 else 0
    
    return {
        "chars": list(text),
        "charStartTimesMs": [int(i * char_duration) for i in range(char_count)],
        "charsDurationsMs": [int(char_duration)] * char_count
    }

def enhanced_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Enhanced optimized handler"""
    job_start = time.perf_counter()
    
    try:
        job_input = job["input"]
        
        # Enhanced health check
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
                    "optimizations": "Enhanced Voice Preloading, Max-Autotune Compilation, Fast Alignment"
                }
            
            yield {
                "status": "healthy",
                "models_loaded": list(PIPELINES.keys()),
                "mode": "enhanced_optimized",
                "shared_model": SHARED_MODEL is not None,
                "preloaded_voices": len(PRELOADED_VOICES),
                "preloaded_voice_list": list(PRELOADED_VOICES),
                **gpu_info
            }
            return
        
        # Route requests to enhanced handlers
        if "text" in job_input:
            yield from handle_enhanced_tts(job_input, job_start)
        elif "websocket_message" in job_input:
            yield from handle_enhanced_websocket(job_input, job_start)
        elif "messages" in job_input:
            yield from handle_enhanced_conversation(job_input, job_start)
        else:
            yield {"error": "Invalid input"}
            
    except Exception as e:
        yield {"error": str(e)}

def handle_enhanced_tts(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Enhanced TTS handling"""
    text = job_input.get("text", "")
    if not text:
        yield {"error": "No text"}
        return
    
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    context_id = f"enh{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    
    yield from generate_enhanced_audio(text, voice_id, voice_settings, context_id, job_start)

def handle_enhanced_websocket(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Enhanced WebSocket handling"""
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
    
    yield from generate_enhanced_audio(text, voice_id, voice_settings, context_id, job_start)

def handle_enhanced_conversation(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Enhanced conversation handling"""
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
        yield from generate_enhanced_audio(text, voice_id, voice_settings, context_id, job_start, i)
        yield {"type": "message_complete", "contextId": context_id, "message_index": i}

def generate_enhanced_audio(
    text: str,
    voice_id: str,
    voice_settings: Dict[str, Any],
    context_id: str,
    job_start: float,
    message_index: Optional[int] = None
) -> Generator[Dict[str, Any], None, float]:
    """Generate audio with enhanced optimizations"""
    
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
        # Check if voice is pre-loaded (should be for common voices)
        voice_preloaded = voice_id in PRELOADED_VOICES
        
        # Enhanced inference with autocast and torch.no_grad
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()), torch.no_grad():
            # Generate with enhanced optimizations
            for result in pipeline(text, voice=voice_id, speed=speed):
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter() - job_start
                    
                    yield {
                        "type": "first_chunk",
                        "contextId": context_id,
                        "latency_ms": int(first_chunk_time * 1000),
                        "gpu_memory_before": f"{gpu_memory_before:.2f}GB",
                        "voice_preloaded": voice_preloaded,
                        "enhanced_optimized": True
                    }
                
                if result.audio is not None:
                    chunk_count += 1
                    
                    # Optimized audio conversion
                    audio_np = result.audio.detach().cpu().numpy() if hasattr(result.audio, 'detach') else result.audio.numpy()
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
        
        # Enhanced final metrics
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / 24000.0
            generation_time = time.perf_counter() - generation_start
            
            gpu_memory_after = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            
            yield {"alignment": create_minimal_alignment(text, audio_duration * 1000), "contextId": context_id}
            
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
                    "voice_preloaded": voice_preloaded,
                    "enhanced_optimizations": "Voice Preloading, Max-Autotune, Autocast, Fast Alignment",
                    "model_device": str(next(SHARED_MODEL.parameters()).device) if SHARED_MODEL else "unknown"
                }
            }
            
            return audio_duration
        else:
            yield {
                "error": f"No audio generated",
                "contextId": context_id,
                "debug": {
                    "text_length": len(text),
                    "voice_id": voice_id,
                    "voice_preloaded": voice_preloaded
                }
            }
            return 0.0
        
    except Exception as e:
        yield {
            "error": f"Enhanced audio generation failed: {str(e)}",
            "contextId": context_id
        }
        return 0.0
    finally:
        # Efficient memory cleanup
        if torch.cuda.is_available() and chunk_count > 0:
            torch.cuda.empty_cache()

# Initialize with enhanced optimizations
enhanced_force_gpu_usage()

# Start RunPod with enhanced handler
runpod.serverless.start({
    "handler": enhanced_handler,
    "return_aggregate_stream": True
})