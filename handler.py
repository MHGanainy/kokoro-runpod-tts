#!/usr/bin/env python3
"""
GPU-Optimized Kokoro TTS RunPod Handler with Ultra-Low Latency
Maximizes GPU utilization for minimum time-to-first-byte
"""

import runpod
import base64
import time
import numpy as np
import json
from typing import Dict, Generator, Any, Optional, List, Tuple
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import PyTorch and check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import Kokoro
try:
    from kokoro import KPipeline
except ImportError:
    logger.error("Kokoro not installed. Please install with: pip install kokoro>=0.9.4")
    raise

# Global pipelines - loaded once at container start
PIPELINES = {}
LOAD_START_TIME = time.time()
DEVICE = None

def setup_gpu_optimization():
    """Configure optimal GPU settings"""
    global DEVICE
    
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:0")
        
        # GPU optimization settings
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Memory optimization
        torch.cuda.empty_cache()
        
        # Set optimal number of threads for GPU usage
        torch.set_num_threads(2)  # Reduced for GPU workloads
        
        logger.info(f"GPU optimization enabled: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        DEVICE = torch.device("cpu")
        torch.set_num_threads(4)  # More threads for CPU fallback
        logger.warning("GPU not available, falling back to CPU")
    
    return DEVICE

def initialize_pipelines():
    """GPU-optimized pipeline initialization"""
    global PIPELINES
    
    # Setup GPU first
    device = setup_gpu_optimization()
    
    languages = {
        'a': 'American English',
        'b': 'British English',
    }
    
    logger.info("Pre-loading Kokoro models with GPU acceleration...")
    
    for lang_code, lang_name in languages.items():
        try:
            start = time.time()
            
            # Initialize pipeline (Kokoro handles GPU internally)
            pipeline = KPipeline(lang_code=lang_code)
            
            # Aggressive warm-up for GPU optimization
            logger.info(f"GPU warm-up for {lang_name}...")
            
            # Multiple warm-up runs to optimize GPU kernels
            warm_texts = ["test", "hello world", "this is a longer test sentence"]
            warm_voices = ['af_bella', 'af_sarah', 'am_adam', 'am_michael']
            
            for warm_text in warm_texts:
                for voice in warm_voices[:2]:  # Limit to reduce startup time
                    try:
                        # Generate and discard to warm up GPU
                        list(pipeline(warm_text, voice=voice))
                    except Exception as e:
                        logger.warning(f"Warm-up warning for {voice}: {e}")
            
            # Clear GPU cache after warmup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations complete
            
            PIPELINES[lang_code] = pipeline
            load_time = time.time() - start
            logger.info(f"Loaded {lang_name} in {load_time:.2f}s")
            
            # GPU memory info after loading
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory used: {memory_used:.2f}GB, cached: {memory_cached:.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load {lang_name}: {e}")
    
    total_time = time.time() - LOAD_START_TIME
    logger.info(f"All models loaded in {total_time:.2f}s")
    
    # Final GPU optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set GPU to maximum performance mode
        try:
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        except:
            pass

# Initialize pipelines on import
initialize_pipelines()

def create_alignment_data(text: str, audio_duration_ms: float) -> dict:
    """Create ElevenLabs-compatible alignment data"""
    chars = list(text)
    if not chars:
        return {"chars": [], "charStartTimesMs": [], "charsDurationsMs": []}
    
    char_duration_ms = audio_duration_ms / len(chars) if chars else 0
    
    return {
        "chars": chars,
        "charStartTimesMs": [int(i * char_duration_ms) for i in range(len(chars))],
        "charsDurationsMs": [int(char_duration_ms) for _ in chars]
    }

def websocket_style_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """GPU-optimized WebSocket-style streaming handler"""
    job_start = time.perf_counter()  # High precision timing
    
    try:
        job_input = job["input"]
        
        # Determine request type with fast path for single requests
        if "text" in job_input and not any(k in job_input for k in ["messages", "websocket_message"]):
            # Direct single TTS - fastest GPU path
            yield from handle_single_tts(job_input, job_start)
        elif "websocket_message" in job_input:
            yield from handle_websocket_message(job_input, job_start)
        elif "messages" in job_input:
            yield from handle_websocket_conversation(job_input, job_start)
        else:
            yield {"error": "Invalid input format"}
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        yield {
            "type": "error",
            "error": str(e),
            "timestamp": time.perf_counter() - job_start
        }

def handle_single_tts(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """GPU-optimized single TTS request"""
    text = job_input.get("text", "")
    if not text:
        yield {"error": "No text provided"}
        return
    
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    context_id = f"single-{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    
    # Send minimal start event
    yield {
        "type": "generation_start",
        "contextId": context_id,
        "text": text,
        "character_count": len(text),
        "timestamp": time.perf_counter() - job_start
    }
    
    # GPU-accelerated generation
    yield from generate_gpu_optimized_audio(text, voice_id, voice_settings, context_id, job_start)
    
    # Send completion
    yield {
        "type": "generation_complete",
        "contextId": context_id,
        "timestamp": time.perf_counter() - job_start
    }

def handle_websocket_message(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """GPU-optimized WebSocket message handling"""
    ws_msg = job_input.get("websocket_message", {})
    text = ws_msg.get("text", "").strip()
    context_id = ws_msg.get("context_id") or ws_msg.get("contextId", f"ws-{int(time.perf_counter() * 1000000) & 0xFFFFFF}")
    voice_settings = ws_msg.get("voice_settings", {})
    voice_id = job_input.get("voice_id", "af_bella")
    
    # Handle control messages
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
    
    # Send start event
    yield {
        "type": "generation_start",
        "contextId": context_id,
        "text": text,
        "character_count": len(text),
        "timestamp": time.perf_counter() - job_start
    }
    
    # GPU generation
    yield from generate_gpu_optimized_audio(text, voice_id, voice_settings, context_id, job_start)
    
    # Send completion
    yield {
        "type": "generation_complete",
        "contextId": context_id,
        "timestamp": time.perf_counter() - job_start
    }

def handle_websocket_conversation(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """GPU-optimized conversation handling"""
    messages = job_input.get("messages", [])
    context_id = job_input.get("context_id", f"conv-{int(time.perf_counter() * 1000000) & 0xFFFFFF}")
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    
    yield {"type": "connection_established", "contextId": context_id, "message_count": len(messages)}
    
    for i, message in enumerate(messages):
        text = message if isinstance(message, str) else message.get("text", "").strip()
        if not text:
            continue
            
        yield {"type": "message_start", "contextId": context_id, "message_index": i, "text": text}
        
        # GPU generation for each message
        yield from generate_gpu_optimized_audio(text, voice_id, voice_settings, context_id, job_start, i)
        
        yield {"type": "message_complete", "contextId": context_id, "message_index": i}
    
    yield {"type": "conversation_complete", "contextId": context_id, "total_messages": len(messages)}

def generate_gpu_optimized_audio(
    text: str,
    voice_id: str,
    voice_settings: Dict[str, Any],
    context_id: str,
    job_start: float,
    message_index: Optional[int] = None
) -> Generator[Dict[str, Any], None, float]:
    """Ultra-fast GPU-optimized audio generation"""
    
    # Fast pipeline selection
    lang_code = voice_id[0] if voice_id and voice_id[0] in PIPELINES else 'a'
    pipeline = PIPELINES.get(lang_code, PIPELINES['a'])
    
    speed = voice_settings.get("speed", 1.0)
    
    audio_chunks = []
    chunk_count = 0
    first_chunk_time = None
    generation_start = time.perf_counter()
    
    try:
        # GPU-accelerated streaming generation
        for graphemes, phonemes, audio in pipeline(text, voice=voice_id, speed=speed):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - job_start
                
                # Send first chunk latency
                yield {
                    "type": "first_chunk",
                    "contextId": context_id,
                    "latency_ms": int(first_chunk_time * 1000),
                    "timestamp": first_chunk_time
                }
            
            # Optimized audio processing
            if hasattr(audio, 'cpu'):
                # Move from GPU to CPU efficiently
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio
            
            # Fast PCM conversion
            audio_pcm = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_pcm.tobytes()
            audio_chunks.append(audio_np)
            
            chunk_count += 1
            
            # Minimal chunk data for maximum speed
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
            audio_duration_ms = audio_duration * 1000
            generation_time = time.perf_counter() - generation_start
            
            # Fast alignment
            alignment = create_alignment_data(text, audio_duration_ms)
            yield {"alignment": alignment, "contextId": context_id}
            
            # Final metrics
            yield {
                "isFinal": True,
                "contextId": context_id,
                "metadata": {
                    "total_chunks": chunk_count,
                    "audio_duration_ms": int(audio_duration_ms),
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
        # Cleanup GPU memory after each generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def health_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-aware health check"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB",
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "gpu_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
        }
    else:
        gpu_info = {"gpu_available": False}
    
    return {
        "status": "healthy",
        "models_loaded": list(PIPELINES.keys()),
        "load_time": f"{time.time() - LOAD_START_TIME:.2f}s",
        "mode": "gpu_optimized",
        "device": str(DEVICE),
        **gpu_info
    }

def main_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Main GPU-optimized handler"""
    job_input = job.get("input", {})
    
    if job_input.get("health_check"):
        yield health_handler(job)
        return
    
    yield from websocket_style_handler(job)

# Start with GPU optimization
runpod.serverless.start({
    "handler": main_handler,
    "return_aggregate_stream": True
})