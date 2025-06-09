#!/usr/bin/env python3
"""
GPU Diagnostic Handler - Ensures actual GPU usage for inference
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

# Force GPU usage
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Aggressive GPU settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(1)
    
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

# Global state
SHARED_MODEL = None
PIPELINES = {}
LOAD_START_TIME = time.perf_counter()

def force_gpu_usage():
    """Force all Kokoro operations to use GPU"""
    global SHARED_MODEL, PIPELINES
    
    try:
        logger.warning("Initializing Kokoro with forced GPU usage...")
        start = time.perf_counter()
        
        # Create model and immediately move to GPU
        SHARED_MODEL = KModel().eval()
        
        if torch.cuda.is_available():
            print(f"Moving model to GPU: {DEVICE}")
            SHARED_MODEL = SHARED_MODEL.to(DEVICE)
            
            # Verify model is on GPU
            for name, param in SHARED_MODEL.named_parameters():
                if not param.is_cuda:
                    print(f"⚠️ Parameter {name} not on GPU!")
                    param.data = param.data.to(DEVICE)
            
            # Force GPU memory allocation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Test inference to ensure GPU usage
            print("Testing GPU inference...")
            dummy_input = torch.LongTensor([[0, 1, 2, 0]]).to(DEVICE)
            dummy_ref = torch.randn(1, 256).to(DEVICE)
            
            with torch.no_grad():
                _ = SHARED_MODEL.forward_with_tokens(dummy_input, dummy_ref, 1.0)
            
            torch.cuda.synchronize()
            print(f"✅ GPU inference test completed")
        
        model_time = time.perf_counter() - start
        logger.warning(f"Model loaded and GPU-optimized in {model_time:.2f}s")
        
        # Initialize pipelines with explicit GPU device
        languages = {'a': 'American English', 'b': 'British English'}
        
        for lang_code, lang_name in languages.items():
            try:
                pipeline_start = time.perf_counter()
                
                # Create pipeline with explicit device
                pipeline = KPipeline(
                    lang_code=lang_code,
                    model=SHARED_MODEL,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                
                # Pre-load and move voices to GPU
                common_voices = get_common_voices(lang_code)
                for voice in common_voices:
                    try:
                        voice_tensor = pipeline.load_voice(voice)
                        if torch.cuda.is_available():
                            # Ensure voice tensor is on GPU
                            voice_tensor = voice_tensor.to(DEVICE)
                            pipeline.voices[voice] = voice_tensor
                    except Exception as e:
                        print(f"⚠️ Voice {voice} loading failed: {e}")
                
                PIPELINES[lang_code] = pipeline
                pipeline_time = time.perf_counter() - pipeline_start
                logger.warning(f"Pipeline {lang_name} ready in {pipeline_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to initialize {lang_name}: {e}")
        
        # Final GPU memory check
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"📊 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        total_time = time.perf_counter() - LOAD_START_TIME
        logger.warning(f"Complete GPU initialization in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"GPU initialization failed: {e}")
        raise

def get_common_voices(lang_code: str) -> list:
    """Get common voices for pre-loading"""
    return {
        'a': ['af_bella', 'af_sarah', 'am_adam'],
        'b': ['bf_emma', 'bm_george']
    }.get(lang_code, ['af_bella'])

def create_minimal_alignment(text: str, audio_duration_ms: float) -> dict:
    """Fast alignment calculation"""
    if not text:
        return {"chars": [], "charStartTimesMs": [], "charsDurationsMs": []}
    
    char_count = len(text)
    char_duration = audio_duration_ms / char_count
    
    return {
        "chars": list(text),
        "charStartTimesMs": [int(i * char_duration) for i in range(char_count)],
        "charsDurationsMs": [int(char_duration)] * char_count
    }

def gpu_diagnostic_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Handler with GPU diagnostics"""
    job_start = time.perf_counter()
    
    try:
        job_input = job["input"]
        
        # Enhanced health check with GPU info
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
                    "model_on_gpu": next(SHARED_MODEL.parameters()).is_cuda if SHARED_MODEL else False
                }
            
            yield {
                "status": "healthy",
                "models_loaded": list(PIPELINES.keys()),
                "mode": "gpu_diagnostic",
                "shared_model": SHARED_MODEL is not None,
                **gpu_info
            }
            return
        
        # Route requests
        if "text" in job_input:
            yield from handle_gpu_tts(job_input, job_start)
        elif "websocket_message" in job_input:
            yield from handle_gpu_websocket(job_input, job_start)
        elif "messages" in job_input:
            yield from handle_gpu_conversation(job_input, job_start)
        else:
            yield {"error": "Invalid input"}
            
    except Exception as e:
        yield {"error": str(e)}

def handle_gpu_tts(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """TTS with GPU diagnostics"""
    text = job_input.get("text", "")
    if not text:
        yield {"error": "No text"}
        return
    
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    context_id = f"gpu{int(time.perf_counter() * 1000000) & 0xFFFFFF}"
    
    yield from generate_with_gpu_monitoring(text, voice_id, voice_settings, context_id, job_start)

def handle_gpu_websocket(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """WebSocket with GPU monitoring"""
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
    
    yield from generate_with_gpu_monitoring(text, voice_id, voice_settings, context_id, job_start)

def handle_gpu_conversation(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Conversation with GPU monitoring"""
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
        yield from generate_with_gpu_monitoring(text, voice_id, voice_settings, context_id, job_start, i)
        yield {"type": "message_complete", "contextId": context_id, "message_index": i}

def generate_with_gpu_monitoring(
    text: str,
    voice_id: str,
    voice_settings: Dict[str, Any],
    context_id: str,
    job_start: float,
    message_index: Optional[int] = None
) -> Generator[Dict[str, Any], None, float]:
    """Generate audio with GPU utilization monitoring"""
    
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
        # Force GPU computation
        if torch.cuda.is_available():
            # Ensure voice is on GPU
            voice_tensor = pipeline.load_voice(voice_id)
            if not voice_tensor.is_cuda:
                voice_tensor = voice_tensor.to(DEVICE)
                pipeline.voices[voice_id] = voice_tensor
        
        # Generate with forced GPU usage
        for result in pipeline(text, voice=voice_id, speed=speed):
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - job_start
                
                yield {
                    "type": "first_chunk",
                    "contextId": context_id,
                    "latency_ms": int(first_chunk_time * 1000),
                    "gpu_memory_before": f"{gpu_memory_before:.2f}GB"
                }
            
            if result.audio is not None:
                chunk_count += 1
                
                # Force GPU synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Convert audio
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
        
        # Final metrics with GPU info
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
                    "model_device": str(next(SHARED_MODEL.parameters()).device) if SHARED_MODEL else "unknown"
                }
            }
            
            return audio_duration
        
        return 0.0
            
    except Exception as e:
        yield {"error": str(e), "contextId": context_id}
        return 0.0
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Initialize with forced GPU usage
force_gpu_usage()

# Start RunPod
runpod.serverless.start({
    "handler": gpu_diagnostic_handler,
    "return_aggregate_stream": True
})