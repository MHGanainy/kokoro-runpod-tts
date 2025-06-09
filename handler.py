#!/usr/bin/env python3
"""
Kokoro TTS RunPod Serverless Handler with WebSocket-Style Streaming
Provides ElevenLabs-compatible streaming interface using RunPod's streaming API
"""

import runpod
import base64
import time
import numpy as np
import json
import uuid
from typing import Dict, Generator, Any, Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Kokoro
try:
    from kokoro import KPipeline
except ImportError:
    logger.error("Kokoro not installed. Please install with: pip install kokoro>=0.9.4")
    raise

# Global pipelines - loaded once at container start
PIPELINES = {}
LOAD_START_TIME = time.time()

def initialize_pipelines():
    """Pre-load all language models for instant access"""
    global PIPELINES
    
    languages = {
        'a': 'American English',
        'b': 'British English',
    }
    
    logger.info("Pre-loading Kokoro models...")
    
    for lang_code, lang_name in languages.items():
        try:
            start = time.time()
            pipeline = KPipeline(lang_code=lang_code)
            
            # Warm up with dummy inference
            logger.info(f"Warming up {lang_name} model...")
            try:
                list(pipeline("test", voice='af_bella'))
            except Exception as e:
                logger.warning(f"Warm-up warning for {lang_name}: {e}")
            
            PIPELINES[lang_code] = pipeline
            logger.info(f"Loaded {lang_name} in {time.time() - start:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load {lang_name}: {e}")
    
    total_time = time.time() - LOAD_START_TIME
    logger.info(f"All models loaded in {total_time:.2f}s")

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

def calculate_word_times(text: str, audio_duration: float, cumulative_time: float = 0) -> List[Tuple[str, float]]:
    """Calculate word timestamps based on text and audio duration"""
    words = text.split()
    if not words:
        return []
    
    # Simple linear distribution of words across audio duration
    word_duration = audio_duration / len(words)
    word_times = []
    
    for i, word in enumerate(words):
        timestamp = cumulative_time + (i * word_duration)
        word_times.append((word, timestamp))
    
    return word_times

def websocket_style_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    WebSocket-style streaming handler that mimics ElevenLabs WebSocket behavior
    Supports both single requests and multi-message conversations
    """
    job_start = time.time()
    
    try:
        job_input = job["input"]
        
        # Determine request type
        if "messages" in job_input:
            # Multi-message conversation (WebSocket-style)
            yield from handle_websocket_conversation(job_input, job_start)
        elif "websocket_message" in job_input:
            # Single WebSocket-style message
            yield from handle_websocket_message(job_input, job_start)
        else:
            # Traditional single TTS request
            yield from handle_single_tts(job_input, job_start)
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        yield {
            "type": "error",
            "error": str(e),
            "timestamp": time.time() - job_start
        }

def handle_websocket_conversation(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Handle conversation with multiple messages (like WebSocket session)"""
    messages = job_input.get("messages", [])
    context_id = job_input.get("context_id", f"conv-{int(time.time())}")
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    
    # Send connection established event
    yield {
        "type": "connection_established",
        "contextId": context_id,
        "message_count": len(messages),
        "timestamp": time.time() - job_start
    }
    
    cumulative_time = 0.0
    
    # Process each message
    for i, message in enumerate(messages):
        if isinstance(message, str):
            text = message
        else:
            text = message.get("text", "").strip()
            
        if not text:
            continue
            
        # Send message start event
        yield {
            "type": "message_start",
            "contextId": context_id,
            "message_index": i,
            "text": text,
            "timestamp": time.time() - job_start
        }
        
        # Generate audio for this message
        audio_duration = yield from generate_streaming_audio(
            text, voice_id, voice_settings, context_id, job_start, 
            message_index=i, cumulative_time=cumulative_time
        )
        
        if audio_duration > 0:
            cumulative_time += audio_duration
        
        # Send message complete event
        yield {
            "type": "message_complete",
            "contextId": context_id,
            "message_index": i,
            "timestamp": time.time() - job_start
        }
    
    # Send conversation complete event
    yield {
        "type": "conversation_complete",
        "contextId": context_id,
        "total_messages": len(messages),
        "total_time": time.time() - job_start,
        "timestamp": time.time() - job_start
    }

def handle_websocket_message(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Handle single WebSocket-style message"""
    ws_msg = job_input.get("websocket_message", {})
    text = ws_msg.get("text", "").strip()
    context_id = ws_msg.get("context_id") or ws_msg.get("contextId", f"ws-{int(time.time())}")
    voice_settings = ws_msg.get("voice_settings", {})
    
    # Extract voice_id from the job input or use default
    voice_id = job_input.get("voice_id", "af_bella")
    
    # Handle special WebSocket messages
    if ws_msg.get("close_socket"):
        yield {
            "type": "socket_closed",
            "contextId": context_id,
            "timestamp": time.time() - job_start
        }
        return
        
    if ws_msg.get("close_context"):
        yield {
            "type": "context_closed",
            "contextId": context_id,
            "timestamp": time.time() - job_start
        }
        return
    
    # Handle initial space (context initialization)
    if text == " ":
        yield {
            "type": "context_initialized",
            "contextId": context_id,
            "timestamp": time.time() - job_start
        }
        return
    
    if not text:
        yield {
            "type": "empty_message",
            "contextId": context_id,
            "timestamp": time.time() - job_start
        }
        return
    
    # Send generation start event
    yield {
        "type": "generation_start",
        "contextId": context_id,
        "text": text,
        "character_count": len(text),
        "timestamp": time.time() - job_start
    }
    
    # Generate audio
    yield from generate_streaming_audio(
        text, voice_id, voice_settings, context_id, job_start
    )
    
    # Send generation complete event
    yield {
        "type": "generation_complete",
        "contextId": context_id,
        "total_time": time.time() - job_start,
        "timestamp": time.time() - job_start
    }

def handle_single_tts(job_input: Dict[str, Any], job_start: float) -> Generator[Dict[str, Any], None, None]:
    """Handle traditional single TTS request"""
    text = job_input.get("text", "")
    voice_id = job_input.get("voice_id", "af_bella")
    voice_settings = job_input.get("voice_settings", {})
    context_id = job_input.get("context_id", f"single-{int(time.time())}")
    
    if not text:
        yield {"error": "No text provided"}
        return
    
    # Send start event
    yield {
        "type": "generation_start",
        "contextId": context_id,
        "text": text,
        "voice_id": voice_id,
        "character_count": len(text),
        "timestamp": time.time() - job_start
    }
    
    # Generate audio
    yield from generate_streaming_audio(
        text, voice_id, voice_settings, context_id, job_start
    )
    
    # Send completion event
    yield {
        "type": "generation_complete",
        "contextId": context_id,
        "total_time": time.time() - job_start,
        "timestamp": time.time() - job_start
    }

def generate_streaming_audio(
    text: str,
    voice_id: str,
    voice_settings: Dict[str, Any],
    context_id: str,
    job_start: float,
    message_index: Optional[int] = None,
    cumulative_time: float = 0.0
) -> Generator[Dict[str, Any], None, float]:
    """Generate streaming audio with real-time chunk delivery"""
    
    # Get pipeline
    lang_code = voice_id[0] if voice_id and voice_id[0] in PIPELINES else 'a'
    pipeline = PIPELINES.get(lang_code, PIPELINES['a'])
    
    # Extract voice settings
    speed = voice_settings.get("speed", 1.0)
    
    audio_chunks = []
    chunk_count = 0
    total_samples = 0
    first_chunk_time = None
    generation_start = time.time()
    
    try:
        # Stream each audio chunk as it's generated
        for graphemes, phonemes, audio in pipeline(text, voice=voice_id, speed=speed):
            if first_chunk_time is None:
                first_chunk_time = time.time() - job_start
                
                # Send first chunk event
                yield {
                    "type": "first_chunk",
                    "contextId": context_id,
                    "latency_ms": int(first_chunk_time * 1000),
                    "message_index": message_index,
                    "timestamp": time.time() - job_start
                }
            
            # Convert audio to proper format
            if hasattr(audio, 'cpu'):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio
            
            # Convert to PCM16
            audio_pcm = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_pcm.tobytes()
            audio_chunks.append(audio_np)
            
            chunk_count += 1
            total_samples += len(audio_np)
            
            # Send audio chunk (ElevenLabs-compatible format)
            chunk_data = {
                "audio": base64.b64encode(audio_bytes).decode('utf-8'),
                "contextId": context_id,
                "isFinal": False,
                "chunk_number": chunk_count,
                "chunk_samples": len(audio_np),
                "total_samples": total_samples,
                "sample_rate": 24000,
                "format": "pcm_16000",
                "timestamp": time.time() - job_start
            }
            
            if message_index is not None:
                chunk_data["message_index"] = message_index
                
            # This yield sends the chunk immediately!
            yield chunk_data
        
        # Calculate final metrics
        audio_duration = 0
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / 24000.0  # 24kHz sample rate
            audio_duration_ms = audio_duration * 1000
            generation_time = time.time() - generation_start
            
            # Send alignment data (ElevenLabs-compatible)
            alignment = create_alignment_data(text, audio_duration_ms)
            alignment_data = {
                "alignment": alignment,
                "contextId": context_id,
                "message_index": message_index,
                "timestamp": time.time() - job_start
            }
            yield alignment_data
            
            # Send final completion with audio metrics
            completion_data = {
                "isFinal": True,
                "contextId": context_id,
                "metadata": {
                    "total_chunks": chunk_count,
                    "audio_duration_ms": int(audio_duration_ms),
                    "generation_time_ms": int(generation_time * 1000),
                    "real_time_factor": generation_time / audio_duration if audio_duration > 0 else 0,
                    "character_count": len(text),
                    "cumulative_time": cumulative_time
                },
                "message_index": message_index,
                "timestamp": time.time() - job_start
            }
            yield completion_data
        
        return audio_duration
            
    except Exception as e:
        error_data = {
            "error": str(e),
            "contextId": context_id,
            "message_index": message_index,
            "timestamp": time.time() - job_start
        }
        yield error_data
        return 0.0

# Health check handler for monitoring
def health_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": list(PIPELINES.keys()),
        "load_time": f"{time.time() - LOAD_START_TIME:.2f}s",
        "timestamp": time.time(),
        "capabilities": [
            "websocket_style_streaming",
            "elevenlabs_compatible",
            "multi_message_conversations",
            "word_timestamps",
            "real_time_audio_chunks"
        ]
    }

# Route handler based on input
def main_handler(job: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """Main router handler"""
    job_input = job.get("input", {})
    
    # Route to health check
    if job_input.get("health_check"):
        yield health_handler(job)
        return
    
    # Route to WebSocket-style handler
    yield from websocket_style_handler(job)

# Start the serverless function with streaming enabled
runpod.serverless.start({
    "handler": main_handler,
    "return_aggregate_stream": True  # Critical for streaming!
})