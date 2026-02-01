"""
DubYou Enterprise - Main Application
FastAPI-based multilingual real-time voice translation system
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
import structlog

from config.settings import get_settings
from services.asr.streaming_whisper import get_asr_service
from services.translation.nllb_translator import get_translation_service
from services.tts.multilingual_tts import get_tts_service
from api.websocket_handler import WebSocketManager
from api.models import (
    HealthResponse,
    TranslationRequest,
    TranslationResponse,
    EnrollmentRequest
)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
settings = get_settings()

# Metrics
translation_counter = Counter(
    'translations_total',
    'Total number of translations',
    ['source_lang', 'target_lang']
)
translation_duration = Histogram(
    'translation_duration_seconds',
    'Time spent on translation'
)
websocket_connections = Counter(
    'websocket_connections_total',
    'Total WebSocket connections',
    ['status']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting DubYou Enterprise")
    
    # Initialize services
    logger.info("Initializing AI services...")
    asr_service = get_asr_service()
    translation_service = get_translation_service()
    tts_service = get_tts_service()
    
    logger.info("All services initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down DubYou Enterprise")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Enterprise multilingual real-time voice translation system",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
ws_manager = WebSocketManager()


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        services={
            "asr": "operational",
            "translation": "operational",
            "tts": "operational"
        }
    )


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST
    )


# Translation API endpoint
@app.post("/api/v1/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    translation_service=Depends(get_translation_service)
):
    """
    Translate text between languages
    
    Args:
        request: Translation request with text and language codes
        
    Returns:
        Translation response with translated text and metadata
    """
    try:
        with translation_duration.time():
            result = await translation_service.translate(
                text=request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                detect_emotion=request.detect_emotion
            )
        
        # Update metrics
        translation_counter.labels(
            source_lang=result.source_language,
            target_lang=result.target_language
        ).inc()
        
        return TranslationResponse(
            translated_text=result.translated_text,
            source_language=result.source_language,
            target_language=result.target_language,
            confidence=result.confidence,
            emotion=result.emotion
        )
        
    except Exception as e:
        logger.error("Translation error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Voice enrollment endpoint
@app.post("/api/v1/enroll")
async def enroll_voice(
    request: EnrollmentRequest,
    tts_service=Depends(get_tts_service)
):
    """
    Enroll user voice for voice cloning
    
    Args:
        request: Enrollment request with audio data
        
    Returns:
        User ID and enrollment status
    """
    try:
        # Generate unique user ID
        user_id = str(uuid.uuid4())
        
        # Extract speaker embedding
        embedding = tts_service.tts_engine.extract_speaker_embedding(
            audio=request.audio_data,
            user_id=user_id
        )
        
        logger.info("Voice enrolled successfully", user_id=user_id)
        
        return {
            "user_id": user_id,
            "status": "enrolled",
            "message": "Voice profile created successfully"
        }
        
    except Exception as e:
        logger.error("Enrollment error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time translation
@app.websocket("/ws/translate/{session_id}")
async def websocket_translate(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for real-time bidirectional translation
    
    Supports:
    - Person A speaks Language X, hears Language Y
    - Person B speaks Language Y, hears Language X
    
    Protocol:
    1. Client connects with session_id and configuration
    2. Client sends audio chunks
    3. Server processes: ASR → Translation → TTS
    4. Server sends back translated audio
    """
    await websocket.accept()
    websocket_connections.labels(status="connected").inc()
    
    try:
        # Register connection
        connection_info = await ws_manager.register_connection(
            websocket, session_id
        )
        
        logger.info(
            "WebSocket connected",
            session_id=session_id,
            user_id=connection_info.get("user_id")
        )
        
        # Get services
        asr_service = get_asr_service()
        translation_service = get_translation_service()
        tts_service = get_tts_service()
        
        # Get user configuration
        config = connection_info.get("config", {})
        source_lang = config.get("source_lang", "eng_Latn")
        target_lang = config.get("target_lang", "hin_Deva")
        user_id = connection_info.get("user_id")
        
        # Main processing loop
        async for message in ws_manager.receive_messages(websocket):
            try:
                if message["type"] == "audio":
                    # Receive audio chunk
                    audio_data = message["data"]
                    
                    # Step 1: Speech-to-Text (ASR)
                    asr_result = await asr_service.transcribe_audio(
                        audio=audio_data,
                        language=source_lang
                    )
                    
                    if not asr_result.text:
                        continue
                    
                    # Send intermediate transcript
                    await websocket.send_json({
                        "type": "transcript",
                        "text": asr_result.text,
                        "language": source_lang,
                        "is_final": asr_result.is_final
                    })
                    
                    # Step 2: Translation
                    translation_result = await translation_service.translate(
                        text=asr_result.text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        detect_emotion=True
                    )
                    
                    # Send translation
                    await websocket.send_json({
                        "type": "translation",
                        "text": translation_result.translated_text,
                        "source_language": source_lang,
                        "target_language": target_lang,
                        "emotion": translation_result.emotion
                    })
                    
                    # Step 3: Text-to-Speech (TTS)
                    tts_audio, sample_rate = await tts_service.text_to_speech(
                        text=translation_result.translated_text,
                        language=target_lang,
                        user_id=user_id,
                        emotion=translation_result.emotion
                    )
                    
                    if len(tts_audio) > 0:
                        # Send audio back
                        await websocket.send_json({
                            "type": "audio",
                            "data": tts_audio.tolist(),
                            "sample_rate": sample_rate
                        })
                    
                    # Update metrics
                    translation_counter.labels(
                        source_lang=source_lang,
                        target_lang=target_lang
                    ).inc()
                
                elif message["type"] == "config":
                    # Update configuration
                    source_lang = message.get("source_lang", source_lang)
                    target_lang = message.get("target_lang", target_lang)
                    user_id = message.get("user_id", user_id)
                    
                    await websocket.send_json({
                        "type": "config_updated",
                        "source_lang": source_lang,
                        "target_lang": target_lang
                    })
                
            except Exception as e:
                logger.error(
                    "Error processing message",
                    session_id=session_id,
                    error=str(e)
                )
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", session_id=session_id)
        websocket_connections.labels(status="disconnected").inc()
    
    except Exception as e:
        logger.error(
            "WebSocket error",
            session_id=session_id,
            error=str(e)
        )
        websocket_connections.labels(status="error").inc()
    
    finally:
        # Unregister connection
        await ws_manager.unregister_connection(websocket, session_id)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "translate": "/api/v1/translate",
            "enroll": "/api/v1/enroll",
            "websocket": "/ws/translate/{session_id}",
            "metrics": "/metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        access_log=True,
        reload=settings.debug
    )
