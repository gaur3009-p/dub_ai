"""
API Models
Pydantic models for request/response validation
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: Dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.now)


class TranslationRequest(BaseModel):
    """Request for text translation"""
    text: str = Field(..., min_length=1, max_length=5000)
    source_lang: Optional[str] = Field(None, description="Source language code")
    target_lang: str = Field(..., description="Target language code")
    detect_emotion: bool = Field(default=True)
    preserve_emotion: bool = Field(default=True)
    
    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class TranslationResponse(BaseModel):
    """Response for text translation"""
    translated_text: str
    source_language: str
    target_language: str
    confidence: float = Field(ge=0.0, le=1.0)
    emotion: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class EnrollmentRequest(BaseModel):
    """Request for voice enrollment"""
    audio_data: List[float] = Field(..., description="Audio data as float array")
    sample_rate: int = Field(default=16000)
    user_name: Optional[str] = None
    
    @validator("audio_data")
    def validate_audio(cls, v):
        if not v or len(v) < 16000:  # Minimum 1 second
            raise ValueError("Audio must be at least 1 second long")
        if len(v) > 16000 * 60:  # Maximum 60 seconds
            raise ValueError("Audio must be less than 60 seconds long")
        return v


class EnrollmentResponse(BaseModel):
    """Response for voice enrollment"""
    user_id: str
    status: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


class WebSocketConfigMessage(BaseModel):
    """WebSocket configuration message"""
    type: str = "config"
    source_lang: str = Field(default="eng_Latn")
    target_lang: str = Field(default="hin_Deva")
    user_id: Optional[str] = None
    enable_emotion: bool = Field(default=True)
    enable_voice_cloning: bool = Field(default=True)


class WebSocketAudioMessage(BaseModel):
    """WebSocket audio message"""
    type: str = "audio"
    data: List[float]
    sample_rate: int = Field(default=16000)
    timestamp: Optional[datetime] = None


class WebSocketTranscriptMessage(BaseModel):
    """WebSocket transcript message"""
    type: str = "transcript"
    text: str
    language: str
    is_final: bool = False
    confidence: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class WebSocketTranslationMessage(BaseModel):
    """WebSocket translation message"""
    type: str = "translation"
    text: str
    source_language: str
    target_language: str
    emotion: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class WebSocketErrorMessage(BaseModel):
    """WebSocket error message"""
    type: str = "error"
    message: str
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    created_at: datetime
    connection_count: int
    languages: Dict[str, str]  # user_id -> language
    active: bool = True


class UserProfile(BaseModel):
    """User profile"""
    user_id: str
    user_name: Optional[str] = None
    preferred_language: str = Field(default="eng_Latn")
    voice_enrolled: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: Optional[datetime] = None


class LanguageInfo(BaseModel):
    """Language information"""
    code: str
    name: str
    native_name: str
    script: str
    supported: bool = True


class SupportedLanguagesResponse(BaseModel):
    """Response for supported languages"""
    languages: List[LanguageInfo]
    total_count: int


class StatsResponse(BaseModel):
    """System statistics response"""
    active_connections: int
    active_sessions: int
    total_translations: int
    uptime_seconds: float
    avg_translation_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AudioMetadata(BaseModel):
    """Audio metadata"""
    sample_rate: int
    channels: int = 1
    duration_seconds: float
    format: str = "float32"
