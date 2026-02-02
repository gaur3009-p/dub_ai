"""
Enterprise Configuration Management
Kaggle-safe centralized configuration with validation
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =========================
# Database Configuration
# =========================
class DatabaseConfig(BaseSettings):
    """Database configuration (external only on Kaggle)"""

    mongodb_url: str = Field(
        default="",
        description="MongoDB Atlas connection string"
    )
    mongodb_database: str = Field(default="dubyou_enterprise")

    redis_url: str = Field(
        default="",
        description="Redis Cloud / Upstash URL"
    )

    postgres_url: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix="DB_")

    @field_validator("mongodb_url", "redis_url")
    def validate_external_services(cls, v, info):
        if not v:
            raise ValueError(
                f"{info.field_name} must be set (Kaggle cannot run local services)"
            )
        return v


# =========================
# AI / ML Models
# =========================
class AIModelConfig(BaseSettings):
    """AI/ML Model configuration"""

    # ASR
    whisper_model: str = Field(default="large-v3")
    whisper_device: str = Field(default="cpu")  # Kaggle-safe default
    whisper_compute_type: str = Field(default="float16")

    # Translation
    nllb_model: str = Field(default="facebook/nllb-200-distilled-600M")
    translation_batch_size: int = Field(default=4)
    max_translation_length: int = Field(default=512)

    # TTS
    tts_model: str = Field(default="facebook/mms-tts-eng")
    tts_vocoder: str = Field(default="facebook/mms-tts-eng")
    tts_sample_rate: int = Field(default=16000)

    # Voice
    voice_encoder_model: str = Field(default="speechbrain/spkrec-ecapa-voxceleb")
    speaker_embedding_dim: int = Field(default=192)

    # Emotion
    emotion_model: str = Field(
        default="j-hartmann/emotion-english-distilroberta-base"
    )

    model_config = SettingsConfigDict(env_prefix="AI_")


# =========================
# Audio Processing
# =========================
class AudioConfig(BaseSettings):
    """Audio processing configuration"""

    sample_rate: int = Field(default=16000)
    vad_threshold: float = Field(default=0.5)
    min_speech_duration: float = Field(default=0.3)
    max_speech_duration: float = Field(default=30.0)
    chunk_size: int = Field(default=1024)

    @field_validator("vad_threshold")
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")
        return v

    model_config = SettingsConfigDict(env_prefix="AUDIO_")


# =========================
# Security
# =========================
class SecurityConfig(BaseSettings):
    """Security configuration"""

    secret_key: str = Field(default="")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

    @field_validator("secret_key")
    def validate_secret_key(cls, v):
        if not v:
            raise ValueError(
                "SEC_SECRET_KEY must be set via environment variable"
            )
        return v

    model_config = SettingsConfigDict(env_prefix="SEC_")


# =========================
# Rate Limiting
# =========================
class RateLimitConfig(BaseSettings):
    """Rate limiting configuration"""

    per_minute: int = Field(default=60)
    per_hour: int = Field(default=1000)
    per_day: int = Field(default=10000)

    model_config = SettingsConfigDict(env_prefix="RATE_")


# =========================
# Main Settings
# =========================
class Settings(BaseSettings):
    """Main application settings"""

    # App
    app_name: str = Field(default="DubYou Enterprise")
    app_version: str = Field(default="2.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)

    # Server (mostly unused on Kaggle)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)

    # Nested configs
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ai_models: AIModelConfig = Field(default_factory=AIModelConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    # Sessions
    session_timeout: int = Field(default=3600)
    max_sessions_per_user: int = Field(default=3)

    # Storage (Kaggle-safe)
    base_storage_path: str = Field(default="/kaggle/working")
    audio_storage_path: str = Field(default="/kaggle/working/audio")
    embeddings_storage_path: str = Field(default="/kaggle/working/embeddings")
    max_upload_size: int = Field(default=10 * 1024 * 1024)

    # Monitoring
    enable_metrics: bool = Field(default=False)
    sentry_dsn: Optional[str] = None

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Features
    enable_voice_cloning: bool = Field(default=True)
    enable_emotion_detection: bool = Field(default=True)
    enable_multilingual: bool = Field(default=True)

    supported_languages: List[str] = Field(
        default=[
            "eng_Latn",
            "hin_Deva",
            "spa_Latn",
            "fra_Latn",
            "deu_Latn",
            "cmn_Hans",
            "ara_Arab",
            "jpn_Jpan",
            "kor_Hang",
            "por_Latn",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("environment")
    def validate_environment(cls, v):
        if v not in {"development", "staging", "production"}:
            raise ValueError("Invalid environment")
        return v


# =========================
# Singleton
# =========================
settings = Settings()


def get_settings() -> Settings:
    return settings
