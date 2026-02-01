"""
Enterprise Configuration Management
Centralized configuration with validation and environment-specific settings
"""

from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    mongodb_url: str = Field(default="mongodb://localhost:27017")
    mongodb_database: str = Field(default="dubyou_enterprise")
    redis_url: str = Field(default="redis://localhost:6379/0")
    postgres_url: Optional[str] = None
    
    model_config = SettingsConfigDict(env_prefix="DB_")


class AIModelConfig(BaseSettings):
    """AI/ML Model configuration"""
    
    # ASR (Speech-to-Text)
    whisper_model: str = Field(default="large-v3")
    whisper_device: str = Field(default="cuda")
    whisper_compute_type: str = Field(default="float16")
    
    # Translation (NLLB-200)
    nllb_model: str = Field(default="facebook/nllb-200-distilled-600M")
    translation_batch_size: int = Field(default=8)
    max_translation_length: int = Field(default=512)
    
    # TTS (MMS-TTS - Multilingual)
    tts_model: str = Field(default="facebook/mms-tts-eng")
    tts_vocoder: str = Field(default="facebook/mms-tts-eng")
    tts_sample_rate: int = Field(default=16000)
    
    # Voice Encoding
    voice_encoder_model: str = Field(default="speechbrain/spkrec-ecapa-voxceleb")
    speaker_embedding_dim: int = Field(default=192)
    
    # Emotion Detection
    emotion_model: str = Field(default="j-hartmann/emotion-english-distilroberta-base")
    
    model_config = SettingsConfigDict(env_prefix="AI_")


class AudioConfig(BaseSettings):
    """Audio processing configuration"""
    sample_rate: int = Field(default=16000)
    vad_threshold: float = Field(default=0.5)
    min_speech_duration: float = Field(default=0.3)
    max_speech_duration: float = Field(default=30.0)
    chunk_size: int = Field(default=1024)
    
    @validator("vad_threshold")
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")
        return v
    
    model_config = SettingsConfigDict(env_prefix="AUDIO_")


class SecurityConfig(BaseSettings):
    """Security configuration"""
    secret_key: str = Field(default="change-me-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if v == "change-me-in-production":
            raise ValueError("Secret key must be changed in production")
        return v
    
    model_config = SettingsConfigDict(env_prefix="SEC_")


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration"""
    per_minute: int = Field(default=60)
    per_hour: int = Field(default=1000)
    per_day: int = Field(default=10000)
    
    model_config = SettingsConfigDict(env_prefix="RATE_")


class Settings(BaseSettings):
    """Main application settings"""
    
    # Application
    app_name: str = Field(default="DubYou Enterprise")
    app_version: str = Field(default="2.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=4)
    
    # Nested configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ai_models: AIModelConfig = Field(default_factory=AIModelConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    
    # Session Management
    session_timeout: int = Field(default=3600)
    max_sessions_per_user: int = Field(default=3)
    
    # Storage
    audio_storage_path: str = Field(default="/data/audio")
    embeddings_storage_path: str = Field(default="/data/embeddings")
    max_upload_size: int = Field(default=10485760)  # 10MB
    
    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090)
    sentry_dsn: Optional[str] = None
    
    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_file: Optional[str] = None
    
    # Features
    enable_voice_cloning: bool = Field(default=True)
    enable_emotion_detection: bool = Field(default=True)
    enable_multilingual: bool = Field(default=True)
    supported_languages: List[str] = Field(
        default=[
            "eng_Latn",  # English
            "hin_Deva",  # Hindi
            "spa_Latn",  # Spanish
            "fra_Latn",  # French
            "deu_Latn",  # German
            "cmn_Hans",  # Chinese (Simplified)
            "ara_Arab",  # Arabic
            "jpn_Jpan",  # Japanese
            "kor_Hang",  # Korean
            "por_Latn",  # Portuguese
        ]
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = {"development", "staging", "production"}
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
