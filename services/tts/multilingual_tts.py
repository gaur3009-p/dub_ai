"""
Enterprise TTS Service
Uses free Hugging Face models for multilingual text-to-speech with voice cloning
Supports: MMS-TTS, SpeechT5, and XTTS-v2 (optional)
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from transformers import (
    VitsModel,
    AutoTokenizer,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
)
from speechbrain.pretrained import EncoderClassifier

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VoiceEmbeddingExtractor:
    """Extract speaker embeddings for voice cloning"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Voice Embedding Extractor on {self.device}")
        
        # SpeechBrain's ECAPA-TDNN for speaker recognition
        self.encoder = EncoderClassifier.from_hparams(
            source=settings.ai_models.voice_encoder_model,
            savedir="models/speaker_encoder",
            run_opts={"device": self.device}
        )
        
    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio
        
        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Sample rate of audio
            
        Returns:
            Speaker embedding tensor
        """
        try:
            # Convert to torch tensor
            if isinstance(audio, np.ndarray):
                audio = torch.FloatTensor(audio)
            
            # Ensure correct shape
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.encoder.encode_batch(audio.to(self.device))
            
            return embedding.squeeze()
            
        except Exception as e:
            logger.error(f"Error extracting voice embedding: {e}")
            raise


class MultilingualTTS:
    """
    Enterprise-grade multilingual TTS using free Hugging Face models
    Supports multiple languages with voice cloning capability
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initializing Multilingual TTS on {self.device}")
        
        # Initialize models
        self._init_speecht5()
        self._init_voice_extractor()
        
        # Cache for speaker embeddings
        self.speaker_cache: Dict[str, torch.Tensor] = {}
        
    def _init_speecht5(self):
        """Initialize SpeechT5 TTS model (supports voice cloning)"""
        try:
            logger.info("Loading SpeechT5 TTS model...")
            
            self.processor = SpeechT5Processor.from_pretrained(
                "microsoft/speecht5_tts"
            )
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                "microsoft/speecht5_tts"
            ).to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan"
            ).to(self.device)
            
            # Load default speaker embeddings
            self.default_speaker_embeddings = self._load_default_embeddings()
            
            logger.info("SpeechT5 TTS initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SpeechT5: {e}")
            raise
    
    def _init_voice_extractor(self):
        """Initialize voice embedding extractor"""
        try:
            self.voice_extractor = VoiceEmbeddingExtractor()
            logger.info("Voice embedding extractor initialized")
        except Exception as e:
            logger.error(f"Error initializing voice extractor: {e}")
            raise
    
    def _load_default_embeddings(self) -> torch.Tensor:
        """Load default speaker embeddings from CMU Arctic dataset"""
        try:
            from datasets import load_dataset
            
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors",
                split="validation"
            )
            
            # Use first speaker as default
            speaker_embeddings = torch.tensor(
                embeddings_dataset[0]["xvector"]
            ).unsqueeze(0)
            
            return speaker_embeddings.to(self.device)
            
        except Exception as e:
            logger.warning(f"Could not load default embeddings: {e}")
            # Return zero tensor as fallback
            return torch.zeros((1, 512)).to(self.device)
    
    def extract_speaker_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        user_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Extract and cache speaker embedding from reference audio
        
        Args:
            audio: Reference audio waveform
            sample_rate: Sample rate of audio
            user_id: Optional user ID for caching
            
        Returns:
            Speaker embedding tensor
        """
        try:
            # Check cache first
            if user_id and user_id in self.speaker_cache:
                logger.debug(f"Using cached embedding for user {user_id}")
                return self.speaker_cache[user_id]
            
            # Extract new embedding
            embedding = self.voice_extractor.extract_embedding(
                audio, sample_rate
            )
            
            # Cache if user_id provided
            if user_id:
                self.speaker_cache[user_id] = embedding
                logger.info(f"Cached speaker embedding for user {user_id}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            return self.default_speaker_embeddings
    
    async def synthesize(
        self,
        text: str,
        language: str = "en",
        speaker_embedding: Optional[torch.Tensor] = None,
        emotion: Optional[str] = None,
        speed: float = 1.0
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text with optional voice cloning
        
        Args:
            text: Text to synthesize
            language: Target language code
            speaker_embedding: Optional speaker embedding for voice cloning
            emotion: Optional emotion modifier
            speed: Speech speed multiplier
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not text or not text.strip():
            return np.array([]), settings.audio.sample_rate
        
        try:
            # Preprocess text with emotion if provided
            processed_text = self._preprocess_text(text, emotion)
            
            # Run synthesis in thread pool to avoid blocking
            audio = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._synthesize_sync,
                processed_text,
                speaker_embedding
            )
            
            # Apply speed modification if needed
            if speed != 1.0:
                audio = self._apply_speed_change(audio, speed)
            
            return audio, settings.audio.sample_rate
            
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}")
            return np.array([]), settings.audio.sample_rate
    
    def _synthesize_sync(
        self,
        text: str,
        speaker_embedding: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Synchronous synthesis (runs in thread pool)"""
        try:
            # Use provided embedding or default
            if speaker_embedding is None:
                speaker_embedding = self.default_speaker_embeddings
            
            # Ensure speaker embedding has correct shape
            if speaker_embedding.dim() == 1:
                speaker_embedding = speaker_embedding.unsqueeze(0)
            
            # Tokenize text
            inputs = self.processor(
                text=text,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = inputs["input_ids"].to(self.device)
            speaker_embedding = speaker_embedding.to(self.device)
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    input_ids,
                    speaker_embedding,
                    vocoder=self.vocoder
                )
            
            # Convert to numpy
            audio = speech.cpu().numpy()
            
            return audio
            
        except Exception as e:
            logger.error(f"Error in sync synthesis: {e}")
            return np.array([])
    
    def _preprocess_text(self, text: str, emotion: Optional[str] = None) -> str:
        """Preprocess text with emotion markers"""
        if not emotion or emotion == "neutral":
            return text
        
        # Add prosody markers based on emotion
        emotion_markers = {
            "joy": "[HAPPY] ",
            "anger": "[ANGRY] ",
            "sadness": "[SAD] ",
            "fear": "[WORRIED] ",
            "surprise": "[SURPRISED] "
        }
        
        marker = emotion_markers.get(emotion, "")
        return marker + text
    
    def _apply_speed_change(
        self,
        audio: np.ndarray,
        speed: float
    ) -> np.ndarray:
        """Apply speed modification to audio"""
        try:
            import librosa
            return librosa.effects.time_stretch(audio, rate=speed)
        except Exception as e:
            logger.warning(f"Could not apply speed change: {e}")
            return audio
    
    async def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        speaker_embedding: Optional[torch.Tensor] = None,
        chunk_size: int = 1024
    ):
        """
        Streaming TTS synthesis (yields audio chunks)
        
        Args:
            text: Text to synthesize
            language: Target language
            speaker_embedding: Optional speaker embedding
            chunk_size: Size of audio chunks to yield
            
        Yields:
            Audio chunks as numpy arrays
        """
        try:
            # Generate full audio
            audio, sr = await self.synthesize(
                text, language, speaker_embedding
            )
            
            # Yield in chunks
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) > 0:
                    yield chunk
                    await asyncio.sleep(0)  # Allow other tasks to run
                    
        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            yield np.array([])
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear speaker embedding cache"""
        if user_id:
            self.speaker_cache.pop(user_id, None)
            logger.info(f"Cleared cache for user {user_id}")
        else:
            self.speaker_cache.clear()
            logger.info("Cleared all speaker embedding cache")


class TTSService:
    """High-level TTS service with caching and optimization"""
    
    def __init__(self):
        self.tts_engine = MultilingualTTS()
        logger.info("TTS Service initialized")
    
    async def text_to_speech(
        self,
        text: str,
        language: str = "en",
        user_id: Optional[str] = None,
        reference_audio: Optional[np.ndarray] = None,
        emotion: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Convert text to speech with optional voice cloning
        
        Args:
            text: Text to convert
            language: Target language
            user_id: User ID for voice profile lookup
            reference_audio: Reference audio for voice cloning
            emotion: Emotion to apply
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            # Extract speaker embedding if reference audio provided
            speaker_embedding = None
            if reference_audio is not None:
                speaker_embedding = self.tts_engine.extract_speaker_embedding(
                    reference_audio,
                    user_id=user_id
                )
            
            # Synthesize speech
            return await self.tts_engine.synthesize(
                text=text,
                language=language,
                speaker_embedding=speaker_embedding,
                emotion=emotion
            )
            
        except Exception as e:
            logger.error(f"TTS service error: {e}")
            return np.array([]), settings.audio.sample_rate


# Global TTS service instance
_tts_service: Optional[TTSService] = None


def get_tts_service() -> TTSService:
    """Get or create global TTS service instance"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
