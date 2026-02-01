"""
Enterprise ASR (Automatic Speech Recognition) Service
Real-time speech-to-text with streaming support using Whisper
"""

import torch
import numpy as np
from typing import Optional, AsyncIterator, Tuple, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from datetime import datetime

from faster_whisper import WhisperModel
import webrtcvad

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TranscriptionResult:
    """Result of a transcription"""
    text: str
    language: str
    confidence: float
    timestamp: datetime
    duration: float
    is_final: bool = False


class VADProcessor:
    """
    Voice Activity Detection processor
    Detects speech segments in audio streams
    """
    
    def __init__(self, aggressiveness: int = 2):
        """
        Initialize VAD processor
        
        Args:
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = 16000
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        self.speech_frames = []
        self.silence_frames = 0
        self.max_silence_frames = 30  # ~900ms of silence
        
        logger.info(f"VAD initialized with aggressiveness {aggressiveness}")
    
    def process_frame(self, frame: bytes) -> bool:
        """
        Process audio frame and detect speech
        
        Args:
            frame: Audio frame as bytes (must be 16-bit PCM)
            
        Returns:
            True if speech is detected
        """
        try:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            return is_speech
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            return False
    
    def process_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Process audio and return speech segments
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Tuple of (has_speech, speech_segments)
        """
        # Convert to 16-bit PCM bytes
        audio_int16 = (audio * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        has_speech = False
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        # Process in frames
        for i in range(0, len(audio_bytes) - self.frame_size, self.frame_size):
            frame = audio_bytes[i:i + self.frame_size]
            
            if len(frame) < self.frame_size:
                break
            
            is_speech = self.process_frame(frame)
            
            if is_speech:
                has_speech = True
                if not in_speech:
                    speech_start = i // 2  # Convert bytes to samples
                    in_speech = True
                self.silence_frames = 0
            else:
                if in_speech:
                    self.silence_frames += 1
                    if self.silence_frames > self.max_silence_frames:
                        speech_end = i // 2
                        speech_segments.append((speech_start, speech_end))
                        in_speech = False
        
        # Handle final segment
        if in_speech:
            speech_segments.append((speech_start, len(audio)))
        
        return has_speech, speech_segments


class StreamingWhisperASR:
    """
    Streaming ASR using Faster Whisper
    Optimized for low-latency real-time transcription
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = settings.ai_models.whisper_compute_type
        self.model_size = settings.ai_models.whisper_model
        
        logger.info(
            f"Initializing Whisper ASR: model={self.model_size}, "
            f"device={self.device}, compute_type={self.compute_type}"
        )
        
        # Initialize Whisper model
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=4,
            num_workers=1
        )
        
        # VAD for speech detection
        self.vad = VADProcessor(aggressiveness=2)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Buffer for streaming
        self.audio_buffer = []
        self.buffer_duration = 0.0
        self.max_buffer_duration = 30.0  # 30 seconds max
        
        logger.info("Whisper ASR initialized successfully")
    
    async def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> TranscriptionResult:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio array (float32, -1.0 to 1.0)
            language: Source language code (None for auto-detect)
            task: "transcribe" or "translate"
            
        Returns:
            TranscriptionResult object
        """
        if audio is None or len(audio) < 1600:  # Minimum 0.1s at 16kHz
            return TranscriptionResult(
                text="",
                language=language or "unknown",
                confidence=0.0,
                timestamp=datetime.now(),
                duration=0.0
            )
        
        try:
            start_time = datetime.now()
            
            # Run transcription in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio,
                language,
                task
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return TranscriptionResult(
                text=result["text"],
                language=result["language"],
                confidence=result["confidence"],
                timestamp=datetime.now(),
                duration=duration,
                is_final=True
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return TranscriptionResult(
                text="",
                language=language or "unknown",
                confidence=0.0,
                timestamp=datetime.now(),
                duration=0.0
            )
    
    def _transcribe_sync(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> dict:
        """Synchronous transcription (runs in thread pool)"""
        try:
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.abs(audio).max()
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio,
                language=language,
                task=task,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=30,
                    min_silence_duration_ms=500
                ),
                condition_on_previous_text=False
            )
            
            # Combine segments
            text_parts = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                if segment.avg_logprob > -1.0:  # Filter low confidence
                    text_parts.append(segment.text.strip())
                    total_confidence += np.exp(segment.avg_logprob)
                    segment_count += 1
            
            text = " ".join(text_parts)
            confidence = (
                total_confidence / segment_count
                if segment_count > 0
                else 0.0
            )
            
            return {
                "text": text,
                "language": info.language,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Sync transcription error: {e}")
            return {
                "text": "",
                "language": language or "unknown",
                "confidence": 0.0
            }
    
    async def transcribe_streaming(
        self,
        audio_chunks: AsyncIterator[np.ndarray],
        language: Optional[str] = None,
        window_duration: float = 3.0
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Streaming transcription with sliding window
        
        Args:
            audio_chunks: Async iterator of audio chunks
            language: Source language
            window_duration: Duration of sliding window in seconds
            
        Yields:
            TranscriptionResult objects
        """
        buffer = []
        buffer_duration = 0.0
        window_samples = int(window_duration * settings.audio.sample_rate)
        
        async for chunk in audio_chunks:
            if chunk is None or len(chunk) == 0:
                continue
            
            # Add to buffer
            buffer.append(chunk)
            buffer_duration += len(chunk) / settings.audio.sample_rate
            
            # Concatenate buffer
            audio = np.concatenate(buffer)
            
            # Check for speech
            has_speech, _ = self.vad.process_audio(
                audio, settings.audio.sample_rate
            )
            
            if has_speech:
                # Keep only last window_duration seconds
                if len(audio) > window_samples:
                    audio = audio[-window_samples:]
                
                # Transcribe
                result = await self.transcribe(audio, language)
                
                if result.text:
                    yield result
            
            # Manage buffer size
            if buffer_duration > window_duration:
                # Remove oldest chunks
                samples_to_remove = int(
                    (buffer_duration - window_duration) * settings.audio.sample_rate
                )
                removed = 0
                while removed < samples_to_remove and buffer:
                    chunk_len = len(buffer[0])
                    buffer.pop(0)
                    removed += chunk_len
                
                buffer_duration = sum(
                    len(c) / settings.audio.sample_rate for c in buffer
                )


class ASRService:
    """High-level ASR service with language detection and caching"""
    
    def __init__(self):
        self.asr_engine = StreamingWhisperASR()
        self.transcription_cache = {}
        logger.info("ASR Service initialized")
    
    async def transcribe_audio(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio with optional caching
        
        Args:
            audio: Audio array
            language: Source language (None for auto-detect)
            session_id: Optional session ID for context
            
        Returns:
            TranscriptionResult
        """
        return await self.asr_engine.transcribe(audio, language)
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        language: Optional[str] = None
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Transcribe streaming audio
        
        Args:
            audio_stream: Async iterator of audio chunks
            language: Source language
            
        Yields:
            TranscriptionResult objects
        """
        async for result in self.asr_engine.transcribe_streaming(
            audio_stream, language
        ):
            yield result


# Global ASR service instance
_asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    """Get or create global ASR service instance"""
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
