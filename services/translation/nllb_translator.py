"""
Enterprise Translation Service
Multilingual neural translation using NLLB-200 (No Language Left Behind)
Supports 200+ languages with emotion preservation
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class TranslationResult:
    """Result of a translation"""
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    emotion: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EmotionDetector:
    """
    Emotion detection from text using RoBERTa
    Detects: joy, anger, sadness, fear, surprise, neutral
    """
    
    def __init__(self):
        logger.info("Initializing Emotion Detector")
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load emotion classification pipeline
        self.classifier = pipeline(
            "text-classification",
            model=settings.ai_models.emotion_model,
            device=self.device,
            top_k=None
        )
        
        logger.info("Emotion Detector initialized successfully")
    
    async def detect(
        self,
        text: str,
        threshold: float = 0.3
    ) -> str:
        """
        Detect emotion from text
        
        Args:
            text: Input text
            threshold: Minimum confidence threshold
            
        Returns:
            Detected emotion label
        """
        if not text or len(text.strip()) < 3:
            return "neutral"
        
        try:
            # Run in thread pool
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.classifier,
                text[:512]  # Truncate to model max length
            )
            
            # Get highest scoring emotion
            if results and len(results[0]) > 0:
                top_emotion = max(results[0], key=lambda x: x['score'])
                
                if top_emotion['score'] >= threshold:
                    return top_emotion['label'].lower()
            
            return "neutral"
            
        except Exception as e:
            logger.error(f"Emotion detection error: {e}")
            return "neutral"


class LanguageDetector:
    """
    Automatic language detection
    Uses fasttext-based language identification
    """
    
    def __init__(self):
        self.supported_languages = set(settings.supported_languages)
        logger.info(f"Language detector initialized with {len(self.supported_languages)} languages")
    
    async def detect(self, text: str) -> str:
        """
        Detect language from text
        
        Args:
            text: Input text
            
        Returns:
            Detected language code (NLLB format)
        """
        # Simple heuristic-based detection
        # In production, use proper language detection like fasttext
        
        if not text:
            return "eng_Latn"
        
        # Check for common patterns
        # Hindi (Devanagari script)
        if any('\u0900' <= c <= '\u097F' for c in text):
            return "hin_Deva"
        
        # Chinese (CJK)
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "cmn_Hans"
        
        # Arabic
        if any('\u0600' <= c <= '\u06FF' for c in text):
            return "ara_Arab"
        
        # Japanese
        if any('\u3040' <= c <= '\u309F' for c in text) or any('\u30A0' <= c <= '\u30FF' for c in text):
            return "jpn_Jpan"
        
        # Korean
        if any('\uAC00' <= c <= '\uD7AF' for c in text):
            return "kor_Hang"
        
        # Default to English
        return "eng_Latn"


class NLLB200Translator:
    """
    Neural translation using NLLB-200 model
    Supports 200+ languages with high quality
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing NLLB-200 Translator on {self.device}")
        
        # Load model and tokenizer
        self.model_name = settings.ai_models.nllb_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        # Enable optimizations
        if self.device == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Language mapping cache
        self.language_cache = {}
        
        logger.info("NLLB-200 Translator initialized successfully")
    
    def _get_language_code(self, lang: str) -> str:
        """Convert language code to NLLB format if needed"""
        # Map common codes to NLLB codes
        lang_map = {
            "en": "eng_Latn",
            "hi": "hin_Deva",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "zh": "cmn_Hans",
            "ar": "ara_Arab",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "pt": "por_Latn",
        }
        
        return lang_map.get(lang, lang)
    
    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None
    ) -> str:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (NLLB format)
            target_lang: Target language code (NLLB format)
            max_length: Maximum output length
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        try:
            # Convert language codes
            source_lang = self._get_language_code(source_lang)
            target_lang = self._get_language_code(target_lang)
            
            # Run translation in thread pool
            translated = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._translate_sync,
                text,
                source_lang,
                target_lang,
                max_length
            )
            
            return translated
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original on error
    
    def _translate_sync(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None
    ) -> str:
        """Synchronous translation (runs in thread pool)"""
        try:
            # Set source language
            self.tokenizer.src_lang = source_lang
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.ai_models.max_translation_length
            ).to(self.device)
            
            # Get target language token ID
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(
                target_lang
            )
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length or settings.ai_models.max_translation_length,
                    num_beams=3,
                    length_penalty=1.0,
                    early_stopping=True,
                    do_sample=False
                )
            
            # Decode
            translated = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True
            )[0]
            
            return translated.strip()
            
        except Exception as e:
            logger.error(f"Sync translation error: {e}")
            return text
    
    async def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str
    ) -> List[str]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # Process in batches for efficiency
        batch_size = settings.ai_models.translation_batch_size
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.translate(text, source_lang, target_lang)
                for text in batch
            ])
            results.extend(batch_results)
        
        return results


class EmotionPreservingTranslator:
    """
    Translation service with emotion detection and preservation
    Combines translation with emotional context
    """
    
    def __init__(self):
        self.translator = NLLB200Translator()
        self.emotion_detector = EmotionDetector()
        self.language_detector = LanguageDetector()
        
        # Emotion markers for different languages
        self.emotion_markers = {
            "eng_Latn": {
                "joy": "[HAPPY] ",
                "anger": "[ANGRY] ",
                "sadness": "[SAD] ",
                "fear": "[WORRIED] ",
                "surprise": "[SURPRISED] ",
                "neutral": ""
            },
            "hin_Deva": {
                "joy": "[खुश] ",
                "anger": "[गुस्सा] ",
                "sadness": "[उदास] ",
                "fear": "[चिंतित] ",
                "surprise": "[हैरान] ",
                "neutral": ""
            },
            "spa_Latn": {
                "joy": "[FELIZ] ",
                "anger": "[ENOJADO] ",
                "sadness": "[TRISTE] ",
                "fear": "[PREOCUPADO] ",
                "surprise": "[SORPRENDIDO] ",
                "neutral": ""
            }
        }
        
        logger.info("Emotion-Preserving Translator initialized")
    
    async def translate_with_emotion(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "eng_Latn",
        detect_emotion: bool = True,
        preserve_emotion: bool = True
    ) -> TranslationResult:
        """
        Translate text with emotion detection and preservation
        
        Args:
            text: Text to translate
            source_lang: Source language (auto-detect if None)
            target_lang: Target language
            detect_emotion: Whether to detect emotion
            preserve_emotion: Whether to add emotion markers
            
        Returns:
            TranslationResult object
        """
        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text="",
                source_language=source_lang or "unknown",
                target_language=target_lang,
                confidence=0.0
            )
        
        try:
            # Auto-detect source language if not provided
            if source_lang is None:
                source_lang = await self.language_detector.detect(text)
            
            # Detect emotion if requested
            emotion = "neutral"
            if detect_emotion:
                emotion = await self.emotion_detector.detect(text)
            
            # Translate text
            translated = await self.translator.translate(
                text, source_lang, target_lang
            )
            
            # Add emotion marker if requested
            if preserve_emotion and emotion != "neutral":
                markers = self.emotion_markers.get(target_lang, {})
                marker = markers.get(emotion, "")
                translated = marker + translated
            
            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.9,  # TODO: Calculate actual confidence
                emotion=emotion
            )
            
        except Exception as e:
            logger.error(f"Translation with emotion error: {e}")
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_language=source_lang or "unknown",
                target_language=target_lang,
                confidence=0.0
            )


class TranslationService:
    """High-level translation service with caching and optimization"""
    
    def __init__(self):
        self.translator = EmotionPreservingTranslator()
        self.translation_cache: Dict[str, TranslationResult] = {}
        self.max_cache_size = 1000
        logger.info("Translation Service initialized")
    
    def _get_cache_key(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Generate cache key for translation"""
        return f"{source_lang}:{target_lang}:{hash(text)}"
    
    async def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: str = "eng_Latn",
        use_cache: bool = True,
        detect_emotion: bool = True
    ) -> TranslationResult:
        """
        Translate text with optional caching
        
        Args:
            text: Text to translate
            source_lang: Source language (auto-detect if None)
            target_lang: Target language
            use_cache: Whether to use cache
            detect_emotion: Whether to detect emotion
            
        Returns:
            TranslationResult
        """
        # Check cache
        if use_cache and source_lang:
            cache_key = self._get_cache_key(text, source_lang, target_lang)
            if cache_key in self.translation_cache:
                logger.debug(f"Cache hit for translation: {cache_key}")
                return self.translation_cache[cache_key]
        
        # Perform translation
        result = await self.translator.translate_with_emotion(
            text, source_lang, target_lang, detect_emotion
        )
        
        # Cache result
        if use_cache and len(self.translation_cache) < self.max_cache_size:
            cache_key = self._get_cache_key(
                text, result.source_language, target_lang
            )
            self.translation_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear translation cache"""
        self.translation_cache.clear()
        logger.info("Translation cache cleared")


# Global translation service instance
_translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """Get or create global translation service instance"""
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service
