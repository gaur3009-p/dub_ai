"""
Download and cache AI models
Pre-downloads all required models to avoid delays during runtime
"""

import os
from pathlib import Path
import logging

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    pipeline
)
from faster_whisper import WhisperModel
from speechbrain.pretrained import EncoderClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def download_whisper():
    """Download Whisper model"""
    logger.info("Downloading Whisper model...")
    try:
        model = WhisperModel(
            "large-v3",
            device="cpu",
            compute_type="int8",
            download_root=str(MODELS_DIR / "whisper")
        )
        logger.info("✓ Whisper model downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download Whisper: {e}")
        return False


def download_nllb():
    """Download NLLB translation model"""
    logger.info("Downloading NLLB-200 model...")
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        cache_dir = str(MODELS_DIR / "nllb")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        logger.info("✓ NLLB-200 model downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download NLLB: {e}")
        return False


def download_speecht5():
    """Download SpeechT5 TTS model"""
    logger.info("Downloading SpeechT5 TTS model...")
    try:
        cache_dir = str(MODELS_DIR / "tts")
        
        processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=cache_dir
        )
        model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=cache_dir
        )
        vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            cache_dir=cache_dir
        )
        logger.info("✓ SpeechT5 TTS model downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download SpeechT5: {e}")
        return False


def download_speaker_encoder():
    """Download speaker encoder model"""
    logger.info("Downloading speaker encoder model...")
    try:
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(MODELS_DIR / "speaker_encoder"),
            run_opts={"device": "cpu"}
        )
        logger.info("✓ Speaker encoder model downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download speaker encoder: {e}")
        return False


def download_emotion_model():
    """Download emotion detection model"""
    logger.info("Downloading emotion detection model...")
    try:
        cache_dir = str(MODELS_DIR / "emotion")
        
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=-1,  # CPU
            model_kwargs={"cache_dir": cache_dir}
        )
        logger.info("✓ Emotion detection model downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download emotion model: {e}")
        return False


def download_speaker_embeddings():
    """Download default speaker embeddings"""
    logger.info("Downloading default speaker embeddings...")
    try:
        from datasets import load_dataset
        
        cache_dir = str(MODELS_DIR / "embeddings")
        
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors",
            split="validation",
            cache_dir=cache_dir
        )
        logger.info("✓ Speaker embeddings downloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to download speaker embeddings: {e}")
        return False


def main():
    """Download all models"""
    logger.info("=" * 50)
    logger.info("DubYou Enterprise - Model Download")
    logger.info("=" * 50)
    
    models = [
        ("Whisper (ASR)", download_whisper),
        ("NLLB-200 (Translation)", download_nllb),
        ("SpeechT5 (TTS)", download_speecht5),
        ("Speaker Encoder", download_speaker_encoder),
        ("Emotion Detection", download_emotion_model),
        ("Speaker Embeddings", download_speaker_embeddings),
    ]
    
    results = []
    
    for name, download_func in models:
        logger.info(f"\n--- {name} ---")
        success = download_func()
        results.append((name, success))
    
    logger.info("\n" + "=" * 50)
    logger.info("Download Summary")
    logger.info("=" * 50)
    
    for name, success in results:
        status = "✓" if success else "✗"
        logger.info(f"{status} {name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        logger.info("\n✓ All models downloaded successfully!")
        return 0
    else:
        logger.warning("\n⚠ Some models failed to download")
        return 1


if __name__ == "__main__":
    exit(main())
