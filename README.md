# DubYou Enterprise - Multilingual Real-Time Voice Translation

<p align="center">
  <strong>Enterprise-grade multilingual live conversation system with real-time translation and voice cloning</strong>
</p>

## 🌟 Features

### Core Capabilities
- **🎙️ Real-Time Speech Recognition**: Streaming ASR using Whisper (large-v3)
- **🌍 Multilingual Translation**: Support for 200+ languages using NLLB-200
- **🔊 Voice Cloning**: Preserve speaker identity across languages using SpeechT5
- **😊 Emotion Preservation**: Detect and maintain emotional tone in translations
- **🔄 Bidirectional Communication**: Person A speaks X, hears Y; Person B speaks Y, hears X

### Technical Features
- **FastAPI** backend with WebSocket support
- **Production-ready** with Docker and Docker Compose
- **GPU-accelerated** inference with CUDA support
- **Monitoring** with Prometheus and Grafana
- **Scalable** architecture with Redis caching and MongoDB storage
- **Free Hugging Face Models** - No paid APIs required

## 📋 Supported Languages

The system supports 200+ languages through NLLB-200, including:

- **English** (eng_Latn)
- **Hindi** (hin_Deva)
- **Spanish** (spa_Latn)
- **French** (fra_Latn)
- **German** (deu_Latn)
- **Chinese Simplified** (cmn_Hans)
- **Arabic** (ara_Arab)
- **Japanese** (jpn_Jpan)
- **Korean** (kor_Hang)
- **Portuguese** (por_Latn)
- And 190+ more languages...

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Person A  │ ←─────→ │   DubYou     │ ←─────→ │   Person B  │
│ (English)   │  WebSocket  Enterprise  │  WebSocket  (Hindi)   │
└─────────────┘         └──────────────┘         └─────────────┘
                               │
                   ┌───────────┴───────────┐
                   │                       │
            ┌──────▼──────┐         ┌─────▼─────┐
            │   Services   │         │  Storage  │
            ├─────────────┤         ├───────────┤
            │ • ASR       │         │ • MongoDB │
            │ • Translation│         │ • Redis   │
            │ • TTS       │         │ • Postgres│
            │ • Emotion   │         └───────────┘
            └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA 11.8+ (for optimal performance)
- 16GB+ RAM recommended
- 20GB+ disk space for models

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd dubyou-enterprise
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the services**
```bash
docker-compose up -d
```

4. **Verify installation**
```bash
curl http://localhost:8000/health
```

### Development Setup

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python main.py
```

## 📡 API Usage

### REST API

#### Translate Text
```bash
curl -X POST http://localhost:8000/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "eng_Latn",
    "target_lang": "hin_Deva",
    "detect_emotion": true
  }'
```

#### Enroll Voice
```bash
curl -X POST http://localhost:8000/api/v1/enroll \
  -H "Content-Type: application/json" \
  -d '{
    "audio_data": [0.1, 0.2, ...],
    "sample_rate": 16000,
    "user_name": "John Doe"
  }'
```

### WebSocket API

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/translate/session-123');

// Configure session
ws.send(JSON.stringify({
  type: 'config',
  source_lang: 'eng_Latn',
  target_lang: 'hin_Deva',
  user_id: 'user-456'
}));

// Send audio chunk
ws.send(JSON.stringify({
  type: 'audio',
  data: audioArray,
  sample_rate: 16000
}));

// Receive messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'transcript':
      console.log('Transcript:', message.text);
      break;
    case 'translation':
      console.log('Translation:', message.text);
      break;
    case 'audio':
      playAudio(message.data);
      break;
  }
};
```

## 🎯 Use Cases

### 1. Live Bilingual Conversations
Enable real-time conversations between people speaking different languages:
- Person A speaks English → Person B hears Hindi (in Person A's voice)
- Person B speaks Hindi → Person A hears English (in Person B's voice)

### 2. International Meetings
Support multilingual meetings with automatic translation and voice preservation.

### 3. Customer Support
Provide customer support in multiple languages with consistent voice quality.

### 4. Language Learning
Practice conversations with native-like pronunciation in target languages.

## 🔧 Configuration

### Environment Variables

See `.env.example` for all configuration options:

```bash
# Application
APP_NAME=DubYou Enterprise
ENVIRONMENT=production

# AI Models
WHISPER_MODEL=large-v3
NLLB_MODEL=facebook/nllb-200-distilled-600M
TTS_MODEL=microsoft/speecht5_tts

# Performance
WORKERS=4
TRANSLATION_BATCH_SIZE=8
```

### Model Configuration

Models are automatically downloaded on first run. To pre-download:

```bash
python scripts/download_models.py
```

## 📊 Monitoring

### Prometheus Metrics
Available at: `http://localhost:9090`

Metrics include:
- Translation count and duration
- WebSocket connection stats
- Model inference times
- Error rates

### Grafana Dashboards
Available at: `http://localhost:3000` (admin/admin)

Pre-configured dashboards for:
- System overview
- Translation performance
- WebSocket connections
- Resource utilization

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_translation.py

# Run with coverage
pytest --cov=services tests/
```

## 🔒 Security Considerations

1. **Authentication**: Implement JWT authentication for production
2. **Rate Limiting**: Configure appropriate rate limits
3. **HTTPS**: Use SSL/TLS in production (configured in nginx)
4. **API Keys**: Secure sensitive endpoints
5. **Input Validation**: All inputs are validated with Pydantic

## 📈 Performance Optimization

### GPU Acceleration
- Ensure CUDA is properly configured
- Use `float16` precision for faster inference
- Batch processing for multiple requests

### Caching
- Redis caching for frequent translations
- Speaker embedding caching for voice cloning
- Model output caching

### Scaling
- Horizontal scaling with load balancer
- Multiple worker processes
- Dedicated GPU per worker for optimal performance

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Use smaller models
   - Add more RAM/VRAM

2. **Slow Translation**
   - Check GPU utilization
   - Verify CUDA installation
   - Use model quantization

3. **WebSocket Disconnects**
   - Increase timeout settings
   - Check network stability
   - Review logs for errors

### Logs

```bash
# View application logs
docker-compose logs -f dubyou-app

# View specific service logs
docker-compose logs -f mongodb
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Whisper** by OpenAI for speech recognition
- **NLLB-200** by Meta AI for translation
- **SpeechT5** by Microsoft for voice synthesis
- **Hugging Face** for model hosting
- All open-source contributors

## 📞 Support

- **Issues**: GitHub Issues
- **Documentation**: [docs/](docs/)
- **Email**: support@dubyou-enterprise.com

## 🗺️ Roadmap

- [ ] Support for more languages
- [ ] Mobile app integration
- [ ] Browser extension
- [ ] Real-time transcription export
- [ ] Multi-party conversations (3+ people)
- [ ] Advanced emotion models
- [ ] Custom voice training
- [ ] On-device processing option

---

<p align="center">
  Made with ❤️ for breaking language barriers
</p>
