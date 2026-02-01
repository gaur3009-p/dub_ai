"""
Example WebSocket Client for DubYou Enterprise
Demonstrates real-time bidirectional translation
"""

import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import queue
from datetime import datetime

# Configuration
SERVER_URL = "ws://localhost:8000/ws/translate"
SESSION_ID = "demo-session-123"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024


class DubYouClient:
    """WebSocket client for real-time translation"""
    
    def __init__(
        self,
        session_id: str,
        user_id: str,
        source_lang: str = "eng_Latn",
        target_lang: str = "hin_Deva"
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        self.websocket = None
        self.audio_queue = queue.Queue()
        self.running = False
        
    async def connect(self):
        """Connect to WebSocket server"""
        uri = f"{SERVER_URL}/{self.session_id}"
        
        try:
            self.websocket = await websockets.connect(uri)
            print(f"✓ Connected to {uri}")
            
            # Send configuration
            await self.send_config()
            
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    async def send_config(self):
        """Send client configuration"""
        config = {
            "type": "config",
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "user_id": self.user_id
        }
        
        await self.websocket.send(json.dumps(config))
        print(f"✓ Configuration sent")
    
    async def send_audio(self, audio_data: np.ndarray):
        """Send audio chunk to server"""
        message = {
            "type": "audio",
            "data": audio_data.tolist(),
            "sample_rate": SAMPLE_RATE
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def receive_messages(self):
        """Receive and handle messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"Error receiving messages: {e}")
    
    async def handle_message(self, message: dict):
        """Handle incoming message"""
        msg_type = message.get("type")
        
        if msg_type == "transcript":
            text = message.get("text", "")
            is_final = message.get("is_final", False)
            marker = "✓" if is_final else "..."
            print(f"{marker} Transcript: {text}")
        
        elif msg_type == "translation":
            text = message.get("text", "")
            emotion = message.get("emotion", "neutral")
            print(f"📝 Translation ({emotion}): {text}")
        
        elif msg_type == "audio":
            audio_data = np.array(message.get("data", []), dtype=np.float32)
            sample_rate = message.get("sample_rate", SAMPLE_RATE)
            
            if len(audio_data) > 0:
                print(f"🔊 Playing translated audio ({len(audio_data)} samples)")
                self.play_audio(audio_data, sample_rate)
        
        elif msg_type == "error":
            error_msg = message.get("message", "Unknown error")
            print(f"❌ Error: {error_msg}")
        
        elif msg_type == "user_joined":
            user_id = message.get("user_id")
            print(f"👋 User joined: {user_id}")
        
        elif msg_type == "user_left":
            user_id = message.get("user_id")
            print(f"👋 User left: {user_id}")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input"""
        if status:
            print(f"Audio status: {status}")
        
        # Add to queue
        self.audio_queue.put(indata.copy())
    
    async def stream_audio(self):
        """Stream audio from microphone"""
        print("🎤 Starting audio capture...")
        
        # Start audio stream
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=CHUNK_SIZE,
            callback=self.audio_callback
        )
        
        with stream:
            self.running = True
            
            while self.running:
                try:
                    # Get audio chunk from queue
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    
                    # Send to server
                    await self.send_audio(audio_chunk.flatten())
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Error streaming audio: {e}")
                    break
    
    @staticmethod
    def play_audio(audio_data: np.ndarray, sample_rate: int):
        """Play audio through speakers"""
        try:
            sd.play(audio_data, sample_rate)
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    async def run(self):
        """Run client"""
        if not await self.connect():
            return
        
        print("\n" + "="*50)
        print("DubYou Client Running")
        print("="*50)
        print(f"Source Language: {self.source_lang}")
        print(f"Target Language: {self.target_lang}")
        print(f"User ID: {self.user_id}")
        print("\nSpeak into your microphone...")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")
        
        try:
            # Run audio streaming and message receiving concurrently
            await asyncio.gather(
                self.stream_audio(),
                self.receive_messages()
            )
        except KeyboardInterrupt:
            print("\n\nStopping client...")
        finally:
            self.running = False
            if self.websocket:
                await self.websocket.close()
            print("✓ Client stopped")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DubYou Client")
    parser.add_argument(
        "--session",
        default=SESSION_ID,
        help="Session ID"
    )
    parser.add_argument(
        "--user",
        default="user-1",
        help="User ID"
    )
    parser.add_argument(
        "--source",
        default="eng_Latn",
        help="Source language code"
    )
    parser.add_argument(
        "--target",
        default="hin_Deva",
        help="Target language code"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = DubYouClient(
        session_id=args.session,
        user_id=args.user,
        source_lang=args.source,
        target_lang=args.target
    )
    
    # Run
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
