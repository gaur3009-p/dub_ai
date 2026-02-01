"""
WebSocket Connection Manager
Handles real-time bidirectional communication for live translation
"""

import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from fastapi import WebSocket, WebSocketDisconnect
import structlog

from config.settings import get_settings

logger = structlog.get_logger()
settings = get_settings()


class ConnectionInfo:
    """Information about a WebSocket connection"""
    
    def __init__(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: Optional[str] = None
    ):
        self.websocket = websocket
        self.session_id = session_id
        self.user_id = user_id
        self.connected_at = datetime.now()
        self.last_activity = datetime.now()
        self.config: Dict[str, Any] = {}
        self.audio_buffer = []
        
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_active(self, timeout_seconds: int = 300) -> bool:
        """Check if connection is still active"""
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed < timeout_seconds


class SessionManager:
    """Manages translation sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Set[ConnectionInfo]] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        
    def create_session(self, session_id: str) -> None:
        """Create a new session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = set()
            logger.info("Session created", session_id=session_id)
    
    def add_to_session(
        self,
        session_id: str,
        connection: ConnectionInfo
    ) -> None:
        """Add connection to session"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id].add(connection)
        
        if connection.user_id:
            self.user_sessions[connection.user_id] = session_id
        
        logger.info(
            "Connection added to session",
            session_id=session_id,
            user_id=connection.user_id,
            total_connections=len(self.sessions[session_id])
        )
    
    def remove_from_session(
        self,
        session_id: str,
        connection: ConnectionInfo
    ) -> None:
        """Remove connection from session"""
        if session_id in self.sessions:
            self.sessions[session_id].discard(connection)
            
            if connection.user_id:
                self.user_sessions.pop(connection.user_id, None)
            
            # Remove empty sessions
            if not self.sessions[session_id]:
                del self.sessions[session_id]
                logger.info("Session removed (no connections)", session_id=session_id)
    
    def get_session_connections(self, session_id: str) -> Set[ConnectionInfo]:
        """Get all connections in a session"""
        return self.sessions.get(session_id, set())
    
    def get_other_connections(
        self,
        session_id: str,
        exclude_connection: ConnectionInfo
    ) -> Set[ConnectionInfo]:
        """Get other connections in the same session"""
        connections = self.get_session_connections(session_id)
        return {c for c in connections if c != exclude_connection}
    
    async def broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        exclude_connection: Optional[ConnectionInfo] = None
    ) -> None:
        """Broadcast message to all connections in session"""
        connections = self.get_session_connections(session_id)
        
        for connection in connections:
            if connection != exclude_connection:
                try:
                    await connection.websocket.send_json(message)
                except Exception as e:
                    logger.error(
                        "Error broadcasting to connection",
                        session_id=session_id,
                        user_id=connection.user_id,
                        error=str(e)
                    )


class WebSocketManager:
    """
    Manages WebSocket connections for real-time translation
    
    Features:
    - Connection lifecycle management
    - Session management for multi-user conversations
    - Message routing between connected clients
    - Timeout handling
    """
    
    def __init__(self):
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.session_manager = SessionManager()
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info("WebSocket Manager initialized")
    
    def _start_cleanup_task(self):
        """Start background task for connection cleanup"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(60)  # Check every minute
                    await self._cleanup_inactive_connections()
                except Exception as e:
                    logger.error("Cleanup task error", error=str(e))
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_inactive_connections(self):
        """Remove inactive connections"""
        timeout = settings.session_timeout
        inactive = []
        
        for conn_id, conn_info in self.active_connections.items():
            if not conn_info.is_active(timeout):
                inactive.append((conn_id, conn_info))
        
        for conn_id, conn_info in inactive:
            logger.info(
                "Removing inactive connection",
                connection_id=conn_id,
                session_id=conn_info.session_id
            )
            await self.unregister_connection(
                conn_info.websocket,
                conn_info.session_id
            )
    
    async def register_connection(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Register a new WebSocket connection
        
        Args:
            websocket: WebSocket instance
            session_id: Session identifier
            user_id: Optional user identifier
            config: Optional connection configuration
            
        Returns:
            Connection information dictionary
        """
        # Create connection info
        connection = ConnectionInfo(websocket, session_id, user_id)
        
        if config:
            connection.config = config
        
        # Generate unique connection ID
        connection_id = f"{session_id}:{user_id or id(websocket)}"
        
        # Store connection
        self.active_connections[connection_id] = connection
        
        # Add to session
        self.session_manager.add_to_session(session_id, connection)
        
        logger.info(
            "Connection registered",
            connection_id=connection_id,
            session_id=session_id,
            user_id=user_id,
            total_connections=len(self.active_connections)
        )
        
        # Notify other connections in the session
        await self.session_manager.broadcast_to_session(
            session_id,
            {
                "type": "user_joined",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            },
            exclude_connection=connection
        )
        
        return {
            "connection_id": connection_id,
            "session_id": session_id,
            "user_id": user_id,
            "config": connection.config
        }
    
    async def unregister_connection(
        self,
        websocket: WebSocket,
        session_id: str
    ) -> None:
        """
        Unregister a WebSocket connection
        
        Args:
            websocket: WebSocket instance
            session_id: Session identifier
        """
        # Find and remove connection
        connection_id = None
        connection_info = None
        
        for conn_id, conn in self.active_connections.items():
            if conn.websocket == websocket and conn.session_id == session_id:
                connection_id = conn_id
                connection_info = conn
                break
        
        if connection_id and connection_info:
            # Remove from session
            self.session_manager.remove_from_session(
                session_id,
                connection_info
            )
            
            # Remove from active connections
            del self.active_connections[connection_id]
            
            logger.info(
                "Connection unregistered",
                connection_id=connection_id,
                session_id=session_id,
                user_id=connection_info.user_id,
                remaining_connections=len(self.active_connections)
            )
            
            # Notify other connections
            await self.session_manager.broadcast_to_session(
                session_id,
                {
                    "type": "user_left",
                    "user_id": connection_info.user_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def receive_messages(
        self,
        websocket: WebSocket
    ):
        """
        Async generator for receiving messages from WebSocket
        
        Args:
            websocket: WebSocket instance
            
        Yields:
            Parsed message dictionaries
        """
        try:
            while True:
                # Receive message
                data = await websocket.receive()
                
                # Update activity timestamp
                for conn in self.active_connections.values():
                    if conn.websocket == websocket:
                        conn.update_activity()
                        break
                
                # Parse message based on type
                if "text" in data:
                    message = json.loads(data["text"])
                    yield message
                
                elif "bytes" in data:
                    # Binary data (audio)
                    audio_data = np.frombuffer(
                        data["bytes"],
                        dtype=np.float32
                    )
                    yield {
                        "type": "audio",
                        "data": audio_data
                    }
        
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected normally")
            raise
        
        except Exception as e:
            logger.error("Error receiving message", error=str(e))
            raise
    
    async def send_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        sender_websocket: Optional[WebSocket] = None
    ) -> None:
        """
        Send message to all connections in a session
        
        Args:
            session_id: Session identifier
            message: Message to send
            sender_websocket: Optional sender to exclude from broadcast
        """
        connections = self.session_manager.get_session_connections(session_id)
        
        for connection in connections:
            if sender_websocket and connection.websocket == sender_websocket:
                continue
            
            try:
                await connection.websocket.send_json(message)
            except Exception as e:
                logger.error(
                    "Error sending message",
                    session_id=session_id,
                    user_id=connection.user_id,
                    error=str(e)
                )
    
    async def send_audio_to_peer(
        self,
        session_id: str,
        audio_data: np.ndarray,
        sender_websocket: WebSocket,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send translated audio to peer in the session
        
        Args:
            session_id: Session identifier
            audio_data: Audio array to send
            sender_websocket: Sender's WebSocket
            metadata: Optional metadata to include
        """
        # Get other connections in session
        sender_connection = None
        for conn in self.active_connections.values():
            if conn.websocket == sender_websocket:
                sender_connection = conn
                break
        
        if not sender_connection:
            return
        
        other_connections = self.session_manager.get_other_connections(
            session_id,
            sender_connection
        )
        
        # Prepare message
        message = {
            "type": "audio",
            "data": audio_data.tolist(),
            "sender_id": sender_connection.user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            message.update(metadata)
        
        # Send to other connections
        for connection in other_connections:
            try:
                await connection.websocket.send_json(message)
            except Exception as e:
                logger.error(
                    "Error sending audio to peer",
                    session_id=session_id,
                    peer_user_id=connection.user_id,
                    error=str(e)
                )
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session information dictionary
        """
        connections = self.session_manager.get_session_connections(session_id)
        
        return {
            "session_id": session_id,
            "connection_count": len(connections),
            "users": [
                {
                    "user_id": conn.user_id,
                    "connected_at": conn.connected_at.isoformat(),
                    "last_activity": conn.last_activity.isoformat(),
                    "config": conn.config
                }
                for conn in connections
            ]
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall WebSocket statistics"""
        return {
            "active_connections": len(self.active_connections),
            "active_sessions": len(self.session_manager.sessions),
            "connections_by_session": {
                session_id: len(connections)
                for session_id, connections in self.session_manager.sessions.items()
            }
        }


# Global WebSocket manager instance
_ws_manager: Optional[WebSocketManager] = None


def get_websocket_manager() -> WebSocketManager:
    """Get or create global WebSocket manager instance"""
    global _ws_manager
    if _ws_manager is None:
        _ws_manager = WebSocketManager()
    return _ws_manager
