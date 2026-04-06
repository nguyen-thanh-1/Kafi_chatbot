from typing import List, Dict, Optional
import time

class SessionManager:
    def __init__(self, max_history: int = 20):
        # In-memory dictionary: session_id -> List of message dicts
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves history for a given session."""
        return self.sessions.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        """Adds a message to the session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        self.sessions[session_id].append({"role": role, "content": content})
        
        # Limit history size to prevent memory leaks/bloat
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]

    def clear_session(self, session_id: str):
        """Removes a session's history."""
        if session_id in self.sessions:
            del self.sessions[session_id]

# Singleton instance for the whole app
_manager_instance = None

def get_session_manager() -> SessionManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SessionManager()
    return _manager_instance
