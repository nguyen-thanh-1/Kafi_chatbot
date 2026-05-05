from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Dict, Optional
from src.utils.app_config import BASE_DIR

class KnowledgeFileMetadata:
    def __init__(self):
        self.db_path = Path(BASE_DIR) / "data" / "knowledge_meta.sqlite"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL, -- 'processing', 'completed', 'error'
                    created_at REAL NOT NULL,
                    error_message TEXT
                )
            """)

    def add_file(self, filename: str) -> int:
        with sqlite3.connect(str(self.db_path)) as con:
            cur = con.execute(
                "INSERT INTO knowledge_files (filename, status, created_at) VALUES (?, ?, ?)",
                (filename, "processing", time.time())
            )
            return cur.lastrowid

    def update_status(self, file_id: int, status: str, error_message: Optional[str] = None):
        with sqlite3.connect(str(self.db_path)) as con:
            con.execute(
                "UPDATE knowledge_files SET status = ?, error_message = ? WHERE id = ?",
                (status, error_message, file_id)
            )

    def get_files(self) -> List[Dict]:
        with sqlite3.connect(str(self.db_path)) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM knowledge_files ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def delete_file(self, file_id: int) -> Optional[str]:
        with sqlite3.connect(str(self.db_path)) as con:
            row = con.execute("SELECT filename FROM knowledge_files WHERE id = ?", (file_id,)).fetchone()
            if row:
                filename = row[0]
                con.execute("DELETE FROM knowledge_files WHERE id = ?", (file_id,))
                return filename
            return None

_meta_instance = None

def get_knowledge_manager():
    global _meta_instance
    if _meta_instance is None:
        _meta_instance = KnowledgeFileMetadata()
    return _meta_instance
