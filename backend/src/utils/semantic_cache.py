from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import numpy as np

from src.utils.app_config import AppConfig, BASE_DIR
from src.utils.logger import logger


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32, copy=False).tobytes()


def _from_blob(blob: bytes, dim: int) -> np.ndarray:
    vec = np.frombuffer(blob, dtype=np.float32)
    if vec.size != dim:
        return np.zeros((dim,), dtype=np.float32)
    return vec


@dataclass(frozen=True)
class CacheHit:
    response: str
    similarity: float
    route: str
    created_at: float


class SemanticCache:
    def __init__(self):
        cfg = AppConfig.get_pipeline_config().get("cache", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.similarity_threshold = float(cfg.get("similarity_threshold", 0.92))
        self.response_similarity_threshold = float(cfg.get("response_similarity_threshold", self.similarity_threshold))
        self.max_entries = int(cfg.get("max_entries", 5000))
        self.top_k_scan = int(cfg.get("top_k_scan", 2000))

        db_rel = cfg.get("db_path", ".cache/semantic_cache.sqlite")
        self.db_path = Path(BASE_DIR) / db_rel
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._dim: Optional[int] = None

        if self.enabled:
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    query_text TEXT NOT NULL,
                    query_hash TEXT NOT NULL,
                    route TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    query_embedding BLOB NOT NULL,
                    response_embedding_dim INTEGER,
                    response_embedding BLOB,
                    response_text TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_hash ON semantic_cache(query_hash);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_semantic_cache_created ON semantic_cache(created_at);")

            # Migration for older DBs (add response embedding columns if missing)
            cols = [r[1] for r in con.execute("PRAGMA table_info(semantic_cache);").fetchall()]
            if "response_embedding_dim" not in cols:
                con.execute("ALTER TABLE semantic_cache ADD COLUMN response_embedding_dim INTEGER;")
            if "response_embedding" not in cols:
                con.execute("ALTER TABLE semantic_cache ADD COLUMN response_embedding BLOB;")

    def lookup(self, query_text: str, query_vec: np.ndarray) -> Optional[CacheHit]:
        if not self.enabled:
            return None

        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query_vec.size == 0:
            return None

        with self._lock:
            qh = _sha256(query_text.strip())
            dim = int(query_vec.size)
            self._dim = self._dim or dim

            with self._connect() as con:
                # Exact match short-circuit
                row = con.execute(
                    """
                    SELECT created_at, route, response_text
                    FROM semantic_cache
                    WHERE query_hash = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (qh,),
                ).fetchone()
                if row is not None:
                    created_at, route, response_text = row
                    return CacheHit(
                        response=str(response_text),
                        similarity=1.0,
                        route=str(route),
                        created_at=float(created_at),
                    )

                rows = con.execute(
                    """
                    SELECT created_at, route, response_text, embedding_dim, query_embedding
                    FROM semantic_cache
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (self.top_k_scan,),
                ).fetchall()

            if not rows:
                return None

            best: Optional[CacheHit] = None
            q = query_vec
            q_norm = float(np.linalg.norm(q) + 1e-12)

            for created_at, route, response_text, emb_dim, emb_blob in rows:
                if int(emb_dim) != dim:
                    continue
                v = _from_blob(emb_blob, dim)
                v_norm = float(np.linalg.norm(v) + 1e-12)
                sim = float(np.dot(q, v) / (q_norm * v_norm))
                if best is None or sim > best.similarity:
                    best = CacheHit(
                        response=str(response_text),
                        similarity=sim,
                        route=str(route),
                        created_at=float(created_at),
                    )

            if best is None:
                return None
            if best.similarity >= self.similarity_threshold:
                return best
            return None

    def has_similar_response(self, response_vec: np.ndarray) -> Optional[Tuple[float, str]]:
        if not self.enabled:
            return None

        response_vec = np.asarray(response_vec, dtype=np.float32).reshape(-1)
        if response_vec.size == 0:
            return None

        with self._lock:
            dim = int(response_vec.size)
            with self._connect() as con:
                rows = con.execute(
                    """
                    SELECT response_text, response_embedding_dim, response_embedding
                    FROM semantic_cache
                    WHERE response_embedding_dim IS NOT NULL AND response_embedding IS NOT NULL
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (self.top_k_scan,),
                ).fetchall()

            if not rows:
                return None

            best_sim = -1.0
            best_text = ""
            q = response_vec
            q_norm = float(np.linalg.norm(q) + 1e-12)

            for response_text, emb_dim, emb_blob in rows:
                if int(emb_dim) != dim:
                    continue
                v = _from_blob(emb_blob, dim)
                v_norm = float(np.linalg.norm(v) + 1e-12)
                sim = float(np.dot(q, v) / (q_norm * v_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_text = str(response_text)

            if best_sim >= self.response_similarity_threshold:
                return (float(best_sim), best_text)
            return None

    def store(
        self,
        query_text: str,
        query_vec: np.ndarray,
        route: str,
        response_text: str,
        *,
        response_vec: Optional[np.ndarray] = None,
    ) -> None:
        if not self.enabled:
            return

        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if query_vec.size == 0:
            return

        with self._lock:
            created_at = time.time()
            qh = _sha256(query_text.strip())
            dim = int(query_vec.size)
            self._dim = self._dim or dim

            r_dim = None
            r_blob = None
            if response_vec is not None:
                rv = np.asarray(response_vec, dtype=np.float32).reshape(-1)
                if rv.size > 0:
                    r_dim = int(rv.size)
                    r_blob = sqlite3.Binary(_to_blob(rv))

            try:
                with self._connect() as con:
                    con.execute(
                        """
                        INSERT INTO semantic_cache (
                            created_at, query_text, query_hash, route,
                            embedding_dim, query_embedding,
                            response_embedding_dim, response_embedding,
                            response_text
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            float(created_at),
                            str(query_text),
                            str(qh),
                            str(route),
                            int(dim),
                            sqlite3.Binary(_to_blob(query_vec)),
                            r_dim,
                            r_blob,
                            str(response_text),
                        ),
                    )

                    # Keep size bounded
                    con.execute(
                        """
                        DELETE FROM semantic_cache
                        WHERE id NOT IN (
                            SELECT id FROM semantic_cache
                            ORDER BY id DESC
                            LIMIT ?
                        )
                        """,
                        (self.max_entries,),
                    )
            except Exception as e:
                logger.warning(f"[cache] store failed: {e}")


_cache_instance: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance
