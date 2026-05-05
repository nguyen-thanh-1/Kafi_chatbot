from __future__ import annotations

import os
import json
import uuid
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.utils.app_config import AppConfig, BASE_DIR
from src.utils.embeddings import get_embedder
from src.utils.reranker import get_reranker
from src.utils.logger import logger


# ═══════════════════════════════════════════════════════════
# EMBEDDING CACHE (from RAG Pro V2)
# ═══════════════════════════════════════════════════════════
class EmbeddingCache:
    """Cache embeddings to avoid re-embedding on subsequent runs."""

    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.cache: dict = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"[embedding_cache] loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"[embedding_cache] load failed: {e}")
                self.cache = {}

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
        logger.info(f"[embedding_cache] saved {len(self.cache)} embeddings")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        return self.cache.get(self._hash(text))

    def set(self, text: str, embedding: np.ndarray):
        self.cache[self._hash(text)] = embedding


# ═══════════════════════════════════════════════════════════
# SEMANTIC CHUNKING (from RAG Pro V2)
# ═══════════════════════════════════════════════════════════
def chunk_text_semantic(
    text: str,
    min_size: int = 400,
    max_size: int = 800,
    overlap: int = 50,
) -> List[str]:
    """
    Semantic chunking — split by natural paragraph boundaries.

    Strategy:
    1. Split by paragraphs (\\n\\n)
    2. Merge small paragraphs together
    3. Split paragraphs that are too large

    Result: Fewer chunks, better context quality.
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks: List[str] = []
    current_words: List[str] = []
    current_size = 0

    for para in paragraphs:
        words = para.split()
        para_size = len(words)

        # If a single paragraph exceeds max_size, split it
        if para_size > max_size:
            # Flush what we have
            if current_words:
                chunks.append(' '.join(current_words))
                current_words = []
                current_size = 0

            # Split the large paragraph with overlap
            for i in range(0, len(words), max_size - overlap):
                chunk_words = words[i:i + max_size]
                if len(chunk_words) >= min_size // 2:
                    chunks.append(' '.join(chunk_words))

        # If adding this paragraph would exceed max_size, flush first
        elif current_size + para_size > max_size:
            if current_words:
                chunks.append(' '.join(current_words))
            current_words = words
            current_size = para_size

        # Otherwise, keep accumulating
        else:
            current_words.extend(words)
            current_size += para_size

    # Flush remaining
    if current_words:
        chunks.append(' '.join(current_words))

    return chunks


# ═══════════════════════════════════════════════════════════
# KNOWLEDGE BASE (Qdrant + Semantic Chunking + Cache)
# ═══════════════════════════════════════════════════════════
class KnowledgeBase:
    def __init__(self):
        cfg = AppConfig.get_pipeline_config().get("knowledge", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.collection_name = cfg.get("collection_name", "kafi_knowledge")
        self.chunk_min = int(cfg.get("chunk_min", 400))
        self.chunk_max = int(cfg.get("chunk_max", 800))
        self.chunk_overlap = int(cfg.get("chunk_overlap", 50))
        self.top_k_retrieve = int(cfg.get("top_k_retrieve", 10))

        # Embedding cache (persistent on disk)
        cache_path = str(Path(BASE_DIR) / "data" / "embedding_cache.pkl")
        self.embedding_cache = EmbeddingCache(cache_path)

        # Initialize Qdrant Client (Local Storage)
        try:
            qdrant_path = cfg.get("qdrant_path", "data/qdrant_db")
            storage_path = Path(BASE_DIR) / qdrant_path
            storage_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(storage_path))
            logger.info(f"[knowledge] Qdrant local storage initialized at {storage_path}")
        except Exception as e:
            logger.error(f"[knowledge] failed to initialize Qdrant: {e}")
            self.client = None

    def _ensure_collection(self, vector_size: int):
        if not self.client:
            return

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(f"[knowledge] creating collection {self.collection_name} with dim {vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text("text") + "\n"
            return text
        except Exception as e:
            logger.error(f"[knowledge] PDF extraction failed: {e}")
            return ""

    def ingest_pdf(self, pdf_path: str):
        """Index a PDF file into Qdrant with semantic chunking and embedding cache."""
        if not self.client:
            logger.error("[knowledge] Qdrant client not initialized")
            return

        logger.info(f"[knowledge] ingesting PDF: {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return

        # Semantic chunking (from RAG Pro V2)
        chunks = chunk_text_semantic(
            text,
            min_size=self.chunk_min,
            max_size=self.chunk_max,
            overlap=self.chunk_overlap,
        )
        logger.info(f"[knowledge] semantic chunking produced {len(chunks)} chunks")

        if not chunks:
            return

        # Embed with cache
        embedder = get_embedder()
        cached_count = 0
        to_embed_texts = []
        to_embed_indices = []

        for i, chunk in enumerate(chunks):
            cached = self.embedding_cache.get(chunk)
            if cached is not None:
                cached_count += 1
            else:
                to_embed_texts.append(chunk)
                to_embed_indices.append(i)

        # Embed only the new ones
        if to_embed_texts:
            logger.info(f"[knowledge] embedding {len(to_embed_texts)} new chunks (cached: {cached_count})")
            new_embeddings = embedder.embed(to_embed_texts).vectors
            for text, emb in zip(to_embed_texts, new_embeddings):
                self.embedding_cache.set(text, emb)
        else:
            logger.info(f"[knowledge] all {len(chunks)} chunks served from cache!")

        # Build the full embedding array
        sample = self.embedding_cache.get(chunks[0])
        dim = sample.shape[0]
        all_embeddings = np.zeros((len(chunks), dim), dtype=np.float32)
        for i, chunk in enumerate(chunks):
            all_embeddings[i] = self.embedding_cache.get(chunk)

        # Save cache to disk
        self.embedding_cache.save()

        self._ensure_collection(dim)

        # Prepare and upsert points
        source_name = os.path.basename(pdf_path)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=all_embeddings[i].tolist(),
                payload={"text": chunks[i], "source": source_name},
            )
            for i in range(len(chunks))
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"[knowledge] ingested {len(chunks)} chunks into {self.collection_name}")

    def delete_file_vectors(self, filename: str):
        """Remove all vectors associated with a specific file."""
        if not self.client:
            return

        logger.info(f"[knowledge] deleting vectors for file: {filename}")
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=filename),
                            ),
                        ],
                    ),
                ),
            )
            logger.info(f"[knowledge] deleted vectors for: {filename}")
        except Exception as e:
            logger.error(f"[knowledge] delete failed: {e}")

    def retrieve(self, query: str) -> List[str]:
        """Search Qdrant and rerank results."""
        if not self.enabled or not self.client:
            return []

        try:
            embedder = get_embedder()
            query_vector = embedder.embed([query]).vectors[0].tolist()

            # 1. Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k_retrieve,
            )

            candidates = [hit.payload["text"] for hit in search_result if hit.payload]

            if not candidates:
                return []

            # 2. Rerank
            reranker = get_reranker()
            return reranker.rerank(query, candidates)
        except Exception as e:
            logger.error(f"[knowledge] retrieval failed: {e}")
            return []


_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
