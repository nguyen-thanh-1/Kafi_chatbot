from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import List, Optional

import numpy as np

from src.utils.app_config import AppConfig
from src.utils.logger import logger
from src.utils.vram import delta_vram, format_vram, get_vram_snapshot


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray  # shape: (n, d), float32


class EmbeddingManager:
    def __init__(self):
        cfg = AppConfig.get_pipeline_config().get("embeddings", {}) or {}
        self.model_name = cfg.get("huggingface_model", "BAAI/bge-m3")
        self.normalize = bool(cfg.get("normalize", True))
        self.device = str(cfg.get("device") or ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        self._lock = Lock()
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> bool:
        if self.is_loaded:
            return True

        with self._lock:
            if self.is_loaded:
                return True

            before = get_vram_snapshot()
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                device = self.device
                if device.lower() == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                self._model = SentenceTransformer(self.model_name, device=device, trust_remote_code=True)
            except Exception as e:
                logger.warning(f"[embeddings] failed to load SentenceTransformer: {e}")
                self._model = None
                return False
            after = get_vram_snapshot()
            logger.info(
                f"[embeddings] loaded model; vram={format_vram(after)} (delta {delta_vram(before, after)})"
            )
            return True

    def embed(self, texts: List[str], batch_size: Optional[int] = None) -> EmbeddingResult:
        if not self.ensure_loaded():
            raise RuntimeError("Embedding model is not available")

        # Optimal batch sizes: GPU=128, CPU=32
        if batch_size is None:
            batch_size = 128 if 'cuda' in str(self._model.device) else 32

        vecs = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        return EmbeddingResult(vectors=vecs)


_embed_instance: Optional[EmbeddingManager] = None


def get_embedder() -> EmbeddingManager:
    global _embed_instance
    if _embed_instance is None:
        _embed_instance = EmbeddingManager()
    return _embed_instance
