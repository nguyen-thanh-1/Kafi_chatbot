from __future__ import annotations

from threading import Lock
from typing import List, Optional, Tuple

from src.utils.app_config import AppConfig
from src.utils.logger import logger
from src.utils.vram import delta_vram, format_vram, get_vram_snapshot


class RerankerManager:
    def __init__(self):
        cfg = AppConfig.get_pipeline_config().get("reranker", {}) or {}
        self.model_name = cfg.get("huggingface_model", "Qwen/Qwen3-Reranker-0.6B")
        self.device = str(cfg.get("device") or ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
        self.top_k = int(cfg.get("top_k", 3))
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
                from sentence_transformers import CrossEncoder
                import torch

                device = self.device
                if device.lower() == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                
                logger.info(f"[reranker] loading model {self.model_name} on {device}...")
                self._model = CrossEncoder(self.model_name, device=device)
            except Exception as e:
                logger.warning(f"[reranker] failed to load CrossEncoder: {e}")
                self._model = None
                return False
            
            after = get_vram_snapshot()
            logger.info(
                f"[reranker] loaded model; vram={format_vram(after)} (delta {delta_vram(before, after)})"
            )
            return True

    def rerank(self, query: str, candidates: List[str]) -> List[str]:
        """
        Rerank a list of candidate strings against a query.
        Returns the top-k candidates.
        """
        if not candidates:
            return []
            
        if not self.ensure_loaded():
            logger.warning("[reranker] model not loaded, returning original candidates order")
            return candidates[:self.top_k]

        try:
            pairs = [[query, c] for c in candidates]
            scores = self._model.predict(pairs)
            
            # Sort by score descending
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [x[0] for x in ranked[:self.top_k]]
        except Exception as e:
            logger.error(f"[reranker] prediction failed: {e}")
            return candidates[:self.top_k]


_rerank_instance: Optional[RerankerManager] = None


def get_reranker() -> RerankerManager:
    global _rerank_instance
    if _rerank_instance is None:
        _rerank_instance = RerankerManager()
    return _rerank_instance
