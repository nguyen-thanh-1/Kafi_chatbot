from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from src.utils.app_config import AppConfig
from src.utils.hf_textgen import HFTextGen
from src.utils.logger import logger
from src.utils.vram import delta_vram, format_vram, get_vram_snapshot


class Routes(str, Enum):
    FINANCIAL = "FINANCIAL"
    KNOWLEDGE = "KNOWLEDGE"
    GENERAL = "GENERAL"


ROUTER_SYSTEM_PROMPT = "Bạn là một router. Chỉ trả về 1 từ: FINANCIAL, KNOWLEDGE, GENERAL."


class RouterManager:
    def __init__(self):
        cfg = AppConfig.get_pipeline_config().get("router", {}) or {}
        model_name = cfg.get("huggingface_model", "Qwen/Qwen3-4B-Instruct-2507")
        quant = cfg.get("quantization", "4bit")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 8))
        self.temperature = float(cfg.get("temperature", 0.0))

        router_cfg = AppConfig.get_router_config()
        self.system_prompt = str(
            cfg.get("system_prompt")
            or router_cfg.get("system_prompt")
            or ROUTER_SYSTEM_PROMPT
        )

        self._model = HFTextGen(model_name, quantization=quant)

    def ensure_loaded(self) -> bool:
        before = get_vram_snapshot()
        ok = self._model.ensure_loaded()
        after = get_vram_snapshot()
        if ok:
            logger.info(
                f"[router] loaded model; vram={format_vram(after)} (delta {delta_vram(before, after)})"
            )
        return ok

    def classify(self, user_text: str) -> Tuple[Routes, str]:
        if not self._model.is_loaded:
            try:
                self.ensure_loaded()
            except Exception:
                return (Routes.GENERAL, "router_not_loaded")

        try:
            res = self._model.generate_chat(
                system=self.system_prompt,
                user=user_text,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop_at_newline=True,
            )
            raw = (res.text or "").strip().upper()
            if raw in (r.value for r in Routes):
                return (Routes(raw), res.raw)
            # Be tolerant: pick the first known token found
            for r in Routes:
                if r.value in raw:
                    return (r, res.raw)
            return (Routes.GENERAL, res.raw)
        except Exception as e:
            return (Routes.GENERAL, f"router_error: {e}")


_router_instance: Optional[RouterManager] = None


def get_router() -> RouterManager:
    global _router_instance
    if _router_instance is None:
        _router_instance = RouterManager()
    return _router_instance
