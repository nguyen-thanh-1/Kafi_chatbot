from __future__ import annotations

from enum import Enum
import re
from typing import Optional, Tuple

from src.utils.app_config import AppConfig
from src.utils.hf_textgen import HFTextGen
from src.utils.logger import logger
from src.utils.vram import delta_vram, format_vram, get_vram_snapshot


class SafetyDecision(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


GUARDRAILS_SYSTEM_PROMPT = "Bạn là guardrails. Chỉ trả về: SAFE hoặc UNSAFE."


class GuardrailsManager:
    def __init__(self):
        cfg = AppConfig.get_pipeline_config().get("guardrails", {}) or {}
        model_name = cfg.get("huggingface_model", "Qwen/Qwen3Guard-Gen-0.6B")
        quant = cfg.get("quantization", "4bit")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 6))
        self.temperature = float(cfg.get("temperature", 0.0))

        guard_cfg = AppConfig.get_guardrails_config()
        self.system_prompt = str(
            cfg.get("system_prompt")
            or guard_cfg.get("system_prompt")
            or GUARDRAILS_SYSTEM_PROMPT
        )

        self._model = HFTextGen(model_name, quantization=quant)

    def ensure_loaded(self) -> bool:
        before = get_vram_snapshot()
        ok = self._model.ensure_loaded()
        after = get_vram_snapshot()
        if ok:
            logger.info(
                f"[guardrails] loaded model; vram={format_vram(after)} (delta {delta_vram(before, after)})"
            )
        return ok

    def check(self, content: str) -> Tuple[SafetyDecision, str]:
        # Fast heuristic blocklist for common sensitive/credential requests (Vietnamese + English).
        # This prevents false negatives when the moderation model returns non-binary labels.
        lowered = (content or "").lower()
        heuristic_patterns = (
            r"\botp\b",
            r"mã\s*otp",
            r"mã\s*xác\s*thực",
            r"verification\s*code",
            r"2fa",
            r"one[-\s]*time\s*password",
            r"mật\s*khẩu",
            r"\bpassword\b",
            r"\bpasscode\b",
            r"tài\s*khoản.*mật\s*khẩu",
            r"token",
            r"session\s*token",
            r"api\s*key",
            r"secret\s*key",
            r"cookie",
            r"cccd",
            r"căn\s*cước",
            r"chứng\s*minh\s*nhân\s*dân",
            r"số\s*thẻ",
            r"credit\s*card",
            r"\bcvv\b",
            r"\bpin\b",
            r"hack",
            r"malware",
            r"lừa\s*đảo",
            r"scam",
            r"bypass",
        )
        if any(re.search(p, lowered, flags=re.IGNORECASE) for p in heuristic_patterns):
            return (SafetyDecision.UNSAFE, "heuristic_unsafe")

        if not self._model.is_loaded:
            try:
                self.ensure_loaded()
            except Exception:
                return (SafetyDecision.SAFE, "guardrails_not_loaded")

        try:
            res = self._model.generate_chat(
                system=self.system_prompt,
                user=content,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop_at_newline=True,
            )
            combined = ((res.text or "").strip() + "\n" + (res.raw or "")).strip()
            raw = combined.upper()

            # Prefer exact label parsing (avoid substring matches like "SAFETY").
            if re.search(r"\bUNSAFE\b", raw):
                return (SafetyDecision.UNSAFE, res.raw)
            # Some guard models output labels like "Safety: Safe"
            if re.search(r"\bSAFE\b", raw):
                return (SafetyDecision.SAFE, res.raw)

            # Fallback mapping for non-compliant outputs (e.g., "Safety: Controversial")
            unsafe_markers = (
                "SENSITIVE",
                "DANGEROUS",
                "ILLEGAL",
                "HATE",
                "HARASS",
                "SELF-HARM",
                "SELF HARM",
                "SUICIDE",
                "WEAPON",
                "EXPLOSIVE",
                "MALWARE",
                "HACK",
                "SCAM",
                "FRAUD",
                "JAILBREAK",
                "INJECTION",
                "PRIVACY",
                "CREDENTIAL",
                "PASSWORD",
                "OTP",
            )
            if any(m in raw for m in unsafe_markers):
                return (SafetyDecision.UNSAFE, res.raw)

            return (SafetyDecision.SAFE, res.raw)
        except Exception as e:
            return (SafetyDecision.SAFE, f"guardrails_error: {e}")


_guardrails_instance: Optional[GuardrailsManager] = None


def get_guardrails() -> GuardrailsManager:
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = GuardrailsManager()
    return _guardrails_instance
