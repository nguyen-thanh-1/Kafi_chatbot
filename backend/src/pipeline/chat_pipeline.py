from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Generator, List, Optional

from src.agents.financial_agent import get_financial_agent
from src.utils.app_config import AppConfig
from src.utils.embeddings import get_embedder
from src.utils.guardrails import SafetyDecision, get_guardrails
from src.utils.logger import logger
from src.utils.router import Routes, get_router
from src.utils.semantic_cache import CacheHit, get_cache


@dataclass(frozen=True)
class PipelineTrace:
    input_safety: str
    cache_hit: bool
    cache_similarity: float
    route: str
    output_safety: str


class ChatPipeline:
    def __init__(self):
        self._lock = Lock()
        self._last_trace: Optional[PipelineTrace] = None

        self._cfg = AppConfig.get_pipeline_config() or {}
        self._startup_cfg = (self._cfg.get("startup", {}) or {})

        self.guardrails = get_guardrails()
        self.router = get_router()
        self.embedder = get_embedder()
        self.cache = get_cache()

    def get_last_trace(self) -> Optional[PipelineTrace]:
        with self._lock:
            return self._last_trace

    def warmup(self) -> None:
        if bool(self._startup_cfg.get("preload_guardrails", True)):
            try:
                self.guardrails.ensure_loaded()
            except Exception as e:
                logger.warning(f"[startup] guardrails preload failed: {e}")
        if bool(self._startup_cfg.get("preload_router", True)):
            try:
                self.router.ensure_loaded()
            except Exception as e:
                logger.warning(f"[startup] router preload failed: {e}")
        if bool(self._startup_cfg.get("preload_embeddings", True)):
            try:
                self.embedder.ensure_loaded()
            except Exception as e:
                logger.warning(f"[startup] embeddings preload failed: {e}")

    def _stream_with_periodic_output_guardrails(
        self,
        *,
        chunk_iter: Generator[str, None, None],
        check_every_chars: int = 300,
    ) -> Generator[str, None, str]:
        """
        Stream chunks to client while buffering full response.
        Periodically checks output safety; if UNSAFE, append a stop notice and stop streaming.
        Returns the final buffered response.
        """
        response_text = ""
        next_check_at = check_every_chars

        for chunk in chunk_iter:
            response_text += chunk
            yield chunk

            if len(response_text) >= next_check_at:
                decision_mid, raw_mid = self.guardrails.check(response_text)
                if decision_mid == SafetyDecision.UNSAFE:
                    logger.info(f"[guardrails] output=UNSAFE mid raw={raw_mid.strip()[:120]}")
                    tail = (
                        "\n\n[Thông báo] Mình dừng trả lời vì nội dung có thể không an toàn. "
                        "Bạn có thể nêu mục tiêu hợp lệ/an toàn để mình hỗ trợ lại."
                    )
                    response_text += tail
                    yield tail
                    break
                next_check_at += check_every_chars

        return response_text

    def _history_context_hash(self, history: List[Dict[str, str]]) -> str:
        context = [
            {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
            for message in history
        ]
        payload = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()

    def process(self, user_text: str, history: List[Dict[str, str]], session_id: Optional[str] = None) -> Generator[str, None, None]:
        context_hash = self._history_context_hash(history)

        # (1) Input guardrails
        decision_in, raw_in = self.guardrails.check(user_text)
        logger.info(f"[guardrails] input={decision_in.value} raw={raw_in.strip()[:120]}")
        if decision_in == SafetyDecision.UNSAFE:
            # Do not forward unsafe user content; ask LLM to produce a safe refusal (Vietnamese).
            agent = get_financial_agent()
            unsafe_prompt = (
                "[UNSAFE_INPUT]\n"
                "The user's content is likely unsafe/invalid.\n"
                "Task: refuse politely in Vietnamese, do NOT repeat the user's content, "
                "and guide them to provide a safe/valid request."
            )

            gen = agent.process_chat(unsafe_prompt, history)
            response_text = ""
            for chunk in gen:
                response_text += chunk
                yield chunk

            # Output guardrails (should be SAFE)
            decision_out, raw_out = self.guardrails.check(response_text)
            logger.info(f"[guardrails] output={decision_out.value} raw={raw_out.strip()[:120]}")

            trace = PipelineTrace(
                input_safety="UNSAFE",
                cache_hit=False,
                cache_similarity=0.0,
                route=Routes.GENERAL.value,
                output_safety=decision_out.value,
            )
            with self._lock:
                self._last_trace = trace
            return

        # (2) Cache check (semantic)
        query_vec = None
        cache_hit: Optional[CacheHit] = None
        if self.cache.enabled:
            try:
                query_vec = self.embedder.embed([user_text]).vectors[0]
                cache_hit = self.cache.lookup(
                    user_text,
                    query_vec,
                    session_id=session_id,
                    context_hash=context_hash,
                )
            except Exception as e:
                logger.warning(f"[cache] embedding/lookup failed: {e}")

        if cache_hit is not None:
            logger.info(
                f"[cache] HIT similarity={cache_hit.similarity:.3f} route={cache_hit.route} age_s={int(time.time() - cache_hit.created_at)}"
            )

            # Cache-hit path: user prompt + cached reference answer -> LLM
            agent = get_financial_agent()
            augmented_prompt = (
                "[CACHE_HIT]\n"
                "Bạn đã từng trả lời câu hỏi tương tự trước đây.\n"
                "Dưới đây là câu trả lời tham chiếu (không cần trích nguyên văn, hãy diễn đạt tự nhiên hơn):\n"
                f"{cache_hit.response}\n\n"
                f"Câu hỏi của người dùng: {user_text}"
            )
            
            gen = agent.process_chat(augmented_prompt, history)
            response_text = yield from self._stream_with_periodic_output_guardrails(chunk_iter=gen)

            decision_out, raw_out = self.guardrails.check(response_text)
            logger.info(f"[guardrails] output={decision_out.value} raw={raw_out.strip()[:120]}")
            if decision_out == SafetyDecision.UNSAFE:
                tail = (
                    "\n\n[Thông báo] Mình không thể tiếp tục vì nội dung có thể không an toàn. "
                    "Bạn có thể cho mình biết mục tiêu hợp lệ/an toàn để mình hỗ trợ theo cách khác."
                )
                yield tail
                response_text += tail

            trace = PipelineTrace(
                input_safety=decision_in.value,
                cache_hit=True,
                cache_similarity=float(cache_hit.similarity),
                route=str(cache_hit.route),
                output_safety=decision_out.value,
            )
            with self._lock:
                self._last_trace = trace
            return

        logger.info("[cache] MISS")

        # (3) Router
        route, router_raw = self.router.classify(user_text)
        logger.info(f"[router] route={route.value} raw={router_raw.strip()[:120]}")

        # (4) Call tools/models (RAG not wired yet -> all routes go to LLM for now)
        agent = get_financial_agent()
        gen = agent.process_chat(user_text, history)
        response_text = yield from self._stream_with_periodic_output_guardrails(chunk_iter=gen)

        # (5) Output guardrails
        decision_out, raw_out = self.guardrails.check(response_text)
        logger.info(f"[guardrails] output={decision_out.value} raw={raw_out.strip()[:120]}")
        if decision_out == SafetyDecision.UNSAFE:
            tail = (
                "\n\n[Thông báo] Mình không thể tiếp tục vì nội dung có thể không an toàn. "
                "Bạn có thể cho mình biết mục tiêu hợp lệ/an toàn để mình hỗ trợ theo cách khác."
            )
            yield tail
            response_text += tail

        # (6) Save cache (skip duplicates by response embedding)
        if self.cache.enabled and decision_out == SafetyDecision.SAFE:
            try:
                if query_vec is None:
                    query_vec = self.embedder.embed([user_text]).vectors[0]

                response_vec = self.embedder.embed([response_text]).vectors[0]
                dup = self.cache.has_similar_response(
                    response_vec,
                    session_id=session_id,
                    context_hash=context_hash,
                )
                if dup is not None:
                    logger.info(f"[cache] DUPLICATE_RESPONSE similarity={dup[0]:.3f} (skip store)")
                else:
                    self.cache.store(
                        user_text,
                        query_vec,
                        route.value,
                        response_text,
                        session_id=session_id,
                        context_hash=context_hash,
                        response_vec=response_vec,
                    )
                    logger.info("[cache] STORED")
            except Exception as e:
                logger.warning(f"[cache] store failed: {e}")

        trace = PipelineTrace(
            input_safety=decision_in.value,
            cache_hit=False,
            cache_similarity=0.0,
            route=route.value,
            output_safety=decision_out.value,
        )
        with self._lock:
            self._last_trace = trace


_pipeline_instance: Optional[ChatPipeline] = None


def get_chat_pipeline() -> ChatPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = ChatPipeline()
    return _pipeline_instance

