"""
Eval script cho rag_llama.py
- Load trực tiếp RAG pipeline (FAISS, Embedding, Reranker, LLaMA 3.1-8B)
- Dùng test_dataset.json làm tập đánh giá
- Đẩy kết quả lên Opik project: "rag_llama"
- Chạy bằng: uv run python rag_systems/optimize_rag/eval_rag_llama.py
- Metrics: ROUGE-L, Levenshtein Ratio, Exact-Match, Custom Context Faithfulness
"""

import os
import sys
import json
import time
import logging

# ── suppress noisy logs ──────────────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# ── std / third-party ────────────────────────────────────────────────────────
import faiss
import fitz  # PyMuPDF
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

# ── local LLM path ───────────────────────────────────────────────────────────
LLM_DIR = r"C:\Users\Admin\Desktop\Special_Subject_AI\llm_models"
sys.path.insert(0, LLM_DIR)
from Llama_3_1_8B_Instruct_v2 import generate_response, _load_model

# ── Opik ─────────────────────────────────────────────────────────────────────
import opik
from opik import track, opik_context
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    LevenshteinRatio,
    ROUGE,
    BaseMetric,
)
from opik.evaluation.metrics.score_result import ScoreResult

# ── Gemini Judge ─────────────────────────────────────────────────────────────
import google.generativeai as genai

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_NAME    = "rag_llama_v2"
OPIK_WORKSPACE  = "chatbot-rag"
DATASET_NAME    = "NLP-Book-QA-Dataset"

PDF_PATH        = r"C:\Users\Admin\Desktop\Special_Subject_AI\data\courses_v2\nlp-book.pdf"
INDEX_PATH      = r"C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\optimize_rag\faiss.index"
META_PATH       = r"C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\optimize_rag\chunks.json"
TEST_DATASET    = r"C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\optimize_rag\test_dataset.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL  = "BAAI/bge-reranker-base"

CHUNK_SIZE      = 400
OVERLAP         = 50
BATCH_SIZE      = 64
TOP_K_RETRIEVE  = 8
TOP_K_RERANK    = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE — sao chép logic từ rag_llama.py (không import FastAPI server)
# ══════════════════════════════════════════════════════════════════════════════

# Globals
_tokenizer_embed = None
_embed_model     = None
_reranker        = None
_index           = None
_chunks          = []
_query_cache     = {}

def _load_rag_components():
    global _tokenizer_embed, _embed_model, _reranker, _index, _chunks

    print(f"\n{'='*60}")
    print("  Loading RAG components for rag_llama …")
    print(f"{'='*60}")

    print("  [1/4] Loading embedding tokenizer …")
    _tokenizer_embed = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    print("  [2/4] Loading SentenceTransformer embedding model …")
    _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("  [3/4] Loading CrossEncoder reranker …")
    _reranker = CrossEncoder(RERANKER_MODEL, device=device)

    print("  [4/4] Loading LLaMA 3.1-8B-Instruct (4-bit) …")
    _load_model()   # force-load global model

    # ── Load or build FAISS index ─────────────────────────────────────────
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        print("  Loading existing FAISS index …")
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _chunks = json.load(f)
        print(f"  Index loaded: {len(_chunks)} chunks.")
    else:
        print("  No FAISS index found — building from PDF …")
        _build_index_from_pdf()

    print("  RAG components ready!\n")


def _extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)


def _chunk_text(text: str):
    tokens = _tokenizer_embed.encode(text, add_special_tokens=False, truncation=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunks.append(_tokenizer_embed.decode(tokens[start:end]))
        start += CHUNK_SIZE - OVERLAP
    return chunks


def _embed_chunks(chunks):
    return _embed_model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def _build_faiss_index(embeddings):
    dim   = embeddings.shape[1]
    nlist = min(100, max(1, len(embeddings) // 39))
    quantizer = faiss.IndexFlatIP(dim)
    idx = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    idx.train(embeddings)
    idx.add(embeddings)
    idx.nprobe = min(10, nlist)
    return idx


def _build_index_from_pdf():
    global _index, _chunks
    text   = _extract_text_from_pdf(PDF_PATH)
    _chunks = _chunk_text(text)
    embs   = _embed_chunks(_chunks)
    _index = _build_faiss_index(embs)
    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_chunks, f)
    print(f"  Built & saved index with {len(_chunks)} chunks.")


def _rerank(query: str, candidates: list) -> list:
    pairs  = [[query, c] for c in candidates]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked[:TOP_K_RERANK]]


def _retrieve(query: str) -> list:
    if not _index or not _chunks:
        return []
    if query in _query_cache:
        return _query_cache[query]
    q_emb = _embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, I  = _index.search(q_emb, TOP_K_RETRIEVE)
    candidates = [_chunks[i] for i in I[0] if i < len(_chunks)]
    contexts   = _rerank(query, candidates)
    _query_cache[query] = contexts
    return contexts


def _generate_answer(query: str, contexts: list) -> str:
    context_text = "\n\n".join(contexts)
    system_prompt = "You are an expert academic assistant. Use ONLY the provided context to answer the question."
    user_input    = f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"
    response = generate_response(
        user_input=user_input,
        system_prompt=system_prompt,
        max_new_tokens=512,
        temperature=0.2,
    )
    return response.strip()


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM GEMINI METRICS
# ══════════════════════════════════════════════════════════════════════════════

class GeminiJudgeMetric(BaseMetric):
    """Base class for Gemini-based metrics."""
    def __init__(self, name: str, model_name: str = "gemini-2.5-flash"):
        super().__init__(name=name)
        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def _query_gemini(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            return response.text
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return "{}"

class HallucinationMetric(GeminiJudgeMetric):
    """Chấm điểm Hallucination: Câu trả lời có bịa đặt thông tin không? (1.0 = Không bịa đặt, 0.0 = Bịa đặt hoàn toàn)"""
    def __init__(self):
        super().__init__(name="hallucination_metric")

    def score(self, output: str, context: list, **kwargs) -> ScoreResult:
        context_str = "\n".join(context)
        prompt = f"""
        Bạn là một chuyên gia đánh giá hệ thống RAG. 
        Hãy đánh giá mức độ ẢO GIÁC (Hallucination) của câu trả lời so với ngữ cảnh được cung cấp.
        Câu trả lời có dựa hoàn toàn trên ngữ cảnh không? Có thông tin nào KHÔNG có trong ngữ cảnh nhưng lại xuất hiện trong câu trả lời không?

        Ngữ cảnh: {context_str}
        Câu trả lời: {output}

        Trả về kết quả dưới định dạng JSON:
        {{
            "score": <giá trị từ 0.0 đến 1.0, trong đó 1.0 là TRUNG THỰC HOÀN TOÀN (Không có ảo giác) và 0.0 là BỊA ĐẶT HOÀN TOÀN>,
            "reason": "<giải thích ngắn gọn bằng tiếng Việt>"
        }}
        """
        res_text = self._query_gemini(prompt)
        try:
            res = json.loads(res_text)
            return ScoreResult(name=self.name, value=res.get("score", 0.0), reason=res.get("reason", ""))
        except:
            return ScoreResult(name=self.name, value=0.0, reason="Failed to parse Gemini response")

class AnswerRelevanceMetric(GeminiJudgeMetric):
    """Chấm điểm Relevance: Câu trả lời có trả lời đúng trọng tâm câu hỏi không?"""
    def __init__(self):
        super().__init__(name="answer_relevance_metric")

    def score(self, input: str, output: str, **kwargs) -> ScoreResult:
        prompt = f"""
        Bạn là một chuyên gia đánh giá hệ thống RAG.
        Hãy đánh giá độ LIÊN QUAN (Answer Relevance) của câu trả lời đối with câu hỏi.
        Câu trả lời có giải quyết được đúng và đủ vấn đề người dùng hỏi không?

        Câu hỏi: {input}
        Câu trả lời: {output}

        Trả về kết quả dưới định dạng JSON:
        {{
            "score": <giá trị từ 0.0 đến 1.0, trong đó 1.0 là HOÀN TOÀN LIÊN QUAN và 0.0 là KHÔNG LIÊN QUAN>,
            "reason": "<giải thích ngắn gọn bằng tiếng Việt>"
        }}
        """
        res_text = self._query_gemini(prompt)
        try:
            res = json.loads(res_text)
            return ScoreResult(name=self.name, value=res.get("score", 0.0), reason=res.get("reason", ""))
        except:
            return ScoreResult(name=self.name, value=0.0, reason="Failed to parse Gemini response")


# ══════════════════════════════════════════════════════════════════════════════
# OPIK — evaluation task
# ══════════════════════════════════════════════════════════════════════════════

@track(project_name=PROJECT_NAME)
def rag_llama_task(dataset_item: dict) -> dict:
    """
    Hàm này được Opik gọi cho từng item trong dataset.
    Trả về dict với các key phù hợp với metrics.
    """
    question        = dataset_item["question"]
    expected_output = dataset_item["expected_output"]

    # ── Retrieve ──────────────────────────────────────────────────────────
    t0       = time.perf_counter()
    contexts = _retrieve(question)
    t_ret    = time.perf_counter() - t0

    # ── Generate ──────────────────────────────────────────────────────────
    t1     = time.perf_counter()
    answer = _generate_answer(question, contexts)
    t_gen  = time.perf_counter() - t1

    # ── Log timing metadata vào Opik span ─────────────────────────────────
    opik_context.update_current_trace(
        metadata={
            "retrieval_time_s": round(t_ret, 3),
            "generation_time_s": round(t_gen, 3),
            "num_contexts": len(contexts),
            "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
        }
    )

    return {
        "input":           question,
        "output":          answer,
        "expected_output": expected_output,
        "context":         contexts,
        "reference":       expected_output,  # alias cho một số metrics
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Configure Opik ─────────────────────────────────────────────────
    opik.configure(
        api_key=os.environ.get("OPIK_API_KEY", ""),
        workspace=OPIK_WORKSPACE,
        use_local=False,
    )
    client = opik.Opik(workspace=OPIK_WORKSPACE)

    # ── 2. Load / create dataset on Opik ─────────────────────────────────
    print(f"\nPreparing Opik dataset: '{DATASET_NAME}' …")
    dataset = client.get_or_create_dataset(name=DATASET_NAME)

    with open(TEST_DATASET, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    dataset.insert(raw_items)   # Opik deduplicates automatically
    print(f"Dataset ready — {len(raw_items)} items (deduplicated by Opik).")

    # ── 3. Load RAG pipeline ──────────────────────────────────────────────
    _load_rag_components()

    # ── 4. Define metrics (no external LLM needed) ────────────────────────
    metrics = [
        LevenshteinRatio(),
        ROUGE(rouge_type="rougeL"),
        HallucinationMetric(),
        AnswerRelevanceMetric(),
    ]

    # ── 5. Run evaluation ─────────────────────────────────────────────────
    print(f"\nStarting evaluation on project '{PROJECT_NAME}' …")
    print(f"Items to evaluate: {len(raw_items)}")
    print("Metrics: LevenshteinRatio, ROUGE-L\n")

    experiment_result = evaluate(
        dataset=dataset,
        task=rag_llama_task,
        scoring_metrics=metrics,
        experiment_name=f"rag_llama_eval_{time.strftime('%Y%m%d_%H%M%S')}",
        project_name=PROJECT_NAME,
        experiment_config={
            "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "quantization": "4-bit NF4 (bitsandbytes)",
            "embedding_model": EMBEDDING_MODEL,
            "reranker_model": RERANKER_MODEL,
            "chunk_size": CHUNK_SIZE,
            "overlap": OVERLAP,
            "top_k_retrieve": TOP_K_RETRIEVE,
            "top_k_rerank": TOP_K_RERANK,
            "dataset": TEST_DATASET,
        },
        nb_samples=len(raw_items),
    )

    # ── 6. Print summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Evaluation complete! Project: {PROJECT_NAME}")
    print(f"  Opik dashboard: https://www.comet.com/opik/{OPIK_WORKSPACE}/projects")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
