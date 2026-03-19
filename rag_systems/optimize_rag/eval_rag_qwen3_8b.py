"""
Eval script cho rag_qwen3_8b.py
- Load trực tiếp RAG pipeline (FAISS, Embedding, Reranker, Qwen3-8B)
- Dùng test_dataset.json làm tập đánh giá
- Đẩy kết quả lên Opik project: "rag_qwen3_8b"
- Chạy bằng: uv run python rag_systems/optimize_rag/eval_rag_qwen3_8b.py
- Metrics: ROUGE-L, Levenshtein Ratio
"""

import os
import sys
import json
import time
import logging
from threading import Thread

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
from transformers import AutoTokenizer, TextIteratorStreamer

# ── local LLM path ───────────────────────────────────────────────────────────
LLM_DIR = r"C:\Users\Admin\Desktop\Special_Subject_AI\llm_models"
sys.path.insert(0, LLM_DIR)
from Qwen3_8B import QwenChat

# ── Opik ─────────────────────────────────────────────────────────────────────
import opik
from opik import track, opik_context
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    LevenshteinRatio,
    ROUGE,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
PROJECT_NAME    = "rag_qwen3_8b"
OPIK_WORKSPACE  = "chatbot-rag"
DATASET_NAME    = "NLP-Book-QA-Dataset"

PDF_PATH        = r"C:\Users\Admin\Desktop\Special_Subject_AI\data\courses_v2\nlp-book.pdf"
INDEX_PATH      = r"C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\optimize_rag\faiss_qwen3_8b.index"
META_PATH       = r"C:\Users\Admin\Desktop\Special_Subject_AI\rag_systems\optimize_rag\chunks_qwen3_8b.json"
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
# RAG PIPELINE — sao chép logic từ rag_qwen3_8b.py (không import FastAPI server)
# ══════════════════════════════════════════════════════════════════════════════

_tokenizer_embed = None
_embed_model     = None
_reranker        = None
_index           = None
_chunks          = []
_query_cache     = {}
_qwen: QwenChat  = None   # type: ignore


def _load_rag_components():
    global _tokenizer_embed, _embed_model, _reranker, _index, _chunks, _qwen

    print(f"\n{'='*60}")
    print("  Loading RAG components for rag_qwen3_8b …")
    print(f"{'='*60}")

    print("  [1/4] Loading embedding tokenizer …")
    _tokenizer_embed = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

    print("  [2/4] Loading SentenceTransformer embedding model …")
    _embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print("  [3/4] Loading CrossEncoder reranker …")
    _reranker = CrossEncoder(RERANKER_MODEL, device=device)

    print("  [4/4] Loading Qwen3-8B (4-bit, Non-Thinking mode) …")
    _qwen = QwenChat(enable_thinking=False)

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
    text    = _extract_text_from_pdf(PDF_PATH)
    _chunks = _chunk_text(text)
    embs    = _embed_chunks(_chunks)
    _index  = _build_faiss_index(embs)
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


def _generate_answer_qwen(query: str, contexts: list) -> str:
    """
    Reproduce chính xác logic generate_answer() từ rag_qwen3_8b.py:
    - System prompt học thuật, strict context-only
    - Greedy decoding (do_sample=False) để sát kết quả RAG thật
    - reset_chat() cho mỗi câu hỏi để tránh cross-contamination
    """
    context_text = "\n\n".join(contexts)

    _qwen.system_prompt = (
        "You are an expert academic assistant. "
        "Use ONLY the provided context to answer the question. "
        "If the context does not contain relevant information, "
        "explicitly say you cannot find it in the provided documents. "
        "Do NOT use any external knowledge outside the context."
    )
    _qwen.reset_chat()

    user_input = f"Context:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"

    # ── Greedy generation (same as rag_qwen3_8b.py) ───────────────────────
    _qwen.history.append({"role": "user", "content": user_input})

    text = _qwen.tokenizer.apply_chat_template(
        _qwen.history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = _qwen.tokenizer([text], return_tensors="pt").to(_qwen.model.device)
    streamer = TextIteratorStreamer(
        _qwen.tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=1.1,
        pad_token_id=_qwen.tokenizer.pad_token_id or _qwen.tokenizer.eos_token_id,
        eos_token_id=_qwen.tokenizer.eos_token_id,
    )
    t = Thread(target=_qwen.model.generate, kwargs=generation_kwargs)
    t.start()
    full_response = ""
    for new_text in streamer:
        full_response += new_text
    t.join()

    _qwen.history.append({"role": "assistant", "content": full_response})
    return full_response.strip()


# ══════════════════════════════════════════════════════════════════════════════
# OPIK — evaluation task
# ══════════════════════════════════════════════════════════════════════════════

@track(project_name=PROJECT_NAME)
def rag_qwen3_task(dataset_item: dict) -> dict:
    """
    Hàm này được Opik gọi cho từng item trong dataset.
    """
    question        = dataset_item["question"]
    expected_output = dataset_item["expected_output"]

    # ── Retrieve ──────────────────────────────────────────────────────────
    t0       = time.perf_counter()
    contexts = _retrieve(question)
    t_ret    = time.perf_counter() - t0

    # ── Generate ──────────────────────────────────────────────────────────
    t1     = time.perf_counter()
    answer = _generate_answer_qwen(question, contexts)
    t_gen  = time.perf_counter() - t1

    # ── Log timing metadata vào Opik span ─────────────────────────────────
    opik_context.update_current_trace(
        metadata={
            "retrieval_time_s":  round(t_ret, 3),
            "generation_time_s": round(t_gen, 3),
            "num_contexts":      len(contexts),
            "llm_model":         "Qwen/Qwen3-8B",
            "embedding_model":   EMBEDDING_MODEL,
            "reranker_model":    RERANKER_MODEL,
        }
    )

    return {
        "input":           question,
        "output":          answer,
        "expected_output": expected_output,
        "context":         contexts,
        "reference":       expected_output,
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

    dataset.insert(raw_items)
    print(f"Dataset ready — {len(raw_items)} items (deduplicated by Opik).")

    # ── 3. Load RAG pipeline ──────────────────────────────────────────────
    _load_rag_components()

    # ── 4. Define metrics (no external LLM needed) ────────────────────────
    metrics = [
        LevenshteinRatio(),
        ROUGE(rouge_type="rougeL"),
    ]

    # ── 5. Run evaluation ─────────────────────────────────────────────────
    print(f"\nStarting evaluation on project '{PROJECT_NAME}' …")
    print(f"Items to evaluate: {len(raw_items)}")
    print("Metrics: LevenshteinRatio, ROUGE-L\n")

    experiment_result = evaluate(
        dataset=dataset,
        task=rag_qwen3_task,
        scoring_metrics=metrics,
        experiment_name=f"rag_qwen3_8b_eval_{time.strftime('%Y%m%d_%H%M%S')}",
        project_name=PROJECT_NAME,
        experiment_config={
            "llm_model": "Qwen/Qwen3-8B",
            "quantization": "4-bit NF4 (bitsandbytes)",
            "thinking_mode": False,
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
