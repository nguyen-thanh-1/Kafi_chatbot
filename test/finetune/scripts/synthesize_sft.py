from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên về tài chính bằng tiếng Việt. "
    "Bạn ưu tiên giải thích rõ ràng, có cấu trúc, nêu rủi ro và giả định. "
    "Bạn không bịa số liệu; nếu thiếu dữ liệu thì hỏi lại hoặc nói không đủ thông tin."
)


GEN_PROMPT_TEMPLATE = """Bạn sẽ tạo dữ liệu huấn luyện SFT cho LLM (tiếng Việt) về tài chính.
Dựa trên ĐOẠN TÀI LIỆU bên dưới, hãy tạo {k} cặp hỏi-đáp.

Yêu cầu:
- 60% câu hỏi dạng giải thích kiến thức (khái niệm, công thức, ví dụ).
- 40% câu hỏi dạng tư vấn giao dịch/risk (quản trị vốn, kịch bản, cảnh báo rủi ro).
- Câu trả lời phải bám vào nội dung đoạn tài liệu; nếu tài liệu không đủ, nói rõ hạn chế rồi đưa gợi ý chung an toàn.
- Trả về JSON (không markdown), dạng:
[
  {{"question": "...", "answer": "..."}},
  ...
]

ĐOẠN TÀI LIỆU:
{chunk}
"""


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
    # Fallback
    out = ""
    for m in messages:
        out += f"{m['role'].upper()}: {m['content']}\n"
    if add_generation_prompt:
        out += "ASSISTANT: "
    return out


def generate_pairs(model, tokenizer, chunk_text: str, k: int, max_new_tokens: int) -> List[Dict[str, str]]:
    user = GEN_PROMPT_TEMPLATE.format(k=k, chunk=chunk_text[:8000])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    prompt = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    # Parse JSON safely-ish
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return []

    pairs: List[Dict[str, str]] = []
    for item in data:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if q and a:
            pairs.append({"question": q, "answer": a})
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="chunks.jsonl")
    ap.add_argument("--out", required=True, help="Output sft.jsonl (messages)")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--examples-per-chunk", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    return run(
        chunks=args.chunks,
        out=args.out,
        base_model=args.base_model,
        examples_per_chunk=args.examples_per_chunk,
        max_chunks=args.max_chunks,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )


def run(
    *,
    chunks: str,
    out: str,
    base_model: str,
    examples_per_chunk: int,
    max_chunks: int,
    max_new_tokens: int,
    seed: int,
) -> int:
    random.seed(seed)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    written = 0
    chunks_seen = 0
    with open(chunks, "r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            chunk_text = row["text"]

            pairs = generate_pairs(
                model,
                tokenizer,
                chunk_text=chunk_text,
                k=examples_per_chunk,
                max_new_tokens=max_new_tokens,
            )

            for p in pairs:
                ex = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": p["question"]},
                        {"role": "assistant", "content": p["answer"]},
                    ],
                    "meta": {"source": row.get("source"), "chunk_id": row.get("chunk_id"), "synthetic": True},
                }
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                written += 1

            chunks_seen += 1
            if chunks_seen >= max_chunks:
                break

    print(f"Wrote {written} SFT examples to {out_path} (from {chunks_seen} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
