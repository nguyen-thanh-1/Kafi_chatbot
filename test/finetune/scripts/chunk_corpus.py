from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils_text import chunk_by_tokens


def run(
    *,
    corpus: str,
    out: str,
    tokenizer: str,
    chunk_tokens: int,
    overlap_tokens: int,
    min_chars: int,
) -> int:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_id = 0
    with open(corpus, "r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row["doc_id"]
            source = row["source"]
            text = row["text"]

            chunks = chunk_by_tokens(
                text,
                tokenizer_name_or_path=tokenizer,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )

            for idx, ch in enumerate(chunks):
                if len(ch) < min_chars:
                    continue
                fout.write(
                    json.dumps(
                        {
                            "chunk_id": f"{doc_id}:{idx}",
                            "doc_id": doc_id,
                            "source": source,
                            "text": ch,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                chunk_id += 1

    print(f"Wrote {chunk_id} chunks to {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="JSONL from build_corpus.py")
    ap.add_argument("--out", required=True, help="Output JSONL chunks path")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="Tokenizer name/path for token-based chunking")
    ap.add_argument("--chunk-tokens", type=int, default=900)
    ap.add_argument("--overlap-tokens", type=int, default=120)
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()
    return run(
        corpus=args.corpus,
        out=args.out,
        tokenizer=args.tokenizer,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chars=args.min_chars,
    )


if __name__ == "__main__":
    raise SystemExit(main())
