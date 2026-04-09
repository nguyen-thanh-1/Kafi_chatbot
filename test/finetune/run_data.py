from __future__ import annotations

import argparse
import sys
from pathlib import Path


FINETUNE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = FINETUNE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_corpus import run as build_corpus_run  # noqa: E402
from chunk_corpus import run as chunk_corpus_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(FINETUNE_DIR / "data" / "raw"), help="Raw docs dir")
    ap.add_argument("--input-file", default=None, help="Optional single file to process")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B", help="Tokenizer for chunking")
    ap.add_argument("--chunk-tokens", type=int, default=900)
    ap.add_argument("--overlap-tokens", type=int, default=120)
    ap.add_argument("--min-chars", type=int, default=400)
    args = ap.parse_args()

    processed_dir = FINETUNE_DIR / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = processed_dir / "corpus.jsonl"
    chunks_path = processed_dir / "chunks.jsonl"

    build_corpus_run(
        args.raw,
        str(corpus_path),
        input_file=args.input_file,
    )
    chunk_corpus_run(
        corpus=str(corpus_path),
        out=str(chunks_path),
        tokenizer=args.tokenizer,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        min_chars=args.min_chars,
    )

    print(f"OK: {chunks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
