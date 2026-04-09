from __future__ import annotations

import argparse
import sys
from pathlib import Path


FINETUNE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = FINETUNE_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from finetune_qlora import run as finetune_run  # noqa: E402
from synthesize_sft import run as synthesize_sft_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(FINETUNE_DIR / "data" / "processed" / "sft.jsonl"), help="SFT JSONL output")
    ap.add_argument("--chunks", default=str(FINETUNE_DIR / "data" / "processed" / "chunks.jsonl"), help="Chunks JSONL input")
    ap.add_argument("--synth-model", default="Qwen/Qwen3-8B", help="Model used to synthesize SFT from chunks")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B", help="Base model to finetune (QLoRA)")
    ap.add_argument("--output-dir", default=str(FINETUNE_DIR / "checkpoint"), help="Checkpoint output dir")
    ap.add_argument("--max-length", type=int, default=2048)

    ap.add_argument("--examples-per-chunk", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-train-epochs", type=int, default=1)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--resume-from-checkpoint", default=None)
    args = ap.parse_args()

    # If dataset doesn't exist yet, synthesize it from chunks.
    data_path = Path(args.data)
    if not data_path.exists():
        if not Path(args.chunks).exists():
            raise SystemExit(f"Missing chunks file: {args.chunks}. Run run_data.py first.")
        synthesize_sft_run(
            chunks=args.chunks,
            out=args.data,
            base_model=args.synth_model,
            examples_per_chunk=args.examples_per_chunk,
            max_chunks=args.max_chunks,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )

    return finetune_run(
        data=args.data,
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
