from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: peft. Install test/finetune/requirements.txt") from e


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
    out = ""
    for m in messages:
        out += f"{m['role'].upper()}: {m['content']}\n"
    if add_generation_prompt:
        out += "ASSISTANT: "
    return out


class ChatSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.rows: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "messages" not in row:
                    continue
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        messages = row["messages"]

        # Full conversation with assistant answer
        full_text = _apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)

        # Prompt-only: exclude assistant content but add generation prompt
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        prompt_text = _apply_chat_template(self.tokenizer, prompt_msgs, add_generation_prompt=True)

        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        labels = input_ids.copy()
        prompt_len = min(len(prompt["input_ids"]), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # Left-pad to max length in batch (common for decoder-only)
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids = []
    attention_mask = []
    labels = []
    for f in features:
        pad = max_len - len(f["input_ids"])
        input_ids.append([0] * pad + f["input_ids"])
        attention_mask.append([0] * pad + f["attention_mask"])
        labels.append([-100] * pad + f["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="sft.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--output-dir", required=True, help="Folder checkpoint output")
    ap.add_argument("--max-length", type=int, default=2048)

    ap.add_argument("--num-train-epochs", type=int, default=1)
    ap.add_argument("--per-device-train-batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--resume-from-checkpoint", default=None)
    args = ap.parse_args()

    return run(
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


def run(
    *,
    data: str,
    base_model: str,
    output_dir: str,
    max_length: int,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    warmup_ratio: float,
    save_steps: int,
    logging_steps: int,
    resume_from_checkpoint: str | None,
) -> int:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # For QLoRA
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)

    # Use EOS as pad to avoid mismatches
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = ChatSFTDataset(data, tokenizer=tokenizer, max_length=max_length)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collate)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))
    print(f"Saved final adapter to {out_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
