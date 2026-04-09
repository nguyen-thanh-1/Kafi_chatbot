# Finetune Qwen/Qwen3-8B (VN Finance)

1) Put your documents into `test/finetune/data/raw/` (`.pdf`, `.txt`, `.md`).
2) Run exactly 2 commands from `C:\Users\Admin\Desktop\Kafi_chatbot`:

```powershell
uv run .\test\finetune\run_data.py
uv run .\test\finetune\run_tune.py
```

Outputs:
- SFT dataset: `test/finetune/data/processed/sft.jsonl`
- Checkpoints/adapters: `test/finetune/checkpoint/`

Note:
- `run_data.py` only reads files + chunks (no model load).
- `run_tune.py` will synthesize the SFT dataset (loads a model) and then finetune (loads the base model).
