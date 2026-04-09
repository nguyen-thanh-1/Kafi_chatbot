from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils_text import iter_files, read_document


def run(
    input_dir: str,
    out: str,
    *,
    input_file: str | None = None,
) -> int:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        paths = [Path(input_file)] if input_file else list(iter_files(input_dir))
        for path in paths:
            doc = read_document(path)
            if not doc.text.strip():
                name = path.name.encode("utf-8", errors="backslashreplace").decode("utf-8")
                print(f"Skip empty text: {name} (PDF may be image-based)")
                continue
            row = {"doc_id": doc.doc_id, "source": doc.source_path, "text": doc.text}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} documents to {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Input folder with docs (.txt/.md/.pdf)")
    ap.add_argument("--input-file", default=None, help="Optional single file to process")
    ap.add_argument("--out", required=True, help="Output JSONL corpus path")
    args = ap.parse_args()
    return run(
        args.input_dir,
        args.out,
        input_file=args.input_file,
    )


if __name__ == "__main__":
    raise SystemExit(main())
