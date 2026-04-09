import json
import os
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def load_financial_phrasebank_allagree(limit: int = 1000):
    """
    Load Financial PhraseBank (Sentences_AllAgree.txt) without `datasets`.

    The installed `datasets` (v4+) no longer supports dataset scripts, and
    `financial_phrasebank` is script-based on the Hub. We download the official
    zip from the dataset repo and parse the text file ourselves.
    """
    # Prefer a local zip if provided (useful when offline or behind a proxy)
    local_zip = os.getenv("FINPHRASEBANK_ZIP")
    if local_zip and Path(local_zip).exists():
        zip_path = str(Path(local_zip))
    else:
        cache_dir = Path(__file__).resolve().parents[1] / "data" / ".hf_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            zip_path = hf_hub_download(
                repo_id="financial_phrasebank",
                repo_type="dataset",
                filename="data/FinancialPhraseBank-v1.0.zip",
                cache_dir=str(cache_dir),
                # Avoid inheriting a broken system proxy config (common on Windows)
                proxies={"http": None, "https": None},
            )
        except Exception as e:
            proxy_env = {k: os.getenv(k) for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"] if os.getenv(k)}
            hint = (
                "Failed to download FinancialPhraseBank-v1.0.zip from Hugging Face.\n"
                "If you are behind a proxy, either:\n"
                "  - set FINPHRASEBANK_ZIP to a local path of FinancialPhraseBank-v1.0.zip, or\n"
                "  - fix/unset your proxy env vars (HTTP_PROXY/HTTPS_PROXY).\n"
                f"Detected proxy env: {proxy_env}\n"
            )
            raise RuntimeError(hint) from e

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Files live under FinancialPhraseBank-v1.0/
        target = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
        with zf.open(target) as f:
            # Dataset file is ISO-8859-1 encoded (per upstream dataset script)
            lines = f.read().decode("iso-8859-1").splitlines()

    records = []
    for line in lines:
        if not line.strip():
            continue
        sentence, label = line.rsplit("@", 1)
        records.append({"sentence": sentence, "label": label.strip()})
        if limit and len(records) >= limit:
            break

    return records


def main():
    save_path = Path(r"C:\Users\Admin\Desktop\Kafi_chatbot\test\data\dataset")
    save_path.mkdir(parents=True, exist_ok=True)

    train = load_financial_phrasebank_allagree(limit=1000)
    with open(save_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
