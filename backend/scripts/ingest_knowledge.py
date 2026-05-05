import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.knowledge_base import get_knowledge_base
from src.utils.logger import logger

def ingest_file(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    kb = get_knowledge_base()
    print(f"Ingesting {file_path} into Knowledge Base (Qdrant)...")
    kb.ingest_pdf(file_path)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest PDF into Knowledge Base")
    parser.add_argument("file", help="Path to PDF file")
    
    args = parser.parse_args()
    ingest_file(args.file)
