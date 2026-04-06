#!/usr/bin/env python3
"""
RAG Lite - Index Only Script
Fast indexing with MiniLM-L6-v2 (GPU)
"""

import argparse
import time
import gc
import torch
from rag_lite import (
    RAGLite, 
    get_embedder, 
    COURSES_FOLDER,
    EMBEDDING_MODEL,
    CHUNK_SIZE
)


def cleanup_memory():
    """Free RAM and VRAM after indexing"""
    global _embedder
    
    # Clear global embedder
    import rag_lite
    if hasattr(rag_lite, '_embedder') and rag_lite._embedder is not None:
        del rag_lite._embedder
        rag_lite._embedder = None
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("   🧹 Memory cleaned up")


def main():
    parser = argparse.ArgumentParser(description="RAG Lite - Index Documents")
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index all files')
    parser.add_argument('--folder', type=str, default=COURSES_FOLDER, help='Folder to index')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG LITE - INDEX ONLY")
    print("═" * 60)
    print(f"   📊 Embedding: {EMBEDDING_MODEL} (GPU)")
    print(f"   ⚡ Chunking:  Recursive ({CHUNK_SIZE} chars)")
    print(f"   📁 Folder:    {args.folder}")
    print("═" * 60)
    
    print("\n🔄 Loading embedding model...")
    get_embedder()
    
    print("\n" + "═" * 60)
    print("📁 INDEXING")
    print("═" * 60)
    
    start_time = time.time()
    
    rag = RAGLite()
    if not args.force:
        rag.load()
    
    total_chunks = rag.index_folder(args.folder, force=args.force)
    
    elapsed = time.time() - start_time
    
    print("\n" + "═" * 60)
    print("✅ INDEXING COMPLETE")
    print("═" * 60)
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Storage: {rag.vector_store.storage_dir}")
    
    # Cleanup memory
    print("\n🔄 Cleaning up memory...")
    cleanup_memory()
    
    print("\n💡 Bây giờ có thể chạy rag_query.py để query!")
    print("═" * 60)


if __name__ == "__main__":
    main()
