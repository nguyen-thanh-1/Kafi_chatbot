"""
RAG Pro V2 - INDEX ONLY
═══════════════════════════════════════════════════════════
Chỉ dùng để index tài liệu, KHÔNG query.
Embedding chạy trên GPU để tối ưu tốc độ.

USAGE:
  uv run rag_index.py --force     # Re-index tất cả
  uv run rag_index.py             # Chỉ index file mới
  
PERFORMANCE:
  - Embedding GPU: 3-4x nhanh hơn CPU
  - VRAM: ~3GB (chỉ Embedding)
  - Index 800-page PDF: ~2-3 phút
═══════════════════════════════════════════════════════════
"""

import os
import time
import argparse
from pathlib import Path

# Import từ rag_pro_v2
from rag_pro_v2 import (
    RAG_STORAGE,
    COURSES_FOLDER,
    SUPPORTED_EXTENSIONS,
    EmbeddingCache,
    IndexTracker,
    VectorStore,
    read_file,
    chunk_text_semantic,
    EMBEDDING_CACHE_FILE,
    TRACKER_FILE,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
)

# ═══════════════════════════════════════════════════════════
# EMBEDDING MODEL - FORCE GPU
# ═══════════════════════════════════════════════════════════
_embedder = None

def get_embedder_gpu():
    """Load embedding model trên GPU"""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch
        
        EMBEDDING_MODEL = "BAAI/bge-m3"
        print(f"   📥 Loading {EMBEDDING_MODEL} on GPU...")
        
        # Force GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        
        print(f"   ✅ Embedding model loaded ({device.upper()})")
        
        if device == 'cpu':
            print("   ⚠️  WARNING: GPU not available, using CPU (slower)")
    
    return _embedder


def embed_texts_gpu(texts, cache, batch_size=128):
    """Embed texts với GPU và cache"""
    import numpy as np
    
    if not texts:
        return np.array([])
    
    # Get cached and to-embed
    cached_indices, to_embed, embed_indices = cache.get_batch(texts)
    
    # Initialize result array
    embeddings = np.zeros((len(texts), 1024), dtype=np.float32)
    
    # Fill cached embeddings
    for idx in cached_indices:
        embeddings[idx] = cache.get(texts[idx])
    
    # Embed new texts on GPU
    if to_embed:
        print(f"   🔄 Embedding {len(to_embed)} new chunks (cached: {len(cached_indices)})...")
        
        embedder = get_embedder_gpu()
        new_embeddings = embedder.encode(
            to_embed,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Fill new embeddings and cache them
        for i, (text, emb) in enumerate(zip(to_embed, new_embeddings)):
            idx = embed_indices[i]
            embeddings[idx] = emb
            cache.set(text, emb)
    else:
        print(f"   ✅ All {len(texts)} chunks from cache!")
    
    return embeddings


# ═══════════════════════════════════════════════════════════
# INDEXER
# ═══════════════════════════════════════════════════════════
class RAGIndexer:
    """Chỉ dùng để index, không query"""
    
    def __init__(self):
        self.vector_store = VectorStore(RAG_STORAGE)
        self.tracker = IndexTracker(TRACKER_FILE)
        self.cache = EmbeddingCache(EMBEDDING_CACHE_FILE)
    
    def index_file(self, file_path: str) -> int:
        """Index một file"""
        text = read_file(file_path)
        chunks = chunk_text_semantic(text)
        
        if chunks:
            # Embed với GPU
            embeddings = embed_texts_gpu(chunks, self.cache)
            
            # Add to vector store
            self.vector_store.add_chunks(chunks, os.path.basename(file_path), embeddings)
            self.tracker.mark_indexed(file_path, len(chunks))
        
        return len(chunks)
    
    def index_folder(self, folder: str, force: bool = False) -> int:
        """Index toàn bộ folder"""
        if force:
            self.vector_store.clear()
            self.tracker.indexed_files = {}
            self.tracker._save()
            # Don't clear cache - reuse it!
        
        if not os.path.exists(folder):
            print(f"⚠️ Folder không tồn tại: {folder}")
            return 0
        
        # Find files
        all_files = [f for f in os.listdir(folder) 
                     if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        
        # Filter new files
        new_files = [f for f in all_files 
                     if self.tracker.needs_indexing(os.path.join(folder, f))]
        
        if not new_files:
            print(f"✅ Không có file mới. Database: {self.tracker.get_indexed_count()} files, {self.tracker.get_total_chunks()} chunks")
            return 0
        
        print(f"\n🆕 Phát hiện {len(new_files)} file cần index:")
        
        total_chunks = 0
        for i, filename in enumerate(new_files, 1):
            file_path = os.path.join(folder, filename)
            try:
                print(f"   [{i}/{len(new_files)}] {filename}...", end=" ", flush=True)
                start = time.time()
                chunks = self.index_file(file_path)
                elapsed = time.time() - start
                print(f"✅ {chunks} chunks ({elapsed:.1f}s)")
                total_chunks += chunks
            except Exception as e:
                print(f"❌ {e}")
        
        # Save cache
        self.cache.save()
        
        # Print stats
        cache_stats = self.cache.get_stats()
        print(f"\n📊 Indexing Stats:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Cache hit rate: {cache_stats['hit_rate']:.1f}%")
        
        return total_chunks


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RAG Pro V2 - Index Only")
    parser.add_argument('--force', '-f', action='store_true', help='Force re-index')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG PRO V2 - INDEX ONLY")
    print("═" * 60)
    print(f"   📊 Embedding: BAAI/bge-m3 (GPU)")
    print(f"   ⚡ Chunking:  Semantic ({MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE} words)")
    print(f"   💾 Cache:     Enabled")
    print(f"   📁 Folder:    {COURSES_FOLDER}")
    print("═" * 60)
    
    # Initialize
    print("\n🔄 Loading embedding model...")
    indexer = RAGIndexer()
    
    # Load embedding model
    get_embedder_gpu()
    
    # Index
    print("\n" + "═" * 60)
    print("📁 INDEXING")
    print("═" * 60)
    
    start = time.time()
    total_chunks = indexer.index_folder(COURSES_FOLDER, force=args.force)
    elapsed = time.time() - start
    
    print("\n" + "═" * 60)
    print("✅ INDEXING COMPLETE")
    print("═" * 60)
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Storage: {RAG_STORAGE}")
    print("\n💡 Bây giờ có thể chạy rag_query.py để query!")
    print("═" * 60)


if __name__ == "__main__":
    main()
