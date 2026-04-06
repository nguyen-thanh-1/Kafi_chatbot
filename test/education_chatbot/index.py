#!/usr/bin/env python3
"""
Index script for Education Chatbot
Index all files in data folder
"""

import os
import time
from rag_engine import RAGHybrid, get_embedder, get_reranker, DATA_DIR


def main():
    print("═" * 60)
    print("📚 EDUCATION CHATBOT - INDEX")
    print("═" * 60)
    
    # Load embedder
    print("\n🔄 Loading embedding model...")
    get_embedder()
    
    # Initialize RAG
    rag = RAGHybrid()
    
    # Get all files in data folder
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"\n📁 Created {DATA_DIR} folder")
        print("   Put your PDF, TXT, MD, CSV files here and run again!")
        return
    
    files = []
    for root, dirs, filenames in os.walk(DATA_DIR):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.pdf', '.txt', '.md', '.csv']:
                files.append(os.path.join(root, filename))
    
    if not files:
        print(f"\n❌ No files found in {DATA_DIR}")
        print("   Put your PDF, TXT, MD, CSV files there and run again!")
        return
    
    print(f"\n📁 Found {len(files)} files:")
    for f in files:
        print(f"   • {os.path.basename(f)}")
    
    # Index
    print("\n" + "═" * 60)
    print("🔄 INDEXING")
    print("═" * 60)
    
    start = time.time()
    total_chunks = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\n   📄 {filename}...")
        
        # Check if already indexed
        if filename in rag.vector_store.files:
            print(f"      ⏭️ Already indexed, skipping")
            total_chunks += rag.vector_store.files[filename].get('chunks', 0)
            continue
        
        try:
            chunks = rag.index_file(filepath, filename)
            total_chunks += chunks
            print(f"      ✅ {chunks} chunks")
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    elapsed = time.time() - start
    
    print("\n" + "═" * 60)
    print("✅ INDEXING COMPLETE")
    print("═" * 60)
    print(f"   Total files: {len(files)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Time: {elapsed:.1f}s")
    print("\n💡 Run: streamlit run app.py")
    print("═" * 60)


if __name__ == "__main__":
    main()
