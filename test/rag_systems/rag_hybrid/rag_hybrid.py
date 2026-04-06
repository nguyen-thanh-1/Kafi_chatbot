"""
RAG Hybrid - 2-Stage RAG with Question Routing
═══════════════════════════════════════════════════════════

ARCHITECTURE:
    User Question
          │
          ▼
    [Question Router]
          │
    ┌─────┴─────────────┐
    │                   │
    ▼                   ▼
  rag_lite           rag_pro
  (fast)             (deep)
    │                   │
    ▼                   ▼
  LLM + Prior      Strict RAG
  Knowledge        (No hallucination)

ROUTING RULES:
- rag_pro: "theo tài liệu", "trong sách", "chương X", specific citations
- rag_lite: General knowledge, definitions, common concepts

PROMPTS:
- rag_lite: HYBRID (context + LLM general knowledge)
- rag_pro: STRICT (only document context)

═══════════════════════════════════════════════════════════
"""

import os
import sys
import time
import re
from typing import Tuple, Optional

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
SIMILARITY_THRESHOLD = 0.5  # Below this, use LLM general knowledge

# Keywords that trigger rag_pro (strict mode)
RAG_PRO_KEYWORDS = [
    "theo tài liệu", "trong sách", "trong tài liệu", "theo sách",
    "chương", "trang", "section", "chapter", "page",
    "được định nghĩa", "được mô tả", "được giải thích",
    "so sánh trong tài liệu", "trích dẫn", "quote",
    "theo như", "dựa theo", "như đã nói"
]

# Keywords that trigger rag_lite (fast + hybrid)
RAG_LITE_KEYWORDS = [
    "là gì", "what is", "định nghĩa", "definition",
    "giải thích", "explain", "có nghĩa là gì",
    "tại sao", "why", "như thế nào", "how",
    "ví dụ", "example", "ứng dụng", "application"
]


# ═══════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════
HYBRID_PROMPT = """Based on the following context, answer the question.

RULES:
1. Prefer using the provided context if relevant
2. If context is insufficient, you may use general AI knowledge
3. Clearly indicate when the answer is based on general knowledge
4. Answer in the same language as the question

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

STRICT_PROMPT = """Based on the following context, answer the question accurately.

IMPORTANT RULES:
1. ONLY use information from the context below
2. If the answer is NOT in the context, say "Tôi không tìm thấy thông tin này trong tài liệu."
3. Be specific and cite which part of the context you're using
4. Answer in the same language as the question

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

NO_CONTEXT_PROMPT = """Answer the following question using your general AI knowledge.

RULES:
1. Be accurate and educational
2. Answer in the same language as the question
3. If you're unsure, indicate your uncertainty

QUESTION: {question}

ANSWER:"""


# ═══════════════════════════════════════════════════════════
# QUESTION ROUTER
# ═══════════════════════════════════════════════════════════
class QuestionRouter:
    """
    Route questions to appropriate RAG pipeline:
    - rag_pro: Document-specific questions (strict mode)
    - rag_lite: General knowledge questions (hybrid mode)
    - llm_only: When no relevant context found
    """
    
    def __init__(self):
        pass
    
    def classify(self, question: str, context_score: float = 0.0) -> str:
        """
        Classify question into routing mode
        
        Returns:
            "rag_pro" - Use strict RAG with only document context
            "rag_lite" - Use hybrid RAG with LLM general knowledge
            "llm_only" - Use LLM without RAG context
        """
        question_lower = question.lower()
        
        # Rule 1: Check for rag_pro keywords (document-specific)
        for keyword in RAG_PRO_KEYWORDS:
            if keyword in question_lower:
                return "rag_pro"
        
        # Rule 2: Check similarity score
        if context_score < SIMILARITY_THRESHOLD:
            return "llm_only"
        
        # Rule 3: Default to rag_lite (hybrid)
        return "rag_lite"
    
    def get_prompt(self, mode: str, question: str, context: str) -> str:
        """Get appropriate prompt based on mode"""
        if mode == "rag_pro":
            return STRICT_PROMPT.format(context=context, question=question)
        elif mode == "rag_lite":
            return HYBRID_PROMPT.format(context=context, question=question)
        else:  # llm_only
            return NO_CONTEXT_PROMPT.format(question=question)


# ═══════════════════════════════════════════════════════════
# RAG HYBRID PIPELINE
# ═══════════════════════════════════════════════════════════
class RAGHybrid:
    """
    Hybrid RAG system that intelligently routes between:
    - rag_lite (fast, hybrid prompt)
    - rag_pro (deep, strict prompt)
    """
    
    def __init__(self):
        self.router = QuestionRouter()
        self.rag_lite = None
        self.rag_pro = None
        self.llm = None
    
    def _load_rag_lite(self):
        """Lazy load rag_lite"""
        if self.rag_lite is None:
            print("   📦 Loading RAG Lite...")
            from rag_systems.rag_lite.rag_lite import RAGLite, get_embedder, get_reranker
            self.rag_lite = RAGLite()
            self.rag_lite.load()
            get_embedder()
            get_reranker()
            print("   ✅ RAG Lite ready")
    
    def _load_rag_pro(self):
        """Lazy load rag_pro"""
        if self.rag_pro is None:
            print("   📦 Loading RAG Pro...")
            from rag_systems.rag_pro.rag_pro_v2 import RAGProV2, get_embedding_model, get_reranker
            self.rag_pro = RAGProV2()
            self.rag_pro.load()
            print("   ✅ RAG Pro ready")
    
    def _load_llm(self):
        """Load LLM"""
        if self.llm is None:
            print("   📥 Loading Llama 3.1 8B...")
            from llm_models.Llama_3_1_8B_Instruct_v2 import generate_response, _load_model
            _load_model()
            self.llm = generate_response
            print("   ✅ Llama 3.1 8B loaded (GPU)")
    
    def query_lite(self, question: str) -> Tuple[str, float]:
        """Query using rag_lite (fast)"""
        self._load_rag_lite()
        
        # Get context
        from rag_systems.rag_lite.rag_lite import TOP_K_RETRIEVE, TOP_K_RERANK, rerank
        
        retrieved = self.rag_lite.vector_store.search(question, TOP_K_RETRIEVE)
        if not retrieved:
            return "", 0.0
        
        reranked = rerank(question, retrieved, TOP_K_RERANK)
        if not reranked:
            return "", 0.0
        
        # Get top score
        top_score = reranked[0][1] if reranked else 0.0
        
        # Build context
        context_parts = []
        for i, (chunk, score) in enumerate(reranked, 1):
            context_parts.append(f"[Đoạn {i}]\n{chunk}")
        context = "\n\n---\n\n".join(context_parts)
        
        return context, top_score
    
    def query_pro(self, question: str) -> Tuple[str, float]:
        """Query using rag_pro (deep)"""
        self._load_rag_pro()
        
        from rag_systems.rag_pro.rag_pro_v2 import TOP_K_RETRIEVE, TOP_K_RERANK, rerank
        
        retrieved = self.rag_pro.vector_store.search(question, TOP_K_RETRIEVE)
        if not retrieved:
            return "", 0.0
        
        reranked = rerank(question, retrieved, TOP_K_RERANK)
        if not reranked:
            return "", 0.0
        
        top_score = reranked[0][1] if reranked else 0.0
        
        context_parts = []
        for i, (chunk, score) in enumerate(reranked, 1):
            context_parts.append(f"[Đoạn {i}]\n{chunk}")
        context = "\n\n---\n\n".join(context_parts)
        
        return context, top_score
    
    def generate(self, prompt: str) -> str:
        """Generate answer using LLM"""
        self._load_llm()
        
        system_prompt = "You are a helpful educational AI assistant. Be accurate, clear, and helpful."
        
        response = self.llm(
            user_input=prompt,
            history=[],
            system_prompt=system_prompt,
            max_new_tokens=700,
            temperature=0.21,
        )
        
        return response
    
    def query(self, question: str, verbose: bool = True) -> str:
        """
        Main query method with intelligent routing
        """
        import torch
        start = time.time()
        
        # Step 1: Quick classification based on keywords
        initial_mode = self.router.classify(question)
        
        if verbose:
            print(f"   🔍 Initial classification: {initial_mode}")
        
        # Step 2: Get context based on initial classification
        if initial_mode == "rag_pro":
            if verbose:
                print(f"   📚 Using RAG Pro (strict mode)...")
            context, score = self.query_pro(question)
            mode = "rag_pro"
        else:
            if verbose:
                print(f"   ⚡ Using RAG Lite (fast mode)...")
            context, score = self.query_lite(question)
            
            # Re-evaluate based on score
            mode = self.router.classify(question, score)
            if mode == "llm_only":
                if verbose:
                    print(f"   ℹ️  Low relevance score ({score:.2f}) → Using LLM general knowledge")
        
        if verbose:
            print(f"   📊 Context score: {score:.2f}")
            print(f"   🎯 Final mode: {mode}")
        
        # Step 3: Generate answer
        if verbose:
            print(f"   🤖 Generating answer...")
        
        prompt = self.router.get_prompt(mode, question, context)
        answer = self.generate(prompt)
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        elapsed = time.time() - start
        if verbose:
            print(f"   ⏱️ Total: {elapsed:.1f}s")
        
        return answer
    
    def preload_lite(self):
        """Preload rag_lite for fast queries"""
        self._load_rag_lite()
        self._load_llm()
    
    def preload_all(self):
        """Preload all models (uses more memory)"""
        self._load_rag_lite()
        self._load_rag_pro()
        self._load_llm()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Hybrid - 2-Stage RAG System")
    parser.add_argument('--query', '-q', type=str, help='Single query mode')
    parser.add_argument('--preload', choices=['lite', 'all'], default='lite', 
                        help='Preload mode: lite (default) or all')
    args = parser.parse_args()
    
    print("═" * 60)
    print("🚀 RAG HYBRID - 2-Stage RAG System")
    print("═" * 60)
    print("   📊 Strategy: Question Router → rag_lite / rag_pro")
    print("   ⚡ Fast mode: RAG Lite + LLM General Knowledge")
    print("   📚 Deep mode: RAG Pro (Strict Document Only)")
    print("═" * 60)
    
    rag = RAGHybrid()
    
    print(f"\n🔄 Preloading ({args.preload} mode)...")
    if args.preload == 'all':
        rag.preload_all()
    else:
        rag.preload_lite()
    
    # Single query mode
    if args.query:
        print("\n" + "═" * 60)
        print(f"\n❓ {args.query}")
        print("\n🤖 Đang xử lý...")
        answer = rag.query(args.query)
        print(f"\n📝 Trả lời:\n{answer}")
        return
    
    # Interactive mode
    print("\n" + "═" * 60)
    print("💬 INTERACTIVE MODE")
    print("═" * 60)
    print("Gõ câu hỏi. 'exit' để thoát.")
    print("")
    print("💡 Tips:")
    print("   - 'NLP là gì?' → Fast mode (hybrid)")
    print("   - 'Theo tài liệu, NLP là gì?' → Deep mode (strict)")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🧑 Bạn: ").strip()
            
            if question.lower() in ["exit", "quit", "q"]:
                print("👋 Tạm biệt!")
                break
            
            if not question:
                continue
            
            print("\n🤖 Đang xử lý...")
            answer = rag.query(question)
            print(f"\n📝 Trả lời:\n{answer}")
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break


if __name__ == "__main__":
    main()
