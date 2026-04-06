"""
PageIndex + Llama 3.1 8B RAG System
Hệ thống RAG hoàn chỉnh sử dụng PageIndex methodology
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from pathlib import Path

# Import PageIndex core
from pageindex_core import LocalPageIndex, format_context_for_prompt

# ==================== LLM Wrapper ====================
class LlamaLLM:
    """Wrapper cho Llama 3.1 8B model"""
    
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load model với 4-bit quantization"""
        print(f"🔄 Đang load model {self.model_id}...")
        
        try:
            # Cấu hình quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=bnb_config,
            )
            
            print("✅ Model đã load thành công!")
            
        except Exception as e:
            print(f"❌ Lỗi khi load model: {e}")
            print("\n💡 Thử load model không quantization...")
            
            # Fallback: Load without quantization
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                print("✅ Model đã load thành công (FP16)!")
            except Exception as e2:
                print(f"❌ Không thể load model: {e2}")
                raise
    
    def generate(self, prompt, max_new_tokens=512, temperature=0.2):
        """Sinh text từ prompt"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model chưa được load!")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode (chỉ lấy phần mới sinh ra)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self, messages, max_new_tokens=512, temperature=0.2):
        """Chat với history"""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return self.generate(prompt, max_new_tokens, temperature)


# ==================== RAG System ====================
class PageIndexRAG:
    """
    RAG system kết hợp PageIndex với Llama 3.1 8B
    
    Đặc điểm:
    - Vectorless retrieval: Không dùng vector database
    - Tree-structured indexing: Cấu trúc phân cấp tự nhiên
    - LLM-based reasoning: Sử dụng LLM để trả lời
    """
    
    def __init__(self, documents_dir="./courses", model_id="meta-llama/Llama-3.1-8B-Instruct"):
        # Khởi tạo PageIndex
        self.page_index = LocalPageIndex(documents_dir)
        self.page_index.build_index()
        
        # Khởi tạo LLM
        self.llm = LlamaLLM(model_id)
        
        # System prompt
        self.system_prompt = """Bạn là một trợ lý AI giáo dục thông minh và chuyên nghiệp.

NHIỆM VỤ:
- Trả lời câu hỏi dựa trên thông tin từ tài liệu được cung cấp
- Giải thích rõ ràng, chi tiết và có cấu trúc
- Sử dụng ví dụ cụ thể khi cần thiết

QUY TẮC BẮT BUỘC:
1. Trả lời HOÀN TOÀN bằng tiếng Việt
2. Dựa vào thông tin trong tài liệu để trả lời
3. Nếu thông tin không có trong tài liệu, hãy nói rõ "Thông tin này không có trong tài liệu"
4. Trích dẫn nguồn khi cần thiết
5. Trả lời ngắn gọn nhưng đầy đủ ý
6. Không bịa đặt thông tin không có trong tài liệu
"""
    
    def query(self, question, max_new_tokens=512, temperature=0.2, max_sections=3):
        """
        Truy vấn hệ thống RAG
        
        Args:
            question: Câu hỏi của người dùng
            max_new_tokens: Số token tối đa để sinh
            temperature: Nhiệt độ sampling
            max_sections: Số sections tối đa để retrieve
            
        Returns:
            response: Câu trả lời
            sources: Danh sách nguồn tham khảo
        """
        # Lấy context từ PageIndex
        context, sources = self.page_index.get_context(question, max_sections=max_sections)
        
        if context is None:
            return "⚠️ Không tìm thấy thông tin liên quan trong tài liệu. Vui lòng thêm tài liệu hoặc hỏi câu hỏi khác.", []
        
        # Xây dựng prompt
        user_prompt = format_context_for_prompt(question, context, sources)
        
        # Tạo messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Gọi LLM
        try:
            response = self.llm.chat(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            return response, sources
            
        except Exception as e:
            return f"❌ Lỗi khi sinh câu trả lời: {e}", []
    
    def rebuild_index(self):
        """Xây dựng lại index (khi có tài liệu mới)"""
        print("\n🔄 Đang xây dựng lại index...")
        self.page_index = LocalPageIndex(self.page_index.documents_dir)
        self.page_index.build_index()
    
    def get_statistics(self):
        """Lấy thống kê về hệ thống"""
        return self.page_index.get_statistics()


# ==================== Interactive Interface ====================
def main():
    print("=" * 70)
    print("🚀 PageIndex + Llama 3.1 8B RAG System")
    print("=" * 70)
    print("\n📌 Đặc điểm của PageIndex:")
    print("  ✅ Không sử dụng vector database (vectorless)")
    print("  ✅ Cấu trúc cây phân cấp tự nhiên (tree-structured)")
    print("  ✅ Reasoning-based retrieval (LLM-powered)")
    print("  ✅ Bảo toàn ngữ cảnh tài liệu (context-preserving)")
    print("=" * 70)
    
    # Khởi tạo RAG system
    print("\n🔧 Đang khởi tạo hệ thống...")
    try:
        rag = PageIndexRAG(documents_dir="./courses")
    except Exception as e:
        print(f"\n❌ Lỗi khởi tạo: {e}")
        print("\n💡 Vui lòng kiểm tra:")
        print("  1. Model Llama 3.1 8B đã được download chưa")
        print("  2. GPU có đủ VRAM không (tối thiểu 6GB)")
        print("  3. Thư mục ./courses có tài liệu chưa")
        return
    
    # Hiển thị thống kê
    stats = rag.get_statistics()
    print(f"\n📊 Thống kê hệ thống:")
    print(f"  • Tổng số tài liệu: {stats['total_documents']}")
    print(f"  • Tổng số sections: {stats['total_sections']}")
    if stats['documents']:
        print(f"  • Danh sách tài liệu:")
        for doc in stats['documents']:
            print(f"    - {doc}")
    
    print("\n✅ Hệ thống đã sẵn sàng!")
    print("\n📝 Lệnh đặc biệt:")
    print("  • 'rebuild' - Xây dựng lại index từ tài liệu")
    print("  • 'stats' - Hiển thị thống kê hệ thống")
    print("  • 'exit' hoặc 'quit' - Thoát chương trình")
    print("=" * 70)
    
    # Interactive loop
    while True:
        print("\n")
        user_input = input("💬 Câu hỏi của bạn: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Tạm biệt!")
            break
        
        if user_input.lower() == "rebuild":
            rag.rebuild_index()
            stats = rag.get_statistics()
            print(f"✅ Đã rebuild! Tổng: {stats['total_documents']} docs, {stats['total_sections']} sections")
            continue
        
        if user_input.lower() == "stats":
            stats = rag.get_statistics()
            print(f"\n📊 Thống kê:")
            print(f"  • Tài liệu: {stats['total_documents']}")
            print(f"  • Sections: {stats['total_sections']}")
            print(f"  • Danh sách: {', '.join(stats['documents'])}")
            continue
        
        print("\n🤖 Đang xử lý...")
        print("=" * 70)
        
        try:
            response, sources = rag.query(user_input, max_new_tokens=512, temperature=0.2)
            
            print("\n📝 Trả lời:")
            print(response)
            
            if sources:
                print("\n📚 Nguồn tham khảo:")
                for idx, source in enumerate(sources, 1):
                    print(f"  {idx}. {source}")
            
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            import traceback
            traceback.print_exc()
        
        print("=" * 70)


if __name__ == "__main__":
    main()
