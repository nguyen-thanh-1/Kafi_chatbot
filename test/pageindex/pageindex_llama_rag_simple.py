"""
PageIndex-inspired RAG System with Llama 3.1 8B
Sử dụng LLM từ file Llama_3_1_8B_Instruct_v2.py
"""

import sys
from pathlib import Path
import importlib.util

# Import LLM module từ file có sẵn
def import_llm_module():
    """Import module Llama từ file có sẵn"""
    module_path = Path(__file__).parent / "Llama_3_1_8B_Instruct_v2.py"
    
    spec = importlib.util.spec_from_file_location("llama_module", module_path)
    llama_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llama_module)
    
    return llama_module

# Load LLM
print("Đang import Llama 3.1 8B model...")
llama = import_llm_module()
print("Model đã sẵn sàng!")

# ==================== PageIndex Implementation ====================
class LocalPageIndex:
    """
    Local implementation của PageIndex - Tree-structured document indexing
    Không cần vector database, sử dụng cấu trúc phân cấp tự nhiên
    """
    
    def __init__(self, documents_dir="./courses"):
        self.documents_dir = Path(documents_dir)
        self.index = {}
        self.documents = {}
        
    def build_index(self):
        """Xây dựng index phân cấp từ tài liệu"""
        print(f"\n📚 Đang xây dựng PageIndex từ {self.documents_dir}...")
        
        if not self.documents_dir.exists():
            print(f"⚠️  Tạo thư mục: {self.documents_dir}")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Xử lý tất cả file .txt
        txt_files = list(self.documents_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"⚠️  Không tìm thấy file .txt nào trong {self.documents_dir}")
            return
        
        for file_path in txt_files:
            self._index_document(file_path)
        
        print(f"✅ Đã index {len(self.documents)} tài liệu với {sum(len(d['sections']) for d in self.documents.values())} sections")
        
    def _index_document(self, file_path):
        """Index một tài liệu với cấu trúc phân cấp"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Lỗi đọc file {file_path}: {e}")
            return
        
        doc_name = file_path.stem
        
        # Tách tài liệu thành sections (theo đoạn văn)
        sections = [s.strip() for s in content.split('\n\n') if s.strip()]
        
        # Tạo cấu trúc phân cấp
        doc_structure = {
            'name': doc_name,
            'path': str(file_path),
            'sections': []
        }
        
        for idx, section in enumerate(sections):
            # Xác định tiêu đề (dòng đầu tiên nếu ngắn hoặc kết thúc bằng :)
            lines = section.split('\n')
            if len(lines) > 1 and (lines[0].endswith(':') or len(lines[0]) < 100):
                title = lines[0].strip(':').strip()
                content_text = '\n'.join(lines[1:]).strip()
            else:
                title = f"Phần {idx + 1}"
                content_text = section
            
            doc_structure['sections'].append({
                'title': title,
                'content': content_text,
                'index': idx
            })
        
        self.documents[doc_name] = doc_structure
        self.index[doc_name] = {
            'sections': [s['title'] for s in doc_structure['sections']],
            'path': str(file_path)
        }
        
        print(f"  📄 {doc_name}: {len(doc_structure['sections'])} sections")
    
    def search(self, query, top_k=3):
        """
        Tìm kiếm sections liên quan sử dụng reasoning
        Trả về các sections liên quan nhất
        """
        if not self.documents:
            return []
        
        relevant_sections = []
        
        for doc_name, doc_data in self.documents.items():
            for section in doc_data['sections']:
                # Tính điểm liên quan
                score = self._calculate_relevance(query, section['title'], section['content'])
                
                if score > 0:
                    relevant_sections.append({
                        'document': doc_name,
                        'title': section['title'],
                        'content': section['content'],
                        'score': score
                    })
        
        # Sắp xếp theo điểm và lấy top_k
        relevant_sections.sort(key=lambda x: x['score'], reverse=True)
        return relevant_sections[:top_k]
    
    def _calculate_relevance(self, query, title, content):
        """Tính điểm liên quan đơn giản (có thể nâng cấp với LLM)"""
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        score = 0
        
        # Tìm các từ khóa trong query
        query_terms = [term for term in query_lower.split() if len(term) > 2]
        
        for term in query_terms:
            # Tiêu đề có trọng số cao hơn
            if term in title_lower:
                score += 5
            # Nội dung có trọng số thấp hơn
            if term in content_lower:
                score += 1
        
        return score
    
    def get_context(self, query, max_sections=3):
        """Lấy context đã format cho LLM prompt"""
        sections = self.search(query, top_k=max_sections)
        
        if not sections:
            return None, []
        
        context_parts = []
        sources = []
        
        for section in sections:
            context_parts.append(
                f"📖 [{section['document']}] - {section['title']}\n{section['content']}"
            )
            sources.append(f"{section['document']} - {section['title']}")
        
        context = "\n\n" + "="*60 + "\n\n".join(context_parts)
        
        return context, sources


# ==================== RAG System ====================
class PageIndexRAG:
    """RAG system kết hợp LocalPageIndex với Llama 3.1 8B"""
    
    def __init__(self, documents_dir="./courses"):
        self.page_index = LocalPageIndex(documents_dir)
        self.page_index.build_index()
        
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
"""
    
    def query(self, question, max_new_tokens=512, temperature=0.2):
        """
        Truy vấn hệ thống RAG
        
        Args:
            question: Câu hỏi của người dùng
            max_new_tokens: Số token tối đa để sinh
            temperature: Nhiệt độ sampling
            
        Returns:
            response: Câu trả lời
            sources: Danh sách nguồn tham khảo
        """
        # Lấy context từ PageIndex
        context, sources = self.page_index.get_context(question, max_sections=3)
        
        if context is None:
            return "⚠️ Không tìm thấy thông tin liên quan trong tài liệu. Vui lòng thêm tài liệu hoặc hỏi câu hỏi khác.", []
        
        # Xây dựng prompt với context
        user_prompt = f"""Dựa vào các thông tin sau từ tài liệu:

{context}

{'='*60}

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên thông tin trong tài liệu ở trên. Nếu cần, hãy tổng hợp từ nhiều nguồn."""
        
        # Tạo history với system prompt
        history = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Gọi LLM
        try:
            response = llama.generate_response(
                user_input=user_prompt,
                history=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            return response.strip(), sources
            
        except Exception as e:
            return f"❌ Lỗi khi sinh câu trả lời: {e}", []
    
    def rebuild_index(self):
        """Xây dựng lại index (khi có tài liệu mới)"""
        print("\n🔄 Đang xây dựng lại index...")
        self.page_index = LocalPageIndex(self.page_index.documents_dir)
        self.page_index.build_index()


# ==================== Interactive Interface ====================
def main():
    print("=" * 70)
    print("🚀 PageIndex + Llama 3.1 8B RAG System")
    print("=" * 70)
    print("\n📌 Đặc điểm của PageIndex:")
    print("  ✅ Không sử dụng vector database")
    print("  ✅ Cấu trúc cây phân cấp tự nhiên")
    print("  ✅ Reasoning-based retrieval")
    print("  ✅ Bảo toàn ngữ cảnh tài liệu")
    print("=" * 70)
    
    # Khởi tạo RAG system
    rag = PageIndexRAG(documents_dir="./courses")
    
    print("\n✅ Hệ thống đã sẵn sàng!")
    print("\n📝 Lệnh đặc biệt:")
    print("  • 'rebuild' - Xây dựng lại index từ tài liệu")
    print("  • 'exit' hoặc 'quit' - Thoát chương trình")
    print("=" * 70)
    
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
            continue
        
        print("\n🤖 Đang xử lý...")
        print("=" * 70)
        
        try:
            response, sources = rag.query(user_input)
            
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
