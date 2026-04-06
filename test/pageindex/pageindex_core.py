"""
PageIndex-inspired RAG System
Phiên bản đơn giản không cần load lại model
"""

from pathlib import Path

# ==================== PageIndex Implementation ====================
class LocalPageIndex:
    """
    Local implementation của PageIndex - Tree-structured document indexing
    Không cần vector database, sử dụng cấu trúc phân cấp tự nhiên
    
    Đặc điểm:
    - Không chunking tùy ý: Tổ chức theo sections tự nhiên
    - Không vector search: Sử dụng keyword matching và reasoning
    - Bảo toàn cấu trúc: Giữ nguyên hierarchy của tài liệu
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
            print(f"💡 Vui lòng thêm file .txt vào thư mục {self.documents_dir}")
            return
        
        # Xử lý tất cả file .txt
        txt_files = list(self.documents_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"⚠️  Không tìm thấy file .txt nào trong {self.documents_dir}")
            print(f"💡 Vui lòng thêm file .txt vào thư mục này")
            return
        
        for file_path in txt_files:
            self._index_document(file_path)
        
        total_sections = sum(len(d['sections']) for d in self.documents.values())
        print(f"✅ Đã index {len(self.documents)} tài liệu với {total_sections} sections")
        
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
            if len(lines) > 1:
                first_line = lines[0].strip()
                # Kiểm tra xem dòng đầu có phải tiêu đề không
                if (first_line.endswith(':') or 
                    first_line.endswith('.') == False and len(first_line) < 100 or
                    first_line.startswith('#')):
                    title = first_line.strip(':').strip('#').strip()
                    content_text = '\n'.join(lines[1:]).strip()
                else:
                    # Lấy 50 ký tự đầu làm tiêu đề
                    title = first_line[:50] + "..." if len(first_line) > 50 else first_line
                    content_text = section
            else:
                title = section[:50] + "..." if len(section) > 50 else section
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
        """
        Tính điểm liên quan sử dụng keyword matching
        Trong PageIndex thực tế, bước này sẽ sử dụng LLM reasoning
        """
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        score = 0
        
        # Tách query thành các từ khóa (bỏ từ quá ngắn)
        query_terms = [term for term in query_lower.split() if len(term) > 2]
        
        for term in query_terms:
            # Tiêu đề có trọng số cao hơn (5 điểm)
            if term in title_lower:
                score += 5
            # Nội dung có trọng số thấp hơn (1 điểm)
            if term in content_lower:
                score += 1
        
        # Bonus nếu có nhiều từ khóa xuất hiện
        matching_terms = sum(1 for term in query_terms if term in content_lower)
        if matching_terms > len(query_terms) * 0.5:  # Hơn 50% từ khóa match
            score += 3
        
        return score
    
    def get_context(self, query, max_sections=3):
        """Lấy context đã format cho LLM prompt"""
        sections = self.search(query, top_k=max_sections)
        
        if not sections:
            return None, []
        
        context_parts = []
        sources = []
        
        for idx, section in enumerate(sections, 1):
            context_parts.append(
                f"[Nguồn {idx}: {section['document']} - {section['title']}]\n{section['content']}"
            )
            sources.append(f"{section['document']} - {section['title']}")
        
        context = "\n\n" + ("-" * 60 + "\n\n").join(context_parts)
        
        return context, sources
    
    def get_statistics(self):
        """Lấy thống kê về index"""
        total_docs = len(self.documents)
        total_sections = sum(len(d['sections']) for d in self.documents.values())
        
        return {
            'total_documents': total_docs,
            'total_sections': total_sections,
            'documents': list(self.documents.keys())
        }


# ==================== Utility Functions ====================
def format_context_for_prompt(query, context, sources):
    """Format context thành prompt cho LLM"""
    
    if context is None:
        return None
    
    prompt = f"""Dựa vào các thông tin sau từ tài liệu:

{context}

{'='*60}

Câu hỏi: {query}

Hãy trả lời câu hỏi dựa trên thông tin trong tài liệu ở trên. Nếu cần, hãy tổng hợp từ nhiều nguồn.
Trả lời bằng tiếng Việt, rõ ràng và chi tiết."""

    return prompt


def demo_pageindex():
    """Demo PageIndex system"""
    print("=" * 70)
    print("🚀 PageIndex Demo - Tree-structured Document Indexing")
    print("=" * 70)
    print("\n📌 Đặc điểm của PageIndex:")
    print("  ✅ Không sử dụng vector database")
    print("  ✅ Cấu trúc cây phân cấp tự nhiên")
    print("  ✅ Reasoning-based retrieval")
    print("  ✅ Bảo toàn ngữ cảnh tài liệu")
    print("=" * 70)
    
    # Khởi tạo PageIndex
    page_index = LocalPageIndex(documents_dir="./courses")
    page_index.build_index()
    
    # Hiển thị thống kê
    stats = page_index.get_statistics()
    print(f"\n📊 Thống kê:")
    print(f"  • Tổng số tài liệu: {stats['total_documents']}")
    print(f"  • Tổng số sections: {stats['total_sections']}")
    if stats['documents']:
        print(f"  • Danh sách tài liệu:")
        for doc in stats['documents']:
            print(f"    - {doc}")
    
    print("\n" + "=" * 70)
    print("💡 Bạn có thể sử dụng PageIndex để:")
    print("  1. Tìm kiếm thông tin trong tài liệu")
    print("  2. Lấy context cho LLM")
    print("  3. Xây dựng hệ thống RAG")
    print("=" * 70)
    
    # Interactive search
    print("\n🔍 Thử nghiệm tìm kiếm (gõ 'exit' để thoát):")
    
    while True:
        query = input("\n💬 Nhập câu hỏi: ").strip()
        
        if not query or query.lower() in ['exit', 'quit']:
            break
        
        context, sources = page_index.get_context(query, max_sections=3)
        
        if context is None:
            print("⚠️  Không tìm thấy thông tin liên quan")
            continue
        
        print("\n📚 Nguồn tìm thấy:")
        for idx, source in enumerate(sources, 1):
            print(f"  {idx}. {source}")
        
        print("\n📝 Context cho LLM:")
        print("-" * 70)
        prompt = format_context_for_prompt(query, context, sources)
        print(prompt)
        print("-" * 70)
    
    print("\n👋 Tạm biệt!")


if __name__ == "__main__":
    demo_pageindex()
