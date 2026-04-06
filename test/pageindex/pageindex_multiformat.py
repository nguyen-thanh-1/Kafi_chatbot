"""
PageIndex RAG System - Multi-Format Support
Hỗ trợ: TXT, PDF, DOCX, MD
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai không được cài đặt. Chạy: uv pip install google-generativeai")

# Import PDF reader
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  pypdf không được cài đặt. Chạy: uv pip install pypdf")

# Import DOCX reader
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️  python-docx không được cài đặt. Chạy: uv pip install python-docx")

# Import PageIndex core
from pageindex_core import format_context_for_prompt


# ==================== Document Readers ====================
class DocumentReader:
    """Base class cho document readers"""
    
    @staticmethod
    def read_txt(file_path: Path) -> str:
        """Đọc file TXT"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def read_pdf(file_path: Path) -> str:
        """Đọc file PDF"""
        if not PDF_SUPPORT:
            raise ImportError("pypdf chưa được cài đặt")
        
        reader = PdfReader(str(file_path))
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text.strip():
                text_parts.append(f"[Trang {page_num}]\n{text}")
        
        return "\n\n".join(text_parts)
    
    @staticmethod
    def read_docx(file_path: Path) -> str:
        """Đọc file DOCX"""
        if not DOCX_SUPPORT:
            raise ImportError("python-docx chưa được cài đặt")
        
        doc = Document(str(file_path))
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        return "\n\n".join(paragraphs)
    
    @staticmethod
    def read_md(file_path: Path) -> str:
        """Đọc file Markdown"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @classmethod
    def read_file(cls, file_path: Path) -> Tuple[str, str]:
        """
        Đọc file dựa trên extension
        
        Returns:
            (content, file_type)
        """
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.txt':
                return cls.read_txt(file_path), 'txt'
            elif ext == '.pdf':
                return cls.read_pdf(file_path), 'pdf'
            elif ext == '.docx':
                return cls.read_docx(file_path), 'docx'
            elif ext in ['.md', '.markdown']:
                return cls.read_md(file_path), 'md'
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {ext}")
        except Exception as e:
            raise Exception(f"Lỗi đọc file {file_path}: {e}")


# ==================== Multi-Format PageIndex ====================
class MultiFormatPageIndex:
    """
    PageIndex hỗ trợ nhiều định dạng file
    Hỗ trợ: TXT, PDF, DOCX, MD
    """
    
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.markdown']
    
    def __init__(self, documents_dir="./courses"):
        self.documents_dir = Path(documents_dir)
        self.index = {}
        self.documents = {}
        self.reader = DocumentReader()
        
    def build_index(self):
        """Xây dựng index từ tất cả file được hỗ trợ"""
        print(f"\n📚 Đang xây dựng PageIndex từ {self.documents_dir}...")
        
        if not self.documents_dir.exists():
            print(f"⚠️  Tạo thư mục: {self.documents_dir}")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Tìm tất cả file được hỗ trợ
        all_files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            all_files.extend(self.documents_dir.glob(f"*{ext}"))
        
        if not all_files:
            print(f"⚠️  Không tìm thấy file nào trong {self.documents_dir}")
            print(f"💡 Định dạng hỗ trợ: {', '.join(self.SUPPORTED_EXTENSIONS)}")
            return
        
        # Index từng file
        for file_path in all_files:
            self._index_document(file_path)
        
        total_sections = sum(len(d['sections']) for d in self.documents.values())
        print(f"✅ Đã index {len(self.documents)} tài liệu với {total_sections} sections")
        
    def _index_document(self, file_path: Path):
        """Index một tài liệu với cấu trúc phân cấp"""
        try:
            # Đọc file
            content, file_type = self.reader.read_file(file_path)
            
            if not content.strip():
                print(f"⚠️  File rỗng: {file_path}")
                return
            
        except Exception as e:
            print(f"⚠️  Lỗi đọc file {file_path}: {e}")
            return
        
        doc_name = file_path.stem
        
        # Tách tài liệu thành sections
        sections = self._split_into_sections(content, file_type)
        
        # Tạo cấu trúc phân cấp
        doc_structure = {
            'name': doc_name,
            'path': str(file_path),
            'type': file_type,
            'sections': []
        }
        
        for idx, section in enumerate(sections):
            # Xác định tiêu đề
            title, content_text = self._extract_title_and_content(section, idx)
            
            doc_structure['sections'].append({
                'title': title,
                'content': content_text,
                'index': idx
            })
        
        self.documents[doc_name] = doc_structure
        self.index[doc_name] = {
            'sections': [s['title'] for s in doc_structure['sections']],
            'path': str(file_path),
            'type': file_type
        }
        
        print(f"  📄 {doc_name} ({file_type.upper()}): {len(doc_structure['sections'])} sections")
    
    def _split_into_sections(self, content: str, file_type: str) -> List[str]:
        """Tách nội dung thành sections dựa trên file type"""
        
        if file_type == 'md':
            # Markdown: tách theo headers
            sections = []
            current_section = []
            
            for line in content.split('\n'):
                if line.startswith('#'):  # Header
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section.append(line)
            
            if current_section:
                sections.append('\n'.join(current_section))
            
            return [s.strip() for s in sections if s.strip()]
        
        elif file_type == 'pdf':
            # PDF: tách theo trang hoặc đoạn văn
            # Nếu có marker [Trang X], tách theo đó
            if '[Trang' in content:
                import re
                sections = re.split(r'\[Trang \d+\]', content)
                return [s.strip() for s in sections if s.strip()]
            else:
                # Fallback: tách theo đoạn văn
                return [s.strip() for s in content.split('\n\n') if s.strip()]
        
        else:
            # TXT, DOCX: tách theo đoạn văn
            return [s.strip() for s in content.split('\n\n') if s.strip()]
    
    def _extract_title_and_content(self, section: str, idx: int) -> Tuple[str, str]:
        """Trích xuất title và content từ section"""
        lines = section.split('\n')
        
        if not lines:
            return f"Phần {idx + 1}", section
        
        first_line = lines[0].strip()
        
        # Markdown header
        if first_line.startswith('#'):
            title = first_line.lstrip('#').strip()
            content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else section
            return title, content
        
        # Tiêu đề kết thúc bằng :
        if first_line.endswith(':'):
            title = first_line.rstrip(':').strip()
            content = '\n'.join(lines[1:]).strip() if len(lines) > 1 else section
            return title, content
        
        # Dòng đầu ngắn (có thể là tiêu đề)
        if len(first_line) < 100 and len(lines) > 1:
            return first_line, '\n'.join(lines[1:]).strip()
        
        # Fallback: lấy 50 ký tự đầu làm tiêu đề
        title = first_line[:50] + "..." if len(first_line) > 50 else first_line
        return title, section
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Tìm kiếm sections liên quan"""
        if not self.documents:
            return []
        
        relevant_sections = []
        
        for doc_name, doc_data in self.documents.items():
            for section in doc_data['sections']:
                score = self._calculate_relevance(query, section['title'], section['content'])
                
                if score > 0:
                    relevant_sections.append({
                        'document': doc_name,
                        'title': section['title'],
                        'content': section['content'],
                        'type': doc_data['type'],
                        'score': score
                    })
        
        relevant_sections.sort(key=lambda x: x['score'], reverse=True)
        return relevant_sections[:top_k]
    
    def _calculate_relevance(self, query: str, title: str, content: str) -> float:
        """Tính điểm liên quan"""
        query_lower = query.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        score = 0
        query_terms = [term for term in query_lower.split() if len(term) > 2]
        
        for term in query_terms:
            if term in title_lower:
                score += 5
            if term in content_lower:
                score += 1
        
        # Bonus cho nhiều từ khóa match
        matching_terms = sum(1 for term in query_terms if term in content_lower)
        if matching_terms > len(query_terms) * 0.5:
            score += 3
        
        return score
    
    def get_context(self, query: str, max_sections: int = 3) -> Tuple[Optional[str], List[str]]:
        """Lấy context cho LLM"""
        sections = self.search(query, top_k=max_sections)
        
        if not sections:
            return None, []
        
        context_parts = []
        sources = []
        
        for idx, section in enumerate(sections, 1):
            context_parts.append(
                f"[Nguồn {idx}: {section['document']} ({section['type'].upper()}) - {section['title']}]\n{section['content']}"
            )
            sources.append(f"{section['document']} ({section['type'].upper()}) - {section['title']}")
        
        context = "\n\n" + ("-" * 60 + "\n\n").join(context_parts)
        return context, sources
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê"""
        stats = {
            'total_documents': len(self.documents),
            'total_sections': sum(len(d['sections']) for d in self.documents.values()),
            'documents': [],
            'by_type': {}
        }
        
        for doc_name, doc_data in self.documents.items():
            stats['documents'].append({
                'name': doc_name,
                'type': doc_data['type'],
                'sections': len(doc_data['sections'])
            })
            
            # Count by type
            file_type = doc_data['type']
            if file_type not in stats['by_type']:
                stats['by_type'][file_type] = 0
            stats['by_type'][file_type] += 1
        
        return stats


# ==================== LLM Wrapper ====================
class GeminiLLM:
    """Wrapper cho Gemini 2.5 Pro API"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai chưa được cài đặt")
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key không tìm thấy. "
                "Vui lòng set GEMINI_API_KEY environment variable hoặc truyền api_key parameter."
            )
        
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Khởi tạo Gemini API client"""
        print(f"🔄 Đang kết nối Gemini API ({self.model_name})...")
        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Gemini API sẵn sàng!")
        except Exception as e:
            raise RuntimeError(f"Không thể kết nối Gemini API: {e}")
    
    def chat(self, messages: List[Dict], max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        """Chat với Gemini API"""
        try:
            # Convert messages to Gemini format
            # Gemini expects: system instruction + user/model messages
            system_instruction = None
            chat_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_instruction = msg["content"]
                elif msg["role"] == "user":
                    chat_messages.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    chat_messages.append({"role": "model", "parts": [msg["content"]]})
            
            # Create chat with system instruction if available
            if system_instruction:
                model = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
            else:
                model = self.model
            
            # Generate response
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            )
            
            # If we have chat history, use chat mode
            if len(chat_messages) > 1:
                chat = model.start_chat(history=chat_messages[:-1])
                response = chat.send_message(
                    chat_messages[-1]["parts"][0],
                    generation_config=generation_config
                )
            else:
                # Single message
                response = model.generate_content(
                    chat_messages[0]["parts"][0],
                    generation_config=generation_config
                )
            
            return response.text.strip()
            
        except Exception as e:
            raise RuntimeError(f"Lỗi Gemini API: {e}")


# ==================== Multi-Format RAG System ====================
class MultiFormatRAG:
    """RAG system hỗ trợ nhiều định dạng file"""
    
    def __init__(self, documents_dir="./courses", api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.page_index = MultiFormatPageIndex(documents_dir)
        self.page_index.build_index()
        
        self.llm = GeminiLLM(api_key=api_key, model_name=model_name)
        
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
    
    def query(self, question: str, max_new_tokens: int = 512, temperature: float = 0.2) -> Tuple[str, List[str]]:
        """Truy vấn hệ thống RAG"""
        context, sources = self.page_index.get_context(question, max_sections=3)
        
        if context is None:
            return "⚠️ Không tìm thấy thông tin liên quan trong tài liệu.", []
        
        user_prompt = format_context_for_prompt(question, context, sources)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.llm.chat(messages, max_new_tokens, temperature)
            return response, sources
        except Exception as e:
            return f"❌ Lỗi: {e}", []
    
    def rebuild_index(self):
        """Rebuild index"""
        print("\n🔄 Đang rebuild index...")
        self.page_index = MultiFormatPageIndex(self.page_index.documents_dir)
        self.page_index.build_index()
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê"""
        return self.page_index.get_statistics()


# ==================== Main ====================
def main():
    print("=" * 70)
    print("🚀 PageIndex Multi-Format RAG System (Gemini API)")
    print("=" * 70)
    print("\n📌 Hỗ trợ định dạng:")
    print("  ✅ TXT - Text files")
    print("  ✅ PDF - PDF documents")
    print("  ✅ DOCX - Word documents")
    print("  ✅ MD - Markdown files")
    print("\n🤖 LLM: Gemini 2.0 Flash Exp (API)")
    print("=" * 70)
    
    try:
        rag = MultiFormatRAG(api_key="Your_api_key", documents_dir="./courses", model_name="gemini-2.5-flash")
    except Exception as e:
        print(f"\n❌ Lỗi khởi tạo: {e}")
        return
    
    stats = rag.get_statistics()
    print(f"\n📊 Thống kê:")
    print(f"  • Tổng tài liệu: {stats['total_documents']}")
    print(f"  • Tổng sections: {stats['total_sections']}")
    print(f"  • Theo loại:")
    for file_type, count in stats['by_type'].items():
        print(f"    - {file_type.upper()}: {count} files")
    
    print("\n✅ Hệ thống sẵn sàng!")
    print("\n📝 Lệnh: rebuild | stats | exit")
    print("=" * 70)
    
    while True:
        print("\n")
        user_input = input("💬 Câu hỏi: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Tạm biệt!")
            break
        
        if user_input.lower() == "rebuild":
            rag.rebuild_index()
            stats = rag.get_statistics()
            print(f"✅ Rebuild xong! {stats['total_documents']} docs, {stats['total_sections']} sections")
            continue
        
        if user_input.lower() == "stats":
            stats = rag.get_statistics()
            print(f"\n📊 Thống kê chi tiết:")
            for doc in stats['documents']:
                print(f"  • {doc['name']} ({doc['type'].upper()}): {doc['sections']} sections")
            continue
        
        print("\n🤖 Đang xử lý...")
        print("=" * 70)
        
        try:
            response, sources = rag.query(user_input)
            print(f"\n📝 Trả lời:\n{response}")
            
            if sources:
                print(f"\n📚 Nguồn:")
                for idx, source in enumerate(sources, 1):
                    print(f"  {idx}. {source}")
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
        
        print("=" * 70)


if __name__ == "__main__":
    main()
