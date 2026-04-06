# Test PDF support cho PageIndex

from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io

def create_sample_pdf(output_path="./courses/sample_document.pdf"):
    """Tạo một file PDF mẫu với nội dung tiếng Việt"""
    
    # Tạo PDF writer
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    # Nội dung mẫu
    content = """
    Machine Learning và Ứng dụng
    
    Machine Learning là một nhánh của trí tuệ nhân tạo (AI) cho phép máy tính
    học từ dữ liệu mà không cần được lập trình cụ thể. Hệ thống ML có khả năng
    tự động cải thiện hiệu suất thông qua kinh nghiệm.
    
    Các loại Machine Learning:
    
    1. Supervised Learning (Học có giám sát)
    - Huấn luyện với dữ liệu được gán nhãn
    - Ví dụ: Phân loại email spam, dự đoán giá nhà
    
    2. Unsupervised Learning (Học không giám sát)
    - Tìm patterns trong dữ liệu không có nhãn
    - Ví dụ: Phân cụm khách hàng, giảm chiều dữ liệu
    
    3. Reinforcement Learning (Học tăng cường)
    - Học thông qua thử và sai với rewards
    - Ví dụ: Game AI, robot tự động
    """
    
    # Vẽ text (đơn giản, không dùng font tiếng Việt)
    y_position = 750
    for line in content.strip().split('\n'):
        line = line.strip()
        if line:
            can.drawString(50, y_position, line)
            y_position -= 20
            if y_position < 50:  # New page if needed
                can.showPage()
                y_position = 750
    
    can.save()
    
    # Lưu PDF
    packet.seek(0)
    with open(output_path, 'wb') as f:
        f.write(packet.getvalue())
    
    print(f"✅ Đã tạo file PDF mẫu: {output_path}")
    return output_path


def test_pdf_reading(pdf_path):
    """Test đọc PDF"""
    print(f"\n📖 Đang đọc PDF: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    print(f"📄 Số trang: {len(reader.pages)}")
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        print(f"\n--- Trang {page_num} ---")
        print(text[:200] + "..." if len(text) > 200 else text)


if __name__ == "__main__":
    import os
    
    # Tạo thư mục nếu chưa có
    os.makedirs("./courses", exist_ok=True)
    
    # Tạo PDF mẫu
    pdf_path = create_sample_pdf()
    
    # Test đọc PDF
    test_pdf_reading(pdf_path)
    
    print("\n✅ Test hoàn tất!")
    print("💡 Bây giờ bạn có thể chạy: python pageindex_multiformat.py")
