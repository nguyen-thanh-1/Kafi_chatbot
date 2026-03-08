import math

class RLMEnvironment:
    """Môi trường chứa dữ liệu cực dài (ví dụ: 1 triệu token)"""
    def __init__(self, full_text):
        self.data = full_text
        self.total_tokens = len(full_text.split()) # Giả định đơn giản

    def peek(self, start, end):
        """Lấy một đoạn nhỏ dữ liệu để 'liếc' qua nội dung"""
        words = self.data.split()
        return " ".join(words[start:end])

    def completion(self, sub_problem, data_slice):
        """
        Đây là hàm đệ quy: LLM gọi chính nó trên một phần dữ liệu nhỏ hơn.
        Trong thực tế, đây là một lời gọi API tới mô hình RLM-Qwen3.
        """
        # Giả lập logic của LLM con xử lý một phần dữ liệu
        print(f"-> Đang xử lý đoạn: {data_slice[0]} đến {data_slice[1]}")
        return f"[Kết quả tóm tắt cho {sub_problem} tại {data_slice}]"

# --- LUỒNG HOẠT ĐỘNG CỦA RLM ---

def rlm_main_logic(env, query):
    """
    Mô phỏng cách LLM 'gốc' phân tích và ra lệnh đệ quy
    """
    threshold = 10000  # Ngưỡng mà LLM có thể đọc trực tiếp (Context Window)
    
    if env.total_tokens <= threshold:
        # Nếu dữ liệu đủ nhỏ, đọc và trả lời trực tiếp
        return "Trả lời trực tiếp dựa trên toàn bộ context."
    
    else:
        # 1. PHÂN TÍCH (Analysis Phase)
        # LLM sẽ 'peek' một vài đoạn để hiểu cấu trúc văn bản
        preview = env.peek(0, 500)
        print(f"LLM đang liếc qua 500 token đầu tiên: {preview[:100]}...")

        # 2. PHÂN RÃ (Decomposition Phase)
        # LLM tự quyết định chia nhỏ dữ liệu thành các phần (ví dụ: 4 phần)
        num_chunks = 4
        chunk_size = math.ceil(env.total_tokens / num_chunks)
        
        results = []
        # 3. GỌI ĐỆ QUY SONG SONG (Parallel Recursion)
        # Tương đương với hàm llm_batch trong bài báo
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, env.total_tokens)
            
            # LLM gọi chính nó để xử lý từng mảnh
            res = env.completion(sub_problem="Tìm thông tin quan trọng liên quan đến " + query, 
                                 data_slice=(start, end))
            results.append(res)
        
        # 4. TỔNG HỢP (Aggregation Phase)
        # LLM gốc nhận các kết quả trung gian và đưa ra câu trả lời cuối cùng
        final_answer = f"Tổng hợp từ {len(results)} kết quả con: " + " ".join(results)
        return final_answer

# Giả lập sử dụng
big_data = "Nội dung cực dài lên tới hàng triệu token..." * 10000 
env = RLMEnvironment(big_data)
answer = rlm_main_logic(env, "Tóm tắt các ý chính của tài liệu")
print("\n--- KẾT QUẢ CUỐI CÙNG ---")
print(answer)