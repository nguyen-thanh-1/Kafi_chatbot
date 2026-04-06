import json
import math

class ID_Model_Interface:
    """Mô phỏng giao diện của một mô hình Instruction-Driven (như Qwen3-RLM)"""
    def __init__(self, model_name="RLM-Qwen3-8B"):
        self.model_name = model_name

    def chat(self, system_prompt, user_prompt):
        # Trong thực tế, đây sẽ là lời gọi API tới LLM
        # LLM sẽ trả về kết quả dưới dạng văn bản hoặc cấu trúc JSON để thực thi code
        print(f"\n[Model {self.model_name} is thinking...]")
        return "Mô hình đã xử lý logic và trả về instruction tiếp theo."

class LongContextEnvironment:
    """Môi trường quản lý dữ liệu cực lớn (Virtual Context Space)"""
    def __init__(self, document_content):
        self.document = document_content
        self.tokens = document_content.split() # Giả lập tokenization
        self.total_size = len(self.tokens)

    def read_segment(self, start, end):
        return " ".join(self.tokens[start:end])

class RLMAgent:
    def __init__(self, model_interface):
        self.llm = model_interface
        self.memory = [] # Lưu trữ các kết quả trung gian

    def solve(self, env, task, depth=0):
        """Hàm đệ quy chính điều khiển luồng suy luận"""
        indent = "  " * depth
        print(f"{indent}> Đang xử lý ở độ sâu đệ quy: {depth}")

        # Bước 1: PEAK - Mô hình 'liếc' qua cấu trúc để lập kế hoạch
        sample = env.read_segment(0, 1000)
        
        # Bước 2: REASONING - Mô hình quyết định có cần đệ quy không
        # Nếu task quá phức tạp hoặc dữ liệu quá dài (> 5000 tokens)
        if env.total_size > 5000 and depth < 2:
            print(f"{indent}[ID Mode] Phát hiện dữ liệu lớn ({env.total_size} tokens). Khởi tạo chiến thuật đệ quy...")
            
            # Chia nhỏ (Decomposition)
            mid_point = env.total_size // 2
            
            # Gọi đệ quy cho nửa đầu
            env_left = LongContextEnvironment(" ".join(env.tokens[:mid_point]))
            res_left = self.solve(env_left, f"Phân tích phần 1 của: {task}", depth + 1)
            
            # Gọi đệ quy cho nửa sau
            env_right = LongContextEnvironment(" ".join(env.tokens[mid_point:]))
            res_right = self.solve(env_right, f"Phân tích phần 2 của: {task}", depth + 1)
            
            # Bước 3: AGGREGATION - Tổng hợp kết quả từ các nhánh
            return self.aggregate(res_left, res_right, task)
        
        else:
            # Base Case: Đọc trực tiếp và trả lời nếu dữ liệu đã đủ nhỏ
            print(f"{indent}[ID Mode] Dữ liệu trong tầm kiểm soát. Đang trích xuất câu trả lời...")
            return f"Kết quả từ mảnh dữ liệu {env.total_size} tokens"

    def aggregate(self, part1, part2, original_task):
        """Hàm tổng hợp kết quả (Reduce phase trong MapReduce)"""
        return f"TỔNG HỢP: {part1} + {part2} -> Đáp án cho: {original_task}"

# --- THỰC THI MÔ PHỎNG ---

# 1. Khởi tạo tài liệu giả lập (12.000 tokens)
massive_doc = "Dữ liệu nghiên cứu khoa học... " * 3000
env = LongContextEnvironment(massive_doc)

# 2. Khởi tạo Model và Agent
id_model = ID_Model_Interface()
agent = RLMAgent(id_model)

# 3. Chạy tác vụ
final_result = agent.solve(env, "Tìm mối liên hệ giữa các thí nghiệm trong tài liệu")

print("\n" + "="*50)
print("KẾT QUẢ CUỐI CÙNG:")
print(final_result)