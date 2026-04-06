"""
RLM Hybrid - Hệ thống Recursive Language Model tối ưu
======================================================
Kết hợp điểm mạnh từ 4 file:
  - basic_rlm_chatgpt.py  → Hệ thống Action + Scratchpad
  - basic_rlm_gemini.py   → Map-Reduce chia-để-trị
  - rlm_idmodel_chatgpt.py → Agent loop thật với LLM
  - rlm_idmodel_gemini.py  → Đệ quy nhị phân có kiểm soát

Cải tiến:
  - JSON Action thay vì Code Generation (tránh lỗi syntax)
  - Tích hợp Llama 3.1 8B từ module có sẵn
  - 5 tools: SEARCH, READ, SUMMARIZE_CHUNKS, CALL_SUBTASK, FINAL
  - Giới hạn an toàn: max_steps=10, max_depth=3
"""

import re
import json
import math
import sys
import os

# Fix encoding cho Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Thêm thư mục gốc của project vào sys.path để import module LLM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_models.Llama_3_1_8B_Instruct_v2 import generate_response

# =====================================================================
# LOGGING UTILITY - Ghi log ra file UTF-8 (tránh lỗi encoding Windows)
# =====================================================================
LOG_FILE = os.path.join(os.path.dirname(__file__), "rlm_run.log")

def log(msg):
    """In ra console và ghi vào file log UTF-8."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode())
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Xóa file log cũ khi bắt đầu
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)


# =====================================================================
# 1️⃣ ENVIRONMENT — Quản lý tài liệu (từ basic_rlm_chatgpt.py)
# =====================================================================

class Environment:
    """Môi trường chứa tài liệu dài, cung cấp các tools cho LLM tương tác."""
    
    def __init__(self, document: str):
        self.document = document
        self.tokens = document.split()
        self.total_tokens = len(self.tokens)
    
    def search(self, keyword: str) -> list:
        """Tìm vị trí (character index) của keyword trong tài liệu."""
        matches = [m.start() for m in re.finditer(re.escape(keyword), self.document, re.IGNORECASE)]
        return matches[:10]  # Giới hạn 10 kết quả
    
    def read(self, start: int, length: int) -> str:
        """Đọc một đoạn tài liệu từ vị trí start, lấy length ký tự."""
        length = min(length, 2000)  # Giới hạn đọc tối đa 2000 ký tự
        return self.document[start:start + length]
    
    def get_metadata(self) -> dict:
        """Trả về thông tin tổng quan của tài liệu."""
        return {
            "total_characters": len(self.document),
            "total_tokens_approx": self.total_tokens,
            "preview": self.document[:300]
        }


# =====================================================================
# 2️⃣ MAP-REDUCE PROCESSOR — Xử lý tài liệu dài (từ basic_rlm_gemini.py
#    + rlm_idmodel_gemini.py)
# =====================================================================

class MapReduceProcessor:
    """
    Chia tài liệu thành nhiều chunks nhỏ, gọi LLM tóm tắt từng chunk,
    rồi tổng hợp kết quả (Map-Reduce pattern).
    """
    
    def __init__(self, chunk_size: int = 3000):
        self.chunk_size = chunk_size  # Số ký tự mỗi chunk
    
    def process(self, document: str, task: str) -> str:
        """Xử lý tài liệu dài bằng Map-Reduce."""
        
        # --- MAP PHASE: Chia nhỏ và xử lý từng chunk ---
        chunks = self._split_into_chunks(document)
        log(f"\n[Map-Reduce] Chia tai lieu thanh {len(chunks)} chunks")
        
        chunk_results = []
        for i, chunk in enumerate(chunks):
            log(f"  [Map] Dang xu ly chunk {i+1}/{len(chunks)}...")
            
            prompt = (
                f"Nhiệm vụ: {task}\n\n"
                f"Đoạn văn bản (chunk {i+1}/{len(chunks)}):\n"
                f"---\n{chunk}\n---\n\n"
                f"Hãy trích xuất thông tin quan trọng liên quan đến nhiệm vụ trên. "
                f"Trả lời ngắn gọn bằng tiếng Việt."
            )
            
            result = generate_response(
                user_input=prompt,
                system_prompt="Bạn là trợ lý AI chuyên phân tích văn bản. Trả lời ngắn gọn, chính xác.",
                max_new_tokens=300,
                temperature=0.1
            )
            chunk_results.append(f"[Chunk {i+1}]: {result}")
        
        # --- REDUCE PHASE: Tổng hợp kết quả ---
        log(f"  [Map-Reduce] Dang tong hop {len(chunk_results)} ket qua...")
        
        combined = "\n".join(chunk_results)
        aggregation_prompt = (
            f"Nhiệm vụ gốc: {task}\n\n"
            f"Dưới đây là kết quả phân tích từ {len(chunks)} đoạn văn bản khác nhau:\n"
            f"---\n{combined}\n---\n\n"
            f"Hãy tổng hợp và đưa ra câu trả lời cuối cùng. "
            f"Trả lời bằng tiếng Việt."
        )
        
        final_result = generate_response(
            user_input=aggregation_prompt,
            system_prompt="Bạn là trợ lý AI chuyên tổng hợp thông tin. Hãy tổng hợp chính xác và súc tích.",
            max_new_tokens=500,
            temperature=0.1
        )
        
        return final_result
    
    def _split_into_chunks(self, text: str) -> list:
        """Chia text thành các chunks có kích thước chunk_size."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks


# =====================================================================
# 3️⃣ RLM CONTROLLER — Trung tâm điều khiển Agent (kết hợp tất cả)
# =====================================================================

SYSTEM_PROMPT = """Bạn là một Recursive Language Model (RLM) Agent.
Bạn KHÔNG có quyền truy cập trực tiếp vào toàn bộ tài liệu.
Bạn phải sử dụng các công cụ (tools) để tương tác với tài liệu.

## Chiến lược ưu tiên (QUAN TRỌNG):

1. Với câu hỏi tìm kiếm thông tin cụ thể: SEARCH → READ → FINAL (3 bước)
2. Với câu hỏi tóm tắt/phân tích nội dung dài: SUMMARIZE_CHUNKS → FINAL (2 bước)
3. CHỈ dùng CALL_SUBTASK khi câu hỏi có NHIỀU phần hoàn toàn khác nhau cần tách ra.
4. Nếu preview đã chứa đủ thông tin → dùng FINAL ngay lập tức!
5. KHÔNG BAO GIỜ dùng CALL_SUBTASK hoặc SUMMARIZE_CHUNKS trong Agent con (subtask).

## Các công cụ có sẵn:

1. SEARCH — Tìm vị trí keyword trong tài liệu (trả về danh sách vị trí ký tự)
2. READ — Đọc một đoạn văn bản tại vị trí cụ thể (dùng sau SEARCH)
3. SUMMARIZE_CHUNKS — Tóm tắt toàn bộ tài liệu dài bằng Map-Reduce (chỉ dùng khi tài liệu > 5000 tokens)
4. CALL_SUBTASK — Tạo Agent con giải quyết câu hỏi phụ (hạn chế dùng)
5. FINAL — Đưa ra câu trả lời cuối cùng (dùng ngay khi có đủ thông tin)

## Quy tắc:

- Mỗi lượt, trả về ĐÚNG MỘT JSON object. KHÔNG viết gì khác ngoài JSON.
- Khi đã có đủ thông tin trong scratchpad → dùng FINAL ngay, đừng tìm thêm.
- Câu trả lời FINAL phải bằng tiếng Việt.

## Format JSON:

{"action": "SEARCH", "keyword": "từ khóa"}
{"action": "READ", "start": 0, "length": 500}
{"action": "SUMMARIZE_CHUNKS", "task": "mô tả nhiệm vụ"}
{"action": "CALL_SUBTASK", "question": "câu hỏi phụ"}
{"action": "FINAL", "answer": "câu trả lời cuối cùng bằng tiếng Việt"}
"""


class RLMController:
    """
    Trung tâm điều khiển RLM Agent.
    - Agent loop multi-step (từ rlm_idmodel_chatgpt.py)
    - Scratchpad memory (từ basic_rlm_chatgpt.py)
    - JSON action parsing (cải tiến, thay thế exec() code)
    - Đệ quy CALL_SUBTASK (từ rlm_idmodel_gemini.py)
    - Map-Reduce SUMMARIZE_CHUNKS (từ basic_rlm_gemini.py)
    """
    
    def __init__(self, env: Environment, max_steps: int = 6, max_depth: int = 2):
        self.env = env
        self.max_steps = max_steps
        self.max_depth = max_depth
        self.scratchpad = ""
        self.map_reduce = MapReduceProcessor(chunk_size=3000)
    
    def run(self, question: str, depth: int = 0) -> str:
        """Vòng lặp Agent chính: LLM suy luận → chọn action → thực thi → lặp lại."""
        
        indent = "  " * depth
        metadata = self.env.get_metadata()
        
        log(f"\n{indent}{'='*50}")
        log(f"{indent}[RLM Agent] depth={depth} | Cau hoi: {question}")
        log(f"{indent}[Doc] {metadata['total_tokens_approx']} tokens")
        log(f"{indent}{'='*50}")
        
        for step in range(self.max_steps):
            log(f"\n{indent}--- Buoc {step + 1}/{self.max_steps} ---")
            
            # Xây dựng prompt cho LLM
            user_prompt = self._build_user_prompt(question, metadata)
            
            # Gọi LLM thật
            llm_response = generate_response(
                user_input=user_prompt,
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=300,
                temperature=0.1
            )
            
            log(f"{indent}[LLM] -> {llm_response[:200]}")
            
            # Parse JSON action từ output của LLM
            action = self._parse_action(llm_response)
            
            if action is None:
                log(f"{indent}[WARN] Khong the parse JSON action. Thu lai...")
                self.scratchpad += f"\n[Loi] Buoc {step+1}: Output khong dung format JSON. Hay tra ve DUNG JSON."
                continue
            
            # Thực thi action
            action_type = action.get("action", "").upper()
            log(f"{indent}[Action] {action_type}")
            
            # ---- SEARCH ----
            if action_type == "SEARCH":
                keyword = action.get("keyword", "")
                # LLM đôi khi trả về list keyword thay vì string
                if isinstance(keyword, list):
                    keyword = keyword[0] if keyword else ""
                keyword = str(keyword)
                results = self.env.search(keyword)
                feedback = f"SEARCH('{keyword}') -> Tim thay {len(results)} vi tri: {results}"
                log(f"{indent}  [SEARCH] {feedback}")
                self.scratchpad += f"\n[Buoc {step+1}] {feedback}"
            
            # ---- READ ----
            elif action_type == "READ":
                start = int(action.get("start", 0))
                length = int(action.get("length", 500))
                text = self.env.read(start, length)
                feedback = f"READ({start}, {length}) -> \"{text[:150]}...\""
                log(f"{indent}  [READ] {feedback}")
                self.scratchpad += f"\n[Buoc {step+1}] READ({start}, {length}) -> \"{text}\""
            
            # ---- SUMMARIZE_CHUNKS ----
            elif action_type == "SUMMARIZE_CHUNKS":
                task_desc = action.get("task", question)
                log(f"{indent}  [SUMMARIZE_CHUNKS] Bat dau Map-Reduce cho: {task_desc}")
                summary = self.map_reduce.process(self.env.document, task_desc)
                feedback = f"SUMMARIZE_CHUNKS -> {summary[:200]}..."
                log(f"{indent}  [SUMMARIZE_CHUNKS] {feedback}")
                self.scratchpad += f"\n[Buoc {step+1}] SUMMARIZE_CHUNKS -> {summary}"
            
            # ---- CALL_SUBTASK ----
            elif action_type == "CALL_SUBTASK":
                sub_question = action.get("question", "")
                if depth >= self.max_depth:
                    feedback = f"CALL_SUBTASK bi tu choi: da dat do sau toi da ({self.max_depth})"
                    log(f"{indent}  [BLOCKED] {feedback}")
                    self.scratchpad += f"\n[Buoc {step+1}] {feedback}. Hay dung FINAL de tra loi."
                else:
                    log(f"{indent}  [SUBTASK] Goi RLM con cho: {sub_question}")
                    sub_rlm = RLMController(self.env, max_steps=3, max_depth=self.max_depth)
                    sub_answer = sub_rlm.run(sub_question, depth=depth + 1)
                    feedback = f"CALL_SUBTASK('{sub_question}') -> {sub_answer}"
                    log(f"{indent}  [SUBTASK] {feedback}")
                    self.scratchpad += f"\n[Buoc {step+1}] {feedback}"
            
            # ---- FINAL ----
            elif action_type == "FINAL":
                answer = action.get("answer", "Khong co cau tra loi.")
                log(f"{indent}  [FINAL] {answer}")
                return answer
            
            else:
                log(f"{indent}  [UNKNOWN] Action khong hop le: {action_type}")
                self.scratchpad += f"\n[Buoc {step+1}] Action '{action_type}' khong hop le. Dung: SEARCH, READ, SUMMARIZE_CHUNKS, CALL_SUBTASK, FINAL"
        
        # Fallback: nếu hết bước mà chưa FINAL, cố gắng tổng hợp từ scratchpad
        if self.scratchpad.strip():
            log(f"{indent}[AUTO-FINAL] Het buoc, tu dong tong hop tu scratchpad...")
            fallback_prompt = (
                f"Dua tren thong tin da thu thap:\n{self.scratchpad}\n\n"
                f"Hay tra loi cau hoi: {question}\n"
                f"Tra loi ngan gon bang tieng Viet."
            )
            fallback_answer = generate_response(
                user_input=fallback_prompt,
                system_prompt="Ban la tro ly AI. Tra loi chinh xac, ngan gon bang tieng Viet.",
                max_new_tokens=200,
                temperature=0.1
            )
            log(f"{indent}[AUTO-FINAL] {fallback_answer}")
            return fallback_answer
        
        return f"Da dat gioi han {self.max_steps} buoc ma chua co FINAL."
    
    def _build_user_prompt(self, question: str, metadata: dict) -> str:
        """Xây dựng prompt gửi cho LLM mỗi bước."""
        remaining = self.max_steps - len([l for l in self.scratchpad.split('\n') if l.startswith('[Buoc')])
        prompt = f"""## Thong tin tai lieu:
- Tong do dai: {metadata['total_characters']} ky tu (~{metadata['total_tokens_approx']} tokens)
- Preview: "{metadata['preview'][:200]}..."

## Scratchpad (lich su cac buoc da thuc hien):
{self.scratchpad if self.scratchpad else "(Chua co buoc nao)"}

## Cau hoi can tra loi:
{question}

## LUU Y: Ban con khoang {remaining} buoc. Neu scratchpad da co du thong tin, hay dung FINAL ngay!

Hay tra ve DUNG MOT JSON object cho action tiep theo. KHONG viet gi khac ngoai JSON."""
        return prompt
    
    def _parse_action(self, llm_output: str) -> dict:
        """Trích xuất JSON action từ output của LLM."""
        # Thử parse trực tiếp
        try:
            return json.loads(llm_output.strip())
        except json.JSONDecodeError:
            pass
        
        # Tìm JSON object trong output (LLM có thể thêm text xung quanh)
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, llm_output)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if "action" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # Fallback: tìm pattern cũ style (SEARCH, READ, FINAL)
        if "FINAL" in llm_output:
            # Cố extract answer
            final_match = re.search(r"FINAL\(['\"](.+?)['\"]\)", llm_output)
            if final_match:
                return {"action": "FINAL", "answer": final_match.group(1)}
        
        return None


# =====================================================================
# 4️⃣ THỰC THI MÔ PHỎNG
# =====================================================================

if __name__ == "__main__":
    # Tài liệu mẫu
    long_document = """
Albert Einstein (14 tháng 3, 1879 – 18 tháng 4, 1955) là một nhà vật lý lý thuyết người Đức, 
được coi là một trong những nhà khoa học vĩ đại nhất mọi thời đại. Ông sinh ra tại thành phố 
Ulm, Vương quốc Württemberg, Đế quốc Đức. Gia đình ông chuyển tới Munich khi ông mới 1 tuổi.

Einstein nổi tiếng nhất với Thuyết tương đối hẹp (1905) và Thuyết tương đối rộng (1915). 
Thuyết tương đối hẹp giới thiệu phương trình nổi tiếng E=mc², cho thấy mối quan hệ giữa 
khối lượng và năng lượng. Thuyết tương đối rộng mở rộng lý thuyết này sang trường hấp dẫn, 
mô tả hấp dẫn như sự cong vênh của không-thời gian.

Năm 1921, Einstein nhận giải Nobel Vật lý cho công trình nghiên cứu hiệu ứng quang điện. 
Ông di cư sang Hoa Kỳ vào năm 1933 khi Adolf Hitler lên nắm quyền ở Đức. Ông làm việc 
tại Viện Nghiên cứu Cao cấp Princeton cho đến cuối đời.

Einstein qua đời ngày 18 tháng 4 năm 1955 tại Princeton, New Jersey, Hoa Kỳ, hưởng thọ 76 tuổi.
Tro cốt của ông được rải theo ý nguyện tại một địa điểm không được công bố.
"""

    log("KHOI DONG HE THONG RLM HYBRID")
    log("=" * 60)
    
    # Khoi tao
    env = Environment(long_document)
    rlm = RLMController(env, max_steps=6, max_depth=2)
    
    # Chay cau hoi
    question = "Einstein sinh ra o dau va mat nam bao nhieu tuoi?"
    answer = rlm.run(question)
    
    log("\n" + "=" * 60)
    log(f"KET QUA CUOI CUNG: {answer}")
    log("=" * 60)
    log(f"\n>> Xem chi tiet tai: {LOG_FILE}")
