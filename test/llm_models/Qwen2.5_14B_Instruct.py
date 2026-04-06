import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch

class QwenChat:
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        print(f"Loading model: {model_name}...")
        
        # 1. Cấu hình Quantization (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. Tải Tokenizer và Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16
        )
        print("Model loaded successfully!")
        
        # QUAN TRỌNG: Tạo danh sách token không phải tiếng Việt để chặn
        print("Đang tạo danh sách chặn token ngoại ngữ (Trung, Nga, Nhật, Hàn...)...")
        self.bad_words_ids = self._get_non_vietnamese_bad_words()
        print(f"Đã chặn {len(self.bad_words_ids)} token ngoại ngữ!")
        
        # System prompt ngắn - cứng - hiệu quả
        self.system_prompt = """Bạn là trợ lý AI.
CHỈ được trả lời bằng tiếng Việt.
CẤM mọi ngôn ngữ khác.
Trả lời ngắn gọn, tự nhiên."""
        
        # Lịch sử hội thoại
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset_chat(self):
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("\n[Hệ thống] Đã xóa lịch sử trò chuyện.\n")

    def _get_non_vietnamese_bad_words(self):
        """Chặn token KHÔNG PHẢI tiếng Việt/Latin (Trung, Nga, Nhật, Hàn, Ả Rập...)"""
        bad_words = []
        
        def is_allowed_char(ch):
            """Chỉ cho phép: Latin cơ bản, Vietnamese diacritics, số, dấu câu thông dụng"""
            # ASCII printable (bao gồm Latin cơ bản, số, dấu câu)
            if ord(ch) < 128:
                return True
            # Vietnamese diacritics (Latin Extended)
            if '\u00c0' <= ch <= '\u01b0':  # À-ư
                return True
            if '\u1ea0' <= ch <= '\u1ef9':  # Vietnamese specific: Ạ-ỹ
                return True
            # Common punctuation và symbols
            if ch in '–—''""…•·×÷±≠≤≥':
                return True
            return False
        
        for i in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode([i])
            # Nếu có BẤT KỲ ký tự nào không được phép -> chặn token đó
            if any(not is_allowed_char(ch) for ch in token):
                bad_words.append([i])
        
        return bad_words

    def chat_stream(self, user_input):
        # Wrap input với format mạnh để ép tiếng Việt
        wrapped_input = (
            "Trả lời NGẮN GỌN, CHÍNH XÁC, "
            "CHỈ bằng TIẾNG VIỆT. "
            "TUYỆT ĐỐI KHÔNG dùng tiếng Trung.\n\n"
            f"Câu hỏi: {user_input}"
        )
        
        # Thêm câu hỏi của user vào lịch sử (lưu input gốc để hiển thị)
        self.history.append({"role": "user", "content": wrapped_input})

        # Format input theo template
        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Streamer để in ra màn hình ngay lập tức
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Cấu hình generate chuẩn - GREEDY CLEAN
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=1024, #512
            
            # GREEDY DECODING - không cần temperature/top_p/top_k
            do_sample=False,
            num_beams=1,
            
            repetition_penalty=1.2,
            bad_words_ids=self.bad_words_ids,  # CHẶN TOKEN NGOẠI NGỮ
            
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Chạy generation trong thread riêng
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\nAssistant: ", end="", flush=True)
        
        full_response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text
            
        print("\n") # Xuống dòng khi xong
        
        # Thêm câu trả lời vào lịch sử
        self.history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    qwen = QwenChat()
    
    print("="*50)
    print("Chào bạn! Tôi là trợ lý AI (Qwen 2.5 14B).")
    print("Gõ 'exit' để thoát, 'clear' để xóa lịch sử.")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "thoát"]:
                print("Tạm biệt!")
                break
            
            if user_input.lower() in ["clear", "reset", "xóa"]:
                qwen.reset_chat()
                continue
                
            qwen.chat_stream(user_input)
            
        except KeyboardInterrupt:
            print("\nĐã dừng cuộc trò chuyện.")
            break
        except Exception as e:
            print(f"\n[Lỗi] {e}")
