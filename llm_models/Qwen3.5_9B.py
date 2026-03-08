import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

from threading import Thread
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch
import warnings

warnings.filterwarnings('ignore')

class QwenChat:
    def __init__(self, model_name="Qwen/Qwen3.5-9B"):
        print(f"Loading model: {model_name}...")
        
        # Khắc phục lỗi 1: Out of Memory (OOM) bằng tính năng Quantization (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Cấu hình Tokenizer và Processor cho model đa phương thức
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        # Khắc phục lỗi 3: Đi lạc đề, lặp lại tiếng Trung
        self.system_prompt = """Bạn là một trợ lý AI thông minh đến từ Việt Nam.
NHIỆM VỤ: Trả lời câu hỏi của người dùng một cách chính xác, tự nhiên.
YÊU CẦU BẮT BUỘC:
1. LUÔN LUÔN trả lời bằng Tiếng Việt.
2. TUYỆT ĐỐI KHÔNG sử dụng tiếng Trung Quốc (Chinese) trong bất kỳ tình huống nào.
3. Nếu câu hỏi là ngôn ngữ khác, hãy dịch ý và trả lời lại bằng Tiếng Việt.
4. Không lặp lại câu hỏi của người dùng.
5. Trả lời ngắn gọn, súc tích, đi thẳng vào vấn đề.
6. TUYỆT ĐỐI KHÔNG sử dụng chế độ suy nghĩ (Thinking process) hoặc thẻ <think>. Trả lời trực tiếp ngay lập tức. """
        
        # Lọc các token ngoại ngữ để ép sinh Tiếng Việt
        print("Đang tạo danh sách chặn token ngoại ngữ (Trung, Nga, Nhật, Hàn...)...")
        self.bad_words_ids = self._get_non_vietnamese_bad_words()
        if self.bad_words_ids:
            print(f"Đã chặn {len(self.bad_words_ids)} token ngoại ngữ!")
        
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset_chat(self):
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("\n[Hệ thống] Đã xóa lịch sử trò chuyện.\n")

    def _get_non_vietnamese_bad_words(self):
        """Chặn token KHÔNG PHẢI tiếng Việt/Latin để khắc phục tuyệt đối việc tạo ra các ngôn ngữ không mong muốn"""
        bad_words = []
        if not hasattr(self.tokenizer, 'vocab_size'):
            return None
            
        def is_allowed_char(ch):
            if ord(ch) < 128:
                return True
            if '\u00c0' <= ch <= '\u01b0' or '\u1ea0' <= ch <= '\u1ef9':
                return True
            if ch in '–—''""…•·×÷±≠≤≥':
                return True
            return False
            
        try:
            for i in range(self.tokenizer.vocab_size):
                token = self.tokenizer.decode([i])
                if any(not is_allowed_char(ch) for ch in token):
                    bad_words.append([i])
            return bad_words
        except:
            return None

    def chat_stream(self, user_input):
        # Format input mạnh mẽ nhằm ép tuân thủ Tiếng Việt và CHẶN SUY NGHĨ (Thinking)
        wrapped_input = (
            "TRẢ LỜI TRỰC TIẾP, KHÔNG SUY NGHĨ (NO THINKING). "
            "CHỈ bằng TIẾNG VIỆT. "
            "TUYỆT ĐỐI KHÔNG dùng tiếng Trung hoặc thẻ <think>.\n\n"
            f"Câu hỏi: {user_input}"
        )
        
        self.history.append({"role": "user", "content": wrapped_input})

        # Format template
        if hasattr(self, "processor") and self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                text = self.processor.apply_chat_template(
                    self.history,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                text = f"System: {self.system_prompt}\nUser: {wrapped_input}\nAssistant:"
        else:
            try:
                text = self.tokenizer.apply_chat_template(
                    self.history,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                text = f"System: {self.system_prompt}\nUser: {wrapped_input}\nAssistant:"
            
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Khắc phục lỗi 4: Lặp từ và tắt Thinking Mode bằng Sampling chuẩn (theo README Qwen3.5)
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            
            # Chế độ Instruct thuần túy (không suy nghĩ)
            do_sample=True,         # README khuyến nghị sampling cho non-thinking mode
            temperature=0.7,        # Chuẩn non-thinking
            top_p=0.8,              # Chuẩn non-thinking
            top_k=20,               # Chuẩn non-thinking
            repetition_penalty=1.1,
            
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        # Add eos_token_id conditionally due to fallback logic above
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            
        if self.bad_words_ids:
            generation_kwargs['bad_words_ids'] = self.bad_words_ids

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\nAssistant: ", end="", flush=True)
        
        full_response = ""
        is_thinking = False
        
        for new_text in streamer:
            chunk = new_text
            
            # Xử lý đoạn text suy luận <think> ... </think> của dòng Qwen3.5
            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "")
            
            if "</think>" in chunk:
                is_thinking = False
                chunk = chunk.replace("</think>", "")
                
            # Chỉ in ra phần nội dung thật (không in thinking process)
            if not is_thinking and chunk.strip() != "":
                print(chunk, end="", flush=True)
                full_response += chunk
            
        print("\n") # Xuống dòng khi sinh xong
        
        # Lưu câu hỏi gốc vào lịch sử để không bị dư "TUYỆT ĐỐI KHÔNG..." ở các câu sau
        self.history[-1]["content"] = user_input
        self.history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    qwen = QwenChat()
    
    print("="*60)
    print("Chào bạn! Tôi là trợ lý AI (Qwen3.5-9B).")
    print("Gõ 'exit' để thoát, 'clear' để xóa lịch sử.")
    print("Ghi chú: Lịch xử lý ngầm (Thinking) đã được ẩn để xuất câu trả lời ngắn gọn nhất.")
    print("="*60 + "\n")

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
