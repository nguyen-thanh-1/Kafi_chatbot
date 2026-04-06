import sys
import codecs

from threading import Thread
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch
import warnings

warnings.filterwarnings('ignore')

class QwenVLChat:
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct"):
        print(f"Loading model: {model_name}...")
        
        # Khắc phục lỗi 1: Out of Memory (OOM) bằng tính năng Quantization (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Khắc phục lỗi 2: Tokenizer và Model loading (Hỗ trợ Vision-Language module)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.tokenizer = self.processor.tokenizer
        except Exception:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        # Khắc phục lỗi 3: Đi lạc đề, ảo giác ngôn ngữ (sinh Tiếng Trung)
        self.system_prompt = """Bạn là một trợ lý AI thông minh, đa phương thức (Vision-Language) đến từ Việt Nam.
NHIỆM VỤ: Trả lời câu hỏi của người dùng một cách chính xác, tự nhiên.
YÊU CẦU BẮT BUỘC:
1. LUÔN LUÔN trả lời bằng Tiếng Việt.
2. TUYỆT ĐỐI KHÔNG sử dụng tiếng Trung Quốc (Chinese) trong bất kỳ tình huống nào.
3. Nếu câu hỏi là ngôn ngữ khác, hãy dịch ý và trả lời lại bằng Tiếng Việt.
4. Không lặp lại câu hỏi của người dùng.
5. Trả lời ngắn gọn, súc tích, đi thẳng vào vấn đề."""
        
        # Khắc phục lỗi cực mạnh (như trong Qwen2.5 14B): Lọc các token ngoại ngữ để ép sinh Tiếng Việt
        print("Đang tạo danh sách chặn token ngoại ngữ (Trung, Nga, Nhật, Hàn...)...")
        self.bad_words_ids = self._get_non_vietnamese_bad_words()
        if self.bad_words_ids:
            print(f"Đã chặn {len(self.bad_words_ids)} token ngoại ngữ!")
        
        self.history = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        ]

    def reset_chat(self):
        self.history = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
        ]
        print("\n[Hệ thống] Đã xóa lịch sử trò chuyện.\n")

    def _get_non_vietnamese_bad_words(self):
        """Chặn token KHÔNG PHẢI tiếng Việt/Latin để khắc phục tuyệt đối việc tạo ra các ngôn ngữ không mong muốn"""
        bad_words = []
        if not hasattr(self.tokenizer, 'vocab_size'):
            return None
            
        def is_allowed_char(ch):
            # Cấp phép: ASCII, Latin Extended (Tiếng Việt), Punctuation
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

    def chat_stream(self, user_input, image_path=None):
        # Format input mạnh mẽ nhằm ép tuân thủ Tiếng Việt
        wrapped_input = (
            "Trả lời NGẮN GỌN, CHÍNH XÁC, "
            "CHỈ bằng TIẾNG VIỆT tĩnh. "
            "TUYỆT ĐỐI KHÔNG dùng tiếng Trung.\n\n"
            f"Câu hỏi: {user_input}"
        )
        
        # Xử lý input dưới dạng cấu trúc đa phương thức (List of dicts)
        if image_path:
             content = [
                 {"type": "image", "image": image_path},
                 {"type": "text", "text": wrapped_input}
             ]
        else:
             content = [{"type": "text", "text": wrapped_input}]
             
        self.history.append({"role": "user", "content": content})

        # Apply chat template
        if self.processor and hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = self.tokenizer.apply_chat_template(
                self.history,
                tokenize=False,
                add_generation_prompt=True
            )
            
        # Tokenizer inference
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Khắc phục lỗi 4: Lặp từ và nội dung lan man -> Greedy Decoding + Repetition Penalty
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=False,        # Sử dụng Greedy decoding để tránh bị "Hallucination"
            num_beams=1,
            repetition_penalty=1.2, # Xử phạt những từ lặp lại liên tục
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        if self.bad_words_ids:
            generation_kwargs['bad_words_ids'] = self.bad_words_ids

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\nAssistant: ", end="", flush=True)
        
        full_response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text
            
        print("\n") # Xuống dòng khi sinh xong
        
        self.history.append({"role": "assistant", "content": [{"type": "text", "text": full_response}]})

if __name__ == "__main__":
    qwen_vl = QwenVLChat()
    
    print("="*60)
    print("Chào bạn! Tôi là trợ lý AI Đa Phương Thức (Qwen3-VL-8B).")
    print("Gõ 'exit' để thoát, 'clear' để xóa lịch sử.")
    print("Ghi chú: Bản code dựng khung VL dựa theo Qwen để hạn chế lỗi")
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
                qwen_vl.reset_chat()
                continue
                
            qwen_vl.chat_stream(user_input)
            
        except KeyboardInterrupt:
            print("\nĐã dừng cuộc trò chuyện.")
            break
        except Exception as e:
            print(f"\n[Lỗi] {e}")
