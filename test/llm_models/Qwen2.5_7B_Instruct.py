import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch

class QwenChat:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
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
        
        # System prompt cố định, cực mạnh về ngôn ngữ
        self.system_prompt = """Bạn là một trợ lý AI thông minh, hữu ích và thân thiện đến từ Việt Nam.
NHIỆM VỤ: Trả lời câu hỏi của người dùng một cách chính xác, tự nhiên.
YÊU CẦU BẮT BUỘC:
1. LUÔN LUÔN trả lời bằng Tiếng Việt.
2. TUYỆT ĐỐI KHÔNG sử dụng tiếng Trung Quốc (Chinese) trong bất kỳ tình huống nào.
3. Nếu câu hỏi là ngôn ngữ khác, hãy dịch ý và trả lời lại bằng Tiếng Việt.
4. Không lặp lại câu hỏi của người dùng.
5. Trả lời ngắn gọn, súc tích, đi thẳng vào vấn đề."""
        
        # Lịch sử hội thoại
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset_chat(self):
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]
        print("\n[Hệ thống] Đã xóa lịch sử trò chuyện.\n")

    def chat_stream(self, user_input):
        # Thêm câu hỏi của user vào lịch sử
        self.history.append({"role": "user", "content": user_input})

        # Format input theo template
        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Streamer để in ra màn hình ngay lập tức
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Cấu hình sinh văn bản (đã tinh chỉnh để giảm ảo giác/tiếng Trung)
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=2048, 
            temperature=0.3,     # Giảm thêm xuống 0.3
            top_p=0.85,          # Giảm top_p để loại bỏ các token lạ
            repetition_penalty=1.2, # Tăng phạt lặp
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
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
    print("Chào bạn! Tôi là trợ lý AI (Qwen 2.5).")
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
