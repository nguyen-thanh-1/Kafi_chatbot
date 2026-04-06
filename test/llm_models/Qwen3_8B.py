import sys
import codecs

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch
import warnings

warnings.filterwarnings('ignore')

class QwenChat:
    def __init__(self, model_name="Qwen/Qwen3-8B", enable_thinking=False):
        """
        Khởi tạo Qwen3-8B Chat.
        
        Args:
            model_name: Tên model trên HuggingFace (mặc định: Qwen/Qwen3-8B)
            enable_thinking: Bật/tắt chế độ Thinking (suy luận).
                             - True : Model sẽ sinh quá trình suy nghĩ <think>...</think> trước khi trả lời.
                             - False: Model trả lời trực tiếp như Qwen2.5-Instruct (nhanh hơn, ngắn gọn hơn).
                             Mặc định: False (Non-thinking mode)
        """
        print(f"Loading model: {model_name}...")
        self.enable_thinking = enable_thinking

        # Khắc phục lỗi 1: Out of Memory (OOM) bằng tính năng Quantization (4-bit)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Tải Tokenizer và Model (Qwen3-8B là text-only, dùng AutoModelForCausalLM)
        # Yêu cầu transformers >= 4.51.0
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print("Model loaded successfully!")

        # Khắc phục lỗi 2: Đi lạc đề, ảo giác ngôn ngữ (sinh Tiếng Trung)
        self.system_prompt = """Bạn là một trợ lý AI thông minh đến từ Việt Nam.
NHIỆM VỤ: Trả lời câu hỏi của người dùng một cách chính xác, tự nhiên.
YÊU CẦU BẮT BUỘC:
1. LUÔN LUÔN trả lời bằng Tiếng Việt.
2. TUYỆT ĐỐI KHÔNG sử dụng tiếng Trung Quốc (Chinese) trong bất kỳ tình huống nào.
3. Nếu câu hỏi là ngôn ngữ khác, hãy dịch ý và trả lời lại bằng Tiếng Việt.
4. Không lặp lại câu hỏi của người dùng.
5. Trả lời ngắn gọn, súc tích, đi thẳng vào vấn đề."""

        # Lọc các token ngoại ngữ để ép sinh Tiếng Việt
        print("Đang tạo danh sách chặn token ngoại ngữ (Trung, Nga, Nhật, Hàn...)...")
        self.bad_words_ids = self._get_non_vietnamese_bad_words()
        if self.bad_words_ids:
            print(f"Đã chặn {len(self.bad_words_ids)} token ngoại ngữ!")

        # Lịch sử hội thoại
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

        mode_label = "Thinking (Suy luận)" if self.enable_thinking else "Non-Thinking (Trực tiếp)"
        print(f"Chế độ hoạt động: {mode_label}")

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
            # Cấp phép: ASCII, Latin Extended (Tiếng Việt), Punctuation
            if ord(ch) < 128:
                return True
            if '\u00c0' <= ch <= '\u01b0' or '\u1ea0' <= ch <= '\u1ef9':
                return True
            if ch in '–—\u2018\u2019\u201c\u201d\u2026\u2022\u00b7\u00d7\u00f7\u00b1\u2260\u2264\u2265':
                return True
            return False

        try:
            for i in range(self.tokenizer.vocab_size):
                token = self.tokenizer.decode([i])
                if any(not is_allowed_char(ch) for ch in token):
                    bad_words.append([i])
            return bad_words
        except Exception:
            return None

    def chat_stream(self, user_input):
        """
        Sinh câu trả lời theo dạng streaming.
        
        Qwen3-8B hỗ trợ 2 chế độ qua enable_thinking trong apply_chat_template:
          - enable_thinking=True : Thinking Mode  (Temperature=0.6, TopP=0.95, TopK=20)
          - enable_thinking=False: Non-Thinking Mode (Temperature=0.7, TopP=0.8, TopK=20)
        """
        # Format input để ép tuân thủ Tiếng Việt
        if self.enable_thinking:
            # Thinking mode: câu hỏi tự nhiên, để model tự suy luận
            wrapped_input = (
                "Trả lời CHỈ bằng TIẾNG VIỆT. "
                "TUYỆT ĐỐI KHÔNG dùng tiếng Trung.\n\n"
                f"Câu hỏi: {user_input}"
            )
        else:
            # Non-thinking mode: yêu cầu trả lời trực tiếp, ngắn gọn
            wrapped_input = (
                "TRẢ LỜI TRỰC TIẾP, KHÔNG SUY NGHĨ (NO THINKING). "
                "CHỈ bằng TIẾNG VIỆT. "
                "TUYỆT ĐỐI KHÔNG dùng tiếng Trung hoặc thẻ <think>.\n\n"
                f"Câu hỏi: {user_input}"
            )

        self.history.append({"role": "user", "content": wrapped_input})

        # Apply chat template với enable_thinking tương ứng (yêu cầu transformers >= 4.51.0)
        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking  # API chính thức của Qwen3
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Cấu hình generation theo khuyến nghị chính thức từ Qwen3 README:
        # - Thinking   : Temperature=0.6, TopP=0.95, TopK=20, KHÔNG dùng greedy decoding
        # - Non-Thinking: Temperature=0.7, TopP=0.8,  TopK=20
        if self.enable_thinking:
            generation_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        else:
            generation_kwargs = dict(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        if self.bad_words_ids:
            generation_kwargs['bad_words_ids'] = self.bad_words_ids

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\nAssistant: ", end="", flush=True)

        full_response = ""
        is_thinking = False

        for new_text in streamer:
            chunk = new_text

            # Xử lý và ẩn đoạn suy luận <think>...</think> (chỉ relevant khi enable_thinking=True)
            if "<think>" in chunk:
                is_thinking = True
                chunk = chunk.replace("<think>", "")

            if "</think>" in chunk:
                is_thinking = False
                chunk = chunk.replace("</think>", "")

            # Chỉ in ra phần câu trả lời thật (không in thinking process)
            if not is_thinking and chunk.strip() != "":
                print(chunk, end="", flush=True)
                full_response += chunk

        print("\n")  # Xuống dòng khi sinh xong

        # Lưu câu hỏi gốc vào lịch sử (bỏ phần prefix wrapping)
        self.history[-1]["content"] = user_input
        self.history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    # Có thể thay đổi enable_thinking=True để bật chế độ suy luận
    qwen = QwenChat(enable_thinking=False)

    print("="*60)
    print("Chào bạn! Tôi là trợ lý AI (Qwen3-8B).")
    print("Gõ 'exit' để thoát, 'clear' để xóa lịch sử.")
    print("Ghi chú: Quá trình suy luận nội bộ <think> đã được ẩn.")
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
