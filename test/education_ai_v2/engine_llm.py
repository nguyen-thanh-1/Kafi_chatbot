"""
LLM Engine - Wrapper cho Llama 3.1 8B với streaming
Tích hợp Subject Detection và Educational Prompts
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
from typing import List, Generator, Tuple

from prompts import detect_subject, get_subject_emoji, get_subject_name, get_rag_prompt, get_system_prompt
from utils import route_language, contains_cjk, clean_response


class EducationalLLM:
    """Llama 3.1 8B với streaming support và Educational Features"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        if self._initialized:
            return
        
        print(f"Loading LLM: {model_id}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_config,
        )
        
        self._initialized = True
        print("LLM loaded successfully!")
    
    def analyze_question(self, question: str) -> dict:
        """Phân tích câu hỏi: môn học, ngôn ngữ"""
        subject = detect_subject(question)
        language = route_language(question)
        
        return {
            "subject": subject,
            "subject_name": get_subject_name(subject, language),
            "subject_emoji": get_subject_emoji(subject),
            "language": language,
            "language_name": "Tiếng Việt" if language == "vi" else "English"
        }
    
    def stream_chat(
        self,
        user_input: str,
        system_prompt: str,
        history: List[dict] = None,
        max_tokens: int = 2048,
        temperature: float = 0.2
    ) -> Generator[str, None, None]:
        """Stream response tokens"""
        
        if history is None:
            history = []
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
    
    def answer_with_context(
        self,
        question: str,
        context_chunks: List[Tuple[str, float]],
        stream: bool = True
    ):
        """
        Trả lời câu hỏi dựa trên context từ RAG
        Tự động detect môn học và chọn prompt phù hợp
        """
        
        if not context_chunks:
            msg = "Tôi không tìm thấy thông tin liên quan trong tài liệu."
            if stream:
                yield msg
                return
            return msg
        
        # Analyze question
        analysis = self.analyze_question(question)
        subject = analysis["subject"]
        language = analysis["language"]
        
        # Build context
        context_parts = []
        for i, (chunk, score) in enumerate(context_chunks, 1):
            context_parts.append(f"[Đoạn {i}] (relevance: {score:.2f})\n{chunk}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Get RAG prompt with subject-specific instructions
        prompt = get_rag_prompt(question, context, subject, language)
        
        # System prompt for strict RAG
        system_prompt = f"""Bạn là trợ lý giáo dục thông minh, chuyên về {analysis['subject_name']}.
Bạn CHỈ trả lời dựa trên thông tin trong tài liệu được cung cấp.
TUYỆT ĐỐI KHÔNG bịa đặt thông tin không có trong tài liệu."""
        
        if stream:
            cjk_detected = False
            for token in self.stream_chat(prompt, system_prompt, temperature=0.15):
                # Check for CJK characters
                if contains_cjk(token):
                    cjk_detected = True
                yield token
            
            if cjk_detected:
                yield "\n\n⚠️ *Phát hiện ký tự lạ trong response*"
        else:
            response = ""
            for token in self.stream_chat(prompt, system_prompt, temperature=0.15):
                response += token
            return clean_response(response)
    
    def chat_without_rag(
        self,
        question: str,
        history: List[dict] = None,
        stream: bool = True
    ):
        """
        Chat trực tiếp không dùng RAG (cho câu hỏi không liên quan đến tài liệu)
        """
        analysis = self.analyze_question(question)
        system_prompt = get_system_prompt(analysis["subject"], analysis["language"])
        
        if stream:
            for token in self.stream_chat(question, system_prompt, history, temperature=0.3):
                yield token
        else:
            response = ""
            for token in self.stream_chat(question, system_prompt, history, temperature=0.3):
                response += token
            return clean_response(response)
