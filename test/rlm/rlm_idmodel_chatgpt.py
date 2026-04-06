import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# =====================================
# 1️⃣ Load Real LLM (ID model thay tại đây)
# =====================================

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # Bạn có thể đổi ID model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

def call_llm(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =====================================
# 2️⃣ Environment
# =====================================

class Environment:
    def __init__(self, document):
        self.document = document

    def search(self, keyword):
        matches = [m.start() for m in re.finditer(keyword, self.document)]
        return matches

    def read(self, start, length):
        return self.document[start:start+length]

    def get_metadata(self):
        return {
            "length": len(self.document),
            "preview": self.document[:200]
        }


# =====================================
# 3️⃣ RLM Controller
# =====================================

class RLMController:
    def __init__(self, env, max_steps=10):
        self.env = env
        self.scratchpad = ""
        self.max_steps = max_steps

    def build_prompt(self):
        metadata = self.env.get_metadata()

        return f"""
You are a Recursive Language Model.
You do NOT have access to the full document.
You can write Python code using these tools:

- env.search(keyword)
- env.read(start, length)
- call_subtask(question)

When ready, return FINAL("your answer")

Metadata:
Document length: {metadata['length']}
Preview: {metadata['preview']}

Scratchpad:
{self.scratchpad}

Write Python code only.
"""

    def call_subtask(self, question):
        print("🔁 Subtask:", question)
        sub_rlm = RLMController(self.env, max_steps=5)
        return sub_rlm.run(question)

    def run(self, question="Answer the question."):
        for step in range(self.max_steps):
            print(f"\n===== STEP {step+1} =====")

            prompt = self.build_prompt() + f"\nUser Question: {question}"
            llm_output = call_llm(prompt)

            print("🧠 LLM RAW OUTPUT:\n", llm_output)

            # Extract Python code block
            code_match = re.search(r"```python(.*?)```", llm_output, re.DOTALL)
            if not code_match:
                print("⚠️ No valid Python code found.")
                break

            code = code_match.group(1)

            print("⚙️ Executing code:\n", code)

            # Sandbox execution context
            local_context = {
                "env": self.env,
                "call_subtask": self.call_subtask,
                "FINAL": lambda x: x
            }

            try:
                exec(code, {}, local_context)

                if "result" in local_context:
                    self.scratchpad += f"\nResult: {local_context['result']}"

                if "FINAL" in code:
                    return local_context.get("result", "Finished.")

            except Exception as e:
                self.scratchpad += f"\nError: {str(e)}"

        return "Max steps reached."

long_doc = """
Albert Einstein was a theoretical physicist.
He developed the theory of relativity.
He was born in Germany in 1879.
"""

env = Environment(long_doc)
rlm = RLMController(env)

answer = rlm.run("Where was Einstein born?")
print("\n🎯 FINAL ANSWER:", answer)