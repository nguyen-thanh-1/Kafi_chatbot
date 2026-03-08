import re

# =====================================
# 1. Base LLM (Giả lập)
# =====================================

class BaseLLM:
    def generate(self, metadata, scratchpad):
        """
        metadata: thông tin về prompt (độ dài, preview)
        scratchpad: lịch sử suy luận
        """

        # Giả lập logic ra quyết định
        if "SEARCH" not in scratchpad:
            return "SEARCH('Einstein')"
        
        if "READ" not in scratchpad:
            return "READ(0, 500)"
        
        return "FINAL('Answer found: Albert Einstein was a physicist.')"


# =====================================
# 2. Environment (giữ prompt rất dài)
# =====================================

class Environment:
    def __init__(self, long_text):
        self.long_text = long_text

    def search(self, keyword):
        matches = [m.start() for m in re.finditer(keyword, self.long_text)]
        return matches

    def read(self, start, length):
        return self.long_text[start:start+length]


# =====================================
# 3. Recursive Language Model Controller
# =====================================

class RLMController:
    def __init__(self, llm, environment):
        self.llm = llm
        self.env = environment
        self.scratchpad = ""

    def run(self):
        metadata = {
            "length": len(self.env.long_text),
            "preview": self.env.long_text[:100]
        }

        while True:
            action = self.llm.generate(metadata, self.scratchpad)
            print("LLM Action:", action)

            if action.startswith("SEARCH"):
                keyword = action.split("'")[1]
                results = self.env.search(keyword)
                self.scratchpad += f"\nSEARCH -> {results}"

            elif action.startswith("READ"):
                numbers = re.findall(r'\d+', action)
                start, length = map(int, numbers)
                text = self.env.read(start, length)
                self.scratchpad += f"\nREAD -> {text}"

            elif action.startswith("FINAL"):
                answer = action.split("'")[1]
                return answer

            elif action.startswith("CALL_SUBTASK"):
                sub_question = ...
                sub_rlm = RLMController(self.llm, self.env)
                sub_answer = sub_rlm.run()
                self.scratchpad += f"\nCALL_SUBTASK -> {sub_answer}"
                
            else:
                raise Exception("Unknown action")

long_document = """
Albert Einstein was a theoretical physicist who developed the theory of relativity.
He was born in Germany.
"""

llm = BaseLLM()
env = Environment(long_document)
rlm = RLMController(llm, env)

result = rlm.run()
print("Final Output:", result)