from src.utils.llm import get_llm
from typing import List, Dict

class FinancialAgent:
    def __init__(self):
        self.llm = get_llm()

    def process_chat(self, user_input: str, history: List[Dict[str, str]]):
        """
        Coordinates the LLM to process a financial chat request.
        """
        # Logic for pre-processing or additional context could go here
        # For now, it simply delegates to the LLM
        return self.llm.generate_response(user_input, history)

# Singleton helper
_agent_instance = None

def get_financial_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FinancialAgent()
    return _agent_instance
