from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.schemas.chat_schema import ChatRequest, ChatResponse
from src.agents.financial_agent import get_financial_agent
from typing import List

router = APIRouter(prefix="/api/chat", tags=["chatbot"])

@router.post("", response_model=None)
async def chat_endpoint(request: ChatRequest):
    """
    Chatbot endpoint for financial advice.
    This uses the FinancialAgent to coordinate with the LLM.
    """
    agent = get_financial_agent()
    
    # Pre-process history from schema to dict format required by agent/llm
    processed_history = []
    for msg in request.history:
        processed_history.append({"role": msg.role, "content": msg.content})
    
    def generate():
        try:
            # Delegate to the agent ("the brain")
            for chunk in agent.process_chat(request.message, processed_history):
                yield chunk
        except Exception as e:
            yield f"\n[Error from Agent] {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")
