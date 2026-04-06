from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.schemas.chat_schema import ChatRequest, ChatResponse
from src.agents.financial_agent import get_financial_agent
from typing import List

from src.conversation.session_manager import get_session_manager

router = APIRouter(prefix="/api/chat", tags=["chatbot"])

@router.post("", response_model=None)
async def chat_endpoint(request: ChatRequest):
    """
    Chatbot endpoint with session-based history management.
    Uses the FinancialAgent and SessionManager.
    """
    agent = get_financial_agent()
    session_manager = get_session_manager()
    
    # Identify history source: either manual history or from session_id
    if request.session_id:
        history = session_manager.get_history(request.session_id)
    else:
        # Fallback to manual history from request
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # Add the current user question to history (if using a session)
    if request.session_id:
        session_manager.add_message(request.session_id, "user", request.message)
    
    def generate():
        full_response = ""
        try:
            # Delegate to the agent ("the brain")
            for chunk in agent.process_chat(request.message, history):
                full_response += chunk
                yield chunk
            
            # Save the complete AI response back to the session history
            if request.session_id:
                session_manager.add_message(request.session_id, "assistant", full_response)
                
        except Exception as e:
            yield f"\n[Error from Agent] {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")
