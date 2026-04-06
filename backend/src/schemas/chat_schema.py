from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="The new message from the user")
    session_id: Optional[str] = Field(None, description="Unique session ID for conversation context")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Manual conversation history (optional if session_id is used)")

class ChatResponse(BaseModel):
    # This could be used for non-streaming responses if needed
    response: str
