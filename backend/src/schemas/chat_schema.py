from pydantic import BaseModel, Field
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the speaker (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="The new message from the user")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation history")

class ChatResponse(BaseModel):
    # This could be used for non-streaming responses if needed
    response: str
