from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from src.schemas.chat_schema import ChatRequest, ModelSelectRequest, AvailableModel
from typing import List
import shutil
import os

from src.conversation.session_manager import get_session_manager
from src.utils.llm import get_llm
from src.utils.app_config import AppConfig
from src.pipeline.chat_pipeline import get_chat_pipeline
from src.utils.knowledge_base import get_knowledge_base
from src.utils.knowledge_manager import get_knowledge_manager
from src.utils.logger import logger

router = APIRouter(prefix="/api/chat", tags=["chatbot"])

@router.get("/models", response_model=List[AvailableModel])
async def get_models():
    """Returns a list of available AI models."""
    llm_cfg = AppConfig.get_llm_config()
    model_list = llm_cfg.get("model_list", [])
    return [{"id": m.get("id"), "name": m.get("name")} for m in model_list]

@router.get("/current-model")
async def get_current_model():
    """Returns the ID of the currently loaded model."""
    llm = get_llm()
    return {"model_id": llm.current_model_id}

@router.post("/model")
async def select_model(request: ModelSelectRequest):
    """Switches the active AI model."""
    llm = get_llm()
    success = llm.switch_model(request.model_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load model {request.model_id}")
    return {"status": "success", "model_id": request.model_id}

@router.post("", response_model=None)
async def chat_endpoint(request: ChatRequest):
    """
    Chatbot endpoint with session-based history management.
    Uses the ChatPipeline and SessionManager.
    """
    pipeline = get_chat_pipeline()
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
            for chunk in pipeline.process(request.message, history):
                full_response += chunk
                yield chunk

            if request.session_id:
                session_manager.add_message(request.session_id, "assistant", full_response)
        except Exception as e:
            yield f"\n[Error from Pipeline] {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")


@router.get("/trace")
async def get_last_trace():
    """Returns the last pipeline routing/cache/guardrails decision for debugging."""
    pipeline = get_chat_pipeline()
    trace = pipeline.get_last_trace()
    return trace.__dict__ if trace is not None else {}

@router.post("/upload-knowledge")
async def upload_knowledge(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_dir = os.path.join("data", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    # Save file immediately
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    km = get_knowledge_manager()
    file_id = km.add_file(file.filename)

    def process_indexing(path: str, fid: int):
        try:
            kb = get_knowledge_base()
            kb.ingest_pdf(path)
            get_knowledge_manager().update_status(fid, "completed")
        except Exception as e:
            logger.error(f"Background indexing failed: {e}")
            get_knowledge_manager().update_status(fid, "error", error_message=str(e))
        finally:
            if os.path.exists(path):
                os.remove(path)

    # Run indexing in background
    background_tasks.add_task(process_indexing, file_path, file_id)
    
    return {"message": "Đã bắt đầu nạp tài liệu. Bạn có thể theo dõi trạng thái ở bảng quản lý.", "id": file_id}

@router.get("/knowledge/files")
async def list_knowledge_files():
    """List all knowledge files and their processing status"""
    km = get_knowledge_manager()
    return km.get_files()

@router.delete("/knowledge/files/{file_id}")
async def delete_knowledge_file(file_id: int):
    """Delete a knowledge file from the database and Qdrant"""
    km = get_knowledge_manager()
    filename = km.delete_file(file_id)
    if filename:
        kb = get_knowledge_base()
        kb.delete_file_vectors(filename)
        return {"message": f"File '{filename}' deleted from knowledge base"}
    raise HTTPException(status_code=404, detail="File not found")
