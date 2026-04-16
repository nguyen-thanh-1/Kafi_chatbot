from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from src.routers import chatbot, market_data
from src.utils.llm import get_llm
from src.pipeline.chat_pipeline import get_chat_pipeline
from src.utils.logger import logger
from src.utils.app_config import AppConfig
import uvicorn

app = FastAPI(title="Kafi Chatbot API", version="0.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chatbot.router)
app.include_router(market_data.router)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Kafi Chatbot API is running"}

@app.on_event("startup")
def _preload_default_model():
    # Preload optional pipeline models in the background so the API becomes responsive immediately.
    def _load():
        try:
            pipeline = get_chat_pipeline()
            pipeline.warmup()

            # Preload main LLM only if configured (can be very heavy on VRAM).
            startup_cfg = (AppConfig.get_pipeline_config().get("startup", {}) or {})
            if bool(startup_cfg.get("preload_main_llm", False)):
                logger.info("[startup] preloading main LLM (default model) ...")
                get_llm().ensure_loaded()
                logger.info("[startup] main LLM ready.")
        except Exception:
            logger.exception("[startup] preload failed")

    Thread(target=_load, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
