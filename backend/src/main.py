from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from src.routers import chatbot, market_data
from src.utils.llm import get_llm
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
    # Load the default model in the background so the API becomes responsive immediately.
    def _load():
        try:
            get_llm().ensure_loaded()
        except Exception:
            pass

    Thread(target=_load, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
