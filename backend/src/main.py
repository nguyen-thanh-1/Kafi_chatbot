from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import chatbot
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

@app.get("/")
def read_root():
    return {"status": "online", "message": "Kafi Chatbot API is running"}

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
