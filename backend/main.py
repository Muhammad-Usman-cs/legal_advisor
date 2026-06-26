from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine
import models
from routers import auth, chat

models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Legal Advisor API",
    description="An AI-powered chatbot for Pakistani law using RAG and conversation memory.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_origin_regex=r"https://.*\.ngrok-free\.dev",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])


@app.get("/", tags=["Health"])
async def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "message": "Legal Advisor API is running"}
