from fastapi import FastAPI, APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..core.database import get_db, init_db
from ..services.agent_service import CollegeChatbotAgent
from pydantic import BaseModel
from typing import List
import os

app = FastAPI()
router = APIRouter()

# Initialize database
init_db()

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class IndexRequest(BaseModel):
    urls: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    agent = CollegeChatbotAgent(db)
    response = await agent.get_response(request.message)
    return ChatResponse(response=response)

@router.post("/index/documents")
async def index_documents():
    from ..indexer.content_indexer import ContentIndexer
    indexer = ContentIndexer()
    try:
        documents_dir = os.getenv("DOCUMENTS_DIR")
        if not documents_dir:
            raise HTTPException(status_code=400, detail="DOCUMENTS_DIR not configured")
        indexer.index_documents(documents_dir)
        return {"status": "success", "message": "Documents indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index/website")
async def index_website(request: IndexRequest):
    from ..indexer.content_indexer import ContentIndexer
    indexer = ContentIndexer()
    try:
        indexer.index_website(request.urls)
        return {"status": "success", "message": "Website content indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)
