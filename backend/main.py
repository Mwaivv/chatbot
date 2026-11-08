import os
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import anthropic
from dotenv import load_dotenv

from database import get_db
import models

load_dotenv()

app = FastAPI(title="AI Chat App API", version="1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Anthropic Client
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key) if anthropic_api_key else None

class MessageCreate(BaseModel):
    conversation_id: Optional[int] = None
    content: str
    user_id: int = 1  # Default demo user id for setup

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Chat App API with FastAPI and Anthropic!"}

@app.post("/chat")
def chat_endpoint(payload: MessageCreate, db: Session = Depends(get_db)):
    if not client:
        raise HTTPException(status_code=500, detail="Anthropic API key is not configured in your .env file.")

    # 1. Handle or create a new conversation
    conversation_id = payload.conversation_id
    if not conversation_id:
        db_conv = models.Conversation(user_id=payload.user_id, title=payload.content[:30] + "...")
        db.add(db_conv)
        db.commit()
        db.refresh(db_conv)
        conversation_id = db_conv.id
    else:
        db_conv = db.query(models.Conversation).filter(models.Conversation.id == conversation_id).first()
        if not db_conv:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    # 2. Save the user's message to the database
    user_msg = models.Message(
        conversation_id=conversation_id,
        role="user",
        content=payload.content
    )
    db.add(user_msg)
    db.commit()

    # 3. Retrieve chat history for context window
    past_messages = db.query(models.Message).filter(
        models.Message.conversation_id == conversation_id
    ).order_by(models.Message.created_at.asc()).all()

    formatted_messages = [
        {"role": m.role, "content": m.content} for m in past_messages
    ]

    try:
        # 4. Call Anthropic Claude API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=formatted_messages
        )
        assistant_reply = response.content[0].text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 5. Save the assistant's reply to the database
    assistant_msg = models.Message(
        conversation_id=conversation_id,
        role="assistant",
        content=assistant_reply
    )
    db.add(assistant_msg)
    db.commit()

    return {
        "conversation_id": conversation_id,
        "user_message": payload.content,
        "assistant_response": assistant_reply
    }