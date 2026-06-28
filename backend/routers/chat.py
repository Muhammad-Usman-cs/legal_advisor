from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models import User, Conversation, Message
from schemas import ChatRequest, ChatResponse, ConversationResponse, ConversationWithMessages
from security import get_current_user
from services.ai_service import legal_chain

# ---------------------------------------------------------------------------
# All routes here are protected — every endpoint requires a valid JWT.
# Depends(get_current_user) handles token verification automatically.
# If the token is missing or invalid, FastAPI returns 401 before the
# route function even starts.
# ---------------------------------------------------------------------------

router = APIRouter()

HISTORY_LIMIT = 20  # number of past messages sent to Groq for memory


# ---------------------------------------------------------------------------
# SEND A MESSAGE
# ---------------------------------------------------------------------------

@router.post("/message", response_model=ChatResponse)
def send_message(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Main chat endpoint. Full flow:
      1. Get existing conversation or create a new one
      2. Fetch the last N messages for memory
      3. Pass message + history to the LangChain chain, which internally:
           - embeds the query and retrieves relevant law sections (MMR)
           - builds the prompt with context + history + question
           - calls Groq and returns the reply
      4. Save both the user message and AI reply to the database
      5. Return the AI reply

    conversation_id in the request is optional — omit it to start a new chat.
    """

    # --- Step 1: resolve conversation ---
    if request.conversation_id:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == current_user.id   # users can only access their own chats
        ).first()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    else:
        # Auto-title the conversation from the first 60 chars of the opening message
        title = request.message[:60] + ("..." if len(request.message) > 60 else "")
        conversation = Conversation(user_id=current_user.id, title=title)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    # --- Step 2: fetch conversation history for memory ---
    past_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(HISTORY_LIMIT)
        .all()
    )
    # Reverse so they are oldest-first (chronological order for the prompt)
    history = [
        {"role": msg.role, "content": msg.content}
        for msg in reversed(past_messages)
    ]

    # --- Step 3 + 4: retrieval and generation happen inside the LangChain chain ---
    reply = legal_chain.get_reply(
        user_message=request.message,
        history=history,
    )

    # --- Step 5: persist both messages ---
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message
    )
    ai_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=reply
    )
    db.add_all([user_msg, ai_msg])
    db.commit()

    # --- Step 6: return ---
    return ChatResponse(reply=reply, conversation_id=conversation.id)


# ---------------------------------------------------------------------------
# LIST ALL CONVERSATIONS
# ---------------------------------------------------------------------------

@router.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns all conversations for the logged-in user, newest first.
    Used to populate the sidebar in the frontend.
    """
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return conversations


# ---------------------------------------------------------------------------
# GET ONE CONVERSATION WITH ALL MESSAGES
# ---------------------------------------------------------------------------

@router.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns a single conversation with its full message history.
    Used when the user clicks a conversation in the sidebar to load it.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation


# ---------------------------------------------------------------------------
# DELETE A CONVERSATION
# ---------------------------------------------------------------------------

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Deletes a conversation and all its messages.
    The cascade="all, delete-orphan" on the Conversation model means
    SQLAlchemy automatically deletes child messages — no manual cleanup needed.
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == current_user.id
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    db.delete(conversation)
    db.commit()
