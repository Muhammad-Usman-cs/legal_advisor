from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# HOW PYDANTIC SCHEMAS WORK IN FASTAPI
#
# Schemas are NOT database models. They define the shape of data moving
# in and out of the API. FastAPI uses them to:
#   1. Validate incoming request data (wrong type or missing field = 422 error)
#   2. Serialize outgoing response data (converts SQLAlchemy objects to JSON)
#   3. Generate the interactive /docs page automatically
#
# Rule of thumb:
#   - "Create" schemas  → what the CLIENT sends to us (input)
#   - "Response" schemas → what we send back to the CLIENT (output)
#   - Never expose hashed_password in a Response schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# USER SCHEMAS
# ---------------------------------------------------------------------------

class UserCreate(BaseModel):
    """Data required to register a new user."""
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr          # FastAPI validates this is a real email format
    password: str = Field(min_length=6, max_length=72)


class UserResponse(BaseModel):
    """
    What we return after creating or fetching a user.
    Notice: no password field — we never expose it.

    model_config tells Pydantic it's okay to read data from a SQLAlchemy
    object (ORM mode). Without this, Pydantic only accepts plain dicts.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime


# ---------------------------------------------------------------------------
# AUTH SCHEMAS
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    """Credentials sent by the user to log in."""
    username: str
    password: str


class Token(BaseModel):
    """JWT token returned after a successful login."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """
    Internal schema — NOT exposed in the API.
    Used to hold data decoded from a JWT token inside the app.
    """
    username: Optional[str] = None


# ---------------------------------------------------------------------------
# CONVERSATION SCHEMAS
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    """Optional title when starting a new chat session."""
    title: str = Field(default="New Conversation", max_length=200)


class ConversationResponse(BaseModel):
    """A conversation summary (without its messages)."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# MESSAGE SCHEMAS
# ---------------------------------------------------------------------------

class MessageResponse(BaseModel):
    """
    A single message in a conversation.
    role will be either 'user' or 'assistant'.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    created_at: datetime


class ConversationWithMessages(BaseModel):
    """A full conversation including all its messages — used for loading chat history."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    messages: list[MessageResponse] = []


# ---------------------------------------------------------------------------
# CHAT SCHEMAS
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    What the frontend sends when the user types a message.
    - conversation_id: if None, the backend creates a new conversation automatically.
    - message: the user's question (1–2000 characters).
    """
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    """What the backend returns after getting a reply from Groq."""
    reply: str
    conversation_id: int        # so the frontend knows which conversation to update
