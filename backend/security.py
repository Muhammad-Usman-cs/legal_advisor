import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from dotenv import load_dotenv
import os

from database import get_db
from schemas import TokenData

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))

# ---------------------------------------------------------------------------
# PASSWORD HASHING
# Using bcrypt directly — passlib 1.7.4 is unmaintained and incompatible
# with bcrypt 4.0+. Direct bcrypt calls are simpler and version-stable.
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# JWT TOKENS
#
# A JWT has three parts: header.payload.signature
# We put the username in the payload under the key "sub" (subject).
# The signature is created with SECRET_KEY — tampered tokens fail verification.
# ---------------------------------------------------------------------------

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> TokenData:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    username: str | None = payload.get("sub")
    if username is None:
        raise JWTError("Token missing subject")
    return TokenData(username=username)


# ---------------------------------------------------------------------------
# OAUTH2 SCHEME
#
# OAuth2PasswordBearer tells FastAPI where the login endpoint is.
# It reads the "Authorization: Bearer <token>" header automatically.
# Any route that declares `token = Depends(oauth2_scheme)` gets the raw token.
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ---------------------------------------------------------------------------
# get_current_user DEPENDENCY
#
# This is the reusable guard for all protected routes.
# Any route that declares `user = Depends(get_current_user)` will:
#   1. Extract the JWT from the Authorization header
#   2. Decode and verify it
#   3. Load the user from the database
#   4. Inject the User object into the route function
# If any step fails, FastAPI returns 401 automatically — the route never runs.
# ---------------------------------------------------------------------------

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    from models import User  # local import to avoid circular imports

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token_data = decode_token(token)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return user
