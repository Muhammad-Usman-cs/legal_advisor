from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from database import get_db
from models import User
from schemas import UserCreate, UserResponse, Token
from security import hash_password, verify_password, create_access_token, get_current_user

# ---------------------------------------------------------------------------
# APIRouter groups related endpoints together.
# The prefix "/auth" and tag "Authentication" are applied in main.py when
# this router is included, so we don't repeat them here.
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Create a new user account.

    FastAPI automatically:
      - Parses the JSON body into a UserCreate object
      - Validates all fields (email format, min lengths, etc.)
      - Returns a 422 error if validation fails — no manual checks needed

    We still manually check uniqueness because that's a business rule,
    not something Pydantic can verify (it would need to query the DB).
    """
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password)
    )
    db.add(new_user)
    db.commit()
    # db.refresh loads the auto-generated fields (id, created_at) back into the object
    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    Log in and receive a JWT access token.

    OAuth2PasswordRequestForm is a FastAPI built-in that reads
    'username' and 'password' from a form body (application/x-www-form-urlencoded).
    This matches the OAuth2 spec and works with the /docs 'Authorize' button.
    """
    user = db.query(User).filter(User.username == form_data.username).first()

    # We check both "user not found" and "wrong password" with the same error.
    # Giving different messages for each would let attackers enumerate valid usernames.
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": user.username})
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """
    Returns the currently logged-in user's profile.
    Depends(get_current_user) handles all token verification — this
    route only runs if the token is valid. A great route to test auth is working.
    """
    return current_user
