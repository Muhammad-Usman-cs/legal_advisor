from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:admin@localhost:5432/legal_advisor")

# The engine is the core connection to the database.
# PostgreSQL handles concurrent connections natively via its own connection pool,
# so no extra connect_args are needed unlike SQLite.
engine = create_engine(DATABASE_URL)

# SessionLocal is a factory that creates new database sessions.
# Each request gets its own session, uses it, then closes it.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base is the parent class all our models will inherit from.
# SQLAlchemy uses it to track which classes are database tables.
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that provides a database session per request.
    The 'yield' makes this a context manager: code after yield runs on cleanup.
    FastAPI calls this automatically when a route declares it as a dependency.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
