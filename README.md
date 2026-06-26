# Legal Advisor — Pakistani Law Chatbot

A full-stack AI chatbot that answers Pakistani legal questions using actual law documents. Built with FastAPI + PostgreSQL on the backend and React + Vite on the frontend.

---

## Prerequisites

Make sure these are installed before starting:

- **Python 3.10+** — [python.org](https://python.org)
- **PostgreSQL 14+** — [postgresql.org](https://postgresql.org)
- **Node.js 18+** — [nodejs.org](https://nodejs.org)
- **A Groq API key** — free at [console.groq.com](https://console.groq.com)

---

## 1. Database Setup (PostgreSQL)

Open **psql** or **pgAdmin** and run:

```sql
CREATE DATABASE legal_advisor;
```

That is all PostgreSQL setup you need. SQLAlchemy creates the tables (`users`, `conversations`, `messages`) automatically when the backend starts.

---

## 2. Backend Setup

### 2.1 Create and activate a virtual environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 2.3 Configure environment variables

Create a file called `.env` inside the `backend/` folder with the following content:

```env
SECRET_KEY=<generate a long random string>
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DATABASE_URL=postgresql://<username>:<password>@localhost:5432/legal_advisor
GROQ_API_KEY=<your key from console.groq.com>
```

Replace `<username>` and `<password>` with your PostgreSQL credentials (commonly `postgres` / `admin` or whatever you set during installation).

To generate a secure `SECRET_KEY` you can run:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 3. Ingest Law Documents (run once)

The ingest script reads PDFs from the `documents/` folder, chunks them, creates embeddings, and loads them into ChromaDB. You only need to run this once — the data persists between server restarts.

Drop any Pakistani law PDFs (e.g. the Pakistan Penal Code) into the `documents/` folder at the root of the project, then run from the `backend/` directory with the virtual environment active:

```bash
# Ingest all PDFs and .txt files from documents/
python ingest.py

# To wipe ChromaDB and re-ingest (e.g. after updating a document)
python ingest.py --reset

# Ingest a single file
python ingest.py --file "path/to/some_law.pdf"
```

On first run the embedding model (~120 MB) is downloaded from HuggingFace automatically. You will see output like:

```
Found 1 file(s) to ingest:
  • Pakistan Penal Code.pdf

[reading] Pakistan Penal Code.pdf
  Extracted 1,243,872 characters of text
  Split into 847 chunks
  Embedding and storing...

Done. 847 total chunks stored in ChromaDB.
```

---

## 4. Run the Backend

From the `backend/` directory with the virtual environment active:

```bash
uvicorn main:app --reload
```

The API is now running at:

| URL | Purpose |
|---|---|
| `http://localhost:8000` | API root |
| `http://localhost:8000/docs` | Interactive API docs (test every endpoint here) |

The `--reload` flag restarts the server automatically when you change a Python file. Remove it in production.

---

## 5. Run the Frontend

Open a **new terminal** (keep the backend running), then:

```bash
cd frontend
npm install
npm run dev
```

The UI is now running at **`http://localhost:5173`**.

Vite proxies all `/auth` and `/chat` requests to `localhost:8000` automatically, so no CORS issues during development.

---

## Full Startup Checklist

```
[ ] PostgreSQL is running
[ ] legal_advisor database exists
[ ] backend/.env is configured
[ ] Virtual environment is active
[ ] python ingest.py has been run at least once
[ ] uvicorn main:app --reload  (terminal 1)
[ ] npm run dev                (terminal 2)
```

---

## Adding More Law Documents

Drop any `.pdf` or `.txt` file into the `documents/` folder, then run:

```bash
python ingest.py           # adds the new file alongside existing data
python ingest.py --reset   # clears everything and re-ingests all documents
```

---

## Project Structure

```
legal_advisor/
├── documents/              # Drop Pakistani law PDFs here
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── database.py         # PostgreSQL connection
│   ├── models.py           # SQLAlchemy table definitions
│   ├── schemas.py          # Pydantic request/response shapes
│   ├── security.py         # Password hashing + JWT
│   ├── ingest.py           # One-time document ingestion script
│   ├── .env                # Secrets (never commit this)
│   ├── requirements.txt    # Python dependencies
│   ├── chroma_db/          # Vector store (created after running ingest.py)
│   ├── routers/
│   │   ├── auth.py         # /auth/register, /auth/login, /auth/me
│   │   └── chat.py         # /chat/message, /chat/conversations
│   └── services/
│       └── ai_service.py   # ChromaDB search + Groq chat
└── frontend/
    ├── src/
    │   ├── api/client.js       # All fetch calls to the backend
    │   ├── context/AuthContext.jsx
    │   ├── pages/              # LoginPage, RegisterPage, ChatPage
    │   └── components/         # Sidebar, ChatWindow, ChatInput, MessageBubble
    └── package.json
```
