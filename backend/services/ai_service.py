import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# This is the personality and instruction set given to Groq on every request.
# It is injected as the first "system" message in the conversation.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable and empathetic Pakistani legal advisor.

Your role:
- Answer legal questions strictly based on Pakistani law (Constitution of Pakistan,
  PPC, CPC, CRPC, family law, property law, labour law, etc.)
- When law sections are provided in the context below, cite them by name and section number
- Clearly distinguish between general legal information and formal legal advice
- Always recommend consulting a licensed Pakistani lawyer (Advocate) for serious matters
- Respond in the same language the user writes in (English or Roman/Urdu)
- Keep answers structured: state the relevant law first, then explain its practical effect

What you must NOT do:
- Invent law sections or cite laws that were not provided to you
- Give definitive rulings — you inform, not adjudicate
- Discuss laws of other countries unless explicitly asked for comparison
"""


# ---------------------------------------------------------------------------
# EMBEDDING SERVICE  (sentence-transformers + ChromaDB)
#
# sentence-transformers runs entirely locally — no API key, no rate limits.
# Model: paraphrase-multilingual-MiniLM-L12-v2
#   - Supports 50+ languages including Urdu
#   - Fast on CPU, good quality for semantic search
#
# ChromaDB stores the vectors on disk in ./chroma_db so they persist
# between server restarts. You only need to ingest documents once.
# ---------------------------------------------------------------------------

class EmbeddingService:
    def __init__(self):
        print("Loading embedding model... (first run downloads ~120MB)")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # PersistentClient saves the vector store to disk
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # get_or_create_collection: safe to call on every startup
        # cosine distance is best for semantic similarity of text
        self.collection = self.client.get_or_create_collection(
            name="pakistani_law",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"ChromaDB ready — {self.collection.count()} law sections loaded.")

    def add_documents(
        self,
        texts: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None
    ) -> None:
        """
        Embed and store a batch of law document chunks in ChromaDB.
        Call this once when ingesting your Pakistani law documents.

        texts     — the actual text of each law section
        ids       — unique string IDs (e.g. "ppc_section_302")
        metadatas — optional dicts with extra info (e.g. {"source": "PPC", "section": "302"})
        """
        embeddings = self.model.encode(texts, show_progress_bar=True).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas or [{} for _ in texts]
        )
        print(f"Added {len(texts)} documents. Total: {self.collection.count()}")

    def search(self, query: str, n_results: int = 5) -> list[str]:
        """
        Find the most relevant law sections for a given user query.
        Returns a list of raw text strings to inject into the Groq prompt.
        """
        total = self.collection.count()
        if total == 0:
            return []  # no documents ingested yet

        query_embedding = self.model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, total)  # can't request more than what exists
        )
        return results["documents"][0] if results["documents"] else []


# ---------------------------------------------------------------------------
# GROQ CHAT SERVICE
#
# Groq is a hosted API that runs open-source models (Llama 3.1) at very
# high speed. The interface is identical to OpenAI's Chat Completions API.
#
# How RAG + memory works together:
#   1. We search ChromaDB for law sections relevant to the user's message
#   2. We inject those sections into the system prompt
#   3. We include the full conversation history so Groq "remembers" the chat
#   4. Groq sees: [system + law context] + [prior messages] + [new message]
# ---------------------------------------------------------------------------

class GroqChatService:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def get_reply(
        self,
        user_message: str,
        history: list[dict],       # [{"role": "user"|"assistant", "content": "..."}]
        law_context: list[str]     # retrieved law sections from ChromaDB
    ) -> str:
        """
        Build the full prompt and call Groq.

        The messages list sent to Groq looks like:
          [
            {"role": "system",    "content": "<SYSTEM_PROMPT + law sections>"},
            {"role": "user",      "content": "previous message 1"},
            {"role": "assistant", "content": "previous reply 1"},
            ...
            {"role": "user",      "content": "<current user message>"}
          ]
        """
        system_content = SYSTEM_PROMPT
        if law_context:
            sections = "\n\n---\n\n".join(law_context)
            system_content += f"\n\nRelevant Pakistani Law Sections:\n{sections}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,    # low temperature = more consistent, factual answers
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# SINGLETONS
#
# We create one instance of each service when the module is first imported.
# This means the embedding model and ChromaDB load once at startup,
# not on every request. Both instances are imported directly in chat.py.
# ---------------------------------------------------------------------------

embedding_service = EmbeddingService()
groq_service = GroqChatService()
