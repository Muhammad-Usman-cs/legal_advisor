import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# Strict grounding rule added: model must only cite sections in [LAW CONTEXT].
# Without this, the LLM fills gaps from training memory and invents article
# numbers that sound plausible but are wrong.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable Pakistani legal advisor operating in STRICT GROUNDING MODE.

RULES YOU MUST NEVER BREAK:
- Only cite law sections that appear word-for-word in the [LAW CONTEXT] block below.
- If the answer requires a section NOT present in [LAW CONTEXT], respond:
  "I could not find the specific provision in my loaded documents. Please verify on
   pakistanlaw.pk or consult a licensed Advocate."
- Never invent, guess, or recall section numbers from memory.

Your role:
- Answer legal questions based on Pakistani law (Constitution, PPC, CPC, CRPC, family law, property law, labour law)
- Cite sections by name and number when they appear in [LAW CONTEXT]
- Clearly distinguish between legal information and formal legal advice
- Always recommend consulting a licensed Pakistani Advocate for serious matters
- Respond in the same language the user writes in (English or Roman/Urdu)
- Keep answers structured: relevant law first, then its practical effect

[LAW CONTEXT]
{context}"""

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pakistani_law"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _format_docs(docs: list) -> str:
    # ---------------------------------------------------------------------------
    # Converts a list of LangChain Document objects into a single string that
    # gets injected into the {context} slot in SYSTEM_PROMPT above.
    # Each document is separated by --- so the LLM can distinguish sections.
    # ---------------------------------------------------------------------------
    if not docs:
        return "No relevant law sections found in the loaded documents."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# LegalAdvisorChain
#
# This single class replaces the old EmbeddingService + GroqChatService pair.
# LangChain lets us express the full RAG pipeline as one composable chain
# using LCEL (LangChain Expression Language) and the | pipe operator.
#
# Pipeline (what happens on every get_reply() call):
#   search_query  ──► retriever ──► _format_docs ──► {context}  ─┐
#   question      ───────────────────────────────► {question}    ─┤─► prompt ──► llm ──► str
#   history       ───────────────────────────────► {history}     ─┘
# ---------------------------------------------------------------------------

class LegalAdvisorChain:
    def __init__(self):
        # -----------------------------------------------------------------
        # 1. EMBEDDINGS — HuggingFaceEmbeddings wraps sentence-transformers.
        #    BAAI/bge-small-en-v1.5 is a retrieval-optimised model trained on
        #    100M+ question-passage pairs. Much better than the old paraphrase
        #    model for matching user questions to law document passages.
        #    normalize_embeddings=True is required by BGE for cosine similarity.
        # -----------------------------------------------------------------
        print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # -----------------------------------------------------------------
        # 2. VECTOR STORE — LangChain's Chroma wrapper.
        #    Same ./chroma_db directory as before; LangChain manages the
        #    ChromaDB client internally so we don't call chromadb directly.
        # -----------------------------------------------------------------
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = self.vectorstore._collection.count()
        print(f"ChromaDB ready — {count} law sections loaded.")

        # -----------------------------------------------------------------
        # 3. RETRIEVER — as_retriever() converts the vector store into a
        #    LangChain Retriever object that the chain can call directly.
        #
        #    search_type="mmr": Maximal Marginal Relevance.
        #    Instead of returning the top-5 most similar chunks (which may
        #    all be from the same section), MMR fetches 15 candidates then
        #    picks 5 that are both relevant AND diverse.
        #    lambda_mult=0.7 means 70% relevance weight, 30% diversity weight.
        # -----------------------------------------------------------------
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7},
        )

        # -----------------------------------------------------------------
        # 4. LLM — ChatGroq wraps the Groq API with LangChain's chat model
        #    interface. It returns a LangChain AIMessage object, not a raw
        #    string, which is why StrOutputParser() is needed at the end.
        # -----------------------------------------------------------------
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
        )

        # -----------------------------------------------------------------
        # 5. PROMPT — ChatPromptTemplate.from_messages() builds a structured
        #    prompt with named slots ({context}, {question}, {history}).
        #
        #    MessagesPlaceholder(variable_name="history") inserts the full
        #    list of prior HumanMessage/AIMessage objects at that position.
        #    This is how LangChain handles multi-turn conversation memory.
        # -----------------------------------------------------------------
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # -----------------------------------------------------------------
        # 6. CHAIN (LCEL — LangChain Expression Language)
        #
        #    The | operator chains Runnables together. A dict of Runnables
        #    runs in parallel and merges results into one dict.
        #
        #    itemgetter("search_query") extracts that key from the input dict,
        #    passes it to the retriever, then to _format_docs to get a string.
        #    itemgetter("question") and itemgetter("history") pass through unchanged.
        #
        #    The merged dict feeds into the prompt template, which fills
        #    {context}, {question}, {history}. The result is a list of
        #    BaseMessages that the LLM receives. StrOutputParser pulls out
        #    just the text content from the AIMessage the LLM returns.
        # -----------------------------------------------------------------
        self.chain = (
            {
                "context": itemgetter("search_query") | self.retriever | _format_docs,
                "question": itemgetter("question"),
                "history": itemgetter("history"),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_reply(self, user_message: str, history: list[dict]) -> str:
        # -----------------------------------------------------------------
        # Build a context-aware search query.
        # If the user asks a follow-up ("what about for minors?"), the
        # raw message alone won't retrieve the right section. Prepending
        # the last user turn gives the retriever more context.
        # -----------------------------------------------------------------
        recent_user_turns = [m["content"] for m in history if m["role"] == "user"][-1:]
        search_query = " ".join(recent_user_turns + [user_message])

        # -----------------------------------------------------------------
        # Convert history from our DB format {"role": ..., "content": ...}
        # to LangChain message objects (HumanMessage / AIMessage).
        # MessagesPlaceholder in the prompt expects this exact format.
        # -----------------------------------------------------------------
        lc_history = []
        for msg in history:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            else:
                lc_history.append(AIMessage(content=msg["content"]))

        return self.chain.invoke({
            "question": user_message,
            "search_query": search_query,
            "history": lc_history,
        })


# ---------------------------------------------------------------------------
# SINGLETON
# Loaded once at module import time. The embedding model and ChromaDB
# connection are shared across all requests — same pattern as before.
# ---------------------------------------------------------------------------

legal_chain = LegalAdvisorChain()
