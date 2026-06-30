import logging
import os
import re
from operator import itemgetter

from dotenv import load_dotenv
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "pakistani_law"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
# Assumption: ms-marco-MiniLM-L-6-v2 chosen because sentence-transformers is
# already in the project. It is ~24 MB and fast on CPU. Flag if a larger/
# domain-specific reranker is preferred (e.g. bge-reranker-base).
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_K    = 15   # candidates from sparse retrieval
DENSE_K   = 15   # candidates from dense retrieval (before RRF)
FINAL_K   = 5    # chunks sent to the LLM after reranking

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
#
# Updated with explicit [SRC-N] citation format (soft enforcement).
# Each retrieved chunk is tagged [SRC-1], [SRC-2], ... in [LAW CONTEXT].
# The model is told to cite these IDs after every factual claim so the
# hard-enforcement step can verify them post-generation.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledgeable Pakistani legal advisor operating in STRICT GROUNDING MODE.

RULES YOU MUST NEVER BREAK:
- Only cite law sections that appear in the [LAW CONTEXT] block below.
- Each section is tagged with a source ID like [SRC-1]. After every factual claim,
  you MUST append the source ID in square brackets, e.g.:
  "Theft is punishable by up to three years imprisonment [SRC-2]."
- If the answer requires a section NOT present in [LAW CONTEXT], respond:
  "I could not find the specific provision in my loaded documents. Please verify on
   pakistanlaw.pk or consult a licensed Advocate."
- Never invent, guess, or recall section numbers from memory.

Your role:
- Answer legal questions based on Pakistani law (Constitution, PPC, CPC, CRPC,
  family law, property law, labour law)
- Clearly distinguish between legal information and formal legal advice
- Always recommend consulting a licensed Pakistani Advocate for serious matters
- Respond in the same language the user writes in (English or Roman/Urdu)
- Keep answers structured: relevant law first, then its practical effect

[LAW CONTEXT]
{context}"""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _format_docs_with_ids(docs: list[Document]) -> tuple[str, set[str]]:
    """
    Tag each retrieved chunk with a stable [SRC-N] ID and build the context
    string injected into the prompt.

    Returns:
        context_str — the formatted string for the {context} slot
        valid_ids   — set of IDs like {"SRC-1", "SRC-2"} that actually exist
                      in this retrieval batch; used by _enforce_citations().
    """
    if not docs:
        return "No relevant law sections found in the loaded documents.", set()

    parts = []
    valid_ids: set[str] = set()
    for i, doc in enumerate(docs, start=1):
        src_id = f"SRC-{i}"
        valid_ids.add(src_id)
        # Include the source filename so the LLM can mention the document name
        source = doc.metadata.get("source", "unknown document")
        page   = doc.metadata.get("page", "")
        loc    = f"{source}, p.{page}" if page != "" else source
        parts.append(f"[{src_id}] ({loc})\n{doc.page_content}")

    return "\n\n---\n\n".join(parts), valid_ids


def _enforce_citations(reply: str, valid_ids: set[str]) -> tuple[str, list[str]]:
    """
    HARD citation enforcement — runs after the LLM produces its reply.

    Two checks:
    1. Any [SRC-N] the model cited that is NOT in valid_ids is a hallucinated
       source — the model invented a reference that was never retrieved.
    2. If the reply makes no citations at all but valid_ids is non-empty, the
       model ignored the grounding instruction entirely.

    Neither check modifies the reply text (we flag, not silently delete).
    Warnings are logged server-side; they do not appear in the user-facing reply.

    Returns (reply, list_of_warning_strings).
    """
    warnings: list[str] = []

    # Extract every [SRC-N] token the model wrote
    cited_tokens = set(re.findall(r'\[SRC-\d+\]', reply))
    # Normalise to the bare ID (strip brackets) for set comparison
    cited_ids = {t[1:-1] for t in cited_tokens}  # "SRC-1", "SRC-2", ...

    # Check 1: hallucinated citations
    hallucinated = cited_ids - valid_ids
    if hallucinated:
        warnings.append(
            f"CITATION_HALLUCINATION: model cited {hallucinated} "
            f"which were not in the retrieved set {valid_ids}"
        )

    # Check 2: no citations at all despite having retrieved content
    if valid_ids and not cited_ids:
        warnings.append(
            "CITATION_MISSING: model produced a reply with no [SRC-N] citations "
            "even though law sections were provided."
        )

    for w in warnings:
        logger.warning(w)

    return reply, warnings


# ---------------------------------------------------------------------------
# LegalAdvisorChain
#
# Full RAG pipeline:
#   get_reply()
#     ├─ 1. SPARSE  : BM25Retriever          → top-15 by keyword match
#     ├─ 2. DENSE   : ChromaDB MMR retriever → top-15 by semantic similarity
#     ├─ 3. FUSION  : EnsembleRetriever (RRF)→ merged & de-duplicated ranked list
#     ├─ 4. RERANK  : CrossEncoder           → top-5 by cross-encoder score
#     ├─ 5. TAG     : _format_docs_with_ids  → [SRC-N] tagged context string
#     ├─ 6. GENERATE: prompt | llm | parser  → reply text
#     └─ 7. ENFORCE : _enforce_citations     → log any citation violations
# ---------------------------------------------------------------------------

class LegalAdvisorChain:
    def __init__(self):

        # -----------------------------------------------------------------
        # DENSE: embeddings + ChromaDB vector store
        # -----------------------------------------------------------------
        print("Loading embedding model (BAAI/bge-small-en-v1.5)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = self.vectorstore._collection.count()
        print(f"ChromaDB ready — {count} law sections loaded.")

        # Dense retriever: MMR fetches 30 candidates, returns 15 diverse ones.
        # fetch_k > k so MMR has enough candidates to enforce diversity.
        self.dense_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": DENSE_K, "fetch_k": DENSE_K * 2, "lambda_mult": 0.7},
        )

        # -----------------------------------------------------------------
        # SPARSE: BM25Retriever built from ChromaDB's full document set.
        # BM25 excels at exact keyword matches — critical for legal queries
        # like "Section 379" or "Article 25" where the number matters more
        # than semantic similarity.
        # The index is built once at startup from whatever is in ChromaDB.
        # It must be rebuilt (restart the server) after re-ingestion.
        # -----------------------------------------------------------------
        self.bm25_retriever = self._build_bm25_retriever(count)

        # -----------------------------------------------------------------
        # FUSION: EnsembleRetriever uses Reciprocal Rank Fusion (RRF)
        # internally to merge the BM25 and dense ranked lists.
        #
        # RRF formula per document: score = Σ weight_i / (k + rank_i)
        # where k=60 (LangChain default, reduces sensitivity to top ranks).
        #
        # weights=[0.4, 0.6]: sparse gets 40%, dense gets 60%.
        # Dense slightly favoured because semantic understanding matters more
        # for open-ended legal questions. Flip to [0.6, 0.4] if users mostly
        # search by section number.
        # -----------------------------------------------------------------
        if self.bm25_retriever is not None:
            self.retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.dense_retriever],
                weights=[0.4, 0.6],
            )
        else:
            # No documents ingested yet — fall back to dense-only
            self.retriever = self.dense_retriever

        # -----------------------------------------------------------------
        # RERANKER: CrossEncoder scores (query, chunk) pairs jointly.
        # Unlike bi-encoders (which embed query and chunk separately), a
        # cross-encoder sees both at once and gives a much more accurate
        # relevance score. Used as a second-pass precision step after the
        # cheap first-pass retrieval above.
        # ms-marco-MiniLM-L-6-v2 is ~24 MB and runs in ~80ms/request on CPU.
        # -----------------------------------------------------------------
        print(f"Loading reranker ({RERANKER_MODEL})...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # -----------------------------------------------------------------
        # LLM + GENERATION CHAIN
        # The retrieval pipeline now runs explicitly in get_reply(), so the
        # LCEL chain handles only prompt → LLM → parse (no retriever pipe).
        # -----------------------------------------------------------------
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Generation-only chain: context is pre-built by get_reply()
        self.generation_chain = prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------

    def _build_bm25_retriever(self, doc_count: int):
        """
        Load all documents from ChromaDB and build a BM25 index.
        Fetches in batches to avoid SQLite's SQL-variable limit (hit at ~110K chunks).
        """
        if doc_count == 0:
            logger.warning("ChromaDB is empty — BM25 retriever skipped. Run ingest.py first.")
            return None

        docs: list[Document] = []
        batch_size = 5000
        offset = 0
        print(f"Building BM25 index over {doc_count} chunks (batching {batch_size} at a time)...")
        while offset < doc_count:
            raw = self.vectorstore.get(
                include=["documents", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            for text, meta in zip(raw["documents"], raw["metadatas"]):
                docs.append(Document(page_content=text, metadata=meta or {}))
            offset += batch_size

        retriever = BM25Retriever.from_documents(docs, k=BM25_K)
        print(f"BM25 index built over {len(docs)} chunks.")
        return retriever

    def _rerank(self, query: str, docs: list[Document], top_k: int = FINAL_K) -> list[Document]:
        """
        Second-pass precision scoring with a cross-encoder.
        The cross-encoder reads (query, chunk) together — not separately —
        so it catches relevance signals that bi-encoder embeddings miss.
        Returns the top_k documents sorted by cross-encoder score descending.
        """
        if not docs:
            return []
        pairs  = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)  # float array, one per pair
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]

    def get_reply(self, user_message: str, history: list[dict]) -> str:
        # Build context-aware search query (last user turn + current message)
        recent_user = [m["content"] for m in history if m["role"] == "user"][-1:]
        search_query = " ".join(recent_user + [user_message])

        # Convert history to LangChain message objects for MessagesPlaceholder
        lc_history = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in history
        ]

        # ── Step 1 + 2 + 3: Hybrid retrieval with RRF fusion ──────────────
        candidates = self.retriever.invoke(search_query)

        # ── Step 4: Rerank with cross-encoder ─────────────────────────────
        reranked = self._rerank(search_query, candidates, top_k=FINAL_K)

        # ── Step 5: Tag chunks with [SRC-N] IDs (soft citation support) ───
        context, valid_ids = _format_docs_with_ids(reranked)

        # ── Step 6: Generate ───────────────────────────────────────────────
        reply = self.generation_chain.invoke({
            "context":  context,
            "question": user_message,
            "history":  lc_history,
        })

        # ── Step 7: Hard citation enforcement (log violations server-side) ─
        reply, _ = _enforce_citations(reply, valid_ids)

        return reply


# ---------------------------------------------------------------------------
# SINGLETON — loaded once at module import; shared across all requests.
# ---------------------------------------------------------------------------
legal_chain = LegalAdvisorChain()
