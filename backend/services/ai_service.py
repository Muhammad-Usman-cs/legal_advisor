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
# LAW DETECTION FOR METADATA-FILTERED RETRIEVAL
#
# When the user's query mentions a specific law, restrict ChromaDB to only
# that law's chunks before retrieval.  This reduces competition from 110K
# chunks to ~2-5K law-specific chunks, so the correct section wins.
#
# The filter is passed as a ChromaDB `where` clause:
#   {"law_key": {"$eq": "ppc"}}
# The `law_key` values must match what ingest.py writes into chunk metadata.
# ---------------------------------------------------------------------------
_LAW_SIGNALS: list[tuple[str, list[str]]] = [
    ("ppc", [
        "ppc", "pakistan penal code", "penal code",
        # common section numbers that only appear in the PPC
        "section 302", "section 300", "section 379", "section 392",
        "section 420", "section 499", "section 311", "section 354",
        # PPC-specific offence names
        "theft", "murder", "robbery", "dacoity", "defamation",
        "culpable homicide", "attempt to murder", "extortion",
        "criminal breach of trust", "cheating", "mischief", "hurt",
        "criminal force", "abduction", "kidnapping",
    ]),
    ("constitution", [
        "constitution", "fundamental rights", "basic rights",
        "article 25", "article 10", "article 8", "article 19", "article 20",
        "parliament", "senate", "national assembly",
        "federalism", "provincial assembly", "writ petition",
    ]),
    ("crpc", [
        "crpc", "code of criminal procedure", "criminal procedure",
        "first information report", "fir", "bail", "cognizable",
        "magistrate", "sessions court", "warrant", "challan",
    ]),
    ("cpc", [
        "cpc", "code of civil procedure", "civil procedure",
        "civil suit", "plaint", "written statement", "decree",
        "civil court", "execution of decree",
    ]),
    ("family", [
        "muslim family laws", "family courts", "divorce", "khula",
        "maintenance", "dower", "mehr", "nikah", "dissolution of marriage",
    ]),
]


def detect_law_filter(query: str) -> tuple[dict | None, str | None]:
    """
    Detect which Pakistani law the query is about.

    Returns:
        (chroma_filter, law_key) where chroma_filter is the ChromaDB `where`
        dict and law_key is the plain string (e.g. "ppc").
        Both are None when no specific law is detected.
    """
    q = query.lower()
    for law_key, signals in _LAW_SIGNALS:
        if any(signal in q for signal in signals):
            return {"law_key": {"$eq": law_key}}, law_key
    return None, None

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

BM25_K    = 20   # candidates from sparse retrieval
DENSE_K   = 20   # candidates from dense retrieval (before RRF)
FINAL_K   = 5    # chunks sent to the LLM after reranking

# Cross-encoder relevance floor.
# ms-marco-MiniLM outputs an UNBOUNDED logit per (query, chunk) pair — NOT a
# 0-1 probability. Higher = more relevant; empirically a score above ~0 means
# the chunk is on-topic, negative means off-topic. Any chunk below this floor
# is dropped from the final set. When NO chunk clears the floor, the LLM
# receives an empty [LAW CONTEXT] and the system prompt's "I could not find
# the specific provision" path fires — instead of the model treating weak,
# irrelevant chunks as authoritative and hallucinating around them.
#
# 0.0 is a conservative starting point. CALIBRATE it: _rerank() logs the top
# scores per query, so run a few known-good and known-bad questions, watch the
# logs, and raise/lower the floor until on-topic chunks pass and off-topic
# ones are rejected.
RERANK_SCORE_FLOOR = 0.0

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


def _strip_source_tags(reply: str) -> str:
    """
    Remove the internal [SRC-N] citation markers before the reply is shown
    to the user.

    The [SRC-N] tags exist only so _enforce_citations() can verify the model
    grounded its answer in the retrieved chunks. They are internal plumbing —
    opaque IDs that mean nothing to an end user — so they must be stripped
    from the user-facing text. This runs AFTER _enforce_citations() so the
    grounding check still sees the original markers.

    Also removes the whitespace immediately preceding each tag so that
    "imprisonment [SRC-2]." cleanly becomes "imprisonment." rather than
    leaving a double space before the period.
    """
    cleaned = re.sub(r'\s*\[SRC-\d+\]', '', reply)   # drop tags + leading space
    cleaned = re.sub(r' {2,}', ' ', cleaned)          # collapse any doubled spaces
    return cleaned.strip()


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
            query_instruction="Represent this sentence for searching relevant passages: ",
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
            temperature=0.0,
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

    def _rrf_merge(self, *ranked_lists: list[Document], k: int = 60) -> list[Document]:
        """
        Reciprocal Rank Fusion across multiple ranked document lists.

        RRF score for each document = sum of 1 / (k + rank_i) across all lists
        where rank_i is its 1-based position in list i (0 if absent).

        k=60 is the standard default — it dampens the impact of the top rank
        so that a document ranked #1 in one list and #3 in another beats a
        document ranked #1 in one list and absent from all others.

        Used in place of EnsembleRetriever when the dense retriever needs a
        dynamic per-query metadata filter (EnsembleRetriever uses a fixed
        retriever object configured at startup, so it can't vary the filter).
        """
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = doc.page_content[:120]          # content fingerprint for dedup
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                doc_map[key] = doc
        return [doc_map[key] for key in sorted(scores, key=scores.__getitem__, reverse=True)]

    def retrieve(self, query: str) -> list[Document]:
        """
        Run the full retrieval pipeline for a query: law detection → filtered
        BM25 + dense → RRF merge → CrossEncoder rerank.
        Exposed so evaluate.py can inspect what the LLM actually receives.
        """
        law_filter, law_key = detect_law_filter(query)

        if law_filter is not None:
            dense_docs = self.vectorstore.similarity_search(query, k=DENSE_K, filter=law_filter)
            bm25_raw   = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
            bm25_docs  = [d for d in bm25_raw if d.metadata.get("law_key") == law_key]
            candidates = self._rrf_merge(bm25_docs, dense_docs)
        else:
            candidates = self.retriever.invoke(query)

        return self._rerank(query, candidates, top_k=FINAL_K)

    def _rerank(self, query: str, docs: list[Document], top_k: int = FINAL_K) -> list[Document]:
        """
        Second-pass precision scoring with a cross-encoder.
        The cross-encoder reads (query, chunk) together — not separately —
        so it catches relevance signals that bi-encoder embeddings miss.
        Returns up to top_k documents sorted by cross-encoder score descending,
        keeping only those whose score clears RERANK_SCORE_FLOOR. If none clear
        it, returns [] — deliberately triggering the abstention path downstream
        (empty context → "I could not find the specific provision" reply).
        """
        if not docs:
            return []
        pairs  = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)  # unbounded logits, one per pair
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        # Log the best scores so RERANK_SCORE_FLOOR can be calibrated on real queries.
        top_preview = [round(float(s), 2) for s, _ in ranked[:top_k]]
        logger.info("Rerank top-%d scores: %s (floor=%.2f)",
                    top_k, top_preview, RERANK_SCORE_FLOOR)

        kept = [doc for score, doc in ranked[:top_k] if float(score) >= RERANK_SCORE_FLOOR]
        if not kept:
            logger.warning(
                "No chunk cleared the rerank floor (%.2f) for query %r — "
                "returning empty context (abstention path).",
                RERANK_SCORE_FLOOR, query[:100],
            )
        return kept

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
        # detect_law_filter() checks for law-specific signals in the query.
        # If a law is detected, dense retrieval is scoped to that law's chunks
        # only, eliminating cross-law false positives (e.g. Hudood Ordinance
        # chunks that reference the PPC outscoring actual PPC sections).
        # BM25 still searches all chunks — it's an in-memory index so no
        # ChromaDB filter can be applied, but RRF down-weights its results
        # when the dense retriever is focused and returns strong matches.
        law_filter, law_key = detect_law_filter(search_query)

        if law_filter is not None:
            logger.info("Applying law filter: %s", law_filter)

            # Dense: ChromaDB restricts to law_key chunks only
            dense_docs = self.vectorstore.similarity_search(
                search_query, k=DENSE_K, filter=law_filter
            )

            # BM25 has no native metadata filter — it returns results from all
            # laws. Post-filter to the same law_key so Hudood Ordinance / other
            # law chunks that merely reference the PPC don't contaminate the
            # RRF merge and beat actual PPC sections.
            bm25_raw  = self.bm25_retriever.invoke(search_query) if self.bm25_retriever else []
            bm25_docs = [d for d in bm25_raw if d.metadata.get("law_key") == law_key]

            candidates = self._rrf_merge(bm25_docs, dense_docs)
        else:
            # No specific law detected — search all chunks via EnsembleRetriever
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

        # ── Step 8: Strip internal [SRC-N] markers from the user-facing text ─
        # Enforcement above already inspected the original markers; the user
        # should never see these opaque internal IDs.
        reply = _strip_source_tags(reply)

        return reply


# ---------------------------------------------------------------------------
# SINGLETON — loaded once at module import; shared across all requests.
# ---------------------------------------------------------------------------
legal_chain = LegalAdvisorChain()
