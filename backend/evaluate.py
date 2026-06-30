"""
RAG Evaluation Script — measures retrieval + generation quality using RAGAS.

The full 967-law corpus stays in ChromaDB untouched.
Evaluation uses a SEPARATE handcrafted Q&A set — not a split of the corpus.

Why you can't split legal data for evaluation
---------------------------------------------
Standard ML train/test splits work when examples are independent (e.g. images).
Pakistani law is NOT independent — answering "Is Section 302 related to Section
311?" requires BOTH sections to be in the retrieval corpus simultaneously.
Holding out sections for a "test set" would make correct answers impossible.

The RAG evaluation solution
----------------------------
1. Keep ALL 967 laws in ChromaDB (the retrieval corpus is never split).
2. Create a separate evaluation set: handcrafted questions with known answers.
3. RAGAS evaluates: did the system retrieve the right section + answer correctly?
   - Context Precision  : were the retrieved chunks actually relevant?
   - Context Recall     : did retrieval include the section containing the answer?
   - Faithfulness       : does the answer stay within what was retrieved?
   - Answer Relevancy   : does the answer address the question?

Run:
    cd backend
    venv\\Scripts\\activate
    pip install ragas datasets
    python evaluate.py
"""

import sys
sys.path.insert(0, ".")

from services.ai_service import legal_chain

# ---------------------------------------------------------------------------
# EVALUATION DATASET
#
# These are hand-verified question/answer pairs drawn from the law corpus.
# ground_truth = the correct answer as it appears in the actual law text.
# Add more entries as you verify more sections from your 967-law dataset.
# ---------------------------------------------------------------------------
EVAL_SET = [
    {
        "question": "What is the punishment for theft under the Pakistan Penal Code?",
        "ground_truth": (
            "Under Section 379 of the Pakistan Penal Code, whoever commits theft "
            "shall be punished with imprisonment of either description for a term "
            "which may extend to three years, or with fine, or with both."
        ),
    },
    {
        "question": "How does the PPC define murder?",
        "ground_truth": (
            "Section 300 of the PPC defines murder as culpable homicide amounting "
            "to murder when the act is done with the intention of causing death, or "
            "with the intention of causing such bodily injury as the offender knows "
            "to be likely to cause death, or is sufficient in the ordinary course "
            "of nature to cause death."
        ),
    },
    {
        "question": "What is the punishment for robbery under Pakistani law?",
        "ground_truth": (
            "Section 392 of the PPC states that whoever commits robbery shall be "
            "punished with rigorous imprisonment for a term which may extend to ten "
            "years, and shall also be liable to fine."
        ),
    },
    {
        "question": "What are the fundamental rights guaranteed by the Constitution of Pakistan?",
        "ground_truth": (
            "Part II of the Constitution of Pakistan 1973 (Articles 8-28) guarantees "
            "fundamental rights including equality before law (Article 25), freedom "
            "of movement (Article 15), freedom of speech (Article 19), freedom of "
            "religion (Article 20), and the right to a fair trial (Article 10-A)."
        ),
    },
    {
        "question": "What constitutes defamation under the Pakistan Penal Code?",
        "ground_truth": (
            "Section 499 of the PPC defines defamation as making or publishing any "
            "imputation concerning a person intending to harm, or knowing or having "
            "reason to believe that it will harm, the reputation of such person. "
            "Section 500 prescribes punishment of simple imprisonment up to two years, "
            "or fine, or both."
        ),
    },
    # ── Add more entries here as you verify them against your documents ──
    # {
    #     "question": "...",
    #     "ground_truth": "...",
    # },
]


def collect_rag_outputs(eval_set: list[dict]) -> list[dict]:
    """
    Run each question through the full RAG pipeline and collect:
      - the generated answer
      - the retrieved context chunks (before generation)
    """
    rows = []
    total = len(eval_set)
    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        print(f"  [{i}/{total}] {question[:70]}...")

        # ── Retrieve (same pipeline as get_reply, but exposed here) ────────
        recent_user   = []                             # no prior history in eval
        search_query  = question
        candidates    = legal_chain.retriever.invoke(search_query)
        reranked      = legal_chain._rerank(search_query, candidates)
        contexts      = [doc.page_content for doc in reranked]

        # ── Generate ────────────────────────────────────────────────────────
        answer = legal_chain.get_reply(question, history=[])

        rows.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,    # list of strings — what RAGAS inspects
            "ground_truth": item["ground_truth"],
        })

    return rows


def _patch_langchain_community():
    """
    RAGAS 0.4.3 imports langchain_community.chat_models.vertexai at startup,
    but langchain-community 0.4.x removed that module (moved to langchain-google-vertexai).
    Inject a shim into sys.modules before RAGAS loads so the import succeeds.
    """
    import sys
    from types import ModuleType
    key = "langchain_community.chat_models.vertexai"
    if key not in sys.modules:
        try:
            import langchain_google_vertexai as _gv
            shim = ModuleType(key)
            shim.ChatVertexAI = getattr(_gv, "ChatVertexAI", None)
            sys.modules[key] = shim
        except ImportError:
            sys.modules[key] = ModuleType(key)  # empty shim — vertexai not installed


def run_ragas(rows: list[dict]):
    """Score the collected outputs with RAGAS metrics."""
    _patch_langchain_community()
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Run: pip install ragas datasets")
        return

    dataset = Dataset.from_list(rows)

    # Reuse the same LLM and embedding model already configured in the project
    judge_llm = LangchainLLMWrapper(
        ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    )
    judge_emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    print("\nScoring with RAGAS...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    print("\n" + "="*50)
    print("RAGAS SCORES")
    print("="*50)
    print(results)
    print()

    # Per-question breakdown
    df = results.to_pandas()
    print("Per-question breakdown:")
    cols = ["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    df.to_csv("eval_results.csv", index=False)
    print("\nFull results saved to eval_results.csv")

    # Guidance based on scores
    print("\n── What your scores mean ──")
    print(results)
    scores = {
        "faithfulness":      float(results.get("faithfulness",      0)),
        "answer_relevancy":  float(results.get("answer_relevancy",  0)),
        "context_precision": float(results.get("context_precision", 0)),
        "context_recall":    float(results.get("context_recall",    0)),
    }
    for metric, score in scores.items():
        if score < 0.7:
            if metric == "faithfulness":
                print(f"  ⚠  {metric}={score:.2f} — LLM is hallucinating. Tighten system prompt grounding rules.")
            elif metric == "context_precision":
                print(f"  ⚠  {metric}={score:.2f} — Retriever pulls irrelevant chunks. Consider raising reranker threshold.")
            elif metric == "context_recall":
                print(f"  ⚠  {metric}={score:.2f} — Retriever misses the right section. Check chunking or increase BM25_K/DENSE_K.")
            elif metric == "answer_relevancy":
                print(f"  ⚠  {metric}={score:.2f} — Answer is off-topic. Lower temperature or tighten the prompt.")
        else:
            print(f"  ✓  {metric}={score:.2f}")


def run_retrieval_only(rows: list[dict]):
    """
    Lightweight evaluation without RAGAS — just prints what was retrieved
    so you can manually verify retrieval quality.
    Useful before installing ragas or when debugging a specific question.
    """
    print("\n── Retrieval-only inspection ──")
    for row in rows:
        print(f"\nQ: {row['question']}")
        print(f"Retrieved {len(row['contexts'])} chunks:")
        for i, ctx in enumerate(row['contexts'], 1):
            print(f"  [{i}] {ctx[:150].strip()}...")
        print(f"Answer (first 200 chars): {row['answer'][:200]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print retrieved chunks only — no RAGAS scoring (no extra install needed)"
    )
    args = parser.parse_args()

    print(f"Running evaluation on {len(EVAL_SET)} questions...")
    print("(Full 967-law corpus stays in ChromaDB — no data is split)\n")

    rows = collect_rag_outputs(EVAL_SET)

    if args.retrieval_only:
        run_retrieval_only(rows)
    else:
        run_ragas(rows)
