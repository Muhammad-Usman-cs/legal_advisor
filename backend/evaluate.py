"""
RAG Evaluation Script -- measures retrieval + generation quality using RAGAS.

The full 967-law corpus stays in ChromaDB untouched.
Evaluation uses a SEPARATE handcrafted Q&A set -- not a split of the corpus.

Why you can't split legal data for evaluation
---------------------------------------------
Standard ML train/test splits work when examples are independent (e.g. images).
Pakistani law is NOT independent -- answering "Is Section 302 related to Section
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
]


# ---------------------------------------------------------------------------
# RAG OUTPUT COLLECTION
# ---------------------------------------------------------------------------

def collect_rag_outputs(eval_set: list[dict]) -> list[dict]:
    """Run each question through the full RAG pipeline and collect answer + contexts."""
    rows = []
    total = len(eval_set)
    for i, item in enumerate(eval_set, 1):
        question = item["question"]
        print(f"  [{i}/{total}] {question[:70]}...")

        reranked = legal_chain.retrieve(question)
        contexts = [doc.page_content for doc in reranked]
        answer     = legal_chain.get_reply(question, history=[])

        rows.append({
            "question":     question,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })

    return rows


# ---------------------------------------------------------------------------
# RAGAS SCORING
# ---------------------------------------------------------------------------

def _patch_langchain_community():
    """
    Shim for RAGAS 0.4.3 which imports langchain_community.chat_models.vertexai
    at startup. That module was removed in langchain-community 0.4.x. Inject an
    empty module so the import does not crash before we even start scoring.
    """
    from types import ModuleType
    key = "langchain_community.chat_models.vertexai"
    if key not in sys.modules:
        try:
            import langchain_google_vertexai as _gv
            shim = ModuleType(key)
            shim.ChatVertexAI = getattr(_gv, "ChatVertexAI", None)
            sys.modules[key] = shim
        except ImportError:
            sys.modules[key] = ModuleType(key)


def run_ragas(rows: list[dict]):
    _patch_langchain_community()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
        from langchain_groq import ChatGroq
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Install with:  pip install ragas==0.4.3 datasets==5.0.0")
        sys.exit(1)

    dataset = Dataset.from_list(rows)

    # Use the small 8B model as judge to avoid burning the 100K/day token quota
    # on evaluation calls (the 70B model is reserved for actual user queries).
    judge_llm = LangchainLLMWrapper(
        ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    )
    judge_emb = RagasHFEmbeddings(model="BAAI/bge-small-en-v1.5")

    print("\nScoring with RAGAS...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    print("\n" + "=" * 50)
    print("RAGAS SCORES")
    print("=" * 50)
    print(results)
    print()

    df = results.to_pandas()
    print("Per-question breakdown:")
    cols      = ["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))

    df.to_csv("eval_results.csv", index=False)
    print("\nFull results saved to eval_results.csv")

    print("\n-- What your scores mean --")
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scores = {
        col: float(df[col].mean()) if col in df.columns else 0.0
        for col in metric_cols
    }
    for metric, score in scores.items():
        if score < 0.7:
            if metric == "faithfulness":
                print(f"  [!] {metric}={score:.2f} -- LLM is hallucinating. Tighten system prompt grounding rules.")
            elif metric == "context_precision":
                print(f"  [!] {metric}={score:.2f} -- Retriever pulls irrelevant chunks. Consider metadata filtering.")
            elif metric == "context_recall":
                print(f"  [!] {metric}={score:.2f} -- Retriever misses the right section. Increase BM25_K/DENSE_K.")
            elif metric == "answer_relevancy":
                print(f"  [!] {metric}={score:.2f} -- Answer is off-topic. Lower temperature or tighten the prompt.")
        else:
            print(f"  [ok] {metric}={score:.2f}")


# ---------------------------------------------------------------------------
# RETRIEVAL-ONLY INSPECTION (no RAGAS install needed)
# ---------------------------------------------------------------------------

def run_retrieval_only(rows: list[dict]):
    """Print retrieved chunks for manual inspection -- no RAGAS scoring."""
    print("\n-- Retrieval-only inspection --")
    for row in rows:
        print(f"\nQ: {row['question']}")
        print(f"Retrieved {len(row['contexts'])} chunks:")
        for i, ctx in enumerate(row['contexts'], 1):
            print(f"  [{i}] {ctx[:200].strip()}...")
        print(f"Answer (first 200 chars): {row['answer'][:200]}...")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Print retrieved chunks only -- no RAGAS scoring (no extra install needed)",
    )
    args = parser.parse_args()

    print(f"Running evaluation on {len(EVAL_SET)} questions...")
    print("(Full 967-law corpus stays in ChromaDB -- no data is split)\n")

    rows = collect_rag_outputs(EVAL_SET)

    if args.retrieval_only:
        run_retrieval_only(rows)
    else:
        run_ragas(rows)
