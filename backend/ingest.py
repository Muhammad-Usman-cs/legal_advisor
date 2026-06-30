"""
Document ingestion script — loads Pakistani law documents into ChromaDB.

Supported sources (auto-detected in priority order):
  1. JSON file  — the pdf_data.json dataset (967 laws pre-extracted from PDFs)
  2. PDF files  — individual PDFs in the documents directory
  3. TXT files  — plain text law documents

Usage (run from backend/ with venv active):
    python ingest.py                    # auto-detects source in ../documents/
    python ingest.py --reset            # clears ChromaDB first, then re-ingests
    python ingest.py --dir path/to/     # custom documents directory
    python ingest.py --file mylaw.pdf   # single PDF or TXT file
    python ingest.py --json path/to.json # explicit JSON dataset path
"""

import argparse
import json
import re
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "pakistani_law"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"

# BGE-small-en-v1.5 max sequence length is 512 tokens (~2000 chars).
# 500 chars keeps every chunk well within that limit.
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

# Entries shorter than this after stripping are almost certainly empty
# PDF extractions — skip them to avoid polluting the vector store.
MIN_TEXT_LENGTH = 200


# ---------------------------------------------------------------------------
# TITLE EXTRACTION
# The pdf_data.json filenames are unreadable hashes like
# "administrator00532129aba2e10fe634ab8fbd94c50b.pdf".
# We extract a human-readable title from the text instead so that the
# metadata attached to every chunk is meaningful for debugging retrieval.
# ---------------------------------------------------------------------------

# Common patterns found at the start of Pakistani law documents
_TITLE_PATTERNS = re.compile(
    r'(THE\s+[A-Z][A-Z\s,\(\)]+(?:ACT|ORDINANCE|CODE|ORDER|RULES?|REGULATIONS?)'
    r'|[A-Z][A-Z\s,\(\)]+(?:ACT|ORDINANCE|CODE|ORDER|RULES?|REGULATIONS?)'
    r'|CONSTITUTION\s+OF\s+PAKISTAN)',
    re.IGNORECASE
)

def _extract_title(text: str, fallback: str) -> str:
    """
    Extract the law title from the first ~500 chars of text.
    Falls back to the filename if no recognisable title pattern is found.
    """
    head = text[:500]
    match = _TITLE_PATTERNS.search(head)
    if match:
        # Collapse internal whitespace and trim
        title = re.sub(r'\s+', ' ', match.group(0)).strip()
        return title[:120]
    # Second attempt: first non-empty line that is longer than 15 chars
    for line in head.splitlines():
        line = line.strip()
        if len(line) > 15:
            return line[:120]
    return fallback


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_json_dataset(path: Path) -> list[Document]:
    """
    Load the pdf_data.json dataset.
    Each entry becomes one Document — the splitter will chunk it afterwards.
    Entries with very little text (failed PDF extractions) are skipped.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document] = []
    skipped = 0
    for item in data:
        text     = item.get("text", item.get("content", "")).strip()
        filename = item.get("file_name", "unknown.pdf")

        if len(text) < MIN_TEXT_LENGTH:
            skipped += 1
            continue

        title = _extract_title(text, fallback=filename)
        docs.append(Document(
            page_content=text,
            metadata={
                "source":    title,       # human-readable law name
                "file_name": filename,    # original filename for traceability
            }
        ))

    print(f"  Loaded {len(docs)} laws ({skipped} skipped — below {MIN_TEXT_LENGTH} char minimum)")
    return docs


def load_pdf(path: Path) -> list[Document]:
    """PyPDFLoader returns one Document per page with page number metadata."""
    loader = PyPDFLoader(str(path))
    docs   = loader.load()
    print(f"  Loaded {len(docs)} pages from {path.name}")
    return docs


def load_txt(path: Path) -> list[Document]:
    loader = TextLoader(str(path), autodetect_encoding=True)
    docs   = loader.load()
    print(f"  Loaded {len(docs)} document(s) from {path.name}")
    return docs


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    elif suffix == ".txt":
        return load_txt(path)
    elif suffix == ".json":
        return load_json_dataset(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest Pakistani law documents into ChromaDB.")
    parser.add_argument("--dir",   default=None, help="Directory to scan (default: ../documents/)")
    parser.add_argument("--file",  default=None, help="Ingest a single PDF or TXT file")
    parser.add_argument("--json",  default=None, help="Explicit path to the JSON law dataset")
    parser.add_argument("--reset", action="store_true", help="Clear ChromaDB before ingesting")
    args = parser.parse_args()

    # ── Embeddings ─────────────────────────────────────────────────────────
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Optional reset ──────────────────────────────────────────────────────
    if args.reset:
        print("Resetting ChromaDB collection...")
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        db.delete_collection()
        print("Collection cleared.\n")

    # ── Resolve source ──────────────────────────────────────────────────────
    docs_dir = Path(args.dir) if args.dir else Path(__file__).parent.parent / "documents"

    if args.file:
        # Single file explicitly specified
        all_raw_docs = load_file(Path(args.file))

    elif args.json:
        # Explicit JSON dataset path
        all_raw_docs = load_json_dataset(Path(args.json))

    else:
        # Auto-detect: prefer JSON dataset, fall back to individual PDFs/TXTs
        if not docs_dir.exists():
            print(f"Error: documents directory not found at {docs_dir}")
            sys.exit(1)

        json_files = list(docs_dir.glob("*.json"))
        pdf_files  = [f for f in docs_dir.iterdir() if f.suffix.lower() in (".pdf", ".txt")]

        if json_files:
            # JSON dataset found — use it (covers all 967 laws at once)
            print(f"Found JSON dataset: {json_files[0].name}")
            all_raw_docs = load_json_dataset(json_files[0])
        elif pdf_files:
            print(f"Found {len(pdf_files)} PDF/TXT file(s)")
            all_raw_docs = []
            for p in pdf_files:
                print(f"\n[reading] {p.name}")
                try:
                    all_raw_docs.extend(load_file(p))
                except Exception as e:
                    print(f"  ERROR: {e}")
        else:
            print(f"No .json, .pdf, or .txt files found in {docs_dir}")
            sys.exit(1)

    if not all_raw_docs:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    # ── Split ───────────────────────────────────────────────────────────────
    # RecursiveCharacterTextSplitter tries paragraph breaks first (\n\n),
    # then newlines, then sentence ends, then spaces — never cuts mid-word.
    # chunk_overlap=100 ensures a sentence at a chunk boundary appears in
    # both adjacent chunks so no context is lost at the seam.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_raw_docs)
    print(f"\nSplit {len(all_raw_docs)} documents → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # ── Embed + store ───────────────────────────────────────────────────────
    print("Embedding and storing in ChromaDB (this may take a few minutes)...")

    # Process in batches so progress is visible and memory stays bounded
    BATCH = 500
    db = None
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PATH,
                collection_name=COLLECTION_NAME,
            )
        else:
            db.add_documents(batch)
        print(f"  Stored {min(i + BATCH, len(chunks))} / {len(chunks)} chunks...")

    total = db._collection.count()
    print(f"\nDone. {total} total chunks in ChromaDB.")


if __name__ == "__main__":
    main()
