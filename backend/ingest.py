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

# 800 chars ≈ 200 tokens — well within BGE-small's 512-token limit.
# Increased from 500 to accommodate a full legal provision in one chunk.
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 50

# Entries shorter than this are failed PDF extractions — skip them.
MIN_TEXT_LENGTH = 200

# Post-split: discard fragments shorter than this (leftover TOC lines).
MIN_CHUNK_CHARS = 80


# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Safe PDF extraction cleanup:
    - Normalise line endings
    - Remove leading whitespace after newlines (e.g. "\n 379." → "\n379.")
    - Collapse multiple spaces within a line to one
    """
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Remove indentation that interferes with section-number detection
    text = re.sub(r'\n[ \t]+', '\n', text)
    # Collapse runs of spaces (not newlines) to one space
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text


# ---------------------------------------------------------------------------
# TOC STRIPPING
# ---------------------------------------------------------------------------

def strip_toc(text: str) -> str:
    """
    Remove the Table of Contents that many Pakistani law PDFs have at the top.

    Why this matters:
        The PPC has a TOC at chars 0-30,000 listing entries like:
            "\\n379. Punishment for theft  \\n380. Theft in dwelling house..."
        The ACTUAL provision is at char 381,587:
            "\\n379. Punishment for theft.  Whoever commits theft shall be
             punished with imprisonment..."
        Without stripping, the TOC entries get indexed and retrieved instead
        of the real provisions — causing context_precision = 0.00.

    Detection:
        In the TOC, each section title is short so consecutive section-number
        matches are only ~30-50 chars apart.  In actual content, each section
        has paragraph text so the gap is 200+ chars.  The first section whose
        gap to the NEXT section exceeds 200 chars is where real content begins.
    """
    section_re = re.compile(r'\n\d+[A-Z]?\.\s')
    matches = list(section_re.finditer(text))

    if len(matches) < 5:
        return text  # too few sections to have a meaningful TOC

    for i in range(len(matches) - 1):
        gap = matches[i + 1].start() - matches[i].start()
        if gap > 200:
            # This section has paragraph content after it — real content starts here
            return text[matches[i].start():]

    return text  # no TOC pattern found, use full text


# ---------------------------------------------------------------------------
# TITLE EXTRACTION
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LAW KEY DETECTION
# ---------------------------------------------------------------------------

# Normalised law identifiers written into chunk metadata so ai_service.py
# can filter ChromaDB to a specific law without knowing exact document titles.
_LAW_KEY_PATTERNS: list[tuple[str, list[str]]] = [
    ("ppc",          ["pakistan penal code", "penal code (xlv"]),
    ("constitution", ["constitution of pakistan"]),
    ("cpc",          ["code of civil procedure"]),
    ("crpc",         ["code of criminal procedure", "criminal procedure code"]),
    ("family",       ["muslim family laws", "family courts act",
                      "dissolution of muslim marriages"]),
    ("contract",     ["contract act"]),
    ("labour",       ["industrial relations", "labour", "employment"]),
    ("property",     ["transfer of property", "registration act",
                      "land acquisition", "stamp act"]),
]

def _detect_law_key(title: str, filename: str) -> str:
    combined = (title + " " + filename).lower()
    for key, patterns in _LAW_KEY_PATTERNS:
        if any(p in combined for p in patterns):
            return key
    return "other"


# ---------------------------------------------------------------------------
# TITLE EXTRACTION
# ---------------------------------------------------------------------------

_TITLE_PATTERNS = re.compile(
    r'(THE\s+[A-Z][A-Z\s,\(\)]+(?:ACT|ORDINANCE|CODE|ORDER|RULES?|REGULATIONS?)'
    r'|[A-Z][A-Z\s,\(\)]+(?:ACT|ORDINANCE|CODE|ORDER|RULES?|REGULATIONS?)'
    r'|CONSTITUTION\s+OF\s+PAKISTAN)',
    re.IGNORECASE
)

def _extract_title(text: str, fallback: str) -> str:
    """Extract a human-readable law title from the first 500 chars of text."""
    head = text[:500]
    match = _TITLE_PATTERNS.search(head)
    if match:
        return re.sub(r'\s+', ' ', match.group(0)).strip()[:120]
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
    Each entry is cleaned, stripped of its TOC, then passed to the splitter.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: list[Document] = []
    skipped = 0
    toc_stripped = 0

    for item in data:
        text     = item.get("text", item.get("content", "")).strip()
        filename = item.get("file_name", "unknown.pdf")

        if len(text) < MIN_TEXT_LENGTH:
            skipped += 1
            continue

        text = clean_text(text)

        stripped = strip_toc(text)
        if len(stripped) < len(text) * 0.9:   # >10% removed → TOC was found
            toc_stripped += 1

        title   = _extract_title(text, fallback=filename)
        law_key = _detect_law_key(title, filename)
        docs.append(Document(
            page_content=stripped,
            metadata={
                "source":    title,
                "file_name": filename,
                "law_key":   law_key,   # used by ai_service for filtered retrieval
            }
        ))

    print(f"  Loaded {len(docs)} laws "
          f"({skipped} skipped — below {MIN_TEXT_LENGTH} chars, "
          f"{toc_stripped} TOCs removed)")
    return docs


def load_pdf(path: Path) -> list[Document]:
    """PyPDFLoader returns one Document per page with page-number metadata."""
    loader = PyPDFLoader(str(path))
    docs   = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    print(f"  Loaded {len(docs)} pages from {path.name}")
    return docs


def load_txt(path: Path) -> list[Document]:
    loader = TextLoader(str(path), autodetect_encoding=True)
    docs   = loader.load()
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
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

    # ── Embeddings ─────────────────────────────────────────────────────────────
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Optional reset ──────────────────────────────────────────────────────────
    if args.reset:
        print("Resetting ChromaDB collection...")
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        db.delete_collection()
        print("Collection cleared.\n")

    # ── Resolve source ──────────────────────────────────────────────────────────
    docs_dir = Path(args.dir) if args.dir else Path(__file__).parent.parent / "documents"

    if args.file:
        all_raw_docs = load_file(Path(args.file))
    elif args.json:
        all_raw_docs = load_json_dataset(Path(args.json))
    else:
        if not docs_dir.exists():
            print(f"Error: documents directory not found at {docs_dir}")
            sys.exit(1)

        json_files = list(docs_dir.glob("*.json"))
        pdf_files  = [f for f in docs_dir.iterdir() if f.suffix.lower() in (".pdf", ".txt")]

        if json_files:
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

    # ── Split ───────────────────────────────────────────────────────────────────
    # Section-aware splitting:
    #   Primary separator is a section-number pattern so each chunk begins with
    #   its own section number, e.g. "379. Punishment for theft.  Whoever..."
    #   This keeps the section number + provision text together in one chunk,
    #   which is the root fix for context_precision = 0.00.
    #
    # keep_separator="start" — the matched section pattern stays at the START
    #   of each new chunk, not discarded, so "379." is always in the chunk.
    #
    # chunk_size=800 — safe for BGE-small (512 tokens ≈ 2000 chars; 800 chars
    #   ≈ 200 tokens). Large enough to hold a complete legal provision.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            r'\n\d+[A-Z]?\.\s',   # \n379.  or \n381A.  — section boundary
            r'\n\n',
            r'\n',
            r'\.  ',              # double-space after period (PPC provision format)
            r' ',
            r'',
        ],
        is_separator_regex=True,
        keep_separator="start",
    )
    chunks = splitter.split_documents(all_raw_docs)

    # Discard very short fragments — these are residual TOC lines or page headers
    # that slipped through (e.g. "Page 22 of 179" or a lone section title).
    before = len(chunks)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]
    discarded = before - len(chunks)

    print(f"\nSplit {len(all_raw_docs)} documents → {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, {discarded} short fragments discarded)")

    # ── Embed + store ───────────────────────────────────────────────────────────
    print("Embedding and storing in ChromaDB (this may take a few minutes)...")

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
