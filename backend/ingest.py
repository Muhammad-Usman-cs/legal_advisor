"""
Document ingestion script — loads Pakistani law documents into ChromaDB.

Usage (run from the backend/ directory with venv active):
    python ingest.py                   # ingests all PDFs and .txt files from ../documents/
    python ingest.py --reset           # clears ChromaDB first, then re-ingests
    python ingest.py --dir path/to/    # custom documents directory
    python ingest.py --file mylaw.pdf  # ingest a single file

Run this once. After that, uvicorn main:app serves the populated vector store.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pypdf

# ---------------------------------------------------------------------------
# We import the singleton from ai_service so we use the same ChromaDB
# collection and embedding model that the running API uses.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from services.ai_service import embedding_service


# ---------------------------------------------------------------------------
# CHUNKING
#
# Why chunk?  The sentence-transformers model has a 512-token limit.
# Feeding a 500-page PDF as one string would get truncated and produce
# a useless embedding. Instead we split into small, focused passages.
#
# Strategy:
#   1. Try to split on section headings (e.g. "302.", "Section 5", "CHAPTER I")
#      so each chunk maps to one legal section — ideal for citation.
#   2. If a section is still too large, split further with character-level
#      chunking + overlap so we don't cut a sentence mid-thought.
# ---------------------------------------------------------------------------

SECTION_PATTERNS = [
    r'\n(?=\d{1,4}\.\s+[A-Z])',          # "302. Punishment of..."
    r'\n(?=Section\s+\d+)',               # "Section 5..."
    r'\n(?=CHAPTER\s+[IVXLC]+)',          # "CHAPTER XIV..."
    r'\n(?=Article\s+\d+)',               # "Article 25..."
    r'\n(?=Schedule\s+[IVX]+)',           # "Schedule I..."
]

CHUNK_SIZE = 1200    # characters per chunk  (~180 words)
OVERLAP    = 200     # overlap between chunks (~30 words) so context isn't lost at boundaries


def split_by_sections(text: str) -> list[str]:
    """
    Try to split the text on legal section headings.
    Returns a list of sections; falls back to the whole text as one block
    if no headings are detected.
    """
    combined = "|".join(SECTION_PATTERNS)
    parts = re.split(combined, text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [text]


def chunk_text(text: str) -> list[str]:
    """
    Split a block of text into overlapping fixed-size chunks.
    Used as the second pass when a section is still larger than CHUNK_SIZE.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]

        # Try to break at a sentence boundary (. or \n) so we don't cut mid-sentence
        if end < len(text):
            break_point = max(
                chunk.rfind(". "),
                chunk.rfind(".\n"),
                chunk.rfind("\n\n"),
            )
            if break_point > CHUNK_SIZE // 2:
                end = start + break_point + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - OVERLAP  # step back by overlap amount

    return [c for c in chunks if len(c) > 50]  # drop tiny leftover fragments


def smart_chunk(text: str) -> list[str]:
    """
    Full chunking pipeline:
      1. Split on section headings
      2. For any section still > CHUNK_SIZE, split it further with overlap
    """
    sections = split_by_sections(text)
    final_chunks = []
    for section in sections:
        if len(section) <= CHUNK_SIZE:
            final_chunks.append(section.strip())
        else:
            final_chunks.extend(chunk_text(section))
    return final_chunks


# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------

def extract_pdf(path: Path) -> str:
    """Extract all text from a PDF file."""
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_txt(path: Path) -> str:
    """Read a plain text file, trying common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any supported encoding.")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    elif suffix == ".txt":
        return extract_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}  (only .pdf and .txt are supported)")


# ---------------------------------------------------------------------------
# INGEST ONE FILE
# ---------------------------------------------------------------------------

def ingest_file(path: Path) -> int:
    """
    Extract, chunk, embed, and store one document.
    Returns the number of chunks added to ChromaDB.
    """
    print(f"\n[reading] {path.name}")
    text = extract_text(path)
    print(f"  Extracted {len(text):,} characters of text")

    chunks = smart_chunk(text)
    print(f"  Split into {len(chunks)} chunks")

    # Build unique IDs: "<filename_stem>_chunk_<index>"
    # e.g. "pakistan_penal_code_chunk_0042"
    stem = re.sub(r'\s+', '_', path.stem.lower())
    ids = [f"{stem}_chunk_{i:04d}" for i in range(len(chunks))]

    # Metadata lets you filter or display the source in the future
    metadatas = [
        {"source": path.name, "chunk_index": i}
        for i in range(len(chunks))
    ]

    print(f"  Embedding and storing...")
    embedding_service.add_documents(texts=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest Pakistani law documents into ChromaDB.")
    parser.add_argument("--dir",   default=None, help="Directory containing documents (default: ../documents/)")
    parser.add_argument("--file",  default=None, help="Ingest a single file instead of a directory")
    parser.add_argument("--reset", action="store_true", help="Delete all existing ChromaDB data before ingesting")
    args = parser.parse_args()

    # --- optional reset ---
    if args.reset:
        print("⚠  Resetting ChromaDB collection...")
        embedding_service.client.delete_collection("pakistani_law")
        # Re-create the collection after deletion
        embedding_service.collection = embedding_service.client.get_or_create_collection(
            name="pakistani_law",
            metadata={"hnsw:space": "cosine"}
        )
        print("   Collection cleared.\n")

    # --- resolve paths ---
    if args.file:
        files = [Path(args.file)]
    else:
        docs_dir = Path(args.dir) if args.dir else Path(__file__).parent.parent / "documents"
        if not docs_dir.exists():
            print(f"Error: documents directory not found at {docs_dir}")
            sys.exit(1)
        files = [f for f in docs_dir.iterdir() if f.suffix.lower() in (".pdf", ".txt")]
        if not files:
            print(f"No .pdf or .txt files found in {docs_dir}")
            sys.exit(1)

    print(f"Found {len(files)} file(s) to ingest:")
    for f in files:
        print(f"  • {f.name}")

    # --- ingest ---
    total_chunks = 0
    for file_path in files:
        try:
            total_chunks += ingest_file(file_path)
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")

    print(f"\nDone. {total_chunks} total chunks stored in ChromaDB.")
    print(f"Total documents in collection: {embedding_service.collection.count()}")


if __name__ == "__main__":
    main()
