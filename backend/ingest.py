"""
Document ingestion script — loads Pakistani law documents into ChromaDB using LangChain.

Usage (run from the backend/ directory with venv active):
    python ingest.py                   # ingests all PDFs and .txt files from ../documents/
    python ingest.py --reset           # clears ChromaDB first, then re-ingests
    python ingest.py --dir path/to/    # custom documents directory
    python ingest.py --file mylaw.pdf  # ingest a single file

Run this once. After that, uvicorn main:app serves the populated vector store.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# LangChain replaces our manual pypdf calls and custom chunking logic.
#
# PyPDFLoader    — LangChain's PDF loader. Wraps pypdf but returns a list of
#                  LangChain Document objects (one per page), each carrying
#                  metadata like {"source": "Pakistan Penal Code.pdf", "page": 3}.
#
# TextLoader     — Same idea for plain .txt files.
#
# RecursiveCharacterTextSplitter — Battle-tested chunker that tries to split
#                  on paragraph breaks (\n\n) first, then newlines, then
#                  sentence ends (". "), then spaces, falling back to raw
#                  character splits. Much more robust than our custom regex
#                  which relied on exact PDF formatting that pypdf often
#                  doesn't reproduce.
# ---------------------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pakistani_law"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# CHUNK SIZE
#
# Old value was 1200 characters. That was larger than the old embedding
# model's 128-token limit (~600 chars), so the second half of every chunk
# was silently truncated and never embedded.
#
# BAAI/bge-small-en-v1.5 supports 512 tokens (~2000 chars), so 500 chars
# is safely within the limit while keeping chunks focused on one topic.
# CHUNK_OVERLAP ensures a sentence that falls at a boundary appears in both
# the preceding and the following chunk so context is never lost.
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_file(path: Path) -> list:
    """
    Use the appropriate LangChain loader for the file type.
    Both loaders return a list of Document objects — same interface,
    so the rest of the pipeline doesn't need to know the file type.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        # PyPDFLoader splits the PDF into one Document per page.
        # Each Document has page_content (extracted text) and metadata
        # {"source": "filename.pdf", "page": 0}.
        loader = PyPDFLoader(str(path))
    elif suffix == ".txt":
        # autodetect_encoding tries utf-8, then latin-1, etc.
        loader = TextLoader(str(path), autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file type: {suffix} (only .pdf and .txt are supported)")

    docs = loader.load()
    print(f"  Loaded {len(docs)} pages/sections from {path.name}")
    return docs


def main():
    parser = argparse.ArgumentParser(description="Ingest Pakistani law documents into ChromaDB.")
    parser.add_argument("--dir",   default=None, help="Directory containing documents (default: ../documents/)")
    parser.add_argument("--file",  default=None, help="Ingest a single file instead of a directory")
    parser.add_argument("--reset", action="store_true", help="Delete all existing ChromaDB data before ingesting")
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # EMBEDDINGS — same model used by ai_service.py so queries and
    # documents live in the same vector space.
    # -----------------------------------------------------------------
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # -----------------------------------------------------------------
    # OPTIONAL RESET — delete the collection so we start clean.
    # Useful when you update a document and want to re-embed everything.
    # -----------------------------------------------------------------
    if args.reset:
        print("Resetting ChromaDB collection...")
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        db.delete_collection()
        print("Collection cleared.\n")

    # -----------------------------------------------------------------
    # COLLECT FILES
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # LOAD — each file becomes a list of Document objects
    # -----------------------------------------------------------------
    all_docs = []
    for file_path in files:
        print(f"\n[reading] {file_path.name}")
        try:
            docs = load_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ERROR loading {file_path.name}: {e}")

    if not all_docs:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # SPLIT — RecursiveCharacterTextSplitter breaks each Document into
    # smaller chunks. It tries paragraph breaks first, then newlines,
    # then sentence ends, falling back to character-level splits.
    #
    # Unlike our old regex approach, this works regardless of whether
    # section headings are formatted consistently in the PDF.
    #
    # The metadata from the loader (source, page) is automatically
    # preserved on every child chunk — so each chunk knows which page
    # of which document it came from.
    # -----------------------------------------------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"\nSplit into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # -----------------------------------------------------------------
    # EMBED + STORE — Chroma.from_documents() embeds every chunk and
    # writes them to the persistent vector store in one call.
    #
    # If the collection already exists (no --reset), from_documents()
    # adds to it rather than replacing it, so you can add new law PDFs
    # incrementally without re-ingesting existing ones.
    # -----------------------------------------------------------------
    print("Embedding and storing in ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    total = db._collection.count()
    print(f"\nDone. {total} total chunks stored in ChromaDB.")


if __name__ == "__main__":
    main()
