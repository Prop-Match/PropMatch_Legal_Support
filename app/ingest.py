import argparse
import sys
from pathlib import Path

from app.chunking import chunk_corpus, chunk_support_faqs
from app.config import get_settings
from app.vector_store import get_vector_store


def ingest_laws() -> int:
    settings = get_settings()
    corpus_dir = settings.laws_dir.resolve()
    chunks = chunk_corpus(corpus_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No law chunks found in {corpus_dir}")
    store = get_vector_store(collection_name=settings.chroma_legal_collection)
    count = store.ingest(chunks)
    print(f"✅ Ingested {count} law chunks into collection '{settings.chroma_legal_collection}'.")
    return count


def ingest_support() -> int:
    settings = get_settings()
    support_dir = Path("docs/support_faqs").resolve()
    if not support_dir.exists():
        print(f"Warning: {support_dir} does not exist.")
        return 0
    chunks = chunk_support_faqs(support_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print(f"Warning: No support FAQ chunks found in {support_dir}")
        return 0
    store = get_vector_store(collection_name=settings.chroma_support_collection)
    count = store.ingest(chunks)
    print(f"✅ Ingested {count} support chunks into collection '{settings.chroma_support_collection}'.")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest law corpus and support FAQs into ChromaDB")
    parser.add_argument(
        "--target",
        choices=["legal", "support", "all"],
        default="all",
        help="Target collection to ingest (default: all)",
    )
    args = parser.parse_args()

    try:
        if args.target in ("legal", "all"):
            ingest_laws()
        if args.target in ("support", "all"):
            ingest_support()
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
