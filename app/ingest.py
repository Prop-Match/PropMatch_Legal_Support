import argparse
import sys

from app.chunking import chunk_corpus
from app.config import get_settings
from app.vector_store import get_vector_store


def ingest() -> int:
    settings = get_settings()
    corpus_dir = settings.laws_dir.resolve()
    chunks = chunk_corpus(corpus_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError(f"No law chunks found in {corpus_dir}")
    count = get_vector_store().ingest(chunks)
    print(f"Ingested {count} chunks into collection '{settings.chroma_collection}'.")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the supplied law corpus into ChromaDB")
    parser.parse_args()
    try:
        ingest()
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
