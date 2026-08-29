#!/bin/sh
set -e

echo "=== PropMatch Legal Support — Container Startup ==="

# Run corpus ingestion if the vector store is empty (embedded mode).
# In client-server mode (CHROMA_HOST set), ingestion is a manual step.
if [ -z "$CHROMA_HOST" ]; then
  echo "Embedded ChromaDB mode — ingesting corpus..."
  python -m app.ingest --target all
  echo "Ingestion complete."
else
  echo "Client-server ChromaDB mode — skipping auto-ingestion."
fi

echo "Starting Uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8001}"
