# PropMatch Legal Support Service Plan

## Goal

Build a standalone FastAPI service in this directory that provides the legal
chatbot required by the PropMatch frontend. It will retrieve relevant passages
from the supplied Egyptian real-estate laws in ChromaDB and use the same ITI
LLM API pattern already used by the NestJS `FormOptimizerService`.

## Confirmed contract and constraints

- Main endpoint: `POST /legal-chat/stream`.
- Request JSON: `{ "message": "..." }`, 1–2000 characters.
- Response: Server-Sent Events (`text/event-stream`).
- Streaming chunks:
  - `{"type":"token","value":"..."}`
  - final `{"type":"done","id":"...","declined":false}`
- Off-topic questions receive the Arabic graceful decline and finish with
  `declined: true`.
- Chat history is session-scoped and is not persisted.
- Answers are informational, cite retrieved law/article metadata where
  available, and include a non-binding legal-information disclaimer.
- The only RAG corpus is `laws/egypt_real_estate_laws_txt_for_rag/`.
- ChromaDB runs from its official Docker image; the application connects over
  HTTP and does not embed an in-process vector database.
- LLM calls follow the existing backend pattern: bearer `SBG_API_KEY`,
  configurable ITI chat URL, model ID, messages, system prompt, and max tokens.

## Architecture

1. Settings and application lifecycle validate configuration and initialize
   shared clients.
2. Corpus ingestion parses law text into article-aware, overlapping chunks and
   stores content plus source metadata in ChromaDB.
3. Environment-selected LangChain `Embeddings` adapters call ITI by default or
   Cohere Embed v2 as a hosted fallback. Ingestion uses `search_document`;
   retrieval uses `search_query`. Provider changes require a separate Chroma
   collection and re-ingestion; no local PyTorch model is installed.
4. A relevance guard combines explicit domain cues with vector distance. It
   rejects clearly unrelated questions before an LLM call.
5. LangChain Chroma retrieves top passages and a LangChain chat prompt builds a
   constrained Arabic request for the ITI LLM API; the service streams the
   resulting answer using the frontend's SSE format.
6. A buffered `POST /legal-chat` endpoint mirrors the same behavior for API
   clients and diagnostics.
7. NestJS exposes the public authenticated endpoints and proxies/pipes FastAPI
   using an internal service credential; the frontend only targets NestJS.
8. Health/readiness and ingestion endpoints/scripts support operations.

## Execution steps

1. Scaffold the FastAPI package and typed settings.
2. Implement article-aware chunking, deterministic IDs, manifest metadata,
   Chroma collection management, and idempotent ingestion.
3. Implement embeddings, retrieval, relevance guard, prompt construction, and
   resilient parsing of the ITI LLM response formats.
4. Implement buffered and SSE routes, validation, CORS, error handling, and
   health endpoints.
5. Add Dockerfile and Docker Compose with the official ChromaDB image.
6. Add unit/API tests with external dependencies mocked.
7. Add `.env.example`, `.dockerignore`, `.gitignore`, and a complete README
   covering architecture, setup, ingestion, running, API examples, and NestJS/
   frontend integration.
8. Run tests and static/compile checks, then document any remaining operational
   requirements.

## Definition of done

- The frontend-compatible stream contract is tested.
- Off-topic decline is tested and avoids the LLM.
- Corpus chunking and idempotent ingestion behavior are tested.
- No secret is committed.
- Docker Compose starts FastAPI plus ChromaDB.
- README and `.env.example` are sufficient for a new developer to run the
  service and connect it to PropMatch.
