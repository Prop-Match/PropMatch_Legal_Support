# PropMatch Legal Support API

Standalone FastAPI retrieval-augmented generation (RAG) service for the
PropMatch legal chatbot. It answers Arabic questions about Egyptian rental and
real-estate law using only the supplied law corpus, retrieves evidence from a
Dockerized ChromaDB vector store, and calls the same ITI LLM provider used by
the NestJS form optimizer.

## What was built

- NestJS-compatible `POST /legal-chat/stream` Server-Sent Events endpoint.
- Buffered `POST /legal-chat` endpoint for non-streaming clients/debugging.
- Internal service-key authentication with NestJS-provided user context.
- Strict 1–2000 character request validation and JSON errors before streaming.
- Arabic legal-domain relevance guard with a graceful off-topic decline.
- Article-aware, overlapping chunking of all nine files in
  `laws/egypt_real_estate_laws_txt_for_rag/`.
- Deterministic chunk IDs and idempotent Chroma `upsert` ingestion.
- Metadata retention: law title, article/section, filename, source URL,
  extraction method, and sequence.
- LangChain-based RAG using ITI embeddings by default or Cohere Embed v2 as a
  hosted fallback, with LangChain's Chroma integration. No local model is used.
- Grounded Arabic prompt that forbids invented legal claims/article numbers.
- Legal-information disclaimer on every substantive answer.
- Defensive parsing of every LLM response shape already handled by the NestJS
  optimizer (`output_text`, `reply`, `content`, `choices`, and `message`).
- Liveness/readiness endpoints, Dockerfile, Docker Compose, automated tests,
  `.env.example`, `PLAN.md`, and `AGENT_CONTEXT.md`.
- Authenticated NestJS gateway endpoints that proxy buffered answers and pipe
  SSE frames without buffering.

Chat content is not persisted. Identity documents and eKYC data are never sent
to the LLM or vector database.

## Request flow

```text
LegalChatbot.tsx
  -> Next.js /api/backend/legal-chat/stream (attaches httpOnly JWT)
  -> NestJS /api/legal-chat/stream (validates JWT)
  -> FastAPI /legal-chat/stream (validates internal service key)
  -> selected ITI or Cohere embedding API (`search_query`)
  -> LangChain Chroma top-k Egyptian-law passages
  -> LangChain legal prompt -> ITI LLM with retrieved context only
  -> SSE token frames + terminal done frame
```

The ITI endpoint currently returns a complete answer rather than native token
events. FastAPI splits that answer at word boundaries to preserve the existing
frontend's progressive SSE experience, matching the pattern used by the NestJS
`FormOptimizerService`.

## Prerequisites

- Docker Engine with Docker Compose v2, or Python 3.11/3.12 for local running.
- An ITI student LLM API key (`SBG_API_KEY`).
- A long random internal service key shared only with `propmatch_backend`.

## Run with Docker Compose

From this directory:

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```dotenv
SBG_API_KEY=your-real-key
INTERNAL_SERVICE_API_KEY=the-same-random-key-used-by-propmatch-backend
```

Start FastAPI and the official `chromadb/chroma` image:

```bash
docker compose up --build -d
```

Ingest the supplied laws once (and again whenever the corpus, embedding model,
or chunk settings change):

```bash
docker compose run --rm api python -m app.ingest
```

Check readiness and open the API docs:

```bash
curl http://localhost:8001/health/ready
```

- Swagger UI: <http://localhost:8001/docs>
- ChromaDB: <http://localhost:8000>

The expected readiness result after ingestion contains `"status":"ok"` and a
positive `collection_count`.

## Run locally with only Chroma in Docker

```bash
cp .env.example .env
```

The checked-in `.env.example` already uses `CHROMA_HOST=localhost`, which is
correct when Python runs on the host and only Chroma runs in Docker. Then run:

```bash
docker compose up -d chroma
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m app.ingest
uvicorn app.main:app --reload --port 8001
```

Do not set `CHROMA_HOST=chroma` for this local-Python workflow: Docker service
names resolve only between containers on the Compose network. For the full
Docker workflow, Compose overrides the value to `chroma` automatically.

## Connect NestJS and the frontend

In `../propmatch_backend/.env.development`:

```dotenv
LEGAL_SUPPORT_API_URL=http://localhost:8001
LEGAL_SUPPORT_INTERNAL_API_KEY=the-same-random-key-used-by-fastapi
LEGAL_SUPPORT_TIMEOUT_MS=120000
```

In `../propmatch_frontend/.env.local`, configure only NestJS:

```dotenv
NESTJS_API_URL=http://localhost:3001/api
API_MOCKING=disabled
```

Restart all services after changing environment variables. The frontend never
targets FastAPI: its BFF sends the user JWT to NestJS, NestJS authenticates the
user, and NestJS calls FastAPI using the internal key.

For isolated development without NestJS authentication, set
`AUTH_REQUIRED=false` in the legal service `.env`. Do not use that setting in a
shared or production environment.

## API contract

### Streamed answer

```bash
curl -N http://localhost:3001/api/legal-chat/stream \
  -H 'Authorization: Bearer YOUR_NESTJS_ACCESS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{"message":"ما هي مدة الإخطار قبل إنهاء عقد الإيجار؟"}'
```

Response frames:

```text
data: {"type":"token","value":"وفقًا "}

data: {"type":"token","value":"للقانون..."}

data: {"type":"done","id":"msg_...","declined":false}

```

For an unrelated question, the service does not call the LLM and responds with
the scoped Arabic decline. The final frame contains `"declined":true`.

### Buffered answer

`POST /legal-chat` accepts the same body and returns:

```json
{
  "id": "msg_...",
  "content": "الإجابة...",
  "declined": false,
  "sources": [
    {
      "title": "القانون المدني المصري رقم 131 لسنة 1948",
      "article": "المادة 558",
      "file": "01_civil_law_131_1948.txt",
      "source_url": "https://..."
    }
  ]
}
```

## Configuration

All supported settings are documented in `.env.example`. The most important
ones are:

| Variable | Purpose |
|---|---|
| `SBG_API_KEY` | Bearer credential for the ITI LLM API |
| `INTERNAL_SERVICE_API_KEY` | Shared NestJS-to-FastAPI credential |
| `AUTH_REQUIRED` | Require internal authentication; keep `true` outside isolated development |
| `JWT_SECRET` | Optional direct-JWT fallback, unused when the internal key is configured |
| `CHROMA_HOST`, `CHROMA_PORT` | Chroma HTTP server address |
| `CHROMA_COLLECTION` | Vector collection name |
| `EMBEDDING_PROVIDER` | `iti` (default) or the hosted `cohere` fallback |
| `EMBEDDING_API_URL` | ITI `/api/v1/student/embed` endpoint |
| `EMBEDDING_API_KEY` | Optional separate key; blank reuses `SBG_API_KEY` |
| `EMBEDDING_MODEL_ID` | ITI embedding model ID |
| `COHERE_API_KEY` | Cohere trial or production API key |
| `COHERE_MODEL_ID` | Cohere embedding model; defaults to `embed-v4.0` |
| `COHERE_OUTPUT_DIMENSION` | Cohere vector size: `256`, `512`, `1024`, or `1536` |
| `COHERE_MAX_RETRIES` | Maximum retries when Cohere returns HTTP 429 |
| `COHERE_RETRY_WAIT_SECONDS` | Wait used when a 429 response omits `Retry-After` |
| `EMBEDDING_BATCH_SIZE` | Maximum texts sent per embedding API request |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Corpus chunking controls |
| `RETRIEVAL_TOP_K` | Passages supplied to the LLM |
| `RELEVANCE_MAX_DISTANCE` | Cosine-distance ceiling used by the fallback topic guard |

If the embedding provider, model, or output dimension changes, select a new
`CHROMA_COLLECTION` and re-ingest. Never query a collection with a different
provider: embedding spaces are incompatible even when their dimensions match.
Ingestion sends `input_type=search_document`; live questions send
`input_type=search_query`.

The implementation deliberately contains no PyTorch, sentence-transformers, or
local embedding path. Provider selection is explicit rather than automatic
per-request failover, preventing mixed vectors from corrupting retrieval.

### Embedding-provider troubleshooting

Before ingesting the full corpus, verify one query embedding:

```bash
python -c 'from app.embeddings import get_embeddings; print(len(get_embeddings().embed_query("عقد إيجار")))'
```

The default is the approved catalog identifier
`amazon.titan-embed-text-v2:0:8k`. An API response such as `MODEL_NOT_ALLOWED`,
`REGION_NOT_ALLOWED`, or Bedrock `Model not found` is an ITI account/gateway
configuration problem, not a Chroma connection problem. Ask the ITI API
administrator to enable a text embedding model in an approved region, then set
its exact identifier in `EMBEDDING_MODEL_ID` and rerun ingestion.

To use Cohere temporarily, set `EMBEDDING_PROVIDER=cohere`, add
`COHERE_API_KEY`, choose a fresh collection name such as
`egypt_real_estate_laws_cohere_v4_1024`, and rerun ingestion. Cohere trial keys
are appropriate only for development and evaluation, not production use.
Initial trial-key ingestion can pause for a minute when Cohere's token window
is exhausted; the adapter retries the same batch and deterministic IDs make
rerunning ingestion safe.

For local Python plus Dockerized Chroma, `CHROMA_HOST` must be `localhost`.
Use `chroma` only when the FastAPI process also runs inside Docker Compose.

## Tests and quality checks

```bash
source .venv/bin/activate
pytest
ruff check app tests
python -m compileall -q app tests
```

Tests mock Chroma, embeddings, authentication, and the external LLM. They cover
chunk metadata/IDs, provider response parsing, off-topic behavior, disclaimer
enforcement, request validation, and the exact frontend SSE shape.

## Production notes

- Pin `chromadb/chroma` to an organization-approved version after deployment
  validation; `latest` is used here so the requested official image is easy to
  start during development.
- Put Chroma on a private network and do not expose port 8000 publicly.
- Terminate TLS at the gateway and restrict CORS to the deployed frontend.
- Keep `.env` out of version control and rotate API/internal service secrets normally.
- The law bundle warns that OCR-derived documents may contain errors. Answers
  are informational and should be checked against official publications and a
  qualified lawyer for consequential decisions.
- Monitor 502 (LLM provider) and 503 (retrieval/Chroma) responses separately.
- Re-run ingestion after corpus updates. `upsert` makes repeated ingestion safe,
  but removed source chunks are not automatically deleted; use a new collection
  for controlled corpus releases.

## Key files

- `app/main.py` — FastAPI routes, errors, health, SSE serialization.
- `app/rag.py` — relevance guard, prompt context, sources, disclaimer.
- `app/vector_store.py` — LangChain Chroma collection, ingestion, retrieval.
- `app/embeddings.py` — selectable ITI and Cohere LangChain adapters.
- `app/chunking.py` — corpus parser and deterministic chunks.
- `app/llm.py` — ITI LLM API client.
- `app/auth.py` — internal service-key verification and optional JWT fallback.
- `app/ingest.py` — ingestion CLI.
- `PLAN.md` / `AGENT_CONTEXT.md` — execution plan and future-agent handoff.
