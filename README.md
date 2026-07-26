# PropMatch Unified AI Service API

Standalone FastAPI retrieval-augmented generation (RAG) microservice for the PropMatch platform, hosting both the **Legal Chatbot** and the **Customer Support AI Assistant**.

It answers Arabic legal questions regarding Egyptian real-estate law using `laws/`, answers platform usage questions using `docs/support_faqs/`, evaluates multi-factor escalation rules for human support handoff, retrieves evidence from a Dockerized ChromaDB vector store, and streams answers using the shared ITI LLM provider.

---

## Features & Endpoints

- **Legal Chat Stream**: `POST /legal-chat/stream` (SSE tokens + legal disclaimer enforcement).
- **Legal Chat Buffered**: `POST /legal-chat` (buffered JSON answer with cited law sources).
- **Support Chat Stream**: `POST /support/stream` (under implementation; not production-ready).
- **Health Probes**: `GET /health/live` and `GET /health/ready`.
- **Internal Key Security**: Validates `X-Internal-Service-Key` header sent by NestJS BFF.
- **Dedicated Vector Collections**:
  - `CHROMA_LEGAL_COLLECTION`: `egypt_real_estate_laws_v1`
  - `CHROMA_SUPPORT_COLLECTION`: `support_kb_v1`

---

## Architecture Flow

```text
UnifiedAiAssistant.tsx (Next.js Frontend)
  -> /api/backend/support/ai-chat/stream  OR  /api/backend/legal-chat/stream
  -> NestJS Gateway (Validates JWT Cookie)
  -> FastAPI /support/stream  OR  /legal-chat/stream (Validates Internal Security Key)
  -> ITI Embedding API (`amazon.titan-embed-text-v2:0:8k`)
  -> ChromaDB Top-k Retrieval (`CHROMA_HOST:8000`)
  -> Grounded ITI LLM Generation (`openai.gpt-oss-120b-1:0`)
  -> SSE Token Frames + Terminal Done Frame
```

---

## Endpoint Reference

Interactive OpenAPI documentation is available at `http://localhost:8001/docs`
while the service is running. Chat endpoints are internal APIs: the browser calls
the Next.js BFF, NestJS authenticates the user, and only NestJS calls FastAPI.

### `GET /health/live`

**Purpose:** Docker/Kubernetes liveness probe. It answers whether the FastAPI
process is running and able to receive HTTP requests.

**How it works:** It returns immediately and deliberately does not contact
ChromaDB, the embedding provider, or the LLM provider.

```json
{
  "status": "ok",
  "chroma": "not_checked",
  "collection_count": null
}
```

### `GET /health/ready`

**Purpose:** Readiness probe used before sending chat traffic to the service.

**How it works:** It sends a heartbeat to ChromaDB and counts indexed legal
chunks. A reachable but empty collection is reported as `degraded`; an
unreachable ChromaDB instance is reported as `down`.

```json
{
  "status": "ok",
  "chroma": "up",
  "collection_count": 420
}
```

This endpoint currently checks the legal vector store. It should be extended to
report legal and support collections separately when support RAG is completed.

### `POST /legal-chat`

**Purpose:** Return one completed legal RAG answer as JSON. It is useful for
diagnostics, automated tests, and clients that do not need SSE.

**Authentication headers sent by NestJS:**

```http
X-Internal-Service-Key: <shared-private-key>
X-PropMatch-User-Id: <authenticated-user-id>
X-PropMatch-User-Role: TENANT
Content-Type: application/json
```

**Request:**

```json
{
  "message": "ما هي مدة الإخطار قبل إنهاء عقد الإيجار؟"
}
```

The message is trimmed and must contain between 1 and 2000 characters.

**How it works:**

1. `CurrentUser` validates the internal key and NestJS user headers.
2. `LegalRagService` embeds the question and queries the legal Chroma collection.
3. Clearly unrelated passages are removed and off-topic questions are declined.
4. Retrieved passages and their metadata are inserted into the legal prompt.
5. `ItiLlmClient` calls the configured ITI model.
6. The required legal-information disclaimer and unique sources are returned.

**Response:**

```json
{
  "id": "msg_...",
  "content": "الإجابة القانونية...",
  "declined": false,
  "sources": [
    {
      "title": "القانون المدني المصري رقم 131 لسنة 1948",
      "article": "المادة 563",
      "file": "01_civil_law_131_1948.txt",
      "source_url": "https://example.com/source"
    }
  ]
}
```

### `POST /legal-chat/stream`

**Purpose:** Production endpoint used by the PropMatch legal-chat frontend.

**Request and authentication:** Same as `POST /legal-chat`.

**How it works:** The complete grounded answer is generated first. Because the
current ITI provider returns a complete answer rather than native model tokens,
FastAPI divides the answer into progressive SSE fragments. NestJS pipes those
fragments through the Next.js BFF to `LegalChatbot`.

```text
data: {"type":"token","value":"وفقًا "}

data: {"type":"token","value":"للقانون..."}

data: {"type":"done","id":"msg_...","declined":false}

```

The terminal `done` frame is sent exactly once. `declined=true` means the legal
agent rejected an off-topic question without presenting it as legal advice.

### `POST /support/stream` (Under Implementation)

**Intended purpose:** Answer PropMatch usage questions and recommend a human
handoff when the user explicitly requests one, reports a payment/security
emergency, or repeatedly fails to resolve an issue.

**Intended ownership boundary:** FastAPI may recommend escalation, but it must
not create a ticket. NestJS owns `SupportTicket`, `SupportMessage`, PostgreSQL,
ticket status transitions, and Socket.IO notifications.

The current route and support service are incomplete. In particular, the stable
request model does not yet include support history, the support SSE response is
not yet finalized, and no buffered `POST /support/chat` endpoint exists. Do not
integrate the frontend with this route until its contract and tests are complete.

### Error Responses

Validation and expected HTTP failures use the NestJS-compatible shape:

```json
{
  "statusCode": 400,
  "message": ["validation message"]
}
```

Common statuses are `400` for invalid input, `401` for an invalid internal key,
`502` for LLM-provider failure, and `503` for retrieval or ChromaDB failure.

---

## How The Code Fits Together

| File | Responsibility |
|---|---|
| `app/main.py` | Creates FastAPI, installs shared error handlers, registers routers, and exposes health probes. |
| `app/auth.py` | Verifies the NestJS internal key and converts trusted headers into user context. |
| `app/config.py` | Loads typed `.env` settings without exposing secret values in representations. |
| `app/routers/legal_router.py` | Defines buffered and SSE legal HTTP contracts. |
| `app/rag.py` | Runs the stable legal retrieval, relevance, prompt, disclaimer, and source pipeline. |
| `app/embeddings.py` | Adapts ITI or Cohere embeddings to the LangChain interface. |
| `app/vector_store.py` | Connects to the legal Chroma collection for ingestion and similarity search. |
| `app/llm.py` | Calls the ITI generation API and normalizes supported response shapes. |
| `app/chunking.py` | Splits law files by article and overlapping windows with deterministic IDs. |
| `app/ingest.py` | Loads the law corpus and idempotently upserts chunks into ChromaDB. |
| `app/routers/support_router.py` | Contains the in-progress support SSE endpoint. |
| `app/services/support_rag.py` | Contains the in-progress support retrieval and answer orchestration. |
| `app/services/escalation.py` | Produces advisory handoff decisions for NestJS to validate and execute. |

---

## Environment Variables (`.env`)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment settings:

| Variable                    | Purpose                        | Default / Example                |
| --------------------------- | ------------------------------ | -------------------------------- |
| `APP_NAME`                  | Service Title                  | `"PropMatch Unified AI Service"` |
| `SBG_API_KEY`               | ITI Student API Key            | `your-iti-key`                   |
| `INTERNAL_SERVICE_API_KEY`  | Shared secret key with NestJS  | `long-random-internal-key`       |
| `CHROMA_HOST`               | ChromaDB container host        | `localhost`                      |
| `CHROMA_PORT`               | ChromaDB container port        | `8000`                           |
| `CHROMA_LEGAL_COLLECTION`   | Legal vector collection name   | `egypt_real_estate_laws_v1`      |
| `CHROMA_SUPPORT_COLLECTION` | Support vector collection name | `support_kb_v1`                  |

---

## Data Ingestion into ChromaDB

Ingest law documents (`laws/`) and support FAQs (`docs/support_faqs/`) into ChromaDB:

### Using Docker Compose (Recommended)

```bash
docker compose up -d chroma
docker compose run --rm api python -m app.ingest --target all
```

### Using Local Python

```bash
source .venv/bin/activate
python -m app.ingest --target all
```

Check service readiness:

```bash
curl http://localhost:8001/health/ready
```

---

## Running the Service

### Docker Compose

```bash
docker compose up --build -d
```

### Local Python

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8001
```

---

## Tests & Quality Checks

Run pytest with mocked external APIs:

```bash
source .venv/bin/activate
pytest
```
