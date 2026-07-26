# PropMatch Unified AI Service Plan (Legal RAG & Customer Support Assistant)

## Goal

Build a single, unified standalone FastAPI service in this directory (`PropMatch_Legal_Support`) that provides both the **Legal RAG Assistant** and the **Customer Support AI Assistant** for the PropMatch platform.

It retrieves relevant passages from Dockerized ChromaDB vector collections (`CHROMA_LEGAL_COLLECTION` and `CHROMA_SUPPORT_COLLECTION`), evaluates multi-factor escalation rules for human support handoff, and calls the shared ITI LLM API provider (`openai.gpt-oss-120b-1:0`).

---

## Confirmed Contracts & Constraints

### 1. Legal Assistant Domain

- Main Streamed Endpoint: `POST /legal-chat/stream`
- Buffered Endpoint: `POST /legal-chat`
- Request JSON: `{ "message": "..." }`, 1–2000 characters.
- Response: Server-Sent Events (`text/event-stream`).
- Chunks: `{"type":"token","value":"..."}` and final `{"type":"done","id":"...","declined":false}`.
- Off-topic legal questions receive the Arabic graceful decline (*"أقدر أساعدك فقط في أسئلة الإيجار والقانون العقاري في مصر."*).
- Legal Disclaimer: Enforces non-binding legal information disclaimer on all substantive responses.
- Vector Collection: `CHROMA_LEGAL_COLLECTION` (`egypt_real_estate_laws_v1`).

### 2. Customer Support Assistant Domain

- Main Streamed Endpoint: `POST /support/stream`
- Request JSON: `{ "message": "...", "history": [...] }`, 1–2000 characters per message, max 10 history items.
- Escalation Recommendation: Emits `{"type":"escalate","shouldEscalate":true,"reason":"...","priority":"HIGH"}` when escalation rules trigger.
- Multi-Factor Escalation: Evaluates explicit human requests, payment/account emergencies, and 4+ unresolved follow-up attempts.
- Vector Collection: `CHROMA_SUPPORT_COLLECTION` (`support_kb_v1`).

### 3. Shared Infrastructure

- Security: Requires `X-Internal-Service-Key` matching NestJS `INTERNAL_SERVICE_API_KEY`.
- Vector Database: External Dockerized ChromaDB container on `CHROMA_HOST:8000`.
- LLM Provider: ITI Chat API (`http://apiaccess.iti.net.eg/api/v1/student/chat`).

---

## Architecture & Modular Routers

1. **`app/main.py`**: Single FastAPI application registering both domain routers (`legal_router` and `support_router`).
2. **`app/routers/legal_router.py`**: Handles `POST /legal-chat/stream` and `POST /legal-chat`.
3. **`app/routers/support_router.py`**: Handles `POST /support/stream` and `POST /support/chat`.
4. **`app/services/legal_rag.py`**: Queries legal collection and enforces Egyptian law context + disclaimer.
5. **`app/services/support_rag.py`**: Queries platform FAQ collection (`docs/support_faqs/`).
6. **`app/services/escalation.py`**: Evaluates 3 escalation rules for human support handoff.
7. **`app/ingest.py`**: CLI script supporting `--target legal`, `--target support`, or `--target all`.

---

## Definition of Done

- Both `POST /legal-chat/stream` and `POST /support/stream` contracts function cleanly.
- `POST /legal-chat/stream` endpoint name remains 100% backward compatible.
- Support escalation emits `escalate` chunk when triggered.
- Vector collections (`CHROMA_LEGAL_COLLECTION` & `CHROMA_SUPPORT_COLLECTION`) ingest data without errors.
- Unit tests pass for both legal and support pipelines with mocked external APIs.
