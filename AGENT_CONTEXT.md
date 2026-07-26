# Agent Context: PropMatch Unified AI Microservice

Read `PLAN.md` before making changes.

## Repository Findings & Scope

- **Frontend**: `../propmatch_frontend` (Next.js 16, Arabic RTL, Dual-Mode Assistant).
- **Backend Gateway**: `../propmatch_backend` (NestJS BFF).
- **AI Microservice Location**: This repository (`PropMatch_Legal_Support`).
- **Corpus Files**:
  - Law Corpus: `laws/egypt_real_estate_laws_txt_for_rag/` (Egyptian Real Estate Law No. 4 of 1996).
  - Support FAQs: `docs/support_faqs/` (Platform guides, tenant/landlord rules, eKYC, PayMob policies).

---

## API Endpoints & Contracts

### 1. Legal Assistant Endpoint (`POST /legal-chat/stream`)

- **Preserved Route**: `POST /legal-chat/stream` (100% backward compatible).
- **Collection**: `CHROMA_LEGAL_COLLECTION` (`egypt_real_estate_laws_v1`).
- **SSE Frame Format**:

  ```text
  data: {"type":"token","value":"جزء من الإجابة "}

  data: {"type":"done","id":"msg_uuid","declined":false}
  ```

- **Disclaimer**: Appends legal disclaimer on all answers.

### 2. Customer Support Endpoint (`POST /support/stream`)

- **Route**: `POST /support/stream`.
- **Collection**: `CHROMA_SUPPORT_COLLECTION` (`support_kb_v1`).
- **Escalation Frame Format**:

  ```text
  data: {"type":"escalate","shouldEscalate":true,"reason":"طلب المستخدم التحدث مع موظف دعم فني بشكل صريح","priority":"HIGH"}

  data: {"type":"token","value":"تم تحويل طلبك..."}

  data: {"type":"done","id":"msg_uuid","escalated":true}
  ```

---

## Security & BFF Gateway

The Next.js frontend sends requests to NestJS (`/api/backend/legal-chat/stream` or `/api/backend/support/ai-chat/stream`). NestJS validates the user JWT and proxies the stream to this FastAPI service sending:

- `X-Internal-Service-Key` (matching `INTERNAL_SERVICE_API_KEY`)
- `X-PropMatch-User-Id`
- `X-PropMatch-User-Role`

Do not expose direct browser-to-FastAPI routes.

---

## LLM & Vector Store Setup

- **LLM Provider**: ITI Student Chat API (`http://apiaccess.iti.net.eg/api/v1/student/chat`).
- **Model**: `openai.gpt-oss-120b-1:0`.
- **Vector DB**: Dockerized ChromaDB container (`CHROMA_HOST:8000`).
- **Ingestion CLI**: Run `python -m app.ingest --target all`.
