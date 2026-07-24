# Agent Context: PropMatch Legal Support

Read `PLAN.md` before making changes.

## Repository findings

- Frontend: `../propmatch_frontend` (Next.js, Arabic RTL).
- Existing backend: `../propmatch_backend` (NestJS).
- New service location: this `legal support` directory.
- Law corpus: `laws/egypt_real_estate_laws_txt_for_rag/` (nine UTF-8 text
  files plus `manifest.json`). Some sources are OCR and may contain errors.

## Frontend API contract

The component `../propmatch_frontend/src/features/legal/components/LegalChatbot.tsx`
calls `streamPost("legal-chat/stream", { message })`.

The SSE parser expects frames separated by a blank line and JSON after `data:`:

```text
data: {"type":"token","value":"جزء من الإجابة "}

data: {"type":"done","id":"msg_uuid","declined":false}

```

The mock backend currently requires authentication, rejects blank messages,
and uses this off-topic behavior:

```text
أقدر أساعدك فقط في أسئلة الإيجار والقانون العقاري في مصر.
```

The frontend's generic BFF forwards `/api/backend/*` only to NestJS. NestJS owns
`/api/legal-chat` and `/api/legal-chat/stream`, validates the user JWT, then
calls this FastAPI service with `X-Internal-Service-Key` and user-context
headers. Do not add a direct frontend-to-FastAPI route.

## LLM provider contract

Mirror `../propmatch_backend/src/properties/services/FormOptimizer.service.ts`:

- URL: `http://apiaccess.iti.net.eg/api/v1/student/chat` (make configurable).
- Header: `Authorization: Bearer ${SBG_API_KEY}`.
- Body fields: `model_id`, `messages`, `system_prompt`, `max_tokens`.
- Existing model: `openai.gpt-oss-120b-1:0`.
- Provider output may be in `output_text`, `reply`, `content`, `choices`, or
  `message`; parse these defensively.

## Legal/RAG rules

- Retrieve only from the supplied corpus.
- Prefer article-aware chunks; retain law title, filename, article, source URL,
  and extraction method as Chroma metadata.
- Treat retrieved OCR text as potentially imperfect.
- Tell the model to use only supplied context, admit when context is
  insufficient, avoid fabricated article numbers, and answer in Arabic.
- Every substantive response must include a concise disclaimer that it is
  general legal information and not binding legal advice.
- Never persist messages or send identity/KYC data to ChromaDB or the LLM.
- Do not add unrelated property semantic-search behavior; this service is only
  the legal assistant.

## Operational choices

- Python 3.12 target.
- FastAPI + Uvicorn + HTTPX + LangChain + LangChain Chroma.
- ChromaDB is an external container based on the official image.
- Embeddings use an environment-selected custom LangChain adapter. ITI
  `/student/embed` is the default; Cohere Embed v2 is the temporary hosted
  fallback. A provider/model switch requires a separate Chroma collection and
  full re-ingestion. Never mix embedding spaces or add a local PyTorch model.
- External LLM, embedding API calls, and Chroma are mocked in unit tests.
- Secrets belong only in `.env`; commit `.env.example`, never real keys.
