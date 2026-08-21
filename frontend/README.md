# WTH — What The Human frontend

Production-oriented Next.js frontend for the locked WTH v1 reading experience.

## Contract source of truth

- `contracts/openapi.json` — copied from the verified FastAPI artifact.
- `types/generated/openapi.ts` — generated from that local OpenAPI artifact with `npm run generate:api`.
- `tests/fixtures/query_success.json` and `tests/fixtures/chunk_success.json` — normalized copies of the captured live responses used only as test fixtures.

Do not edit `types/generated/openapi.ts` by hand. Replace `contracts/openapi.json` when the backend contract changes, then regenerate.

## Environment

Copy `.env.example` to `.env.local`:

```bash
NEXT_PUBLIC_WTH_API_BASE_URL=http://127.0.0.1:8000
```

Optional:

```bash
NEXT_PUBLIC_WTH_GITHUB_URL=https://github.com/<owner>/<repo>
```

No secret belongs in a `NEXT_PUBLIC_` variable. The browser has no direct Supabase, Gemini, Groq, pgvector, or provider connection.

## Commands

```bash
npm install
npm run generate:api
npm run dev
npm run lint
npm run typecheck
npm test
npm run build
```

The development frontend is served by Next.js (normally at `http://localhost:3000`) and calls the FastAPI backend configured by `NEXT_PUBLIC_WTH_API_BASE_URL`.

## Netlify

The project is intended to be deployed from this `frontend/` directory. Configure the environment variable `NEXT_PUBLIC_WTH_API_BASE_URL` to the deployed HTTPS FastAPI origin. The backend must allow the Netlify site origin through CORS because the browser calls the API directly.

The included `netlify.toml` uses `npm run build`. Netlify's Next.js runtime should handle the `.next` output; do not convert this app to a static export because the App Router/runtime integration should remain intact.

## Architecture notes

- Exactly three user-facing routes: `/`, `/try-these`, `/about`.
- `/api/query` returns one complete JSON response. There is no SSE/EventSource/WebSocket implementation.
- The staged Science → Advaita → Samkhya → synthesis reveal is intentionally simulated in `ResponseView.tsx` after the atomic response has arrived.
- Citation chips never guess `chunk_id`. They resolve a response-scoped `citation_ref` through top-level `claim_level_citations`, then request `/api/chunk/{encodedChunkId}`.
- API text is rendered as text; there is no `dangerouslySetInnerHTML` and no mojibake replacement heuristic.
