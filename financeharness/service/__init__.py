"""HTTP+SSE service: exposes the harness over a versioned API.

`POST /research` (sync JSON, or SSE when stream=true) is the core; alongside it:
`POST /clarify` (scoping), `POST /compact` (session summarization), `GET
/sessions`
+ `/sessions/{id}` (resume), `GET /status`, `GET /models`, `GET /health`. The
agent
runs async per request with its own cache/registry (per-request isolation); the
same trajectory shape backs both the sync body and the streamed `done` event.

Run: ``uvicorn financeharness.service.app:app`` (the FastAPI object lives in the
``app`` submodule; it is not re-exported here, to keep the submodule importable).
"""
