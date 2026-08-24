# FinanceHarness console

A React console over the [HTTP+SSE service](../financeharness/service/app.py).
One question at a time, streamed: the plan the agent maintains, every tool call
with the data that came back, the sources filling as pages are read, and the
report as it is written.

User-facing setup lives in the [main README](../README.md#web-console-react).
This file is for working on the console itself.

## Run it

```bash
npm install
npm run dev          # :5173, proxying /api to the service on :8080
```

The service must be running alongside (`make serve` from the repo root).
`FH_API_URL` retargets the proxy:

```bash
FH_API_URL=http://127.0.0.1:8000 npm run dev
```

`npm run build` emits `dist/`, which `fh serve` picks up and serves at `/` — one
process for API and UI. A build talks to its own origin with no `/api` prefix;
`VITE_API_BASE` overrides that for a build behind someone else's proxy.

## How it is put together

```
src/
  api.js          the service seam — fetch wrappers + the SSE frame parser
  useRun.js       the run reducer: SSE frames in, the view's whole world out
  App.jsx         layout and the one-question-at-a-time state machine
  styles.css      custom properties: one accent, three status colours, greyscale
  components/
    Composer.jsx  question, mode, backbone, run/stop
    Trace.jsx     rounds → tool calls → arguments → grounding data
    Report.jsx    streamed GFM markdown, copy / markdown / trajectory export
    Sources.jsx   the bibliography, numbered to match the report's [N] markers
    Plan.jsx      the agent's own checklist, from update_plan
    Clarify.jsx   the scoping dialog, when a question is genuinely ambiguous
    Sessions.jsx  multi-turn sessions, context meter, compaction
    McpPanel.jsx  which MCP data sources this run can reach, and how to serve it
```

No state library and no router: the interesting state is one run, and
[`useRun.js`](src/useRun.js) owns it.

### The streaming contract

`POST /research` with `stream: true` returns Server-Sent Events. `EventSource`
cannot issue a POST, so [`api.js`](src/api.js) reads the body itself, buffers
until it has a complete `\n\n`-terminated frame, and yields `{event, data}`.
Keepalive comments (`: ping`) are skipped — without them a long tool call would
look like a dropped connection.

[`useRun.js`](src/useRun.js) folds those frames into state, which is where the
protocol's subtleties live:

- `token` frames **accumulate** into the report; the later `answer` frame is
  authoritative and **replaces** it. Reconciling matters — the grounding pass
  can rewrite the draft after it has already streamed.
- `tool_call` and `tool_result` pair by `call_id`, not by order. Several tools
  can be in flight in one round.
- `done` carries the full trajectory — exactly what a non-streaming client would
  have received — so the end state never depends on having seen every frame.
- A stream that ends *without* `done` still settles the UI, rather than leaving
  a spinner running forever.

The frame types are the single source of truth in
[`service/events.py`](../financeharness/service/events.py); `EVENT_TYPES` there
rejects any frame not on the list, so the two ends cannot drift silently.

### Adding a panel

Most additions are a new frame type plus a place to put it:

1. Add the frame to `EVENT_TYPES` in `service/events.py` and emit it.
2. Handle it in the `reducer` in `useRun.js` — keep components free of protocol
   knowledge.
3. Render it. The right rail (`Plan`, `Sources`) is for run context; the sidebar
   is for things that outlive a run.

## Conventions

- Plain JSX, no TypeScript, matching the repo's dependency-light bias.
- Comments explain *why* — the protocol subtleties above are worth a sentence;
  `useState` is not.
- Colour lives in `styles.css` custom properties: one accent, three status
  colours (ok / warn / bad), greyscale for everything else. The palette is
  deliberately narrow so the figures and citations draw the eye.
