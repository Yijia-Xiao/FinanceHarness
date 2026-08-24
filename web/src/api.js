// The service seam: one place that knows the HTTP + SSE protocol.
//
// The research endpoint streams Server-Sent Events over a POST, which EventSource
// cannot do, so the stream is read off fetch's body and parsed here. Frame types
// are documented in financeharness/service/events.py.

// In dev the Vite proxy exposes the Python service under /api (and strips the
// prefix). A production build is served *by* that service, so it talks to the
// same origin with no prefix. VITE_API_BASE overrides both — for a build that
// sits behind someone else's proxy.
const BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? '/api' : '')

async function json(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const getHealth = (profile) =>
  json(`/health${profile ? `?profile=${encodeURIComponent(profile)}` : ''}`)
export const getModels = () => json('/models')
export const getSessions = () => json('/sessions')
export const getSession = (id) => json(`/sessions/${encodeURIComponent(id)}`)
export const getMcp = (probe) => json(`/mcp?probe=${probe ? 'true' : 'false'}`)

export const getStatus = (sessionId, profile) => {
  const q = new URLSearchParams()
  if (sessionId) q.set('session_id', sessionId)
  if (profile) q.set('profile', profile)
  return json(`/status?${q.toString()}`)
}

export const clarify = (body) =>
  json('/clarify', { method: 'POST', body: JSON.stringify({ ...body, stream: false }) })

export const compact = (body) =>
  json('/compact', { method: 'POST', body: JSON.stringify(body) })

/**
 * Parse a raw SSE byte stream into {event, data} frames.
 *
 * Yields nothing for keepalive comments (`: ping`) and tolerates a frame split
 * across chunk boundaries — the buffer only releases complete frames.
 */
export async function* readEventStream(response, signal) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let split
      while ((split = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, split)
        buffer = buffer.slice(split + 2)
        const frame = parseFrame(raw)
        if (frame) yield frame
      }
      if (signal?.aborted) return
    }
  } finally {
    reader.cancel().catch(() => {})
  }
}

function parseFrame(raw) {
  let event = 'message'
  const dataLines = []
  for (const line of raw.split('\n')) {
    if (!line || line.startsWith(':')) continue // keepalive / comment
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (!dataLines.length) return null
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    return { event, data: { raw: dataLines.join('\n') } }
  }
}

/** Start a streaming research run; returns an async iterator of frames. */
export async function streamResearch(body, signal) {
  const res = await fetch(`${BASE}/research`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  })
  if (!res.ok || !res.body) {
    const detail = await res.text().catch(() => '')
    throw new Error(detail || `research failed: ${res.status}`)
  }
  return readEventStream(res, signal)
}
