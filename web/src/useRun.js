// The run reducer — SSE frames in, the view's whole world out.
//
// One place folds the event protocol into state so components stay dumb:
// tokens accumulate into the report, `answer` supersedes them, tool calls pair
// with their results by call_id, and `done` carries the authoritative trajectory.

import { useCallback, useMemo, useReducer, useRef } from 'react'
import { streamResearch } from './api'

const initial = {
  status: 'idle', // idle | running | done | error
  question: '',
  mode: null,
  round: 0,
  phase: null,
  report: '', // streamed, then reconciled to the authoritative answer
  reasoning: '',
  tools: [], // {call_id, name, args, round, ok, elapsed_s, result, progress}
  sources: [],
  plan: [],
  mcp: [],
  timeline: [], // {id, kind, label, detail}
  trajectory: null,
  error: null,
}

let seq = 0
const mark = (kind, label, detail) => ({ id: ++seq, kind, label, detail, at: Date.now() })

function reducer(state, action) {
  if (action.type === 'reset') {
    return { ...initial, status: 'running', question: action.question }
  }
  if (action.type === 'abort') {
    return { ...state, status: 'done' }
  }
  if (action.type === 'fail') {
    return { ...state, status: 'error', error: action.error }
  }
  if (action.type !== 'frame') return state

  const { event, data } = action.frame
  switch (event) {
    case 'run_start':
      return {
        ...state,
        mode: data.mode,
        timeline: [...state.timeline, mark('run', `${data.mode} run started`)],
      }
    case 'round_start':
      return {
        ...state,
        round: data.round,
        phase: null,
        timeline: [...state.timeline, mark('round', `Round ${data.round}`)],
      }
    case 'mcp':
      return {
        ...state,
        mcp: data.servers || [],
        timeline: [
          ...state.timeline,
          mark(
            'mcp',
            `MCP: ${(data.servers || []).filter((s) => s.connected).length} server(s) connected`,
          ),
        ],
      }
    case 'tool_call':
      return {
        ...state,
        tools: [
          ...state.tools,
          {
            call_id: data.call_id,
            name: data.name,
            args: data.args,
            round: data.round,
            pending: true,
          },
        ],
      }
    case 'tool_progress':
      return {
        ...state,
        tools: state.tools.map((t) =>
          t.call_id === data.call_id ? { ...t, progress: data.detail } : t,
        ),
      }
    case 'tool_result':
      return {
        ...state,
        tools: state.tools.map((t) =>
          t.call_id === data.call_id
            ? {
                ...t,
                pending: false,
                ok: data.ok,
                elapsed_s: data.elapsed_s,
                result: data.result,
              }
            : t,
        ),
      }
    case 'source':
      return state.sources.some((s) => s.url === data.url)
        ? state
        : { ...state, sources: [...state.sources, data] }
    case 'plan':
      return { ...state, plan: data.items || [] }
    case 'reasoning':
      return { ...state, reasoning: state.reasoning + (data.text || '') }
    case 'token':
      return { ...state, report: state.report + (data.text || '') }
    case 'phase':
      return {
        ...state,
        phase: data.label,
        timeline: [...state.timeline, mark('phase', data.label)],
      }
    case 'answer':
      return { ...state, report: data.content || state.report }
    case 'error':
      return {
        ...state,
        error: data.error,
        timeline: [...state.timeline, mark('error', data.error)],
      }
    case 'done': {
      const traj = data.trajectory
      return {
        ...state,
        status: state.error && !traj ? 'error' : 'done',
        trajectory: traj,
        report: traj?.prediction || state.report,
        mcp: traj?.mcp?.length ? traj.mcp : state.mcp,
        phase: null,
      }
    }
    default:
      return state
  }
}

export function useRun() {
  const [state, dispatch] = useReducer(reducer, initial)
  const controller = useRef(null)

  const start = useCallback(async (body) => {
    controller.current?.abort()
    const ac = new AbortController()
    controller.current = ac
    dispatch({ type: 'reset', question: body.question })
    try {
      const frames = await streamResearch(body, ac.signal)
      for await (const frame of frames) {
        dispatch({ type: 'frame', frame })
      }
      // A stream that ends without `done` (a dropped connection) still settles.
      dispatch({ type: 'abort' })
    } catch (err) {
      if (ac.signal.aborted) dispatch({ type: 'abort' })
      else dispatch({ type: 'fail', error: String(err.message || err) })
    }
  }, [])

  const stop = useCallback(() => {
    controller.current?.abort()
    dispatch({ type: 'abort' })
  }, [])

  const derived = useMemo(
    () => ({
      running: state.status === 'running',
      toolCount: state.tools.length,
      failedTools: state.tools.filter((t) => t.ok === false).length,
      mcpTools: state.tools.filter((t) => t.name?.startsWith('mcp_')).length,
    }),
    [state.status, state.tools],
  )

  return { run: state, ...derived, start, stop }
}
