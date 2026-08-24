// The console. One question at a time: scope it if it's ambiguous, stream the
// run, and keep the evidence next to the answer.

import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'
import { useRun } from './useRun'
import Clarify from './components/Clarify'
import Composer from './components/Composer'
import McpPanel from './components/McpPanel'
import Plan from './components/Plan'
import Report from './components/Report'
import Sessions from './components/Sessions'
import Sources from './components/Sources'
import Trace from './components/Trace'

const newSessionId = () =>
  (crypto.randomUUID?.() || `s-${Math.random().toString(36).slice(2)}`)

export default function App() {
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState('research')
  const [profile, setProfile] = useState('')
  const [models, setModels] = useState([])
  const [health, setHealth] = useState(null)
  const [sessions, setSessions] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [status, setStatus] = useState(null)
  const [compacting, setCompacting] = useState(false)
  const [mcp, setMcp] = useState({ configured: [], servers: [] })
  const [probing, setProbing] = useState(false)
  const [scoping, setScoping] = useState(false)
  const [clarifyState, setClarifyState] = useState(null)
  const [tab, setTab] = useState('report')
  const [error, setError] = useState(null)

  const { run, running, start, stop, mcpTools } = useRun()
  const pickedTab = useRef(false)
  const mainRef = useRef(null)

  // --- bootstrap ------------------------------------------------------------

  useEffect(() => {
    api.getModels()
      .then((data) => {
        setModels(data.models || [])
        setProfile((current) => current || data.default || '')
      })
      .catch((err) => setError(String(err.message || err)))
    api.getSessions().then((d) => setSessions(d.sessions || [])).catch(() => {})
    // Config only on load — probing spawns child processes, so that's on request.
    api.getMcp(false).then(setMcp).catch(() => {})
  }, [])

  useEffect(() => {
    if (!profile) return
    let live = true
    api.getHealth(profile).then((h) => live && setHealth(h)).catch(() => live && setHealth(null))
    return () => { live = false }
  }, [profile])

  const refreshStatus = useCallback(() => {
    if (!sessionId) return setStatus(null)
    api.getStatus(sessionId, profile).then(setStatus).catch(() => {})
  }, [sessionId, profile])

  useEffect(refreshStatus, [refreshStatus])

  // --- running --------------------------------------------------------------

  const launch = useCallback(
    async (clarifications) => {
      setError(null)
      setClarifyState(null)
      pickedTab.current = false
      setTab('trace')
      await start({
        question,
        mode,
        profile: profile || undefined,
        session_id: sessionId || undefined,
        clarifications,
      })
      api.getSessions().then((d) => setSessions(d.sessions || [])).catch(() => {})
      refreshStatus()
    },
    [question, mode, profile, sessionId, start, refreshStatus],
  )

  const onRun = useCallback(async () => {
    setError(null)
    setScoping(true)
    try {
      // The scoping pass is fail-open by design, so a scoping failure must not
      // block the research — fall through and run.
      const result = await api.clarify({ question, profile: profile || undefined })
      if (!result.sufficient && result.questions?.length) {
        setClarifyState(result)
        return
      }
    } catch {
      /* proceed unscoped */
    } finally {
      setScoping(false)
    }
    launch()
  }, [question, profile, launch])

  // Follow the report as it streams, unless the reader has scrolled away.
  useEffect(() => {
    const el = mainRef.current
    if (!el || !running || tab !== 'report') return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 220
    if (nearBottom) el.scrollTop = el.scrollHeight
  }, [run.report, running, tab])

  // Move to the report once it starts arriving — unless the reader chose a tab.
  useEffect(() => {
    if (run.report && !pickedTab.current && tab !== 'report') setTab('report')
  }, [run.report, tab])

  const chooseTab = (next) => {
    pickedTab.current = true
    setTab(next)
  }

  const probeMcp = async () => {
    setProbing(true)
    try {
      setMcp(await api.getMcp(true))
    } catch (err) {
      setError(String(err.message || err))
    } finally {
      setProbing(false)
    }
  }

  const onCompact = async () => {
    setCompacting(true)
    try {
      await api.compact({ session_id: sessionId, profile: profile || undefined })
      refreshStatus()
    } finally {
      setCompacting(false)
    }
  }

  const selectSession = async (id) => {
    setSessionId(id)
    setTab('report')
  }

  const backboneReady = health?.backbone?.ready
  const readerReady = health?.reader?.ready

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          FinanceHarness<small>deep research</small>
        </div>

        <span className="pill">
          <span className={`dot ${backboneReady ? 'ok' : health ? 'bad' : ''}`} />
          {health ? `${health.backbone.profile} · ${health.backbone.model}` : 'checking…'}
        </span>
        {health && (
          <span className="pill">
            <span className={`dot ${readerReady ? 'ok' : 'warn'}`} />
            reader {health.reader.profile}
          </span>
        )}
        {mcpTools > 0 && (
          <span className="pill">
            <span className="dot ok" />{mcpTools} MCP call{mcpTools === 1 ? '' : 's'}
          </span>
        )}

        <span className="spacer" />

        {running && (
          <span className="pill">
            <span className="dot live" />
            round {run.round}{run.phase ? ` · ${run.phase}` : ''}
          </span>
        )}
        {sessionId && <span className="pill">session {sessionId.slice(0, 8)}</span>}
      </header>

      <div className="body">
        <aside className="sidebar">
          <Sessions
            sessions={sessions}
            sessionId={sessionId}
            onSelect={selectSession}
            onNew={() => setSessionId(null)}
            onCompact={onCompact}
            status={status}
            compacting={compacting}
          />
          <div className="panel">
            <h3>Thread</h3>
            <button
              className="tiny ghost"
              onClick={() => setSessionId((id) => id || newSessionId())}
              disabled={!!sessionId}
            >
              {sessionId ? 'threading this session' : 'start a session'}
            </button>
          </div>
          <McpPanel
            configured={mcp.configured}
            servers={mcp.servers}
            liveServers={run.mcp}
            onProbe={probeMcp}
            probing={probing}
          />
        </aside>

        <main className="center">
          <Composer
            question={question}
            setQuestion={setQuestion}
            mode={mode}
            setMode={setMode}
            profile={profile}
            setProfile={setProfile}
            models={models}
            scoping={scoping}
            running={running}
            onRun={onRun}
            onStop={stop}
          />

          <div className="main" ref={mainRef}>
            {(error || run.error) && <div className="banner">{error || run.error}</div>}

            {health && !backboneReady && (
              <div className="banner">
                The {health.backbone.profile} backbone isn’t answering — set its API key (or pick
                another backbone) before running.
              </div>
            )}

            <div className="tabs">
              <button className={tab === 'report' ? 'on' : ''} onClick={() => chooseTab('report')}>
                Report
              </button>
              <button className={tab === 'trace' ? 'on' : ''} onClick={() => chooseTab('trace')}>
                Trace
                {!!run.tools.length && <span className="badge">{run.tools.length}</span>}
              </button>
            </div>

            {tab === 'report' ? (
              <Report
                report={run.report}
                running={running}
                reasoning={run.reasoning}
                trajectory={run.trajectory}
                question={run.question}
              />
            ) : (
              <Trace
                tools={run.tools}
                timeline={run.timeline}
                running={running}
                phase={run.phase}
              />
            )}
          </div>
        </main>

        <aside className="rail">
          <Plan plan={run.plan} />
          <Sources sources={run.sources} />
        </aside>
      </div>

      {clarifyState && (
        <Clarify
          result={clarifyState}
          question={question}
          onSubmit={(clarifications) => launch(clarifications)}
          onSkip={() => launch()}
        />
      )}
    </div>
  )
}
