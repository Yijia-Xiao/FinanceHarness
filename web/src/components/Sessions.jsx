// Multi-turn sessions. Threading a run into a session lets the next question
// build on the last; compacting summarises the older turns to free context.

export default function Sessions({
  sessions,
  sessionId,
  onSelect,
  onNew,
  onCompact,
  status,
  compacting,
}) {
  return (
    <div className="panel">
      <h3>
        Sessions
        <span style={{ flex: 1 }} />
        <button className="tiny ghost" onClick={onNew}>new</button>
      </h3>

      <div
        className={`card clickable ${!sessionId ? 'active' : ''}`}
        onClick={onNew}
        role="button"
      >
        <div className="title">One-shot</div>
        <div className="sub">no history threaded</div>
      </div>

      {sessions.map((session) => (
        <div
          key={session.id}
          className={`card clickable ${session.id === sessionId ? 'active' : ''}`}
          onClick={() => onSelect(session.id)}
          role="button"
        >
          <div className="title">{session.title || session.id.slice(0, 8)}</div>
          <div className="sub">{session.turns} turn{session.turns === 1 ? '' : 's'}</div>
        </div>
      ))}

      {sessionId && status && (
        <div style={{ marginTop: 10 }}>
          <div className="sub" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="meter">
              <i style={{ width: `${Math.min(100, status.used_pct || 0)}%` }} />
            </span>
            {status.used_pct}% of context
          </div>
          <button
            className="tiny ghost"
            style={{ marginTop: 6 }}
            onClick={onCompact}
            disabled={compacting}
          >
            {compacting ? 'compacting…' : 'compact history'}
          </button>
        </div>
      )}
    </div>
  )
}
