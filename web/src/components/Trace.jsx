// The agent's timeline: every tool call, its arguments, and the grounding data
// it came back with. This is the audit trail behind the report's numbers.

function argSummary(args) {
  if (!args || !Object.keys(args).length) return ''
  return Object.entries(args)
    .map(([key, value]) => {
      let shown
      if (Array.isArray(value)) shown = `[${value.length} items]`
      else if (value && typeof value === 'object') shown = '{…}'
      else shown = String(value)
      if (shown.length > 88) shown = `${shown.slice(0, 88)}…`
      return `${key}=${shown}`
    })
    .join('  ')
}

export default function Trace({ tools, timeline, running, phase }) {
  if (!tools.length && !timeline.length) {
    return (
      <p className="empty">
        No tool activity yet. The trace fills in as the agent searches, reads, pulls data and computes.
      </p>
    )
  }

  let lastRound = null
  return (
    <div className="trace">
      {tools.map((tool) => {
        const showRound = tool.round !== lastRound
        lastRound = tool.round
        const isMcp = tool.name?.startsWith('mcp_')
        return (
          <div key={tool.call_id}>
            {showRound && <div className="round-sep">── round {tool.round}</div>}
            <div className="step">
              <div className="gutter">
                {tool.pending ? '…' : tool.ok ? `${tool.elapsed_s ?? 0}s` : 'FAIL'}
              </div>
              <div style={{ minWidth: 0, flex: 1 }}>
                <div>
                  <span className={isMcp ? 'name mcp' : 'name'}>{tool.name}</span>
                  {isMcp && <span className="tag" style={{ marginLeft: 6 }}>mcp</span>}
                  {tool.ok === false && <span className="fail"> — failed</span>}
                </div>
                {!!argSummary(tool.args) && <div className="args">{argSummary(tool.args)}</div>}
                {tool.pending && tool.progress && <div className="args">{tool.progress}</div>}
                {tool.result && <pre className="out">{tool.result}</pre>}
              </div>
            </div>
          </div>
        )
      })}
      {running && (
        <div className="step">
          <div className="gutter"><span className="dot live" /></div>
          <div className="args">{phase ? `${phase}…` : 'thinking…'}</div>
        </div>
      )}
    </div>
  )
}
