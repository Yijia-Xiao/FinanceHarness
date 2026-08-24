// The ask. Question in, backbone + mode chosen, run or stop.

const EXAMPLES = [
  "Estimate NVDA's intrinsic value with a DCF.",
  'Is AAPL expensive versus its peers right now?',
  "What's the consensus view on TSM, and does the web support it?",
  'Compare the risk profile of MSFT and NVDA over the past year.',
]

const MODES = [
  ['research', 'research — web-first'],
  ['analytical', 'analytical — numbers-first'],
  ['auto', 'auto — agent decides'],
]

export default function Composer({
  question,
  setQuestion,
  mode,
  setMode,
  profile,
  setProfile,
  models,
  scoping,
  running,
  onRun,
  onStop,
}) {
  const submit = (event) => {
    event.preventDefault()
    if (!running && question.trim()) onRun()
  }

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a finance question — the agent plans, gathers evidence, computes, and cites."
        onKeyDown={(e) => {
          // Enter runs; Shift+Enter is a newline. A research question is usually one line.
          if (e.key === 'Enter' && !e.shiftKey) submit(e)
        }}
        disabled={running}
        spellCheck="false"
      />
      <div className="controls">
        <select value={mode} onChange={(e) => setMode(e.target.value)} disabled={running}>
          {MODES.map(([value, label]) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
        <select value={profile} onChange={(e) => setProfile(e.target.value)} disabled={running}>
          {models.map((m) => (
            <option key={m.name} value={m.name} disabled={!m.available}>
              {m.name} — {m.model}{m.available ? '' : ' (no key)'}
            </option>
          ))}
        </select>
        <span className="spacer" style={{ flex: 1 }} />
        {running ? (
          <button type="button" onClick={onStop}>Stop</button>
        ) : (
          <button type="submit" className="primary" disabled={!question.trim() || scoping}>
            {scoping ? 'Scoping…' : 'Research'}
          </button>
        )}
      </div>
      {!running && !question && (
        <div className="examples">
          {EXAMPLES.map((example) => (
            <button key={example} type="button" onClick={() => setQuestion(example)}>
              {example}
            </button>
          ))}
        </div>
      )}
    </form>
  )
}
