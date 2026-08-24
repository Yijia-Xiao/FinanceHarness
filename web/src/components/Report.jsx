// The deliverable. Markdown (GFM — reports lean on tables), streamed live, with
// the citation markers left intact so a claim stays traceable to its source.

import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function Report({ report, running, reasoning, trajectory, question }) {
  if (!report && !running) {
    return (
      <p className="empty">
        The cited report lands here. Ask something above to start a run.
      </p>
    )
  }

  const download = () => {
    const blob = new Blob([report], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${(question || 'report').slice(0, 48).replace(/[^\w-]+/g, '_')}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  const saveTrajectory = () => {
    const blob = new Blob([JSON.stringify(trajectory, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'trajectory.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <div className="report-head">
        {running && <span className="pill"><span className="dot live" />writing</span>}
        {trajectory && (
          <span className="pill">
            {trajectory.rounds} rounds · {trajectory.elapsed_s}s · {trajectory.termination}
          </span>
        )}
        <span className="spacer" />
        {!!report && (
          <>
            <button className="tiny ghost" onClick={() => navigator.clipboard?.writeText(report)}>
              Copy
            </button>
            <button className="tiny ghost" onClick={download}>Markdown</button>
            {trajectory && (
              <button className="tiny ghost" onClick={saveTrajectory}>Trajectory</button>
            )}
          </>
        )}
      </div>

      {!report && running && reasoning && (
        <details open>
          <summary className="empty">thinking…</summary>
          <pre className="out" style={{ whiteSpace: 'pre-wrap' }}>{reasoning.slice(-1400)}</pre>
        </details>
      )}

      <article className="report">
        <Markdown remarkPlugins={[remarkGfm]}>{report}</Markdown>
        {running && <span className="cursor" />}
      </article>
    </div>
  )
}
