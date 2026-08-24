// The bibliography, filling live as `visit` reads pages. The numbers match the
// [N] markers in the report.

const host = (url) => {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

export default function Sources({ sources }) {
  return (
    <div className="panel">
      <h3>Sources <span className="count">{sources.length || ''}</span></h3>
      {!sources.length && <p className="empty">Pages the agent reads appear here, numbered.</p>}
      {sources.map((source) => (
        <a
          key={`${source.index}-${source.url}`}
          className="card clickable source"
          href={source.url}
          target="_blank"
          rel="noreferrer noopener"
          style={{ display: 'block', textDecoration: 'none', color: 'inherit' }}
        >
          <div className="row">
            <span className="idx">{source.index}</span>
            <span className="title">{source.title || host(source.url)}</span>
          </div>
          <div className="host">{host(source.url)}</div>
          {source.headline && <div className="sub">{source.headline}</div>}
        </a>
      ))}
    </div>
  )
}
