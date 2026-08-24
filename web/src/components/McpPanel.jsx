// MCP, both directions.
//
// Inbound: the external servers this harness borrows tools from — a local
// process holding private data, or a remote endpoint. Probing dials each one and
// reports what it actually exposes.
// Outbound: the snippet that points an MCP client (Claude Desktop, an IDE) at
// this harness, so an LLM elsewhere gets the finance toolkit.

import { useState } from 'react'

const TOOL_PREVIEW = 8 // a real provider exposes dozens; name the first few

const CLIENT_SNIPPET = (cwd) =>
  JSON.stringify(
    {
      mcpServers: {
        financeharness: {
          command: 'uv',
          args: ['run', '--directory', cwd, 'fh', 'mcp'],
          env: { GEMINI_API_KEY: '…' },
        },
      },
    },
    null,
    2,
  )

function Server({ server }) {
  const state = server.connected ? 'ok' : server.error ? 'bad' : 'warn'
  return (
    <div className="mcp-server">
      <div className="head">
        <span className={`dot ${state}`} />
        <span className="name">{server.name}</span>
        <span className="tag">{server.transport}</span>
        <span style={{ flex: 1 }} />
        {server.connected && (
          <span className="tool">
            {server.tools?.length || 0} tools
            {server.resource_count ? ` · ${server.resource_count} res` : ''}
          </span>
        )}
      </div>
      <div className="target">{server.target}</div>
      {server.catalog_mode === 'index' && (
        <div className="sub" title="Too many tools to list in the prompt: the agent searches them, then loads what it needs.">
          indexed — the agent discovers these on demand
        </div>
      )}
      {server.error && <div className="sub" style={{ color: 'var(--bad)' }}>{server.error}</div>}
      {!!server.tools?.length && (
        <div className="tool" style={{ marginTop: 5 }}>
          {server.tools.slice(0, TOOL_PREVIEW).join(', ')}
          {server.tools.length > TOOL_PREVIEW && ` … +${server.tools.length - TOOL_PREVIEW}`}
        </div>
      )}
    </div>
  )
}

export default function McpPanel({ configured, servers, onProbe, probing, liveServers }) {
  const [showSnippet, setShowSnippet] = useState(false)
  // A live run's own view wins: it reports what that run could actually reach.
  const shown = liveServers?.length ? liveServers : servers
  const known = shown?.length ? shown : null

  return (
    <div className="panel">
      <h3>
        MCP data sources
        <span style={{ flex: 1 }} />
        <button className="tiny ghost" onClick={onProbe} disabled={probing}>
          {probing ? '…' : 'probe'}
        </button>
      </h3>

      {known
        ? known.map((server) => <Server key={server.name} server={server} />)
        : configured?.length
          ? configured.map((c) => (
              <div className="mcp-server" key={c.name}>
                <div className="head">
                  <span className={`dot ${c.enabled ? '' : 'warn'}`} />
                  <span className="name">{c.name}</span>
                  <span className="tag">{c.transport}</span>
                  <span style={{ flex: 1 }} />
                  {!c.enabled && <span className="tool">disabled</span>}
                </div>
                <div className="target">{c.target}</div>
              </div>
            ))
          : (
            <p className="empty">
              None configured. Add a local process or an endpoint to configs/mcp.json and its
              tools join every run.
            </p>
          )}

      <h3 style={{ marginTop: 14 }}>Serve this harness</h3>
      <p className="empty" style={{ fontStyle: 'normal' }}>
        <code>fh mcp</code> exposes these tools, the skills, and <code>deep_research</code> to any
        MCP client.
      </p>
      <button className="tiny ghost" onClick={() => setShowSnippet((v) => !v)}>
        {showSnippet ? 'hide' : 'client config'}
      </button>
      {showSnippet && (
        <pre className="out" style={{ marginTop: 6, fontSize: 11 }}>
          {CLIENT_SNIPPET('/path/to/FinanceHarness')}
        </pre>
      )}
    </div>
  )
}
