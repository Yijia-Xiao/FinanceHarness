// The agent's own checklist, as it maintains it through update_plan.

// Statuses come from update_plan: pending | in_progress | completed.
const MARK = { completed: '✓', in_progress: '▸', pending: '·' }

export default function Plan({ plan }) {
  if (!plan.length) return null
  return (
    <div className="panel">
      <h3>Plan <span className="count">{plan.filter((i) => i.status === 'completed').length}/{plan.length}</span></h3>
      {plan.map((item, index) => (
        <div key={`${index}-${item.step}`} className={`plan-item ${item.status}`}>
          <span className="mark">{MARK[item.status] || '·'}</span>
          <span>{item.step}</span>
        </div>
      ))}
    </div>
  )
}
