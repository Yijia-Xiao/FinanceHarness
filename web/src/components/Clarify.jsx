// The scoping pass. The backbone asks only when a question is genuinely
// ambiguous, so this dialog is the exception, not the rule — and every question
// is skippable, because proceeding on a sensible default is usually right.

import { useState } from 'react'

export default function Clarify({ result, question, onSubmit, onSkip }) {
  const [answers, setAnswers] = useState({})
  const [free, setFree] = useState({})

  const set = (id, value) => setAnswers((prev) => ({ ...prev, [id]: value }))

  const submit = () => {
    const payload = result.questions
      .map((q) => ({
        question: q.question,
        answer: (free[q.id] || answers[q.id] || '').trim(),
      }))
      .filter((entry) => entry.answer)
    onSubmit(payload)
  }

  return (
    <div className="scrim" role="dialog" aria-modal="true">
      <div className="modal">
        <h2>Before I dig in</h2>
        <p className="lede">
          “{question}” could go a few ways. Answer what matters, skip the rest — anything
          unanswered gets a sensible default.
        </p>

        {!!result.assumptions?.length && (
          <div className="assumptions">
            Proceeding with: {result.assumptions.join(' · ')}
          </div>
        )}

        {result.questions.map((q) => (
          <div className="question" key={q.id}>
            <div className="q">{q.question}</div>
            <div className="options">
              {q.options?.map((option) => (
                <button
                  key={option}
                  className={`tiny ${answers[q.id] === option ? 'on' : ''}`}
                  onClick={() => {
                    set(q.id, option)
                    setFree((prev) => ({ ...prev, [q.id]: '' }))
                  }}
                >
                  {option}
                </button>
              ))}
            </div>
            {q.allow_free_text && (
              <input
                type="text"
                placeholder="or type your own"
                value={free[q.id] || ''}
                onChange={(e) => setFree((prev) => ({ ...prev, [q.id]: e.target.value }))}
                style={{ width: '100%' }}
              />
            )}
          </div>
        ))}

        <div className="actions">
          <button onClick={onSkip}>Skip, use defaults</button>
          <button className="primary" onClick={submit}>Research</button>
        </div>
      </div>
    </div>
  )
}
