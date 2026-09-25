'use client'

import { useState } from 'react'
import { ArrowUp, Check, Copy, Database, GitBranch, Sparkles, ThumbsDown, ThumbsUp } from 'lucide-react'
import { PageFrame, apiFetch } from '@/components/app-shell'

// ── Types matching api/main.py response ───────────────────────────────────────
interface SqlResult {
  sql: string
  explanation?: string
  provider_used: string
  confidence: string
  cache_hit?: boolean
  used_definitions?: string[]
  warning?: string
}

interface QueryResponse {
  query_id: string
  session_id: string
  latency_ms: number
  status: 'ok' | 'conflict_detected' | 'error'
  question: string
  intent?: string
  sql_result?: SqlResult
  conflict?: {
    conflict_id: string
    matched_definition: string
    similarity: number
    message: string
  }
  conflict_id?: string
  message?: string
}

// Unique session ID per browser tab so multi-turn memory works
const SESSION_ID = typeof crypto !== 'undefined'
  ? crypto.randomUUID().slice(0, 16)
  : 'dd-session-' + Math.random().toString(36).slice(2, 10)

export default function AskDataPage() {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [copied, setCopied] = useState(false)
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)
  const [resolving, setResolving] = useState(false)
  const [mergedText, setMergedText] = useState('')
  const [showMerge, setShowMerge] = useState(false)

  // Submit question to backend
  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!question.trim() || loading) return
    setLoading(true)
    setResult(null)
    setFeedback(null)
    try {
      const data: QueryResponse = await apiFetch('/api/query', {
        method: 'POST',
        body: JSON.stringify({ question: question.trim(), session_id: SESSION_ID }),
      })
      setResult(data)
    } catch (err) {
      setResult({ query_id: '', session_id: SESSION_ID, latency_ms: 0, status: 'error', question, message: String(err) })
    } finally {
      setLoading(false)
    }
  }

  // Send thumbs up / down feedback
  async function sendFeedback(rating: 1 | -1) {
    if (!result?.query_id || feedback) return
    setFeedback(rating === 1 ? 'up' : 'down')
    try {
      await apiFetch('/api/feedback', {
        method: 'POST',
        body: JSON.stringify({ query_id: result.query_id, rating }),
      })
    } catch { /* silent */ }
  }

  // Resolve a conflict from the chat window
  async function resolveConflict(action: 'approve_a' | 'approve_b' | 'merge' | 'reject') {
    if (!result?.conflict_id || resolving) return
    setResolving(true)
    try {
      const body: Record<string, unknown> = { conflict_id: result.conflict_id, action }
      if (action === 'merge') body.merged_definition = mergedText
      await apiFetch('/api/hitl/resolve', { method: 'POST', body: JSON.stringify(body) })
      // After resolution re-submit to get the actual answer
      if (action !== 'reject') {
        const data: QueryResponse = await apiFetch('/api/query', {
          method: 'POST',
          body: JSON.stringify({ question: question.trim(), session_id: SESSION_ID }),
        })
        setResult(data)
      } else {
        setResult(null)
      }
    } catch { /* silent */ } finally {
      setResolving(false)
      setShowMerge(false)
    }
  }

  function copySQL() {
    const sql = result?.sql_result?.sql || ''
    navigator.clipboard?.writeText(sql)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const sqlResult = result?.sql_result

  return (
    <PageFrame eyebrow="CONVERSATIONAL ANALYTICS / 01" title="Ask your data anything.">
      <div className="chat-page" style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>

        {/* Intro / suggestions */}
        <div className="chat-intro" style={{ width: '100%', maxWidth: 980 }}>
          <div className="chat-orb"><Sparkles /></div>
          <h2>From question<br /><em>to confident answer.</em></h2>
          <p>Ask in plain language. Definition Drift routes your question through governed metrics and returns an answer you can trust.</p>
          <div className="suggestions">
            <button onClick={() => setQuestion('What was our net revenue last month?')}>Net revenue last month <ArrowUp /></button>
            <button onClick={() => setQuestion('Which products are driving growth?')}>Products driving growth <ArrowUp /></button>
            <button onClick={() => setQuestion('Give me gross sales SQL')}>Gross sales SQL <ArrowUp /></button>
          </div>
        </div>

        {/* Query input */}
        <form className="query-box" style={{ order: 2, width: '88%', maxWidth: 980, margin: '0 auto' }} onSubmit={submit}>
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Ask about your business metrics or data..."
            aria-label="Ask about your business metrics or data"
            disabled={loading}
          />
          <button type="submit" aria-label="Submit question" disabled={loading || !question.trim()}>
            <ArrowUp />
          </button>
        </form>

        {/* Response area */}
        <div className="conversation" style={{ order: 3, width: '100%', minHeight: 560 }}>

          {/* Empty state */}
          {!result && !loading && (
            <div className="empty-chat">
              <Database />
              <p>Your answers will appear here</p>
              <small>Start with a question about your business metrics.</small>
            </div>
          )}

          {/* Loading skeleton */}
          {loading && (
            <div className="answer-card" style={{ opacity: 0.6 }}>
              <div className="answer-head">
                <div className="answer-icon"><Sparkles /></div>
                <div><b>Analysing…</b><small>Mapping your question across the definition graph</small></div>
              </div>
              <p className="answer-text" style={{ color: 'var(--text-muted)' }}>
                Routing intent → checking governance layer → generating SQL…
              </p>
            </div>
          )}

          {/* Error */}
          {result?.status === 'error' && !loading && (
            <div className="answer-card" style={{ borderColor: 'rgba(239,68,68,.4)' }}>
              <div className="answer-head">
                <div className="answer-icon" style={{ background: 'rgba(239,68,68,.15)' }}>⚠</div>
                <div><b>Error</b><small>{result.message || 'Something went wrong'}</small></div>
              </div>
            </div>
          )}

          {/* Conflict detected → HITL card */}
          {result?.status === 'conflict_detected' && !loading && (
            <div className="conflict-card">
              <div className="conflict-title">
                <span><GitBranch /> Definition Conflict Detected</span>
                <small>HUMAN REVIEW REQUIRED</small>
              </div>
              <p>{result.message}</p>
              <blockquote>
                <strong>Matched:</strong> {result.conflict?.matched_definition} &nbsp;
                <span style={{ opacity: 0.6 }}>({Math.round((result.conflict?.similarity || 0) * 100)}% similar)</span>
              </blockquote>
              <div className="conflict-actions">
                <button onClick={() => resolveConflict('approve_b')} disabled={resolving}>✓ Keep existing</button>
                <button onClick={() => resolveConflict('approve_a')} disabled={resolving}>Create new</button>
                <button onClick={() => setShowMerge(!showMerge)} disabled={resolving}>Merge</button>
                <button onClick={() => resolveConflict('reject')} disabled={resolving}>Reject</button>
              </div>
              {showMerge && (
                <>
                  <textarea
                    value={mergedText}
                    onChange={e => setMergedText(e.target.value)}
                    placeholder="Edit the combined definition…"
                    rows={4}
                    style={{ width: '100%', marginTop: 12, padding: 10, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 8, color: 'inherit', resize: 'vertical' }}
                  />
                  <button className="solid-button" style={{ marginTop: 8 }} onClick={() => resolveConflict('merge')} disabled={resolving || !mergedText.trim()}>
                    Save merge
                  </button>
                </>
              )}
            </div>
          )}

          {/* Successful answer */}
          {result?.status === 'ok' && sqlResult && !loading && (
            <div className="answer-card">
              <div className="answer-head">
                <div className="answer-icon"><Check /></div>
                <div>
                  <b>Answer generated</b>
                  <small>Definition Drift found a governed path</small>
                </div>
                {sqlResult.cache_hit && <span className="cache-badge">Cache Hit</span>}
              </div>

              {/* Explanation */}
              {sqlResult.explanation && (
                <p className="answer-text">{sqlResult.explanation}</p>
              )}

              {/* Warning */}
              {sqlResult.warning && (
                <p style={{ color: '#f59e0b', fontSize: 13, margin: '8px 0' }}>⚠ {sqlResult.warning}</p>
              )}

              {/* SQL block */}
              {sqlResult.sql && (
                <div className="sql-block">
                  <div>
                    <span>SQL EXPRESSION</span>
                    <button onClick={copySQL}>{copied ? 'Copied' : <><Copy /> Copy</>}</button>
                  </div>
                  <code style={{ whiteSpace: 'pre-wrap', display: 'block' }}>{sqlResult.sql}</code>
                </div>
              )}

              {/* Meta */}
              <div className="answer-meta">
                <span>Provider: {sqlResult.provider_used || 'unknown'}</span>
                <span>Latency: {result.latency_ms}ms</span>
                <span>Intent: {result.intent || '—'}</span>
                {sqlResult.used_definitions?.length ? (
                  <span>Defs: {sqlResult.used_definitions.join(', ')}</span>
                ) : null}
                <div className="feedback">
                  <button
                    aria-label="Upvote"
                    onClick={() => sendFeedback(1)}
                    style={{ opacity: feedback && feedback !== 'up' ? 0.3 : 1 }}
                  >
                    <ThumbsUp />
                  </button>
                  <button
                    aria-label="Downvote"
                    onClick={() => sendFeedback(-1)}
                    style={{ opacity: feedback && feedback !== 'down' ? 0.3 : 1 }}
                  >
                    <ThumbsDown />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        <style jsx>{`
          .chat-intro { width:100%; max-width:980px; }
          .chat-intro h2 { font-size:clamp(42px,5vw,72px); }
          .suggestions { display:flex; flex-wrap:wrap; gap:10px; margin-top:24px; }
          .suggestions button { transition:transform .2s ease, border-color .2s ease, background .2s ease; }
          .suggestions button:hover { transform:translateY(-2px); border-color:#60b1d1; background:rgba(96,177,209,.1); }
          .query-box { order:2; width:min(88%,980px); margin:0 auto; }
          .conversation { order:3; width:100%; min-height:560px; }
          .conversation:has(.empty-chat) { display:flex; align-items:center; justify-content:center; }
          .answer-card, .conflict-card { min-height:520px; }
          @media (max-width:720px) { .query-box { width:100%; } .conversation { min-height:440px; } .answer-card, .conflict-card { min-height:400px; } }
        `}</style>
      </div>
    </PageFrame>
  )
}
