'use client'

import { useState } from 'react'
import { Check, GitMerge, Plus, X } from 'lucide-react'
import useSWR from 'swr'
import { PageFrame, apiFetch, conflicts as fallbackConflicts } from '@/components/app-shell'

interface Conflict {
  id: string
  question_a: string   // backend field name
  def_b: string        // matched definition name
  similarity: number   // 0-1 float
  status: string
}

export default function HitlPage() {
  const { data, mutate } = useSWR('/api/hitl/queue', apiFetch, {
    fallbackData: { conflicts: fallbackConflicts },
    shouldRetryOnError: false,
    refreshInterval: 10000, // auto-refresh every 10 seconds
  })

  // Backend returns { pending_count, conflicts: [...] }
  const items: Conflict[] = data?.conflicts ?? fallbackConflicts

  const [resolved, setResolved] = useState<string[]>([])
  const [mergeId, setMergeId] = useState<string | null>(null)
  const [mergeText, setMergeText] = useState('')
  const [busy, setBusy] = useState<string | null>(null)

  async function resolve(id: string, action: 'approve_a' | 'approve_b' | 'merge' | 'reject', extra?: string) {
    setBusy(id)
    try {
      const body: Record<string, unknown> = { conflict_id: id, action }
      if (action === 'merge') body.merged_definition = extra
      await apiFetch('/api/hitl/resolve', { method: 'POST', body: JSON.stringify(body) })
      setResolved(prev => [...prev, id])
      await mutate()
    } catch { /* silent */ } finally {
      setBusy(null)
      setMergeId(null)
      setMergeText('')
    }
  }

  const pending = items.filter(item => !resolved.includes(item.id))

  return (
    <PageFrame eyebrow="HUMAN-IN-THE-LOOP / 03" title="Resolve definition conflicts.">
      <div className="toolbar">
        <p>Data steward review keeps ambiguity from becoming organisational drift.</p>
        <span className="queue-count">{pending.length} pending conflict{pending.length !== 1 ? 's' : ''}</span>
      </div>

      <div className="table-card">
        <div className="table-head">
          <span>Question</span>
          <span>Matched definition</span>
          <span>Similarity</span>
          <span>Actions</span>
        </div>

        {pending.length === 0 && (
          <div style={{ padding: '40px 24px', textAlign: 'center', opacity: 0.5 }}>
            <Check size={32} style={{ marginBottom: 8 }} />
            <p>No pending conflicts. Great governance!</p>
          </div>
        )}

        {pending.map((item) => (
          <div className="table-row" key={item.id}>
            {/* Question */}
            <div>
              <b>{item.question_a}</b>
            </div>

            {/* Matched definition */}
            <strong>{item.def_b}</strong>

            {/* Similarity score */}
            <span className="score">
              {item.similarity != null
                ? `${Math.round(item.similarity * 100)}%`
                : '—'}
            </span>

            {/* Actions */}
            <div className="row-actions" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {/* Keep existing (approve_b = use the already-approved definition) */}
              <button
                title="Keep existing"
                onClick={() => resolve(item.id, 'approve_b')}
                disabled={busy === item.id}
                aria-label="Keep existing"
              >
                <Check size={14} /> Keep
              </button>

              {/* Create new (approve_a = register the new question as its own metric) */}
              <button
                title="Create new"
                onClick={() => resolve(item.id, 'approve_a')}
                disabled={busy === item.id}
                aria-label="Create new definition"
              >
                <Plus size={14} /> Create
              </button>

              {/* Merge */}
              <button
                title="Merge definitions"
                onClick={() => { setMergeId(item.id); setMergeText('') }}
                disabled={busy === item.id}
                aria-label="Merge definitions"
              >
                <GitMerge size={14} /> Merge
              </button>

              {/* Reject */}
              <button
                title="Reject and discard"
                onClick={() => resolve(item.id, 'reject')}
                disabled={busy === item.id}
                aria-label="Reject"
              >
                <X size={14} /> Reject
              </button>
            </div>

            {/* Inline merge editor */}
            {mergeId === item.id && (
              <div style={{ gridColumn: '1/-1', padding: '12px 0', display: 'flex', flexDirection: 'column', gap: 8 }}>
                <textarea
                  rows={3}
                  value={mergeText}
                  onChange={e => setMergeText(e.target.value)}
                  placeholder="Write the merged definition…"
                  style={{ width: '100%', padding: 10, background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 8, color: 'inherit', resize: 'vertical' }}
                />
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="solid-button" onClick={() => resolve(item.id, 'merge', mergeText)} disabled={!mergeText.trim() || busy === item.id}>
                    Save merge
                  </button>
                  <button onClick={() => setMergeId(null)}>Cancel</button>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </PageFrame>
  )
}
