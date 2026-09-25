'use client'

import { useState } from 'react'
import { Plus, Tag } from 'lucide-react'
import useSWR from 'swr'
import { PageFrame, apiFetch, definitions as fallbackDefs } from '@/components/app-shell'

interface Definition {
  id: string
  name: string
  description: string
  sql_expr?: string
  sql?: string        // some versions return sql
  tags: string[]
  owner?: string
  approved: boolean
}

export default function DefinitionsPage() {
  const [open, setOpen] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState('')

  // Form state
  const [name, setName] = useState('')
  const [desc, setDesc] = useState('')
  const [sql, setSql] = useState('')
  const [tags, setTags] = useState('')

  const { data, mutate } = useSWR('/api/definitions', apiFetch, {
    fallbackData: { definitions: fallbackDefs },
    shouldRetryOnError: false,
  })

  // Backend returns { count, definitions: [...] } — handle both shapes
  const items: Definition[] = data?.definitions ?? data ?? fallbackDefs

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    if (!name.trim() || !desc.trim()) return
    setSaving(true)
    setSaveError('')
    try {
      await apiFetch('/api/definitions', {
        method: 'POST',
        body: JSON.stringify({
          name: name.trim(),
          description: desc.trim(),
          sql_expr: sql.trim() || null,
          tags: tags.split(',').map(t => t.trim()).filter(Boolean),
          approved: true,
        }),
      })
      await mutate()          // refresh the list
      setOpen(false)
      setName(''); setDesc(''); setSql(''); setTags('')
    } catch (err: unknown) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  return (
    <PageFrame eyebrow="DEFINITION GOVERNANCE / 02" title="A shared language for metrics.">
      <div className="toolbar">
        <p>Official definitions keep every question aligned to the same source of truth.</p>
        <button className="solid-button" onClick={() => setOpen(true)}><Plus /> Create Definition</button>
      </div>

      <div className="definition-grid">
        {items.map((item) => (
          <article className="definition-card" key={item.id || item.name}>
            <div className="card-top">
              <span className="metric-icon"><Tag /></span>
              {item.approved && <span style={{ fontSize: 11, color: '#34d399', fontWeight: 600 }}>✓ APPROVED</span>}
              {item.owner && <span className="owner">{item.owner}</span>}
            </div>
            <h3>{item.name}</h3>
            <p>{item.description}</p>
            <code>{item.sql_expr || item.sql}</code>
            <div className="tags">
              {(Array.isArray(item.tags)
                ? item.tags
                : typeof item.tags === 'string'
                  ? JSON.parse(item.tags as string)
                  : []
              ).map((tag: string) => <span key={tag}>#{tag}</span>)}
            </div>
          </article>
        ))}
      </div>

      {/* Create Definition Modal */}
      {open && (
        <div className="modal-backdrop" onClick={() => setOpen(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setOpen(false)}>×</button>
            <p className="app-eyebrow">NEW GOVERNED METRIC</p>
            <h2>Create Definition</h2>
            <form onSubmit={handleCreate} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <label>
                Metric name *
                <input
                  required
                  value={name}
                  onChange={e => setName(e.target.value)}
                  placeholder="e.g. Conversion Rate"
                />
              </label>
              <label>
                Description *
                <textarea
                  required
                  value={desc}
                  onChange={e => setDesc(e.target.value)}
                  placeholder="What does this metric mean?"
                  rows={3}
                />
              </label>
              <label>
                SQL expression
                <input
                  value={sql}
                  onChange={e => setSql(e.target.value)}
                  placeholder="SUM(...) / COUNT(...)"
                />
              </label>
              <label>
                Tags (comma-separated)
                <input
                  value={tags}
                  onChange={e => setTags(e.target.value)}
                  placeholder="revenue, finance"
                />
              </label>
              {saveError && <p style={{ color: '#ef4444', fontSize: 13 }}>⚠ {saveError}</p>}
              <button className="solid-button" type="submit" disabled={saving}>
                {saving ? 'Saving…' : 'Save definition'}
              </button>
            </form>
          </div>
        </div>
      )}
    </PageFrame>
  )
}
