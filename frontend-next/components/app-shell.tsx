'use client'

import Link from 'next/link'
import { BarChart3, BookOpen, Bot, CircleAlert, Database, Menu, Settings, X } from 'lucide-react'
import { useState } from 'react'

const nav = [
  { href: '/', label: 'Home', icon: Bot },
  { href: '/ask-data', label: 'Ask data', icon: Bot },
  { href: '/definitions', label: 'Definitions', icon: BookOpen },
  { href: '/hitl', label: 'HITL queue', icon: CircleAlert },
  { href: '/analytics', label: 'Analytics', icon: BarChart3 },
]

export function AppShell({ children, title, eyebrow }: { children: React.ReactNode; title: string; eyebrow: string }) {
  const [open, setOpen] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [compact, setCompact] = useState(false)
  return <div className={`app-shell${compact ? ' compact-mode' : ''}`}><aside className={open ? 'sidebar open' : 'sidebar'}><div className="app-logo"><span className="logo-grid"><i /><i /><i /></span><span><b>DEFINITION DRIFT</b><small>INTELLIGENCE OS</small></span></div><nav className="app-nav">{nav.map(({ href, label, icon: Icon }) => <Link key={href} href={href} onClick={() => setOpen(false)}><Icon />{label}</Link>)}</nav><div className="sidebar-bottom"><button className="sources-link" type="button" onClick={() => setSettingsOpen(true)}><Database /> <span>Connected sources<small>4 systems online</small></span></button><button className="settings-link" type="button" onClick={() => setSettingsOpen(true)}><Settings /> Settings</button></div></aside><button className="mobile-menu" aria-label="Toggle navigation" onClick={() => setOpen(!open)}>{open ? <X /> : <Menu />}</button><main className="app-main"><header className="app-header"><div><p className="app-eyebrow">{eyebrow}</p><h1>{title}</h1></div><div className="status-chip"><span /> Systems nominal</div></header>{children}</main>{settingsOpen && <div className="settings-backdrop" role="presentation" onMouseDown={() => setSettingsOpen(false)}><section className="settings-panel" role="dialog" aria-modal="true" aria-labelledby="settings-title" onMouseDown={(event) => event.stopPropagation()}><div className="settings-heading"><div><p className="app-eyebrow">WORKSPACE PREFERENCES</p><h2 id="settings-title">Settings</h2></div><button type="button" aria-label="Close settings" onClick={() => setSettingsOpen(false)}><X /></button></div><label className="setting-row"><span><b>Compact workspace</b><small>Reduce spacing across dashboard panels.</small></span><input type="checkbox" checked={compact} onChange={(event) => setCompact(event.target.checked)} /></label><label className="setting-row"><span><b>Connected sources</b><small>Show the four connected systems as online.</small></span><input type="checkbox" defaultChecked /></label><button className="solid-button settings-done" type="button" onClick={() => setSettingsOpen(false)}>Save settings</button></section></div>}</div>
}

export function PageFrame({ children, title, eyebrow }: { children: React.ReactNode; title: string; eyebrow: string }) { return <AppShell title={title} eyebrow={eyebrow}>{children}</AppShell> }

// ── API helpers ───────────────────────────────────────────────────────────────
export const apiUrl = process.env.NEXT_PUBLIC_API_URL || ''
export const adminToken = process.env.NEXT_PUBLIC_ADMIN_TOKEN || ''

/**
 * apiFetch — wraps fetch with base URL + JSON content-type.
 * Automatically adds X-API-Key for write operations (POST/PUT/DELETE).
 */
export const apiFetch = (path: string, options?: RequestInit) => {
  const method = options?.method?.toUpperCase() || 'GET'
  const isWrite = ['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string> || {}),
  }
  if (isWrite && adminToken) {
    headers['X-API-Key'] = adminToken
  }
  return fetch(`${apiUrl}${path}`, { ...options, headers }).then(async r => {
    if (!r.ok) throw new Error(`Request failed: ${r.status}`)
    return r.json()
  })
}

// ── Fallback / demo data (used when API is unreachable) ───────────────────────
export const definitions = [
  { id: 'gross-sales', name: 'Gross Sales', description: 'Total value of all orders before discounts, returns, and taxes.', sql: 'SUM(SalesAmount)', tags: ['revenue', 'commerce'], owner: 'Finance', approved: true },
  { id: 'net-revenue', name: 'Net Revenue', description: 'Gross sales minus discounts and returns, excluding tax.', sql: 'SUM(SalesAmount - ReturnAmount)', tags: ['revenue', 'finance'], owner: 'Finance', approved: true },
  { id: 'active-customers', name: 'Active Customers', description: 'Unique customers with at least one completed order in the period.', sql: 'COUNT(DISTINCT CustomerKey)', tags: ['customers', 'growth'], owner: 'Growth', approved: true },
  { id: 'repeat-rate', name: 'Repeat Purchase Rate', description: 'Share of customers who placed more than one completed order.', sql: 'repeat_customers / total_customers', tags: ['retention'], owner: 'Growth', approved: true },
]

export const conflicts = [
  { id: 'conf-102', question: 'What was our revenue last month?', matched: 'Net Revenue', score: '82%', definition: 'Revenue means gross sales less refunds, before tax.' },
  { id: 'conf-098', question: 'How many active accounts do we have?', matched: 'Active Customers', score: '76%', definition: 'Active account means a customer with a login in the last 30 days.' },
]
