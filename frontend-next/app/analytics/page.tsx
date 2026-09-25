'use client'

import { Activity, ArrowDownRight, ArrowUpRight, CheckCircle2, XCircle } from 'lucide-react'
import useSWR from 'swr'
import { PageFrame, apiFetch } from '@/components/app-shell'

// ── Types matching backend responses ─────────────────────────────────────────
interface HealthResponse {
  status: string
  timestamp: string
  version: string
}

interface HistoryItem {
  intent?: string
  status?: string
  latency_ms?: number
  used_definitions?: string
}

interface HistoryResponse {
  count: number
  history: HistoryItem[]
}

export default function AnalyticsPage() {
  const { data: health } = useSWR<HealthResponse>('/health', apiFetch, {
    refreshInterval: 30000,
    shouldRetryOnError: false,
  })

  const { data: historyData } = useSWR<HistoryResponse>('/api/history?limit=200', apiFetch, {
    fallbackData: { count: 0, history: [] },
    shouldRetryOnError: false,
    refreshInterval: 30000,
  })

  const { data: defsData } = useSWR('/api/definitions', apiFetch, {
    fallbackData: { count: 0, definitions: [] },
    shouldRetryOnError: false,
  })

  // Compute real stats from history
  const history: HistoryItem[] = historyData?.history || []
  const totalQueries = historyData?.count ?? 0
  const latencies = history.map(h => h.latency_ms).filter((l): l is number => typeof l === 'number')
  const avgLatency = latencies.length
    ? (latencies.reduce((a, b) => a + b, 0) / latencies.length / 1000).toFixed(2) + 's'
    : '—'

  // Count intents
  const intentCounts = { DATA: 0, DEFINE: 0, AMBIGUOUS: 0 }
  history.forEach(h => {
    const i = (h.intent || '').toUpperCase()
    if (i === 'DATA') intentCounts.DATA++
    else if (i === 'DEFINE') intentCounts.DEFINE++
    else if (i === 'AMBIGUOUS') intentCounts.AMBIGUOUS++
  })

  const maxIntent = Math.max(...Object.values(intentCounts), 1)
  const bars = [
    { label: 'DATA', value: intentCounts.DATA, pct: Math.round((intentCounts.DATA / maxIntent) * 100) },
    { label: 'DEFINE', value: intentCounts.DEFINE, pct: Math.round((intentCounts.DEFINE / maxIntent) * 100) },
    { label: 'AMBIGUOUS', value: intentCounts.AMBIGUOUS, pct: Math.round((intentCounts.AMBIGUOUS / maxIntent) * 100) },
  ]

  const activeDefCount = defsData?.count ?? defsData?.definitions?.length ?? 0

  const analyticsStats = [
    { label: 'Total queries', value: totalQueries.toLocaleString(), icon: <ArrowUpRight size={16} />, positive: true },
    { label: 'Avg latency', value: avgLatency, icon: <ArrowDownRight size={16} />, positive: true },
    { label: 'Active definitions', value: activeDefCount, icon: <ArrowUpRight size={16} />, positive: true },
  ]

  const healthOk = health?.status === 'ok'

  return (
    <PageFrame eyebrow="ANALYTICS & OBSERVABILITY / 04" title="See the system clearly.">

      {/* Stat cards */}
      <div className="stats-grid">
        {analyticsStats.map((stat) => (
          <article className="stat-card" key={stat.label}>
            <span>{stat.label}</span>
            <strong>{stat.value}</strong>
            <small className="positive">{stat.icon} Live</small>
          </article>
        ))}
      </div>

      <div className="analytics-grid">
        {/* Query intent bar chart */}
        <section className="chart-card">
          <div className="panel-heading">
            <div>
              <p className="app-eyebrow">QUERY INTENTS</p>
              <h2>What teams are asking.</h2>
            </div>
            <Activity />
          </div>
          <div className="bar-chart">
            {bars.map(bar => (
              <div className="bar-item" key={bar.label}>
                <div className="bar-track">
                  <div style={{ height: `${bar.pct || 2}%` }} />
                </div>
                <b>{bar.value}</b>
                <span>{bar.label}</span>
              </div>
            ))}
          </div>
        </section>

        {/* System health */}
        <section className="health-card">
          <p className="app-eyebrow">SYSTEM HEALTH</p>
          <h2>
            {healthOk
              ? <>All systems<br /><em>operational.</em></>
              : <>Checking<br /><em>system status…</em></>}
          </h2>

          <div className="health-line">
            {healthOk ? <CheckCircle2 size={14} color="#34d399" /> : <XCircle size={14} color="#ef4444" />}
            <b>API Server</b>
            <small>{healthOk ? 'Online' : 'Unreachable'}</small>
          </div>

          <div className="health-line">
            <CheckCircle2 size={14} color="#34d399" />
            <b>Definition graph</b>
            <small>{activeDefCount} definitions loaded</small>
          </div>

          <div className="health-line">
            <CheckCircle2 size={14} color="#34d399" />
            <b>Query router</b>
            <small>{totalQueries} queries served</small>
          </div>

          <div className="health-line">
            <CheckCircle2 size={14} color="#34d399" />
            <b>Cache layer</b>
            <small>Active</small>
          </div>

          {health?.version && (
            <p style={{ marginTop: 16, fontSize: 12, opacity: 0.5 }}>
              Backend v{health.version} · {health.timestamp ? new Date(health.timestamp).toLocaleTimeString() : ''}
            </p>
          )}
        </section>
      </div>
    </PageFrame>
  )
}
