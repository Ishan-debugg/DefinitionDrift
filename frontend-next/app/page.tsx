'use client'

import { useState } from 'react'
import { ArrowRight, Check, ChevronDown, Circle, Menu, Play, Plus, X } from 'lucide-react'

const layers = [
  { number: '01', name: 'ROUTING LAYER', detail: 'Classify', tone: 'silver' },
  { number: '02', name: 'GOVERNANCE LAYER', detail: 'Align', tone: 'violet' },
  { number: '03', name: 'ORCHESTRATION LAYER', detail: 'Coordinate', tone: 'cyan' },
  { number: '04', name: 'GENERATION LAYER', detail: 'Synthesize', tone: 'silver' },
  { number: '05', name: 'VALIDATION LAYER', detail: 'Verify', tone: 'violet' },
]

const dataCards = [
  { label: 'orders_daily', x: '8%', y: '19%', delay: '0s' },
  { label: 'CUSTOMER', x: '5%', y: '58%', delay: '1.5s' },
  { label: 'revenue_metrics', x: '75%', y: '21%', delay: '.8s' },
  { label: 'DIM_PROD', x: '79%', y: '62%', delay: '2.2s' },
]

export default function Page() {
  const [activeLayer, setActiveLayer] = useState(2)
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <main className="site-shell">
      <div className="noise" aria-hidden="true" />
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Definition Drift home"><span className="brand-mark" aria-hidden="true"><span className="agent-eye" /><span className="agent-eye" /><span className="data-ring" /></span><span><strong>DEFINITION DRIFT</strong><small>DATA INTELLIGENCE</small></span></a>
        <button className="menu-button" aria-label="Toggle menu" aria-expanded={menuOpen} onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X size={18} /> : <Menu size={18} />}</button>
        <nav className={menuOpen ? 'nav open' : 'nav'} aria-label="Primary">
          <a href="#engine">Platform</a><a href="#layers">Layers</a><a href="#signal">Signal</a><a href="#docs">Docs</a><a className="login" href="/ask-data">Ask data</a><a className="nav-cta" href="#engine">Explore platform <ArrowRight size={15} /></a>
        </nav>
      </header>
      <section id="top" className="hero">
        {dataCards.map((card) => <div key={card.label} className="ghost-card" style={{ left: card.x, top: card.y, animationDelay: card.delay }}><span>{card.label}</span><i /><i /><i /><i /></div>)}
        <div className="hero-content"><p className="eyebrow"><span className="pulse-dot" /> ANALYTICS INFRASTRUCTURE / 01</p><h1><span>Talk to your data.</span><em>We&apos;ll handle the definitions</em></h1><p className="hero-copy">A living intelligence layer for teams that need to move from raw data to confident decisions.</p><div className="hero-actions"><a className="primary-button" href="#engine">Explore the engine <ArrowRight size={16} /></a><a className="text-button" href="#signal"><Play size={14} fill="currentColor" /> Watch the system</a></div><p className="hero-note"><Check size={14} /> Built for modern data teams, analysts, and operators.</p></div>
        <div className="scroll-cue"><span>SCROLL TO DISCOVER</span><ChevronDown size={16} /></div>
      </section>
      <section id="engine" className="engine-section"><div className="section-intro"><p className="eyebrow">THE DEFINITION DRIFT ENGINE</p><h2>Five layers.<br /><span>One source of truth.</span></h2><p>Data becomes useful when every layer speaks the same language. Definition Drift coordinates the journey from source to signal.</p></div><div id="layers" className="layer-stack" aria-label="Data layers">{layers.map((layer, index) => <button key={layer.number} className={`layer-card ${activeLayer === index ? 'active' : ''} tone-${layer.tone}`} onClick={() => setActiveLayer(index)}><span className="layer-number">{layer.number}</span><span className="layer-name">{layer.name}</span><span className="layer-detail">{layer.detail}</span><span className="layer-line" /></button>)}</div><div className="engine-foot"><span>ACTIVE LAYER / {layers[activeLayer].number}</span><span className="engine-status"><Circle size={7} fill="currentColor" /> SYSTEMS NOMINAL</span></div></section>
      <section id="signal" className="signal-section"><div className="signal-copy"><p className="eyebrow">SEE THE SYSTEM AT WORK</p><h2>Raw inputs.<br /><em>Clear outcomes.</em></h2><p>Every source is mapped, modeled, and made ready for the questions that move your business forward.</p><a className="outline-button" href="/analytics">View analytics <ArrowRight size={15} /></a></div><div className="signal-visual"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="core"><span>DEFINITION DRIFT</span><small>INTELLIGENCE CORE</small></div><div className="signal-label label-a">RAW DATA <b>01</b></div><div className="signal-label label-b">TRUSTED SIGNAL <b>05</b></div><div className="orbit-node node-a" /><div className="orbit-node node-b" /></div></section>
      <section className="robot-panel" aria-label="Definition Drift connected data assistant"><div className="robot-panel-copy"><p className="eyebrow">CONNECTED INTELLIGENCE</p><h2>Ask anything.<br /><em>Find the signal.</em></h2><p>Definition Drift connects your data systems so every question starts with shared context.</p><a className="primary-button" href="/ask-data">Ask your data <ArrowRight size={16} /></a></div><div className="robot-panel-art"><img src="/definition-drift-robot.png" alt="Friendly data assistant connected to business data sources" /></div></section>
      <footer id="contact"><div className="brand"><span className="brand-mark" aria-hidden="true"><span className="agent-eye" /><span className="agent-eye" /><span className="data-ring" /></span><span><strong>DEFINITION DRIFT</strong><small>DATA INTELLIGENCE</small></span></div><span>© 2026 Definition Drift Systems</span><span>BUILT FOR CLARITY <Plus size={13} /></span></footer>
    </main>
  )
}

export { layers }
