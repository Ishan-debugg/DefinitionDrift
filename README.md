<div align="center">
  <img src="./frontend-next/public/icon-dark-32x32.png" alt="DefinitionDrift Logo" width="120" height="120">
  
  <h1>DefinitionDrift</h1>
  <strong>AI Data Intelligence OS with Human-in-the-Loop Governance</strong>
  
  <p>
    <a href="https://definitiondrift.ishan-tarkas.in">View Live Demo</a> • 
    <a href="#features">Features</a> • 
    <a href="#architecture">Architecture</a>
  </p>
</div>

---

## 🛑 The Problem
Traditional Text-to-SQL AI agents suffer from a massive flaw: **Metric Hallucination**. If you ask an AI *"What is our net revenue?"*, the AI will blindly guess the SQL formula based on the column names it sees in the database. 

Different analysts define "Net Revenue" differently (e.g., should it include tax? returns? shipping?). When the AI guesses the formula, it provides **highly convincing, yet completely inaccurate data**, causing business leaders to make decisions based on false metrics.

## 🚀 The Solution: DefinitionDrift
**DefinitionDrift** is an enterprise-grade AI Data Engineering assistant that completely solves metric hallucination by introducing a **Strict Governance Layer**.

Instead of guessing, the AI queries a centralized "Definition Registry". If a definition for *"Net Revenue"* is approved by an Admin, the AI is forced to use the exact, company-approved SQL formula. 

### 🛡️ Human-in-the-Loop (HITL)
If a user asks about a metric that has **not** been approved yet (e.g., *"What is our total margin?"*), the AI's execution is immediately paused. A **Human-in-the-Loop (HITL)** alert is triggered, forcing a human Admin to review, correct, and approve the new metric definition before the AI is ever allowed to use it.

---

## ✨ Key Features

- **🗣️ Natural Language to SQL:** Ask complex business questions in plain English. The agent perfectly structures highly advanced SQL queries using CTEs and Joins.
- **🏛️ Governed Definition Registry:** A centralized truth for all business metrics (Gross Sales, Net Revenue, etc.).
- **🚦 Automated Conflict Detection:** The Intent Router uses Vector Embeddings to detect when a user is asking about an unapproved metric, instantly blocking rogue queries.
- **👨‍⚖️ Admin HITL Queue:** A dedicated dashboard for Data Engineers to review, merge, and approve pending AI metric requests.
- **🧠 Multi-Turn Memory:** The agent remembers previous queries, allowing users to ask follow-up questions seamlessly.

---

## 💻 Tech Stack

### Frontend
- **Framework:** Next.js 14 (App Router)
- **Styling:** Tailwind CSS + Glassmorphism UI
- **Icons:** Lucide React
- **Deployment:** Vercel

### Backend
- **Framework:** FastAPI (Python)
- **Agent Orchestration:** LangGraph (Stateful AI Agents)
- **LLM Engine:** Groq Cloud (Llama-3.1 8B for lightning-fast routing & SQL generation)
- **Vector Search:** TF-IDF / Cosine Similarity (Fallback embedded locally)
- **Database:** SQLite / LibSQL (Turso-compatible)
- **Deployment:** Render

---

## 🌐 Live Demo
The application is fully deployed and connected to a live database. 
You can test the AI agent and trigger the Human-in-the-Loop governance layer here:

👉 **[definitiondrift.ishan-tarkas.in](https://definitiondrift.ishan-tarkas.in)**

### Demo Prompts to Try:
1. **The "Perfect Match"** *(Tests Approved Definitions)*
   > *"What is our net revenue?"*
2. **The "HITL Trigger"** *(Tests the Governance Blocking Layer)*
   > *"Update the gross margin definition"*
3. **The "Complex Reasoning"** *(Tests the LLM)*
   > *"Compare the gross sales to the net revenue for our top 3 products."*

---

## 🏗️ Architecture

1. **Intent Router:** The user's query hits a fast LLM classifier. If it's a simple data request, it moves forward. If the user is trying to define a new metric, it routes to the Conflict Agent.
2. **Conflict Agent (Vector Search):** The AI searches the Definition Registry using embeddings to see if this metric already exists or if it conflicts with an unapproved metric.
3. **HITL Interrupt:** If a conflict or unapproved metric is found, the LangGraph state is paused. A ticket is sent to the Admin Dashboard.
4. **SQL Generator:** Once approved, the query + approved definitions are sent to Groq. Groq synthesizes the final, mathematically accurate SQL.
5. **Validation:** The SQL is parsed into an Abstract Syntax Tree (AST) to verify it safely matches the actual database schema before being shown to the user.

---

<div align="center">
  <small>Mail - tarkasishan7536@gmail.com</small>
</div>
