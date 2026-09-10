# DefinitionDrift

> An agentic Natural Language-to-SQL system with governed business definitions, semantic retrieval, conflict detection, SQL validation, schema-drift monitoring, and human-in-the-loop review.

---

## 📌 Overview

DefinitionDrift is an agentic Natural Language-to-SQL system designed to solve a common problem in analytics systems: **the same business term can have different meanings across teams**.

Instead of directly converting a user's question into SQL, DefinitionDrift introduces a governance layer between natural language and database execution.

The system:

1. Understands the user's natural-language question.
2. Retrieves relevant approved business definitions.
3. Detects potential definition conflicts.
4. Uses an LLM to generate SQL.
5. Parses and validates the generated SQL.
6. Checks SQL against the database schema.
7. Executes only validated SQL.
8. Returns the result to the user.
9. Stores conversation context for future queries.
10. Monitors the database for schema drift.

---

## 🎯 Problem Statement

Traditional Text-to-SQL systems generally follow:

```text
Natural Language
       ↓
      LLM
       ↓
      SQL
       ↓
   Database
