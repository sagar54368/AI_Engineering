# AI Engineering Mastery — Two Linked Projects (free, laptop-only)

> Target: one 16 GB Apple-Silicon laptop · Docker · uv · free tools only.
> Researched 2026-10-04 (deep notes in `research/`). Free tiers and library APIs move fast — each file flags what to re-verify.

## Why two projects, not one
One mega-project would mix two different skills and you'd learn neither cleanly:

| | **Project 1 — NiveshRAG** | **Project 2 — Portfolio Research Desk** |
|---|---|---|
| Core skill | RAG (basics → advanced) + MLOps + LLMOps | Agentic AI + LangChain/LangGraph/LangSmith + AI automations |
| What it is | Cited Q&A over Indian financial documents (annual reports, results, concall transcripts, SEBI circulars, RBI policy) | Give a stock universe → multi-agent desk (fundamental / technical / news / risk / regime pillars) builds and manages a **paper** portfolio with human approval, scheduled reports, MCP access |
| Why this order | Agents are only as good as their tools. P1 builds the best tool P2 will use (filings search + financials SQL) and the eval/MLOps habits P2 needs | Reuses P1 as an MCP/agentic-RAG tool, P1's LiteLLM gateway, MLflow, eval harness — you add only the agent layer |
| Classical ML model | Query-intent classifier (TF-IDF+LR → SetFit) — routes queries, full MLflow champion/challenger | Market-regime model (HMM + LightGBM) — walk-forward, drift, delayed labels |
| Portfolio story | "Ablation table: each RAG technique's measured lift on Indian filings" | "Auditable multi-agent desk: deterministic spine, agent leaves, HITL, evals in CI" |

**Why not both at once:** you'd debug agent loops before your retriever is trustworthy — every agent failure would be ambiguous (bad retrieval? bad prompt? bad routing?). Sequencing removes that ambiguity.

## Files
| File | Read when |
|---|---|
| `00_SETUP_AND_STACK.md` | Day 1 — install, keys, model routing across free providers (Gemini, NVIDIA, Groq, OpenRouter, Ollama), RAM budget, every tool choice in one table |
| `PROJECT_1_NiveshRAG.md` | Weeks 1–12 — 13 phases, each: build → concepts → why → alternatives → eval gate |
| `PROJECT_2_PortfolioDesk.md` | Weeks 13–22 — 11 phases, same format |
| `research/01..04_*.md` | Deep reference: every technique with intuition, papers, verified links |

## How to use the outlines
1. Do phases **in order**; never start a phase until the previous **eval gate** passes. The gate is the proof you understood it.
2. For every phase write a 5-line note in `docs/learning_log.md`: *what failed before, what fixed it, the number that moved.* That log becomes your interview stories.
3. Build each technique **once by hand (plain Python), then with the framework** (LangChain/LlamaIndex/LangGraph). Hand-built = intuition; framework = industry fluency.
4. Commit after every phase with a tag (`p1-phase3`). Results go to MLflow + `eval/results/*.json`.

## Rules (non-negotiable)
- **Keys only in `.env`** (gitignored) + `gitleaks` pre-commit. Never paste keys into chats, issues or notebooks; rotate any key that leaks.
- **Free tiers = public data only.** Gemini free tier, NVIDIA API catalog, OpenRouter `:free` may log/review prompts. Anything private → local Ollama only.
- **Paper trading only.** No order-placement tool exists anywhere in P2 (capability removal > prompting). Outputs are educational, not investment advice (SEBI RA rules).
- Respect source ToS: no Screener.in scraping, polite throttled downloads of public SEBI/RBI/exchange PDFs, commit URL manifests not PDFs.

## Time budget (part-time ~12–15 h/week)
P1: ~12 weeks · P2: ~10 weeks. Shortcut path if time-boxed: P1 phases 0–6 + 11–12 (core RAG + MLOps/LLMOps), then P2 phases 0–7.
