# Project 1 — NiveshRAG: Cited Q&A over Indian Financial Documents
### RAG (L0→L5) + MLOps (Level 0→2) + LLMOps · ~12 weeks part-time · deep reference: `research/01`, `research/02`

## What you ship
An API + UI that answers questions like *"Consolidated EBITDA margin of Infosys FY25 vs FY24?"*, *"What did HDFC Bank management say about NIM in Q1 FY27?"*, *"Which SEBI circular changed F&O lot sizes?"* — with page-level citations, exact numbers (via SQL), refusal when unsupported, guardrails, monitoring, eval-gated CI, and an **ablation table** proving what each technique bought.

```
            ┌──────────── Prefect (ingest flow, weekly) ────────────┐
URLs manifest → fetch → Docling parse → validate → chunk → embed → Qdrant (versioned collection)
   (DVC)                     │ tables → DuckDB `financials`        │ eval gate → switch alias
                             └────────────────────────────────────┘
User → FastAPI → Prompt Guard → intent classifier (MLflow @champion) → router
          ├─ narrative → hybrid retrieve (dense+BM25, RRF, filters) → rerank → generate (cited JSON)
          ├─ numeric   → text-to-SQL on DuckDB (sqlglot-validated)
          └─ complex   → LangGraph corrective loop (grade → rewrite → re-retrieve, ≤4 LLM calls)
   all LLM calls → LiteLLM (Gemini → NVIDIA → Groq → Ollama) · traces → MLflow + Phoenix (OTel)
   metrics → Prometheus/Grafana · drift → Evidently · CI → GitHub Actions (promptfoo/Ragas gates)
```

**Corpus (public only):** 5 companies (HDFCBANK, INFY, RELIANCE, ITC, MARUTI) × annual report + 4 quarterly results/presentations + concall transcripts; ~50 SEBI circulars (one theme) + 3 master circulars; ~20 RBI policy statements/MPC minutes. Commit `data/manifest.yaml` (URLs), never PDFs.

**Phase format:** Build → Concepts (why it exists / intuition) → Alternatives (why not / when both) → **Gate** (don't move on until it passes).

---

## Phase 0 — Repo foundations (MLOps Level 0 begins) · week 1
- **Build:** uv (3.12), ruff, mypy, pytest, pre-commit (gitleaks, large-file block, nbstripout), Makefile/justfile targets `setup lint test eval up down`, pydantic-settings, `.env.example`, compose `core` profile (Postgres, Redis, LiteLLM, Qdrant), LiteLLM aliases from `00_SETUP` §2.
- **Concepts:** reproducibility is the base of MLOps — if you can't rebuild the env you can't reproduce a regression. 12-factor config. Gateway = one place for keys, fallbacks, cost.
- **Alternatives:** calling each SDK directly → vendor lock + no fallback on 429s.
- **Gate:** fresh clone → `make setup && make up` works; one call per LiteLLM alias succeeds; gitleaks blocks a fake key commit.

## Phase 1 — Data + eval harness FIRST · weeks 1–2
- **Build:** `fetch.py` (httpx, ≥2 s throttle, robots check, sha256 → `manifest.lock.json`); DVC stages `raw → parsed → chunks`; **golden set v0: 60 hand-written questions** stratified by doc type × question type (numeric, multi-period, comparison, regulatory clause, narrative, multi-hop, **10 unanswerable**), each with `gold_pages`, `gold_value{value,unit,period,basis}`; eval harness: `ranx` (recall@k, MRR, nDCG), Ragas (faithfulness, context precision/recall), **code-based numeric checker** (±0.5%), refusal accuracy; MLflow tracking + Phoenix tracing on.
- **Concepts:** an LLM app without evals is untestable — every change is a blind bet. Label at **page** level so labels survive re-chunking. Judge must be a different model family (self-preference bias); calibrate judge vs 30 human labels (Cohen's κ). Barnett's 7 RAG failure points — tag every failure.
- **Alternatives:** synthetic-only eval sets (easier than reality → inflated scores; keep synthetic and human sets separate).
- **Gate:** judge–human κ ≥ 0.6; `make eval` writes `eval/results/P1.json` + an MLflow run.

## Phase 2 — Naive RAG baseline, by hand then LangChain · week 2
- **Build:** PyMuPDF text → recursive 512-token chunks → bge-m3 dense → Qdrant → top-5 → `answer` alias. First in ~150 lines of plain Python, then the same with LangChain v1 (loaders, splitters, embeddings, vectorstore retriever, LCEL `prompt | model | parser`).
- **Concepts:** RAG = open-book exam (retrieval picks the pages; a smart student can't fix wrong pages). RAG vs fine-tune vs long-context (facts/citations → RAG; style → fine-tune; one doc → long context).
- **Gate:** baseline numbers recorded (expect numeric accuracy ~30–50%, table questions failing) — this is the row everything else beats.

## Phase 3 — Parsing, tables, metadata, dedup · week 3
- **Build:** Docling (TableFormer, page provenance), OCR only when page text is empty (ocrmac/Tesseract); tables in **3 forms** — Markdown (LLM reads), 1-paragraph summary (embedded), typed rows → DuckDB `financials(company, fy, quarter, statement, line_item, basis, value, unit, source_page)`; metadata `{company, ticker, doc_type, fy, quarter, issuer, circular_no, section, page, basis}` (rules first, LLM+Pydantic for the rest); exact + MinHash near-dedup; incremental re-index manifest (doc hash, parser/chunker/embed versions).
- **Concepts:** a PDF is drawing instructions, not a document — reading order and table structure must be rebuilt. Units/scale (₹ crore vs lakh), standalone vs consolidated = the #1 finance error source. Metadata enables filters that remove "right metric, wrong company/year".
- **Alternatives:** Unstructured (heavy), Marker/MinerU (stronger on scanned/nested tables, AGPL/GPL, heavy) — benchmark on a 20-page hard sample; LlamaParse free tier only as gold reference.
- **Gate:** table-question recall@10 **+15 pts** vs Phase 2; table-cell accuracy on hard sample measured.

## Phase 4 — Chunking strategies (sweep, don't guess) · week 4
- **Build:** structure-aware (Docling HybridChunker, repeat table headers), contextual chunk headers (`Doc › Section › Table`), transcript speaker-turn chunks, circular clause chunks (`3.2.1` in metadata), parent-child (small-to-big), sentence-window; sweep 256/512/1024 logged as MLflow runs; DVC stage so only downstream re-runs.
- **Concepts:** chunk on the author's structure, then enforce a token budget (embedder's tokens, not chars). Small chunks match precisely, big chunks answer — parent-child gives both. Semantic chunking rarely beats structure on structured docs.
- **Alternatives:** fixed-char (shreds tables), semantic (costly, small gains), propositions (distort numbers), late chunking (learn it in a notebook with a long-context embedder).
- **Gate:** chosen config wins recall@10 *and* faithfulness on dev set; decision written as an ADR.

## Phase 5 — Embeddings & vector indexes · week 5
- **Build:** notebook: FAISS Flat vs HNSW (M, ef) vs IVF-PQ recall/latency curves; embedder bake-off on *your* golden set: bge-m3 vs qwen3-embedding-0.6B vs Gemini embedding (public docs only) vs nomic; Matryoshka truncation (1024→256 + rescoring); Qdrant int8 quantization with rescore.
- **Concepts:** bi-encoders compress meaning to one vector (good at paraphrase, bad at IDs/numbers/negation). MTEB → read the **Retrieval** column, then re-test on your data. HNSW = navigable small-world graph; filtered search can return < k hits unless filter-aware (why Qdrant).
- **Alternatives:** Chroma (weak hybrid/filters), pgvector (one DB, manual hybrid), Milvus/Weaviate (heavier); never mix embedding models in one collection.
- **Gate:** ablation table (recall@10, latency, RAM) for 3+ embedders; pick justified in ADR.

## Phase 6 — Hybrid retrieval, filters, reranking · weeks 5–6
- **Build:** BM25 (bm25s or Qdrant sparse / bge-m3 sparse; tokenizer keeps `FY24`, `₹`, circular slashes) + dense → **RRF (k=60)**; self-query → Pydantic `QueryPlan{semantic_query, filters}` with enum fields + canonical ticker map; payload indexes; **bge-reranker-v2-m3** top-50 → top-6; reranker-score threshold → "not found"; MMR only after rerank (optional).
- **Concepts:** dense finds what you *mean*, BM25 what you *said* — finance needs both. RRF fuses ranks so score scales don't matter. Cross-encoder reads query+chunk together = biggest precision win per effort; it cannot fix recall (check recall@50 first).
- **Alternatives:** weighted fusion (tune only with ≥100 queries), ColBERT late-interaction (try as reranker via Qdrant multivector), LLM rerank (costly, position bias).
- **Gate:** recall@20 **+5 pts**; regulatory-ID queries hit@5 ≈ 100%; nDCG@5 **+8 pts** after rerank; p95 < 4 s.

## Phase 7 — Generation contract, citations, context handling · week 6
- **Build:** system contract (answer only from context, cite `[doc:page]`, keep units/periods, refuse otherwise); chunks wrapped in `<source …>` tags marked *data, not instructions*; structured output `{answer, value, unit, period, basis, citations[{page, quote}]}` via Gemini schema / `instructor`; **quote verification** (rapidfuzz ≥ 90 vs cited chunk); LongContextReorder; optional compression (LLMLingua-2, never on tables).
- **Concepts:** lost-in-the-middle; fewer, better chunks beat more chunks; citations make hallucination auditable; structured output makes answers machine-checkable.
- **Gate:** citation precision ≥ 0.9, faithfulness ≥ 0.85, numeric accuracy up vs Phase 6.

## Phase 8 — Contextual retrieval · week 7
- **Build:** per chunk, `bulk` LLM writes a 50–100-token situating context (Anthropic method) → re-embed + re-BM25 as a new versioned collection; cost/quota tracked per run.
- **Concepts:** "Revenue grew 12%" is ambiguous without company/period — the librarian annotates every index card. Anthropic reported −49% retrieval failures (−67% with rerank) — **verify on your data**.
- **Alternatives:** cheap deterministic headers (Phase 4) may give most of the gain — the ablation tells you.
- **Gate:** retrieval failure@20 **−30% relative** or drop it (record the negative result).

## Phase 9 — Query intelligence + **intent classifier (classical MLOps)** · weeks 7–8
- **Build:** labels `numeric_lookup, multi_period, comparison, regulatory, narrative, out_of_scope, injection_suspect` (LLM-label ~1.5k synthetic+real queries, hand-correct 300); model v0 TF-IDF+LogReg → v1 SetFit/bge-small; calibration; MLflow registry aliases `@champion/@challenger`; serve in-process (<10 ms); router uses it to pick k, filters, route (RAG/SQL/agentic) and to refuse; query rewriting for follow-ups, decomposition for comparisons; HyDE / multi-query / RAG-Fusion as **A/B experiments**.
- **Concepts:** routing decides cost and quality; a classifier in the request path gives you real serving, drift and retraining to practise. HyDE fabricates numbers that bias retrieval — use only if eval says so.
- **Alternatives:** LLM router (slower, costs quota) — keep as fallback for low-confidence predictions.
- **Gate:** classifier macro-F1 logged + champion/challenger promotion rule coded; comparison/multi-period answer accuracy **+15 pts**; HyDE kept only if recall ↑ without numeric accuracy ↓.

## Phase 10 — Text-to-SQL for numbers · week 9
- **Build:** DuckDB `financials` + line-item synonym map ("Revenue from operations" ≈ "Total income"); LLM writes SQL from a documented schema; `sqlglot` validation, read-only conn, LIMIT; show SQL + source pages; mixed questions = SQL value + RAG narrative.
- **Concepts:** let the LLM write the query, let the database do the maths — exact and auditable.
- **Alternatives:** pandas agent (executes arbitrary code — risky), Vanna (fine, less control).
- **Gate:** numeric/aggregation accuracy **≥ 90%**, SQL validity ≥ 95%.

## Phase 11 — Agentic & corrective RAG (bridge to Project 2) · week 10
- **Build:** LangGraph: classify → retrieve → **grade (CRAG)** → rewrite/re-route → generate → **groundedness check (Self-RAG-lite)**; max 2 loops / ≤4 LLM calls; adaptive routing by complexity; expose `search_filings`, `query_financials`, `get_page` as a **FastMCP server** (P2 will mount it).
- **Concepts:** fixed pipelines can't recover from bad retrieval; agents can, at a cost in latency/determinism. Try the cheap version (reranker-score gate) before an LLM grader.
- **Gate:** multi-hop subset **+10 pts**; avg LLM calls/query ≤ 4; a Claude Code client calls the MCP tools successfully.

## Phase 12 — Production hardening + automation (MLOps Level 1) · week 11
- **Build:**
  - FastAPI (SSE streaming, `/healthz`, `/metrics`), multi-stage Docker, compose profiles `core/mlops/obs`, Streamlit UI with PDF page preview.
  - **Prefect flows:** `ingest_and_index` (new filings → validate → index new collection → eval → switch alias only if gate passes); `train_intent_classifier` (Pandera + Evidently checks → train → challenger → promote if better); `freshness` crawl of SEBI/RBI.
  - Caching: exact cache + **entity/period-keyed** semantic cache (τ ≥ 0.95; never across FY/ticker mismatch) — measure false-hit rate.
  - Guardrails: Prompt Guard 2 on queries **and retrieved chunks**, Presidio (PAN/Aadhaar), number-in-source check, disclaimer; ACL as a mandatory retrieval filter (never via prompt).
  - MLflow Prompt Registry (prompts in git → synced → loaded by alias, version on every trace).
- **Concepts:** Level 1 = pipelines automated, evals gate promotion, every answer traceable to (code SHA, prompt version, index version, model alias, classifier version).
- **Gate:** 5 scheduled ingest runs succeed unattended; planted-injection canary doc blocked; 0 ACL leaks on adversarial tests; cache false-hit rate reported.

## Phase 13 — LLMOps loop + CI/CD + monitoring (MLOps Level 2) · week 12
- **Build:**
  - GitHub Actions: `ci.yml` (lint, type, unit, contract tests with mocked LLMs, image build), `eval.yml` (promptfoo + Ragas/ranx on golden subset, PR comment with deltas, **fail if any slice drops > threshold**), `train.yml` (weekly retrain → challenger), `release.yml` (multi-arch → GHCR → Cloud Run).
  - Monitoring: Prometheus + Grafana (p50/p95, errors, tokens, **shadow cost**, cache hits, guardrail blocks, abstention rate); Evidently weekly drift on query features + classifier outputs → alert.
  - Feedback: 👍/👎 on traces → annotation queue → golden set + classifier data; mine hard negatives.
  - A/B prompts (hash user → variant, bootstrap CIs) + shadow challenger; red-team nightly (promptfoo redteam + garak).
  - Deploy stateless API to Cloud Run (budget alert, max-instances=1) with Qdrant Cloud free; demo on Render.
- **Gate:** a PR that degrades an eval slice **cannot merge**; drift alert fires on a synthetic shifted batch; public demo URL live.

## Stretch (only if a gate shows the need)
| Stretch | Why / when | Gate |
|---|---|---|
| Embedding fine-tune (sentence-transformers MNRL + Matryoshka, synthetic pairs + hard negatives, MPS/Colab) + dual-index migration | Domain vocab (NIM, GNPA, CASA); teaches index versioning | recall@10 +5 on **human** test split |
| Reranker fine-tune from 👎 traces | Production-driven data | nDCG@5 ↑ |
| LightRAG / MS GraphRAG (≤20 docs) / Neo4j for circular `amends/supersedes` chains | Global & relational questions vector RAG fails | win-rate ≥ 60% on global subset, no factoid regression |
| RAPTOR / section-summary nodes | "Overall strategy in the AR" questions | synthesis subset ↑ |
| ColQwen/ColSmol over investor-presentation pages | Charts/infographics parsers miss | chart subset ↑ |
| Rebuild the pipeline in LlamaIndex; port ingest to Dagster assets | Framework fluency | same eval numbers |

## Final deliverables
README with architecture + **cumulative ablation table** (rows = phases, cols = recall@10, nDCG@5, faithfulness, numeric acc, citation precision, p95, LLM calls/query, shadow cost) + leave-one-out ablations · ADRs (`docs/adr/`) · threat model · Grafana screenshot · 3-min demo video · `learning_log.md`.

## Interview "why" cheatsheet
Low recall@50 → parsing/chunking/metadata/hybrid/rewriting (reranker can't help) · good recall, bad precision → reranker/filters/headers · right page, wrong number → tables + SQL + numeric validation · global questions → RAPTOR/LightRAG · multi-step → decomposition → agentic loop (capped) · suspicious jump → judge self-preference, synthetic leakage, noise band.
