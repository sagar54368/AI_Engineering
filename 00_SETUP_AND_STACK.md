# 00 — Setup, Model Routing & Stack Decisions

## 1. Day-1 setup (≈2 h)
```bash
brew install ollama just gitleaks node        # node only for promptfoo (npx)
ollama serve &                                # run NATIVELY — Docker on macOS has no Metal GPU
ollama pull qwen3.5:4b bge-m3 qwen3-embedding:0.6b   # 16 GB / fanless-safe: no 9B by default — heavy work goes to cloud APIs
ollama run qwen3.5:4b --verbose               # record real tok/s in docs/benchmarks.md
uv init niveshrag && cd niveshrag && uv python pin 3.12   # system 3.14 is too new for torch/docling wheels
```
- `.env` (gitignored): `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `LANGSMITH_API_KEY`. Commit `.env.example` with names only.
- Open **AI Studio → Rate limits** and **build.nvidia.com** account page; write the real per-model limits into `config/limits.yaml` (Google no longer publishes free-tier numbers).
- Docker Desktop → Resources → Memory **6 GB**. `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=5m`, `OLLAMA_NUM_PARALLEL=1`, `num_ctx` 8192.
- **On a fanless or 16 GB laptop:** no damage risk (macOS throttles/shuts down before harm), but: sustained local generation throttles and heats it, and memory overcommit → SSD swap wear. Rules: local = only embeddings, reranker, ≤4B fallback, small classifiers; bulk generation / agents / judging → cloud aliases; keep Activity Monitor memory pressure green/yellow; long jobs plugged in, on a hard surface, in batches; Optimized Battery Charging on; optional `macmon` to watch temps/power. Any `qwen3.5:9b` mention below = optional, Docker stopped, short runs only.

## 2. Model routing (LiteLLM aliases — code only ever calls the alias)
| Alias | Primary (free) | Fallback 1 | Fallback 2 | Why this split |
|---|---|---|---|---|
| `bulk` (chunk context, synthetic Q&A, metadata) | Gemini Flash-Lite | Ollama `qwen3.5:4b` | NVIDIA `gpt-oss-20b` | High volume, low stakes; local never rate-limits |
| `router` (classify/rewrite/HyDE) | your trained intent classifier / Ollama `qwen3.5:0.8b` | Gemini Flash-Lite | — | Latency-sensitive; keep off cloud quota |
| `answer` (RAG synthesis w/ citations) | Gemini Flash (latest) | NVIDIA `nemotron-3-super-120b` | Groq `gpt-oss-120b` → Ollama `qwen3.5:9b` | Best quality + long context; 3 families = resilient |
| `agent` (supervisor / analysts, tool calling) | Gemini Flash | NVIDIA `nemotron-3.5-lightning-30b` | Groq `qwen3.8-27b` → Ollama `granite4.2:8b` | Tool-call reliability first |
| `worker` (many cheap calls) | Ollama `qwen3.5:4b` | Groq `gpt-oss-20b` | OpenRouter `:free` | Save cloud quota for supervisor |
| `judge` (eval loop) | NVIDIA `nemotron-3-ultra-550b` | Groq `gpt-oss-120b` | Ollama `qwen3.5:9b` | **Different family from generator (Gemini)** → less self-preference bias |
| `judge_final` (numbers you publish) | optional paid: Claude Haiku Batch (~$1–3/full eval) | — | — | Credible, pinned, cheap |
| `vision` (bad PDF pages, charts) | Gemini Flash | NVIDIA `llama-3.2-11b-vision` | — | Only for low-confidence pages |
| `local_only` (anything private) | Ollama only | — | — | Data-classification routing (interview point) |
| Embeddings | local `bge-m3` / `qwen3-embedding:0.6b` | Gemini embedding (ablation) | — | Unlimited, reproducible; never mix models in one index |
| Rerank | local `bge-reranker-v2-m3` (sentence-transformers, `mps`) | Cohere trial (ablation only) | — | No quota locally |

Notes: set per-deployment `rpm/tpm` = real free limits so LiteLLM routes *before* 429s. Never rotate multiple keys/projects of one provider to multiply free limits (ToS risk). **Pin LiteLLM version + image digest** (malicious PyPI releases 1.82.7/1.82.8, Mar 2026). Verify exact model IDs on day 1 (`research/04` §1).

## 3. Stack decisions (pick · why · why not the alternative · when both)
| Layer | Pick | Why | Not this, because | Use both when |
|---|---|---|---|---|
| Env | uv + pyproject + `uv.lock` | Fast, one tool, reproducible | Poetry slower; conda only for non-Python binaries | Employer repo uses Poetry |
| Quality | ruff · mypy(strict src) · pytest+hypothesis · pre-commit · gitleaks | Catch bugs/leaks before commit | flake8/black/isort = 3 tools ruff replaces | pyright in editor + mypy in CI |
| Config | pydantic-settings (`SecretStr`) | Typed, fails fast, keys never logged | raw `os.getenv` silently returns None | — |
| Data versioning | DVC (local remote) | Hash pointers in git, stage caching (`dvc.yaml`) | lakeFS = team/server; git-lfs no pipelines | Push frozen eval set to HF Hub |
| Validation | Pydantic (records) + Pandera (DataFrames) | Validate at every boundary | Great Expectations too heavy solo | — |
| Parsing | Docling (+PyMuPDF fast path) | Tables + page provenance for citations | Unstructured heavy; Marker/MinerU AGPL/GPL & heavy; LlamaParse cloud | LlamaParse free tier as gold reference |
| Vectors | **Qdrant** (Docker) | Native hybrid sparse+dense, filter-aware HNSW, multivector, quantization | Chroma weak hybrid/filters; FAISS not a DB; pgvector manual hybrid | FAISS notebook to learn index internals; pgvector as one ablation |
| App DB / tables | Postgres (app, MLflow, LiteLLM, LangGraph checkpoints) + DuckDB (financial tables, text-to-SQL) | One relational store for services; DuckDB embedded + fast analytics | SQLite can't serve multi-service concurrency well | SQLite in tests |
| Frameworks | Plain Python first → LangChain v1 (RAG pieces) → LangGraph (loops/agents) | Learn mechanics, then industry APIs | LlamaIndex equally good for RAG — don't stack frameworks | Rebuild one P1 pipeline in LlamaIndex as an exercise |
| Gateway | LiteLLM proxy (+Redis) | Fallbacks on 429, budgets, caching, cost per tag | Portkey OSS fine but less common; per-SDK calls = vendor lock | — |
| Tracking/registry | **MLflow 3** (Postgres backend) | Classic runs + registry aliases + prompt registry + GenAI tracing/judges in one | W&B is SaaS; Neptune/Comet niche | W&B for one fine-tune sweep |
| LLM tracing UI | **Phoenix** (P1 retrieval debugging) · **LangSmith free** (P2, Studio) | Phoenix 1 container, OTel; LangSmith best for LangGraph | Langfuse self-host needs ~16 GB (ClickHouse) | Instrument once with OTel → export to MLflow + Phoenix/LangSmith; Langfuse Cloud Hobby optional |
| Orchestration | Prefect 3 | Plain Python, retries, schedules, light | Airflow 3 heavy (2–4 GB); Dagster steeper | Port one flow to Dagster assets (stretch) |
| Serving | FastAPI + Docker + compose **profiles** | Industry default, testable, SSE | Ray Serve RAM-heavy; BentoML optional | BentoML for the classifier (optional) |
| Local LLM | Ollama native + MLX-LM (LoRA) | Simple + Apple-native speed/fine-tune | vLLM is CUDA-first (vllm-metal is young) | Benchmark Ollama vs MLX once |
| Eval | ranx (retrieval) · Ragas (RAG) · code numeric checks · promptfoo (CI + red-team) · MLflow judges | Deterministic where possible, LLM-judge only where needed | DeepEval/TruLens overlap — pick one judge lib | DeepEval if you prefer pytest-native |
| Guardrails | Prompt Guard 2 (22M) · Presidio (+PAN/Aadhaar) · Pydantic outputs | Cheap, local, defence in depth | NeMo Guardrails extra LLM calls | NeMo for topical rails (stretch) |
| Red-team | promptfoo redteam + garak | Attack generation + vuln scanner | PyRIT heavier | Both (nightly) |
| Monitoring | Prometheus + Grafana + Evidently (+Pushgateway) | System + drift standard trio | NannyML only for delayed labels | NannyML for P2 regime model |
| CI/CD | GitHub Actions (public repo = unlimited) + GHCR | Free, standard | GitLab 400 min | — |
| Deploy | Local compose (main) · Cloud Run (stateless API) · Render free (demo) | Free and realistic | Fly.io no free tier; HF Docker Spaces need PRO | Qdrant Cloud free / Neon pgvector for cloud API |
| Agents (P2) | LangGraph 1.x + LangChain `create_agent` + middleware | State, checkpoints, HITL, Studio | CrewAI less control; AutoGen in maintenance | PydanticAI/ADK rebuild of one pillar (stretch) |
| Automations (P2) | Prefect (data) + n8n self-host (glue, Slack approve buttons) | Code for logic, no-code for glue | Zapier free tiny; Temporal overkill | cron as fallback |

## 3b. Free remote compute (offload the fanless Air) — $0 only
| Option | Free allowance (checked 2026-10-04) | Use it for | Never |
|---|---|---|---|
| **Oracle Always Free — Ampere A1 VM** | **2 OCPU + 12 GB RAM total** (cut from 4/24 on 2026-06-15), 200 GB block storage, 10 TB egress/mo, 2 AMD micro VMs (1 GB each); ARM64 | Always-on services box: Qdrant, Postgres, MLflow, Prefect, Phoenix, n8n, Grafana via docker compose → Air runs only the app + embeddings/reranker | Any shape/resource not labelled **"Always Free-eligible"**; volumes outside the home region (billed); upgrading to Pay-As-You-Go |
| Oracle Autonomous AI Database ×2 | 1 OCPU, 20 GB each | Optional ablation: Oracle AI Vector Search vs Qdrant | — |
| Oracle Object Storage | 20 GB (Always Free account) | DVC remote / MLflow artifacts / backups | Exceeding 20 GB |
| **Kaggle** | 30 GPU-h/week (T4/P100), 9 h sessions | Embedding/reranker fine-tunes, bulk corpus embedding, MLX-impossible jobs | Private data |
| **Google Colab free** | T4, ~15–30 h/week (not guaranteed), ≤12 h sessions | Same as Kaggle, ad-hoc | Long unattended jobs |
| GitHub Actions (public repo) | Unlimited standard minutes | CI, eval gates, scheduled light jobs | macOS runners (10× cost) |

**Oracle billing safety (do these before creating anything):**
1. Billing & Cost Management → confirm account type is **Free Tier** (not *Pay As You Go*). Free Tier accounts are not billed; PAYG bills anything above Always Free limits. **Do not click Upgrade.**
2. Create only resources showing the **Always Free-eligible** tag, in your **home region**.
3. Add a **compartment quota policy** that sets every non-free service to 0 (hard block, unlike budgets which only alert), plus a ₹1 **budget alert** as a tripwire.
4. Idle reclaim: A1 VMs with p95 CPU < 20% **and** network < 20% **and** memory < 20% over 7 days can be reclaimed — keep backups (DVC/MLflow artifacts in Object Storage), don't treat it as the only copy.
5. All infra changes on Oracle are done by you (or with your explicit go-ahead per action); no credentials ever go in the repo.

## 4. RAM budget (16 GB) — run with compose profiles
| Group | Services | RAM |
|---|---|---|
| host | macOS + IDE + browser | 4.5–5.5 GB |
| host | Ollama 4B model + embedder (8B model = +3 GB) | ~3.5 GB |
| `core` | Postgres, Redis, LiteLLM, Qdrant, API (reranker + Prompt Guard on CPU) | 2.5–4 GB |
| `mlops` | MLflow, Prefect | 0.6–1 GB |
| `obs` | Phoenix, Prometheus, Grafana | 0.9–1.7 GB |
**Never together:** Langfuse/Opik self-host, Airflow, Ray, 8B model + training job. Before training: stop `obs` + `ollama stop <model>`.
