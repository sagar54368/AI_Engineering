# 04 — Free/Cheap Models, APIs & Indian-Finance Data (research note)

**Checked:** 2026-10-04 (all "checked" dates below are this date unless noted). Free tiers change monthly — re-verify before relying on a number.
**Context:** Apple Silicon, 16 GB unified memory, macOS, Docker. Projects: (1) RAG over Indian financial docs, (2) LangGraph multi-agent stock-research desk (paper trading only). Gemini free-tier key already available.

Legend: **[V]** = verified on official/primary page today · **[3P]** = only third-party source found · **[UNV]** = could not verify, treat as estimate.

---

## 0. TL;DR

- **Default hosted brain:** Gemini free tier (Flash / Flash-Lite 3.x family). Free, but prompts **are used to improve Google products and may be human-reviewed** → public documents only.
- **Fast fallback:** Groq (gpt-oss-20b/120b, qwen3.8-27b) — ~1K req/day/model free. OpenRouter `:free` as a third fallback (50 req/day unless you top up $10 once → 1,000/day).
- **Local:** Ollama (llama.cpp-Metal backend on 16 GB) running `qwen3.5:4b`/`qwen3.5:9b` or `granite4.2:8b` for agents; `qwen3-embedding:0.6b` or `bge-m3` for embeddings; `bge-reranker-v2-m3` / Qwen3-Reranker-0.6B via sentence-transformers for rerank.
- **Judge:** local small judge for iteration; occasional paid spot-checks with Claude Haiku 4.5 (Batch: $0.50/$2.50 per MTok) or GPT-5.x nano.
- **Big change since 2025:** **GitHub Models was retired on 2026-07-30** — do not plan around it. Cerebras free tier is now a **$5 / 30-day trial**, not a perpetual daily quota. Gemini **Pro models are not on the free tier**.
- **Data:** SEBI, RBI, NSE/BSE filings and bhavcopy are free and public; transcripts are mandated (SEBI LODR) to be on exchange + company sites within 5 working days. Screener.in ToS forbids copying/mirroring — don't scrape it.

---

## 1. Hosted LLM APIs — free tiers & cheap options

### 1.1 Google Gemini

| Model ID (current) | Free tier? | Paid price / 1M tok (in / out) | Data used to improve products? | Notes |
|---|---|---|---|---|
| `gemini-3.8-flash` | Yes [V] | $0.75 / $3.75 through 2026-12-31, then $1.50 / $7.50 [V] | Free: **Yes** · Paid: No [V] | Newest Flash, strongest for agent reasoning/tool use |
| `gemini-3.7-flash` / `gemini-3.6-flash` | Yes [V] | same as 3.8 [V] | Free: Yes | Previous gens; good fallback if 3.8 quota exhausted (quota is per model) |
| `gemini-3.5-flash` | Yes [V] | $1.50 / $9.00 [V] | Free: Yes | "Legacy" per models page; oddly pricier than 3.8 promo |
| `gemini-3.5-flash-lite` | Yes [V] | $0.30 / $2.50 [V] | Free: Yes | Bulk generation, query rewriting, chunk summaries |
| `gemini-3.1-flash-lite` | Yes [V] | $0.25 / $1.50 [V] | Free: Yes | Cheapest paid option if you upgrade |
| `gemini-3.1-pro-preview` | **No** [V] | $2 / $12 (≤200K ctx) [V] | — | Pro is paid-only (since 2026-04-01 per 3P) |
| `gemini-embedding-2` (models page lists `gemini-embedding-2-preview`) | Yes [V] | $0.20 text [V] | Free: Yes | Multimodal (text/image/PDF/audio), 8,192 input tokens, 128–3072 dims (768/1536/3072 recommended) [V] |
| `gemini-embedding-001` | not on pricing page [UNV] | — | — | Text-only, 2,048 input tokens, `task_type` param (RETRIEVAL_DOCUMENT/QUERY etc.) [V] |
| `gemini-2.5-*` | Restricted to past active users [V] | — | — | Don't start new code on 2.5; `gemini-2.0-flash` is shut down [V] |

**Free-tier rate limits (RPM / TPM / RPD):** Google **no longer publishes numbers** in docs — "Rate limits … can be viewed in Google AI Studio" (https://aistudio.google.com/rate-limit), per project, "not guaranteed" [V, docs updated 2026-09-02]. Third-party reports conflict wildly (e.g., Flash-Lite 500–1,500 RPD, 15–30 RPM; 3.5 Flash as low as 20 RPD) [3P]. **Action: open AI Studio → Rate limit page on day 1 and record your actual numbers in `config/limits.yaml`.** Plan for ~10–15 RPM and a few hundred RPD on Flash-Lite, far fewer on full Flash.

Gotchas:
- Quotas are **per project, per model** — rotating models (3.8 → 3.7 → 3.6 → Flash-Lite) multiplies effective daily budget. Don't create multiple projects to evade limits (ToS risk).
- Thinking tokens count toward output/TPM; set a thinking budget low for bulk work.
- Free tier: **no context caching, no Batch API** (paid only) [V].
- EEA/UK/Switzerland users get paid-tier data terms even on free — **India does not**, so free-tier data use applies [V].

Sources: https://ai.google.dev/gemini-api/docs/pricing · https://ai.google.dev/gemini-api/docs/rate-limits · https://ai.google.dev/gemini-api/docs/models · https://ai.google.dev/gemini-api/docs/embeddings · https://ai.google.dev/gemini-api/terms (all checked 2026-10-04)

### 1.2 Other providers

| Provider | Free / cheap offer (checked 2026-10-04) | Best use here | Gotchas | Source |
|---|---|---|---|---|
| **Groq** | Free plan per model: `openai/gpt-oss-120b`, `gpt-oss-20b`, `qwen/qwen3.8-27b` = **30 RPM, 1K RPD, 8K TPM, 200K TPD** [V]. Also Whisper (20 RPM, 2K RPD) | Fast agent fallback, LLM-as-judge (gpt-oss-120b), low-latency tool calls | 8K TPM is tight for RAG prompts with long context (~2–3 big requests/min). Limits are org-level. Small model menu | https://console.groq.com/docs/rate-limits |
| **Cerebras** | **Free Trial: $5 credits, expire after 30 days** [V]; gpt-oss-120b & qwen-3.8-27b at 5 RPM, 1M TPD | One-off burst eval runs (very fast) | No longer a perpetual free tier; after credits → must buy | https://inference-docs.cerebras.ai/support/rate-limits |
| **OpenRouter `:free`** | 20 RPM; **50 req/day** if never purchased; **1,000/day after ≥$10 lifetime purchase** [V]. ~20–28 free models incl. Gemma 4 31B, Qwen3.8 27B, NVIDIA Nemotron 3 family [3P] | Third-tier fallback; trying many open models via one OpenAI-compatible key | Free models rotate/disappear without notice; free endpoints may log prompts (provider-dependent) — check each model's data policy | https://openrouter.ai/docs/api/reference/limits · https://openrouter.ai/collections/free-models |
| **Mistral La Plateforme** ("Experiment" plan) | Free, all models, phone verification, **requires opting into data use for training**; reported ~1 RPS, 500K TPM, 1B tok/month [3P] | Extra free capacity for bulk generation (Mistral Small) | Exact limits now only in Admin Console [UNV]; training opt-in = public data only | https://docs.mistral.ai (tier page 404'd) · https://yangmao.ai/en/providers/mistral/free-api/ |
| **GitHub Models** | **Retired 2026-07-30** (closed to new customers 2026-06-16) [V] | — | Don't use. Old tutorials still reference it | https://daily.dev/posts/github-models-is-being-fully-retired-on-july-30-2026-bz8saavyw |
| **Hugging Face Inference Providers** | **$0.10/month** free credit (Free), $2/month (PRO $9) [V]; pass-through provider pricing | Occasional access to niche hosted models / hf-inference CPU embeddings & rerankers | $0.10 is ~nothing; good mainly for testing | https://huggingface.co/docs/inference-providers/pricing |
| **Together AI** | Conflicting: official docs reportedly "no free trial, $5 minimum purchase"; some report $1–5 signup credit [3P/UNV] | Cheap open-model inference if you pay | Treat as paid | https://www.eesel.ai/blog/together-ai-pricing |
| **Fireworks AI** | ~$1 starter credit, no card [3P] | Same | Tiny | https://pricepertoken.com/endpoints/fireworks/free |
| **Cohere (trial key)** | Free trial: Chat 20 RPM, Embed 2,000 inputs/min, **Rerank 10 RPM**, **1,000 API calls/month total** [V]. Rerank models: `rerank-v4.0-pro`, `rerank-v4.0-fast`, `rerank-v3.5` (multilingual) [V] | **Rerank** quality benchmark vs local reranker (eval only) | 1K calls/month across all endpoints; trial keys not for production (per Cohere docs; commercial-use clause not confirmed on page) | https://docs.cohere.com/docs/rate-limits · https://docs.cohere.com/docs/rerank |
| **DeepSeek** | No free tier. `DeepSeek-Flash`: $0.15–0.30 in (miss) / $0.6–1.2 out; `DeepSeek-V4-Pro`: $0.66–1.32 / $1.98–3.96; cache hits ~$0.003–0.04 [V]. **Off-peak = half price**; peak 01:00–04:00 & 06:00–10:00 UTC Mon–Fri (= 06:30–09:30 & 11:30–15:30 IST) [V] | Cheapest paid capable model for bulk synthetic Q&A generation; 1M context | Data processed in China — public docs only. Range = peak vs off-peak | https://api-docs.deepseek.com/quick_start/pricing |
| **Anthropic** | Small free credit on signup [V]. **Claude Haiku 4.5: $1 / $5 per MTok; Batch $0.50 / $2.50**; cache read $0.10 [V] | Occasional high-quality **LLM-as-judge** spot-checks (batch, 50% off) | Not free; Sonnet 5/5.5 at $2/$10 for gold-label judging of a small set | https://platform.claude.com/docs/en/about-claude/pricing |
| **OpenAI** | GPT-5 nano $0.05/$0.40; GPT-5.4 nano $0.20/$1.25; GPT-5.4 mini $0.75/$4.50; Batch −50% [3P — openai.com returned 403] | Second-opinion judge (cross-family judging reduces self-preference bias) | Verify on https://openai.com/api/pricing before spending | https://benchlm.ai/openai/api-pricing |

---

## 2. Local models on 16 GB Apple Silicon

### 2.1 Memory budget reality

| Item | Value |
|---|---|
| Total unified memory | 16 GB |
| macOS + browser + IDE + Docker (Qdrant/Postgres) | ~6–8 GB |
| Default GPU-wireable limit | ~2/3–3/4 of RAM (≈10.5–12 GB); raisable with `sudo sysctl iogpu.wired_limit_mb=12288` (resets on reboot) [UNV on newest macOS] |
| **Practical model budget** | **≈ 6–8 GB total** for weights + KV cache across everything loaded |
| KV cache | grows with `num_ctx`; set 8K–16K, not 128K/256K. 256K context on a 9B model will OOM/swap |

### 2.2 Runtime choice

| Runtime | Pros | Cons on 16 GB Apple Silicon | Verdict |
|---|---|---|---|
| **Ollama** (v0.30+) | One-line install, model registry, OpenAI-compatible `/v1`, native LiteLLM support (`ollama_chat/…`), tool calling, embeddings API, keeps models warm, Docker-friendly via `host.docker.internal:11434` | **MLX engine only activates on ≥32 GB Macs**; 16 GB silently uses llama.cpp Metal [3P, two sources] → no MLX speedup for you. No rerank endpoint [UNV] | **Primary.** Simplicity + ecosystem beats raw speed for a learner |
| **MLX-LM** (`mlx_lm.server`) | Fastest on Apple Silicon (typically 20–50% higher decode than llama.cpp at same bits) [3P], OpenAI-compatible server, easy LoRA fine-tuning | Python env management, fewer turnkey tool-calling templates, one model per server | **Secondary**: use for speed experiments & LoRA fine-tuning a small judge |
| **llama.cpp** (`llama-server`) | Max control (quant choice, `--reranking` endpoint, grammar/JSON-schema constrained output), GGUF everywhere | Manual model downloads/flags | Use for **local rerank server** or constrained JSON if Ollama falls short |
| **LM Studio** | GUI, MLX + GGUF engines, model browser, OpenAI server | GUI app, less scriptable/reproducible; proprietary app | Good for *exploring* models; don't make it a project dependency |

### 2.3 Recommended local models (tags verified on ollama.com/library 2026-10-04)

Speed estimates are **[UNV] back-of-envelope**: base Apple Silicon ≈ 150 GB/s memory bandwidth → decode tok/s ≈ 0.6–0.7 × bandwidth / weight-size. Published "230 tok/s for 8B on Apple Silicon" figures are for Pro/Max chips — ignore for base Apple Silicon.

| Role | Model (Ollama tag) | Disk / RAM (Q4) | Est. decode tok/s (base Apple Silicon) | Why |
|---|---|---|---|---|
| Agent / chat (main) | `qwen3.5:9b` | 6.6 GB / ~7.5–8.5 GB w/ 8K ctx | ~13–18 | Tools + thinking + vision, 256K ctx (use 8–16K). Best capability that fits [V tags] |
| Agent (light, run alongside embedder) | `qwen3.5:4b` | 3.4 GB / ~4.5 GB | ~25–35 | Tool calling, fast enough for multi-agent loops |
| Agent alt (strong tool-use, Apache-2.0) | `granite4.2:8b` | 5.3 GB | ~17–22 | RAG/function-calling focus, thinking toggle, 128K ctx [V] |
| Agent alt | `gemma4:12b` | 7.7–8.0 GB | ~10–14 | Good tools/vision; **tight** on 16 GB — only with nothing else loaded |
| Tiny router / classifier | `qwen3.5:0.8b` or `granite4.2:3b` | 1.2 / 2.2 GB | 60–100+ / ~40 | Intent routing, query classification, guardrails |
| Small judge | `qwen3.5:9b` (thinking on) or `granite4.2:8b`; MLX alt: `mlx-community/Qwen3.5-9B-4bit` | as above | as above | Use **pairwise/rubric binary** judgments; calibrate against ~50 human labels and a Haiku spot-check |
| Embeddings (multilingual, Hindi/English mix) | `bge-m3` (567M) | ~1.2 GB | ~100s of chunks/s | Dense+sparse+multi-vector, 8K input, proven for RAG [V tag] |
| Embeddings (best quality/size) | `qwen3-embedding:0.6b` (HF: `Qwen/Qwen3-Embedding-0.6B`) | ~1–1.5 GB | fast | MTEB-eng-v2 ~70.7, 32K ctx, 32–1024 dims (MRL), Apache-2.0 [3P] |
| Embeddings (lightweight) | `nomic-embed-text` | ~0.3 GB | very fast | Baseline/ablation; needs `search_document:`/`search_query:` prefixes |
| Reranker | `BAAI/bge-reranker-v2-m3` (sentence-transformers `CrossEncoder`, device `mps`) | ~1.1 GB fp16 | ~20–50 pairs/s [UNV] | Multilingual, reliable; rerank top-30 → top-5 |
| Reranker alt | `Qwen/Qwen3-Reranker-0.6B` (HF; community Ollama builds exist e.g. `dengcao/Qwen3-Reranker-0.6B`) | ~1.2 GB | similar | Stronger on benchmarks; LLM-style reranker needs specific prompt format |
| PDF parsing / OCR | **Docling** (+ `granite-docling-258M`, MLX build; `docling-mlx` accel) | <1 GB | ~1–3 s/page [UNV] | Layout + **table structure** (FinTabNet TEDS 0.97 reported) — critical for financial statements |
| OCR for scanned PDFs | `deepseek-ocr` (3B, Ollama) or `qwen3.5:4b` vision | ~3 GB | slow (sec/page) | Only for scanned docs; most BSE/NSE filings are digital-text PDFs — try `pymupdf` text first |

### 2.4 What NOT to try on 16 GB

- NO: 27B+ models (`qwen3.5:27b` 17 GB, `gemma4:26b/31b`, `qwen3.8:27b`) — will swap/crash. Use them via Groq/OpenRouter instead.
- NO: 35B-A3B MoE "it's only 3B active" — **all weights still need RAM** (22–24 GB).
- NO: Ollama `*-mlx` tags expecting speedups — MLX engine gated to ≥32 GB [3P]; `qwen3.5:9b-mlx` is 8.9 GB vs 6.6 GB GGUF anyway.
- NO: Loading chat 9B + judge 9B + Docling simultaneously. Run pipelines **sequentially**; set `OLLAMA_MAX_LOADED_MODELS=1` (or 2 with a 4B) and `OLLAMA_KEEP_ALIVE=5m`.
- NO: Full 256K context windows; NO: fine-tuning > 4B (LoRA on ≤4B with MLX is OK); NO: Q2/Q3 quants of big models instead of Q4 of a smaller one (quality cliff).
- NO: Running Ollama **inside Docker** on macOS — no Metal GPU in containers. Run Ollama natively; containers call `host.docker.internal`.

---

## 3. Embeddings & rerank — free options compared

| Option | Free allowance (checked 2026-10-04) | Quality / notes | Use |
|---|---|---|---|
| **Local `bge-m3` / `qwen3-embedding:0.6b`** | Unlimited | Near-SOTA for size; zero data leakage; reproducible | **Primary** for both projects |
| **Gemini `gemini-embedding-2`** | Free tier (limits only in AI Studio) [V]; paid $0.20/M | Strong, multimodal (can embed PDF pages/images) | Ablation vs local; **public docs only** (free tier data use) |
| **Voyage AI** | **200M free tokens per model** (50M for `voyage-finance-2`, `voyage-law-2`) [3P]; `rerank-3`/`rerank-3-lite` (released 2026-09-30) also 200M free [3P] | `voyage-finance-2` is domain-tuned — good ablation for finance RAG | Best free *hosted* embedding+rerank budget; verify at voyageai.com/pricing [UNV official] |
| **Jina AI** | **10M tokens one-time** shared across Embeddings/Reranker/Reader; free key 100 RPM / 100K TPM [3P] | jina-embeddings & jina-reranker; Reader API handy for HTML→markdown | Small; one-time. Open weights are often CC-BY-NC — fine for learning, not commercial |
| **Cohere trial** | 1,000 calls/month, rerank 10 RPM [V] | `rerank-v4.0-fast/pro` strong multilingual | Eval-only "gold" reranker to compare against local bge-reranker |
| **HF Inference (hf-inference CPU)** | $0.10/month credit [V] | Embedding/rerank endpoints | Negligible |

**Recommendation:** index with `bge-m3` (or Qwen3-Embedding-0.6B, 1024-d) locally; rerank with `bge-reranker-v2-m3` locally; run a one-time ablation table (local vs Gemini-embedding-2 vs voyage-finance-2; local reranker vs Cohere rerank-v4.0-fast) on a 50–100 question eval set. That ablation is a strong portfolio artifact.

---

## 4. Model routing (implement via LiteLLM Router)

| Task | Primary (free) | Fallback 1 | Fallback 2 / paid escape hatch | Why |
|---|---|---|---|---|
| Bulk generation (chunk summaries, synthetic Q&A, metadata extraction) | `gemini/gemini-3.5-flash-lite` | `ollama_chat/qwen3.5:4b` | `deepseek/…flash` off-peak (paid, cents) | High volume, low stakes; local fallback never rate-limits |
| Query rewrite / HyDE / routing classifier | `ollama_chat/qwen3.5:0.8b`–`4b` | `gemini/gemini-3.5-flash-lite` | `groq/openai/gpt-oss-20b` | Latency-sensitive, tiny prompts; keep off cloud quota |
| RAG answer synthesis (with citations) | `gemini/gemini-3.8-flash` | `groq/openai/gpt-oss-120b` | `ollama_chat/qwen3.5:9b` | Needs best reasoning + long context; Groq's 8K TPM limits big contexts → keep retrieved context ≤6K tokens for fallback |
| Agent reasoning / tool calling (LangGraph supervisor, analyst agents) | `gemini/gemini-3.8-flash` | `groq/qwen/qwen3.8-27b` | `ollama_chat/qwen3.5:9b` or `granite4.2:8b` | Tool-call reliability matters most; 3 families → resilient |
| Agent workers (news summarizer, ratio calculator narrator) | `ollama_chat/qwen3.5:4b` | `gemini/gemini-3.5-flash-lite` | `openrouter/…:free` | Many calls per run; save cloud quota for supervisor |
| LLM-as-judge (iteration loop) | `groq/openai/gpt-oss-120b` | `ollama_chat/qwen3.5:9b` | — | Different family from generator (Gemini) → less self-preference bias |
| LLM-as-judge (final reported numbers) | `anthropic/claude-haiku-4-5` via **Batch** (paid, ~$1–3 per full eval) | GPT-5.x nano/mini | Sonnet 5 on 30-item gold subset | Credible, reproducible, cheap; pin model ID + prompt hash |
| Embeddings | local `bge-m3` / `qwen3-embedding:0.6b` | `gemini/gemini-embedding-2` | Voyage free tokens | Never mix embedding models within one index; store `embed_model` in collection metadata |
| Rerank | local `bge-reranker-v2-m3` | Cohere `rerank-v4.0-fast` (trial) | Voyage `rerank-3-lite` | Local has no quota; cloud for ablations |
| PDF → structured text/tables | Docling (local) | `pymupdf4llm` | `gemini-3.8-flash` vision on hard pages (public PDFs only) | Tables in results PDFs are the hard part |

LiteLLM config notes:
- Use `router_settings: { routing_strategy: simple-shuffle, num_retries: 2, allowed_fails: 3, cooldown_time: 60 }` and per-deployment `rpm`/`tpm` matching each free tier so LiteLLM pre-emptively routes away before 429s.
- Define `fallbacks: [{"rag-synth": ["rag-synth-groq", "rag-synth-local"]}]` using model-group aliases; code only references aliases.
- Add `context_window_fallbacks` (Groq 8K TPM) and log every call (model, tokens, latency, cost) to SQLite/Langfuse for the README's cost table.
- Set `max_budget` for paid providers; keep keys in `.env` (gitignored) — never in config committed to Git.

---

## 5. Free, legal Indian-finance data sources

| Source | What | Access | ToS / etiquette | Verified |
|---|---|---|---|---|
| **SEBI circulars** | ~2,800 circulars; HTML page per circular with PDF link | Listing: `https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0`; detail: `/legal/circulars/<mon-yyyy>/<slug>_<id>.html` | Government public docs; throttle 1 req / 2–3 s; cache locally | [V] latest: 2026-10-01 "Project Jagrook" |
| Also SEBI master circulars, regulations, board meeting minutes, consultation papers | Better for Q&A grounding than individual circulars | Same site, other `ssid` values | same | [V site] |
| **RBI monetary policy** | Policy statements, MPC resolutions, **MPC minutes** (released 14 days after meeting), Governor's statements, press-conference transcripts | `https://www.rbi.org.in/scripts/annualpolicy.aspx`; PDFs on `rbidocs.rbi.org.in/rdocs/PressRelease/PDFs/…` | Public; polite crawling | [V] Aug 2026 MPC minutes present |
| RBI press releases, DBIE (macro time series) | Rates, CPI, credit data | `rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx`; `data.rbi.org.in` (DBIE) | Public | [V] PR page; DBIE [UNV] |
| **Company annual reports & quarterly results** | Reg. 30/33/34 filings: results PDFs, annual reports, investor presentations | BSE: corporate announcements (JSON endpoint `api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w`, unofficial lib `pip install bse`); NSE: corporate-filings pages; **also company IR pages** (most stable URLs) | Exchanges have no official free API; unofficial endpoints need browser-like headers + cookies, may change/break. NSE robots.txt: `Allow: /` (only `/market-data-test` disallowed) [V]. ≤1 req/s, cache, no parallel hammering | [V] robots; endpoints [3P] |
| **Earnings-call transcripts** | Mandatory: SEBI LODR Reg. 46 — transcripts on company website **and** exchanges within **5 working days**; audio within 24 h; retained 5 years | BSE/NSE announcements (category "Analysts/Institutional Investor Meet/Con. Call Updates") + company IR pages | Public disclosure; redistribution of full text in a public repo — store locally, commit only manifest/URLs | [V] (SEBI/NSE guidance, 3P summaries) |
| **NSE bhavcopy (EOD prices)** | Daily OHLC, volume, delivery for all CM securities | UDiFF format (old format discontinued 2024-07-08): `https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip` | Public; one file/day — trivial load. Use `User-Agent` header | [3P] URL pattern — test on day 1 |
| BSE bhavcopy | EOD equity | bseindia.com "Bhavcopy" downloads | same | [UNV] exact URL |
| **yfinance** (`RELIANCE.NS`, `^NSEI`) | Prices, some fundamentals | `pip install yfinance` (≥1.0) | **Unofficial** Yahoo scraping; Yahoo ToS = personal use. Frequent `YFRateLimitError` (2026 issues open); fundamentals for .NS often incomplete/wrong currency/stale; corporate actions may be missing. Cache to parquet, ≤1 req/2 s, batch `yf.download` | [V] GitHub issues |
| **News RSS** | Moneycontrol (`moneycontrol.com/rss/latestnews.xml`, more at `/india/newsarticle/rssfeeds/rssfeeds.php`), ET (`economictimes.indiatimes.com/rss.cms`), Livemint (`livemint.com/rss`) | `feedparser` | RSS = headlines + snippets for personal use; **don't scrape full articles** behind paywalls; attribute + link; poll ≤ every 15 min | [UNV] — ET/Mint pages blocked my fetcher; Moneycontrol [3P] |
| **Screener.in** | Fundamentals, 10-yr financials | Web UI; export to Excel for logged-in users | ToS: may not "modify or copy the materials", "use for any commercial purpose", or "mirror" [V]. **Don't scrape.** Use manual export for personal analysis only, don't commit data | [V] |
| **data.gov.in (OGD)** | 130K+ APIs (CPI, IIP, trade, etc.) | Free API key (My Account → API Access); `pip install datagovindia` | Gov open data licence (GODL); ~1,000 req/day reported [3P] | [3P] |
| **FRED** | 800K+ US/global macro series (Fed funds, UST yields, DXY, crude, INR/USD `DEXINUS`) | Free API key (fredaccount.stlouisfed.org); `fredapi` | ~120 req/min [3P]; attribute FRED | [3P] |
| NSE indices / F&O / FII-DII data | Index constituents, participant OI | NSE "All Reports" page `nseindia.com/all-reports` | Same NSE etiquette | [3P] |

### 5.1 Recommended starter corpus (~250–350 PDFs, ~1–2 GB)

| Bucket | Content | Count | Notes |
|---|---|---|---|
| Companies (5, sector-diverse) | `HDFCBANK` (bank), `INFY` (IT), `RELIANCE` (conglomerate), `ITC` (FMCG), `MARUTI` (auto) | 5 | Avoid recently demerged names (e.g., Tata Motors split) for clean time series |
| Annual reports | FY2025-26 (published ~Jun–Aug 2026); optionally FY2024-25 for YoY questions | 5–10 | 250–450 pages each — biggest parsing cost; start with MD&A + financial statements |
| Quarterly results + investor presentations | Q2 FY26 → Q1 FY27 (4 quarters; Q2 FY27 results arrive mid-Oct 2026 — add as "fresh data" test) | 40 | Results PDFs contain the tables your RAG must cite |
| Earnings-call transcripts | Same 4 quarters | 20 | Great for "what did management say about margins" questions |
| SEBI circulars | 50 most recent (or 50 on one theme: F&O/derivatives, MF, LODR) | 50 | Theme-focused set → better eval questions |
| SEBI master circulars | LODR, Mutual Funds, Stock Brokers | 3–5 | Long, structured docs — good chunking test |
| RBI | Last 8 policy statements + 8 MPC minutes + 4 Monetary Policy Reports | ~20 | Macro context for the agent desk |
| Prices | NSE bhavcopy last 3 years (~750 files) **or** yfinance daily for 5 tickers + `^NSEI` | — | For the paper-trading desk |
| Macro | FRED: `DEXINUS`, `DGS10`, `DCOILBRENTEU`, `FEDFUNDS`; RBI repo history | — | |

### 5.2 Reproducible download recipe

1. `data/manifest.yaml` — one entry per doc: `{id, source, url, company, doc_type, period, published_date}`. **Commit the manifest, not the PDFs** (copyright + repo size).
2. `scripts/fetch.py` — `httpx` with `User-Agent: "<project>/0.1 (contact: your-email)"`, `tenacity` retry with exponential backoff, **≥2 s sleep between requests**, skip if file exists, write `sha256`, size, fetched_at to `data/manifest.lock.json`.
3. For NSE: first GET `https://www.nseindia.com` to obtain cookies, then API calls with the same session; expect occasional 401/403 — back off, don't parallelize.
4. Store raw under `data/raw/<source>/<company>/<period>/<doc_type>.pdf`; parsed output (Docling JSON/markdown) under `data/parsed/…` keyed by sha256 → re-parsing is cached.
5. `make data` = fetch + verify hashes + parse; anyone cloning the repo regenerates the identical corpus (minus URLs that rot — note the fetch date in README).
6. Respect `robots.txt` (check with `urllib.robotparser` in the fetcher), never bypass logins/paywalls/CAPTCHAs.

---

## 6. Privacy & compliance notes

- **Gemini free tier (India):** Google "uses the content you submit … and any generated responses to provide, improve, and develop Google products"; "Human reviewers may read, annotate, and process your API input and output"; "**Do not submit sensitive, confidential, or personal information to the Unpaid Services.**" [V — https://ai.google.dev/gemini-api/terms]. EEA/UK/CH carve-out does **not** apply to India.
- **Implications:**
  - Fine: public SEBI/RBI/exchange filings, public news headlines, synthetic portfolios.
  - **Not fine:** client portfolios, KYC/PII, employer-internal research, broker statements, anything under NDA (e.g., from a job). Same rule for Mistral free (training opt-in), OpenRouter free endpoints, DeepSeek (China-hosted), and likely Groq/Cerebras free (check each).
  - For private data: route to **local Ollama only** (enforce in LiteLLM with a `sensitivity=private` tag → local-only model group) or a paid tier with no-training terms (Gemini paid, Anthropic, OpenAI API default).
  - Mention this data-classification routing in the README — it's an interview talking point.
- **Paper trading only:** the agent desk must never call a broker order API; label outputs "educational, not investment advice" (SEBI regulates investment advice/research analysts — public stock recommendations without RA registration are a regulatory risk).
- **Secrets:** `.env` + `.gitignore`; add `gitleaks`/`detect-secrets` pre-commit hook.

---

## 7. Couldn't verify / re-check list

| Item | Status |
|---|---|
| Gemini free-tier RPM/TPM/RPD per model | Not published; only in AI Studio (login). Third-party numbers conflict |
| Gemini embedding model ID (`gemini-embedding-2` vs `-2-preview`) and free limits | Docs pages disagree on the suffix; test both |
| Mistral Experiment plan exact limits | Official tier page 404; numbers from third parties |
| Together / Fireworks signup credits | Conflicting third-party reports |
| OpenAI mini/nano prices | openai.com blocked (403); third-party only |
| Voyage free-token amounts, rerank-3 | Third-party + Voyage blog title; verify at voyageai.com |
| Ollama MLX ≥32 GB gate on v0.30+ | Two third-party sources agree; not in official docs I could reach |
| ET / Livemint RSS URLs and terms | Sites blocked fetcher |
| Local tok/s numbers | Bandwidth-based estimates; benchmark yourself with `ollama run --verbose` |
| Groq/Cerebras data-retention on free tier | Not checked today |

## 8. Day-1 checklist

1. `brew install ollama` → `ollama pull qwen3.5:4b qwen3.5:9b bge-m3 qwen3-embedding:0.6b` → `ollama run qwen3.5:4b --verbose` (record tok/s).
2. Open AI Studio rate-limit page; write actual Gemini limits into `config/limits.yaml`.
3. Create Groq + OpenRouter keys (consider one-time $10 OpenRouter top-up → 1K free req/day).
4. `pip install litellm sentence-transformers docling yfinance feedparser httpx tenacity`; test `CrossEncoder("BAAI/bge-reranker-v2-m3", device="mps")`.
5. Download 1 annual report + 1 results PDF + 1 transcript for one company; run Docling; inspect tables before scaling to the full corpus.
