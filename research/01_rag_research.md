# RAG Mastery Research Note: Technique Map L0 → L5 (Indian Financial Docs)

> Compiled 2026-10-04. Constraints: Apple Silicon / 16 GB / macOS / Docker, **free tools only**, LLMs = Gemini free tier, Groq, OpenRouter free, Ollama/MLX local.
> Free-tier limits change often (Gemini RPD has moved several times in 2025–26). Check the provider consoles before you rely on any number below.
> Format for each technique: **What / Why (failure it fixes) / Intuition / Skip when / Tool pick (vs alternatives)**.

---

## 0. TL;DR stack (laptop, $0)

| Layer | Default pick | Swap-in / when |
|---|---|---|
| Parse | **Docling** (layout + TableFormer + HybridChunker, MIT, runs on MPS/CPU) | **PyMuPDF** for fast text-only; **MinerU/Marker** for hard tables & scanned docs; **LlamaParse** free tier (10k credits/mo) for a gold-standard comparison |
| Chunk | Docling `HybridChunker` (structure + token aware, repeats table headers) + **contextual chunk headers** | Parent-child for long narrative sections |
| Embed | **bge-m3** (dense+sparse+multi-vector, 8k ctx, MIT) or **Qwen3-Embedding-0.6B** locally; **gemini-embedding-001** (MRL 3072/1536/768) via API | EmbeddingGemma-300M for the smallest footprint; nomic-embed-text-v1.5 through Ollama |
| Store | **Qdrant** (Docker; dense + sparse + payload filters + multivector) | **LanceDB** embedded (zero server, built-in FTS); pgvector if you also want SQL tables; FAISS for experiments |
| Lexical | **bm25s** (or Qdrant sparse / bge-m3 sparse) | SPLADE when you need learned sparse retrieval |
| Fuse | **RRF** (k=60) | Weighted score fusion after calibration |
| Rerank | **bge-reranker-v2-m3** (0.6B, CPU/MPS ok) | Qwen3-Reranker-0.6B/4B (32k ctx, accuracy ceiling), mxbai-rerank-v2, LLM rerank (Gemini Flash) |
| Generate | **Gemini 2.5/3 Flash** (free) with structured output | Groq (Llama/GPT-OSS/Qwen, very fast) or local **Qwen3-8B / Gemma-3-12B Q4** through Ollama/MLX |
| Numbers | **DuckDB text-to-SQL** over extracted tables | Pandas agent (riskier) |
| Orchestrate | Plain Python + **LlamaIndex** *or* **LangGraph** for agentic flows | Haystack; avoid stacking frameworks |
| Eval | **RAGAS** + your own retrieval metrics (`ranx`/pytrec_eval) + **DeepEval** pytest gates | TruLens, Phoenix evals |
| Observe | **Arize Phoenix** (single Docker container, OpenInference/OTel) | **Langfuse** (MIT, self-host; heavier: Postgres+ClickHouse+Redis) |
| Guard | **Presidio** (PII), **Llama Prompt Guard 2** / **LLM Guard** (injection) | NeMo Guardrails for dialogue policy |

**Rule of thumb:** most RAG quality comes from parsing, chunking, hybrid retrieval and reranking, and this is even more true for finance PDFs. Add architectural complexity only when an eval gate says you need it.

---

## L0: Foundations (what RAG is and why)

### 0.1 Core RAG loop
- **What:** Index (parse → chunk → embed → store). Query (embed query → retrieve top-k → put into prompt → LLM answers with citations).
- **Why:** LLMs hallucinate, have stale knowledge and can't see private docs. RAG grounds answers in fresh, citable sources and avoids retraining.
- **Intuition:** an open-book exam. The model is the student and retrieval chooses which pages it may look at. A bad page selection gives a bad answer however smart the student is.
- **Skip when:** the corpus fits in context and is queried rarely (just stuff it in a long-context model), or the task is behavioural/style (fine-tune instead).
- **Failure taxonomy you will debug against** (Barnett et al., "Seven failure points"): missing content, missed top-ranked doc, not in context (cut by k), not extracted, wrong format, wrong specificity, incomplete answer. **Tag every eval failure with one of these.**

### 0.2 RAG vs fine-tuning vs long context
| Need | Use |
|---|---|
| Facts that change, citations, access control | RAG |
| Output format/tone/domain jargon | Fine-tune (LoRA) or few-shot |
| One ≤500-page doc, a few queries | Long context (Gemini 1M) + context caching |
| Thousands of docs, many queries, numbers must be exact | RAG + SQL over tables |

### L0 reading
- Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP*: https://arxiv.org/abs/2005.11401
- Gao et al. 2023, *RAG for LLMs: A Survey*: https://arxiv.org/abs/2312.10997
- Barnett et al. 2024, *Seven Failure Points When Engineering a RAG System*: https://arxiv.org/abs/2401.05856

---

## L1: Ingestion & Parsing (finance PDFs live or die here)

### 1.1 PDF text extraction
- **Why:** Annual reports have 2-column layouts, headers/footers on every page, footnotes, rotated tables, and Hindi+English bilingual pages. Naive extraction interleaves columns, breaks numbers across lines and drops table structure, so retrieval fails before it starts.
- **Intuition:** the PDF is a set of drawing instructions, not a document. You have to rebuild reading order.

| Tool | What it is | Strength | Weakness | Use when |
|---|---|---|---|---|
| **PyMuPDF (fitz) / pymupdf4llm** | C-backed text/coord extraction; `pymupdf4llm` emits Markdown | Very fast (~100s pages/s), blocks + bbox + fonts (font size → heading detection) | No real table structure on complex tables; AGPL licence (matters for commercial closed source) | Baseline, born-digital SEBI/RBI circulars, page-level metadata |
| **pdfplumber** | pdfminer-based, char-level geometry | Precise table extraction with explicit line/text strategies, debuggable visually | Slow; brittle without ruling lines | Surgical extraction of known-layout tables (e.g., RBI policy rate tables) |
| **Unstructured** (OSS) | Partitioners → typed elements (Title, NarrativeText, Table) | Many formats, element types, `chunk_by_title` | Heavy deps; best-quality "hi_res" needs detectron/OCR; the hosted API is not free beyond trial | Mixed formats (HTML, DOCX, emails) |
| **Docling** (IBM, MIT) | Layout model + TableFormer + reading order → `DoclingDocument` → MD/JSON; optional Granite-Docling 258M VLM | Strong tables, structure tree, bbox provenance (page no. for citations), **HybridChunker**, integrates with LlamaIndex/LangChain | Slower than PyMuPDF; weaker on scanned docs than Marker/MinerU (2026 olmOCR-bench) | **Default** for annual reports. Keep provenance for citations |
| **Marker** (Datalab) | PDF → MD/JSON with layout + OCR (Surya), optional LLM assist | Top 2026 olmOCR-bench scores; good equations/tables | GPL code + model weights licence limits commercial use; heavy on 16 GB | Scanned/messy docs; benchmark vs Docling |
| **MinerU** (OpenDataLab) | Pipeline/VLM parser, HTML tables | Best nested-header tables (TEDS) in 2026 comparisons | Big models; AGPL; Linux/CUDA-first (Mac works via CPU/MPS, slow) | Complex multi-level financial tables |
| **LlamaParse** (free 10k credits/mo) | Hosted agentic parser | High quality, zero setup | Cloud (data leaves laptop), credit cap, vendor lock | One-off **gold reference** to score your local parser against |

- **Recommendation:** Docling as the main path. PyMuPDF for fast page text and metadata. On a 20-page "hard sample" (consolidated P&L, segment tables, notes), compare Docling, MinerU and LlamaParse with a table-cell-accuracy check you write yourself. Choose per doc type, not globally.

### 1.2 Table extraction (highest-leverage item for finance)
- **What:** recover rows, columns and headers (including multi-level "FY25 | FY24" × "Standalone | Consolidated") into DataFrame/HTML/Markdown.
- **Why:** "What was consolidated EBITDA margin in FY24?" requires the right cell. Flattened text loses column semantics, so the LLM picks the wrong year or unit (₹ crore vs ₹ lakh).
- **Intuition:** each table gets three representations: (a) **Markdown/HTML** for the LLM to read, (b) a **one-paragraph LLM summary** for embedding (tables embed poorly as raw grids), (c) a **typed DataFrame → DuckDB** for exact numeric queries.
- **Do:** carry units/scale ("₹ in crore"), period, standalone/consolidated, and the table caption into metadata. Repeat header rows when a table is split (Docling `repeat_table_header`). Never split a table row across chunks.
- **Skip when:** the docs are narrative (RBI speeches).
- **Tools:** Docling TableFormer (default) → MinerU (nested headers) → pdfplumber (rule-based, deterministic) → Camelot (lattice tables; less maintained). Use a VLM (Gemini Flash on the page image) as a fallback for low-confidence tables only.

### 1.3 OCR
- **Why:** older NSE filings and signed board letters are scanned images and have no text layer.
- **Detect first:** if `page.get_text()` is near-empty, OCR that page. Don't OCR born-digital pages (it adds errors and is slow).
- **Tools:** **ocrmac / Apple Vision** (via Docling `OcrMacOptions`) is fast and free on Mac; **Tesseract** (`hin+eng` packs for Devanagari); **EasyOCR / RapidOCR** in Docling; **Surya** (Marker). Use a VLM OCR (Gemini Flash, Granite-Docling, olmOCR) for the worst pages.
- **Skip when:** the text layer exists and is clean.

### 1.4 Metadata extraction
- **What:** attach `{company, ticker/ISIN, doc_type (AR/circular/transcript/filing), fiscal_year, quarter, issuer (SEBI/RBI/NSE), circular_no, date, section, page, table_id, standalone|consolidated, language}` to every chunk.
- **Why:** enables pre-filtering ("only Infosys FY24 AR"), which removes cross-company/year confusion, the #1 finance-RAG error. It also powers citations and access control.
- **How:** rules first (filename, regex on "SEBI/HO/…" circular numbers, dates, NSE symbol lists), then an LLM extraction with a Pydantic schema (Gemini structured output) for the rest. Store it as payload in Qdrant / columns in LanceDB.
- **Skip when:** the corpus is a single doc.

### 1.5 Dedup & near-dedup
- **Why:** the same circular appears as master-circular + amendment, reports republish boilerplate (MD&A disclaimers) every year, and transcripts repeat safe-harbour text. Duplicates flood top-k and push out unique evidence.
- **How:** exact = SHA-256 of normalized text. Near = **MinHash LSH** (`datasketch`) at Jaccard ~0.85–0.9 on shingles, or embedding cosine >0.97. Keep the newest and record `supersedes` links (important for SEBI master circulars).
- **Skip when:** the corpus is hand-curated and small.

### 1.6 Incremental re-indexing / freshness
- **What:** re-process only new/changed docs. Use a doc-level content hash plus a chunk-level hash. Upsert changed chunks and delete orphans.
- **Why:** full re-embeds waste free-tier quota and hours. Stale chunks give outdated answers (e.g., an old repo rate).
- **How:** a manifest table (SQLite/DuckDB) `doc_id, source_url, sha256, parser_version, chunker_version, embed_model, indexed_at`. **Version the pipeline**: a parser/chunker/model change triggers a re-index of affected docs. LlamaIndex `IngestionPipeline` + docstore gives dedup/upsert out of the box.
- **Also:** `effective_date` and `superseded_by` metadata, so retrieval can prefer the latest regulation.

### L1 reading
- Docling technical report: https://arxiv.org/abs/2408.09869
- OmniDocBench (parsing benchmark): https://arxiv.org/abs/2412.07626
- FinanceBench (why finance RAG is hard): https://arxiv.org/abs/2311.11944

---

## L2: Chunking

General rule: **chunk on the document's own structure, then enforce a token budget.** Evaluate chunkers with retrieval recall, not by eye. Chroma's study found chunking choice can swing recall by ~9 pts.

| Technique | What | Why / fixes | Intuition | Skip when | Tool |
|---|---|---|---|---|---|
| **Fixed-size (chars)** | Split every N chars with overlap | Baseline only | A paper shredder | Always beyond the baseline (cuts sentences/tables) | `CharacterTextSplitter` |
| **Recursive** | Split on `\n\n` → `\n` → `. ` → ` ` until ≤N | Keeps paragraphs intact | Cut at the largest natural seam that fits | Docs with rich structure (use structure-aware) | LangChain `RecursiveCharacterTextSplitter`, LlamaIndex `SentenceSplitter` |
| **Token-based** | Budget in model tokens (e.g., 256–512) | Prevents silent truncation (bge 512/8192 limits) and gives predictable cost | Measure in the embedder's units | No reason to skip; always measure in tokens | `tiktoken`/HF tokenizer length fn; Docling HybridChunker |
| **Structure/Markdown-aware** | Split by headings/sections/tables (MD headers, Docling tree) | Chunks align with semantic units; headings become metadata/path | Follow the author's ToC | Flat unstructured text (transcripts → speaker-turn chunking instead) | **Docling HybridChunker** (default), `MarkdownHeaderTextSplitter`, Unstructured `chunk_by_title` |
| **Semantic** | Split where adjacent-sentence embedding similarity drops | Topic-coherent chunks in long flowing text | Cut where the topic shifts | Structured docs (headings already do this better). Expensive; gains are often small in benchmarks | LlamaIndex `SemanticSplitterNodeParser`, `chonkie` |
| **Parent-child / small-to-big** | Embed small child chunks (~128–256 tok); return their parent section (~1–2k tok) | Small = precise matching; big = enough context to answer | Search with a magnifying glass, read the whole page | Short docs; tight context budget | LlamaIndex `HierarchicalNodeParser` + `AutoMergingRetriever`; LangChain `ParentDocumentRetriever` |
| **Sentence-window** | Embed single sentences; at query time expand ±k sentences | Max precision on fact lookups | Find the sentence, then read around it | Tables; very short sentences | LlamaIndex `SentenceWindowNodeParser` + `MetadataReplacementPostProcessor` |
| **Late chunking** | Run the long-context embedder over the whole doc, then mean-pool token embeddings per chunk span | Chunk vectors "know" document context (pronouns, "the Company") | Embed first, cut later | Embedder isn't long-context (needs 8k+ ctx, e.g., jina-v3, bge-m3); docs much longer than context | `jina-embeddings-v3` `late_chunking=True`, or DIY with HF token embeddings |
| **Contextual chunk headers** | Prepend `Doc: Infosys AR FY24 > MD&A > Risk` to each chunk before embedding/BM25 | Chunks like "Revenue grew 12%" are ambiguous without entity/period | Every page gets a running header | Self-contained chunks | DIY string prefix (cheap, deterministic) |
| **Anthropic Contextual Retrieval** | LLM writes a 50–100-token situating context per chunk (given the full doc), prepended for embedding **and** BM25 | Anthropic: -49% retrieval failures (contextual embeddings + contextual BM25), -67% with reranking | A librarian annotates each index card | Huge corpora on free tier (1 LLM call per chunk); with Gemini, use context caching on the doc to cut cost | Gemini Flash / local Qwen3-8B; prompt from Anthropic cookbook |
| **Propositions** | LLM rewrites text into atomic, self-contained facts; index those | Fine-grained, decontextualized retrieval units (Dense X Retrieval) | One fact per index card | Costly at ingestion; loses tables/numbers context; risk of LLM distortion of figures | LlamaIndex `DenseXRetrievalPack`, custom prompt |

**Finance-specific chunking rules**
- Tables are atomic chunks (or split by row-groups with repeated headers). Store the table summary as a separate embedded node linked to the table.
- Earnings call transcripts: chunk by **speaker turn / Q&A pair**, with metadata `speaker, role (CEO/CFO/analyst), firm`.
- SEBI/RBI circulars: chunk by numbered paragraph/clause ("3.2.1") and keep the clause id in metadata for citations.
- Start at 400–512 tokens, 10–15% overlap. Tune using the recall@k eval, not intuition.

### L2 reading
- Anthropic, *Introducing Contextual Retrieval*: https://www.anthropic.com/engineering/contextual-retrieval
- Chroma Research, *Evaluating Chunking Strategies*: https://research.trychroma.com/evaluating-chunking
- Late Chunking (Günther et al.): https://arxiv.org/abs/2409.04701
- Dense X Retrieval / propositions: https://arxiv.org/abs/2312.06648

---

## L2: Embeddings

### 2.1 Dense embedding models
- **What:** a bi-encoder maps text to a vector, and cosine similarity approximates semantic relevance.
- **Why:** catches paraphrase ("topline" ≈ "revenue from operations") that BM25 misses.
- **Weakness:** exact tokens (ISINs, circular numbers, "Ind AS 116"), numbers and negation. Hence hybrid (L3).

| Model | Size / dims / ctx | Licence | Notes | Pick when |
|---|---|---|---|---|
| **bge-m3** (BAAI) | 568M / 1024 / 8192 | MIT | Dense + sparse + ColBERT multi-vector in one; 100+ langs (Hindi ok) | **Local default**: one model gives hybrid + multilingual |
| **Qwen3-Embedding** 0.6B/4B/8B | up to 4096 dims (MRL) / 32k | Apache-2.0 | 8B tops MTEB multilingual (2025–26); instruction-aware | 0.6B fits laptop well; 4B/8B quantized for best quality if RAM allows |
| **EmbeddingGemma-300M** | 768 (MRL→128) / 2k | Gemma terms | Tiny, strong for size, on-device | Lowest RAM/latency |
| **nomic-embed-text v1.5/v2** | 768 (MRL) / 8k | Apache-2.0 | `ollama pull nomic-embed-text`; needs `search_query:`/`search_document:` prefixes | Easiest Ollama path |
| **e5 (multilingual-e5-large-instruct)** | 1024 / 512 | MIT | Needs `query:`/`passage:` prefixes | Robust multilingual baseline |
| **gte (Alibaba gte-multilingual / ModernBERT)** | 768 / 8k | Apache-2.0 | Good speed/quality | Alternative to bge-m3 |
| **jina-embeddings-v3/v4** | 1024 (MRL) / 8k | CC-BY-NC (non-commercial) | Native late chunking, task LoRAs | Learning late chunking. Avoid for commercial |
| **gemini-embedding-001** (API) | 3072 → 1536/768 (MRL) / 2k | API, free tier | ~68 MTEB; strong multilingual; `task_type=RETRIEVAL_DOCUMENT/QUERY` | No local compute; but rate-limited, data leaves laptop, vendor lock-in for re-embeds |

- **Choosing via MTEB:** use the **Retrieval** column (not the overall average), filter by size ≤1B for laptop, and check the multilingual (MMTEB) column for Hindi. Leaderboard scores are partly overfit/contaminated, so **always re-run on your own golden set** (50–200 Q→chunk pairs). A 2-pt MTEB gap often flips on domain data.
- **Asymmetric prefixes / instructions:** many models (e5, nomic, Qwen3, bge for queries) expect a query instruction. Forgetting it silently costs recall.
- **Run locally on Apple Silicon:** `sentence-transformers` with `device="mps"`, or Ollama (`/api/embed`), or MLX (`mlx-embeddings`). Batch 32–64.

### 2.2 Dimensions & Matryoshka (MRL)
- **What:** MRL-trained models front-load information, so you can truncate a 1024-d vector to 256-d and re-normalize.
- **Why:** 4× less memory/latency for a small recall loss. Use a **two-stage** setup: search 256-d, rescore at full dims.
- **Skip when:** the model isn't MRL-trained (truncation then degrades badly), or the corpus is small (<1M vectors, where memory is a non-issue).

### 2.3 Quantization
- **What:** store vectors as int8 (scalar, 4× smaller, ~1% loss) or binary (32× smaller, needs rescoring with float/int8).
- **Why:** RAM on a 16 GB laptop; speed. Qdrant has built-in scalar/binary/product quantization with `rescore=True`. sentence-transformers has `quantize_embeddings`.
- **Skip when:** corpus is under ~100k chunks. A finance corpus of a few hundred reports is ~200k–500k chunks, so int8 is a reasonable but optional "production" exercise.

### 2.4 Fine-tuning embeddings
- **What:** contrastive fine-tune (MultipleNegativesRankingLoss, + MatryoshkaLoss) on (query, positive[, hard-negative]) pairs.
- **Why:** domain vocabulary (NIM, GNPA, CASA, PCR, "exceptional items") and Indian entity names; typical +5–15% recall@10 on in-domain data.
- **How (free):** generate synthetic queries per chunk with Gemini Flash. Mine hard negatives with the base model (`sentence_transformers.util.mine_hard_negatives`). Train bge-small/base or Qwen3-0.6B on MPS (fine for ≤50k pairs) or Colab/Kaggle free GPU. Evaluate with `InformationRetrievalEvaluator`.
- **Skip when:** you have no eval set yet, a reranker hasn't been tried (cheaper win), or the corpus changes domain often.
- **Tools:** `sentence-transformers` v3+ `SentenceTransformerTrainer` (default). LlamaIndex finetuning wrappers are thin and not needed.

### L2-embedding reading
- MTEB paper: https://arxiv.org/abs/2210.07316 · leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.13147
- HF blog, *Embedding Quantization*: https://huggingface.co/blog/embedding-quantization

---

## L2: Vector stores & ANN indexes

### Index types
| Index | How | Pros | Cons | Use |
|---|---|---|---|---|
| **Flat (brute force)** | Exact dot product vs all | 100% recall, zero tuning | O(N) per query | <100k vectors (your likely scale!), eval ground truth |
| **HNSW** | Multi-layer proximity graph; params `M`, `ef_construction`, `ef_search` | Best recall/latency; incremental inserts | RAM-heavy (vectors + graph), slow build, deletes are messy | Default in Qdrant/Chroma/pgvector/Weaviate |
| **IVF(-PQ)** | k-means cells, probe `nprobe` cells; PQ compresses | Low memory, fast build at scale | Needs training; recall depends on nprobe; worse for frequent inserts | 10M+ vectors, disk-based (LanceDB, FAISS, Milvus) |
| **DiskANN / on-disk HNSW** | Graph on SSD | Billion-scale | Complexity | Not needed on laptop |

- **Filtered search pitfall:** post-filtering HNSW results can return <k hits when the filter is selective (e.g., one company out of 500). Use engines with **filter-aware HNSW / payload indexes** (Qdrant), or pre-filter then flat-scan small subsets.

### Store comparison (laptop lens)
| Store | Mode | Hybrid/sparse | Filtering | Why pick | Why not |
|---|---|---|---|---|---|
| **Qdrant** | Docker server or `QdrantClient(":memory:"/path)` local | Native sparse vectors, multivector (ColBERT/ColPali), server-side fusion (RRF/DBSF) via Query API | Excellent (payload indexes, filter-aware HNSW) | **Default**: production-like API, hybrid + multivector + quantization in one, light on RAM | One more container; local-file mode is for testing only |
| **LanceDB** | Embedded (files), no server | Native FTS (Tantivy) + vector + reranker hooks | SQL-like `where` | Zero-ops, columnar, multimodal, versioned tables | Smaller ecosystem; index tuning less mature for small-k filters |
| **Chroma** | Embedded / server | No native lexical (DIY hybrid) | Basic metadata `where` | Fastest to prototype | Hybrid, multivector and filtering are weaker; outgrow quickly |
| **FAISS** | Library only | None | None (DIY id maps) | Learn index internals (Flat/HNSW/IVF-PQ), benchmarks | Not a database (no metadata, persistence is DIY) |
| **pgvector** (+ pgvectorscale) | Postgres extension | Postgres FTS (`tsvector`) / ParadeDB BM25 | Full SQL | One DB for chunks + tables + ACLs + text-to-SQL | HNSW in PG uses RAM; tuning; hybrid is manual SQL |
| **Milvus** (Lite/Standalone) | Embedded Lite or multi-container | Sparse + BM25 functions | Good | Massive scale, GPU indexes | Overkill/heavy for 16 GB |
| **Weaviate** | Docker | Native BM25 + hybrid `alpha` | Good | Built-in hybrid, modules | JVM-free but heavier; GraphQL-ish API |

- **Recommendation:** Qdrant in Docker (≈200–400 MB RAM at your scale) for the main project. Use FAISS-Flat in a notebook to understand recall vs HNSW. Consider **pgvector** only if you want a single Postgres for chunks + financial tables + users/ACLs. DuckDB is a lighter alternative for the tables.

### L2-store reading
- HNSW paper (Malkov & Yashunin): https://arxiv.org/abs/1603.09320
- Pinecone, *Faiss: The Missing Manual* (IVF/PQ/HNSW intuition): https://www.pinecone.io/learn/series/faiss/
- Qdrant docs, *Hybrid Queries*: https://qdrant.tech/documentation/concepts/hybrid-queries/

---

## L3: Retrieval

### 3.1 BM25 / sparse lexical
- **What:** term-frequency × IDF with length normalization (BM25). Learned sparse (SPLADE, bge-m3 sparse) = a model assigns weights to vocab terms, including expansions.
- **Why:** exact matches dense models blur: "SEBI/HO/CFD/…/2024/123", "Reg 30 LODR", tickers, "Ind AS 115", specific numbers.
- **Intuition:** dense finds what you *mean*; BM25 finds what you *said*.
- **Skip when:** purely conversational/paraphrastic queries with no rare tokens (rare in finance).
- **Tools:** **bm25s** (pure-Python sparse-matrix BM25, ~500× faster than rank_bm25, saves to disk). Qdrant sparse vectors (with `fastembed` `Qdrant/bm25` or bge-m3 sparse) keep it in-DB. **SPLADE-v3** (`naver/splade-v3`, non-commercial licence) when vocabulary mismatch hurts BM25. Avoid `rank_bm25` beyond toys (slow).
- **Tokenization:** keep alphanumerics like "FY24", "Q3FY25", "₹" and the slashes in circular numbers. Customize the tokenizer.

### 3.2 Hybrid search + RRF
- **What:** run dense and sparse, then fuse. **RRF:** `score(d) = Σ 1/(k + rank_i(d))`, k≈60.
- **Why:** each retriever's misses are partly disjoint, so hybrid reliably beats either alone on BEIR-style and financial benchmarks. RRF needs no score calibration (cosine and BM25 scales differ).
- **Intuition:** a democratic vote on ranks rather than raw scores.
- **Skip when:** a single retriever already hits ~100% recall@k on your eval (rare).
- **Alternatives:** weighted linear fusion (`alpha` in Weaviate) after min-max normalization is better if tuned on eval data. Qdrant DBSF (distribution-based) is another option. Use RRF by default, and tune alpha only once you have ≥100 eval queries.
- **Tools:** Qdrant Query API `prefetch=[dense, sparse], query=FusionQuery(fusion=Fusion.RRF)`; LlamaIndex `QueryFusionRetriever(mode="reciprocal_rerank")`; LangChain `EnsembleRetriever`.

### 3.3 Rerankers (2nd stage)
- **What:** retrieve a wide set (top 50–100), re-score (query, chunk) jointly, keep the top 5–8.
- **Why:** bi-encoders compress a doc to one vector before seeing the query. Cross-encoders attend across both, so they are far more precise. This is typically the **single biggest precision gain** per unit of effort.
- **Intuition:** retriever = fast recall-oriented net; reranker = careful reader.
- **Skip when:** latency budget is <100 ms on CPU with large k, or recall@50 is already the bottleneck (a reranker can't fix missing docs).

| Reranker | Type | Notes | Pick |
|---|---|---|---|
| **bge-reranker-v2-m3** | Cross-encoder 0.6B, Apache-2.0 | Multilingual, the "boring correct default"; ~50 pairs in 1–2 s on MPS | **Default** |
| **Qwen3-Reranker 0.6B/4B** | LLM-based, 32k ctx | Highest open accuracy; 4B is slow on laptop | Upgrade if eval shows gains; 0.6B is laptop-friendly |
| **mxbai-rerank-v2 (base/large)** | Qwen-based, Apache-2.0 | Strong, long ctx | Alternative |
| **ColBERT / late interaction** (ColBERTv2, answerai-colbert-small, bge-m3 colbert vecs) | Token-level MaxSim between query/doc token vectors | Near cross-encoder quality at retrieval-time speed; storage heavy (a vector per token) | As a reranker over top-100 via Qdrant multivector or `pylate`/RAGatouille |
| **LLM rerank** (RankGPT-style listwise, Gemini Flash) | Prompt LLM to order passages | Handles reasoning ("which chunk states FY24 consolidated figure") | Costly/rate-limited; position bias. Use for top-20 only, or as a judge |
| **FlashRank** | Tiny ONNX cross-encoders | CPU-fast | Very low-latency demos |

### 3.4 MMR (Maximal Marginal Relevance)
- **What:** greedily pick chunks maximizing `λ·sim(q,d) − (1−λ)·max sim(d, selected)`.
- **Why:** avoids 5 near-identical boilerplate chunks; increases coverage for multi-aspect questions.
- **Skip when:** single-fact lookups (diversity can drop the one right chunk). Prefer dedup + reranker. Apply MMR *after* reranking if at all.
- **Tools:** LangChain `max_marginal_relevance_search`, LlamaIndex `mmr` mode, or ~15 lines of numpy.

### 3.5 Metadata filtering
- **What:** hard constraints `company=INFY AND fiscal_year=2024 AND doc_type=AR`.
- **Why:** finance questions almost always name an entity and a period. Filters remove the entire class of "right metric, wrong company/year" errors and shrink the search space.
- **Skip when:** the user's query doesn't specify (don't hallucinate filters; fall back to unfiltered + boost).
- **Tool:** Qdrant payload indexes on `company`, `fiscal_year`, `doc_type`.

### 3.6 Self-query retrieval
- **What:** an LLM converts the NL query into `{semantic_query, filters}` using a schema of allowed metadata fields/values.
- **Why:** automates 3.5 from free text ("HDFC Bank's GNPA trend in last 3 years").
- **Pitfalls:** LLM invents field values ("HDFC" vs "HDFCBANK"). Constrain with enums and fuzzy-map entity names to canonical tickers (a symbol table from NSE equity list).
- **Tools:** Gemini structured output → Pydantic `QueryPlan`. LangChain `SelfQueryRetriever` / LlamaIndex `VectorIndexAutoRetriever` exist, but a custom Pydantic version is more controllable.

### L3 reading
- RRF (Cormack et al. 2009): https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- ColBERTv2: https://arxiv.org/abs/2112.01488 · SPLADE: https://arxiv.org/abs/2107.05720
- BM25S: https://arxiv.org/abs/2407.03618 · BEIR benchmark: https://arxiv.org/abs/2104.08663

---

## L3: Query-side transformations

| Technique | What | Why (failure fixed) | Intuition | Skip when | Tool |
|---|---|---|---|---|---|
| **Query rewriting** | LLM rewrites a vague/conversational query into a standalone, keyword-rich one; resolves "it/they" from chat history | Follow-ups ("and last year?") and typos retrieve nothing | Translate user-speak to index-speak | Queries already precise; adds 300–800 ms + 1 LLM call | Prompt + Gemini Flash-Lite / Groq 8B |
| **Multi-query** | Generate 3–5 paraphrases, retrieve each, union | Single phrasing misses relevant chunks (recall) | Ask the librarian 5 ways | Tight rate limits; precise queries | LangChain `MultiQueryRetriever`, LlamaIndex `QueryFusionRetriever(num_queries=4)` |
| **RAG-Fusion** | Multi-query + RRF fuse rankings | Robust recall; ranks docs supported across variants | Consensus across paraphrases | Drift: bad variants pull off-topic docs (the paper notes this) | Same as above with RRF mode |
| **HyDE** | LLM writes a hypothetical answer; embed *that* and search | Short queries vs long passages are asymmetric; answer-like text lands nearer true passages | Search with a fake answer shaped like the real one | Numbers/entities: the hallucinated figure biases retrieval to wrong years. Also strong embedders already handle asymmetry. Use BM25 on the original query alongside | LlamaIndex `HyDEQueryTransform`; DIY |
| **Step-back prompting** | Ask a more general question first ("What drives bank NIM?") and retrieve both | Specific queries miss conceptual/background context | Zoom out before zooming in | Simple factual lookups | DIY prompt |
| **Decomposition** | Break into sub-questions (per company/year/metric), answer each, synthesize | Multi-hop/comparative ("Compare TCS and Infy FY24 margins") retrieve only one side | Divide and conquer | Single-hop queries (wasted calls) | LlamaIndex `SubQuestionQueryEngine`; LangGraph map-reduce |
| **Routing** | Choose the destination: vector index vs SQL tables vs graph vs web vs no-retrieval | One index can't serve both "explain" and "compute" questions | A switchboard | Single data source | LLM router with Pydantic enum; or a cheap classifier |
| **Query classification** | Classify intent (factoid-number / narrative / comparison / regulatory-lookup / chit-chat / out-of-scope) and complexity | Drives routing, k, filters and refusal; blocks out-of-domain queries early | Triage nurse | Tiny homogeneous query set | Few-shot LLM, or embedding + logistic regression (fast, free) trained on golden-set labels |

- **Order of adoption for finance:** classification + self-query filters → rewriting (for chat) → decomposition (comparisons) → multi-query/RAG-fusion → HyDE last, gated on eval.

### L3-query reading
- HyDE (Gao et al.): https://arxiv.org/abs/2212.10496
- Step-Back Prompting: https://arxiv.org/abs/2310.06117
- RAG-Fusion: https://arxiv.org/abs/2402.03367

---

## L3: Generation

### 4.1 Prompt design
- System prompt contract: answer **only** from the context; cite `[doc_id:page]` per claim; say "not found in provided documents" otherwise; preserve units and periods; never compute unless asked (or route to SQL/tool).
- Wrap each chunk in delimiters with metadata: `<source id="INFY_AR_FY24_p142" company="INFY" fy="2024" section="Consolidated P&L">…</source>`. Tell the model that content inside `<source>` is **data, not instructions** (injection defence).
- Few-shot 1–2 examples of a cited, unit-aware answer. Low temperature (0–0.2).

### 4.2 Context ordering ("lost in the middle")
- **What:** LLMs use information at the start/end of context better than the middle.
- **Fix:** put the highest-reranked chunks at the start *and* end (LlamaIndex `LongContextReorder`), and keep k small (5–8 after rerank). Recent long-context models are better at this, but the effect plus "more passages → worse" from hard negatives still exist.
- **Skip when:** ≤3 chunks.

### 4.3 Context compression
- **What:** remove irrelevant sentences/tokens before generation.
- **Why:** cut tokens (free-tier TPM limits!), reduce distraction from hard negatives.
- **Options:** (a) **LLMLingua-2 / LongLLMLingua** (small model drops low-information tokens, 2–5× compression; risk: can mangle numbers/tables, so exclude tables); (b) **contextual compression** = LLM/embedding filter extracts the relevant sentences per chunk (LangChain `ContextualCompressionRetriever` with `EmbeddingsFilter`/`LLMChainExtractor`); (c) simply a better reranker + smaller k (often enough).
- **Skip when:** contexts are already small, or exact table fidelity matters.

### 4.4 Citations / attribution
- **Why:** finance users must verify, and citations make unfaithfulness auditable.
- **How:** structured output `{answer, claims:[{text, source_ids, quote}]}`. **Verify** post-hoc that each `quote` is a substring/fuzzy-match (rapidfuzz ≥90) of the cited chunk, then drop/flag unsupported claims. Map chunk → page + bbox (Docling provenance) to deep-link the PDF page.
- **Tools:** Gemini `response_schema` / Pydantic; `instructor` library for any provider (works with Gemini, Groq, Ollama).

### 4.5 Structured output
- **Why:** downstream UIs/agents/evals need JSON. Numeric answers need `{value, unit, scale, period, basis: standalone|consolidated, source}`.
- **Tools:** Gemini native JSON schema; Ollama `format=<json schema>`; `instructor`; Outlines for constrained local decoding.

### 4.6 Long-context vs RAG trade-off
| Factor | Long context (Gemini 1M) | RAG |
|---|---|---|
| Corpus | One/few docs | Many docs, growing |
| Cost/latency | High per query (cache helps) | Low |
| Quality | Good for synthesis across one doc; degrades with distractors | Precise; depends on retrieval |
| Citations/ACL | Harder | Natural |
| Free tier | TPM limits bite quickly | Fits |
- **Hybrid pattern:** RAG selects the 1–3 relevant *documents/sections*, then a long-context model reads them whole ("retrieve-then-read-long"). Good for "summarize the risk section of X's AR".

### L3-generation reading
- Lost in the Middle (Liu et al.): https://arxiv.org/abs/2307.03172
- LongLLMLingua: https://arxiv.org/abs/2310.06839
- Long-Context LLMs Meet RAG (Jin et al.): https://arxiv.org/abs/2410.05983

---

## L4: Advanced architectures

### 5.1 Text-to-SQL hybrid for numeric tables (**most important for finance**)
- **What:** extracted tables → normalized long table `financials(company, fy, quarter, statement, line_item, basis, value, unit, source_page)` in **DuckDB**. Router sends numeric/aggregation questions ("3-yr revenue CAGR", "top 5 banks by GNPA") to an LLM that writes SQL over a documented schema. Narrative questions go to vector RAG. Mixed questions use both.
- **Why:** LLMs are bad at arithmetic and at picking the right cell among similar tables. SQL is exact and auditable.
- **Intuition:** let the LLM write the query, let the database do the maths.
- **Pitfalls:** line-item naming varies by company ("Revenue from operations" vs "Total income"). Build a **line-item synonym map**/canonicalization. Use a read-only connection, `LIMIT`s and a SQL validator (`sqlglot`). Show the SQL to the user.
- **Skip when:** questions are qualitative.
- **Tools:** DuckDB (embedded, fast) + LlamaIndex `NLSQLTableQueryEngine` or a custom prompt + `sqlglot`. Vanna is an alternative. Postgres if already on pgvector.

### 5.2 Knowledge graphs / GraphRAG
- **What:** LLM extracts entities/relations (Company–subsidiary–of, Promoter–holds–%, Regulator–issues–Circular–amends–Circular) into a graph; retrieval traverses neighbors/communities.
- **Why:** vector RAG fails at **global/sensemaking** ("main themes across all RBI policies in 2025") and **relational multi-hop** ("which group companies of X have related-party transactions with Y").
- **Microsoft GraphRAG:** entity graph → Leiden communities → hierarchical community summaries; *local* (entity-centric) and *global* (map-reduce over summaries) search. Best for global questions; **indexing is very expensive** (many LLM calls per chunk, so it burns free tiers; run on a small subset or with local Ollama).
- **LightRAG:** graph + vectors, dual-level (low = entities, high = themes) retrieval, **incremental updates**, much cheaper indexing (~1/100 of GraphRAG in community comparisons; quality typically within 70–90%+). Ollama/Gemini supported. **Pick LightRAG** for a laptop; try MS GraphRAG on ~20 docs to learn the concept.
- **Neo4j Community (Docker) vs NetworkX:** NetworkX is in-memory and fine for <100k edges/prototyping. Neo4j adds Cypher, persistence, a vector index and a visual browser, which is good for regulatory "amends/supersedes" chains. Start NetworkX → move to Neo4j if you need Cypher queries.
- **Skip when:** questions are local fact lookups (graph adds cost and extraction noise); no eval shows multi-hop/global failures.

### 5.3 Agentic RAG
- **What:** an LLM agent with tools (`vector_search(filters)`, `sql_query`, `get_page`, `graph_query`, `calculator`) decides what to retrieve, iterates and stops.
- **Why:** fixed pipelines can't handle multi-step questions or recover from bad retrieval.
- **Skip when:** latency/cost matter and >80% of queries are single-hop. Agents also add nondeterminism and are harder to eval.
- **Tools:** **LangGraph** (explicit state machine; good for CRAG/Self-RAG/Adaptive flows), LlamaIndex Workflows/AgentWorkflow, PydanticAI. Your MCP experience applies directly: expose the retriever as an **MCP server** so Claude Code/other agents can use it.

### 5.4 Corrective RAG (CRAG)
- **What:** a retrieval evaluator grades the retrieved docs as Correct / Ambiguous / Incorrect. If Incorrect → rewrite the query and re-retrieve (originally web search); if Ambiguous → combine. Docs are also decomposed into strips and filtered.
- **Why:** blindly generating from irrelevant context produces confident hallucinations.
- **Free impl:** a grader LLM (Groq 8B / Gemini Flash-Lite) with a binary relevance schema in LangGraph; fallback = broaden filters / switch to SQL / web search (Tavily free tier or DuckDuckGo).
- **Skip when:** a reranker score threshold already gives the same signal more cheaply. Try a **reranker-score gate** first.

### 5.5 Self-RAG
- **What:** the model emits reflection tokens deciding *whether* to retrieve, whether passages are relevant, and whether its output is supported/useful. The original trains a special LM.
- **Practical version:** a LangGraph loop with an "is grounded?" (hallucination grader) and "answers question?" check, regenerating or re-retrieving on failure.
- **Skip when:** latency matters. Each check is an LLM call, and the loop can blow free-tier RPM.

### 5.6 Adaptive RAG
- **What:** a query-complexity classifier routes to no-retrieval / single-step / iterative multi-step retrieval.
- **Why:** spend compute only where needed: simple lookups stay fast, complex ones get iteration.
- **Free impl:** your L3 query classifier + router. The paper uses a small trained classifier (T5); you can use a few-shot LLM or a logistic regression on embeddings.

### 5.7 Multi-hop retrieval
- **What:** iterative retrieve → reason → retrieve (IRCoT-style), or decomposition (L3).
- **Finance example:** "Who is the auditor of the subsidiary that contributed most to X's FY24 revenue?" needs segment table → subsidiary name → auditor section.
- **Tools:** LangGraph loop with max-hops = 3; or agent with tools.

### 5.8 RAPTOR (hierarchical summaries)
- **What:** cluster chunks (GMM on embeddings, UMAP), summarize clusters recursively into a tree; index all levels (collapsed tree).
- **Why:** questions needing synthesis across a long doc (e.g., "overall strategy described in the AR") have no single chunk with the answer.
- **Skip when:** factoid/numeric QA; frequently changing corpora (the tree must be rebuilt). Cost is O(chunks) summaries.
- **Tools:** LlamaIndex `RaptorPack`, or ~150 lines DIY. Cheaper "poor man's RAPTOR": section-level and doc-level summaries as extra nodes (use the Docling structure tree).

### 5.9 Multimodal RAG (ColPali family)
- **What:** embed **page images** directly with a VLM retriever (ColPali/ColQwen2.5/ColSmol) into multi-vectors (~1k patch vectors × 128-d per page); late-interaction MaxSim at query time; feed the top page images to a VLM (Gemini Flash) for answering.
- **Why:** charts, infographics, complex tables and scanned pages that parsers mangle. It bypasses OCR/parsing entirely.
- **Skip when:** born-digital text docs parse well (text RAG is cheaper/more precise for numbers). Storage is heavy (multi-vector per page; use Qdrant multivector + pooling/binary quantization). ColQwen2.5-3B on 16 GB MPS is feasible but slow for thousands of pages, so index a subset (investor presentations).
- **Tools:** `colpali-engine` (ColQwen2.5 / ColSmol-256M for laptop), **Byaldi** (simple wrapper; prototyping), Qdrant multivector for storage. Generation via Gemini Flash (multimodal, free tier).

### 5.10 Memory
- **What:** conversation memory (short-term: last N turns + rolling summary; long-term: user facts/preferences in a vector/KV store) and **query rewriting with history**.
- **Why:** follow-ups and analyst workflows ("now compare with ICICI").
- **Tools:** LangGraph checkpointers (SQLite), Mem0 (OSS), LlamaIndex `ChatMemoryBuffer`. Keep memory separate from the document index (stops user chatter from polluting retrieval).
- **Skip when:** single-shot QA API.

### L4 reading
- GraphRAG (Edge et al.): https://arxiv.org/abs/2404.16130 · LightRAG: https://arxiv.org/abs/2410.05779
- Self-RAG: https://arxiv.org/abs/2310.11511 · CRAG: https://arxiv.org/abs/2401.15884 · Adaptive-RAG: https://arxiv.org/abs/2403.14403
- RAPTOR: https://arxiv.org/abs/2401.18059 · ColPali: https://arxiv.org/abs/2407.01449 · Agentic RAG survey: https://arxiv.org/abs/2501.09136
- LangGraph agentic RAG tutorials: https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/

---

## L5a: Evaluation (build this before L2, not after)

### 6.1 Retrieval metrics (need a qrels set: query → relevant chunk/page ids)
| Metric | Meaning | Use |
|---|---|---|
| **Hit rate@k** | ≥1 relevant in top-k | Quick sanity metric |
| **Recall@k** | Fraction of relevant items in top-k | **Primary for first stage** (k = rerank pool, e.g., 50) |
| **Precision@k** | Fraction of top-k that are relevant | After rerank (k=5) |
| **MRR** | 1/rank of first relevant | Single-answer factoids |
| **nDCG@k** | Graded relevance, position-discounted | Primary post-rerank ranking quality |
- Label at **page level** or "evidence span" level, not chunk ids, so labels survive re-chunking. Map retrieved chunks → pages to score.
- **Tools:** `ranx` (fast, stat significance tests, fusion), `pytrec_eval`. Report mean ± bootstrap CI. With 100 queries, ±3–5 pts is noise.

### 6.2 Generation metrics
- **Faithfulness / groundedness:** claims in the answer supported by the context (catches hallucination).
- **Answer relevance:** does the answer address the question.
- **Context precision:** relevant chunks ranked high (a retrieval-ranking quality proxy).
- **Context recall:** ground-truth answer statements attributable to the context (needs a reference answer).
- **Answer correctness:** vs reference. For finance add a **deterministic numeric check**: parse value+unit+period and compare with tolerance (e.g., ±0.5%). This beats an LLM judge for numbers.
- **Citation precision/recall:** cited pages ∈ gold pages.
- **Refusal accuracy:** unanswerable questions (~10–15% of the set) should get "not found".

### 6.3 Frameworks
| Tool | Strength | Weakness | Use |
|---|---|---|---|
| **RAGAS** | Standard RAG metrics, synthetic **testset generation** (knowledge-graph based, multi-hop queries), any LLM via LangChain/LlamaIndex wrappers (Gemini, Ollama) | Metric drift across versions; LLM-judge noise; many calls per sample (watch RPM) | **Primary metric engine** + synthetic data |
| **DeepEval** | pytest-style `assert_test`, G-Eval custom criteria, CI gates, RAG metrics | Opinionated; cloud dashboard upsell (optional) | **CI regression gates** |
| **TruLens** | "RAG triad" (context relevance, groundedness, answer relevance), feedback functions, local dashboard | Smaller community now (Snowflake-owned) | Alternative to RAGAS |
| **Arize Phoenix evals** | Evals tied to traces; hallucination/QA/relevance templates; runs locally | Elastic License 2.0 (source-available) | When you already trace in Phoenix |

### 6.4 LLM-as-judge pitfalls
- Position bias, verbosity bias, **self-preference** (judge = generator model family → inflated scores). Use a different family as the judge (e.g., generate with Gemini, judge with Groq Llama/Qwen, or vice versa).
- Judges are bad at numbers/units, so use deterministic checks.
- **Calibrate:** hand-label 50 samples and measure judge-human agreement (Cohen's κ). Only trust metrics with κ ≥ ~0.6.
- Binary/3-point rubrics with reasoning-before-score beat 1–10 scales. Fix temperature 0 and pin judge model + prompt version.
- Free-tier nondeterminism: cache judge outputs keyed by (sample hash, judge version).

### 6.5 Golden dataset (for this project)
- **Size:** start with 50 hand-written Q/A, grow to 150–300. Stratify by doc type (AR / circular / policy / transcript / filing) × question type (numeric lookup, multi-period, comparison across companies, regulatory clause lookup, narrative/why, multi-hop, unanswerable).
- **Per item:** `question, reference_answer, gold_pages[], gold_value{value,unit,period,basis}, doc_ids, difficulty, type`.
- **Synthetic expansion:** RAGAS `TestsetGenerator` or a custom Gemini prompt per chunk ("write a question answerable only from this chunk; include company and FY"). **Filter**: drop questions answerable without context (ask the LLM closed-book), drop duplicates, then manually review a 20% sample. Keep synthetic and human sets **separate** in reporting (synthetic sets are easier).
- **Freeze** a test split; tune only on the dev split.
- Leverage public sets for sanity: FinanceBench (US 10-K, open subset 150 Qs), FinQA/TAT-QA (table+text numeric reasoning).

### L5a reading
- RAGAS paper: https://arxiv.org/abs/2309.15217 · docs: https://docs.ragas.io
- ARES (automated RAG eval with judge calibration): https://arxiv.org/abs/2311.09476
- Judging LLM-as-a-Judge (biases): https://arxiv.org/abs/2306.05685
- Hamel Husain, *Your AI Product Needs Evals*: https://hamel.dev/blog/posts/evals/

---

## L5b: Production

| Concern | Technique | Why / failure fixed | Free tool | Notes / when not |
|---|---|---|---|---|
| **Exact cache** | Hash(normalized query + filters + index version) → answer | Repeat queries burn free-tier quota | `diskcache` / Redis (Docker) | Invalidate on index version bump |
| **Semantic cache** | Embed query; if cos > τ (~0.95) to a cached query, return cached answer | Paraphrased repeats | RedisVL `SemanticCache`, GPTCache, or Qdrant collection | **Dangerous in finance**: "INFY FY24 revenue" ≈ "INFY FY23 revenue" at cos 0.97. Include extracted entities/periods in the cache key; use high τ |
| **Embedding cache** | Cache chunk/query embeddings by content hash | Re-index speed; API quota | LlamaIndex `IngestionCache`, LangChain `CacheBackedEmbeddings` | Always on |
| **Provider caching** | Gemini implicit/explicit context caching | Cheaper repeated long contexts (contextual retrieval over one doc) | Gemini API | Check free-tier support |
| **Latency** | Parallel dense+sparse retrieval; rerank only top-30–50; stream tokens; small router model (Groq 8B ≈ very fast) | p95 < ~3 s target | asyncio, Qdrant prefetch | Measure each stage separately |
| **Cost / quota** | Multi-provider fallback with retry/backoff on 429; budget per request | Free tiers rate-limit (RPM/RPD) | **LiteLLM** (OSS router: Gemini/Groq/OpenRouter/Ollama behind one API) | Keep a local Ollama fallback |
| **Prompt injection (indirect)** | Docs/web pages can contain "ignore previous instructions…" | Retrieved text is attacker-controlled input | **Llama Prompt Guard 2** (22M/86M classifier) on chunks at ingest + on retrieved context; LLM Guard scanners; delimiting + "data not instructions"; no tool side-effects without allowlists | Classifiers are evadable, so use defence in depth: least-privilege tools, output validation |
| **PII** | Detect/mask PAN, Aadhaar, phone, email in ingested docs/outputs | Shareholder lists, transcripts may contain PII; DPDP Act 2023 | **Microsoft Presidio** + custom recognizers (PAN regex `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar 12-digit with Verhoeff check) | Mask at ingest if not needed for answers |
| **Output guardrails** | Groundedness check, citation verification, numeric-format validation, "not investment advice" disclaimer | Hallucinated figures | Your own validators + DeepEval/Phoenix hallucination eval online (sampled) | Sample 5–10% online to save quota |
| **Freshness** | Scheduled crawl of SEBI/RBI/NSE pages → hash diff → incremental index; `effective_date`, `superseded_by`; recency boost in ranking | Stale regulation answers | cron/APScheduler, `httpx`; respect robots/ToS | Show "as of" date in answers |
| **Access control** | Store `acl_groups` in chunk payload; **mandatory filter at retrieval** derived from the authenticated user (never from LLM output) | Data leakage across tenants/users | Qdrant payload filter / Postgres RLS | Don't rely on prompt instructions for ACL |
| **Observability** | Trace every stage (query, rewritten query, filters, retrieved ids + scores, reranked ids, prompt, tokens, latency, cost, judge scores); user feedback 👍/👎 linked to trace | Debug "why did it answer that"; regressions | **Arize Phoenix** (1 container, OTel/OpenInference auto-instrumentation for LlamaIndex/LangChain) or **Langfuse** (MIT, prompt mgmt, datasets, heavier stack) | Phoenix for solo laptop; Langfuse if you want prompt versioning + team UI |
| **Versioning & CI** | Version parser/chunker/embedder/prompt; eval suite runs in CI on every change; release only if gates pass | Silent regressions | DeepEval + pytest + GitHub Actions; `ranx` comparisons | Keep CI eval subset small (≤50 Qs) for quota |
| **Serving** | FastAPI (+ SSE streaming) + Docker Compose (Qdrant, Phoenix, app); MCP server exposing `search_filings`, `query_financials` | Usable product; agent integration | FastAPI, FastMCP | Add Streamlit/Gradio UI with PDF page preview + highlighted bbox |

### L5b reading
- Greshake et al., *Indirect Prompt Injection*: https://arxiv.org/abs/2302.12173
- OWASP Top 10 for LLM Apps (2025): https://genai.owasp.org/llm-top-10/
- Arize Phoenix docs: https://arize.com/docs/phoenix · Langfuse self-hosting: https://langfuse.com/self-hosting

---

## Build plan: "NiveshRAG" (Indian financial-docs RAG), milestones with eval gates

**Corpus (start small, grow):** 10 Nifty companies × 2 years of annual reports (~4–6k pages), ~50 SEBI circulars (LODR, ICDR, MF), RBI MPC statements (last 8), ~20 earnings-call transcripts, a few NSE corporate announcements. Download from official sites manually or via a polite script.

**Global rules:** every milestone ends with `make eval` → a results table committed to `eval/results/Mx.json` (ranx + RAGAS + numeric accuracy + latency p50/p95 + LLM calls/query). A change ships only if its gate passes **and** nothing else regresses by more than the CI noise band.

| # | Milestone | Techniques added | Eval gate (dev set) | Expected movement |
|---|---|---|---|---|
| **M0** | Eval harness first | 60 hand-written golden Qs (stratified, 10 unanswerable), gold pages + gold values; ranx, RAGAS (judge = different model family), numeric checker; Phoenix tracing | Judge-vs-human κ ≥ 0.6 on 30 samples | n/a (baseline infra) |
| **M1** | Naive RAG baseline | PyMuPDF text → recursive 512-token chunks → bge-m3 dense → Qdrant → top-5 → Gemini Flash | Record baseline: recall@5, MRR, faithfulness, numeric acc | Expect poor numeric acc (~30–50%) and table failures |
| **M2** | Parsing & metadata | Docling (tables to MD, provenance) + OCR detection; metadata schema; dedup; incremental manifest | Numeric-lookup **recall@10 ↑ ≥15 pts** vs M1; table questions' answer acc ↑ | Biggest jump for table Qs |
| **M3** | Structure-aware chunking + headers | HybridChunker (repeat table headers), contextual chunk headers; transcript speaker-turn chunks; circular clause chunks; chunk-size sweep (256/512/1024) | Recall@10 ↑; choose size by recall@10 and faithfulness | Fewer "right section, wrong year" misses |
| **M4** | Hybrid retrieval | bm25s / Qdrant sparse + dense, RRF; metadata filters from self-query (canonical ticker map) | **Recall@20 ↑ ≥5 pts**; regulatory-ID queries hit@5 ≈ 100% | Exact-token queries fixed |
| **M5** | Reranking | bge-reranker-v2-m3 over top-50 → top-6; LongContextReorder; reranker threshold → "not found" | **Context precision ↑, nDCG@5 ↑ ≥8 pts**, faithfulness ↑; latency p95 < 4 s | Precision jump; refusal accuracy ↑ |
| **M6** | Generation contract | Structured output `{answer, value, unit, period, basis, citations[quote,page]}`; quote verification; injection-safe delimiting | **Citation precision ≥ 0.9**, faithfulness ≥ 0.85, numeric acc ↑ | Auditable answers |
| **M7** | Contextual retrieval | Gemini (with doc caching) or local Qwen3-8B context per chunk; re-embed + re-BM25 | **Retrieval failure@20 ↓ ≥30% relative** (Anthropic saw −49%; validate on your data) | Cost: one LLM call per chunk; check quota |
| **M8** | Query intelligence | Query classifier + router; rewriting for follow-ups (chat memory); decomposition for comparisons; (optional) multi-query / HyDE A/B | Comparison/multi-period Qs **answer acc ↑ ≥15 pts**; HyDE kept only if recall ↑ without numeric acc ↓ | Multi-entity questions fixed |
| **M9** | Text-to-SQL for financials | Tables → canonical `financials` in DuckDB; line-item synonym map; router: numeric → SQL, narrative → RAG, mixed → both | **Numeric/aggregation acc ≥ 90%** on the numeric subset; SQL validity ≥ 95% | Calculations become exact |
| **M10** | Embedding fine-tune | Synthetic Q-chunk pairs + hard negatives → fine-tune bge-base/Qwen3-0.6B (MRL) on MPS/Colab | Recall@10 ↑ ≥5 pts on **human** test split (not synthetic) | Domain vocab gains |
| **M11** | Agentic / corrective loop | LangGraph: classify → retrieve → grade (CRAG) → rewrite/re-route → generate → groundedness check (Self-RAG-lite), max 2 loops; exposed as an MCP server | Multi-hop subset acc ↑ ≥10 pts; **avg LLM calls/query ≤ 4**; p95 latency budget held | Robustness on hard Qs |
| **M12** | GraphRAG / RAPTOR (experiment) | LightRAG over circulars (amends/supersedes) + company–subsidiary graph; section/doc summary nodes | Global/sensemaking subset win-rate ≥ 60% vs M11 (pairwise judge) **and** no regression on factoids. Otherwise keep as an optional route | Usually niche, so keep it only if gated |
| **M13** | Multimodal (experiment) | ColQwen2.5/ColSmol over investor-presentation pages → Gemini Flash vision | Chart/infographic subset acc ↑ vs text-only | Covers chart-only facts |
| **M14** | Production hardening | Exact + entity-keyed semantic cache; LiteLLM fallback (Gemini→Groq→Ollama); Prompt Guard 2 + Presidio (PAN/Aadhaar); ACL filters; scheduled SEBI/RBI freshness crawl; DeepEval CI gates; Docker Compose; Streamlit UI with PDF-page highlight | Cache hit-rate on replayed log; 0 ACL leaks on adversarial tests; injection test-suite block rate; CI green | Ship-ready |

**Ablation table to publish (portfolio gold):** rows = M1…M11 cumulative, columns = recall@10, nDCG@5, context precision, faithfulness, numeric acc, citation precision, p95 latency, LLM calls/query. Also leave-one-out ablations (remove reranker, remove hybrid, etc.) to show which components matter.

---

## Quick decision cheatsheet
- Low recall@50 → fix parsing/chunking/metadata/hybrid/query rewriting (rerankers can't help).
- Good recall@50 but low precision@5 → reranker, filters, contextual headers.
- Good context but wrong answer → prompt contract, ordering, compression, smaller k, a better generator.
- Numbers wrong but right page retrieved → table representation + text-to-SQL + numeric validation.
- Global "themes across corpus" fails → RAPTOR summaries / LightRAG.
- Multi-step fails → decomposition → agentic loop (cap the iterations).
- Eval scores jump suspiciously → check judge self-preference, synthetic-set leakage, CI noise band.
