# 03 — Agentic AI Mastery via LangChain / LangGraph / LangSmith
### Research note + project design: "AI Portfolio Research Desk" (paper-trading only)

> Researched 2026-10-04. Versions checked: `langchain` 1.4.x (1.4.3, 2026-09-28), `langgraph` 1.2.x (1.2.12, 2026-09-21).
> APIs change fast. Each "verify" tag marks something worth re-checking against the docs before you code it.
> Free stack only: Gemini free tier, Groq, OpenRouter `:free`, Ollama on 16 GB Apple Silicon, plus Docker.

---

## 0. TL;DR verdict on the project

- **Keep the project idea.** It exercises nearly every concept on the list: fan-out, supervisor, HITL, memory, MCP, scheduling, evals and deploy.
- **Refinements (important):**
  1. **The core should be a workflow, with agents only at the leaves.** Universe → per-stock fan-out (`Send`) → pillar analysts → aggregate → construct → risk veto loop → HITL → publish. Only the pillar analysts and the "research question" path need open-ended ReAct autonomy. This is Anthropic's main advice: use the simplest pattern that works.
  2. **Make portfolio construction deterministic.** The LLM proposes views and tilts as structured output. A **convex optimizer** (`PyPortfolioOpt`/`cvxpy`) enforces the constraints. LLMs are bad at arithmetic and constraint satisfaction. The risk agent's hard checks are code, and the LLM only writes the narrative.
  3. **Lower token cost: cheap screen first, deep research second.** A quant screen narrows NIFTY 50 to roughly 10–15 names before the LLM pillars run, because free tiers (~1k req/day on Groq/Gemini) can't pay for 50 stocks × 5 pillars × ReAct loops every day.
  4. **Treat news and RSS as untrusted input.** Prompt injection is possible, so sanitise text and keep tool permissions to read-only.
  5. **Paper trading only, and the code enforces it.** No broker order tools are registered. The "execution" is a paper ledger in SQLite. The Angel One MCP server is mounted with a read-only tool allowlist.
  6. **Compliance note (India):** publishing buy/sell recommendations to *others* can fall under SEBI Research Analyst regulations. Keep the output personal/educational and add a disclaimer.

---

## 1. Foundations

| Concept | What | Why (failure it fixes) | Intuition | When NOT to use |
|---|---|---|---|---|
| **Tool calling** | The model emits a structured `tool_calls` list (name and JSON args). Your code executes it and returns a `ToolMessage`. | LLMs can't fetch live prices or do exact math, so without tools they hallucinate numbers. | The LLM is the planner. Tools are its hands. | When the next step is always the same. Just call the function in code. |
| **Structured output (Pydantic)** | The model is forced to return a schema-valid object (`model.with_structured_output(Schema)`; in agents `response_format=`). | Regex-parsing free text breaks, and later nodes need typed fields (score, confidence, citations). | A typed contract between nodes. | Long-form prose such as the final report narrative. Use plain text there. |
| **ReAct** | Loop of think → act (tool) → observe until done. | Single-shot answers miss multi-hop information needs. | A researcher who iterates. | When the steps are known up front. It wastes tokens and wanders. |
| **Plan-and-execute** | The planner writes a step list, executors run the steps (often with a cheaper model), and an optional re-planner revises it. | ReAct is short-sighted and uses expensive model calls on every step. | PM writes the plan, juniors execute. | Short tasks, or when plans go stale quickly. |
| **Reflection / Reflexion** | A critic critiques the output and the generator revises it, sometimes with memory of past failures. | First drafts have errors, unsupported claims and constraint violations. | Self code-review. | When there's no clear criterion. It loops without converging. Always cap the iterations. |
| **Workflow vs agent** | Workflow = code-defined path. Agent = the LLM chooses the path dynamically. | Agents are flexible but costly and less predictable. Workflows are predictable. | Railway vs taxi. | Don't build an agent when a workflow is enough. |

**Anthropic "Building effective agents" patterns (use as a vocabulary):**

| Pattern | Use in this project |
|---|---|
| Prompt chaining (gates between steps) | Report pipeline: draft → fact-check gate → format |
| Routing (classify → specialised path) | User query router: "research ticker" vs "rebalance" vs "explain the portfolio" |
| Parallelization (sectioning / voting) | Sectioning = 5 pillars in parallel. Voting = 3 sentiment samples, take the majority. |
| Orchestrator-workers (dynamic subtasks) | Research supervisor spawns N per-stock workers via `Send` |
| Evaluator-optimizer (loop until pass) | Risk agent vetoes → constructor revises (max 3 rounds) |
| Autonomous agent | Only the "free research question" path and the pillar internals |

Other principles from the post: keep it simple, make the planning transparent, and put real effort into the agent–computer interface (ACI), meaning tool names, docs and examples. Tool design matters as much as prompts.

---

## 2. LangChain v1 (current: 1.4.x)

**v1 changes:** `langchain` is now a slim package: `create_agent`, middleware, messages, models, tools, structured output. The old chains, `AgentExecutor`, legacy retrievers and so on moved to **`langchain-classic`**. Python ≥ 3.10. Messages carry `content_blocks`, a standard cross-provider content format.

| Building block | Current API | Why | Notes / when not |
|---|---|---|---|
| Models | `init_chat_model("google_genai:gemini-2.5-flash")`, `ChatGroq`, `ChatOllama`, `ChatOpenAI(base_url=openrouter)` | One interface for every provider, so swapping providers is a config change | Use `.with_fallbacks([...])` or `ModelFallbackMiddleware` to handle free-tier 429s |
| Prompts | `ChatPromptTemplate`, `MessagesPlaceholder` | Reusable, testable prompts; LangSmith prompt hub | Inside `create_agent`, use `system_prompt=` or `@dynamic_prompt` middleware |
| LCEL / Runnables | `prompt \| model \| parser`; `RunnableParallel`, `RunnableLambda`, `.batch()`, `.astream()` | Composable pipelines with free batching, streaming, async and tracing | **Not for branching or looping logic.** Use LangGraph for that. LCEL fits linear chains. |
| Tools | `@tool` decorator with typed args and a docstring; `StructuredTool`; `ToolRuntime` for injected state/context | The docstring and arg schema are the ACI the LLM reads | Keep each agent to ≤ ~10–15 tools, or use `LLMToolSelectorMiddleware` |
| Retrievers | `vectorstore.as_retriever()`; wrap as a tool for agentic RAG | Grounding | Wrap your sibling RAG project as a `@tool search_filings(query, ticker)` |
| Output parsers | `PydanticOutputParser`, `JsonOutputParser` | Fallback for models without native structured output (small Ollama models) | Prefer `with_structured_output` (native tool/JSON mode) |
| **`create_agent`** | `from langchain.agents import create_agent` → `create_agent(model, tools, system_prompt=, response_format=, middleware=[...], checkpointer=, store=, context_schema=)` | Production ReAct loop built on LangGraph, returns a compiled graph | **Replaces `langgraph.prebuilt.create_react_agent` (deprecated)** |
| Structured output in agents | `response_format=ToolStrategy(Schema)` or `ProviderStrategy(Schema)`; result in `result["structured_response"]` | Typed final answer | ToolStrategy works with any tool-calling model, which makes it the safer choice for Groq/Ollama |
| **Middleware** | Hooks: `before_agent`, `before_model`, `after_model`, `after_agent`, `wrap_model_call`, `wrap_tool_call`; decorators `@before_model`, `@wrap_tool_call`, `@dynamic_prompt` | Cross-cutting control without forking the agent loop | See the table below |

**Prebuilt middleware (verified on docs):** `HumanInTheLoopMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `ModelFallbackMiddleware`, `ModelRetryMiddleware`, `ToolRetryMiddleware`, `ToolErrorMiddleware`, `PIIMiddleware`, `TodoListMiddleware`, `LLMToolSelectorMiddleware`, `ContextEditingMiddleware`, `LLMToolEmulator` (tests), `FilesystemMiddleware`, `SubAgentMiddleware`, `RubricMiddleware` (LLM-judge self-check), `ShellToolMiddleware`, `ProviderToolSearchMiddleware`.

**Which middleware to use here:** `ModelCallLimitMiddleware` (budget), `ToolCallLimitMiddleware`, `ModelFallbackMiddleware` (Groq → Gemini → Ollama), `ToolRetryMiddleware` (yfinance flakiness), `SummarizationMiddleware` (long news), `HumanInTheLoopMiddleware` (any side-effect tool: email/Slack/ledger write). Write a custom `wrap_tool_call` that sanitises tool output against injection.

**Deep Agents** (`deepagents`, `create_deep_agent`) is a higher-level harness with planning, a virtual filesystem and subagents (the Claude Code style). Learn it **after** you can build the same thing by hand in LangGraph. It's good for the "free research question" path.

Refs: https://docs.langchain.com/oss/python/releases/langchain-v1 · https://docs.langchain.com/oss/python/langchain/agents · https://docs.langchain.com/oss/python/langchain/middleware/built-in

---

## 3. LangGraph (current: 1.2.x)

**Mental model:** a graph of functions over a shared typed state, executed in "supersteps" (Pregel/BSP). Every superstep can be checkpointed, which is where persistence, HITL, time travel and fault tolerance come from.

| Concept | API | Why / failure fixed | Gotchas |
|---|---|---|---|
| **StateGraph** | `StateGraph(State, context_schema=Ctx)`; `add_node`, `add_edge(START, "a")`, `compile()` | Explicit control flow that you can test and visualise | State = `TypedDict` (fast) or Pydantic (validated). Input/output schemas can differ. |
| **Reducers** | `Annotated[list, operator.add]`; `add_messages` | Parallel nodes writing the same key would otherwise overwrite each other | Without a reducer you get last-write-wins, or `InvalidUpdateError` on parallel writes |
| **Conditional edges** | `add_conditional_edges("risk", route_fn, {"revise":"construct","ok":"hitl"})` | Branching and loops | Routing functions should be pure and deterministic |
| **Send (map-reduce)** | `from langgraph.types import Send`; return `[Send("research_stock", {"ticker": t}) for t in shortlist]` | Dynamic fan-out over N items that aren't known at build time | Results need a reducer key. Rate-limit with `max_concurrency` in config to protect free tiers. |
| **Command** | `return Command(goto="node", update={...})`; `graph=Command.PARENT` jumps out of a subgraph | Lets a node update state *and* route in one step, which handoffs need | Annotate the return type `Command[Literal[...]]` so the graph renders correctly |
| **Subgraphs** | Add a compiled graph as a node (shared keys) or call it inside a node (state transform) | Encapsulation. Each pillar becomes its own testable graph with private state. | Streaming needs `subgraphs=True`. Each subgraph has its own checkpoint namespace. |
| **Checkpointers** | `InMemorySaver` (dev), `SqliteSaver`/`AsyncSqliteSaver` (`langgraph-checkpoint-sqlite`), `PostgresSaver` (`langgraph-checkpoint-postgres`), Redis/Mongo | Persistence, pause/resume, crash recovery | Requires `config={"configurable":{"thread_id":...}}`. Use SQLite locally and Postgres in Docker. |
| **Threads** | `thread_id` = one conversation/run history | Isolation between runs, e.g. `thread_id=f"rebalance-{date}"` | — |
| **Time travel** | `graph.get_state_history(cfg)`; re-invoke with a `checkpoint_id`; `graph.update_state(cfg, values)` to fork | Debug "what if the risk agent had seen X". Replay from any step without re-running the expensive earlier steps. | Replay re-executes the nodes after the fork, so their side effects run again. Make side effects idempotent. |
| **interrupt / HITL** | `from langgraph.types import interrupt, Command`; inside a node, `decision = interrupt({"proposal":...})`; resume with `graph.invoke(Command(resume={"approved":True}), cfg)` | Human approval, edits and rejection, hours or days later | **The node restarts from the top on resume**, so put side effects *after* `interrupt` or in a separate node. Static `interrupt_before=[...]` is for debugging. |
| **Streaming** | `stream_mode`: `values`, `updates`, `messages` (tokens), `custom` (`get_stream_writer()`), `checkpoints`, `tasks`, `debug`; `version="v2"` gives a uniform `{type, ns, data}` | UX: show per-stock progress and token streams | v1.2 adds a typed "event streaming" projection API (beta). Prefer `version="v2"` in new code. |
| **Short-term memory** | Checkpointer + `messages` state; trim or summarise | Conversation continuity inside a thread | Context bloat. Use `SummarizationMiddleware` or `trim_messages`. |
| **Long-term memory (Store)** | `InMemoryStore`/`PostgresStore`, `compile(store=)`; in a node use `runtime.store.aput/asearch(namespace, ...)`; `index={"embed":..., "dims":...}` for semantic search | Memory across threads: user risk preferences, past theses per ticker, "lessons learned" | Namespace it, e.g. `("user", uid, "prefs")`, `("ticker", "INFY", "theses")`. Kinds of memory: semantic (facts), episodic (past runs), procedural (prompt updates). |
| **Runtime context** | `context_schema=Ctx`; `runtime.context.user_id` | Static per-run deps (user id, model choice, DB handle) without putting them in state | Replaces the old `config["configurable"]` pattern for app context |
| **Durable execution** | Checkpointer + `durability="exit"\|"async"\|"sync"` on invoke/stream (verify); Functional API `@entrypoint` / `@task` | Survive crashes and long waits. Completed tasks aren't re-run on resume. | Wrap non-deterministic or side-effecting calls in `@task` or separate nodes, and make them idempotent |
| **Errors / retries** | `add_node(..., retry_policy=RetryPolicy(max_attempts=3))`; v1.2: per-node timeouts and node-level error handlers (Saga-style) | Flaky APIs (yfinance, RSS, free-tier 429s) | Retry transient errors only. Turn tool errors into messages so the LLM can recover. |
| **Node caching** | `add_node(..., cache_policy=CachePolicy(ttl=3600))` + `compile(cache=InMemoryCache())` | Avoid re-fetching the same fundamentals within a day | Cache key = node input |
| **Recursion limit** | `config={"recursion_limit": N}` → `GraphRecursionError`; `RemainingSteps` managed value lets a node degrade gracefully | Infinite agent/critic loops | Set it explicitly. Also put loop counters in state (e.g. `risk_rounds`). |
| **Studio / dev server** | `pip install "langgraph-cli[inmem]"`, `langgraph.json`, `langgraph dev`, which opens **LangSmith Studio** (renamed from LangGraph Studio) | Visual graph view, state inspection, interrupt/resume and time travel in a UI | `langgraph dev` is in-memory, no Docker. `langgraph up`/`build` produce a Docker Agent Server. |

**Multi-agent topologies:**

| Topology | How (2026) | Use when | Avoid when |
|---|---|---|---|
| **Supervisor (tool-calling)** | Supervisor agent whose tools are `call_fundamental_analyst(...)` etc. ("subagents as tools") | Central control, parallel specialists, clean context per subagent | Latency-critical chat. Every hop passes through the supervisor. |
| **Swarm / handoffs** | Agents return `Command(goto=other_agent, graph=Command.PARENT)` via handoff tools | Conversational, where the active specialist talks to the user directly | Strict pipelines. Handoffs are harder to audit. |
| **Hierarchical** | Supervisor of supervisors (subgraphs) | Large teams, e.g. Research team + Portfolio team | Small systems, where it's overhead |
| **Network** | Any agent → any agent | Rarely. Research prototypes. | Production. Loops are hard to bound. |
| **Custom workflow** | Hand-built StateGraph with deterministic edges plus agent nodes | **The default for this project** | — |

**Library status:** `langgraph-supervisor` / `langgraph-swarm` were convenience wrappers. The supervisor repo was **archived (Sep 2026)**, and the JS versions don't work with v1 `createAgent`. The docs now recommend building the **supervisor pattern yourself via tools** (more control over context). Learn the ideas and hand-roll them; don't depend on the libraries.

Refs: https://docs.langchain.com/oss/python/langgraph/overview · https://docs.langchain.com/oss/python/langchain/multi-agent · https://docs.langchain.com/oss/python/migrate/langgraph-v1 · https://academy.langchain.com (free "Introduction to LangGraph", "Ambient Agents", "Deep Research with LangGraph", "Deep Agents")

**v1 deprecations/removals to know:** `create_react_agent` → `create_agent`. `langgraph.prebuilt.AgentState*` → `langchain.agents.AgentState`. `HumanInterrupt*` → `langchain.agents.middleware.human_in_the_loop.HITLRequest`/`InterruptOnConfig`. `MessageGraph` and `ValidationNode` **removed**. Python 3.9 dropped. Many 2024-era tutorials use the removed APIs, so be careful which ones you follow.

---

## 4. MCP and A2A

| Item | Current state | Why | When not |
|---|---|---|---|
| **MCP server (FastMCP)** | `from fastmcp import FastMCP`; `@mcp.tool`, `@mcp.resource`, `@mcp.prompt`; transports stdio / Streamable HTTP | Write a tool once and use it from Claude, Cursor, LangGraph or anything else. You already know this. | Tools used by a single in-process agent: a plain `@tool` is simpler |
| **MCP in LangChain: new** | **`langchain.mcp.MCPAdapter`** (`pip install "langchain[mcp]>=1.4"`, beta), built on FastMCP: `async with MCPAdapter(target) as a: tools = await a.list_tools()`. Transport is inferred. MCP elicitation surfaces as a LangGraph `interrupt()`. | One entry point, caching, HITL bridging | Beta. Pin versions. |
| **MCP in LangChain: legacy** | `langchain-mcp-adapters`: `MultiServerMCPClient({...}).get_tools()`, `tool_interceptors=` | Most tutorials use it. Still works. | Migration guide exists. Prefer `MCPAdapter` for new code. |
| **Exposing the desk as MCP** | FastMCP server wrapping the LangGraph SDK client (`run_research(universe)`, `get_latest_report()`, `approve_proposal(id)`). The Agent Server also exposes an MCP endpoint for deployed graphs (verify). | Claude and other agents can call your desk | Never expose approval tools to untrusted clients. Gate them with auth. |
| **A2A** | Agent2Agent protocol: **v1.0 spec (Apr 2026)**, now under the Linux Foundation **Agentic AI Foundation** alongside MCP (Aug 2026). Agent Cards for discovery, task lifecycle, streaming. | Agent ↔ agent across vendors/frameworks (MCP is agent ↔ tool). Supported by ADK and LangGraph Agent Server (verify). | Inside one codebase/graph, where subgraphs/tools suffice. A2A is useful only when **separately owned agents** must interoperate. Learn it as a stretch goal: expose the risk agent as an A2A service. |

Refs: https://modelcontextprotocol.io · https://gofastmcp.com · https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters · https://a2a-protocol.org

---

## 5. Alternative frameworks: when to pick them

| Framework | Sweet spot | Why not the default here | Pick it when |
|---|---|---|---|
| **LangGraph** | Explicit stateful graphs, persistence, HITL, durable execution, mature tooling | Verbose, steep learning curve | Production control and auditability, which is what you want to master |
| **CrewAI** | Role/task "crews", fast prototypes; Flows for deterministic paths | Less control over state and checkpoints, and harder to debug at scale | Quick demos and role-play style teams |
| **AutoGen → Microsoft Agent Framework / AG2** | Conversational multi-agent. AutoGen is in **maintenance mode (Oct 2025)**. **MS Agent Framework GA Apr 2026** (AutoGen + Semantic Kernel). AG2 is the community fork. | Lineage churn; Azure-centric | .NET/Azure shops |
| **OpenAI Agents SDK** | Minimal primitives (agents, handoffs, guardrails, sessions, tracing) | OpenAI-centric (supports other models via LiteLLM). Less graph control. | Simple handoff apps on OpenAI |
| **Google ADK** | Gemini-native, multi-agent, A2A-first, Vertex deploy | Google-centric ecosystem | Gemini-heavy, A2A interop, GCP deploy |
| **PydanticAI** | Type-safe agents, DI, structured output, `pydantic-graph`, Logfire | Smaller ecosystem for persistence and HITL (improving) | Type-safety-first services. Good second framework. |
| **LlamaIndex Workflows** | Event-driven steps; best-in-class RAG ingestion and indexing | Agent orchestration is less mature than LangGraph | RAG-heavy pipelines. You could build the sibling RAG project on it. |
| **smolagents (HF)** | Code-agents (the LLM writes Python), tiny codebase | Code execution needs a sandbox, and it's thin for production | Learning, or local open models doing math or data work |
| **DSPy** | *Programming, not prompting*: signatures + optimizers (MIPROv2, GEPA) tune prompts and few-shots against a metric | Not an orchestrator | **Combine with LangGraph:** optimise one node's prompt, e.g. the sentiment classifier, on a labelled set. Stretch milestone. |

Rule of thumb: you're learning *concepts* (state, checkpoints, handoffs, evals), and these transfer across frameworks. Master LangGraph, then rebuild one pillar in PydanticAI or ADK to see the trade-offs.

---

## 6. AI automations and workflow orchestration

| Tool | Type | Free? | Use when | Not when |
|---|---|---|---|---|
| **cron / launchd** | OS scheduler | Yes | One machine, single job | Retries, observability, multi-step deps, missed runs while the laptop sleeps |
| **n8n (self-host)** | Low-code workflow plus AI nodes | Community Edition free, unlimited executions, under the **Sustainable Use License** (fair-code; fine for internal use, can't resell as a hosted service) | Glue: schedule → call desk API → Slack/email/Sheets; webhooks; approval buttons; non-devs editing flows | Core agent logic. Keep that in code, under version control and tests. |
| **Make / Zapier** | Hosted no-code | Small free tiers (Zapier ~100 tasks/mo) | Quick SaaS-to-SaaS glue | Self-hosting, volume, privacy |
| **Prefect (OSS)** | Python-native orchestration | OSS server free; Prefect Cloud has a free tier | Python data pipelines with retries, schedules, caching and a UI. Good for the **data ingestion layer**. | Sub-second event-driven work |
| **Temporal (OSS)** | Durable workflow engine | OSS self-host free (Docker) | Mission-critical, long-running, exactly-once business processes | Learning projects. LangGraph checkpoints already cover agent durability. Overkill on 16 GB. |
| **LangGraph Agent Server cron** | Built-in cron jobs on deployed assistants | Self-hosted Lite free tier (verify limits) | Scheduling the graph itself | When you're not on the Agent Server |
| **GitHub Actions schedule** | CI cron | Free minutes | Laptop-independent daily runs (secrets in repo settings) | Long jobs; local Ollama |

**Recommendation:** run **Prefect** for data ingestion (prices, RSS, fundamentals snapshots into DuckDB/SQLite) and **n8n** (Docker) for event glue: schedule trigger → HTTP POST to the desk API → wait for webhook → Slack message with Approve/Reject buttons → resume the graph via `Command(resume=...)`. That teaches both code-first and no-code automation. Cron remains the fallback.

Triggers to implement: **schedule** (daily 08:30 IST pre-market), **event** (VIX spike or a big drawdown in a holding → ad-hoc risk review), **webhook** (Slack approval callback), **manual** (Studio / UI).

Refs: https://docs.n8n.io/hosting/ · https://docs.prefect.io · https://docs.temporal.io

---

## 7. Evaluation and observability

### Concepts
| What | Why | How |
|---|---|---|
| **Tracing** | You can't debug a 200-call multi-agent run from logs | Set `LANGSMITH_TRACING=true` plus the key env var; everything auto-traces. Add metadata/tags such as `ticker` and `pillar`. |
| **Datasets** | Regression tests for non-deterministic systems | Curate 30–100 examples: (ticker, date-frozen data snapshot) → expected properties |
| **Evaluators** | Score outputs | Code checks (schema valid, weights sum to 1, constraints held), LLM-as-judge (`openevals`), pairwise |
| **Experiments** | Compare prompt/model/graph versions on the same dataset | `client.evaluate(target, data=, evaluators=)` (or `langsmith.evaluate`), compared in the UI |
| **Trajectory evals** | Agents can get the right answer by the wrong path, or get the wrong answer with a sane path | `agentevals`: `create_trajectory_match_evaluator(trajectory_match_mode="strict\|unordered\|subset\|superset")`, `create_trajectory_llm_as_judge`; graph trajectories for LangGraph |
| **Tool-call accuracy** | Wrong tool or wrong args is the #1 agent bug | Assert the tool name and args for unit-style cases. Measure precision and recall of tool calls. |
| **Online evals** | Prod drift | LangSmith rules on sampled traces; Langfuse evaluators |
| **Annotation queues** | Human labels become dataset growth | Route low-confidence or vetoed runs to the queue, review weekly |
| **Multi-turn simulation** | Chat agents | `openevals` simulated user, `run_multiturn_simulation` |
| **Cost / latency** | Free-tier budgets | Token counts per node in the trace; p50/p95 latency per pillar; tokens per stock |
| **Pytest-style** | Evals in CI | LangSmith pytest/vitest integration (`@pytest.mark.langsmith`) |

### Tool choice
| Tool | Free tier | Pick when | Why not |
|---|---|---|---|
| **LangSmith** | Developer plan: **1 seat, 5k base traces/mo, 14-day retention** | Best LangGraph integration (Studio, graph-aware traces, datasets, experiments, annotation queues) | 5k traces runs out fast with multi-agent runs. Sample them (`LANGSMITH_SAMPLING_RATE`) and trace mainly in evals. |
| **Langfuse (self-host)** | **MIT core, unlimited**: tracing, evals, LLM-judge, annotation queues, prompts, playground. Docker Compose (Postgres + ClickHouse + Redis; heavy on 16 GB) | Unlimited local tracing, OSS, OpenTelemetry | Less LangGraph-native UX |
| **Arize Phoenix** | OSS self-host (single container, lighter) | OTel/OpenInference tracing + evals, notebook-friendly | Smaller eval-workflow surface |

**Recommendation:** use **LangSmith for development and evals**. It's what LangChain jobs expect, and Studio needs it. Add **Phoenix or Langfuse** for unlimited local tracing of the daily scheduled runs. Learning both shows you understand vendor-neutral OTel.

Refs: https://docs.langchain.com/langsmith/evaluation · https://github.com/langchain-ai/agentevals · https://github.com/langchain-ai/openevals · https://langfuse.com/self-hosting · https://arize.com/docs/phoenix

---

## 8. Safety and guardrails

| Risk | Control | Implementation here |
|---|---|---|
| Real-money side effects | **Capability removal** beats prompts | No order-placement tools exist at all. Angel One MCP mounted with an explicit **allowlist** of read-only tools. The paper ledger is the only "execution". |
| Prompt injection via news/RSS/filings | Treat tool output as data | Custom `wrap_tool_call` middleware: strip HTML/scripts, truncate, wrap in `<untrusted_source>` tags, never let news text choose tools. News agent has **no** side-effect tools ("lethal trifecta": private data + untrusted content + exfiltration channel. Break one leg). |
| Tool permissioning | Least privilege per agent | Each pillar gets only its own tools; email/Slack only in the publisher node; `HumanInTheLoopMiddleware(interrupt_on={"send_email": True, "post_slack": True})` |
| Runaway loops / cost | Budgets | `ModelCallLimitMiddleware(run_limit=...)`, `ToolCallLimitMiddleware`, `recursion_limit`, `risk_rounds ≤ 3`, `max_concurrency` on fan-out, per-day token budget in the Store |
| Kill switch | Global stop | Feature flag file/env `DESK_ENABLED=false` checked at START and before side effects; n8n workflow disable; v1.2 `RunControl` graceful shutdown (verify) |
| Hallucinated numbers | Ground the numbers | Every number in the report must come from a tool result: post-check by regex-extracting numbers and matching them to the state; citations required in the schema |
| Code execution (if smolagents/pandas agent) | Sandbox | Docker container with no network, or skip. Prefer predefined analytics tools. |
| PII / secrets | Redaction | `PIIMiddleware`; secrets only in `.env`/keychain; never in state (state is checkpointed!) |
| Regulatory | Disclaimer and scope | "Educational / paper-trading, not investment advice"; don't distribute recommendations publicly |

Refs: https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/ · https://genai.owasp.org/llm-top-10/ · https://docs.langchain.com/oss/python/langchain/guardrails

---

## 9. Deployment and UI

| Option | Free? | Use |
|---|---|---|
| `langgraph dev` | Yes (in-memory, hot reload) | Dev + Studio |
| **Agent Server** (formerly LangGraph Server) via `langgraph build`/`langgraph up` → Docker with Postgres + Redis | Self-hosted Lite/standalone: free up to a usage cap (~1M node executions, verify) and needs a LangSmith key | Prod-like: threads, runs, assistants, cron, streaming, double-texting, background runs, `langgraph_sdk` client |
| **LangSmith Deployment** (formerly LangGraph Platform, Cloud) | Paid (Plus+) | Skip, or use a trial only |
| **FastAPI wrapper** | Yes | Own the API: `/research`, `/proposals/{id}/approve`; compile the graph with `AsyncPostgresSaver`; SSE streaming. Shows you understand what the Agent Server does for you. |
| Docker Compose | Yes | `api` + `postgres` + `n8n` + `phoenix` + optional `ollama` (or native Ollama on macOS for Metal GPU: **run Ollama natively, not in Docker, on Apple Silicon**) |
| Free hosting | Fly.io/Render/Railway free or low tiers, Oracle Cloud Always Free ARM (verify current) | Public demo (without Ollama) |

**UI options:**
| UI | Pick when |
|---|---|
| **LangSmith Studio** | Dev and debugging the graph (not end-user) |
| **agent-chat-ui** (langchain-ai, Next.js) | Chat front-end for any Agent Server graph, supports interrupts out of the box |
| **Streamlit** | Dashboards: portfolio weights, pillar scorecards, approve button. **Best fit for this project's main UI.** |
| **Chainlit** | Chat-first with step visualisation and LangChain callbacks |
| **Gradio** | Quick demos, HF Spaces hosting |

Refs: https://docs.langchain.com/langsmith/cli · https://docs.langchain.com/langsmith/agent-server · https://github.com/langchain-ai/agent-chat-ui

---

## 10. Free LLM strategy (16 GB Apple Silicon)

| Role | Model | Why |
|---|---|---|
| Supervisor / constructor narrative / judge | Gemini 2.5 Flash (free tier; **check your project's quotas in AI Studio**: roughly 10 RPM, a few hundred to ~1.5k RPD) | Best free quality, long context, native structured output |
| Pillar workers (high volume) | Groq: `llama-3.1-8b-instant` (~14.4k RPD), `gpt-oss-20b/120b`, Qwen (~1k RPD each) | Very fast, generous daily limits |
| Fallback / offline / evals at volume | Ollama: Qwen3 4B/8B Q4, Gemma 3 4B, Llama 3.2 3B (≈3–6 GB RAM) | Unlimited, private. Weaker tool calling, so use ToolStrategy and small tool sets. |
| Extra free pool | OpenRouter `:free` models (~20 RPM, 50 RPD, 1k RPD after a one-time $10 credit) | Rotation and fallback |

Use `ModelFallbackMiddleware` and `.with_fallbacks()`. Use LangGraph node caching plus a daily data snapshot so re-runs don't spend tokens. Embeddings: local `nomic-embed-text`/`bge-small` via Ollama, or Gemini embeddings on the free tier.

---

## 11. Free data sources (realistic, 2026)

| Data | Source | Legality / reliability notes |
|---|---|---|
| Prices OHLCV (daily) | **Angel One SmartAPI** (official, free for account holders), **yfinance** (`RELIANCE.NS`) | SmartAPI is the most legitimate. yfinance is unofficial Yahoo scraping with no SLA and occasionally breaks, so cache it. |
| NSE bhavcopy, index constituents, corporate actions | NSE archives / `jugaad-data` | **NSE Terms of Use prohibit automated scraping without consent.** Personal research at very low frequency with caching is a grey area. Never redistribute. Prefer the broker API where possible. |
| India VIX, FII/DII flows | NSE pages (same caveat), broker API | Low-frequency manual/cached pulls |
| Fundamentals | yfinance `.financials` (patchy for India); **Screener.in manual CSV export** (free account, *don't scrape*); company annual reports/filings PDFs from NSE/BSE announcements, which go into your **RAG project** | Screener ToS restricts scraping. Exports for personal use are fine. |
| News | RSS: Economic Times Markets, Moneycontrol, Livemint, Business Standard, Google News RSS query per ticker; **GDELT** (free, global) | RSS is intended for consumption. Store headlines, links and snippets, not full articles. |
| Macro / regime | RBI DBIE, data.gov.in OGD API (free key), FRED API (free key; US rates, DXY, crude), yfinance for `^NSEI`, `^INDIAVIX`, `USDINR=X`, `CL=F` | Official, free |
| Fixed reference | NIFTY 50 constituents CSV (niftyindices.com) refreshed monthly | Manual download |

Design rule: the **Data layer is code (Prefect flows → DuckDB snapshot per date)**, and agents only read snapshots via tools. You get reproducible evals (frozen as-of dates), no rate-limit pain in loops, and protection against look-ahead bias.

---

## 12. Project design: AI Portfolio Research Desk

### 12.1 Agent / node roster
| Node | Kind | Pattern | Model | Responsibility | Tools |
|---|---|---|---|---|---|
| `load_universe` | code | — | — | Resolve universe (NIFTY 50 / custom), load user prefs from Store | Store, constituents CSV |
| `data_snapshot` | code | — | — | Ensure today's DuckDB snapshot exists (else trigger the Prefect flow) | DuckDB |
| `quant_screen` | code | — | — | Momentum/quality/value/vol z-scores → shortlist K (10–15) + current holdings | pandas |
| `regime_agent` | agent (small) | prompt chain | Gemini Flash | Macro/regime label (risk-on/off, rate/FX/VIX context), structured `Regime` | macro tools, RAG (RBI policy docs) |
| `fan_out` | edge | **orchestrator-workers / Send** | — | `Send("stock_research", {ticker, regime})` per shortlisted stock | — |
| `stock_research` (subgraph) | workflow | **parallelization (sectioning)** | — | Runs 4 pillars in parallel for one ticker, then `synthesize` | — |
| ├ `fundamental_analyst` | `create_agent` | ReAct | Groq 70B/120B | Valuation, growth, quality, red flags → `PillarView` | fundamentals tool, **RAG search_filings** (agentic RAG) |
| ├ `technical_analyst` | code + LLM | chain | Groq 8B | Indicators computed in code; LLM interprets → `PillarView` | indicators tool |
| ├ `news_sentiment` | agent | parallelization (**voting** ×3) | Groq 8B / Ollama | Headline sentiment and event detection, injection-hardened | RSS/GDELT tool (read-only) |
| ├ `stock_risk` | code + LLM | chain | Groq 8B | Vol, beta, drawdown, liquidity, event risk → `PillarView` | risk metrics tool |
| └ `synthesize` | LLM | structured output | Gemini Flash | Merge pillars → `StockThesis` (score −2..+2, confidence, citations) | — |
| `aggregate` | reducer | map-reduce | — | `theses: Annotated[list[StockThesis], add]` | — |
| `pm_constructor` | LLM + optimizer | **tool use, deterministic core** | Gemini Flash | LLM turns theses into views/tilts; `PyPortfolioOpt` (Black-Litterman or mean-variance with constraints) computes weights | optimizer tool |
| `risk_manager` | code checks + LLM | **evaluator-optimizer** | Gemini Flash | Hard limits (max weight 10%, sector ≤ 30%, beta band, turnover, liquidity) in code; LLM writes the veto rationale; `Command(goto="pm_constructor")` if fail and rounds < 3 | risk tools |
| `human_approval` | node | **HITL interrupt** | — | `interrupt(proposal)` → approve / edit weights / reject | — |
| `paper_execute` | code | — | — | Write to the paper ledger (idempotent by `run_id`) | SQLite ledger |
| `report_writer` | LLM | prompt chain + **reflection** | Gemini Flash | Markdown/HTML report; critic checks that numbers match state and the disclaimer is present | — |
| `publish` | code | side effect (HITL-gated) | — | Email (SMTP/Gmail) and Slack webhook | email, slack |
| `memory_update` | LLM | — | small | Write episodic memory (theses, outcome later) and lessons to the Store | Store |
| `research_chat` (separate graph) | `create_agent` / deep agent | **routing + supervisor-as-tools** | Gemini | Ad-hoc Q&A: "why is TCS underweight?" Reads state, Store, RAG | read-only tools |

### 12.2 State schema sketch
```python
class PillarView(BaseModel):
    pillar: Literal["fundamental","technical","news","risk"]
    score: float = Field(ge=-2, le=2); confidence: float = Field(ge=0, le=1)
    key_points: list[str]; citations: list[str]; data_asof: date

class StockThesis(BaseModel):
    ticker: str; pillars: list[PillarView]; composite: float
    view: Literal["overweight","neutral","underweight"]; rationale: str

class Proposal(BaseModel):
    weights: dict[str, float]; expected_vol: float; turnover: float
    constraint_report: dict[str, bool]; narrative: str

class DeskState(TypedDict):
    run_id: str; asof: date; universe: list[str]; shortlist: list[str]
    regime: Regime | None
    theses: Annotated[list[StockThesis], operator.add]      # fan-in reducer
    proposal: Proposal | None; risk_feedback: list[str]; risk_rounds: int
    approval: Literal["pending","approved","edited","rejected"]
    report_md: str | None; errors: Annotated[list[str], operator.add]
    messages: Annotated[list[AnyMessage], add_messages]      # for chat path

class Ctx(TypedDict):  # context_schema (runtime, not checkpointed)
    user_id: str; model_tier: Literal["free","local"]; dry_run: bool
```

### 12.3 Topology
```
START
  │
load_universe ─► data_snapshot ─► quant_screen ─► regime_agent
                                                     │
                              ┌────── Send(ticker) × K (max_concurrency=4) ──────┐
                              ▼                      ▼                            ▼
                     ┌─ stock_research subgraph (per ticker) ──────────────────────┐
                     │  START ─┬─► fundamental_analyst (ReAct + RAG tool) ─┐       │
                     │         ├─► technical_analyst ──────────────────────┤       │
                     │         ├─► news_sentiment (vote×3, sanitized) ─────┼─► synthesize ─► END
                     │         └─► stock_risk ─────────────────────────────┘       │
                     └──────────────────────────────────────────────────────────────┘
                              │ (theses reducer: operator.add)
                              ▼
                        aggregate ─► pm_constructor ◄──────────┐ revise (rounds<3)
                                          │                    │
                                          ▼                    │
                                    risk_manager ──── veto ────┘
                                          │ pass (or rounds==3 → flag)
                                          ▼
                                  human_approval  ⟵ interrupt()  (Slack button / Streamlit / Studio)
                               approve/edit │      reject → memory_update → END
                                          ▼
                                   paper_execute ─► report_writer ⇄ critic (≤2) ─► publish (HITL-gated)
                                                                                   │
                                                                             memory_update ─► END

Triggers: n8n schedule (08:30 IST) / event (VIX spike) ─► POST /runs ─► graph
Exposure: FastMCP server ─► {run_research, get_report, get_proposal, explain_position}  (approve NOT exposed)
```

### 12.4 Why each pattern, and what you could have done instead
- **Deterministic spine + agent leaves.** Predictable cost and auditability. Rejected alternative: a fully autonomous supervisor that decides everything, which is non-reproducible and burns free-tier quotas.
- **`Send` fan-out.** The shortlist is dynamic. Rejected alternative: a supervisor calling 15 subagents sequentially, which is slow.
- **Pillars as a subgraph.** Unit-testable per ticker and reusable.
- **Optimizer in code.** Hard constraints must be guaranteed. The LLM's job is views and narrative.
- **Evaluator-optimizer for risk.** Classic generate/critique, bounded by `risk_rounds`.
- **HITL before side effects.** Durable pause, so you can approve from your phone hours later.
- **Supervisor-as-tools for chat.** Recommended in the 2026 docs over the archived `langgraph-supervisor`.

---

## 13. Milestones with eval gates

Each milestone produces a git tag, a short README section, and **a LangSmith experiment that must pass its gate before moving on**.

| # | Milestone | Learn | Deliverable | **Eval gate** |
|---|---|---|---|---|
| M0 | Setup | uv, env, providers, tracing | `uv` project, `.env.example`, `init_chat_model` for Gemini/Groq/Ollama, LangSmith tracing on | One trace per provider visible. Smoke test passes. |
| M1 | Basic chain | Prompts, LCEL, structured output | `news → SentimentView` chain (Pydantic) | 30-headline labelled set: schema validity 100%, accuracy ≥ 75% vs labels; compare 3 models in one experiment |
| M2 | Single tool agent | `@tool`, `create_agent`, middleware | Fundamental analyst ReAct agent with price/fundamentals/RAG tools + `ToolRetry` + `ModelCallLimit` | 20 questions: tool-selection accuracy ≥ 90% (`agentevals` subset match); numbers in the answer match tool outputs (code check) 100%; avg ≤ 6 model calls |
| M3 | Graph | StateGraph, reducers, conditional edges, retries, caching | Single-ticker pillar subgraph (4 pillars parallel → synthesize) | 15 tickers on frozen as-of dates: 0 crashes, all `PillarView` valid, p95 latency < 60 s; LLM-judge "rationale supported by citations" ≥ 0.8 |
| M4 | Multi-agent | Send map-reduce, subgraphs, Command, evaluator-optimizer | Full desk to proposal (no HITL yet) | 10 historical dates: constraint satisfaction 100% (code), risk loop converges ≤ 3 rounds in ≥ 90%, graph-trajectory match (expected node sequence) 100%; tokens per run under budget |
| M5 | HITL + persistence | Checkpointers (SQLite → Postgres), interrupt, Command(resume), time travel | Approve/edit/reject flow in Studio + Streamlit; kill the process mid-run and resume | Resume-after-crash test passes; edit-weights path re-validated by risk; replay from the `risk_manager` checkpoint reproduces the decision; no side effect before approval (assert ledger empty) |
| M6 | Memory | Store, namespaces, semantic search, runtime context | User prefs (risk tolerance, exclusions), per-ticker thesis history, "lessons" from rejections | Memory recall eval: 15 cases where a stored pref must change output (e.g. exclusion list honoured 100%); no cross-user leakage test |
| M7 | MCP | FastMCP server, `MCPAdapter`, allowlists | (a) mount an existing market-data MCP server read-only into pillars; (b) expose the desk as an MCP server, used from Claude Code/Desktop | Contract tests for each MCP tool; a Claude client completes 5 scripted tasks; attempting a non-allowlisted tool fails closed |
| M8 | Automation | n8n, Prefect, webhooks, schedules | Prefect data flow 08:00, n8n 08:30 trigger → run → Slack approval buttons → resume → email report | 5 consecutive scheduled runs succeed; approval round-trip via webhook works; kill switch halts within one step; idempotent re-trigger (no duplicate ledger entries) |
| M9 | Evals & observability | Datasets, experiments, online evals, annotation queue, Phoenix/Langfuse | Golden dataset (≥ 50), CI eval job (pytest + LangSmith), injection red-team set | CI blocks regressions (score drop > 5%); injection set: 0 tool calls triggered by injected text; report numeric-grounding 100%; paper-portfolio backtest dashboard vs NIFTY 50 (informational, not a gate) |
| M10 | Deploy | Agent Server (Docker), FastAPI alt, UIs | `docker compose up`: api/agent-server + postgres + n8n + phoenix; Streamlit + agent-chat-ui | Fresh-clone deploy in < 15 min following the README; load test 3 concurrent runs; trace/cost dashboard screenshot |
| Stretch | DSPy / A2A / 2nd framework | Prompt optimisation, interop | DSPy-optimised sentiment node; risk agent exposed via A2A; one pillar rebuilt in PydanticAI | DSPy lifts the M1 metric by ≥ 5 pts; A2A agent card discoverable and callable |

**Industry-grade signals to show in the repo:** typed state, tests (unit for tools, graph-level with a fake LLM via `GenericFakeChatModel`/`LLMToolEmulator`), eval CI, ADRs explaining pattern choices (§12.4), cost table, threat model (§8), reproducible data snapshots, and a demo video.

---

## 14. Suggested repo layout
```
desk/
  graphs/        desk_graph.py  stock_research.py  chat_graph.py
  agents/        fundamental.py technical.py news.py risk.py constructor.py
  tools/         market.py fundamentals.py news_rss.py rag_tool.py optimizer.py notify.py
  middleware/    sanitize_untrusted.py budget.py
  state.py  context.py  config.py
  data/          prefect_flows.py  snapshot.py (DuckDB)
  mcp_server/    server.py (FastMCP)
  api/           fastapi_app.py
  ui/            streamlit_app.py
evals/           datasets/  evaluators.py  test_graph.py  redteam_injection.jsonl
langgraph.json   docker-compose.yml   n8n/workflows/*.json   docs/adr/*.md
```

---

## 15. Learning sequence (concept → resource)
1. Anthropic "Building effective agents" (patterns vocabulary). Then LangChain Academy **Introduction to LangGraph** (free).
2. LangChain v1 docs: Agents, Middleware, Structured output → M1–M2.
3. LangGraph docs: Graph API, Persistence, Interrupts, Streaming, Memory, Durable execution → M3–M6.
4. Academy **Ambient Agents with LangGraph** (HITL, memory, evals; closest to M5/M6/M9) and **Deep Research with LangGraph** (multi-agent fan-out; closest to M4).
5. LangSmith evaluation docs + `agentevals`/`openevals` READMEs → M9 (start the eval habit at M1, though).
6. MCP spec + FastMCP docs + `MCPAdapter` migration doc → M7.
7. Academy **Introduction to Deep Agents** once you have built the desk by hand.

---

## 16. Verified references (by area)

**Foundations / patterns**
- https://www.anthropic.com/engineering/building-effective-agents
- https://www.anthropic.com/engineering/writing-tools-for-agents
- https://react-lm.github.io/ (ReAct paper) · https://arxiv.org/abs/2303.11366 (Reflexion)

**LangChain v1**
- https://docs.langchain.com/oss/python/releases/langchain-v1
- https://docs.langchain.com/oss/python/langchain/agents
- https://docs.langchain.com/oss/python/langchain/middleware/built-in

**LangGraph**
- https://docs.langchain.com/oss/python/langgraph/overview
- https://docs.langchain.com/oss/python/migrate/langgraph-v1
- https://docs.langchain.com/oss/python/langgraph/streaming · https://docs.langchain.com/oss/python/langgraph/add-memory
- https://docs.langchain.com/oss/python/langchain/multi-agent
- https://academy.langchain.com/courses/ambient-agents · https://academy.langchain.com/courses/deep-research-with-langgraph

**MCP / A2A**
- https://modelcontextprotocol.io · https://gofastmcp.com
- https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters · https://github.com/langchain-ai/langchain-mcp-adapters
- https://a2a-protocol.org · https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year

**Alternatives**
- https://docs.crewai.com · https://ai.pydantic.dev · https://google.github.io/adk-docs/
- https://openai.github.io/openai-agents-python/ · https://dspy.ai · https://huggingface.co/docs/smolagents
- https://learn.microsoft.com/agent-framework/ · https://docs.llamaindex.ai/en/stable/module_guides/workflow/

**Automation**
- https://docs.n8n.io/hosting/ · https://docs.n8n.io/sustainable-use-license/
- https://docs.prefect.io · https://docs.temporal.io

**Evals / observability**
- https://docs.langchain.com/langsmith/evaluation · https://www.langchain.com/pricing
- https://github.com/langchain-ai/agentevals · https://github.com/langchain-ai/openevals
- https://langfuse.com/self-hosting · https://arize.com/docs/phoenix

**Safety**
- https://genai.owasp.org/llm-top-10/
- https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
- https://docs.langchain.com/oss/python/langchain/human-in-the-loop

**Deployment / UI**
- https://docs.langchain.com/langsmith/cli · https://docs.langchain.com/langsmith/agent-server
- https://docs.langchain.com/langsmith/deploy-standalone-server
- https://github.com/langchain-ai/agent-chat-ui · https://docs.streamlit.io · https://docs.chainlit.io

**Data / legal**
- https://www.nseindia.com/static/nse-terms-of-use (automated scraping prohibited without consent)
- https://smartapi.angelbroking.com/docs · https://github.com/ranaroussi/yfinance · https://github.com/jugaad-py/jugaad-data
- https://ai.google.dev/gemini-api/docs/rate-limits · https://console.groq.com/docs/rate-limits · https://openrouter.ai/docs/api-reference/limits

---

## 17. Open items to verify when coding (fast-moving)
- Exact `durability=` parameter name/values and the v1.2 node error-handler / timeout / `RunControl` APIs.
- `MCPAdapter` method names (reference shows `list_tools()`; some docs show `get_tools()`) and beta status.
- Agent Server self-hosted free tier limits and the license-key requirement; whether Agent Server exposes MCP/A2A endpoints on your version.
- Default `recursion_limit` in your installed LangGraph version (set it explicitly regardless).
- Current Gemini/Groq/OpenRouter free quotas (per-project; they change often).
- Ollama tool-calling reliability for your chosen 4–8B model (run the M2 gate on it before relying on it).
