# DevDocs AI — Complete Master Prompt
### Copy everything below this line and paste into a new chat

---

## WHO I AM

I am Jay Thadeshwar — an AI/ML Engineer with 1 year of production experience.

**Current stack I already know:**
- LLMs: LLaMA models, RAG with BAAI/bge-m3 embeddings, prompt engineering
- Vision: DINOv2 image similarity search, vector search with FAISS/Pinecone
- MLOps: FastAPI microservices, Docker, GCP, GitHub Actions CI/CD
- Frontend: Vue.js, Gradio
- Languages: Python, JavaScript, Bash

**My goal:** Build a production-grade, fully deployable AI project that gets me hired at ₹18–25 LPA+ and makes recruiters say "remarkable resume."

---

## THE PROJECT — DevDocs AI

**What it is:**
A production-grade developer documentation intelligence system. Users paste any public GitHub repo URL, ask questions about it in natural language, and get precise, cited, hallucination-scored answers — powered by a multi-stage retrieval pipeline with agentic reasoning and a self-evaluation loop.

**Why it's not basic RAG:**
- Hybrid BM25 + dense retrieval with RRF fusion (not naive cosine search)
- BGE cross-encoder reranking (+15–30% precision improvement)
- HyDE (Hypothetical Document Embeddings) for complex query expansion
- LangGraph agentic multi-hop reasoning (not a linear chain)
- RAGAs + LLM-as-judge automated self-evaluation on every response
- DSPy self-optimization loop — system improves from its own failures
- Langfuse full observability dashboard — traces, latency, cost, scores
- Fully deployed, live, publicly accessible demo

---

## COMPLETE TECH STACK (DO NOT SUBSTITUTE)

| Layer | Tool | Why |
|---|---|---|
| LLM Inference | Groq — LLaMA 3.3 70B | Free 14,400 req/day, 300+ tok/sec, OpenAI-compatible |
| Embeddings | BAAI/bge-m3 | Multilingual, 8192 ctx, top MTEB, I already use it at work |
| Vector DB | Qdrant Cloud | 1GB free forever, native hybrid search (BM25+dense) |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder, free, runs locally |
| Orchestration | LangGraph | Stateful cyclic agent graphs, not linear chains |
| Evaluation | RAGAs + LLM-as-judge | Auto-score faithfulness, relevance, precision, recall |
| Self-optimization | DSPy | Auto-rewrites prompts from failure cases |
| Observability | Langfuse Cloud | 50K events/month free, LangGraph native |
| Backend | FastAPI | I already know this, async, production-grade |
| Deployment | Render (backend) + HuggingFace Spaces (frontend) | Both free |
| Frontend | Gradio | Fast to ship, clean UI |
| CI/CD | GitHub Actions | Free for public repos |
| Dev vector DB | ChromaDB | Local only during development |
| Caching | Redis (local) / in-memory LRU | Cold start optimization, embedding cache |
| A/B Testing | Custom PipelineConfig + Langfuse tags | Compare baseline vs optimized pipelines |
| Failure Classification | Custom classifier module | Tag every failure: retrieval/hallucination/irrelevant/incomplete |
| Cost Tracking | tiktoken + Langfuse metadata | Per-query, per-stage token cost in USD |
| Rate Limiting | slowapi (FastAPI middleware) | Per-IP rate limiting, repo size guards |
| Secrets | GitHub Secrets + python-dotenv | Free, secure |

**API keys needed (all free):**
- `GROQ_API_KEY` — console.groq.com (primary LLM)
- `TOGETHER_API_KEY` — together.ai (fallback LLM if Groq rate-limits)
- `QDRANT_URL` + `QDRANT_API_KEY` — cloud.qdrant.io
- `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` — langfuse.com
- `GITHUB_TOKEN` — for cloning repos (rotate 2 tokens for rate limit handling)
- `HF_TOKEN` — for HuggingFace Spaces deployment

---

## COMPLETE FOLDER STRUCTURE

```
devdocs-ai/
│
├── ingestion/
│   ├── github_loader.py        # Clone repo via API, extract .md .py .rst .txt files
│   ├── chunker.py              # Semantic chunking by heading/section (NOT fixed char splits)
│   ├── embedder.py             # BAAI/bge-m3 embeddings via sentence-transformers
│   ├── indexer.py              # Upsert to Qdrant with metadata (repo, file, section, lang)
│   ├── cache.py                # Embedding cache — skip re-embedding unchanged files
│   └── guards.py               # Repo size limit check, file count limit, malicious input guard
│
├── retrieval/
│   ├── hybrid_retriever.py     # BM25 + dense retrieval + RRF fusion via Qdrant
│   ├── reranker.py             # BGE cross-encoder reranking top-50 → top-5
│   ├── compressor.py           # Strip irrelevant sentences from context before LLM
│   └── query_processor.py      # Query rewriting + HyDE generation via Groq
│
├── agent/
│   ├── graph.py                # LangGraph state machine — full pipeline graph definition
│   ├── nodes.py                # Router, Retriever, Reranker, Generator, Evaluator nodes
│   ├── state.py                # AgentState dataclass
│   ├── memory.py               # Session memory + multi-turn conversation history
│   └── config.py               # PipelineConfig — toggle hyde/reranker/compression per run
│
├── evaluation/
│   ├── ragas_scorer.py         # Auto-score: faithfulness, answer relevance, context precision/recall
│   ├── llm_judge.py            # LLM-as-judge — verify answer is grounded in retrieved context
│   ├── failure_classifier.py   # Classify failure type: retrieval_failure | hallucination |
│   │                           #   irrelevant_context | incomplete_answer | out_of_scope
│   ├── feedback_store.py       # User thumbs up/down → correction store (JSON/SQLite)
│   ├── ab_testing.py           # A/B test: baseline pipeline vs optimized pipeline
│   └── dspy_optimizer.py       # Weekly auto prompt optimization from failure cases
│
├── cost/
│   └── tracker.py              # Per-query, per-stage token cost in USD via tiktoken
│
├── api/
│   ├── main.py                 # FastAPI app + route registration
│   ├── routes/
│   │   ├── ingest.py           # POST /ingest — index a GitHub repo
│   │   ├── query.py            # POST /query — full pipeline, returns answer + citations + scores
│   │   ├── feedback.py         # POST /feedback — thumbs up/down
│   │   ├── metrics.py          # GET /metrics — aggregate RAGAs scores + latency stats
│   │   ├── failures.py         # GET /failures — failure mode breakdown dashboard
│   │   └── ab_results.py       # GET /ab-results — A/B test comparison results
│   └── middleware.py           # Per-IP rate limiting (slowapi), logging, error handling, CORS
│
├── frontend/
│   └── app.py                  # Gradio UI — all components including explainability toggle
│
├── observability/
│   ├── langfuse_client.py      # Trace every pipeline step — selective logging (failures priority)
│   └── metrics_dashboard.py    # Aggregate scores for public shareable dashboard
│
├── tests/
│   ├── test_retrieval.py       # Precision@K, Recall@K unit tests
│   ├── test_pipeline.py        # End-to-end integration tests
│   ├── test_ab.py              # A/B testing framework unit tests
│   ├── test_guards.py          # Security + input validation tests
│   └── gold_standard_qa.json   # 50 Q&A pairs over LangChain/FastAPI/React docs
│
├── scripts/
│   ├── benchmark.py            # Run full benchmark suite, output metrics table for README
│   ├── precompute_embeddings.py # Pre-embed demo repos at startup (cold start fix)
│   └── rotate_github_tokens.py # Auto-rotate between 2 GitHub tokens for rate limit handling
│
├── prompts/                    # DSPy-optimized prompts stored here, versioned in git
│   ├── query_rewriter_v1.json
│   ├── generator_v1.json
│   └── evaluator_v1.json
│
├── .github/
│   └── workflows/
│       ├── ci.yml              # Run tests on every push
│       ├── deploy.yml          # Auto-deploy to Render on merge to main
│       └── optimize.yml        # Weekly cron: run DSPy optimizer, commit updated prompts
│
├── Dockerfile
├── docker-compose.yml          # App + Qdrant local + Redis for dev environment
├── requirements.txt
├── .env.example                # Template for all API keys
├── DECISIONS.md                # WHY every tech choice was made (critical for interviews)
└── README.md                   # Architecture diagram + benchmark table + live demo link
```

---

## FULL QUERY PIPELINE — EVERY STAGE IN ORDER

```
User Query (natural language)
    ↓
[1] Query Router
    — Classify: simple / complex / needs clarification
    — Simple → skip HyDE (saves 200–400ms latency)
    — Unclear → return clarifying question to user
    ↓
[2] Query Rewriter + HyDE
    — Rewrite vague query into retrieval-friendly version (Groq/LLaMA)
    — Generate hypothetical answer (HyDE) → embed it → retrieve against that embedding
    — Dramatically improves recall for complex multi-part questions
    ↓
[3] Hybrid Retrieval (Qdrant)
    — Dense retrieval: BAAI/bge-m3 cosine similarity
    — Sparse retrieval: BM25 keyword matching
    — Fuse both ranked lists via Reciprocal Rank Fusion (RRF)
    — Retrieve top-50 candidates
    ↓
[4] BGE Reranker
    — Cross-encoder reads query + each document jointly
    — Re-scores all 50 candidates
    — Keeps top-5 for LLM context
    — Benchmark: measure Precision@5 before vs after → document improvement %
    ↓
[5] Context Compression
    — Strip irrelevant sentences from top-5 chunks
    — Reduces tokens sent to LLM → lower cost + lower hallucination risk
    ↓
[6] LangGraph Agent — Plan → Act → Verify
    — Generator node: produce answer grounded in compressed context
    — Include source citations (file name, line range, repo link)
    — Confidence check: if low confidence → trigger second retrieval round with refined query
    — Multi-hop loop: handles "how does X connect to Y?" with 2+ retrieval rounds
    ↓
[7] LLM-as-Judge Evaluator
    — Verify: is the answer grounded in retrieved context? (yes/no + score)
    — If answer contradicts context → flag as potential hallucination
    — Score stored in Langfuse + returned to user as confidence badge
    ↓
[8] RAGAs Auto-Scorer
    — Faithfulness: does answer contradict retrieved context?
    — Answer Relevance: does it actually answer the question?
    — Context Precision: how much retrieved context was actually useful?
    — Context Recall: did retrieval capture what was needed?
    ↓
[9] Response to User
    — Answer with inline citations (file + line links to GitHub)
    — Confidence badge (green ≥0.85 / amber 0.65–0.85 / red <0.65)
    — Latency display (total + per-stage breakdown)
    — Thumbs up / thumbs down feedback
    ↓
[10] Langfuse Trace
    — Log entire pipeline: latency per stage, token cost, all scores
    — SELECTIVE LOGGING: always log failures + low-score traces; sample 20% of successful traces
    — Feeds public observability dashboard
    — Failed queries (low score or thumbs down) → correction store → DSPy optimizer
    — Tag each trace with: pipeline_version (baseline | optimized), ab_group (A | B)
```

---

## PIPELINE CONFIG SYSTEM (enables A/B testing + experimentation)

```python
# agent/config.py
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    use_hyde: bool = True               # HyDE query expansion
    use_reranker: bool = True           # BGE cross-encoder reranking
    use_compression: bool = True        # Context compression before LLM
    use_multi_hop: bool = True          # Allow second retrieval round
    use_cache: bool = True              # Embedding + result cache
    top_k_retrieval: int = 50           # Candidates to retrieve
    top_k_rerank: int = 5              # Chunks to pass to LLM
    max_retrieval_rounds: int = 2       # Max multi-hop iterations
    min_confidence_threshold: float = 0.65  # Below this → trigger re-retrieval
    llm_model: str = "llama-3.3-70b-versatile"  # Primary model
    fallback_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Together AI fallback
    pipeline_version: str = "baseline"  # Tag for A/B tracking in Langfuse

# Predefined configs for A/B testing:
BASELINE_CONFIG = PipelineConfig(
    use_hyde=False, use_reranker=False, use_compression=False,
    pipeline_version="baseline"
)
OPTIMIZED_CONFIG = PipelineConfig(
    use_hyde=True, use_reranker=True, use_compression=True,
    pipeline_version="optimized"
)
```

---

## LANGGRAPH STATE MACHINE — NODE DEFINITIONS

```python
# agent/state.py
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    query: str                      # Original user query
    rewritten_query: str            # LLM-rewritten query
    hyde_document: str              # Hypothetical answer for HyDE
    query_type: str                 # "simple" | "complex" | "clarify"
    retrieved_chunks: List[dict]    # Top-50 raw retrieved chunks
    reranked_chunks: List[dict]     # Top-5 after reranking (with rerank scores)
    compressed_context: str         # Context after compression
    answer: str                     # Generated answer
    citations: List[dict]           # Source citations
    confidence_score: float         # LLM-as-judge score
    ragas_scores: dict              # Faithfulness, relevance, precision, recall
    failure_type: Optional[str]     # retrieval_failure|hallucination|irrelevant_context|
                                    #   incomplete_answer|out_of_scope|None (success)
    cost_usd: dict                  # {"total": float, "retrieval": float, "generation": float,
                                    #  "reranking": float, "hyde": float}
    retrieval_round: int            # How many retrieval rounds completed
    session_history: List[dict]     # Multi-turn conversation
    needs_clarification: bool       # True if query is unclear
    clarification_question: str     # Question to ask user
    pipeline_config: dict           # PipelineConfig used for this query (for A/B tracking)
    explainability: dict            # If explainability mode ON: chunk scores, rerank scores,
                                    #   why-chosen reasoning, stage-by-stage reasoning log

# agent/graph.py — nodes:
# router_node → query_rewriter_node → retriever_node → reranker_node
# → compressor_node → generator_node → evaluator_node → ragas_node
# Conditional edges:
# router → clarify (if needs_clarification=True)
# evaluator → retriever (if confidence < 0.65 and retrieval_round < 2)
# evaluator → generator (otherwise)
```

---

## INGESTION PIPELINE — KEY DETAILS

```python
# ingestion/github_loader.py
# - Accept GitHub repo URL → extract owner/repo
# - Use GitHub API (with token) to list all files
# - Filter: .md, .py, .rst, .txt, .ipynb files only
# - Skip: node_modules, .git, __pycache__, dist, build folders
# - Download file content via raw.githubusercontent.com
# - Return list of {filename, content, language, path, repo}

# ingestion/chunker.py
# CRITICAL: DO NOT use fixed character/token splits
# USE: Semantic chunking strategy:
# - For .md/.rst: split by heading (##, ###) — each section = one chunk
# - For .py: split by class/function definition using AST parser
# - For .txt: split by double newline (paragraph boundary)
# - Each chunk must have: content, heading, file_path, repo, char_start, char_end
# - Target chunk size: 512–1024 tokens
# - Overlap: 50 tokens between adjacent chunks

# ingestion/indexer.py
# Qdrant collection setup:
# - Vector size: 1024 (BAAI/bge-m3 output dim)
# - Distance: Cosine
# - Sparse vectors: Enable for BM25 (Qdrant built-in)
# - Payload fields (for filtering): repo_name, file_path, language, section_heading
```

---

## HYBRID RETRIEVAL — KEY DETAILS

```python
# retrieval/hybrid_retriever.py
# Qdrant supports hybrid search natively via query_points with fusion=RRF
# 1. Dense: embed query with BAAI/bge-m3 → cosine similarity search
# 2. Sparse: tokenize query → BM25 sparse vector search
# 3. Fusion: Qdrant RRF (Reciprocal Rank Fusion) combines both ranked lists
# 4. Return top-50 candidates with scores + payload

# retrieval/query_processor.py
# Query Rewriting prompt:
# "You are a search query optimizer. Rewrite this user question into
#  a precise, keyword-rich query for retrieving technical documentation.
#  Original: {query}. Rewritten:"
#
# HyDE prompt:
# "Write a concise technical documentation snippet that would answer
#  this question: {query}. Focus on specifics, code examples, and
#  precise terminology."
```

---

## EVALUATION — KEY DETAILS

```python
# evaluation/ragas_scorer.py
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
# Run on every query — async, non-blocking
# Store scores in Langfuse as trace metadata

# evaluation/llm_judge.py
# Prompt: "Given this context: {context}
#          And this answer: {answer}
#          Is the answer fully supported by the context?
#          Score from 0.0 to 1.0. Respond with JSON: {"score": float, "reason": str}"
# Use Groq/LLaMA 3.3 70B for the judge — different from generator

# evaluation/dspy_optimizer.py
# Run weekly via GitHub Actions cron
# 1. Load all low-score traces from Langfuse (faithfulness < 0.70)
# 2. Use DSPy BootstrapFewShot optimizer
# 3. Optimize: query rewriter prompt, generator prompt, evaluator prompt
# 4. Save optimized prompts to prompts/ folder
# 5. Commit updated prompts back to repo via GitHub API
```

---

## FAILURE CLASSIFIER — EVERY FAILURE GETS TAGGED

```python
# evaluation/failure_classifier.py
# After every query, classify the outcome. This is what makes the
# "Failure Modes Dashboard" possible — and what gives Jay a real answer
# to "How do you debug RAG systems?" in interviews.

FAILURE_TYPES = {
    "retrieval_failure":    "Retrieved chunks don't contain answer (context_recall < 0.5)",
    "hallucination":        "Answer contradicts retrieved context (LLM-as-judge < 0.6)",
    "irrelevant_context":   "Retrieved chunks are off-topic (context_precision < 0.5)",
    "incomplete_answer":    "Answer partially correct but missing key info (faithfulness OK,
                             relevance < 0.6)",
    "out_of_scope":         "Question outside the indexed repo's content",
    "success":              "All scores above threshold, user didn't thumbs-down"
}

# Classification logic (run after RAGAs + LLM-judge):
# if context_recall < 0.5        → retrieval_failure
# elif llm_judge_score < 0.6     → hallucination
# elif context_precision < 0.5   → irrelevant_context
# elif answer_relevance < 0.6    → incomplete_answer
# elif no relevant chunks found  → out_of_scope
# else                           → success

# Store failure_type in:
# - AgentState.failure_type
# - Langfuse trace metadata tag
# - feedback_store for DSPy training

# GET /failures endpoint returns:
# {
#   "failure_breakdown": {
#     "retrieval_failure": {"count": int, "percentage": float, "example_queries": [str]},
#     "hallucination": {"count": int, "percentage": float, "example_queries": [str]},
#     "irrelevant_context": {"count": int, "percentage": float},
#     "incomplete_answer": {"count": int, "percentage": float},
#     "out_of_scope": {"count": int, "percentage": float},
#     "success": {"count": int, "percentage": float}
#   },
#   "top_failing_repos": [str],
#   "most_common_failure": str,
#   "trend_7d": {"improving": bool, "delta": float}
# }
```

---

## A/B TESTING SYSTEM

```python
# evaluation/ab_testing.py
# Compare BASELINE_CONFIG vs OPTIMIZED_CONFIG on identical queries
# Route 50% of traffic to each config (use session_id hash for consistency)

# How it works:
# 1. Hash session_id → assign to group A (baseline) or group B (optimized)
# 2. Run the query through the assigned pipeline config
# 3. Tag Langfuse trace with ab_group: "A" | "B" and pipeline_version
# 4. Aggregate results daily from Langfuse API

# GET /ab-results returns:
# {
#   "baseline": {
#     "avg_faithfulness": float,
#     "avg_latency_ms": int,
#     "avg_cost_usd": float,
#     "hallucination_rate": float,
#     "positive_feedback_rate": float,
#     "sample_size": int
#   },
#   "optimized": { ... same fields ... },
#   "winner": "baseline" | "optimized" | "inconclusive",
#   "faithfulness_delta": float,   # e.g. +0.12 means optimized is 12% more faithful
#   "latency_delta_ms": int,       # negative = optimized is faster
#   "cost_delta_usd": float        # negative = optimized is cheaper
# }

# This directly answers the interview question:
# "How do you validate that your optimizations actually work?"
```

---

## COST TRACKING

```python
# cost/tracker.py
# Track token cost at every LLM call using tiktoken + Groq pricing
# Groq LLaMA 3.3 70B pricing (as of 2026): ~$0.59/1M input, $0.79/1M output tokens

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    PRICING = {
        "llama-3.3-70b-versatile": {"input": 0.59e-6, "output": 0.79e-6},
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": {"input": 0.18e-6, "output": 0.18e-6},
    }
    p = PRICING.get(model, {"input": 0.59e-6, "output": 0.79e-6})
    return (input_tokens * p["input"]) + (output_tokens * p["output"])

# Track cost per stage:
# cost_usd = {
#   "hyde_generation": float,
#   "query_rewriting": float,
#   "generation": float,
#   "llm_judge": float,
#   "ragas_eval": float,
#   "total": float
# }
# Store in AgentState.cost_usd and Langfuse trace metadata
# Surface in /metrics endpoint + Gradio UI cost display
# Target: < $0.003 per query (with compression reducing context tokens)
```

---

## COLD START OPTIMIZATION + CACHING

```python
# ingestion/cache.py
# Two-level cache to eliminate cold start latency:

# Level 1: Embedding cache (in-memory LRU)
# - Key: sha256(chunk_text) → Value: embedding vector
# - Prevents re-embedding identical chunks on repo re-index
# - Size limit: 10,000 entries (~400MB for bge-m3 1024-dim vectors)
from functools import lru_cache

# Level 2: Query result cache (in-memory dict)
# - Key: sha256(query + repo_name) → Value: full pipeline result
# - TTL: 1 hour (stale after repo re-index)
# - Only cache high-confidence results (confidence >= 0.85)
# - Max size: 500 cached queries

# scripts/precompute_embeddings.py
# Run at container startup (before first request)
# Pre-embed all chunks for the 4 demo repos:
#   langchain-ai/langchain, tiangolo/fastapi, facebook/react, pytorch/pytorch
# Store in Qdrant + populate embedding cache
# This ensures the demo is instant for first user — no 30s wait

# Cold start sequence:
# Container starts → precompute_embeddings.py runs → 4 repos indexed
# First user query → hits cache → sub-500ms response
# WITHOUT this: first query = 15–30s indexing wait = bad demo impression
```

---

## SECURITY + ABUSE HANDLING

```python
# api/middleware.py — rate limiting + guards

# 1. Per-IP rate limiting (slowapi)
from slowapi import Limiter
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)

# Limits:
# POST /ingest → 5 requests per hour per IP (expensive operation)
# POST /query  → 30 requests per minute per IP
# POST /feedback → 60 per minute per IP

# 2. Repo guards (ingestion/guards.py)
MAX_REPO_SIZE_MB = 50           # Reject repos > 50MB (prevents abuse)
MAX_FILES_PER_REPO = 500        # Skip repos with too many files
MAX_CHUNK_LIMIT_PER_REPO = 2000 # Stop indexing after 2000 chunks (Qdrant 1GB free tier)
ALLOWED_FILE_EXTENSIONS = {".md", ".py", ".rst", ".txt", ".ipynb"}
BLOCKED_REPO_PATTERNS = [".", "..", "//", "..%2F"]  # Path traversal prevention

# 3. Input sanitization
# - Strip control characters from queries
# - Max query length: 1000 characters
# - Max session_id length: 64 characters
# - Validate GitHub URL format before fetching
# - Reject queries with prompt injection patterns

# 4. Qdrant 1GB free tier guard
# Before indexing: check current collection size
# If > 900MB: return warning {"status": "storage_near_limit", "action": "oldest_repo_evicted"}
# Auto-evict least-recently-used repo to make space

# 5. GitHub API rate limit handling
# Use 2 GitHub tokens, rotate on 403/429
# Cache github file listings for 1 hour (avoid repeated API calls for same repo)
```

---

## FASTAPI ENDPOINTS

```python
# POST /ingest
# Body: {"repo_url": "https://github.com/owner/repo", "branch": "main"}
# Guards: repo size check, file count check, Qdrant storage check, per-IP rate limit
# Process: download → chunk → embed (cache check first) → index to Qdrant
# Response: {"status": "success", "chunks_indexed": int, "repo": str,
#            "storage_used_mb": float, "cache_hits": int}

# POST /query
# Body: {"question": str, "repo_name": str, "session_id": str,
#        "explainability": bool, "pipeline_version": "baseline"|"optimized"}
# Process: full pipeline (router → retriever → reranker → agent → evaluator)
# Response: {
#   "answer": str,
#   "citations": [{"file": str, "section": str, "url": str}],
#   "confidence": float,
#   "failure_type": str | null,
#   "ragas_scores": {"faithfulness": float, "relevance": float,
#                    "context_precision": float, "context_recall": float},
#   "cost_usd": {"total": float, "by_stage": dict},
#   "latency_ms": {"total": int, "retrieval": int, "reranking": int,
#                  "generation": int, "evaluation": int},
#   "explainability": {          # only if explainability=True
#     "retrieved_chunks": [{"text": str, "score": float, "file": str}],
#     "reranked_chunks": [{"text": str, "rerank_score": float}],
#     "why_chosen": str,         # LLM-generated reasoning for answer selection
#     "stage_log": [str]         # Step-by-step reasoning trace
#   },
#   "trace_url": str,            # Langfuse trace URL
#   "ab_group": "A" | "B"
# }

# POST /feedback
# Body: {"trace_id": str, "rating": "up" | "down", "correction": str | null}
# Process: save to feedback_store, flag trace in Langfuse, update failure_type if needed

# GET /metrics
# Response: aggregate scores over last 7 days
# {
#   "avg_faithfulness": float, "avg_latency_p95": int,
#   "hallucination_rate": float, "total_queries": int,
#   "avg_cost_per_query_usd": float, "total_cost_usd": float,
#   "positive_feedback_rate": float, "cache_hit_rate": float
# }

# GET /failures
# Response: failure mode breakdown (see FAILURE CLASSIFIER section)

# GET /ab-results
# Response: A/B test comparison (see A/B TESTING section)

# GET /health
# Response: {"status": "ok", "qdrant_storage_mb": float, "cache_size": int}
# Used by UptimeRobot to keep Render free tier awake
```

---

## GRADIO FRONTEND — UI COMPONENTS

```python
# frontend/app.py — Gradio Blocks UI
# Components:
# 1.  Repo URL input (text) + Index button
# 2.  Question input (text)
# 3.  Answer output (markdown — renders code blocks and citations)
# 4.  Confidence badge (colored label — green/amber/red)
# 5.  Latency breakdown (label per stage)
# 6.  Cost display (per-query USD cost + running total)
# 7.  Citations accordion (expandable list of source files + links)
# 8.  RAGAs scores panel (4 score numbers)
# 9.  Failure type badge (shown only on failures — e.g. "⚠️ Retrieval Failure")
# 10. Thumbs up / thumbs down buttons
# 11. Session ID (hidden state for multi-turn)
# 12. "Show Reasoning" toggle (Explainability Mode)
#     When ON → shows:
#       - Retrieved chunks with cosine similarity scores
#       - Reranked chunks with cross-encoder scores
#       - Why-chosen: LLM explanation of answer selection
#       - Stage-by-stage reasoning log
# 13. Pipeline selector (Baseline vs Optimized) for A/B demo
#
# Pre-loaded repos (demo ready without indexing):
# - langchain-ai/langchain
# - tiangolo/fastapi
# - facebook/react
# - pytorch/pytorch
```

---

## LANGFUSE TRACING — WHAT TO LOG

```python
# observability/langfuse_client.py
# SELECTIVE LOGGING STRATEGY (stay within 50K event/month free tier):
# - ALWAYS log: failures (any failure_type != "success"), low-score traces (faithfulness < 0.75)
# - SAMPLE: 20% of successful traces (random sampling)
# - NEVER log: health checks, cached responses

# Log as a single trace per query with nested spans:
# Trace: {query, repo, session_id, user_id, ab_group, pipeline_version}
#   Span: query_router {input: query, output: query_type, latency}
#   Span: query_rewriter {input: query, output: rewritten_query, tokens, cost, latency}
#   Span: hyde_generation {input: query, output: hyde_doc, tokens, cost, latency}
#   Span: hybrid_retrieval {input: query, output: chunk_count, latency}
#   Span: reranking {input: 50_chunks, output: 5_chunks, latency}
#   Span: compression {input: 5_chunks, output: compressed_tokens, latency}
#   Span: generation {input: context+query, output: answer, tokens, cost, latency}
#   Span: llm_judge {output: confidence_score, latency}
#   Span: ragas_eval {output: all_4_scores, latency}
#   Span: failure_classifier {output: failure_type, latency}
# Scores: faithfulness, relevance, context_precision, context_recall, llm_judge
# Metadata: total_latency, total_cost_usd, retrieval_round_count,
#           failure_type, ab_group, pipeline_version, cache_hit
```

---

## DOCKER + CI/CD

```yaml
# docker-compose.yml (development)
# services:
#   app: FastAPI on port 8000
#   qdrant: Qdrant local on port 6333 (for dev without cloud)
#   redis: Redis on port 6379 (optional — for caching layer in dev)

# .github/workflows/ci.yml
# Triggers: push to any branch
# Steps: checkout → install deps → run pytest tests/ → lint with ruff

# .github/workflows/deploy.yml
# Triggers: merge to main
# Steps: checkout → build Docker image → push to Render via deploy hook
# Secrets needed: RENDER_DEPLOY_HOOK_URL, all API keys as GitHub Secrets

# .github/workflows/optimize.yml
# Triggers: cron schedule — every Monday 2am UTC
# Steps: checkout → install deps → run dspy_optimizer.py →
#        if prompts changed → commit updated prompts/ to repo → push
# This is the self-improving loop running on autopilot

# Render free tier note:
# Service sleeps after 15 min inactivity
# Add UptimeRobot (free) to ping /health every 5 min → keeps it awake
# At startup: container runs precompute_embeddings.py → 4 demo repos ready
```

---

## DECISIONS.md — MUST WRITE THIS (Interviewers ask about every choice)

Document WHY for each of these decisions:
1. Why Groq over OpenAI? (Cost + speed — 300+ tok/sec free)
2. Why Together AI as fallback? (OpenAI-compatible, handles Groq rate limits gracefully)
3. Why BAAI/bge-m3 over OpenAI embeddings? (Cost + multilingual + top MTEB performance)
4. Why Qdrant over Pinecone or ChromaDB? (Native hybrid search + 1GB free forever)
5. Why RRF over weighted sum for fusion? (No hyperparameter tuning needed)
6. Why cross-encoder reranker over bi-encoder? (Reads query+doc jointly → higher precision)
7. Why HyDE for complex queries? (Embeds answer space not question space → better recall)
8. Why LangGraph over LangChain LCEL? (Handles cycles/loops for multi-hop reasoning)
9. Why RAGAs over manual evaluation? (Automated, consistent, reference-free)
10. Why DSPy for optimization? (Systematic prompt optimization vs manual tweaking)
11. Why Langfuse over LangSmith? (50K events/month free vs 5K)
12. Why semantic chunking over fixed splits? (80% of RAG failures start here)
13. Why context compression before LLM? (Lower tokens → lower cost + less hallucination)
14. Why PipelineConfig dataclass over hardcoded flags? (Enables A/B testing + experimentation)
15. Why classify failure types? (Debugging without failure taxonomy is guessing)
16. Why LRU cache for embeddings? (Avoid re-embedding unchanged chunks on re-index)
17. Why selective Langfuse logging? (Stay within 50K free tier — prioritize failures)
18. Why precompute demo embeddings at startup? (Cold start kills first impressions in demos)
19. Why per-IP rate limiting? (Free tier protection + abuse prevention)
20. Why 2 GitHub tokens? (Rotate on rate limit — prevents ingestion failures during demos)

---

## TARGET BENCHMARKS (aim for these, document in README)

| Metric | Target |
|---|---|
| Faithfulness score (RAGAs) | ≥ 0.90 |
| Answer relevance | ≥ 0.85 |
| Context precision | ≥ 0.85 |
| Precision improvement from reranking | +15–30% |
| End-to-end p95 latency | ≤ 2.5s |
| Hallucination rate (LLM-as-judge) | ≤ 5% |
| Positive user feedback rate | ≥ 75% |
| Average cost per query | ≤ $0.003 USD |
| Cache hit rate (demo repos) | ≥ 80% |
| Success rate (no failure classification) | ≥ 85% |
| A/B improvement (optimized vs baseline faithfulness) | ≥ +10% |

---

## THE RESUME BULLET (goal — fill in real numbers when done)

*"Built DevDocs AI — a production-grade developer documentation RAG system with hybrid BM25+dense retrieval (RRF fusion), BGE cross-encoder reranking (+__% precision), LangGraph agentic multi-hop reasoning, and a DSPy self-optimization loop — achieving __% faithfulness score at <__s p95 latency, <__% hallucination rate, and $0.00__ avg cost/query. Features failure mode classification dashboard, A/B testing framework, explainability mode, and LRU embedding cache. Includes automated RAGAs + LLM-as-judge evaluation with Langfuse observability. Live on HuggingFace Spaces."*

---

## 10-WEEK BUILD ORDER (do NOT skip or reorder steps)

| Week | What to build | Why this order |
|---|---|---|
| 1 | GitHub ingestion + semantic chunking + guards | 80% of RAG failures start here; guards prevent abuse from day 1 |
| 2 | Hybrid retrieval BM25+dense+RRF | Core retrieval before adding complexity |
| 3 | Query rewriting + HyDE + PipelineConfig | Config system enables A/B testing from week 8 onward |
| 4 | BGE reranker + context compression | Biggest single precision improvement |
| 5 | LangGraph agent state machine | Orchestration layer over working retrieval |
| 6 | RAGAs + LLM-as-judge + failure classifier | Can't debug what you don't classify |
| 7 | Cost tracker + embedding cache + cold start precompute | Production signals: cost + performance |
| 8 | Feedback loop + DSPy optimizer + A/B testing | Needs evaluation data from week 6 |
| 9 | FastAPI backend + security middleware + Docker + CI/CD | Productionize the working pipeline |
| 10 | Gradio frontend + explainability mode + HuggingFace deploy | Make it public and impressive |
| 11 | Weekly optimize GitHub Actions cron + Langfuse selective logging | Full automation loop running |
| 12 | Benchmarks + README + DECISIONS.md + LinkedIn post | Ship and announce |

---

## HOW TO WORK WITH ME

- Build one week at a time — complete each module before moving to next
- Write production-quality code — type hints, docstrings, error handling
- Write tests alongside each module (tests/ folder)
- After each module: show me the code, explain key decisions
- At the end of each week: run a quick smoke test to confirm the module works
- If there's a simpler free tool that achieves the same result — suggest it
- Flag immediately if any free tier limit might be hit before project completion
- Always use PipelineConfig — never hardcode boolean flags directly in pipeline logic
- Every new feature must be togglable via PipelineConfig (for A/B testing)

---

## ELITE INTERVIEW ANSWERS THIS PROJECT ENABLES

After building this, Jay can answer these senior-level questions with real examples:

**"How do you debug a RAG system?"**
→ "I built a failure classifier that tags every query as retrieval_failure, hallucination,
   irrelevant_context, incomplete_answer, or out_of_scope. I can pull the failure dashboard
   and see exactly which category is causing problems, with example queries for each."

**"How do you validate that your optimizations work?"**
→ "I built an A/B testing system that routes 50% of traffic to baseline vs optimized configs.
   I compare faithfulness delta, latency delta, and cost delta across both groups in Langfuse."

**"How do you handle LLM costs in production?"**
→ "I track cost at every stage — HyDE generation, reranking, generation, evaluation.
   Target is under $0.003/query. Context compression alone reduced generation tokens by ~40%."

**"What happens when your primary LLM provider goes down?"**
→ "Groq is primary. If it returns 429 or 503, the system falls back to Together AI's
   LLaMA endpoint — same model, OpenAI-compatible API, just slower. Transparent to the user."

**"How do you prevent abuse of a free-tier deployment?"**
→ "Per-IP rate limiting via slowapi — 5 ingests/hour, 30 queries/minute. Repo size capped
   at 50MB. Qdrant storage guard evicts LRU repos when approaching 900MB. GitHub tokens
   rotate automatically on rate limit to prevent ingestion failures."

**"Why is your system's first query fast?"**
→ "I precompute embeddings for 4 demo repos at container startup. Combined with an LRU
   embedding cache, first-query latency for pre-indexed repos is under 500ms."

**"What does your observability setup look like?"**
→ "Langfuse traces every query with spans for each pipeline stage — latency, token count,
   cost, RAGAs scores, failure classification. I log all failures but only 20% sample
   successful traces to stay within the free tier."

---

**Start with:** Week 1 — the ingestion pipeline.
Write `github_loader.py`, `chunker.py`, `embedder.py`, `indexer.py`, and `guards.py` in full, production-ready, with error handling and type hints.