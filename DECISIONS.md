# Architectural Decisions - DevDocs AI

This document outlines the key technical decisions and trade-offs made during the development of the DevDocs RAG pipeline.

## 1. Retrieval Strategy: Multi-Hop Hybrid RRF
- **Decision**: Combined Dense (BGE-M3) and Sparse (BM25) retrieval using Reciprocal Rank Fusion (RRF), wrapped in a LangGraph-based multi-hop agent.
- **Why**: Keyword search (BM25) is essential for finding specific function names or error codes, while semantic search (Dense) handles natural language intent. Multi-hop allows the agent to find missing dependencies (e.g., "I found the class, now I need its base class in another file").

## 2. Embedding Model: BAAI/bge-m3
- **Decision**: Selected `bge-m3` over OpenAI `text-embedding-3-small`.
- **Why**: `bge-m3` is a state-of-the-art multilingual and multi-vector model that performs exceptionally well on code-related tasks and can be run locally within a Docker container, reducing external dependencies and costs.

## 3. Query Optimization: Dual-Strategy (HyDE + Rewriting)
- **Decision**: Implemented both HyDE (Hypothetical Document Embeddings) and Query Rewriting via Groq (LLaMA 3.3 70B).
- **Why**: HyDE bridges the semantic gap between a question and its answer by generating a "fake" answer and searching for its nearest neighbors. Rewriting ensures natural language queries are transformed into precise technical search terms.

## 4. Precision: Cross-Encoder Reranking
- **Decision**: Added a `ms-marco-MiniLM-L-6-v2` reranker after the initial retrieval.
- **Why**: Vector similarity alone is often noisy. A Cross-Encoder spends more compute on the top 10-20 results to ensure the most relevant context is actually placed at the top of the LLM prompt.

## 5. Agent Orchestration: LangGraph
- **Decision**: Used LangGraph over LangChain's pre-built agents.
- **Why**: LangGraph provides a precise "State Machine" approach where we define exact nodes (Retrieve, Grade, Generate) and edges. This prevents agent "hallucination loops" and makes the RAG logic highly deterministic and debuggable.

## 6. Observability: RAGAs + Custom LLM-as-judge
- **Decision**: Implemented both RAGAs for standard metrics and a custom `llm_evaluator.py` for specific technical grounding checks.
- **Why**: Automated evaluation is the only way to iterate on RAG. Without "Faithfulness" and "Relevancy" scores, prompt changes are guesswork.
