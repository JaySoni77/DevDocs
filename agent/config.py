from dataclasses import dataclass

@dataclass
class PipelineConfig:
    use_hyde: bool = True               # HyDE query expansion
    use_reranker: bool = False          # BGE cross-encoder reranking (Week 4)
    use_compression: bool = False       # Context compression (Week 4)
    use_multi_hop: bool = False         # Allow second retrieval round (Week 5)
    use_cache: bool = True              # Embedding + result cache
    top_k_retrieval: int = 50           # Candidates to retrieve
    top_k_rerank: int = 5              # Chunks to pass to LLM
    max_retrieval_rounds: int = 2       # Max multi-hop iterations
    min_confidence_threshold: float = 0.65  # Below this → trigger re-retrieval
    llm_model: str = "llama-3.3-70b-versatile"  # Primary model (Groq)
    fallback_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # Together AI fallback
    pipeline_version: str = "baseline"  # Tag for A/B tracking

# Predefined configs for A/B testing:
BASELINE_CONFIG = PipelineConfig(
    use_hyde=False,
    use_reranker=False,
    use_compression=False,
    pipeline_version="baseline"
)

OPTIMIZED_CONFIG = PipelineConfig(
    use_hyde=True,
    use_reranker=False,
    use_compression=False,
    pipeline_version="optimized"
)
