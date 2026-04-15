import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.github_loader import GitHubLoader
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from ingestion.indexer import Indexer
from retrieval.query_processor import QueryProcessor
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.compressor import Compressor
from agent.config import OPTIMIZED_CONFIG

def run_reranking_test(repo_url: str):
    print(f"--- Starting Reranking & Compression Test for {repo_url} ---")
    
    persist_dir = "./test_chroma_db"
    # Note: We reuse the DB if it exists, otherwise it will be empty
    
    loader = GitHubLoader()
    chunker = Chunker()
    embedder = Embedder()
    reranker = Reranker()
    compressor = Compressor()
    processor = QueryProcessor()
    
    retriever = HybridRetriever(persist_directory=persist_dir)
    retriever.set_embedder(embedder)
    retriever.set_reranker(reranker)
    retriever.set_compressor(compressor)
    
    # 1. Ingest & Index (Ensures the test is standalone)
    print("[1] Ingesting and Indexing test repository...")
    repo_name = f"{loader.parse_github_url(repo_url)['owner']}/{loader.parse_github_url(repo_url)['repo']}"
    
    files = loader.fetch_repo_files(repo_url)
    all_chunks = []
    for f in files[:5]: # Top 5 files for speed
        content = loader.fetch_file_content(f["url"])
        chunks = chunker.process_file(f["path"], content)
        all_chunks.extend(chunks)
    
    indexer = Indexer(persist_directory=persist_dir)
    indexer.set_embedder(embedder)
    indexer.index_chunks(all_chunks, repo_name=repo_name)

    query = "How to run tests in this project?"
    print(f"\n[2] Processing Query: '{query}'")
    
    # Dual Retrieval (Week 3)
    rewritten_query = processor.rewrite_query(query)
    hyde_doc = processor.generate_hyde_document(query)
    
    # Precision-Optimized Retrieval (Week 4)
    print("[2] Executing Hybrid Retrieval + Reranking + Compression...")
    results = retriever.retrieve(
        query=rewritten_query, 
        repo_name=repo_name, 
        top_k_retrieval=OPTIMIZED_CONFIG.top_k_retrieval,
        top_k_rerank=OPTIMIZED_CONFIG.top_k_rerank,
        hyde_doc=hyde_doc,
        use_reranker=OPTIMIZED_CONFIG.use_reranker,
        use_compression=OPTIMIZED_CONFIG.use_compression
    )

    print(f"\n[3] Results Breakdown:")
    print(f"    Raw Chunks Retrieved: {results['raw_total']}")
    print(f"    Chunks after Reranking: {len(results['chunks'])}")
    
    print("\n    Top 3 Reranked Chunks:")
    for i, chunk in enumerate(results['chunks'][:3]):
        score = chunk['metadata'].get('rerank_score', 0)
        print(f"    - [{chunk['metadata']['filename']}] Score: {score:.4f}")
        print(f"      Content: {chunk['content'][:100]}...")

    if results['compressed_context']:
        print("\n[4] Compressed Context (Final Input to LLM):")
        print("-" * 40)
        print(results['compressed_context'])
        print("-" * 40)
        print(f"    Character count reduced from {sum(len(c['content']) for c in results['chunks'])} to {len(results['compressed_context'])}")

    print("\n--- Reranking Test Completed Successfully ---")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pypa/sampleproject"
    run_reranking_test(url)
