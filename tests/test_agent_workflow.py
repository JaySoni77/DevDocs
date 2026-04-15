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
from agent.graph import AgentGraph
from agent.config import OPTIMIZED_CONFIG

def run_agent_test(repo_url: str):
    print(f"--- Starting Agent Workflow Test for {repo_url} ---")
    
    persist_dir = "./test_chroma_db"
    
    # 1. Setup
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
    
    # 2. Ingest
    print("[1] Preparing indexed data...")
    repo_name = f"{loader.parse_github_url(repo_url)['owner']}/{loader.parse_github_url(repo_url)['repo']}"
    
    # We only index a few files to keep it fast
    files = loader.fetch_repo_files(repo_url)
    all_chunks = []
    for f in files[:10]: # Index more files for multi-hop to have something to find
        content = loader.fetch_file_content(f["url"])
        chunks = chunker.process_file(f["path"], content)
        all_chunks.extend(chunks)
    
    indexer = Indexer(persist_directory=persist_dir)
    indexer.set_embedder(embedder)
    indexer.index_chunks(all_chunks, repo_name=repo_name)

    # 3. Create Graph
    agent = AgentGraph(retriever, processor, OPTIMIZED_CONFIG)

    # 4. Run complex multi-part query
    query = "How does the ingestion process handle security guards and file limits?"
    print(f"\n[2] Running Agent for Query: '{query}'")
    
    final_state = agent.run(query, repo_name)

    print("\n[3] Execution Summary:")
    print(f"    Total Hops: {final_state['hop_count']}")
    print(f"    Total Chunks Collected: {len(final_state['context'])}")
    print(f"    Final Sufficiency: {final_state['is_sufficient']}")
    
    print("\n    Retrieved Documents Path Trace:")
    for i, c in enumerate(final_state['context']):
        print(f"    - {i+1}: {c['metadata']['filename']}")

    print("\n--- Agent Workflow Test Completed ---")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pypa/sampleproject"
    run_agent_test(url)
