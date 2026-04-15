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
from agent.generator import AnswerGenerator
from agent.graph import AgentGraph
from agent.config import OPTIMIZED_CONFIG

def run_generation_test(repo_url: str):
    print(f"--- Starting Grounded Generation Test for {repo_url} ---")
    
    persist_dir = "./test_chroma_db"
    
    # 1. Setup
    loader = GitHubLoader()
    chunker = Chunker()
    embedder = Embedder()
    reranker = Reranker()
    compressor = Compressor()
    processor = QueryProcessor()
    generator = AnswerGenerator()
    
    retriever = HybridRetriever(persist_directory=persist_dir)
    retriever.set_embedder(embedder)
    retriever.set_reranker(reranker)
    retriever.set_compressor(compressor)
    
    # 2. Ingest
    print("[1] Preparing indexed data...")
    repo_name = f"{loader.parse_github_url(repo_url)['owner']}/{loader.parse_github_url(repo_url)['repo']}"
    
    files = loader.fetch_repo_files(repo_url)
    all_chunks = []
    for f in files[:5]: # Top 5 files
        content = loader.fetch_file_content(f["url"])
        chunks = chunker.process_file(f["path"], content)
        all_chunks.extend(chunks)
    
    indexer = Indexer(persist_directory=persist_dir)
    indexer.set_embedder(embedder)
    indexer.index_chunks(all_chunks, repo_name=repo_name)

    # 3. Create Graph
    agent = AgentGraph(retriever, processor, generator, OPTIMIZED_CONFIG)

    # 4. Run retrieval + generation
    query = "What is this project about and how do I use the sessions?"
    print(f"\n[2] Running Agent for Query: '{query}'")
    
    final_state = agent.run(query, repo_name)

    print("\n[3] Final Generated Answer:")
    print("-" * 50)
    # The final answer is in the last message
    if final_state['messages']:
        print(final_state['messages'][-1]['content'])
    print("-" * 50)

    print("\n--- Generation Test Completed ---")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pypa/sampleproject"
    run_generation_test(url)
