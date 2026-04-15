import sys
import os
import shutil

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.github_loader import GitHubLoader
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from ingestion.indexer import Indexer
from retrieval.hybrid_retriever import HybridRetriever

def run_retrieval_test(repo_url: str):
    print(f"--- Starting Retrieval Test for {repo_url} ---")
    
    # Setup
    persist_dir = "./test_chroma_db"
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        
    loader = GitHubLoader()
    chunker = Chunker()
    embedder = Embedder()
    indexer = Indexer(persist_directory=persist_dir)
    indexer.set_embedder(embedder)
    
    # 1. Ingest
    print("[1] Ingesting repo...")
    files = loader.fetch_repo_files(repo_url)
    all_chunks = []
    for f in files[:10]: # Limit for test speed
        content = loader.fetch_file_content(f["url"])
        chunks = chunker.process_file(f["path"], content)
        all_chunks.extend(chunks)
        
    print(f"    Generated {len(all_chunks)} chunks.")

    # 2. Index
    print("[2] Indexing chunks...")
    repo_name = f"{loader.parse_github_url(repo_url)['owner']}/{loader.parse_github_url(repo_url)['repo']}"
    indexer.index_chunks(all_chunks, repo_name=repo_name)

    # 3. Retrieve with Dual Strategy
    print("[3] Testing Hybrid Retrieval (Dual Strategy)...")
    from retrieval.query_processor import QueryProcessor
    processor = QueryProcessor()
    retriever = HybridRetriever(persist_directory=persist_dir)
    retriever.set_embedder(embedder)
    
    queries = [
        "How to use sessions in nox?",
    ]
    
    for query in queries:
        print(f"\n    Original Query: '{query}'")
        
        # Week 3 Upgrades: Rewrite + HyDE
        rewritten_query = processor.rewrite_query(query)
        hyde_doc = processor.generate_hyde_document(query)
        
        print(f"    Rewritten:      '{rewritten_query}'")
        print(f"    HyDE Generated: {'Yes' if hyde_doc != query else 'No (Fallback)'}")
        
        results = retriever.retrieve(rewritten_query, repo_name=repo_name, top_k=3, hyde_doc=hyde_doc)
        print(f"    Found {len(results)} results:")
        for r in results:
            print(f"    - [{r['metadata']['filename']}] (tokens: {r['metadata']['tokens']})")
            print(f"      Snippet: {r['content'][:100].strip()}...")

    print("\n--- Retrieval Test Completed Successfully ---")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pypa/sampleproject"
    run_retrieval_test(url)
