import os
import json
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

def test_full_pipeline():
    print("--- Starting Headless Pipeline Verification ---")
    repo_url = "https://github.com/pypa/sampleproject"
    
    # Initialize
    loader = GitHubLoader()
    chunker = Chunker()
    embedder = Embedder()
    indexer = Indexer(persist_directory="./test_chroma_db")
    indexer.set_embedder(embedder)
    processor = QueryProcessor()
    generator = AnswerGenerator()
    reranker = Reranker()
    compressor = Compressor()
    
    retriever = HybridRetriever(persist_directory="./test_chroma_db")
    retriever.set_embedder(embedder)
    retriever.set_reranker(reranker)
    retriever.set_compressor(compressor)
    
    agent = AgentGraph(retriever, processor, generator, OPTIMIZED_CONFIG)
    
    # 1. Ingest
    print(f"1. Ingesting {repo_url}...")
    details = loader.parse_github_url(repo_url)
    repo_name = f"{details['owner']}/{details['repo']}"
    files = loader.fetch_repo_files(repo_url)
    print(f"   Found {len(files)} files.")
    
    # 2. Chunk
    print("2. Chunking files...")
    all_chunks = []
    for f in files:
        content = loader.fetch_file_content(f['url'])
        chunks = chunker.process_file(f['filename'], content)
        all_chunks.extend(chunks)
    print(f"   Generated {len(all_chunks)} chunks.")
    
    # 3. Index
    print("3. Indexing chunks...")
    indexer.index_chunks(all_chunks, repo_name)
    
    # 4. Query
    query = "What does the add_one function do?"
    print(f"4. Querying agent: '{query}'")
    state = agent.run(query, repo_name)
    
    print("\n--- Final Answer ---")
    if 'messages' in state and state['messages']:
        print(state['messages'][-1]['content'])
    else:
        print("No answer generated.")
    
    print("\n--- Context Used ---")
    for i, c in enumerate(state.get('context', [])):
        print(f"[{i+1}] {c['metadata']['filename']}: {c['content'][:50]}...")

if __name__ == "__main__":
    test_full_pipeline()
