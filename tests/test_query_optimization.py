import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.query_processor import QueryProcessor
from agent.config import PipelineConfig

def run_query_opt_test():
    print("--- Starting Query Optimization Test ---")
    
    config = PipelineConfig(use_hyde=True)
    processor = QueryProcessor()

    if not processor.client:
        print("[!] Warning: GROQ_API_KEY is not set. Testing fallback behavior.")

    # 1. Test Query Rewriting
    print("\n[1] Testing Query Rewriting...")
    queries = [
        "how to run it?",
        "setup project",
        "error 404 in api"
    ]
    
    for q in queries:
        rewritten = processor.rewrite_query(q)
        print(f"    Original:  '{q}'")
        print(f"    Rewritten: '{rewritten}'")

    # 2. Test HyDE
    print("\n[2] Testing HyDE (Hypothetical Document Generation)...")
    test_q = "How do I configure the GitHub tokens?"
    
    if config.use_hyde:
        print(f"    Input: '{test_q}'")
        hyde_doc = processor.generate_hyde_document(test_q)
        print(f"    HyDE Output Length: {len(hyde_doc)}")
        print(f"    HyDE Snippet: {hyde_doc[:150]}...")
    else:
        print("    HyDE is disabled in config.")

    print("\n--- Query Optimization Test Completed ---")

if __name__ == "__main__":
    run_query_opt_test()
