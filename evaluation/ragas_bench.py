import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI

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

def run_ragas_evaluation(repo_url: str, dataset_path: str = "evaluation/golden_dataset.json"):
    print(f"--- Starting RAGAS Evaluation for {repo_url} ---")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        golden_data = json.load(f)

    # 1. Setup Pipeline (Overriding to 8b for rate limits)
    loader = GitHubLoader()
    processor = QueryProcessor(model="llama-3.1-8b-instant")
    generator = AnswerGenerator(model="llama-3.1-8b-instant")
    compressor = Compressor(model="llama-3.1-8b-instant")
    embedder = Embedder()
    reranker = Reranker()
    
    retriever = HybridRetriever(persist_directory="./test_chroma_db")
    retriever.set_embedder(embedder)
    retriever.set_reranker(reranker)
    retriever.set_compressor(compressor)
    
    agent = AgentGraph(retriever, processor, generator, OPTIMIZED_CONFIG)
    repo_name = f"{loader.parse_github_url(repo_url)['owner']}/{loader.parse_github_url(repo_url)['repo']}"

    # 2. Run Agent on Dataset
    print(f"Running Agent on {len(golden_data)} test questions...")
    results = []
    for i, item in enumerate(golden_data):
        print(f"  [{i+1}/{len(golden_data)}] Query: {item['question']}")
        
        # Run agent
        state = agent.run(item['question'], repo_name)
        
        # Extract results for RAGAS (0.4.x schema)
        results.append({
            "user_input": item['question'],
            "response": state['messages'][-1]['content'] if state['messages'] else "",
            "contexts": [c['content'] for c in state['context']],
            "reference": item['ground_truth']
        })

    # 3. Format for RAGAS
    # Save raw results first for safety in case RAGAS fails
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/agent_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    dataset = Dataset.from_list(results)

    # 4. Setup LLM Judge (Using official Ragas wrapper for Groq)
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Configure the LLM for Judging (Strictly forcing n=1 for Groq)
    class GroqLLM(ChatOpenAI):
        def _generate(self, *args, **kwargs):
            if "n" in kwargs:
                kwargs["n"] = 1
            return super()._generate(*args, **kwargs)

    llm = GroqLLM(
        model_name="llama-3.1-8b-instant",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        max_retries=3,
    )
    
    # Wrap to ensure n=1 is always used
    ragas_llm = LangchainLLMWrapper(llm)

    # Configure the Embeddings (Local BGE-M3)
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)

    # 5. Run Evaluation
    print("Calculating RAGAS metrics (Faithfulness & Relevancy)...")
    try:
        # We'll run one metric at a time to isolate errors
        score = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings
        )
        
        # 6. Output Results
        print("\n" + "="*40)
        print("RAGAS EVALUATION SUMMARY")
        print("="*40)
        df = score.to_pandas()
        print(df[["question", "faithfulness", "answer_relevancy"]].head())
        print("-" * 40)
        print(f"Average Faithfulness: {df['faithfulness'].mean():.4f}")
        print(f"Average Relevancy: {df['answer_relevancy'].mean():.4f}")
        print("="*40)
        
        # Save results
        df.to_csv("evaluation/ragas_results.csv", index=False)
        print("Detailed results saved to evaluation/ragas_results.csv")
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pypa/sampleproject"
    run_ragas_evaluation(url)
