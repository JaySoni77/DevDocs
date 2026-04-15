import os
import json
import pandas as pd
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMEvaluator:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def evaluate_triplet(self, question: str, context: str, answer: str) -> Dict[str, float]:
        """
        Evaluate a single RAG response for Faithfulness and Relevancy.
        """
        prompt = f"""You are an Expert RAG Evaluator. 
Rate the following Answer based on the provided Context and Question.

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Metrics (0.0 to 1.0):
1. FAITHFULNESS: Is the answer derived ONLY from the provided context? (1.0 = perfect, 0.0 = contains hallucinations or external info)
2. RELEVANCY: Does the answer accurately and completely address the question? (1.0 = perfect, 0.0 = irrelevant or incomplete)

Output ONLY the scores in the following format:
FAITHFULNESS: [Score]
RELEVANCY: [Score]"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            
            # Parse scores
            scores = {}
            for line in content.split("\n"):
                if "FAITHFULNESS:" in line:
                    scores["faithfulness"] = float(line.split(":")[1].strip())
                elif "RELEVANCY:" in line:
                    scores["relevancy"] = float(line.split(":")[1].strip())
            
            return scores
        except Exception as e:
            print(f"Error evaluating triplet: {e}")
            return {"faithfulness": 0.0, "relevancy": 0.0}

def run_custom_evaluation(repo_url: str, results_json: str = "evaluation/agent_results.json"):
    print(f"--- Starting Custom LLM Evaluation for {repo_url} ---")
    
    if not os.path.exists(results_json):
        print(f"Error: Agent results not found at {results_json}")
        return

    with open(results_json, "r") as f:
        results = json.load(f)

    evaluator = LLMEvaluator()
    
    final_scores = []
    print(f"Evaluating {len(results)} samples...")
    for i, item in enumerate(results):
        print(f"  [{i+1}/{len(results)}] Scoring: {item['user_input'][:50]}...")
        
        # Combine contexts if multiple
        context_str = "\n---\n".join(item['contexts'])
        
        scores = evaluator.evaluate_triplet(item['user_input'], context_str, item['response'])
        final_scores.append({
            "question": item['user_input'],
            "faithfulness": scores.get("faithfulness", 0.0),
            "relevancy": scores.get("relevancy", 0.0)
        })

    # Summary
    df = pd.DataFrame(final_scores)
    print("\n" + "="*40)
    print("CUSTOM LLM EVALUATION SUMMARY")
    print("="*40)
    print(df.head())
    print("-" * 40)
    print(f"Average Faithfulness: {df['faithfulness'].mean():.4f}")
    print(f"Average Relevancy: {df['relevancy'].mean():.4f}")
    print("="*40)
    
    # Save results
    df.to_csv("evaluation/custom_eval_results.csv", index=False)
    print("Detailed results saved to evaluation/custom_eval_results.csv")

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/pypa/sampleproject"
    run_custom_evaluation(url)
