import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        """
        Initialize the BGE Cross-Encoder reranker.
        """
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Loading reranker model: {model_name} on {device}...")
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of chunks based on their relevance to the query.
        Each chunk is expected to have a 'content' field.
        """
        if not chunks:
            return []

        # Prepare pairs for the cross-encoder: (Query, Chunk_Content)
        pairs = [[query, chunk["content"]] for chunk in chunks]
        
        # Predict relevance scores
        # scores range from 0.0 to 1.0 (or raw logits depending on the model)
        scores = self.model.predict(pairs)
        
        # Zip chunks with their scores
        ranked_chunks = list(zip(chunks, scores))
        
        # Sort by score descending
        sorted_chunks = sorted(ranked_chunks, key=lambda x: x[1], reverse=True)
        
        # Return the top_k chunks with their rerank scores added to metadata
        final_results = []
        for chunk, score in sorted_chunks[:top_k]:
            chunk["metadata"]["rerank_score"] = float(score)
            final_results.append(chunk)
            
        return final_results

if __name__ == "__main__":
    # Small smoke test
    reranker = Reranker()
    test_query = "How to install the project?"
    test_chunks = [
        {"content": "To install, run pip install -r requirements.txt", "metadata": {}},
        {"content": "DevDocs is an AI project for developers.", "metadata": {}},
        {"content": "The sky is blue today.", "metadata": {}}
    ]
    
    results = reranker.rerank(test_query, test_chunks, top_k=2)
    print(f"Reranking Results for '{test_query}':")
    for r in results:
        print(f"Score: {r['metadata']['rerank_score']:.4f} | Content: {r['content']}")
