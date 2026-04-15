import os
from typing import List, Union
import json
import hashlib
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", cache_file: str = "logs/embed_cache.json"):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache_file = cache_file
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, List[float]]:
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed_text(self, text: str) -> List[float]:
        text_hash = self._get_hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        embedding = self.model.encode(text, convert_to_numpy=True).tolist()
        self.cache[text_hash] = embedding
        self._save_cache()
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        results = []
        to_embed = []
        to_embed_indices = []

        for i, text in enumerate(texts):
            text_hash = self._get_hash(text)
            if text_hash in self.cache:
                results.append(self.cache[text_hash])
            else:
                results.append(None)
                to_embed.append(text)
                to_embed_indices.append(i)

        if to_embed:
            print(f"Embedding {len(to_embed)} new items...")
            new_embeddings = self.model.encode(to_embed, convert_to_numpy=True).tolist()
            for idx, emb in zip(to_embed_indices, new_embeddings):
                results[idx] = emb
                self.cache[self._get_hash(texts[idx])] = emb
            self._save_cache()

        return results

if __name__ == "__main__":
    # Small smoke test
    embedder = Embedder()
    test_text = "What is DevDocs AI?"
    vector = embedder.embed_text(test_text)
    print(f"Vector dimension: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")
