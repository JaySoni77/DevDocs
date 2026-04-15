import os
import json
import random
from typing import List, Dict, Any, Optional
from groq import Groq
import chromadb
from dotenv import load_dotenv

load_dotenv()

class DatasetGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Synthetic generation requires an LLM.")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate_qa_triplet(self, chunk: str, filename: str) -> Optional[Dict[str, str]]:
        """
        Generate a (Question, Context, Ground Truth) triplet from a single code chunk.
        """
        prompt = f"""You are a Technical QA Generator. 
Given the following code/documentation chunk from the file '{filename}', 
generate a realistic developer question and its definitive answer (ground truth) 
based ONLY on this chunk.

CHUNK:
{chunk}

Rules:
1. The question should be specific (e.g., 'How do I X?' or 'What does Y do?').
2. The answer must be detailed and include any relevant code or config values from the chunk.
3. If the chunk is just boilerplate (license, metadata), return 'SKIP'.

Output Format:
QUESTION: [Your Question]
ANSWER: [Your Ground Truth Answer]"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            if "SKIP" in content.upper() or "QUESTION:" not in content:
                return None
            
            lines = content.split("\n")
            q = lines[0].replace("QUESTION:", "").strip()
            a = "\n".join(lines[1:]).replace("ANSWER:", "").strip()
            
            return {
                "question": q,
                "ground_truth": a,
                "context": chunk,
                "filename": filename
            }
        except Exception as e:
            print(f"Error generating triplet: {e}")
            return None

    def create_dataset(self, repo_name: str, persist_directory: str = "./test_chroma_db", num_samples: int = 10):
        """
        Fetch random chunks from ChromaDB and generate the dataset.
        """
        client = chromadb.PersistentClient(path=persist_directory)
        collection = client.get_collection("devdocs_chunks")
        
        # Fetch all chunks for the repo
        all_data = collection.get(where={"repo": repo_name})
        if not all_data["documents"]:
            print(f"No chunks found for repo {repo_name} in {persist_directory}")
            return
        
        # Sampling
        indices = list(range(len(all_data["documents"])))
        random.shuffle(indices)
        
        dataset = []
        print(f"Generating {num_samples} QA pairs for {repo_name}...")
        
        for idx in indices:
            if len(dataset) >= num_samples:
                break
            
            chunk = all_data["documents"][idx]
            meta = all_data["metadatas"][idx]
            filename = meta.get("filename", "Unknown")
            
            triplet = self.generate_qa_triplet(chunk, filename)
            if triplet:
                dataset.append(triplet)
                print(f"  [{len(dataset)}/{num_samples}] Generated question for {filename}")

        # Save to file
        output_path = "evaluation/golden_dataset.json"
        os.makedirs("evaluation", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=4)
        
        print(f"Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    import sys
    repo = sys.argv[1] if len(sys.argv) > 1 else "pypa/sampleproject"
    generator = DatasetGenerator()
    generator.create_dataset(repo_name=repo, num_samples=10)
