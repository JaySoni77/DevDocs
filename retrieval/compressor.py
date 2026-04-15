import os
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class Compressor:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found. Compressor will return original chunks.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        self.model = model

    def compress_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Compress the top-ranked chunks into a single, highly relevant context block.
        It strips out sentences that do not contribute to answering the query.
        """
        if not chunks:
            return ""
        
        if not self.client:
            # Fallback: Just join the chunks
            return "\n\n".join([f"Source: {c['metadata'].get('filename', 'Unknown')}\n{c['content']}" for c in chunks])

        # Prepare context for the prompt
        raw_context = "\n\n".join([f"Chunk ID: {i}\nContent: {c['content']}" for i, c in enumerate(chunks)])

        prompt = f"""You are a Precise Context Compressor. 
Your goal is to extract ONLY the sentences from the provided chunks that are directly relevant to the user query.

Rules:
1. Maintain the technical detail of the relevant parts.
2. Remove all conversational filler, licensing headers, or unrelated code blocks.
3. If multiple chunks provide the same info, keep only the clearest one.
4. Output ONLY the compressed context, separated by newlines.

User Query: "{query}"

Raw Chunks:
{raw_context}

Compressed Context:"""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in compress_chunks: {e}")
            return "\n\n".join([c['content'] for c in chunks])

if __name__ == "__main__":
    # Test Compressor
    compressor = Compressor()
    test_q = "How do I install the requirements?"
    test_chunks = [
        {"content": "Installing the project is easy. First, ensure you have Python 3.10+. Then, run pip install -r requirements.txt. This file contains all dependencies.", "metadata": {"filename": "README.md"}},
        {"content": "The project uses the MIT license. You are free to use it for personal projects as long as you include the original license file.", "metadata": {"filename": "LICENSE.txt"}}
    ]
    
    compressed = compressor.compress_chunks(test_q, test_chunks)
    print(f"Query: {test_q}")
    print(f"Compressed Context:\n{compressed}")
