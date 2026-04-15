import os
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class AnswerGenerator:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", cost_tracker=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found. AnswerGenerator will return pre-formatted context.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        self.model = model
        self.cost_tracker = cost_tracker

    def generate_answer(self, query: str, context: List[Dict[str, Any]], compressed_context: Optional[str] = None) -> str:
        """
        Generate a structured, technical answer based on the provided context with citations.
        """
        if not self.client:
            # Fallback for dev mode
            return f"Answer based on {len(context)} chunks: " + (compressed_context or "No compressed context available.")

        # Prepare context with source markers
        if compressed_context:
            formatted_context = f"COMPRESSED RELEVANT INFO:\n{compressed_context}"
        else:
            formatted_context = ""
            for i, chunk in enumerate(context):
                filename = chunk['metadata'].get('filename', 'Unknown')
                content = chunk['content']
                formatted_context += f"--- SOURCE {i+1}: {filename} ---\n{content}\n\n"

        system_prompt = """You are 'DevDocs AI', a senior Technical Architect and Documentation Expert.
Your goal is to answer developer questions based ONLY on the provided source code context.

RULES:
1. TECHNICAL PRECISION: Provide exact code patterns and configuration values.
2. CITATIONS: Every factual claim or code snippet MUST be followed by a source citation like [Source: filename].
3. STRUCTURE: Use Markdown (headers, code blocks, bullet points).
4. GROUNDING: If the answer is not in the context, say 'I do not have enough information in the provided repository to answer this.'
5. NO HALLUCINATION: Do not mention libraries or files not present in the context.
6. CONCISION: Be direct. Developers value their time.

Output format example:
## Implementation Guide
To use the loader, initialize the class... [Source: loader.py]
```python
loader = Loader()
```
"""

        user_prompt = f"""USER QUESTION: {query}

CONTEXT PROVIDED:
{formatted_context}

Grounded, Cited Answer:"""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.1 # Low temperature for factual consistency
            )
            if self.cost_tracker and response.usage:
                self.cost_tracker.log_usage(
                    self.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    {"action": "generate_answer"}
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in generate_answer: {e}")
            return "I encountered an error while generating the answer. Please check the logs."

if __name__ == "__main__":
    # Test Generator
    generator = AnswerGenerator()
    test_q = "How do I configure the GitHub token?"
    test_context = [
        {"content": "The GitHubLoader requires a GITHUB_TOKEN environment variable or an api_key parameter in the constructor.", "metadata": {"filename": "ingestion/github_loader.py"}}
    ]
    
    answer = generator.generate_answer(test_q, test_context)
    print(f"Query: {test_q}")
    print(f"Answer:\n{answer}")
