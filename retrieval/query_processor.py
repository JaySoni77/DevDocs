import os
from typing import Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class QueryProcessor:
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile", cost_tracker=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("Warning: GROQ_API_KEY not found. QueryProcessor will return original queries.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        self.model = model
        self.cost_tracker = cost_tracker
        self.cost_tracker = cost_tracker

    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a vague user query into a precise, keyword-rich search query.
        """
        if not self.client:
            return query

        prompt = f"""You are an expert Search Query Optimizer for technical documentation. 
Your task is to rewrite the user's natural language question into a precise, keyword-rich query that will maximize retrieval from a vector database of source code and documentation.

Rules:
1. Extract core technical concepts (libraries, function names, patterns).
2. Remove conversational filler ("how do i", "can you tell me").
3. If the query is about an error, include common error variations.
4. Output ONLY the rewritten query.

Original Question: {query}
Optimized Retrieval Query:"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in rewrite_query: {e}")
            return query

    def generate_hyde_document(self, query: str) -> str:
        """
        Generate a hypothetical technical documentation snippet that would answer this question.
        """
        if not self.client:
            return query

        prompt = f"""You are a senior Software Engineer writing technical documentation. 
Generate a concise, highly detailed hypothetical documentation snippet that would directly answer this question: "{query}".

Requirements:
1. Use professional, technical language.
2. Include hypothetical code snippets, configuration examples, or API signatures where relevant.
3. Focus on precise terminology (e.g., "middleware", "dependency injection", "rate limiting").
4. Do NOT include any preamble or meta-commentary (like "Here is the snippet").
5. Return ONLY the documentation content.

Hypothetical Documentation:"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=300
            )
            if self.cost_tracker and response.usage:
                self.cost_tracker.log_usage(
                    self.model, 
                    response.usage.prompt_tokens, 
                    response.usage.completion_tokens,
                    {"action": "generate_hyde"}
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in generate_hyde_document: {e}")
            return query

    def generate_multihop_query(self, query: str, context: str) -> str:
        """
        Given the original query and the current context, determine what is missing
        and generate a new search query to find the missing information.
        """
        if not self.client:
            return query

        prompt = f"""You are an expert technical researcher. 
Original Question: {query}
Current Context: {context[:2000]}

Based on the question and the context already found, what specific technical information 
is STILL MISSING to provide a complete answer? 
Generate a concise, keyword-rich search query to find that missing information.

If no information is missing, output 'NONE'.
MISSING INFO QUERY:"""
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in generate_multihop_query: {e}")
            return query

if __name__ == "__main__":
    # Test QueryProcessor
    processor = QueryProcessor()
    test_query = "how to run tests here?"
    
    rewritten = processor.rewrite_query(test_query)
    print(f"Original: {test_query}")
    print(f"Rewritten: {rewritten}")
    
    hyde = processor.generate_hyde_document(test_query)
    print(f"HyDE Doc Snippet: {hyde[:100]}...")
