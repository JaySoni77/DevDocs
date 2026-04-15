import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Pricing in USD per 1M tokens (Approximate Groq/Standard rates)
PRICING = {
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "llama-3.1-70b-versatile": {"prompt": 0.59, "completion": 0.79},
}

class CostTracker:
    def __init__(self, log_file: str = "logs/usage_stats.json"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.current_session_cost = 0.0
        self.current_session_tokens = 0

    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the cost of an LLM call in USD."""
        if model not in PRICING:
            # Default to 8B pricing if unknown
            model_price = PRICING["llama-3.1-8b-instant"]
        else:
            model_price = PRICING[model]
        
        cost = (prompt_tokens / 1_000_000) * model_price["prompt"] + \
               (completion_tokens / 1_000_000) * model_price["completion"]
        
        return cost

    def log_usage(self, model: str, prompt_tokens: int, completion_tokens: int, metadata: Optional[Dict[str, Any]] = None):
        """Append usage data to a persistent log."""
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        self.current_session_cost += cost
        self.current_session_tokens += (prompt_tokens + completion_tokens)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "metadata": metadata or {}
        }
        
        # Simple append-to-JSON-list logic (not efficient for millions of logs, but fine for MVP)
        history = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    history = json.load(f)
            except:
                history = []
        
        history.append(entry)
        with open(self.log_file, "w") as f:
            json.dump(history[-1000:], f, indent=2) # Keep last 1000 entries
        
        return cost

    def get_session_summary(self):
        return {
            "total_tokens": self.current_session_tokens,
            "total_cost_usd": self.current_session_cost
        }

if __name__ == "__main__":
    # Test CostTracker
    tracker = CostTracker()
    cost = tracker.log_usage("llama-3.1-8b-instant", 1000, 500)
    print(f"Test Cost: ${cost:.6f}")
    print(f"Session Summary: {tracker.get_session_summary()}")
