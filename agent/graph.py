from typing import List, Dict, Any, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    """
    The state of our RAG agent.
    """
    query: str                          # Original user query
    current_query: str                  # Potentially rewritten/multi-hop query
    repo_name: str                      # Target repository
    context: List[Dict[str, Any]]       # All retrieved chunks
    compressed_context: Optional[str]   # Final noise-free context
    is_sufficient: bool                 # Does context answer the query?
    hop_count: int                      # Current search round
    messages: Annotated[List[Dict[str, str]], operator.add] # Conversation history
    total_tokens: int                   # Accumulated tokens
    total_cost: float                   # Accumulated cost in USD

class AgentGraph:
    def __init__(self, retriever, processor, generator, config):
        self.retriever = retriever
        self.processor = processor
        self.generator = generator
        self.config = config
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("grade", self.grade_node)
        workflow.add_node("generate", self.generate_node)

        # Build Graph
        workflow.set_entry_point("retrieve")
        
        # Retrieval -> Grade
        workflow.add_edge("retrieve", "grade")
        
        # Grade -> Decide (Edge with logic)
        workflow.add_conditional_edges(
            "grade",
            self.decide_next_step,
            {
                "continue": "retrieve", # Multi-hop
                "generate": "generate", # Sufficient
                "end": END              # Final answer
            }
        )
        
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Orchestrates the retrieval pipeline.
        """
        hop = state.get("hop_count", 0)
        print(f"--- [Node: Retrieve] Hop {hop + 1} ---")
        
        # Determine the query for this hop
        if hop == 0:
            current_query = self.processor.rewrite_query(state["query"])
        else:
            # Multi-hop: generate query based on what's missing
            prev_context = "\n\n".join([c["content"] for c in state["context"]])
            current_query = self.processor.generate_multihop_query(state["query"], prev_context)
            if current_query.upper() == "NONE":
                print("No missing info identified. Ending retrieval.")
                return {"is_sufficient": True}

        # HyDE
        hyde_doc = None
        if self.config.use_hyde:
            hyde_doc = self.processor.generate_hyde_document(current_query)

        # Search
        results = self.retriever.retrieve(
            query=current_query,
            repo_name=state["repo_name"],
            top_k_retrieval=self.config.top_k_retrieval,
            top_k_rerank=self.config.top_k_rerank,
            hyde_doc=hyde_doc,
            use_reranker=self.config.use_reranker,
            use_compression=self.config.use_compression
        )

        return {
            "context": state["context"] + results["chunks"],
            "compressed_context": results["compressed_context"],
            "current_query": current_query,
            "hop_count": hop + 1
        }

    def grade_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Evaluates if the context is enough to answer.
        """
        print("--- [Node: Grade] ---")
        if not state["context"]:
            return {"is_sufficient": False}

        context_str = "\n\n".join([c["content"] for c in state["context"][:5]])
        prompt = f"""You are a Retrieval Grader. 
Evaluate if the following retrieved context is sufficient to answer the user's question completely.

Question: {state["query"]}
Context: {context_str}

Respond with ONLY 'YES' if it is sufficient, or 'NO' if more information is needed.
SUFFICIENT:"""

        try:
            # Use processor's client for grading
            response = self.processor.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.processor.model,
                temperature=0.0,
                max_tokens=5
            )
            if self.processor.cost_tracker and response.usage:
                self.processor.cost_tracker.log_usage(
                    self.processor.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    {"action": "grade_retrieval"}
                )
            result = response.choices[0].message.content.strip().upper()
            is_sufficient = "YES" in result
            print(f"Grading Result: {result}")
            return {"is_sufficient": is_sufficient}
        except Exception as e:
            print(f"Error in grade_node: {e}")
            return {"is_sufficient": True} # Fallback to avoid infinite loops

    def generate_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Final grounded answer generation.
        """
        print("--- [Node: Generate] ---")
        answer = self.generator.generate_answer(
            query=state["query"],
            context=state["context"],
            compressed_context=state["compressed_context"]
        )
        return {"messages": [{"role": "assistant", "content": answer}]}

    def decide_next_step(self, state: AgentState) -> str:
        """
        Determines if we should do another hop or generate.
        """
        if state["is_sufficient"] or state["hop_count"] >= self.config.max_retrieval_rounds:
            return "generate"
        return "continue"

    def run(self, query: str, repo_name: str):
        initial_state = {
            "query": query,
            "repo_name": repo_name,
            "context": [],
            "hop_count": 0,
            "messages": [],
            "total_tokens": 0,
            "total_cost": 0.0
        }
        return self.workflow.invoke(initial_state)
