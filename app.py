import os
import gradio as gr
import json
import time
from datetime import datetime
from ingestion.github_loader import GitHubLoader
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder
from ingestion.indexer import Indexer
from ingestion.guards import IngestionGuards
from retrieval.query_processor import QueryProcessor
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.compressor import Compressor
from agent.generator import AnswerGenerator
from agent.graph import AgentGraph
from agent.config import OPTIMIZED_CONFIG
from utils.cost_tracker import CostTracker

# Configuration
PERSIST_DIR = "./chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# Production Guards & Rate Limiting
guards = IngestionGuards()
INGEST_COOLDOWN = 3600  # 1 hour per IP
last_ingest_time = {} # IP -> last_time

# Initialize Components
cost_tracker = CostTracker()
loader = GitHubLoader()
chunker = Chunker()
embedder = Embedder()
indexer = Indexer(persist_directory=PERSIST_DIR)
indexer.set_embedder(embedder)
processor = QueryProcessor(cost_tracker=cost_tracker)
generator = AnswerGenerator(cost_tracker=cost_tracker)
reranker = Reranker()
compressor = Compressor()

# Shared Retriever
retriever = HybridRetriever(persist_directory=PERSIST_DIR)
retriever.set_embedder(embedder)
retriever.set_reranker(reranker)
retriever.set_compressor(compressor)

agent = AgentGraph(retriever, processor, generator, OPTIMIZED_CONFIG)

# Supported Repo (Global for simplicity)
current_repo = None

def index_repo(repo_url, request: gr.Request):
    global current_repo
    ip = request.client.host
    
    # Rate Limit Check
    if ip in last_ingest_time:
        elapsed = time.time() - last_ingest_time[ip]
        if elapsed < INGEST_COOLDOWN:
            wait_min = int((INGEST_COOLDOWN - elapsed) // 60)
            yield f"⚠️ Rate limit hit. Please wait {wait_min} minutes before indexing another repository."
            return

    try:
        if not repo_url:
            return "Please provide a valid GitHub URL."
        
        guards.validate_repo_url(repo_url)
        info = loader.parse_github_url(repo_url)
        repo_name = f"{info['owner']}/{info['repo']}"
        
        yield f"1/4 | Fetching files from {repo_name}..."
        files = loader.fetch_repo_files(repo_url)
        
        # Guard Check
        guards.validate_file_count(files)
        
        yield f"2/4 | Processing {len(files)} files (using embedding cache)..."
        all_chunks = []
        for i, f in enumerate(files):
            if i % 10 == 0:
                yield f"2/4 | Processing file {i+1}/{len(files)}: {f['filename']}..."
            content = loader.fetch_file_content(f['url'])
            chunks = chunker.process_file(f['filename'], content)
            all_chunks.extend(chunks)
        
        yield f"3/4 | Embedding and Indexing {len(all_chunks)} chunks..."
        indexer.index_chunks(all_chunks, repo_name)
        
        current_repo = repo_name
        last_ingest_time[ip] = time.time() # Mark success
        yield f"✅ Success! Indexed {repo_name}. You can now ask questions."
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def chat_response(message, history, use_hyde, use_reranker, use_compression):
    global current_repo
    if not current_repo:
        yield "⚠️ Please index a repository first using the input above."
        return

    try:
        # Update config based on UI toggles
        agent.config.use_hyde = use_hyde
        agent.config.use_reranker = use_reranker
        agent.config.use_compression = use_compression
        
        state = agent.run(message, current_repo)
        response = state['messages'][-1]['content'] if state['messages'] else "I'm sorry, I couldn't generate an answer."
        
        summary = cost_tracker.get_session_summary()
        metrics = f"\n\n---\n**Session Metrics:** {summary['total_tokens']} tokens | ${summary['total_cost_usd']:.4f} USD"
        yield response + metrics
    except Exception as e:
        yield f"❌ Error: {str(e)}"

def log_feedback(data: gr.LikeData):
    """Save feedback to a JSONL file for DSPy optimization."""
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "repo": current_repo,
        "index": data.index,
        "value": data.value, # "liked" or "disliked"
        "rating": 1 if data.liked else 0
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    print(f"Feedback logged: {feedback_entry}")

# UI Assembly with Premium Theme
# Note: theme is moved back to Blocks for compatibility with earlier Gradio versions (e.g., on Colab)
app_theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
with gr.Blocks() as demo:
    with gr.Column(elem_id="container"):
        gr.Markdown("# 🛸 DevDocs AI: Intelligent Codebase Agent")
        gr.Markdown("Enterprise-grade RAG with SOTA BGE-v2 models and Multi-Hop reasoning.")
        
        with gr.Tabs():
            with gr.Tab("🛰️ Repository Indexing"):
                with gr.Row():
                    repo_input = gr.Textbox(
                        label="GitHub Repository URL", 
                        placeholder="https://github.com/owner/repo",
                        scale=4
                    )
                    index_btn = gr.Button("Build Knowledge Base", variant="primary", scale=1)
                
                status_output = gr.Markdown("Ready to index.")
                
                with gr.Accordion("Retriever Tuning", open=False):
                    gr.Markdown("Fine-tune the RAG strategy for specific codebase architectures.")
                    use_hyde = gr.Checkbox(label="Hypothetical Doc Embeddings (HyDE)", value=OPTIMIZED_CONFIG.use_hyde)
                    use_reranker = gr.Checkbox(label="BAAI Cross-Encoder Reranking", value=OPTIMIZED_CONFIG.use_reranker)
                    use_compression = gr.Checkbox(label="Contextual Prompt Compression", value=OPTIMIZED_CONFIG.use_compression)
            
            with gr.Tab("💬 Technical Agent"):
                chatbot = gr.Chatbot(
                    label="DevDocs Assistant", 
                    show_label=False,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=DevDocs")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Ask a technical question (e.g., 'How is the auth flow implemented?')",
                        show_label=False,
                        scale=9
                    )
                    submit_btn = gr.Button("Ask", variant="primary", scale=1)
                
                clear = gr.Button("Clear Session", variant="secondary")

    # Connect components
    index_btn.click(index_repo, inputs=[repo_input], outputs=[status_output])

    def respond(message, chat_history, hyde, rerank, compress):
        # chat_history is a list of dicts in Gradio 6.x
        for chunk in chat_response(message, chat_history, hyde, rerank, compress):
            new_history = chat_history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": chunk}
            ]
            yield {msg: "", chatbot: new_history}

    msg.submit(respond, [msg, chatbot, use_hyde, use_reranker, use_compression], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot, use_hyde, use_reranker, use_compression], [msg, chatbot])
    chatbot.like(log_feedback, None, None)
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        footer_links=["gradio", "settings"],
        theme=app_theme
    )
