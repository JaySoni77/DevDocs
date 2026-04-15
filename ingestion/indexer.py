import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from ingestion.embedder import Embedder

class Indexer:
    def __init__(self, collection_name: str = "devdocs_chunks", persist_directory: str = "./chroma_db"):
        """
        Initialize the Indexer with ChromaDB.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder: Optional[Embedder] = None

    def set_embedder(self, embedder: Embedder):
        self.embedder = embedder

    def index_chunks(self, chunks: List[Dict[str, Any]], repo_name: str):
        """
        Index a list of chunks into ChromaDB.
        Each chunk expects: {filename, header, content, tokens}
        """
        if not self.embedder:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{repo_name}_{chunk['filename']}_{i}"
            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append({
                "repo": repo_name,
                "filename": chunk["filename"],
                "header": chunk["header"],
                "tokens": chunk["tokens"]
            })
            
        # Batch embed documents
        print(f"Embedding {len(documents)} chunks for repo '{repo_name}'...")
        embeddings = self.embedder.embed_batch(documents)

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully indexed {len(ids)} chunks.")

if __name__ == "__main__":
    # Test Indexer
    embedder = Embedder()
    indexer = Indexer(persist_directory="./test_chroma_db")
    indexer.set_embedder(embedder)
    
    test_chunks = [
        {"filename": "test.py", "header": "func", "content": "def hello(): pass", "tokens": 5},
        {"filename": "readme.md", "header": "intro", "content": "# Welcome to DevDocs", "tokens": 10}
    ]
    indexer.index_chunks(test_chunks, repo_name="test_repo")
