import os
import chromadb
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from ingestion.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.compressor import Compressor

class HybridRetriever:
    def __init__(self, collection_name: str = "devdocs_chunks", persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedder: Optional[Embedder] = None
        self.reranker: Optional[Reranker] = None
        self.compressor: Optional[Compressor] = None

    def set_embedder(self, embedder: Embedder):
        self.embedder = embedder

    def set_reranker(self, reranker: Reranker):
        self.reranker = reranker

    def set_compressor(self, compressor: Compressor):
        self.compressor = compressor

    def reciprocal_rank_fusion(self, dense_results: List[str], sparse_results: List[str], k: int = 60) -> List[str]:
        """
        Combine two ranked lists using RRF.
        Returns a sorted list of chunk IDs.
        """
        scores = {}
        for rank, chunk_id in enumerate(dense_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        for rank, chunk_id in enumerate(sparse_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
        
        # Sort by score descending
        sorted_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [item[0] for item in sorted_ids]

    def retrieve(self, query: str, repo_name: str, top_k_retrieval: int = 50, top_k_rerank: int = 5, hyde_doc: Optional[str] = None, use_reranker: bool = False, use_compression: bool = False) -> Dict[str, Any]:
        if not self.embedder:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        # 1. Dense Search 1 (Rewritten Query)
        query_vector = self.embedder.embed_text(query)
        dense_hits_1 = self.collection.query(
            query_embeddings=[query_vector],
            n_results=50,
            where={"repo": repo_name}
        )
        dense_ids_1 = dense_hits_1["ids"][0]
        
        # 1b. Dense Search 2 (HyDE Document) - DUAL RETRIEVAL
        dense_ids_2 = []
        if hyde_doc:
            hyde_vector = self.embedder.embed_text(hyde_doc)
            dense_hits_2 = self.collection.query(
                query_embeddings=[hyde_vector],
                n_results=50,
                where={"repo": repo_name}
            )
            dense_ids_2 = dense_hits_2["ids"][0]

        # 2. Sparse Search (BM25)
        # Fetch all documents for this repo to build BM25 index
        # NOTE: For local dev, we fetch all. In production (Qdrant), this is native.
        all_chunks = self.collection.get(where={"repo": repo_name})
        if not all_chunks["ids"]:
            return []
            
        tokenized_corpus = [doc.split() for doc in all_chunks["documents"]]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split()
        
        # Get BM25 scores and rank them
        doc_scores = bm25.get_scores(tokenized_query)
        # Zip IDs with scores
        id_scores = list(zip(all_chunks["ids"], doc_scores))
        sparse_ids = [id_score[0] for id_score in sorted(id_scores, key=lambda x: x[1], reverse=True)]
        
        # 3. Fuse Results (Rewritten + Sparse + HyDE)
        # Combine all dense ID sources
        all_dense_ids = dense_ids_1 + dense_ids_2
        fused_ids = self.reciprocal_rank_fusion(all_dense_ids, sparse_ids)[:top_k_retrieval]
        
        # 4. Fetch full data for fused IDs
        retrieved_chunks = []
        chunk_data = self.collection.get(ids=fused_ids)
        
        # Map IDs back to full objects
        id_to_data = {
            id_: {"content": doc, "metadata": meta}
            for id_, doc, meta in zip(chunk_data["ids"], chunk_data["documents"], chunk_data["metadatas"])
        }
        
        for id_ in fused_ids:
            if id_ in id_to_data:
                retrieved_chunks.append(id_to_data[id_])

        # 5. Reranking (optional)
        final_chunks = retrieved_chunks
        if use_reranker and self.reranker:
            print(f"Reranking top {len(retrieved_chunks)} chunks...")
            final_chunks = self.reranker.rerank(query, retrieved_chunks, top_k=top_k_rerank)

        # 6. Compression (optional)
        compressed_context = None
        if use_compression and self.compressor:
            print(f"Compressing top {len(final_chunks)} chunks...")
            compressed_context = self.compressor.compress_chunks(query, final_chunks)

        return {
            "chunks": final_chunks,
            "compressed_context": compressed_context,
            "raw_total": len(retrieved_chunks)
        }

if __name__ == "__main__":
    # Test Retriever
    embedder = Embedder()
    retriever = HybridRetriever(persist_directory="./test_chroma_db")
    retriever.set_embedder(embedder)
    
    # Assuming test data was indexed by Indexer
    results = retriever.retrieve("Welcome to DevDocs", repo_name="test_repo")
    print(f"Retrieved {len(results)} results:")
    for r in results:
        print(f"[{r['metadata']['filename']}] {r['content'][:100]}")
