import os
import math
from typing import List, Dict, Any, Tuple
from collections import defaultdict

# Core RAG/Vector DB libraries
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, 
    ScoredPoint, 
    SearchParams, 
    # RE-IMPORTING the models designed to fix this validation error
    NamedVector, 
    NamedSparseVector 
)
# Embedding Model
from sentence_transformers import SentenceTransformer

# Sparse Vector (BM25-like) Dependencies 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Configuration (MUST match ingestion_pipeline.py settings) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_reports_2025"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STOP_WORDS = set(stopwords.words('english'))

# --- Retrieval-Specific Configuration ---
SEARCH_LIMIT = 5 # Number of chunks to retrieve
RRF_K = 60 # Constant for Reciprocal Rank Fusion (RRF)

# --- Sparse Vector (BM25-like/TF-IDF) Functions (Unchanged) ---

def tokenize(text: str) -> List[str]:
    """Tokenizes text, removes stopwords, and converts to lowercase."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in STOP_WORDS]

def get_sparse_vector_query(text: str) -> SparseVector:
    """
    Generates a SparseVector for the query using Term Frequency (TF).
    """
    tokens = tokenize(text)
    term_frequencies: Dict[str, int] = defaultdict(int)
    for token in tokens:
        term_frequencies[token] += 1

    indices = []
    values = []
    
    for token, tf in term_frequencies.items():
        if tf > 0:
            # Simple hash for index mapping (must match ingestion)
            index = hash(token) % 2**30 
            indices.append(index)
            values.append(float(tf))

    # Qdrant expects indices and values to be sorted
    sorted_pairs = sorted(zip(indices, values))
    sorted_indices = [idx for idx, val in sorted_pairs]
    sorted_values = [val for idx, val in sorted_pairs]
    
    return SparseVector(indices=sorted_indices, values=sorted_values)


def rank_fusion(dense_results: List[ScoredPoint], sparse_results: List[ScoredPoint], k: int, limit: int) -> List[ScoredPoint]:
    """Applies Reciprocal Rank Fusion to merge results."""
    fused_scores = defaultdict(float)
    all_results_map = {p.id: p for p in dense_results + sparse_results}

    # 1. Fuse Dense Results
    for rank, point in enumerate(dense_results):
        point_id = point.id
        fused_scores[point_id] += 1 / (k + rank + 1)

    # 2. Fuse Sparse Results
    for rank, point in enumerate(sparse_results):
        point_id = point.id
        fused_scores[point_id] += 1 / (k + rank + 1)
    
    # 3. Sort by the new fused score
    sorted_fused_items = sorted(
        fused_scores.items(), key=lambda item: item[1], reverse=True
    )

    # 4. Reconstruct ScoredPoint list with new scores
    final_fused_results = []
    for point_id, fused_score in sorted_fused_items:
        original_point = all_results_map[point_id]
        final_fused_results.append(
            ScoredPoint(
                id=original_point.id,
                version=original_point.version,
                score=fused_score,
                payload=original_point.payload,
                vector=None, 
                shard_key=original_point.shard_key
            )
        )
    return final_fused_results[:limit]


# --- Main Retrieval Function ---

def run_retrieval(query: str):
    """Executes the full Hybrid Search against the Qdrant collection."""
    
    print("\nStarting Hybrid Search Retrieval...")
    
    # 1. Initialize Clients and Model
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=600) 
    print(f"Loading Dense Embedding Model: {DENSE_MODEL_NAME}...")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    # 2. Generate Query Vectors
    print(f"Processing Query: '{query}'")
    
    dense_vector = dense_model.encode(query).tolist()
    sparse_vector = get_sparse_vector_query(query)

    # 3. Perform Hybrid Search via Two Separate Searches (Manual RRF)
    print("🔍 Performing Separate Dense and Sparse Searches...")

    # A. Dense (Semantic) Search 
    # Use NamedVector for the dense search (tuple format for query_vector)
    dense_results: List[ScoredPoint] = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense_vectors", dense_vector), 
        limit=SEARCH_LIMIT * 5, 
        with_payload=True,
    )

    # B. Sparse (Keyword) Search 
    # FINAL ATTEMPT: Use the dedicated `NamedSparseVector` model, which is designed
    # to accept the SparseVector object and bypass the NamedVector validation error.
    sparse_query_vector = NamedSparseVector(
        name="sparse_vectors",
        vector=sparse_vector,
    )
    
    # We must pass the NamedSparseVector object as the positional argument.
    sparse_results: List[ScoredPoint] = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        # Pass the NamedSparseVector object, not the dictionary or tuple!
        query_vector=sparse_query_vector, 
        limit=SEARCH_LIMIT * 5, 
        with_payload=True,
    )
    
    # C. Apply RRF
    search_results = rank_fusion(dense_results, sparse_results, RRF_K, SEARCH_LIMIT)


    # 4. Display Results
    print(f"\n✅ Hybrid Search Complete. Retrieved {len(search_results)} documents.")
    print("------------------------------------------------------------------")
    
    if not search_results:
        print("No results found for the query.")
        return

    for i, result in enumerate(search_results):
        payload = result.payload
        company = payload.get('company', 'N/A')
        page = payload.get('page_number', 'N/A')
        text = payload.get('text', 'N/A')
        
        # Display the combined RRF score
        print(f"RANK {i+1} | FUSED SCORE: {result.score:.4f}")
        print(f"SOURCE: {company} Annual Report, Page {page}")
        print("---")
        # Display the retrieved text chunk
        print(text[:400].strip() + "...") 
        print("------------------------------------------------------------------")


if __name__ == "__main__":
    # Example Query 
    QUERY = "What is the capital expenditure planned for the retail sector in 2025 and why?"
    
    # Run the retrieval pipeline
    run_retrieval(QUERY)