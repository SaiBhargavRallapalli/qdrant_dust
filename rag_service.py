# rag_service.py

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
    # FIX: Re-importing the working models for hybrid search
    NamedVector, 
    NamedSparseVector 
)

# Embedding Model
from sentence_transformers import SentenceTransformer

# LLM/Generation Library
from google import genai
from google.genai.errors import APIError

# Sparse Vector Dependencies (Used for query vector generation)
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Configuration (Copied from retrieval_pipeline.py) ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_reports_2025"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STOP_WORDS = set(stopwords.words('english'))

# --- Retrieval-Specific Configuration (Copied) ---
SEARCH_LIMIT = 5 # Number of chunks to retrieve
RRF_K = 60 # Constant for Reciprocal Rank Fusion (RRF)
GEMINI_MODEL = "gemini-2.5-flash" 

# --- Retrieval Utility Functions (Unchanged) ---

def tokenize(text: str) -> List[str]:
    """Tokenizes text, removes stopwords, and converts to lowercase."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in STOP_WORDS]

def get_sparse_vector_query(text: str) -> SparseVector:
    """Generates a SparseVector for the query using Term Frequency (TF)."""
    tokens = tokenize(text)
    term_frequencies: Dict[str, int] = defaultdict(int)
    for token in tokens:
        term_frequencies[token] += 1

    indices = []
    values = []
    
    for token, tf in term_frequencies.items():
        if tf > 0:
            index = hash(token) % 2**30 
            indices.append(index)
            values.append(float(tf))

    sorted_pairs = sorted(zip(indices, values))
    sorted_indices = [idx for idx, val in sorted_pairs]
    sorted_values = [val for idx, val in sorted_pairs]
    
    return SparseVector(indices=sorted_indices, values=sorted_values)


def rank_fusion(dense_results: List[ScoredPoint], sparse_results: List[ScoredPoint], k: int, limit: int) -> List[Dict[str, Any]]:
    """Applies RRF and returns a clean list of context dictionaries."""
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

    # 4. Reformat the final results for the LLM
    final_context = []
    for point_id, fused_score in sorted_fused_items[:limit]:
        original_point = all_results_map[point_id]
        
        # Extract necessary fields and include the score for debugging/citation
        context_data = {
            "score": fused_score,
            "text": original_point.payload.get('text', 'N/A'),
            "company": original_point.payload.get('company', 'N/A'),
            "page_number": original_point.payload.get('page_number', 'N/A'),
        }
        final_context.append(context_data)
        
    return final_context


def retrieve_context(query: str) -> List[Dict[str, Any]]:
    """
    Executes the hybrid retrieval and returns a list of context dicts.
    Refactored from run_retrieval in retrieval_pipeline.py.
    """
    
    # 1. Initialize Clients and Model
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=600) 
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)

    # 2. Generate Query Vectors
    dense_vector = dense_model.encode(query).tolist()
    sparse_vector = get_sparse_vector_query(query)
    
    # FIX: The dictionary format is replaced by the object format that worked
    sparse_query_vector = NamedSparseVector(
        name="sparse_vectors",
        vector=sparse_vector,
    )
    
    # 3. Perform Hybrid Search via Two Separate Searches (Manual RRF)
    # A. Dense (Semantic) Search 
    dense_results: List[ScoredPoint] = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense_vectors", dense_vector), 
        limit=SEARCH_LIMIT * 5, 
        with_payload=True,
    )

    # B. Sparse (Keyword) Search 
    # FIX: Passing the NamedSparseVector object directly
    sparse_results: List[ScoredPoint] = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=sparse_query_vector, # Pass the object, NOT the tuple/dict
        limit=SEARCH_LIMIT * 5, 
        with_payload=True,
    )
    
    # C. Apply RRF and return the clean context list
    return rank_fusion(dense_results, sparse_results, RRF_K, SEARCH_LIMIT)


# --- Final RAG Service Function (Unchanged except for the necessary imports) ---

def run_rag_service(query: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes the full Retrieval-Augmented Generation (RAG) pipeline.
    
    Returns:
        Tuple[str, List[Dict]]: The final answer and the list of retrieved context chunks.
    """
    
    # --- 1. RETRIEVAL PHASE ---
    print("\n--- 1. RETRIEVAL: Hybrid Search ---")
    
    try:
        retrieved_chunks = retrieve_context(query)
        print(f"✅ Retrieved {len(retrieved_chunks)} relevant documents.")
        
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return "An error occurred during retrieval. Please check the Qdrant server connection.", []

    if not retrieved_chunks:
        print("Could not find relevant context.")
        return "I could not find any relevant information in the financial reports to answer your question.", []

    # --- 2. GENERATION PHASE ---
    print("\n--- 2. GENERATION: LLM Synthesis ---")
    
    try:
        # Initialize Gemini client
        if "GEMINI_API_KEY" not in os.environ:
             raise ValueError("GEMINI_API_KEY environment variable is not set.")
             
        gemini_client = genai.Client()
        
        # Format context for the LLM
        context_list = []
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk.get('company', 'N/A')
            page = chunk.get('page_number', 'N/A')
            text = chunk.get('text', 'N/A')
            context_list.append(f"--- SOURCE: {source} (Page {page}, RRF Score: {chunk['score']:.4f}) ---\n{text}")
        
        context_str = "\n\n".join(context_list)
        
        # Construct the detailed RAG prompt
        system_prompt = (
            "You are an expert financial analyst. Your task is to answer the user's question based "
            "ONLY on the provided context. Do not use external knowledge. "
            "For every piece of factual information, cite the source company and page number using "
            "the format [Source: Company, Page X]. "
            "If the information is not present in the context, state 'The required information is not available in the provided reports.' "
        )
        
        user_prompt = f"""
        CONTEXT:
        {context_str}

        QUESTION:
        {query}
        """

        print(f"Sending prompt to {GEMINI_MODEL}...")

        # Call the Gemini API
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[system_prompt, user_prompt],
        )
        
        return response.text, retrieved_chunks
        
    except APIError as e:
        error_msg = f"❌ Gemini API Error: Ensure your GEMINI_API_KEY is correct. Error: {e}"
        print(error_msg)
        return error_msg, retrieved_chunks
    except Exception as e:
        error_msg = f"❌ Generation failed: {e}"
        print(error_msg)
        return error_msg, retrieved_chunks


if __name__ == "__main__":
    # Example Query 
    RAG_QUERY = "What is the capital expenditure planned for the retail sector in 2025 and why is it growing?"
    
    # Run the full RAG pipeline
    final_answer, context_used = run_rag_service(RAG_QUERY)
    
    print("\n" + "="*80)
    print("FINAL RAG ANSWER")
    print("="*80)
    print(final_answer)
    print("="*80)
    
    print("\n--- CONTEXT CHUNKS USED ---")
    for chunk in context_used:
        print(f"Source: {chunk['company']} (Page {chunk['page_number']}) | Score: {chunk['score']:.4f}")
        print(f"Text: {chunk['text'][:100]}...\n")