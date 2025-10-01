import os
import re
import math
from typing import List, Dict, Any, Generator

# Core RAG/Vector DB libraries
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    SparseVectorParams, 
    PointStruct, 
    SparseVector,
    # Updated: Using OptimizersConfigDiff for collection creation
    OptimizersConfigDiff 
)

# Embedding Model
from sentence_transformers import SentenceTransformer

# Document Parsing
from pypdf import PdfReader

# Sparse Vector (BM25-like) Dependencies
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict


# --- Configuration ---
# NOTE: ASSUMES NLTK data ('stopwords', 'punkt', 'punkt_tab') was downloaded 
# separately to avoid in-script I/O and SSL errors.
DATA_DIR = "data"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "financial_reports_2025"
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
STOP_WORDS = set(stopwords.words('english')) # Now safe to initialize
BATCH_SIZE = 500 # Manual batching size for upsert to prevent timeouts

PDF_FILES = {
    "RIL": "RIL-Integrated-Annual-Report-2024-25.pdf",
    "ETERNAL": "eternal-ar-25.pdf",
    "INFOSYS": "infosys-ar-25.pdf",
}

# --- Utility Functions ---

def clean_text(text: str) -> str:
    """Removes common PDF artifacts and extra whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    return text.strip()

def recursive_chunker(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple character-based recursive chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks

def parse_and_chunk_pdf(file_path: str, company: str) -> List[Dict[str, Any]]:
    """Parses a PDF, chunks text, and generates metadata."""
    print(f"📄 Processing {company} report...")
    reader = PdfReader(file_path)
    all_chunks = []
    
    for page_num, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        if not raw_text:
            continue
            
        cleaned_text = clean_text(raw_text)
        
        chunks = recursive_chunker(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for i, chunk in enumerate(chunks):
            metadata = {
                "company": company,
                "year": 2025,
                "source_file": os.path.basename(file_path),
                "page_number": page_num + 1,
                "chunk_id": f"{company}_{page_num+1}_{i}",
                "section_hint": "Financial Report Content"
            }
            all_chunks.append({"text": chunk, "metadata": metadata})
            
    return all_chunks

def batch_points(points_list: List[PointStruct], batch_size: int) -> Generator[List[PointStruct], None, None]:
    """Yields successive n-sized chunks from a list."""
    for i in range(0, len(points_list), batch_size):
        yield points_list[i : i + batch_size]

# --- Sparse Vector (BM25-like/TF-IDF) Functions ---

DOCUMENT_FREQUENCIES: Dict[str, int] = defaultdict(int)
TOTAL_DOCUMENTS = 0

def tokenize(text: str) -> List[str]:
    """Tokenizes text, removes stopwords, and converts to lowercase."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalnum() and t not in STOP_WORDS]

def calculate_doc_frequencies(all_data: List[Dict[str, Any]]):
    """Calculates the document frequency for each term in the entire corpus."""
    global TOTAL_DOCUMENTS
    TOTAL_DOCUMENTS = len(all_data)
    
    for item in all_data:
        text = item['text']
        tokens = set(tokenize(text))
        for token in tokens:
            DOCUMENT_FREQUENCIES[token] += 1
            
def get_sparse_vector(text: str) -> SparseVector:
    """
    Generates a SparseVector approximation (TF-IDF/BM25-like).
    """
    if TOTAL_DOCUMENTS == 0:
        raise ValueError("Document frequencies must be calculated first.") 
        
    tokens = tokenize(text)
    term_frequencies: Dict[str, int] = defaultdict(int)
    for token in tokens:
        term_frequencies[token] += 1

    indices = []
    values = []
    
    for token, tf in term_frequencies.items():
        if token in DOCUMENT_FREQUENCIES:
            # IDF = log(N / df_t)
            idf = math.log(TOTAL_DOCUMENTS / DOCUMENT_FREQUENCIES[token], 10)
            score = tf * idf
            
            if score > 0:
                # Use a simple hash for index mapping
                index = hash(token) % 2**30 
                indices.append(index)
                values.append(float(score))

    # Qdrant expects indices and values to be sorted
    sorted_pairs = sorted(zip(indices, values))
    sorted_indices = [idx for idx, val in sorted_pairs]
    sorted_values = [val for idx, val in sorted_pairs]
    
    return SparseVector(indices=sorted_indices, values=sorted_values)


# --- Main Ingestion Pipeline ---

def run_ingestion():
    """Executes the full data parsing, embedding, and Qdrant ingestion process."""
    
    # 1. Initialize Clients
    print("\nStarting Financial Research Agent Data Ingestion...")
    # FIX: Initialize Qdrant client with an extended timeout (10 minutes)
    qdrant_client = QdrantClient(
        host=QDRANT_HOST, 
        port=QDRANT_PORT,
        timeout=600  
    ) 
    
    # Initialize Dense Model
    print(f"Loading Dense Embedding Model: {DENSE_MODEL_NAME}...")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME)
    DENSE_DIM = dense_model.get_sentence_embedding_dimension()

    # 2. Parse and Chunk All Documents
    all_processed_chunks: List[Dict[str, Any]] = []
    for company, filename in PDF_FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"🛑 Error: File not found at {file_path}. Skipping.")
            continue
        
        chunks = parse_and_chunk_pdf(file_path, company)
        all_processed_chunks.extend(chunks)

    if not all_processed_chunks:
        print("🛑 No data processed. Exiting.")
        return

    # 3. Calculate Document Frequencies for Sparse Vectors
    print(f"\n📊 Calculating corpus statistics across {len(all_processed_chunks)} chunks...")
    calculate_doc_frequencies(all_processed_chunks)
    
    # 4. Generate Dense and Sparse Vectors
    print("🧠 Generating Dense and Sparse embeddings...")
    texts = [item['text'] for item in all_processed_chunks]
    
    # NOTE: Using a single encode call is highly optimized by Sentence Transformers
    dense_vectors = dense_model.encode(texts, convert_to_numpy=True).tolist()
    sparse_vectors = [get_sparse_vector(text) for text in texts]
    
    final_points_data = []
    for i in range(len(all_processed_chunks)):
        chunk = all_processed_chunks[i]
        final_points_data.append({
            "id": i,
            "text": chunk['text'],
            "metadata": chunk['metadata'],
            "dense_vector": dense_vectors[i],
            "sparse_vector": sparse_vectors[i]
        })

    # 5. Qdrant Collection Setup (Hybrid Search)
    print(f"\n⚙️ Configuring Qdrant Collection '{COLLECTION_NAME}' (Dim: {DENSE_DIM})...")
    
    qdrant_client.recreate_collection(collection_name=COLLECTION_NAME,
        vectors_config={
            "dense_vectors": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse_vectors": SparseVectorParams(),
        },
        # FIX: Using OptimizersConfigDiff and providing all required fields
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2,
            memmap_threshold=20000,
            indexing_threshold=10000,
            deleted_threshold=0.2,            
            vacuum_min_vector_number=1000,    
            flush_interval_sec=5,             
        )
    )

    # 6. Data Ingestion (Upsert Points)
    total_points = len(final_points_data)
    print(f"🚀 Ingesting {total_points} points in batches of {BATCH_SIZE} into Qdrant...")
    
    points_to_upsert = [
        PointStruct(
            id=item["id"],
            vector={
                "dense_vectors": item["dense_vector"],
                "sparse_vectors": item["sparse_vector"],
            },
            payload={
                **item["metadata"],
                "text": item["text"]
            }
        )
        for item in final_points_data
    ]
    
    # FIX: Iterate and upsert the points in batches to prevent timeouts
    total_batches = math.ceil(total_points / BATCH_SIZE)
    for i, batch in enumerate(batch_points(points_to_upsert, BATCH_SIZE)):
        print(f"   -> Sending batch {i+1} / {total_batches}...")
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=batch,
        )

    print(f"\n✅ Ingestion complete! Collection '{COLLECTION_NAME}' created with {total_points} points.")
    
if __name__ == "__main__":
    # Ensure NLTK modules are imported to trigger errors early if data is missing
    # We don't use the find() method here because it's prone to the LookupError
    try:
        if not os.path.isdir(DATA_DIR):
            print(f"Error: The directory '{DATA_DIR}' was not found. Please create it and place your PDF files inside.")
        else:
            run_ingestion()
    except LookupError as e:
        print("\n\n**********************************************************************")
        print("CRITICAL NLTK ERROR: NLTK resources are missing.")
        print("Please run these commands to download the data:")
        print(">>> import nltk")
        print(">>> nltk.download('stopwords')")
        print(">>> nltk.download('punkt')")
        # Added the resource that caused the most recent LookupError
        print(">>> nltk.download('punkt_tab')") 
        print("**********************************************************************\n")