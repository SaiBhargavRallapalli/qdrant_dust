# pip install streamlit
import streamlit as st
from rag_service import run_rag_pipeline # Assuming you structured your code this way

st.title("Financial Document Hybrid RAG System")

query = st.text_input("Ask a question about the financial reports:")

if query:
    with st.spinner("Searching and synthesizing answer..."):
        # This calls your combined RAG service function
        final_answer, retrieved_context = run_rag_pipeline(query) 
        
    st.subheader("Synthesized Answer")
    st.write(final_answer)

    # Optional: show the context that was used
    st.subheader("Retrieved Context Chunks (Hybrid RRF)")
    # ... display the retrieved_context nicely ...