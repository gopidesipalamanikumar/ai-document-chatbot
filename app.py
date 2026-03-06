import streamlit as st
from rag_pipeline import build_vector_store

st.title("RAG Question Answering System")

vector_store = build_vector_store()

query = st.text_input("Ask a question")

if query:

    docs = vector_store.similarity_search(query)

    if docs:
        answer = docs[0].page_content
        st.write("Answer:", answer)