import streamlit as st
import logging
import base64
import time
import os

from src.pdf_loader import load_pdf
from src.rag_pipeline import build_vector_store

# ✅ NEW IMPORTS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------- API KEY ----------------
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="AI Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider(
    "Number of retrieved chunks",
    min_value=1,
    max_value=5,
    value=3
)

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    st.rerun()

# ---------------- MULTI PDF ----------------
uploaded_files = st.file_uploader(
    "📄 Upload PDF Documents",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- UI Styling ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        120deg,
        #f6f9fc,
        #eef2f7,
        #e3ecf5
    );
}
.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div style="padding:25px;border-radius:18px;background:rgba(255,255,255,0.9);
box-shadow:0 6px 20px rgba(0,0,0,0.08);margin-bottom:25px;text-align:center;">
<h1>🤖 AI Document Chatbot</h1>
<p style="font-size:18px;color:#555;">
Multi-PDF Chatbot using RAG + LLM + Memory
</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Session ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# ---------------- Cache ----------------
@st.cache_resource
def create_vector_store(docs):
    return build_vector_store(docs)

# ---------------- PDF DOWNLOAD ----------------
def generate_pdf(chat_history):
    doc = SimpleDocTemplate("chat_history.pdf")
    styles = getSampleStyleSheet()
    content = []

    for msg in chat_history:
        text = f"{msg['role'].upper()}: {msg['content']}"
        content.append(Paragraph(text, styles["Normal"]))

    doc.build(content)

# ---------------- MAIN ----------------
if uploaded_files:

    with st.spinner("Processing documents..."):

        all_docs = []

        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())

            docs = load_pdf(file.name)
            all_docs.extend(docs)

        vector_store = create_vector_store(all_docs)

        st.sidebar.header("📄 Document Info")
        st.sidebar.write(f"Total Chunks: {len(all_docs)}")

    col1, col2 = st.columns(2)

    # -------- PDF Viewer --------
    with col1:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📄 Document Preview")

        for file in uploaded_files:
            with open(file.name, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")

            st.markdown(f"### {file.name}")
            pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}"
            width="100%" height="400"></iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- Chat --------
    with col2:

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🤖 Chat with Documents")

        for message in st.session_state.messages:
            avatar = "🧑" if message["role"] == "user" else "🤖"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        query = st.chat_input("Ask a question...")

        if query:
            st.session_state.messages.append(
                {"role": "user", "content": query}
            )

            with st.chat_message("user", avatar="🧑"):
                st.write(query)

            start_time = time.time()

            # ✅ LLM + RAG
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": top_k}
                ),
                memory=st.session_state.memory
            )

            result = qa_chain.run(query)

            response_time = time.time() - start_time

            response_text = f"""
### 🧠 Answer
{result}

### ⚡ Features Used
- Multi-document RAG
- LLM reasoning
- Conversational memory
"""

            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )

            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response_text)
                st.caption(f"⏱ {response_time:.2f} sec")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DOWNLOAD ----------------
if st.session_state.messages:
    if st.button("📥 Download Chat as PDF"):
        generate_pdf(st.session_state.messages)
        with open("chat_history.pdf", "rb") as f:
            st.download_button(
                "⬇ Download",
                f,
                file_name="chat_history.pdf"
            )

# ---------------- Sidebar About ----------------
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📘 About

AI Chatbot using:
- LangChain
- FAISS
- OpenAI LLM
- RAG Architecture
""")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:gray;font-size:14px;">
Built with ❤️ using Streamlit • RAG • LangChain • LLM
</div>
""", unsafe_allow_html=True)