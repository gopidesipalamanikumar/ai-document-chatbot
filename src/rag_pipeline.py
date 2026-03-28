from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_vector_store(documents):

    # ✅ Better splitter (handles structure better than CharacterTextSplitter)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    texts = splitter.split_documents(documents)

    # ✅ Optimized embedding model (fast + accurate)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ✅ Build FAISS index
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store