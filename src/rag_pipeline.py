from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_vector_store(documents):

    # ✅ Better splitter (handles structure better)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    texts = splitter.split_documents(documents)

    # ✅ Use OpenAI embeddings (NO transformers, NO torch issues)
    embeddings = OpenAIEmbeddings()

    # ✅ Build FAISS index
    vector_store = FAISS.from_documents(texts, embeddings)

    return vector_store