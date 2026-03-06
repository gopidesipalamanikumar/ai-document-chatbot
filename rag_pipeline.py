from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

def build_vector_store():

    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = FakeEmbeddings(size=384)

    vector_store = FAISS.from_texts(chunks, embeddings)

    return vector_store