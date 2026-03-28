import logging
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):

    try:
        logging.info(f"Loading PDF: {file_path}")

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        logging.info(f"Loaded {len(documents)} pages")

        return documents

    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        raise e