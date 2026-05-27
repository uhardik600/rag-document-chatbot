import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings import embed_and_store


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def ingest_document(filepath: str, session_id: str) -> int:
    """
    Load a document, split into chunks, embed and store in FAISS.
    Returns the number of chunks indexed.
    """
    ext = filepath.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        loader = PyPDFLoader(filepath)
    elif ext == "txt":
        loader = TextLoader(filepath, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Document appears to be empty or unreadable.")

    texts = [chunk.page_content for chunk in chunks]
    embed_and_store(texts, session_id)

    return len(texts)
