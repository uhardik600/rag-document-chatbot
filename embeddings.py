import os
import faiss
import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_DIR = "vector_stores"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


def get_store_paths(session_id: str):
    index_path = os.path.join(VECTOR_STORE_DIR, f"{session_id}.index")
    texts_path = os.path.join(VECTOR_STORE_DIR, f"{session_id}.pkl")
    return index_path, texts_path


def embed_and_store(texts: list[str], session_id: str):
    """
    Generate embeddings for text chunks and store in a FAISS index.
    """
    vectors = embeddings_model.embed_documents(texts)
    vectors_np = np.array(vectors, dtype="float32")

    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    index_path, texts_path = get_store_paths(session_id)
    faiss.write_index(index, index_path)

    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)


def load_store(session_id: str):
    """
    Load FAISS index and associated texts for a session.
    Returns (index, texts) or (None, None) if not found.
    """
    index_path, texts_path = get_store_paths(session_id)

    if not os.path.exists(index_path) or not os.path.exists(texts_path):
        return None, None

    index = faiss.read_index(index_path)
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    return index, texts


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for similarity search.
    """
    vector = embeddings_model.embed_query(query)
    return np.array([vector], dtype="float32")
