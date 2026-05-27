from embeddings import load_store, embed_query

TOP_K = 4  # Number of most relevant chunks to retrieve


def search_similar_chunks(query: str, session_id: str) -> list[str]:
    """
    Search FAISS index for the top-K most relevant chunks for the given query.
    Returns a list of text chunks, or empty list if session not found.
    """
    index, texts = load_store(session_id)

    if index is None or texts is None:
        return []

    query_vector = embed_query(query)
    k = min(TOP_K, len(texts))
    distances, indices = index.search(query_vector, k)

    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(texts):
            results.append(texts[idx])

    return results
