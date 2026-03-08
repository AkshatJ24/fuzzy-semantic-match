import numpy as np
from src import embeddings as emb


def retrieve(query_embedding: np.ndarray, n: int = 5) -> str:
    """
    Query ChromaDB for top-n semantically similar documents.
    Returns a concatenated string of results (stored in cache as the 'result').
    No cluster filter here — ChromaDB's HNSW index is already fast enough at this scale.
    Cluster filtering is reserved for the cache layer where linear scan cost matters.
    """
    results = emb.query(query_embedding, n_results=n)
    docs = results["documents"][0]
    dists = results["distances"][0]

    output_parts = []
    for i, (doc, dist) in enumerate(zip(docs, dists)):
        similarity = round(1 - dist, 4)  # ChromaDB cosine distance → similarity
        output_parts.append(f"[{i+1}] (sim={similarity})\n{doc[:500]}")

    return "\n\n---\n\n".join(output_parts)
