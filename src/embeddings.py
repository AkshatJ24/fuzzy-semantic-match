import os
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "newsgroups"

# all-MiniLM-L6-v2: 384-dim, ~60ms/batch on CPU, strong semantic quality for
# short-to-medium text. Good fit here since posts are paragraph-length.
_model = SentenceTransformer(os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def embed(texts: list[str]) -> np.ndarray:
    return _model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)


def upsert(ids: list[str], texts: list[str], embeddings: np.ndarray, metadatas: list[dict]):
    _collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )


def query(embedding: np.ndarray, n_results: int = 5, where: dict | None = None) -> dict:
    kwargs = dict(query_embeddings=[embedding.tolist()], n_results=n_results, include=["documents", "distances", "metadatas"])
    if where:
        kwargs["where"] = where
    return _collection.query(**kwargs)


def count() -> int:
    return _collection.count()


def get_all_embeddings() -> tuple[list[str], np.ndarray, list[dict]]:
    """Fetch entire collection for clustering. Only called once during training."""
    result = _collection.get(include=["embeddings", "metadatas"])
    return result["ids"], np.array(result["embeddings"]), result["metadatas"]
