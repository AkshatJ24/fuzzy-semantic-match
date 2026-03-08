import os
import numpy as np
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# Threshold exploration notes (run your own sweep with test queries):
#   0.70 → too permissive: "machine learning" hits "deep learning history" — wrong results
#   0.80 → good recall, occasional semantic drift on short queries
#   0.85 → (default) balanced: catches paraphrases, rejects topic shifts
#   0.92 → too strict: near-identical phrasings still miss; cache rarely fires
# The right value is corpus- and use-case-dependent. It is NOT a hyperparameter
# you can tune on accuracy alone — it encodes your tolerance for semantic approximation.

DEFAULT_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", 0.85))


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    result: str
    dominant_cluster: int
    soft_probs: np.ndarray


@dataclass
class CacheStats:
    hit_count: int = 0
    miss_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return round(self.hit_count / total, 4) if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_entries": self.hit_count + self.miss_count,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_rate,
        }


class SemanticCache:
    """
    Cluster-bucketed semantic cache backed by cosine similarity.

    Lookup is O(n/k) rather than O(n) by restricting the scan to the
    dominant cluster bucket of the incoming query. As the cache grows,
    this bucketing becomes increasingly important.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._store: dict[int, list[CacheEntry]] = {}
        self._stats = CacheStats()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        # Embeddings are already L2-normalised at encode time, so dot product == cosine sim.
        return float(np.dot(a, b))

    def lookup(self, embedding: np.ndarray, cluster_id: int) -> CacheEntry | None:
        bucket = self._store.get(cluster_id, [])
        best_sim, best_entry = -1.0, None

        for entry in bucket:
            sim = self._cosine_sim(embedding, entry.embedding)
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        if best_sim >= self.threshold:
            self._stats.hit_count += 1
            return best_entry, best_sim

        self._stats.miss_count += 1
        return None, best_sim

    def insert(self, entry: CacheEntry):
        self._store.setdefault(entry.dominant_cluster, []).append(entry)

    def flush(self):
        self._store.clear()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    @property
    def total_entries(self) -> int:
        return sum(len(v) for v in self._store.values())
