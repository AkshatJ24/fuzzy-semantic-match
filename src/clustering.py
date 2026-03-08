import os
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("CLUSTER_MODEL_PATH", "./gmm_model.joblib")

# GMM chosen over KMeans/DBSCAN because predict_proba() gives a proper
# probability distribution per document — exactly what fuzzy clustering requires.
# A post about "gun legislation" gets nonzero mass on both politics and firearms.

_gmm: GaussianMixture | None = None


def load_model() -> GaussianMixture:
    global _gmm
    if _gmm is None:
        _gmm = joblib.load(MODEL_PATH)
    return _gmm


def fit(embeddings: np.ndarray, n_components: int) -> GaussianMixture:
    global _gmm
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",  # full is O(d^2) per component — too expensive at 384-dim
        max_iter=200,
        random_state=42,
        verbose=1,
    )
    gmm.fit(embeddings)
    _gmm = gmm
    joblib.dump(gmm, MODEL_PATH)
    return gmm


def bic_sweep(embeddings: np.ndarray, k_range: range) -> dict[int, float]:
    """Fit GMMs over a range of k and return BIC scores to justify n_components choice."""
    scores = {}
    for k in k_range:
        g = GaussianMixture(n_components=k, covariance_type="diag", max_iter=100, random_state=42)
        g.fit(embeddings)
        scores[k] = g.bic(embeddings)
        print(f"  k={k:3d}  BIC={scores[k]:.2f}")
    return scores


def soft_assign(embedding: np.ndarray) -> np.ndarray:
    """Returns probability distribution over all clusters for one embedding."""
    gmm = load_model()
    return gmm.predict_proba(embedding.reshape(1, -1))[0]


def dominant_cluster(embedding: np.ndarray) -> int:
    return int(np.argmax(soft_assign(embedding)))


def assignment_entropy(probs: np.ndarray) -> float:
    """Higher entropy = more ambiguous / boundary document."""
    return float(entropy(probs + 1e-12))
