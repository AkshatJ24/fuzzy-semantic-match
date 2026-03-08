"""
Fits GMM on corpus embeddings.
Outputs: BIC curve, per-cluster sample docs, boundary cases (high-entropy assignments).
Run after prepare_data.py.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import entropy as scipy_entropy
from src import embeddings as emb
from src import clustering

print("Loading embeddings from ChromaDB...")
ids, embeddings, metadatas = emb.get_all_embeddings()
print(f"Loaded {len(ids)} documents.")

# BIC sweep to justify k. Lower BIC = better model.
# We expect an elbow somewhere between 15–25 given 20 labelled categories,
# but semantic overlap means true structure may differ from label count.
print("\nBIC sweep over k=10..28 (step 2):")
bic_scores = clustering.bic_sweep(embeddings, k_range=range(10, 29, 2))

best_k = min(bic_scores, key=bic_scores.get)
print(f"\nBest k by BIC: {best_k}")

print(f"\nFitting final GMM with k={best_k}...")
gmm = clustering.fit(embeddings, n_components=best_k)

# Soft assignment matrix: shape (n_docs, k)
probs = gmm.predict_proba(embeddings)
dominant = np.argmax(probs, axis=1)

print("\n--- Per-cluster sample documents (5 nearest to centroid) ---")
for cluster_id in range(best_k):
    member_mask = dominant == cluster_id
    member_indices = np.where(member_mask)[0]
    if len(member_indices) == 0:
        continue

    centroid = gmm.means_[cluster_id]
    member_embeddings = embeddings[member_indices]
    sims = member_embeddings @ centroid / (np.linalg.norm(member_embeddings, axis=1) * np.linalg.norm(centroid) + 1e-9)
    top5 = member_indices[np.argsort(sims)[-5:][::-1]]

    cats = [metadatas[i]["category"] for i in top5]
    print(f"\nCluster {cluster_id:2d} ({len(member_indices)} docs) — top categories: {set(cats)}")
    for i in top5:
        snippet = emb._collection.get(ids=[ids[i]], include=["documents"])["documents"][0][:120]
        print(f"  [{metadatas[i]['category']}] {snippet}")

# Boundary cases: documents with highest assignment entropy
print("\n--- Top 20 boundary documents (most ambiguous cluster membership) ---")
entropies = np.array([scipy_entropy(p + 1e-12) for p in probs])
boundary_indices = np.argsort(entropies)[-20:][::-1]

for i in boundary_indices:
    top2 = np.argsort(probs[i])[-2:][::-1]
    snippet = emb._collection.get(ids=[ids[i]], include=["documents"])["documents"][0][:120]
    print(f"  entropy={entropies[i]:.3f} clusters={top2} probs={probs[i][top2].round(3)} [{metadatas[i]['category']}]")
    print(f"    {snippet}\n")

print("\nClustering complete. Model saved to", os.getenv("CLUSTER_MODEL_PATH", "./gmm_model.joblib"))
