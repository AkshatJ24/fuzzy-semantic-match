"""
Run once: downloads 20 Newsgroups, cleans it, embeds it, and persists to ChromaDB.
"""

import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import fetch_20newsgroups
from src import embeddings as emb

# Use 'all' split to maximise corpus size for clustering quality.
# remove=('headers', 'footers', 'quotes') strips metadata that would let the
# model cheat — we want semantic content, not sender addresses or mailing list IDs.
newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

raw_texts = newsgroups.data
categories = [newsgroups.target_names[t] for t in newsgroups.target]


def clean(text: str) -> str:
    # Strip lines that are quoted replies — they duplicate content from other posts
    lines = [l for l in text.splitlines() if not l.strip().startswith(">")]
    text = " ".join(lines)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


cleaned = [clean(t) for t in raw_texts]

# Drop posts that are too short to carry semantic signal after cleaning.
# Threshold of 80 chars: roughly 2 short sentences. Below this, embeddings are noisy.
MIN_LEN = 80
filtered = [(i, text, cat) for i, (text, cat) in enumerate(zip(cleaned, categories)) if len(text) >= MIN_LEN]
print(f"Retained {len(filtered)}/{len(raw_texts)} documents after cleaning.")

ids = [f"doc_{i}" for i, _, _ in filtered]
texts = [text for _, text, _ in filtered]
metas = [{"category": cat} for _, _, cat in filtered]

print("Embedding — this takes a few minutes on first run...")
embeddings = emb.embed(texts)

print("Upserting into ChromaDB...")
BATCH = 500
for start in range(0, len(ids), BATCH):
    emb.upsert(ids[start:start+BATCH], texts[start:start+BATCH], embeddings[start:start+BATCH], metas[start:start+BATCH])
    print(f"  {min(start+BATCH, len(ids))}/{len(ids)}")

print(f"Done. Collection size: {emb.count()}")
