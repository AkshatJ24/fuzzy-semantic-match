from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src import embeddings as emb
from src import clustering
from src.cache import SemanticCache, CacheEntry
from src.search import retrieve


cache = SemanticCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load the GMM so first request isn't slow
    clustering.load_model()
    yield


app = FastAPI(title="Semantic Search API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    embedding = emb.embed([req.query])[0]
    probs = clustering.soft_assign(embedding)
    cluster_id = int(probs.argmax())

    cached_entry, sim_score = cache.lookup(embedding, cluster_id)

    if cached_entry is not None:
        return {
            "query": req.query,
            "cache_hit": True,
            "matched_query": cached_entry.query,
            "similarity_score": round(sim_score, 4),
            "result": cached_entry.result,
            "dominant_cluster": cluster_id,
        }

    result = retrieve(embedding)
    cache.insert(CacheEntry(
        query=req.query,
        embedding=embedding,
        result=result,
        dominant_cluster=cluster_id,
        soft_probs=probs,
    ))

    return {
        "query": req.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": round(sim_score, 4),
        "result": result,
        "dominant_cluster": cluster_id,
    }


@app.get("/cache/stats")
def cache_stats():
    s = cache.stats
    return {
        "total_entries": cache.total_entries,
        "hit_count": s.hit_count,
        "miss_count": s.miss_count,
        "hit_rate": s.hit_rate,
    }


@app.delete("/cache")
def flush_cache():
    cache.flush()
    return {"message": "Cache flushed."}
