"""
02b_load_qdrant.py  (optional)
------------------------------
Loads the cached embeddings into a Qdrant collection with ticker/name/sector
payload, then checks that Qdrant's k-NN results match the exact numpy kNN used
by the analysis (src/knn.py).

The analysis does not need Qdrant. Run this only to reproduce the vector-store
setup or to re-verify the neighbor lists.

PREREQ: Qdrant running, e.g.  docker run -p 6333:6333 qdrant/qdrant
"""
import sys
import numpy as np
import pandas as pd

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    sys.exit("Run: pip install qdrant-client")

from knn import exact_knn

COLLECTION = "sp500_companies"
QDRANT_URL = "http://localhost:6333"
K = 10
TIE_TOL = 1e-5      # scores closer than this are treated as an exact tie


def load(client, vecs, meta):
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=vecs.shape[1], distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=i, vector=vecs[i].tolist(),
            payload={"ticker": r["ticker"], "name": r["name"],
                     "sector": r["sector"], "sub_industry": r["sub_industry"]},
        )
        for i, r in meta.iterrows()
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Loaded {client.count(COLLECTION).count} points into Qdrant '{COLLECTION}'.")


def verify(client, vecs):
    idx, scores = exact_knn(vecs, K)
    n = len(vecs)
    ordered = set_equal = 0
    tie_only = []
    for i in range(n):
        pts = client.query_points(COLLECTION, query=vecs[i].tolist(), limit=K + 1).points
        q = [p for p in pts if p.id != i][:K]
        q_ids = [p.id for p in q]
        q_scores = np.array([p.score for p in q])
        if q_ids == idx[i].tolist():
            ordered += 1
        if set(q_ids) == set(idx[i].tolist()):
            set_equal += 1
        if q_ids != idx[i].tolist():
            # every disagreement must sit between neighbors with tied scores
            same_scores = np.allclose(q_scores, scores[i], atol=TIE_TOL)
            tie_only.append(same_scores)
    print(f"Qdrant vs exact numpy kNN over {n} queries x {K} neighbors:")
    print(f"  identical ordered lists : {ordered}/{n}")
    print(f"  identical neighbor sets : {set_equal}/{n}")
    if tie_only:
        print(f"  differing lists explained by exact score ties: {sum(tie_only)}/{len(tie_only)}")
    max_score_diff = 0.0
    for i in range(n):
        pts = client.query_points(COLLECTION, query=vecs[i].tolist(), limit=K + 1).points
        q_scores = np.array([p.score for p in pts if p.id != i][:K])
        max_score_diff = max(max_score_diff, float(np.abs(q_scores - scores[i]).max()))
    print(f"  max |score difference|  : {max_score_diff:.2e}")
    return ordered == n


def main():
    meta = pd.read_csv("data/meta.csv")
    vecs = np.load("data/embeddings.npy")
    client = QdrantClient(url=QDRANT_URL)
    load(client, vecs, meta)
    ok = verify(client, vecs)
    print("PASS: neighbors identical." if ok else "Neighbor lists differ (see above).")


if __name__ == "__main__":
    main()
