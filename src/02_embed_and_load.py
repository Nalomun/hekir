"""
02_embed_and_load.py
--------------------
Embeds each business summary with a local sentence-transformer (no API key--
i was looking at that itemized list dang what the hell push anti-key idea
for later), then loads the vectors into Qdrant with sector/name payload.

PREREQ: Qdrant running in Docker. Default at localhost:6333.
Output: data/embeddings.npy + data/meta.csv (cached so steps 03/04 don't re-embed)
"""
import sys
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers")
from qdrant_client import QdrantClient, models

MODEL = "all-MiniLM-L6-v2"     # 384-dim, fast on CPU. Upgrade: BAAI/bge-small-en-v1.5
COLLECTION = "sp500_companies"
QDRANT_URL = "http://localhost:6333"


def main():
    df = pd.read_csv("data/companies.csv")
    print(f"Loaded {len(df)} companies. Embedding with {MODEL}...")
    model = SentenceTransformer(MODEL)
    vecs = model.encode(
        df["summary"].tolist(), show_progress_bar=True,
        normalize_embeddings=True, batch_size=32,
    ).astype("float32")
    dim = vecs.shape[1]
    print(f"  embeddings: {vecs.shape}")

    # cache for some of the later steps, claude knows how to optimize???
    np.save("data/embeddings.npy", vecs)
    df.to_csv("data/meta.csv", index=False)

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=i, vector=vecs[i].tolist(),
            payload={
                "ticker": r["ticker"], "name": r["name"],
                "sector": r["sector"], "sub_industry": r["sub_industry"],
            },
        )
        for i, r in df.iterrows()
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Loaded {client.count(COLLECTION).count} points into Qdrant '{COLLECTION}'.")
    print("Screenshot the dashboard at http://localhost:6333/dashboard for your report.")


if __name__ == "__main__":
    main()
