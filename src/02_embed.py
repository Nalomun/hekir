"""
02_embed.py
-----------
Embeds each business summary with a local sentence-transformer (no API key).

Output: data/embeddings.npy + data/meta.csv (cached so steps 03/04 don't re-embed)

Loading the vectors into Qdrant is optional and lives in 02b_load_qdrant.py;
the analysis itself uses exact numpy kNN (src/knn.py).
"""
import sys
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers")

MODEL = "all-MiniLM-L6-v2"     # 384-dim, fast on CPU. Upgrade: BAAI/bge-small-en-v1.5


def main():
    df = pd.read_csv("data/companies.csv")
    print(f"Loaded {len(df)} companies. Embedding with {MODEL}...")
    model = SentenceTransformer(MODEL)
    vecs = model.encode(
        df["summary"].tolist(), show_progress_bar=True,
        normalize_embeddings=True, batch_size=32,
    ).astype("float32")
    print(f"  embeddings: {vecs.shape}")

    np.save("data/embeddings.npy", vecs)
    df.to_csv("data/meta.csv", index=False)


if __name__ == "__main__":
    main()
