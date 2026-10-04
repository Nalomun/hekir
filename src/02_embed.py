"""
02_embed.py
-----------
Embeds each business summary with a local sentence-transformer (no API key).

all-MiniLM-L6-v2 reads at most 256 tokens (including [CLS]/[SEP]) and silently
truncates the rest. Most summaries are longer than that, so each summary is
split into sentence-packed chunks that fit the window, each chunk is embedded,
and the chunk vectors are averaged (weighted by token count) and re-normalized.

Output: data/embeddings.npy           chunked + mean-pooled (used by the analysis)
        data/embeddings_truncated.npy  single-pass, truncated at 256 (for comparison)
        data/token_counts.csv          per-firm token count and number of chunks
        data/meta.csv                  (cached so steps 03/04 don't re-embed)

Loading the vectors into Qdrant is optional and lives in 02b_load_qdrant.py;
the analysis itself uses exact numpy kNN (src/knn.py).
"""
import re
import sys
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Run: pip install sentence-transformers")

MODEL = "all-MiniLM-L6-v2"     # 384-dim, fast on CPU. Upgrade: BAAI/bge-small-en-v1.5


def merge_share_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per company.

    Dual-class listings (GOOGL/GOOG, FOXA/FOX, NWSA/NWS) carry identical
    business summaries, so each twin would be the other's cosine-1.0 nearest
    neighbor and the same company would be counted twice. Keep the first
    listed class (Wikipedia order: GOOGL, FOXA, NWSA).
    """
    dup = df.duplicated("summary", keep="first")
    for t in df.loc[dup, "ticker"]:
        kept = df.loc[(df["summary"] == df.loc[df.ticker == t, "summary"].iloc[0]) & ~dup, "ticker"].iloc[0]
        print(f"  merged share class {t} into {kept}")
    return df.loc[~dup].reset_index(drop=True)


def token_len(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def chunk_text(tok, text: str, budget: int) -> list[str]:
    """Greedily pack whole sentences into chunks of at most `budget` tokens.

    A single sentence longer than the budget is split on word boundaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces = []
    for s in sentences:
        if token_len(tok, s) <= budget:
            pieces.append(s)
            continue
        cur = []
        for w in s.split():
            if cur and token_len(tok, " ".join(cur + [w])) > budget:
                pieces.append(" ".join(cur))
                cur = []
            cur.append(w)
        if cur:
            pieces.append(" ".join(cur))
    chunks, cur = [], []
    for p in pieces:
        if cur and token_len(tok, " ".join(cur + [p])) > budget:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(p)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def embed_chunked(model, texts: list[str]):
    tok = model.tokenizer
    budget = model.max_seq_length - 2          # room for [CLS] and [SEP]
    chunks, owner, weight = [], [], []
    for i, t in enumerate(texts):
        for c in chunk_text(tok, t, budget):
            n = token_len(tok, c)
            assert n <= budget, (i, n)
            chunks.append(c); owner.append(i); weight.append(n)
    cvecs = model.encode(chunks, show_progress_bar=True,
                         normalize_embeddings=True, batch_size=32)
    owner, weight = np.array(owner), np.array(weight, dtype=float)
    out = np.zeros((len(texts), cvecs.shape[1]))
    np.add.at(out, owner, cvecs * weight[:, None])
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    n_chunks = np.bincount(owner, minlength=len(texts))
    return out.astype("float32"), n_chunks


def main():
    df = pd.read_csv("data/companies.csv")
    df = merge_share_classes(df)
    print(f"Loaded {len(df)} companies. Embedding with {MODEL}...")
    model = SentenceTransformer(MODEL)
    texts = df["summary"].tolist()

    n_tok = np.array([len(model.tokenizer(t)["input_ids"]) for t in texts])   # incl. special tokens
    over = n_tok > model.max_seq_length
    print(f"  token counts (model tokenizer, incl. [CLS]/[SEP]): "
          f"median {int(np.median(n_tok))}, max {n_tok.max()}")
    print(f"  {over.sum()}/{len(texts)} summaries ({over.mean():.1%}) exceed the "
          f"{model.max_seq_length}-token window")

    truncated = model.encode(texts, show_progress_bar=True,
                             normalize_embeddings=True, batch_size=32).astype("float32")
    vecs, n_chunks = embed_chunked(model, texts)
    print(f"  embeddings: {vecs.shape}; chunks per firm: max {n_chunks.max()}")

    np.save("data/embeddings.npy", vecs)
    np.save("data/embeddings_truncated.npy", truncated)
    pd.DataFrame({"ticker": df["ticker"], "n_tokens": n_tok, "over_256": over,
                  "n_chunks": n_chunks}).to_csv("data/token_counts.csv", index=False)
    df.to_csv("data/meta.csv", index=False)


if __name__ == "__main__":
    main()
