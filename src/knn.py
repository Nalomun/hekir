"""
knn.py
------
Exact cosine k-nearest-neighbor search in numpy (no vector database needed).

The embeddings are L2-normalized, so cosine similarity is the dot product.
At N ~ 500 a full N x N similarity matrix is trivial, and the result is exact.
Ties are broken deterministically by row index (lower index first).
"""
import numpy as np


def exact_knn(vecs: np.ndarray, k: int):
    """Return (idx, scores), each shape (N, k): the k nearest other rows by cosine.

    The query row itself is excluded. Ordering is by descending similarity,
    ties broken by ascending row index.
    """
    v = np.asarray(vecs, dtype=np.float32)
    sims = v @ v.T
    np.fill_diagonal(sims, -np.inf)
    n = len(v)
    cols = np.arange(n)
    idx = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        # lexsort: last key is primary -> sort by -similarity, then by index
        order = np.lexsort((cols, -sims[i]))
        idx[i] = order[:k]
    scores = np.take_along_axis(sims, idx, axis=1)
    return idx, scores
