"""
03_analyze.py
-------------
The core investigation. Uses Qdrant k-NN to answer:
  "Do a company's nearest semantic neighbors stay within its GICS sector,
   or do business descriptions reveal peer structure that sector labels miss?"

Produces result tables:
  data/neighbors.csv            : top-K neighbors per company
  data/cross_sector_by_sector.csv : how 'porous' each sector is
  data/sector_affinity.csv      : sector -> neighbor-sector affinity matrix (row %)
  data/mislabeled.csv           : companies whose semantic peers mostly sit elsewhere
"""
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

COLLECTION = "sp500_companies"
QDRANT_URL = "http://localhost:6333"
K = 10   # neighbors used for metrics (discretionary)


def main():
    meta = pd.read_csv("data/meta.csv")
    vecs = np.load("data/embeddings.npy")
    client = QdrantClient(url=QDRANT_URL)
    sectors = sorted(meta["sector"].unique())

    rows = []
    for i, r in meta.iterrows():
        pts = client.query_points(COLLECTION, query=vecs[i].tolist(), limit=K + 1).points
        nbrs = [p for p in pts if p.id != i][:K]
        for rank, p in enumerate(nbrs, 1):
            rows.append({
                "ticker": r["ticker"], "sector": r["sector"], "rank": rank,
                "nbr_ticker": p.payload["ticker"], "nbr_name": p.payload["name"],
                "nbr_sector": p.payload["sector"], "score": round(p.score, 4),
                "cross": p.payload["sector"] != r["sector"],
            })
    nb = pd.DataFrame(rows)
    nb.to_csv("data/neighbors.csv", index=False)

    overall = nb["cross"].mean()
    print(f"Overall cross-sector neighbor rate: {overall:.1%}")

    # how porous is each sector (share of its companies' neighbors in OTHER sectors)
    by_sector = (nb.groupby("sector")["cross"].mean()
                   .sort_values(ascending=False).rename("cross_sector_rate"))
    by_sector.to_csv("data/cross_sector_by_sector.csv")
    print("\nMost porous sectors (semantic peers leak out):")
    print((by_sector * 100).round(1).to_string())

    # sector -> neighbor-sector affinity, row-normalized to %
    # NOTE: row-% (normalize="index") divides out the SENDING sector but not the
    # RECEIVING one, so a large sector grabs a bigger share of every row purely
    # because it has more firms to be a neighbor. Kept as the descriptive "where
    # do neighbors land" table, but it is NOT a measure of affinity on its own.
    aff = (pd.crosstab(nb["sector"], nb["nbr_sector"], normalize="index") * 100).round(1)
    aff = aff.reindex(index=sectors, columns=sectors, fill_value=0)
    aff.to_csv("data/sector_affinity.csv")

    # ---- size-corrected affinity: lift over a chance baseline -----------------
    # Expected share of sector i's neighbors that land in sector j, if neighbors
    # were drawn at random from the other N-1 firms:
    #     expected_share[i->j] = (count_j - [i==j]) / (N - 1)
    # The (-[i==j]) drops the firm itself from its own sector's pool, so the
    # diagonal of this baseline is exactly the (n-1)/(N-1) chance term used in
    # make_fig4.py. lift = observed / expected:  1.0 = chance, >1 = attraction,
    # <1 = avoidance. This removes the receiving-sector size confound.
    counts = meta["sector"].value_counts()
    N = len(meta)
    expected = pd.DataFrame(
        {j: {i: 100 * (counts[j] - (1 if i == j else 0)) / (N - 1)
             for i in sectors} for j in sectors}
    ).reindex(index=sectors, columns=sectors)
    lift = (aff / expected).round(3)
    lift.to_csv("data/sector_affinity_lift.csv")

    # ---- significance: label-permutation null for each cell -------------------
    # Shuffle neighbor-sector labels (preserving each sector's neighbor count),
    # recompute the lift matrix, and ask how often the shuffled lift is at least
    # as extreme as the observed one. Two-sided empirical p-value per cell.
    rng = np.random.default_rng(42)
    n_perm = 2000
    obs_counts = pd.crosstab(nb["sector"], nb["nbr_sector"]).reindex(
        index=sectors, columns=sectors, fill_value=0)
    row_tot = obs_counts.sum(axis=1)
    exp_counts = expected.div(100).mul(row_tot, axis=0)   # expected COUNTS per cell
    obs_dev = (obs_counts - exp_counts).abs()
    nbr_labels = nb["nbr_sector"].to_numpy()
    sender = nb["sector"].to_numpy()
    ge = pd.DataFrame(0, index=sectors, columns=sectors)
    for _ in range(n_perm):
        shuffled = rng.permutation(nbr_labels)
        ct = pd.crosstab(pd.Series(sender), pd.Series(shuffled)).reindex(
            index=sectors, columns=sectors, fill_value=0)
        dev = (ct - exp_counts).abs()
        ge += (dev >= obs_dev).astype(int)
    pval = ((ge + 1) / (n_perm + 1)).round(4)
    pval.to_csv("data/sector_affinity_pvalue.csv")
    sig = (pval < 0.05).sum().sum()
    print(f"\nSize-corrected affinity (lift) written; {sig} of {pval.size} "
          f"cells differ from chance at p<0.05 ({n_perm} permutations).")

    # companies whose neighbors mostly sit in a DIFFERENT, single sector = case studies
    mis = []
    for tkr, g in nb.groupby("ticker"):
        own = g["sector"].iloc[0]
        top_nbr_sector = g["nbr_sector"].mode().iloc[0]
        cross_rate = g["cross"].mean()
        if cross_rate >= 0.6 and top_nbr_sector != own:
            mis.append({
                "ticker": tkr, "name": meta.loc[meta.ticker == tkr, "name"].iloc[0],
                "gics_sector": own, "semantic_sector": top_nbr_sector,
                "cross_rate": round(cross_rate, 2),
            })
    mis = pd.DataFrame(mis).sort_values("cross_rate", ascending=False)
    mis.to_csv("data/mislabeled.csv", index=False)
    print(f"\n{len(mis)} companies whose semantic peers mostly sit in another sector.")
    print("Top mismatches (great case studies for your write-up):")
    print(mis.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
