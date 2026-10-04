"""
03_analyze.py
-------------
The core investigation. Uses exact cosine k-NN (src/knn.py) to answer:
  "Do a company's nearest semantic neighbors stay within its GICS sector,
   or do business descriptions reveal peer structure that sector labels miss?"

Produces result tables:
  data/neighbors.csv               : top-K neighbors per company
  data/headline.json               : overall in-sector rate vs chance
  data/cross_sector_by_sector.csv  : how 'porous' each sector is
  data/sector_cohesion.csv         : self-retention vs per-sector chance (Table 2 / fig4)
  data/sector_affinity.csv         : sector -> neighbor-sector shares (row %, descriptive)
  data/sector_affinity_lift.csv    : observed / expected (firm-count baseline)
  data/sector_affinity_tests.csv   : per-cell counts, lift, raw and BH-adjusted p
  data/sector_asymmetry_tests.csv  : per-pair C[a->b] vs C[b->a] test, raw and BH p
  data/mismatch_candidates.csv     : firms whose semantic peers mostly sit elsewhere
  data/mismatch_threshold_counts.csv
"""
import json

import numpy as np
import pandas as pd

from knn import exact_knn
import gics_lib as g

K = 10            # neighbors used for metrics (discretionary)
N_PERM = 10_000   # label permutations for the significance tests
SEED = 42
CANDIDATE_THRESHOLD = 0.7   # cross-sector rate (chosen from 0.6/0.7/0.8/0.9; see mismatch_threshold_counts.csv)


def main():
    meta = pd.read_csv("data/meta.csv")
    vecs = np.load("data/embeddings.npy")
    sectors, codes = g.sector_codes(meta)

    idx, scores = exact_knn(vecs, K)
    nb = g.neighbor_table(meta, idx, scores)
    nb.to_csv("data/neighbors.csv", index=False)

    head = g.headline(codes, idx)

    by_sector = (nb.groupby("sector")["cross"].mean()
                   .sort_values(ascending=False).rename("cross_sector_rate"))
    by_sector.to_csv("data/cross_sector_by_sector.csv")
    cohesion = g.sector_cohesion(meta, idx)
    cohesion.to_csv("data/sector_cohesion.csv", index=False)

    # descriptive "where do neighbors land" table (row %). Divides out the
    # SENDING sector only, so it is NOT a measure of affinity on its own.
    aff = (pd.crosstab(nb["sector"], nb["nbr_sector"], normalize="index") * 100).round(1)
    aff.reindex(index=sectors, columns=sectors, fill_value=0).to_csv("data/sector_affinity.csv")

    # ---- significance: firm-label permutation null ---------------------------
    # The kNN graph is fixed; sector labels are permuted across the N firms and
    # the full crosstab is recomputed each time. Each cell is compared with its
    # own permutation distribution (centered on the permuted mean), two-sided.
    # Benjamini-Hochberg across the 110 off-diagonal cells.
    res = g.permutation_test(codes, idx, len(sectors), n_perm=N_PERM, seed=SEED)
    cells = g.cell_tests(sectors, codes, idx, res)
    cells.to_csv("data/sector_affinity_tests.csv", index=False)
    (cells.pivot(index="from_sector", columns="to_sector", values="lift")
          .reindex(index=sectors, columns=sectors).round(3)
          .to_csv("data/sector_affinity_lift.csv"))
    asym = g.asymmetry_tests(sectors, codes, idx, res)
    # size and size-adjusted cohesion of the sending ("donor") and receiving
    # ("attractor") side of each pair's stronger direction
    a_is_sender = asym["diff_count"] > 0
    asym["sender"] = np.where(a_is_sender, asym["sector_a"], asym["sector_b"])
    asym["receiver"] = np.where(a_is_sender, asym["sector_b"], asym["sector_a"])
    asym["count_stronger"] = np.where(a_is_sender, asym["count_a_to_b"], asym["count_b_to_a"])
    asym["count_weaker"] = np.where(a_is_sender, asym["count_b_to_a"], asym["count_a_to_b"])
    coh = cohesion.set_index("sector")
    for side in ("sender", "receiver"):
        asym[f"{side}_n"] = asym[side].map(coh["n"])
        asym[f"{side}_cohesion_lift"] = asym[side].map(coh["lift"]).round(2)
    asym.to_csv("data/sector_asymmetry_tests.csv", index=False)

    head.update({"k": K, "n_perm": N_PERM, "seed": SEED,
                 "p_in_sector": res["p_in_sector"],
                 "offdiag_cells": int((~cells["diagonal"]).sum()),
                 "offdiag_sig_bh05": int(cells["significant_bh05"].sum()),
                 "offdiag_sig_bh05_above": int((cells["significant_bh05"] & (cells["lift"] > 1)).sum()),
                 "offdiag_sig_bh05_below": int((cells["significant_bh05"] & (cells["lift"] < 1)).sum()),
                 "asym_pairs": len(asym), "asym_sig_bh05": int(asym["significant_bh05"].sum())})
    with open("data/headline.json", "w") as f:
        json.dump(head, f, indent=2)

    print(f"N = {head['N']} firms, k = {K}")
    print(f"In-sector neighbor rate: {head['in_sector_rate']:.1%} vs chance "
          f"{head['chance_rate']:.1%} ({head['in_sector_lift']:.1f}x; "
          f"{g.format_p(res['p_in_sector'], N_PERM)})")
    print("\nMost porous sectors (semantic peers leak out):")
    print((by_sector * 100).round(1).to_string())
    print(f"\nOff-diagonal cells significant after BH (q<0.05): "
          f"{head['offdiag_sig_bh05']} of {head['offdiag_cells']} "
          f"({head['offdiag_sig_bh05_above']} above chance, "
          f"{head['offdiag_sig_bh05_below']} below), {N_PERM} permutations")
    print(f"Directional asymmetries significant after BH: "
          f"{head['asym_sig_bh05']} of {head['asym_pairs']} sector pairs")
    sig = asym[asym["significant_bh05"]]
    print(sig[["sender", "receiver", "count_stronger", "count_weaker", "p_bh", "sender_n", "receiver_n",
               "sender_cohesion_lift", "receiver_cohesion_lift"]].round(4).to_string(index=False))
    print(f"  receiver larger than sender: {(sig['receiver_n'] > sig['sender_n']).sum()} of {len(sig)}; "
          f"receiver more cohesive (size-adjusted): "
          f"{(sig['receiver_cohesion_lift'] > sig['sender_cohesion_lift']).sum()} of {len(sig)}")

    # ---- firms whose neighbors mostly sit in a DIFFERENT, single sector -------
    mt = g.mismatch_table(nb).merge(meta[["ticker", "name"]], on="ticker")
    g.threshold_counts(mt).to_csv("data/mismatch_threshold_counts.csv", index=False)
    cand = mt[(mt["cross_rate"] >= CANDIDATE_THRESHOLD - 1e-9)
              & (mt["semantic_sector"] != mt["gics_sector"])]
    cand = (cand[["ticker", "name", "gics_sector", "semantic_sector", "modal_tie", "cross_rate"]]
            .sort_values(["cross_rate", "ticker"], ascending=[False, True]))
    cand.to_csv("data/mismatch_candidates.csv", index=False)
    print(f"\n{len(cand)} mismatch candidates at cross-sector rate >= {CANDIDATE_THRESHOLD}")
    print(g.threshold_counts(mt).to_string(index=False))
    print(cand.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
