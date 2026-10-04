"""
compare_variants.py
-------------------
Headline numbers before and after each correction, all computed with the same
code (gics_lib.py):

  A  as published : 502 listings (share-class twins included), truncated at 256 tokens
  B  twins merged : 499 firms, truncated at 256 tokens
  C  final        : 499 firms, chunked + mean-pooled embeddings

For variant A it also re-runs the original (legacy) significance test, which
shuffled neighbor-sector labels across the 5,020 neighbor slots and measured
deviation from firm-count expected counts, to confirm it reproduces the
published figures.

Output: data/variant_comparison.csv
"""
import importlib
import sys

import numpy as np
import pandas as pd

from knn import exact_knn
import gics_lib as g

K = 10
N_PERM = 10_000
SEED = 42


def legacy_sig_count(nb: pd.DataFrame, sectors, meta, n_perm=2000, seed=42) -> int:
    """The original 03_analyze.py test, verbatim in logic (slot shuffle)."""
    rng = np.random.default_rng(seed)
    counts = meta["sector"].value_counts()
    N = len(meta)
    expected = pd.DataFrame({j: {i: 100 * (counts[j] - (1 if i == j else 0)) / (N - 1)
                                 for i in sectors} for j in sectors}).reindex(index=sectors, columns=sectors)
    obs_counts = pd.crosstab(nb["sector"], nb["nbr_sector"]).reindex(index=sectors, columns=sectors, fill_value=0)
    exp_counts = expected.div(100).mul(obs_counts.sum(axis=1), axis=0)
    obs_dev = (obs_counts - exp_counts).abs()
    nbr_labels, sender = nb["nbr_sector"].to_numpy(), nb["sector"].to_numpy()
    ge = pd.DataFrame(0, index=sectors, columns=sectors)
    for _ in range(n_perm):
        ct = pd.crosstab(pd.Series(sender), pd.Series(rng.permutation(nbr_labels))).reindex(
            index=sectors, columns=sectors, fill_value=0)
        ge += ((ct - exp_counts).abs() >= obs_dev).astype(int)
    pval = (ge + 1) / (n_perm + 1)
    return int((pval < 0.05).sum().sum())


def run_variant(name, meta, vecs, legacy=False) -> dict:
    sectors, codes = g.sector_codes(meta)
    idx, scores = exact_knn(vecs, K)
    nb = g.neighbor_table(meta, idx, scores)
    res = g.permutation_test(codes, idx, len(sectors), n_perm=N_PERM, seed=SEED)
    cells = g.cell_tests(sectors, codes, idx, res).set_index(["from_sector", "to_sector"])
    asym = g.asymmetry_tests(sectors, codes, idx, res).set_index(["sector_a", "sector_b"])
    mt = g.mismatch_table(nb)
    tc = g.threshold_counts(mt).set_index("threshold")["n_candidates"]
    h = g.headline(codes, idx)
    mi, im = cells.loc[("Materials", "Industrials")], cells.loc[("Industrials", "Materials")]
    pair = asym.loc[("Industrials", "Materials")]
    out = {
        "variant": name, "N": h["N"],
        "in_sector_rate": round(h["in_sector_rate"], 4),
        "chance_rate": round(h["chance_rate"], 4),
        "in_sector_lift": round(h["in_sector_lift"], 2),
        "offdiag_sig_bh05_of_110": int(cells["significant_bh05"].sum()),
        "offdiag_sig_bh05_above_chance": int((cells["significant_bh05"] & (cells["lift"] > 1)).sum()),
        "all_cells_raw_p_lt_05_of_121": int((cells["p_raw"] < 0.05).sum()),
        "mat_to_ind_lift": round(mi["lift"], 3), "mat_to_ind_p_raw": mi["p_raw"], "mat_to_ind_p_bh": mi["p_bh"],
        "ind_to_mat_lift": round(im["lift"], 3), "ind_to_mat_p_raw": im["p_raw"], "ind_to_mat_p_bh": im["p_bh"],
        "mat_ind_asym_p_raw": pair["p_raw"], "mat_ind_asym_p_bh": pair["p_bh"],
        "asym_pairs_sig_bh05_of_55": int(asym["significant_bh05"].sum()),
    }
    for t, c in tc.items():
        out[f"candidates_at_{t}"] = int(c)
    if legacy:
        out["legacy_test_cells_p_lt_05_of_121"] = legacy_sig_count(nb, sectors, meta)
    return out


def main():
    embed = importlib.import_module("02_embed")
    from sentence_transformers import SentenceTransformer

    raw = pd.read_csv("data/companies.csv")
    model = SentenceTransformer(embed.MODEL)
    raw_vecs = model.encode(raw["summary"].tolist(), normalize_embeddings=True,
                            batch_size=32).astype("float32")

    meta = pd.read_csv("data/meta.csv")
    rows = [
        run_variant("A_published_502_truncated", raw, raw_vecs, legacy=True),
        run_variant("B_merged_499_truncated", meta, np.load("data/embeddings_truncated.npy")),
        run_variant("C_final_499_chunked", meta, np.load("data/embeddings.npy")),
    ]
    df = pd.DataFrame(rows).set_index("variant").T
    df.to_csv("data/variant_comparison.csv")
    pd.set_option("display.width", 200)
    print(df.to_string())


if __name__ == "__main__":
    sys.exit(main())
