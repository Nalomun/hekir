"""
gics_lib.py
-----------
Shared analysis functions: neighbor tables, chance baselines, the firm-label
permutation test, Benjamini-Hochberg, and mismatch candidates. Used by
03_analyze.py and compare_variants.py so every variant runs the same code.
"""
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


def neighbor_table(meta: pd.DataFrame, idx: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    n, k = idx.shape
    src = np.repeat(np.arange(n), k)
    dst = idx.ravel()
    return pd.DataFrame({
        "ticker": meta["ticker"].to_numpy()[src],
        "sector": meta["sector"].to_numpy()[src],
        "rank": np.tile(np.arange(1, k + 1), n),
        "nbr_ticker": meta["ticker"].to_numpy()[dst],
        "nbr_name": meta["name"].to_numpy()[dst],
        "nbr_sector": meta["sector"].to_numpy()[dst],
        "score": np.round(scores.ravel().astype(float), 4),
        "cross": meta["sector"].to_numpy()[src] != meta["sector"].to_numpy()[dst],
    })


def sector_codes(meta: pd.DataFrame):
    sectors = sorted(meta["sector"].unique())
    codes = meta["sector"].map({s: i for i, s in enumerate(sectors)}).to_numpy()
    return sectors, codes


def crosstab_counts(codes: np.ndarray, idx: np.ndarray, n_sectors: int) -> np.ndarray:
    """C[i, j] = number of (firm in sector i -> neighbor in sector j) edges."""
    k = idx.shape[1]
    src = np.repeat(codes, k)
    dst = codes[idx.ravel()]
    return np.bincount(src * n_sectors + dst, minlength=n_sectors ** 2).reshape(n_sectors, n_sectors)


def expected_counts(codes: np.ndarray, k: int, n_sectors: int) -> np.ndarray:
    """Firm-count baseline: E[i, j] = k * n_i * (n_j - [i == j]) / (N - 1)."""
    n = np.bincount(codes, minlength=n_sectors).astype(float)
    N = len(codes)
    E = k * np.outer(n, n) / (N - 1)
    E[np.diag_indices(n_sectors)] = k * n * (n - 1) / (N - 1)
    return E


def headline(codes: np.ndarray, idx: np.ndarray) -> dict:
    n = np.bincount(codes).astype(float)
    N = len(codes)
    obs = float((codes[idx] == codes[:, None]).mean())
    chance = float((n * (n - 1)).sum() / (N * (N - 1)))
    return {"N": N, "in_sector_rate": obs, "chance_rate": chance, "in_sector_lift": obs / chance}


def sector_cohesion(meta: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
    sectors, codes = sector_codes(meta)
    n = np.bincount(codes, minlength=len(sectors))
    N = len(codes)
    same = (codes[idx] == codes[:, None]).mean(axis=1)
    out = pd.DataFrame({"sector": sectors, "n": n})
    out["self_retention"] = [same[codes == s].mean() for s in range(len(sectors))]
    out["cross_rate"] = 1 - out["self_retention"]
    out["chance"] = (n - 1) / (N - 1)
    out["lift"] = out["self_retention"] / out["chance"]
    return out.sort_values("lift", ascending=False).reset_index(drop=True)


def two_sided_p(obs, perm):
    """Empirical two-sided p, centered on the permutation mean.

    p = (1 + #{|perm - mean| >= |obs - mean|}) / (n_perm + 1); floor is 1/(n_perm+1).
    """
    mu = perm.mean(axis=0)
    extreme = np.abs(perm - mu) >= np.abs(obs - mu) - 1e-9
    return (1 + extreme.sum(axis=0)) / (perm.shape[0] + 1), mu


def permutation_test(codes: np.ndarray, idx: np.ndarray, n_sectors: int,
                     n_perm: int = 10_000, seed: int = 42) -> dict:
    """Keep the kNN graph fixed; permute sector labels across firms.

    Each permutation reassigns the observed labels to firms at random (sector
    sizes preserved), then recomputes the full sector x sector crosstab.
    """
    rng = np.random.default_rng(seed)
    obs = crosstab_counts(codes, idx, n_sectors)
    perm = np.empty((n_perm, n_sectors, n_sectors), dtype=np.int32)
    for b in range(n_perm):
        perm[b] = crosstab_counts(rng.permutation(codes), idx, n_sectors)

    p_cell, mu_cell = two_sided_p(obs, perm)

    # directional asymmetry for each unordered pair a<b: C[a,b] - C[b,a]
    iu = np.triu_indices(n_sectors, 1)
    d_obs = (obs - obs.T)[iu]
    d_perm = (perm - perm.transpose(0, 2, 1))[:, iu[0], iu[1]]
    p_asym, mu_asym = two_sided_p(d_obs, d_perm)

    # overall in-sector count (trace) under the same null
    tr_obs = np.trace(obs)
    tr_perm = np.trace(perm, axis1=1, axis2=2)
    p_trace = (1 + (tr_perm >= tr_obs).sum()) / (n_perm + 1)     # one-sided: more in-sector than chance

    return {"obs": obs, "perm_mean": mu_cell, "p_cell": p_cell,
            "pairs": iu, "d_obs": d_obs, "d_perm_mean": mu_asym, "p_asym": p_asym,
            "p_in_sector": p_trace, "n_perm": n_perm, "seed": seed}


def bh(p: np.ndarray) -> np.ndarray:
    return false_discovery_control(np.asarray(p, dtype=float), method="bh")


def format_p(p: float, n_perm: int) -> str:
    floor = 1 / (n_perm + 1)
    return f"p < 1/{n_perm + 1}" if p <= floor + 1e-15 else f"{p:.4f}"


def cell_tests(sectors, codes, idx, res) -> pd.DataFrame:
    S = len(sectors)
    k = idx.shape[1]
    E = expected_counts(codes, k, S)
    rows = []
    for i in range(S):
        for j in range(S):
            rows.append({"from_sector": sectors[i], "to_sector": sectors[j],
                         "diagonal": i == j, "obs_count": int(res["obs"][i, j]),
                         "expected_count": E[i, j], "perm_mean_count": res["perm_mean"][i, j],
                         "lift": res["obs"][i, j] / E[i, j], "p_raw": res["p_cell"][i, j]})
    df = pd.DataFrame(rows)
    off = ~df["diagonal"]
    df["p_bh"] = np.nan
    df.loc[off, "p_bh"] = bh(df.loc[off, "p_raw"])
    df["significant_bh05"] = df["p_bh"] < 0.05
    df["direction"] = np.where(df["lift"] > 1, "above chance", "below chance")
    df["p_raw_display"] = [format_p(p, res["n_perm"]) for p in df["p_raw"]]
    return df


def asymmetry_tests(sectors, codes, idx, res) -> pd.DataFrame:
    S = len(sectors)
    k = idx.shape[1]
    E = expected_counts(codes, k, S)
    a, b = res["pairs"]
    obs = res["obs"]
    df = pd.DataFrame({
        "sector_a": [sectors[x] for x in a], "sector_b": [sectors[y] for y in b],
        "count_a_to_b": obs[a, b], "count_b_to_a": obs[b, a],
        "lift_a_to_b": obs[a, b] / E[a, b], "lift_b_to_a": obs[b, a] / E[b, a],
        "diff_count": res["d_obs"], "perm_mean_diff": res["d_perm_mean"],
        "p_raw": res["p_asym"],
    })
    df["p_bh"] = bh(df["p_raw"])
    df["significant_bh05"] = df["p_bh"] < 0.05
    df["stronger_direction"] = np.where(df["diff_count"] > 0,
                                        df["sector_a"] + " -> " + df["sector_b"],
                                        df["sector_b"] + " -> " + df["sector_a"])
    df["p_raw_display"] = [format_p(p, res["n_perm"]) for p in df["p_raw"]]
    return df


def mismatch_table(nb: pd.DataFrame) -> pd.DataFrame:
    """Per firm: cross-sector rate and modal neighbor sector.

    Modal sector ties are broken alphabetically (pandas .mode() order), as in
    the original analysis; `modal_tie` records when that happened.
    """
    rows = []
    for tkr, g in nb.groupby("ticker", sort=False):
        vc = g["nbr_sector"].value_counts()
        top = vc[vc == vc.max()].index
        rows.append({"ticker": tkr, "gics_sector": g["sector"].iloc[0],
                     "semantic_sector": sorted(top)[0], "modal_tie": len(top) > 1,
                     "cross_rate": round(g["cross"].mean(), 2)})
    return pd.DataFrame(rows)


def threshold_counts(mt: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    foreign = mt["semantic_sector"] != mt["gics_sector"]
    return pd.DataFrame({"threshold": thresholds,
                         "n_candidates": [int(((mt["cross_rate"] >= t - 1e-9) & foreign).sum())
                                          for t in thresholds]})
