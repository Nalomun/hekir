"""
06_comovement.py
----------------
Do embedding-defined peers co-move more than same-GICS-sector peers?

Specification (fixed before looking at results):
  window   : daily returns 2025-07-01 .. 2026-06-30 from adjusted closes
  returns  : simple daily returns. PRIMARY = raw returns.
             SECONDARY = market-residual returns (OLS residual on the
             equal-weighted mean return of all firms in the sample).
  coverage : a firm needs >= 200 daily returns in the window; pairwise
             correlations use pairwise-complete days (min 100).
  per firm : emb_i  = mean correlation with its 10 embedding peers (peers
                      without price data dropped; firm kept if >= 5 remain)
             gics_i = mean correlation with ALL other same-GICS-sector firms
                      that have price data. This is the exact expected value
                      of the mean over 10 random same-sector peers, so no
                      random draw is needed.
             d_i    = emb_i - gics_i
  test     : mean d over firms; two-sided sign-flip permutation p
             (10,000 draws, seed 42) and a firm-level bootstrap 95% CI
             (10,000 resamples, seed 42).
  subsets  : mismatch candidates at each cross-rate threshold (0.6-0.9), and
             firms by whether any embedding peer sits outside the GICS sector.

Input : data/prices/close.csv.gz (from 05_fetch_prices.py), data/meta.csv,
        data/neighbors.csv
Output: data/comovement_by_firm.csv, data/comovement_summary.csv
"""
import numpy as np
import pandas as pd

import gics_lib as g

START, END = "2025-07-01", "2026-06-30"
MIN_RETURNS = 200
MIN_PAIR_DAYS = 100
MIN_PEERS = 5
N_DRAWS = 10_000
SEED = 42


def load_returns(meta):
    px = pd.read_csv("data/prices/close.csv.gz", index_col=0, parse_dates=True)
    ret = px.pct_change(fill_method=None).loc[START:END]
    n_obs = ret.notna().sum()
    keep = n_obs[n_obs >= MIN_RETURNS].index
    dropped = sorted(set(meta["ticker"]) - set(keep))
    return ret[keep], dropped, len(ret)


def market_residuals(ret):
    mkt = ret.mean(axis=1)
    out = {}
    for c in ret.columns:
        y = ret[c]
        ok = y.notna() & mkt.notna()
        X = np.column_stack([np.ones(ok.sum()), mkt[ok]])
        beta = np.linalg.lstsq(X, y[ok], rcond=None)[0]
        r = pd.Series(np.nan, index=y.index)
        r[ok] = y[ok] - X @ beta
        out[c] = r
    return pd.DataFrame(out)


def per_firm(corr, meta, nb):
    sector = meta.set_index("ticker")["sector"]
    have = set(corr.columns)
    peers = nb.groupby("ticker")["nbr_ticker"].apply(list)
    rows = []
    for t in corr.columns:
        emb = [p for p in peers[t] if p in have]
        same = [p for p in corr.columns if p != t and sector[p] == sector[t]]
        if len(emb) < MIN_PEERS or not same:
            continue
        rows.append({"ticker": t, "sector": sector[t],
                     "n_emb_peers": len(emb), "n_sector_peers": len(same),
                     "emb_corr": corr.loc[t, emb].mean(),
                     "gics_corr": corr.loc[t, same].mean(),
                     "any_cross_peer": any(sector[p] != sector[t] for p in peers[t])})
    df = pd.DataFrame(rows)
    df["diff"] = df["emb_corr"] - df["gics_corr"]
    return df


def test_mean(d: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    obs = d.mean()
    flips = rng.choice([-1.0, 1.0], size=(N_DRAWS, len(d)))
    null = (flips * d).mean(axis=1)
    p = (1 + (np.abs(null) >= abs(obs) - 1e-12).sum()) / (N_DRAWS + 1)
    rng = np.random.default_rng(SEED)
    boot = d[rng.integers(0, len(d), size=(N_DRAWS, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_firms": len(d), "mean_diff": obs, "ci95_lo": lo, "ci95_hi": hi,
            "p_signflip": p, "p_display": g.format_p(p, N_DRAWS),
            "share_firms_diff_gt_0": float((d > 0).mean())}


def main():
    meta = pd.read_csv("data/meta.csv")
    nb = pd.read_csv("data/neighbors.csv")
    # candidates recomputed from the neighbor table so every threshold is
    # available, not just the one written to mismatch_candidates.csv
    mt = g.mismatch_table(nb)
    foreign = mt["semantic_sector"] != mt["gics_sector"]
    ret, dropped, n_days = load_returns(meta)
    print(f"Window {START}..{END}: {n_days} trading days; {ret.shape[1]} firms with "
          f">= {MIN_RETURNS} returns; excluded {len(dropped)}: {dropped}")

    results, by_firm = [], []
    for label, r in (("raw", ret), ("market_residual", market_residuals(ret))):
        corr = r.corr(min_periods=MIN_PAIR_DAYS)
        pf = per_firm(corr, meta, nb)
        pf["returns"] = label
        by_firm.append(pf)
        subsets = {"all firms": pf,
                   "firms with >=1 cross-sector embedding peer": pf[pf["any_cross_peer"]],
                   "firms with all embedding peers in-sector": pf[~pf["any_cross_peer"]]}
        for t in g.THRESHOLDS:
            tick = mt.loc[foreign & (mt["cross_rate"] >= t - 1e-9), "ticker"]
            subsets[f"mismatch candidates (cross-rate >= {t})"] = pf[pf["ticker"].isin(tick)]
        for name, sub in subsets.items():
            res = test_mean(sub["diff"].to_numpy())
            res.update({"returns": label, "subset": name,
                        "mean_emb_corr": sub["emb_corr"].mean(),
                        "mean_gics_corr": sub["gics_corr"].mean()})
            results.append(res)

    pd.concat(by_firm).to_csv("data/comovement_by_firm.csv", index=False)
    summ = pd.DataFrame(results)[["returns", "subset", "n_firms", "mean_emb_corr", "mean_gics_corr",
                                  "mean_diff", "ci95_lo", "ci95_hi", "p_signflip", "p_display",
                                  "share_firms_diff_gt_0"]]
    summ.to_csv("data/comovement_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
