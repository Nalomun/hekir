# Beyond GICS: Vector-Search Company Comparables

Do the standard GICS sector labels actually capture how companies relate to one
another, or do their business descriptions reveal a peer structure that the
labels miss?

This project embeds the business description of every S&P 500 company and uses
exact k-nearest-neighbor search over the vectors to ask a simple question: **do a
company's closest semantic peers stay inside its GICS sector, or do they leak
into others?** The vectors can also be loaded into a
[Qdrant](https://qdrant.tech/) vector database (optional; its neighbors are
verified identical to the numpy search).

This public repository is a snapshot of private work begun in June 2026.

## Key findings

N = 499 companies (dual-class listings merged), k = 10 neighbors, 10,000
firm-label permutations.

- **66.0% of nearest semantic neighbors fall in the firm's own GICS sector,
  against 11.0% expected by chance (6.0×; p < 1/10001).** The embeddings recover
  sector structure strongly.
- **Cohesion varies by sector once size is accounted for.** Self-retention as a
  multiple of each sector's own chance rate runs from Utilities (14.9×) and
  Energy (14.8×) down to Consumer Discretionary (3.6×) and Industrials (3.5×).
- **Six cross-sector cells are above chance after Benjamini-Hochberg
  correction** (of 110 off-diagonal cells): Energy → Utilities (3.7×), Consumer
  Discretionary → Consumer Staples (2.8×), Materials → Consumer Staples (2.6×),
  Consumer Discretionary → Real Estate (1.8×), Communication Services →
  Information Technology (1.5×) and Materials → Industrials (1.4×). Another 85
  cells are below chance, mostly because two-thirds of all neighbors stay
  in-sector.
- **Eight of 55 sector pairs show a significant directional asymmetry** (BH
  q < 0.05). For example, Materials firms have 58 Industrials neighbors, while
  Industrials firms have 23 Materials neighbors (lift 1.39× vs 0.55×,
  q = 0.0055). The receiving sector is larger in 5 of the 8 pairs and more
  cohesive in 5 of the 8, so there is no consistent "small sector into large,
  coherent sector" rule.
- **Embedding peers co-move more than same-sector peers.** Mean daily-return
  correlation with the 10 embedding peers is 0.335, against 0.275 for the
  firm's GICS sector (difference +0.060, 95% CI 0.050–0.070; higher for 72% of
  493 firms; 2025-07-01 to 2026-06-30). The gap is similar on market-residual
  returns (+0.070).
- **For mismatch candidates, the advantage disappears.** These are firms whose
  neighbors mostly sit in one other sector. Their embedding peers co-move no
  more than their GICS sector peers at any cross-rate threshold from 0.6 to
  0.9 (at 0.6: −0.007, 95% CI −0.027 to 0.014, n = 98). The candidate list
  marks where descriptions and labels disagree. It does not show that GICS
  misplaces these firms.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| Build dataset (Wikipedia S&P 500 list + business summaries) | `src/01_fetch_data.py` | `data/companies.csv` |
| Merge share classes; chunked embeddings | `src/02_embed.py` | `data/embeddings.npy`, `data/meta.csv`, `data/token_counts.csv` |
| *(optional)* Load into Qdrant and verify neighbors | `src/02b_load_qdrant.py` | — |
| Exact kNN, lift, permutation tests, mismatch candidates | `src/03_analyze.py` | `data/neighbors.csv`, `data/headline.json`, `data/sector_*.csv`, `data/mismatch_*.csv` |
| Before/after comparison of the corrections | `src/compare_variants.py` | `data/variant_comparison.csv` |
| Daily prices (local only, gitignored) | `src/05_fetch_prices.py` | `data/prices/close.csv.gz` |
| Co-movement test | `src/06_comovement.py` | `data/comovement_*.csv` |
| Figures 1–3 and Table 1 data | `src/04_visualize.py` | `figures/fig1–fig3`, `data/case_studies.csv` |
| Sector-cohesion chart | `src/make_fig4.py` | `figures/fig4` |

## Method notes

- **Embeddings:** `all-MiniLM-L6-v2` (384-dim sentence-transformer), cosine
  similarity. The model reads at most 256 tokens. 292 of 499 summaries (58.5%)
  are longer, measured with the model's own tokenizer. So each summary is packed
  by sentence into chunks that fit, the chunks are embedded, and the chunk
  vectors are averaged (weighted by token count) and re-normalized.
- **Share classes:** GOOGL/GOOG, FOXA/FOX and NWSA/NWS have identical
  descriptions, so each pair is merged into one company (first listed class
  kept).
- **Neighbors:** exact cosine kNN in numpy (`src/knn.py`), with ties broken by
  row index.
- **Lift:** for sector *i* pointing to sector *j*, the chance baseline is
  `expected[i→j] = k · n_i · (n_j − 𝟙[i=j]) / (N − 1)` neighbor slots, and
  `lift = observed / expected`. Lift = 1 is chance.
- **Significance:** the kNN graph is held fixed while sector labels are permuted
  across firms (10,000 permutations, seed 42). Each cell's observed count is
  compared with its own permutation distribution, centered on the permuted
  mean (two-sided; the smallest attainable p is 1/10001). Benjamini-Hochberg is
  applied across the 110 off-diagonal cells. Directional asymmetry,
  C[a→b] − C[b→a], is tested the same way across the 55 sector pairs.
- **Mismatch candidates:** cross-sector rate at or above a threshold, with a
  modal neighbor sector other than the firm's own. Counts at 0.6 / 0.7 / 0.8 /
  0.9 are in `data/mismatch_threshold_counts.csv`.
- **Co-movement:** simple daily returns from adjusted closes. Firms need at
  least 200 returns in the window (493 qualify). Each firm's mean correlation
  with its embedding peers is compared with its mean correlation with all
  other same-sector firms (the exact expectation for 10 random same-sector
  peers). Tests are a sign-flip permutation and a firm bootstrap (10,000 each).
  Peer sets overlap across firms, so the firm-level tests treat
  non-independent observations as independent and the intervals are
  optimistic.
- **Before/after:** `data/variant_comparison.csv` gives the headline numbers
  as published (502 listings, truncated text), after merging share classes,
  and after chunking.

## Run

```bash
pip install -r requirements.txt
./run.sh              # embed -> analyze -> before/after -> co-movement -> figures
./run.sh --fetch      # also rebuild data/companies.csv (the live S&P 500 list will differ)
./run.sh --qdrant     # also load into Qdrant and verify; needs requirements-qdrant.txt and
                      #   docker run -p 6333:6333 qdrant/qdrant
```

The repository ships with `data/companies.csv` (June 2026 snapshot) and all
derived outputs. Price data is downloaded locally on first run and is never
committed.

## License

[MIT](LICENSE)
