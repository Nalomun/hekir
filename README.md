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
  firm's GICS sector (difference +0.060, 95% CI 0.050–0.069; higher for 72% of
  493 firms; 2025-07-01 to 2026-06-30). The gap is similar on market-residual
  returns (+0.070).
- **For mismatch candidates, the advantage disappears.** These are the 83
  firms with at least 70% of neighbors outside their own sector and a
  different modal sector. Their embedding peers co-move no more than their
  GICS sector peers (−0.014, 95% CI −0.036 to 0.010). No threshold from 0.6
  to 0.9 gives a significant difference. The candidate list marks where
  descriptions and labels disagree. It does not show that GICS misplaces
  these firms.

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
- **Mismatch candidates:** cross-sector rate of at least 0.7, with a modal
  neighbor sector other than the firm's own (83 firms). Counts at 0.6 / 0.7 /
  0.8 / 0.9 are 98 / 83 / 57 / 27 (`data/mismatch_threshold_counts.csv`).
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

## Related work

Hoberg and Phillips (2010, 2016) measure product similarity as the cosine
between word vectors drawn from each firm's full 10-K business description,
recomputed annually across all public firms to build time-varying competitor
networks. This study instead applies a pretrained sentence-embedding model to
the short yfinance summaries of S&P 500 firms at a single point in time, and
uses the resulting neighbors to test GICS sectors and return co-movement rather
than to build a replacement classification.

## References

- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society: Series B (Methodological)*, 57(1), 289–300.
  https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Bhojraj, S., Lee, C. M. C., & Oler, D. K. (2003). What's my line? A comparison
  of industry classification schemes for capital market research. *Journal of
  Accounting Research*, 41(5), 745–774.
  https://doi.org/10.1046/j.1475-679X.2003.00122.x
- Hoberg, G., & Phillips, G. (2010). Product market synergies and competition in
  mergers and acquisitions: A text-based analysis. *Review of Financial
  Studies*, 23(10), 3773–3811. https://doi.org/10.1093/rfs/hhq053
- Hoberg, G., & Phillips, G. (2016). Text-based network industries and
  endogenous product differentiation. *Journal of Political Economy*, 124(5),
  1423–1465. https://doi.org/10.1086/688176
- Hoberg, G., & Phillips, G. Hoberg-Phillips Data Library.
  https://hobergphillips.tuck.dartmouth.edu/ (accessed 4 October 2026)
- Lee, C. M. C., Ma, P., & Wang, C. C. Y. (2015). Search-based peer firms:
  Aggregating investor perceptions through internet co-searches. *Journal of
  Financial Economics*, 116(2), 410–431.
  https://doi.org/10.1016/j.jfineco.2015.02.003
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
  Siamese BERT-networks. In *Proceedings of the 2019 Conference on Empirical
  Methods in Natural Language Processing and the 9th International Joint
  Conference on Natural Language Processing (EMNLP-IJCNLP)* (pp. 3982–3992).
  Association for Computational Linguistics. https://doi.org/10.18653/v1/D19-1410
- sentence-transformers/all-MiniLM-L6-v2 [model card]. Hugging Face.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (accessed 4
  October 2026)

## License

[MIT](LICENSE)
