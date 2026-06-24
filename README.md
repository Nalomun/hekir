# Beyond GICS — Vector-Search Company Comparables

Do the standard GICS sector labels actually capture how companies relate to one
another, or do their business descriptions reveal a peer structure that the
labels miss?

This project embeds the business description of every S&P 500 company, loads the
vectors into a [Qdrant](https://qdrant.tech/) vector database, and uses
k-nearest-neighbor search to ask a simple question: **do a company's closest
semantic peers stay inside its GICS sector, or do they leak into others?**

## Key findings

- Across all 502 firms, a large share of nearest semantic neighbors fall outside
  the company's own GICS sector — sector porosity varies widely, from tightly
  self-contained sectors (Utilities, Health Care) to highly porous ones
  (Materials, Consumer Discretionary).
- **Cross-sector affinity is measured as lift over a firm-count baseline**
  (observed ÷ expected share), not as a raw neighbor percentage. This is the
  important methodological point: a large sector captures a bigger share of every
  other sector's neighbors purely because it has more firms, so raw shares
  conflate genuine semantic attraction with sector size. Dividing out the
  receiving sector's size isolates real affinity. Each cell is significance-tested
  against a 2,000-iteration label-permutation null.
- Genuine, size-corrected cross-sector attractions emerge (e.g. Energy ↔
  Utilities, Materials → Consumer Staples), and a directional asymmetry survives
  the correction: some sectors reach toward a neighbor that does not reciprocate.

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| Build dataset (Wikipedia S&P 500 list + business summaries) | `src/01_fetch_data.py` | `data/companies.csv` |
| Embed descriptions and load into Qdrant | `src/02_embed_and_load.py` | `data/embeddings.npy`, `data/meta.csv` |
| k-NN analysis, affinity matrix, lift + permutation test | `src/03_analyze.py` | `data/neighbors.csv`, `data/sector_affinity*.csv`, `data/mislabeled.csv` |
| Figures (embedding map, affinity heatmap, case studies) | `src/04_visualize.py` | `figures/fig1–fig3` |
| Sector-cohesion lift chart | `src/make_fig4.py` | `figures/fig4` |

## Method notes

- **Embeddings:** `all-MiniLM-L6-v2` (384-dim sentence-transformer), cosine
  similarity.
- **Affinity / lift:** for sector *i* pointing to sector *j*, the chance baseline
  is `expected[i→j] = (count_j − 𝟙[i=j]) / (N − 1)` with `N = 502`; `lift =
  observed / expected`. Lift = 1 is chance, > 1 attraction, < 1 avoidance. The
  diagonal of this baseline is the same self-retention chance term used in the
  cohesion chart, generalized to every cell.
- **Significance:** a label-permutation null (2,000 iterations) yields a
  two-sided p-value per affinity cell.

## Run order

```bash
# 1. dependencies
pip install -r requirements.txt

# 2. start Qdrant (vector database)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 3. build the dataset   (full S&P 500 fetch takes a few minutes)
python src/01_fetch_data.py

# 4. embed + load into Qdrant
python src/02_embed_and_load.py

# 5. run the analysis
python src/03_analyze.py

# 6. build the figures
python src/04_visualize.py
python src/make_fig4.py
```

The repository ships with the precomputed `data/` and `figures/` outputs, so the
analysis and figure steps can be re-run without refetching or re-embedding.

## License

[MIT](LICENSE)
