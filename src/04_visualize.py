"""
04_visualize.py
---------------
Builds the figures for the report (all saved to figures/):
  fig1_umap.png            : embedding space colored by GICS sector
  fig2_affinity_heatmap.png: sector -> neighbor-sector affinity
  fig3_case_studies.png    : the case-study firms + their nearest semantic neighbors
and data/case_studies.csv (Table 1: top-5 neighbor cosine range per case study).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

meta = pd.read_csv("data/meta.csv")
vecs = np.load("data/embeddings.npy")

# ---- Fig 1: embedding map colored by sector ----
try:
    import umap
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=42).fit_transform(vecs)
    method = "UMAP"
except Exception as e:
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2).fit_transform(vecs)
    method = "PCA"
    print(f"UMAP unavailable ({e}); used PCA.")

fig, ax = plt.subplots(figsize=(9, 6.5))
for s, g in pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "sector": meta["sector"]}).groupby("sector"):
    ax.scatter(g["x"], g["y"], s=14, alpha=0.75, label=s)
ax.legend(fontsize=7, markerscale=1.4, ncol=2, loc="best")
ax.set_title(f"S&P 500 business-description embedding space ({method}), colored by GICS sector")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig("figures/fig1_umap.png", dpi=150); plt.close()

# ---- Fig 2: size-corrected sector affinity heatmap (log2 lift over chance) ----
# Raw row-% conflates affinity with receiving-sector size, so we plot lift =
# observed / expected on a log2 scale: 0 = chance, +1 = 2x attraction,
# -1 = 2x avoidance. Diverging colormap centered at chance. Cells that are not
# distinguishable from chance (BH-adjusted permutation p>=0.05) get a dot.
lift = pd.read_csv("data/sector_affinity_lift.csv", index_col=0)
tests = pd.read_csv("data/sector_affinity_tests.csv")
pval = (tests.pivot(index="from_sector", columns="to_sector", values="p_bh")
             .reindex(index=lift.index, columns=lift.columns))
# Cells with zero observed neighbors have log2(lift) = -inf; draw them at the
# bottom of the color scale and label them "0" so they are not mistaken for the
# masked diagonal.
log_lift = np.log2(lift.where(lift > 0, 2.0 ** -2.0))
annot = (lift.round(1).astype(str).where(lift > 0, "0")
         + np.where(pval < 0.05, "", " ·"))
# Mask the diagonal: self-retention lift (3-15x) dominates the scale and is
# already the subject of fig4. Masking it lets the cross-sector affinities show.
mask = np.eye(len(log_lift), dtype=bool)
# Cap the scale at +-2 (4x attraction / 0.25x avoidance). Without a cap, near-zero
# avoidance cells (log2 -> large negative) dominate and flatten the attraction side.
# Exact lift values remain in the annotations, so nothing is hidden.
vmax = 2.0
fig, ax = plt.subplots(figsize=(9.5, 7.5))
sns.heatmap(log_lift, mask=mask, annot=annot.values, fmt="", cmap="RdBu_r",
            center=0, vmin=-vmax, vmax=vmax,
            cbar_kws={"label": "log2(lift)   — below chance ◄ 0 ► above chance"},
            linewidths=0.5, linecolor="#E2E8F0", annot_kws={"fontsize": 8}, ax=ax)
# grey out the masked diagonal so it reads as "n/a, see fig4" not "missing"
ax.set_facecolor("#F1F5F9")
ax.set_title("Cross-sector semantic affinity, corrected for sector size\n"
             "(lift over a firm-count baseline; diagonal = self-retention, shown in fig4;\n"
             "· = not distinguishable from chance, BH-adjusted p ≥ 0.05)",
             loc="left", fontsize=11)
ax.set_xlabel("Neighbor sector"); ax.set_ylabel("Company's GICS sector")
plt.tight_layout(); plt.savefig("figures/fig2_affinity_heatmap.png", dpi=150); plt.close()

# ---- Table 1 data + Fig 3: case-study neighbor tables ----
CASE_STUDIES = ["BRK.B", "BR", "ECL", "APTV"]   # Table 1; Figure 3 shows the first three
nb = pd.read_csv("data/neighbors.csv")
cand = pd.read_csv("data/mismatch_candidates.csv").set_index("ticker")
rows = []
for tkr in CASE_STUDIES:
    g5 = nb[nb["ticker"] == tkr].head(5)
    rows.append({"ticker": tkr, "gics_sector": g5["sector"].iloc[0],
                 "is_candidate": tkr in cand.index,
                 "semantic_sector": cand["semantic_sector"].get(tkr),
                 "cross_rate_k10": round(nb.loc[nb["ticker"] == tkr, "cross"].mean(), 2),
                 "top5_cos_min": g5["score"].min(), "top5_cos_max": g5["score"].max(),
                 "top5_neighbors": "; ".join(f"{t} ({s})" for t, s in zip(g5["nbr_ticker"], g5["nbr_sector"]))})
pd.DataFrame(rows).to_csv("data/case_studies.csv", index=False)
picks = CASE_STUDIES[:3]
fig, axes = plt.subplots(len(picks), 1, figsize=(9, 2.2 * len(picks)))
if len(picks) == 1:
    axes = [axes]
for ax, tkr in zip(axes, picks):
    g = nb[nb["ticker"] == tkr].head(5)
    own = g["sector"].iloc[0]
    cell = [[r["nbr_ticker"], r["nbr_name"][:34], r["nbr_sector"], f"{r['score']:.2f}"]
            for _, r in g.iterrows()]
    ax.axis("off")
    ax.set_title(f"{tkr}  (GICS: {own})  — nearest semantic neighbors", loc="left", fontsize=10)
    t = ax.table(cellText=cell, colLabels=["Ticker", "Name", "Sector", "Cos"],
                 loc="center", cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
plt.tight_layout(); plt.savefig("figures/fig3_case_studies.png", dpi=150); plt.close()
print("Saved figures/fig1_umap.png, fig2_affinity_heatmap.png, fig3_case_studies.png")
