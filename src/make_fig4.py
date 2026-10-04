"""
make_fig4.py
------------
Figure 4: sector self-retention as a multiple of its size-based random
baseline, (n - 1)/(N - 1). Reads data/sector_cohesion.csv (from 03_analyze.py).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

coh = pd.read_csv("data/sector_cohesion.csv").sort_values("lift")   # ascending: highest lift on top

labels = coh["sector"].tolist(); lifts = coh["lift"].to_numpy()
intra = (100 * coh["self_retention"]).to_numpy(); ns = coh["n"].to_numpy()
y = np.arange(len(coh))
norm = plt.Normalize(lifts.min(), lifts.max())
colors = plt.cm.viridis(norm(lifts))

fig, ax = plt.subplots(figsize=(9.2, 6.2))
ax.barh(y, lifts, color=colors, edgecolor="white", height=0.72)
for i, (l, iv, n) in enumerate(zip(lifts, intra, ns)):
    ax.text(l + 0.25, i, f"{l:.1f}x   ({iv:.0f}% self, n={n})", va="center", fontsize=10, color="#1E293B")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(0, lifts.max() + 4.2)
ax.set_xlabel("Self-retention of nearest peers, as a multiple of the size-based random baseline  (lift x)", fontsize=11)
ax.set_title("Sector cohesion adjusted for firm count (lift over chance)",
             fontsize=14, fontweight="bold", loc="left", pad=14)
ax.axvline(1.0, color="#94A3B8", linestyle="--", linewidth=1)
ax.text(1.0, len(coh) - 0.3, "  chance (1x)", fontsize=9, color="#64748B", va="top")
for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout(); plt.savefig("figures/fig4_cohesion_lift.png", dpi=150); plt.close()
print("saved figures/fig4_cohesion_lift.png")
