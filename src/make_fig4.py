import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

counts={'Utilities':31,'Health Care':59,'Financials':75,'Consumer Staples':36,'Real Estate':31,
'Information Technology':72,'Energy':21,'Communication Services':23,'Industrials':80,
'Consumer Discretionary':48,'Materials':26}
porosity={'Utilities':9.4,'Health Care':15.6,'Financials':16.5,'Consumer Staples':24.4,'Real Estate':29.7,
'Information Technology':33.2,'Energy':38.6,'Communication Services':48.3,'Industrials':48.4,
'Consumer Discretionary':64.4,'Materials':75.4}
N=502
rows=[]
for s in counts:
    obs=100-porosity[s]; chance=100*(counts[s]-1)/(N-1); rows.append((s,counts[s],obs,obs/chance))
rows.sort(key=lambda r:r[3])  # ascending so highest lift on top in barh

labels=[r[0] for r in rows]; lifts=[r[3] for r in rows]; intra=[r[2] for r in rows]; ns=[r[1] for r in rows]
y=np.arange(len(rows))
norm=plt.Normalize(min(lifts),max(lifts))
colors=plt.cm.viridis(norm(lifts))

fig,ax=plt.subplots(figsize=(9.2,6.2))
ax.barh(y,lifts,color=colors,edgecolor="white",height=0.72)
for i,(l,iv,n) in enumerate(zip(lifts,intra,ns)):
    ax.text(l+0.25,i,f"{l:.1f}x   ({iv:.0f}% self, n={n})",va="center",fontsize=10,color="#1E293B")
ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=11)
ax.set_xlim(0,max(lifts)+4.2)
ax.set_xlabel("Self-retention of nearest peers, as a multiple of the size-based random baseline  (lift x)",fontsize=11)
ax.set_title("Sector cohesion adjusted for firm count (lift over chance)",
             fontsize=14,fontweight="bold",loc="left",pad=14)
ax.axvline(1.0,color="#94A3B8",linestyle="--",linewidth=1)
ax.text(1.0,len(rows)-0.3,"  chance (1x)",fontsize=9,color="#64748B",va="top")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
ax.tick_params(left=False)
plt.tight_layout(); plt.savefig("fig4_cohesion_lift.png",dpi=150); plt.close()
print("saved fig4_cohesion_lift.png")
