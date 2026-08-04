"""Generate the two paper figures:
  fig_face_demo.png      -- §3.7: mean face + genome-driven population difference + profile
  fig_kernel_adapter.png -- §3.5: facial-GWAS variants placed on the ADAPTER<->KERNEL axis
Saved into the cognimed/ dir (next to the .tex) for \\includegraphics.
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mesh_morph, gwas_adapter as ga, classify_kernel_adapter as cka

OUT = r"C:\Users\jacobsme\cognimed"

# ---------------- Figure A: the face demonstration ----------------
V,F,HL = mesh_morph.load(); A = mesh_morph.anchors(V)
eur1 = mesh_morph.morph(V,A,ga.population_dosages("nfe"),exaggerate=1.0)
eas1 = mesh_morph.morph(V,A,ga.population_dosages("eas"),exaggerate=1.0)
disp = np.linalg.norm(eas1-eur1,axis=1)
E=8.0
eurE = mesh_morph.morph(V,A,ga.population_dosages("nfe"),exaggerate=E)
easE = mesh_morph.morph(V,A,ga.population_dosages("eas"),exaggerate=E)

fig,ax = plt.subplots(1,3,figsize=(13,5.2))
ax[0].scatter(V[:,0],V[:,1],c=V[:,2],s=1.4,cmap="bone"); ax[0].set_aspect("equal"); ax[0].axis("off")
ax[0].set_title("(a) FaceBase mean face\n= frozen craniofacial kernel",fontsize=11)
sc=ax[1].scatter(V[:,0],V[:,1],c=disp,s=1.8,cmap="inferno"); ax[1].set_aspect("equal"); ax[1].axis("off")
ax[1].set_title("(b) genome-driven difference\nEast Asian − European (per vertex)",fontsize=11)
cb=plt.colorbar(sc,ax=ax[1],fraction=0.045,pad=0.02); cb.set_label("displacement",fontsize=8); cb.ax.tick_params(labelsize=7)
ax[2].scatter(eurE[:,2],eurE[:,1],s=1.0,c="steelblue",label="European");
ax[2].scatter(easE[:,2],easE[:,1],s=1.0,c="firebrick",alpha=0.55,label="East Asian")
ax[2].set_aspect("equal"); ax[2].axis("off"); ax[2].legend(markerscale=8,fontsize=9,loc="lower left")
ax[2].set_title(f"(c) profile overlay (×{int(E)})\nchin = EDAR, bridge = PAX3",fontsize=11)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_face_demo.png"),dpi=150,bbox_inches="tight")
print("wrote fig_face_demo.png"); plt.close()

# ---------------- Figure B: ADAPTER <-> KERNEL axis ----------------
rows = cka.classify()
DIV = 3.5   # divider: only EDAR (score 5) is a kernel-LEVEL variant; rest are adapter
fig,ax = plt.subplots(figsize=(9.2,5.8))
ax.axvspan(-0.5,DIV,color="#dce8f5"); ax.axvspan(DIV,5.7,color="#fbe6cf")  # adapter / kernel zones
ax.axvline(DIV,ls="--",color="#888",lw=1)
jit={"PAX1":-0.18,"GLI3":+0.18}   # separate the two co-located alar loci
loff={"PAX1":(7,-14),"GLI3":(7,6),"DCHS2":(8,-2)}
for r in rows:
    coding = r["consequence"].startswith("coding")
    x=r["kernel_score"]; y=r["pleiotropy_systems"]+jit.get(r["gene"],0)
    fd=r["freq_divergence"]; sz=90+ (fd*620 if fd else 0)
    ax.scatter(x,y,s=sz, marker="*" if coding else "o",
               c="firebrick" if coding else "steelblue", edgecolor="k", zorder=3)
    note = r["gene"] + ("  (coding, swept)" if coding else "")
    ax.annotate(note,(x,y),xytext=loff.get(r["gene"],(7,6)),textcoords="offset points",
                fontsize=10,fontweight="bold")
ax.set_xlim(-0.5,5.7); ax.set_ylim(0,8.2)
ax.set_xlabel("kernel score  (consequence + pleiotropy + selection-divergence + constraint)",fontsize=10)
ax.set_ylabel("pleiotropy (organ systems)",fontsize=10)
ax.text(1.4,7.5,"ADAPTER\ntunes the LLM promoter-MLP\n(common cis-regulatory variants)",
        ha="center",fontsize=10.5,color="#235",fontweight="bold")
ax.text(4.5,7.5,"KERNEL\nedits the zygote /\ndevelopmental machinery",
        ha="center",fontsize=10.5,color="#834",fontweight="bold")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],marker="*",color="w",markerfacecolor="firebrick",markeredgecolor="k",markersize=14,label="coding variant"),
                   Line2D([],[],marker="o",color="w",markerfacecolor="steelblue",markeredgecolor="k",markersize=10,label="regulatory variant"),
                   Line2D([],[],marker="o",color="w",markerfacecolor="gray",markeredgecolor="k",markersize=14,label="size ∝ freq divergence")],
          loc="center right",fontsize=9,framealpha=0.95)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"fig_kernel_adapter.png"),dpi=150,bbox_inches="tight")
print("wrote fig_kernel_adapter.png"); plt.close()
