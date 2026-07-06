"""
Genome-DERIVED gap-junction operator: weighting the Laplacian by measured connexin expression.
================================================================================================

mosta_train / zesta_train / hesta_train all reconstruct the germ-layer V_m field from 15%
sparse anchors via a gap-junction operator -- but that operator is a UNIFORM k-nearest-
neighbour graph with a single scalar coupling rate. The graph is geometry; the conductance
is hand-uniform. That is the honest weak link: the operator is not genome-conditioned.

Here we make the operator itself come from the genome. A gap junction is a channel built
from connexons on BOTH apposed membranes, so the junctional conductance between two cells
scales with how much connexin each one expresses. We read the measured connexin transcripts
per bin from the SAME Stereo-seq atlas (Cx43/Gja1 is the dominant developmental connexin),
build a per-bin conductance g_i, and weight every graph edge by the geometric mean

    w_ij = sqrt(g_i * g_j)          (two hemichannels in series: both cells must express)

so the diffusion operator is now the real embryo's own connexin field, not a constant. This
instantiates the Paper-#4 claim ("the conductance profile IS the cymatic generator") on real
tissue, and it survives the operon-null result: we are scaling the COUPLING by a measured
gene, not deriving the set-point from it (the null killed accessibility->set-point, not
accessibility->conductance, which is exactly what connexins physically are).

Two tests, per species (mouse E9.5 MOSTA / zebrafish 24 hpf ZESTA / human CS12-13 HESTA):
  (1) RECONSTRUCTION  -- reconstruct the held-out field from 15% anchors with the UNIFORM
      operator vs the connexin-WEIGHTED operator; compare held-out R^2 overall and on the
      tissue-BOUNDARY bins (where a genome-conditioned operator should help most, because
      low-conductance seams preserve the compartment discontinuity uniform coupling smears).
  (2) COUPLING DOMAINS -- do the connexin coupling domains ALIGN with germ-layer boundaries?
      Compare mean edge conductance WITHIN a tissue vs ACROSS a tissue boundary. Levin's
      picture predicts across < within (seams are electrically insulated). Reported honestly
      whether it confirms or not.

HONEST SCOPE: the target is still the germ-layer V_m floor on the real layout; the connexin
read is a real measured gene, coarse at Stereo-seq bin resolution (Cx43 dominant, others
sparse). We report the comparison as-is -- an improvement, a wash, or a boundary-only gain.

Run: python -m medic.gj_operator_train
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/organ_cascade")
V_ZYGOTE = -70.0
K = 8
ANCHOR_FRAC = 0.15
G_FLOOR = 0.05          # minimum edge conductance so the graph never disconnects
torch.manual_seed(0)

# ---- germ-layer V_m floors (same anchors the three train scripts place) ----
MOSTA_VM = {
    "Brain": -65.0, "Spinal cord": -65.0, "Notochord": -60.0,
    "Sclerotome": -50.0, "Mesenchyme": -45.0, "Head mesenchyme": -45.0, "AGM": -45.0,
    "Heart": -40.0, "Surface ectoderm": -35.0,
    "Liver": -32.0, "Lung primordium": -32.0, "Pancreas primordium": -30.0,
}
ZESTA_VM = {
    "Eye": -68.0, "Forebrain": -66.0, "Nervous System": -65.0, "Spinal Cord": -65.0,
    "Neural Rod": -65.0, "Neural Keel": -64.0, "Neural Crest": -55.0, "Otic Vesicle": -46.0,
    "Presumptive Mesoderm, Presumptive Ectoderm": -48.0, "Margin": -46.0,
    "Mesoderm": -45.0, "Somite": -45.0, "Blastodisc": -50.0, "Proliferative Like Cell": -50.0,
    "Epidermal": -35.0, "Hypoblast": -32.0, "Yolk Syncytial Layer": -28.0,
}
HESTA_VM = {
    "Brain": -65.0, "Branchial Arch": -50.0, "Mesenchyme": -45.0, "Mesonephron": -42.0,
    "Heart": -40.0, "Epidermis": -35.0, "Erythroblasts": -30.0,
}


def is_connexin(name: str) -> bool:
    """Real connexin / pannexin gene, excluding CXXC-domain, chemokine (CXCL/CXCR), CXADR."""
    lo = name.lower()
    if lo.startswith("gj") or lo.startswith("panx"):
        return True
    # zebrafish connexins: cx43, cx43.4, cx30.3, cx44.2 ... -> "cx" + a digit
    if lo.startswith("cx") and len(lo) > 2 and lo[2].isdigit():
        return True
    return False


SPECIES = {
    "MOUSE": dict(file="data/mosta/E9.5_E2S2.MOSTA.h5ad", anno_col="annotation",
                  vm=MOSTA_VM, coords="obsm", time=None, tag="mouse E9.5 MOSTA"),
    "FISH":  dict(file="data/zesta/zf_sixtime_slice.h5ad", anno_col="layer_annotation",
                  vm=ZESTA_VM, coords=("spatial_x", "spatial_y"), time="24hpf", tag="zebrafish 24 hpf ZESTA"),
    "HUMAN": dict(file="data/hesta/CS12-13_E1S1_HESTA.h5ad", anno_col="celltype",
                  vm=HESTA_VM, coords=("x", "y"), time=None, tag="human CS12-13 HESTA"),
}


def load_species(spec):
    import anndata as ad
    a = ad.read_h5ad(spec["file"])
    o = a.obs
    anno_all = o[spec["anno_col"]].astype(str).values
    keep = np.array([t in spec["vm"] for t in anno_all])
    if spec["time"] is not None:
        keep = keep & (o["time"].astype(str).values == spec["time"])
    # coordinates
    if spec["coords"] == "obsm":
        xy = np.asarray(a.obsm["spatial"], float)[keep]
    else:
        cx, cy = spec["coords"]
        xy = np.c_[o[cx].values, o[cy].values][keep].astype(float)
    anno = anno_all[keep]
    tissues = sorted(set(anno))
    tid = np.array([tissues.index(t) for t in anno])
    ref = np.array([spec["vm"][t] for t in anno], np.float32)

    # per-bin connexin conductance from measured transcripts
    vn = np.asarray(a.var_names, str)
    gj_idx = [i for i, v in enumerate(vn) if is_connexin(v)]
    X = a.X
    sub = X[:, gj_idx]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    sub = sub[keep]
    gj_genes = [vn[i] for i in gj_idx]
    gj_raw = np.log1p(sub.sum(1)).astype(np.float32)      # pooled connexin load per bin (log)
    return xy, anno, tissues, tid, ref, gj_raw, gj_genes, sub


def build_graph(xy):
    nbr = cKDTree(xy).query(xy, k=K + 1)[1][:, 1:]        # (n, K) exclude self
    return nbr


def smooth_on_graph(v, nbr, rounds=2):
    """Denoise a sparse per-bin signal by averaging with graph neighbours."""
    v = v.copy()
    for _ in range(rounds):
        v = 0.5 * v + 0.5 * v[nbr].mean(1)
    return v


def conductance(gj_raw, nbr):
    """Per-bin conductance g_i in [G_FLOOR, 1]: smoothed, percentile-normalised connexin load."""
    g = smooth_on_graph(gj_raw, nbr, rounds=2)
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    g = np.clip((g - lo) / (hi - lo + 1e-9), 0, 1)
    return (G_FLOOR + (1 - G_FLOOR) * g).astype(np.float32)


def edge_weights(g, nbr, genome=True):
    """(n,K) edge weights. genome: sqrt(g_i*g_j) series hemichannels; else uniform ones."""
    if not genome:
        return np.ones_like(nbr, dtype=np.float32)
    gi = g[:, None]                     # (n,1)
    gj = g[nbr]                         # (n,K)
    return np.sqrt(gi * gj).astype(np.float32)


def reconstruct(ref_t, anchor_t, nbr_t, W_t, kgj, T=200):
    """Diffuse the field from the anchors under a weighted graph operator."""
    Wsum = W_t.sum(1)                                   # (n,)
    V = torch.full((ref_t.shape[0],), V_ZYGOTE)
    for _ in range(T):
        wmean = (W_t * V[nbr_t]).sum(1) / Wsum
        V = torch.where(anchor_t, ref_t, V + kgj * (wmean - V))
    return V


def fit_and_score(ref, nbr, W, anchor, held):
    ref_t = torch.tensor(ref); anchor_t = torch.tensor(anchor)
    nbr_t = torch.tensor(nbr, dtype=torch.long); W_t = torch.tensor(W)
    pg = torch.tensor(1.0, requires_grad=True)
    opt = torch.optim.Adam([pg], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        Vr = reconstruct(ref_t, anchor_t, nbr_t, W_t, torch.sigmoid(pg) * 0.5)
        loss = ((Vr[torch.tensor(held)] - ref_t[torch.tensor(held)]) ** 2).mean()
        loss.backward(); opt.step()
    kgj = float(torch.sigmoid(pg) * 0.5)
    with torch.no_grad():
        V = reconstruct(ref_t, anchor_t, nbr_t, W_t, torch.tensor(kgj)).numpy()
    return V, kgj


def r2_on(v, ref, mask):
    return float(1 - np.sum((v[mask] - ref[mask]) ** 2) /
                 np.sum((ref[mask] - ref[mask].mean()) ** 2))


def run_species(name, spec):
    print(f"\n===== {name}  ({spec['tag']}) =====")
    xy, anno, tissues, tid, ref, gj_raw, gj_genes, gj_sub = load_species(spec)
    n = len(xy)
    nbr = build_graph(xy)
    g = conductance(gj_raw, nbr)
    print(f"  {n} bins, {len(tissues)} tissues; {len(gj_genes)} connexin genes pooled")
    print(f"  conductance CV across bins = {g.std()/g.mean():.2f}  (0 = uniform)")

    # boundary bins: have at least one cross-tissue neighbour
    cross_edge = (tid[nbr] != tid[:, None])              # (n,K) True where neighbour is other tissue
    boundary = cross_edge.any(1)
    # coupling-domain test: mean edge conductance within vs across tissue boundaries
    W_g = edge_weights(g, nbr, genome=True)
    w_within = float(W_g[~cross_edge].mean())
    w_across = float(W_g[cross_edge].mean())
    print(f"  coupling domains: edge conductance within-tissue {w_within:.3f} vs "
          f"across-boundary {w_across:.3f}  (ratio {w_across/w_within:.2f}; <1 = insulated seams)")

    # anchors
    rng = np.random.RandomState(0)
    anchor = np.zeros(n, bool)
    anchor[rng.choice(n, int(ANCHOR_FRAC * n), replace=False)] = True
    held = ~anchor

    W_u = edge_weights(g, nbr, genome=False)
    V_u, k_u = fit_and_score(ref, nbr, W_u, anchor, held)
    V_g, k_g = fit_and_score(ref, nbr, W_g, anchor, held)

    hb = held & boundary
    res = dict(
        n_bins=n, tissues=tissues, n_connexin_genes=len(gj_genes), connexin_genes=gj_genes[:40],
        conductance_cv=float(g.std() / g.mean()),
        w_within=w_within, w_across=w_across, coupling_ratio=w_across / w_within,
        uniform=dict(r2_held=r2_on(V_u, ref, held), r2_boundary=r2_on(V_u, ref, hb), kgj=k_u),
        genome=dict(r2_held=r2_on(V_g, ref, held), r2_boundary=r2_on(V_g, ref, hb), kgj=k_g),
        n_boundary_held=int(hb.sum()),
    )
    du, dg = res["uniform"], res["genome"]
    print(f"  RECONSTRUCTION held-out R^2:  uniform {du['r2_held']:.3f}   genome-derived {dg['r2_held']:.3f}"
          f"   (delta {dg['r2_held']-du['r2_held']:+.3f})")
    print(f"  on BOUNDARY bins only     :  uniform {du['r2_boundary']:.3f}   genome-derived {dg['r2_boundary']:.3f}"
          f"   (delta {dg['r2_boundary']-du['r2_boundary']:+.3f})")
    return dict(res=res, xy=xy, anno=anno, tissues=tissues, ref=ref, g=g,
                V_u=V_u, V_g=V_g, anchor=anchor, boundary=boundary)


def figure(results):
    ncol = 5
    fig, axes = plt.subplots(3, ncol, figsize=(21, 11))
    for r, (name, d) in enumerate(results.items()):
        xy = d["xy"]; x, y = xy[:, 0], xy[:, 1]; ref = d["ref"]
        inv = (name == "MOUSE")   # mouse plotted y-down like its train fig

        def emb(ax):
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if inv:
                ax.invert_yaxis()

        vm_kw = dict(cmap="RdBu_r", vmin=-70, vmax=-25, s=4, linewidths=0)
        ax = axes[r, 0]; sc = ax.scatter(x, y, c=ref, **vm_kw); emb(ax)
        ax.set_ylabel(f"{name}\n{d['tissues'] and ''}", fontsize=10)
        if r == 0: ax.set_title("target germ-layer $V_m$ floor", fontsize=9)

        ax = axes[r, 1]; sc2 = ax.scatter(x, y, c=d["g"], cmap="viridis", s=4, linewidths=0); emb(ax)
        if r == 0: ax.set_title("connexin conductance $g_i$\n(measured, genome-derived)", fontsize=9)
        fig.colorbar(sc2, ax=ax, fraction=0.045)

        ax = axes[r, 2]; ax.scatter(x, y, c=d["V_u"], **vm_kw); emb(ax)
        du = d["res"]["uniform"]
        if r == 0: ax.set_title("reconstructed: UNIFORM operator", fontsize=9)
        ax.text(0.5, -0.06, f"held-out $R^2$={du['r2_held']:.3f}", transform=ax.transAxes,
                ha="center", fontsize=8)

        ax = axes[r, 3]; ax.scatter(x, y, c=d["V_g"], **vm_kw); emb(ax)
        dg = d["res"]["genome"]
        if r == 0: ax.set_title("reconstructed: GENOME-derived operator", fontsize=9)
        ax.text(0.5, -0.06, f"held-out $R^2$={dg['r2_held']:.3f}", transform=ax.transAxes,
                ha="center", fontsize=8)

        ax = axes[r, 4]
        labels = ["all\nheld-out", "boundary\nbins"]
        uni = [du["r2_held"], du["r2_boundary"]]
        gen = [dg["r2_held"], dg["r2_boundary"]]
        xp = np.arange(2)
        ax.bar(xp - 0.2, uni, 0.4, label="uniform", color="0.6")
        ax.bar(xp + 0.2, gen, 0.4, label="genome", color="tab:green")
        ax.set_xticks(xp); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("reconstruction $R^2$", fontsize=8)
        ax.axhline(0, color="k", lw=0.5)
        if r == 0: ax.set_title("uniform vs genome-derived", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Genome-derived gap-junction operator: weighting the diffusion Laplacian by measured connexin "
                 "expression (Cx43/Gja1 dominant)\nacross the three atlases -- the operator becomes the embryo's "
                 "own conductance field rather than a uniform graph.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "gj_operator_train.png", dpi=125, bbox_inches="tight")
    plt.close(fig); print("\nsaved", OUT / "gj_operator_train.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in SPECIES.items():
        results[name] = run_species(name, spec)
    figure(results)
    dump = {name: d["res"] for name, d in results.items()}
    json.dump(dump, open(OUT / "gj_operator_train.json", "w"), indent=2)
    print("saved", OUT / "gj_operator_train.json")

    print("\n==== SUMMARY: uniform vs genome-derived operator (held-out reconstruction R^2) ====")
    for name, d in results.items():
        du, dg = d["res"]["uniform"], d["res"]["genome"]
        print(f"  {name:6s}  all {du['r2_held']:.3f} -> {dg['r2_held']:.3f} "
              f"({dg['r2_held']-du['r2_held']:+.3f})   boundary {du['r2_boundary']:.3f} -> "
              f"{dg['r2_boundary']:.3f} ({dg['r2_boundary']-du['r2_boundary']:+.3f})   "
              f"cond-CV {d['res']['conductance_cv']:.2f} seam-ratio {d['res']['coupling_ratio']:.2f}")


if __name__ == "__main__":
    main()
