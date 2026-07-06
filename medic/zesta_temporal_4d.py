"""
ZESTA temporal 4D: the germ-layer bioelectric field EMERGING across zebrafish development.
==========================================================================================

zesta_train.py fit the spatial NCA on ONE stage (24 hpf) and reported a single scalar
(germ-layer spread) at the other five timepoints. This is the 4D upgrade: run the full
forward NCA field AND the sparse-anchor reconstruction at each of the six real ZESTA
stages (3.3, 5.25, 10, 12, 18, 24 hpf), and track how the field's STRUCTURE emerges as
the embryo differentiates.

The developmental fact the atlas hands us:
  3.3 hpf  1167 bins, ALL Blastodisc                  -> one class, polarity spread 0 mV
  5.25     germ layers appear (epidermis/meso/yolk)   -> spread 7 mV
  10-12    mesoderm + first neural (Keel -> Rod)       -> spread 9-11 mV
  18-24    forebrain / spinal cord / EYE / crest       -> full neural-hyper vs yolk-depol axis

So the bioelectric field is not a fixed backdrop: it CONDENSES out of a uniform blastula
into a structured field as the germ layers separate. We measure that four ways per stage:
  (1) forward NCA field (the pattern itself, rendered on the real slice),
  (2) germ-layer polarity spread (mV),
  (3) the two poles SEPARATING -- mean V_m of the neural domain vs the yolk/ventral domain,
  (4) reconstruction R^2 from 15% anchors -- does the gap-junction operator have anything
      to reconstruct yet? (flat early -> structured late),
  (5) Moran's I of the field on the kNN graph -- spatial organisation, rising with time.
Plus a common-AP-axis profile: register every stage's long axis and overlay the mean-V_m
profile, so the anterior-hyperpolarised gradient can be seen sharpening across development.

HONEST SCOPE: Stereo-seq sections are destructive, so these are six DIFFERENT slices, not
the same cells tracked -- this is the field's population-level emergence across staged real
embryos, not a per-cell time-lapse. Set-points are the same placed germ-layer physiology as
zesta_train (no larp). The 4D content is real developmental structure, computed forward.

Run: python -m medic.zesta_temporal_4d
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/organ_cascade")
ATLAS = Path("data/zesta/zf_sixtime_slice.h5ad")
V_ZYGOTE = -70.0
TIMES = ["3.3hpf", "5.25hpf", "10hpf", "12hpf", "18hpf", "24hpf"]
TIME_HPF = {"3.3hpf": 3.3, "5.25hpf": 5.25, "10hpf": 10, "12hpf": 12, "18hpf": 18, "24hpf": 24}
K = 8
torch.manual_seed(0)

LAYER_VM = {
    "Eye": -68.0, "Forebrain": -66.0, "Nervous System": -65.0, "Spinal Cord": -65.0,
    "Neural Rod": -65.0, "Neural Keel": -64.0, "Neural Crest": -55.0, "Otic Vesicle": -46.0,
    "Presumptive Mesoderm, Presumptive Ectoderm": -48.0, "Margin": -46.0,
    "Mesoderm": -45.0, "Somite": -45.0, "Blastodisc": -50.0, "Proliferative Like Cell": -50.0,
    "Epidermal": -35.0, "Hypoblast": -32.0, "Yolk Syncytial Layer": -28.0,
}
NEURAL = {"Eye", "Forebrain", "Nervous System", "Spinal Cord", "Neural Rod", "Neural Keel", "Neural Crest"}
YOLK_VENTRAL = {"Epidermal", "Hypoblast", "Yolk Syncytial Layer"}


def load_all():
    import anndata as ad
    a = ad.read_h5ad(ATLAS)
    o = a.obs
    tvals = o["time"].astype(str).values
    lay = o["layer_annotation"].astype(str).values
    xy_all = np.c_[o["spatial_x"].values, o["spatial_y"].values].astype(float)
    stages = {}
    for t in TIMES:
        m = (tvals == t) & np.array([x in LAYER_VM for x in lay])
        anno = lay[m]
        stages[t] = dict(xy=xy_all[m], anno=anno,
                         ref=np.array([LAYER_VM[x] for x in anno], np.float32))
    return stages


def nca_field(ref, nbr, k_relax=0.25, k_gj=0.12, T=60):
    ref_t = torch.tensor(ref); nbr_t = torch.tensor(nbr, dtype=torch.long)
    V = torch.full((len(ref),), V_ZYGOTE)
    for _ in range(T):
        V = V + k_relax * (ref_t - V) + k_gj * (V[nbr_t].mean(1) - V)
    return V.numpy()


def reconstruct(ref, nbr, anchor_frac=0.15):
    n = len(ref)
    if n < 40 or np.std(ref) < 1e-3:          # nothing to reconstruct (flat/blastula)
        return float("nan")
    rng = np.random.RandomState(0)
    anchor = np.zeros(n, bool); anchor[rng.choice(n, int(anchor_frac * n), replace=False)] = True
    held = ~anchor
    ref_t = torch.tensor(ref); nbr_t = torch.tensor(nbr, dtype=torch.long)
    anchor_t = torch.tensor(anchor)
    pg = torch.tensor(1.0, requires_grad=True); opt = torch.optim.Adam([pg], lr=0.1)

    def rec(kgj, T=200):
        V = torch.full((n,), V_ZYGOTE)
        for _ in range(T):
            V = torch.where(anchor_t, ref_t, V + kgj * (V[nbr_t].mean(1) - V))
        return V
    for _ in range(150):
        opt.zero_grad()
        V = rec(torch.sigmoid(pg) * 0.5)
        loss = ((V[torch.tensor(held)] - ref_t[torch.tensor(held)]) ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        Vr = rec(torch.sigmoid(pg) * 0.5).numpy()
    h = held
    return float(1 - np.sum((Vr[h] - ref[h]) ** 2) / np.sum((ref[h] - ref[h].mean()) ** 2))


def morans_i(v, nbr):
    if np.std(v) < 1e-6:
        return 0.0
    z = v - v.mean()
    num = np.sum(z[:, None] * z[nbr])          # Σ_i Σ_j∈N(i) z_i z_j
    W = nbr.size
    n = len(v)
    return float((n / W) * num / np.sum(z ** 2))


def ap_profile(xy, v, nbins=12):
    """Register the section long axis (PCA) as AP, orient anterior = low-V_m (neural) end,
    return mean V_m in nbins along AP (nan-padded where empty)."""
    c = xy - xy.mean(0)
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    ap = c @ vt[0]                              # projection on principal axis
    # orient: put the most-hyperpolarised third at ap=0 (anterior)
    if np.std(v) > 1e-6:
        lo = v < np.percentile(v, 33)
        if ap[lo].mean() > ap.mean():
            ap = -ap
    t = (ap - ap.min()) / (ap.max() - ap.min() + 1e-9)
    prof = np.full(nbins, np.nan)
    for b in range(nbins):
        m = (t >= b / nbins) & (t < (b + 1) / nbins)
        if m.sum() > 3:
            prof[b] = v[m].mean()
    return prof


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stages = load_all()
    rows = []
    fields = {}
    profiles = {}
    for t in TIMES:
        d = stages[t]
        xy, anno, ref = d["xy"], d["anno"], d["ref"]
        n = len(xy)
        nbr = cKDTree(xy).query(xy, k=K + 1)[1][:, 1:]
        V = nca_field(ref, nbr)
        fields[t] = (xy, V)
        neural = np.array([a in NEURAL for a in anno])
        yolk = np.array([a in YOLK_VENTRAL for a in anno])
        v_neural = float(V[neural].mean()) if neural.any() else float("nan")
        v_yolk = float(V[yolk].mean()) if yolk.any() else float("nan")
        r2 = reconstruct(ref, nbr)
        mi = morans_i(V, nbr)
        profiles[t] = ap_profile(xy, V)
        rows.append(dict(time=t, hpf=TIME_HPF[t], n_bins=n, spread=float(np.std(ref)),
                         v_neural=v_neural, v_yolk=v_yolk,
                         pole_gap=(v_yolk - v_neural) if (neural.any() and yolk.any()) else float("nan"),
                         recon_r2=r2, morans_i=mi, neural_frac=float(neural.mean())))
        print(f"{t:8s} n={n:5d} spread={rows[-1]['spread']:4.1f}mV  "
              f"Vneural={v_neural:6.1f} Vyolk={v_yolk:6.1f} gap={rows[-1]['pole_gap'] if rows[-1]['pole_gap']==rows[-1]['pole_gap'] else float('nan'):5.1f}  "
              f"recon_R2={r2 if r2==r2 else float('nan'):5.2f}  MoranI={mi:4.2f}")

    _figure(fields, rows, profiles)
    json.dump(dict(stages=rows), open(OUT / "zesta_temporal_4d.json", "w"), indent=2)
    print("\nsaved", OUT / "zesta_temporal_4d.json")


def _figure(fields, rows, profiles):
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0])

    # top row: the field emerging on the real slice at each stage
    for i, t in enumerate(TIMES):
        xy, V = fields[t]
        ax = fig.add_subplot(gs[0, i])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=V, cmap="RdBu_r", vmin=-70, vmax=-25, s=5, linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{t}  (n={len(xy)})", fontsize=9)
        if i == 0:
            ax.set_ylabel("forward NCA field", fontsize=9)
    cax = fig.add_axes([0.92, 0.56, 0.008, 0.32]); fig.colorbar(sc, cax=cax, label="$V_m$ (mV)")

    hpf = [r["hpf"] for r in rows]
    # (a) the two poles separating
    ax = fig.add_subplot(gs[1, 0])
    vn = [r["v_neural"] for r in rows]; vy = [r["v_yolk"] for r in rows]
    ax.plot(hpf, vn, "o-", color="tab:blue", label="neural domain")
    ax.plot(hpf, vy, "o-", color="tab:red", label="yolk / ventral")
    ax.fill_between(hpf, vn, vy, color="0.85", zorder=0)
    ax.set_xlabel("hpf"); ax.set_ylabel("mean $V_m$ (mV)")
    ax.set_title("(a) the two poles separate\n(polarity condenses)", fontsize=9); ax.legend(fontsize=7)

    # (b) germ-layer spread
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(hpf, [r["spread"] for r in rows], "o-", color="tab:purple")
    ax.set_xlabel("hpf"); ax.set_ylabel("germ-layer $V_m$ spread (mV)")
    ax.set_title("(b) polarity spread", fontsize=9)

    # (c) reconstruction R^2 emerging
    ax = fig.add_subplot(gs[1, 2])
    r2 = [r["recon_r2"] for r in rows]
    ax.plot(hpf, r2, "o-", color="tab:green")
    ax.set_xlabel("hpf"); ax.set_ylabel("reconstruction $R^2$ (15% anchors)")
    ax.set_title("(c) operator has structure\nto reconstruct only once it emerges", fontsize=9)
    ax.set_ylim(0, 1)

    # (d) Moran's I
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(hpf, [r["morans_i"] for r in rows], "o-", color="tab:orange")
    ax.set_xlabel("hpf"); ax.set_ylabel("Moran's $I$ of the field")
    ax.set_title("(d) spatial organisation rises", fontsize=9)

    # (e) neural fraction
    ax = fig.add_subplot(gs[1, 4])
    ax.plot(hpf, [r["neural_frac"] for r in rows], "o-", color="tab:cyan")
    ax.set_xlabel("hpf"); ax.set_ylabel("neural-domain fraction")
    ax.set_title("(e) neural domain grows in", fontsize=9)

    # (f) AP profile overlay sharpening
    ax = fig.add_subplot(gs[1, 5])
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(TIMES):
        p = profiles[t]
        xs = np.linspace(0, 1, len(p))
        ax.plot(xs, p, "-", color=cmap(i / 5), label=t, lw=1.6)
    ax.set_xlabel("AP axis (anterior -> posterior)"); ax.set_ylabel("mean $V_m$ (mV)")
    ax.set_title("(f) anterior-hyperpolarised\ngradient sharpens", fontsize=9)
    ax.legend(fontsize=6)

    fig.suptitle("ZESTA temporal 4D: the germ-layer bioelectric field condensing out of a uniform blastula across "
                 "zebrafish development (3.3 -> 24 hpf).\nThe field is computed forward (NCA) on each real staged "
                 "section; polarity, spatial organisation and reconstructability all emerge as the germ layers separate.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.91, 0.94])
    fig.savefig(OUT / "zesta_temporal_4d.png", dpi=125, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "zesta_temporal_4d.png")


if __name__ == "__main__":
    main()
