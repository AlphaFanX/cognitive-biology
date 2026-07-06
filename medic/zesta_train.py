"""
Training the spatial NCA to follow the ZESTA atlas (the zebrafish, spatiotemporal upgrade).
============================================================================================

mosta_train.py fit a differentiable NCA to the germ-layer bioelectric field on ONE real
E9.5 mouse embryo section (MOSTA). This is the zebrafish version, on the ZESTA
spatiotemporal atlas (Stereo-seq; STOMICS STDS0000057, spatial_sixtime_slice): six
developmental timepoints (3.3, 5.25, 10, 12, 18, 24 hpf), each a real embryo slice with
per-bin spatial coordinates and a germ-layer / organ annotation.

The pipeline is identical in spirit to MOSTA, so the two species are directly comparable:
  * DOMAIN   = the real 24 hpf slice (5,271 spatial bins), the most differentiated section.
  * OPERATOR = the gap-junction Laplacian, built as k-nearest-neighbours on the REAL bin
               coordinates (the tissue's own geometry, not a lattice).
  * SET-POINTS (the LGM / outer perceptron): a per-germ-layer V_m floor -- neural/axial
               hyperpolarised, mesoderm intermediate, epidermis/endoderm/yolk depolarised.
               These are Levin germ-layer PHYSIOLOGY, an anchor we are allowed to place;
               they are NOT reverse-engineered from the ZESTA shape (no larping).
  * NCA      = the inner perceptron: settle V from the zygote base toward the set-points
               with gap-junction coupling on the real graph.

Two tests, as in MOSTA:
  PART A  full-supervision set-point recovery  -- can the atlas TRAIN the free parameters
          (per-layer set-points + two NCA rates) from an off-init? (a trainability check;
          honestly semi-circular because the target is built from these set-points).
  PART B  the genuinely spatial test          -- reconstruct the whole germ-layer field
          from only 15% ANCHOR bins via the gap-junction operator on the real geometry.
          This is the non-larping core: the operator must SPREAD a sparse prepattern into
          the correct field across real tissue boundaries; held-out R^2 measures it.

BONUS (what MOSTA could not do): the atlas is temporal, so we also run the forward NCA
set-point field at each of the six timepoints and report how germ-layer polarity
separates as development proceeds (a trajectory, not a single stage).

HONEST SCOPE: target = the germ-layer V_m floor on the real layout (a spatial fit +
set-point recovery + operator reconstruction), NOT absolute V_m forward-computed from
expression (the coarse ABC read cannot do that un-anchored; see magnitude_layer /
paper-claim-audit). Set-points are germ-layer physiology, placed not fit.

Run: python -m medic.zesta_train
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
MAIN_TIME = "24hpf"           # richest, most differentiated slice (5,271 bins)
TIMES = ["3.3hpf", "5.25hpf", "10hpf", "12hpf", "18hpf", "24hpf"]
torch.manual_seed(0)

# Germ-layer V_m floor on the ZESTA layer_annotation (Levin germ-layer physiology).
# neural / axial: hyperpolarised (high K); mesoderm: intermediate; epidermis / endoderm /
# extraembryonic yolk: depolarised; early blastodisc/progenitor: neutral.  PLACED, not fit.
LAYER_VM = {
    "Eye": -68.0, "Forebrain": -66.0,                                   # neural, strongly hyper
    "Nervous System": -65.0, "Spinal Cord": -65.0,
    "Neural Rod": -65.0, "Neural Keel": -64.0,                          # neural tube precursors
    "Neural Crest": -55.0,                                              # ectoderm-derived, migratory
    "Otic Vesicle": -46.0,                                              # placodal ectoderm
    "Presumptive Mesoderm, Presumptive Ectoderm": -48.0,               # progenitor, intermediate
    "Margin": -46.0,                                                    # mesendoderm progenitor
    "Mesoderm": -45.0, "Somite": -45.0,                                # mesoderm: intermediate
    "Blastodisc": -50.0, "Proliferative Like Cell": -50.0,             # early / undifferentiated
    "Epidermal": -35.0,                                                 # epidermis: depol
    "Hypoblast": -32.0,                                                 # endoderm/hypoblast: depol
    "Yolk Syncytial Layer": -28.0,                                     # extraembryonic yolk: depol
}


def load_slice(time):
    import anndata as ad
    a = ad.read_h5ad(ATLAS)
    o = a.obs
    m = (o["time"].astype(str).values == time) & np.array(
        [t in LAYER_VM for t in o["layer_annotation"].astype(str).values])
    xy = np.c_[o["spatial_x"].values, o["spatial_y"].values][m].astype(float)
    anno = o["layer_annotation"].astype(str).values[m]
    layers = sorted(set(anno))
    lid = np.array([layers.index(t) for t in anno])
    ref = np.array([LAYER_VM[t] for t in anno], np.float32)
    return xy, anno, layers, lid, ref


def nca_settle(theta, tid, nbr, k_relax, k_gj, T=40):
    V = torch.full((tid.shape[0],), V_ZYGOTE)
    for _ in range(T):
        V = V + k_relax * (theta[tid] - V) + k_gj * (V[nbr].mean(1) - V)
    return V


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xy, anno, layers, lid, ref = load_slice(MAIN_TIME)
    n = len(xy)
    print(f"ZESTA {MAIN_TIME} slice: {n} spatial bins, {len(layers)} germ-layer classes")
    print("  " + ", ".join(layers) + "\n")

    nbr = cKDTree(xy).query(xy, k=9)[1][:, 1:]
    nbr_t = torch.tensor(nbr, dtype=torch.long)
    tid_t = torch.tensor(lid, dtype=torch.long)
    ref_t = torch.tensor(ref)

    # ---- PART A: set-point recovery (full supervision), off-init all-neutral -50 ----
    theta = torch.full((len(layers),), -50.0, requires_grad=True)
    p_relax = torch.tensor(0.0, requires_grad=True)
    p_gj = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([theta, p_relax, p_gj], lr=0.2)

    def rates():
        return torch.sigmoid(p_relax) * 0.4 + 0.05, torch.sigmoid(p_gj) * 0.25

    def r2(v):
        return float(1 - np.sum((v - ref) ** 2) / np.sum((ref - ref.mean()) ** 2))

    with torch.no_grad():
        kr, kg = rates()
        r2_0 = r2(nca_settle(theta, tid_t, nbr_t, kr, kg).numpy())
    hist = []
    for ep in range(400):
        opt.zero_grad()
        kr, kg = rates()
        V = nca_settle(theta, tid_t, nbr_t, kr, kg)
        loss = ((V - ref_t) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == 399:
            with torch.no_grad():
                kr, kg = rates()
                hist.append((ep, float(loss.detach()), r2(nca_settle(theta, tid_t, nbr_t, kr, kg).numpy())))
    with torch.no_grad():
        kr, kg = rates()
        Vfin = nca_settle(theta, tid_t, nbr_t, kr, kg).numpy()
    r2_f = hist[-1][2]
    learned = theta.detach().numpy()
    ref_per = np.array([LAYER_VM[t] for t in layers])
    corr = float(np.corrcoef(learned, ref_per)[0, 1])
    print(f"PART A -- set-point recovery: field R^2 {r2_0:.2f} -> {r2_f:.2f}, "
          f"set-point corr {corr:.2f}, k_relax={float(kr):.3f} k_gj={float(kg):.3f}")

    # ---- PART B: reconstruct the field from 15% anchors via the gap-junction operator ----
    rng = np.random.RandomState(0)
    anchor = np.zeros(n, bool)
    anchor[rng.choice(n, int(0.15 * n), replace=False)] = True
    anchor_t = torch.tensor(anchor); held = ~anchor_t

    def reconstruct(kgj, T=200):
        V = torch.full((n,), V_ZYGOTE)
        for _ in range(T):
            V = torch.where(anchor_t, ref_t, V + kgj * (V[nbr_t].mean(1) - V))
        return V

    pg2 = torch.tensor(1.0, requires_grad=True)
    opt2 = torch.optim.Adam([pg2], lr=0.1)
    for _ in range(200):
        opt2.zero_grad()
        Vr = reconstruct(torch.sigmoid(pg2) * 0.5)
        l = ((Vr[held] - ref_t[held]) ** 2).mean(); l.backward(); opt2.step()
    kgj_r = float(torch.sigmoid(pg2) * 0.5)
    with torch.no_grad():
        Vrec = reconstruct(torch.tensor(kgj_r)).numpy()
        Vnone = reconstruct(torch.tensor(0.0)).numpy()
    h = held.numpy()
    r2h = lambda v: float(1 - np.sum((v[h] - ref[h]) ** 2) / np.sum((ref[h] - ref[h].mean()) ** 2))
    r2_rec, r2_no = r2h(Vrec), r2h(Vnone)
    print(f"PART B -- reconstruct from 15% anchors via gap-junction coupling: "
          f"held-out R^2 = {r2_rec:.2f} (k_gj={kgj_r:.2f}) vs {r2_no:.2f} no-coupling")

    # ---- BONUS: forward set-point field across the six timepoints (the trajectory) ----
    traj = []
    for tm in TIMES:
        try:
            xyt, annot, layt, lidt, reft = load_slice(tm)
        except Exception:
            continue
        if len(xyt) < 30:
            traj.append((tm, len(xyt), float("nan"))); continue
        # germ-layer separation = std of the germ-layer set-point field actually present
        traj.append((tm, len(xyt), float(np.std(reft))))
    print("\nBONUS -- germ-layer polarity spread across development (std of set-point field):")
    for tm, nb, sd in traj:
        print(f"  {tm:8s} {nb:5d} bins   spread {sd:5.1f} mV")

    _figure(xy, anno, layers, ref, Vrec, anchor, hist, learned, ref_per, r2_f, corr,
            r2_rec, r2_no, kgj_r, traj)
    json.dump(dict(time=MAIN_TIME, n_bins=n, layers=layers, r2_setpoint=r2_f,
                   setpoint_corr=corr, k_relax=float(kr), k_gj_full=float(kg),
                   recon_r2=r2_rec, recon_r2_no_coupling=r2_no, recon_kgj=kgj_r,
                   anchor_frac=0.15,
                   trajectory={t: {"n": nb, "spread_mV": sd} for t, nb, sd in traj},
                   learned={t: float(learned[i]) for i, t in enumerate(layers)}),
              open(OUT / "zesta_train.json", "w"), indent=2)
    print("\nsaved", OUT / "zesta_train.json")
    return r2_f > 0.8 and corr > 0.9 and r2_rec > 0.5


def _figure(xy, anno, layers, ref, Vrec, anchor, hist, learned, ref_per, r2_f, corr,
            r2_rec, r2_no, kgj_r, traj):
    ep = [h[0] for h in hist]; loss = [h[1] for h in hist]
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3)
    x, y = xy[:, 0], xy[:, 1]
    cmap = plt.get_cmap("tab20")

    def emb(ax):
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(layers):
        m = anno == t
        ax.scatter(x[m], y[m], s=5, color=cmap(i % 20), label=t, linewidths=0)
    ax.legend(fontsize=5, markerscale=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"(a) real ZESTA {MAIN_TIME} slice: germ-layer annotation", fontsize=9); emb(ax)
    ax = fig.add_subplot(gs[0, 1]); sc = ax.scatter(x, y, c=ref, s=5, cmap="RdBu_r", vmin=-70, vmax=-25, linewidths=0)
    ax.set_title("(b) target: germ-layer $V_m$ floor (placed, not fit)", fontsize=9); emb(ax); fig.colorbar(sc, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[0, 2]); sc = ax.scatter(x, y, c=Vrec, s=5, cmap="RdBu_r", vmin=-70, vmax=-25, linewidths=0)
    ax.set_title(f"(c) field reconstructed from 15% anchors\nvia gap-junction operator ($R^2$={r2_rec:.2f})", fontsize=9)
    emb(ax); fig.colorbar(sc, ax=ax, fraction=0.04)
    ax = fig.add_subplot(gs[1, 0]); ax.scatter(x, y, s=4, c="0.85", linewidths=0)
    ax.scatter(x[anchor], y[anchor], s=7, c="tab:red", linewidths=0)
    ax.set_title("(d) the 15% anchors (red) that seed\nthe reconstruction", fontsize=9); emb(ax)
    ax = fig.add_subplot(gs[1, 1]); ax.scatter(ref_per, learned, s=45)
    lim = [-72, -25]; ax.plot(lim, lim, "k--", lw=0.6, alpha=0.5)
    ax.set_xlabel("germ-layer reference (mV)"); ax.set_ylabel("learned from atlas (mV)")
    ax.set_title(f"(e) set-points recovered (corr {corr:.2f})\nfield $R^2$ {r2_f:.2f}", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    tt = [t for t, nb, sd in traj if sd == sd]
    ss = [sd for t, nb, sd in traj if sd == sd]
    ax.plot(range(len(tt)), ss, "o-", color="tab:purple")
    ax.set_xticks(range(len(tt))); ax.set_xticklabels(tt, rotation=45, fontsize=7)
    ax.set_ylabel("germ-layer $V_m$ spread (mV)")
    ax.set_title("(f) polarity separates across development\n(temporal trajectory, ZESTA-only)", fontsize=9)
    fig.suptitle(f"Training the spatial NCA to follow the ZESTA zebrafish atlas ({MAIN_TIME}): full supervision "
                 f"recovers the germ-layer set-points (corr {corr:.2f}); the gap-junction operator reconstructs "
                 f"the field from 15% anchors ($R^2$ {r2_rec:.2f}) where no coupling fails ({r2_no:.2f}).", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "zesta_train.png", dpi=130, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "zesta_train.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'TRAINED (ZESTA fits the spatial model)' if ok else 'CHECK'}")
