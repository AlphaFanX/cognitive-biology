"""
Paper figure for the unified embryo: all four heads in one forward pass, and the fascia.
=========================================================================================
Renders unified_embryo.png -- the capstone figure for Paper #6. Top row: stages of the one
embryo (growth + clock-differentiation + convergent extension + neural fold). Bottom row: the
two mechanical coupling systems -- cadherin sorting into tissue compartments, and the integrin/
ECM fascia that binds the sorted tissues into one connected body (versus fragmentation without).

Run: python -m medic.unified_embryo_figure
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.unified_embryo import simulate, FATES

# a discrete colour per fate group
GROUP = {"Forebrain": "neural", "Eye": "neural", "Nervous System": "neural", "Spinal Cord": "neural",
         "Neural Crest": "crest", "Mesoderm": "mesoderm", "Somite": "mesoderm",
         "Epidermal": "epidermis", "Hypoblast": "endoderm", "Yolk Syncytial Layer": "yolk"}
GCOL = {"neural": "#2166ac", "crest": "#762a83", "mesoderm": "#f4a582", "epidermis": "#4daf4a",
        "endoderm": "#d6604d", "yolk": "#b2182b", "other": "#cccccc"}


def components(P, r=0.05):
    pairs = cKDTree(P).query_pairs(r, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    n, lab = connected_components(g, directed=False)
    return n, lab


def main():
    print("simulating with ECM ...")
    frames, m_ecm = simulate(use_ecm=True)
    print("simulating without ECM ...")
    _, m_no = simulate(use_ecm=False)

    fig = plt.figure(figsize=(19, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.2)

    # ---- row 1: stages of the one embryo (side view AP-DV, colour = V_m) ----
    stg = [6, 20, 34, 49]
    for k, s in enumerate(stg):
        born, t, prc2, P, V = frames[s]
        c = frames[-1][3].mean(0)
        Q = (P - c)
        ax = fig.add_subplot(gs[0, k])
        ax.scatter(Q[:, 0], Q[:, 1], c=V, cmap="RdBu_r", vmin=-70, vmax=-25, s=3, linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.7, 0.7)
        ax.set_title(f"{t:.0f} hpf · N={born} · PRC2 {prc2:.2f}", fontsize=9)
        if k == 0:
            ax.set_ylabel("one embryo, all 4 heads\n(colour = $V_m$)", fontsize=9)

    # ---- row 2a: final, coloured by FATE (cadherin sorting into compartments) ----
    P = m_ecm["pos"] - m_ecm["pos"].mean(0); fid = m_ecm["fid"]
    ax = fig.add_subplot(gs[1, 0])
    cols = np.array([GCOL.get(GROUP.get(FATES[j], "other"), "#cccccc") if j >= 0 else "#cccccc" for j in fid])
    ax.scatter(P[:, 0], P[:, 1], c=cols, s=3, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.7, 0.7)
    ax.set_title("(a) cadherins sort tissues\n(colour = fate)", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=GCOL[g], label=g) for g in ["neural", "crest", "mesoderm", "epidermis", "endoderm", "yolk"]]
    ax.legend(handles=handles, fontsize=6, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.22))

    # ---- row 2b: WITHOUT ECM, coloured by connected component (fragments) ----
    Pn = m_no["pos"] - m_no["pos"].mean(0)
    ncn, labn = components(Pn)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(Pn[:, 0], Pn[:, 1], c=labn, cmap="tab10", s=3, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.7, 0.7)
    ax.set_title(f"(b) cadherins alone: FRAGMENTS\n{m_no['ncomp']} components (colour = piece)", fontsize=9)

    # ---- row 2c: WITH ECM, coloured by component (one body) ----
    nce, labe = components(P)
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(P[:, 0], P[:, 1], c=labe, cmap="tab10", s=3, linewidths=0, vmin=0, vmax=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-0.7, 0.7)
    ax.set_title(f"(c) + integrin/ECM fascia: ONE BODY\n{m_ecm['ncomp']} component", fontsize=9)

    # ---- row 2d: the connectivity bar ----
    ax = fig.add_subplot(gs[1, 3])
    ax.bar(["cadherins\nonly", "+ ECM\nfascia"], [m_no["ncomp"], m_ecm["ncomp"]],
           color=["0.6", "#2166ac"])
    ax.set_ylabel("connected components (body pieces)")
    ax.set_title("(d) the fascia is the continuum", fontsize=9)
    for i, v in enumerate([m_no["ncomp"], m_ecm["ncomp"]]):
        ax.text(i, v + 0.06, str(v), ha="center", fontsize=11)

    fig.suptitle("The unified embryo: one grow-from-one-cell forward pass with all four heads intercalated per step "
                 "(division, telomere/PRC2 differentiation,\nconvergent extension, neural fold), and its two mechanical "
                 "couplings---cadherin sorting into tissues, and the integrin/ECM fascia binding them into one body.",
                 fontsize=11)
    for out in ("unified_embryo.png", "data/organ_cascade/unified_embryo.png"):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print(f"saved unified_embryo.png  (ECM {m_ecm['ncomp']} comp / het {m_ecm['het']:.2f}; "
          f"no-ECM {m_no['ncomp']} comp / het {m_no['het']:.2f})")


if __name__ == "__main__":
    main()
