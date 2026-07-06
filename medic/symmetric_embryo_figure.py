"""
Hero figure: the bilaterally symmetric unified embryo (with the electric-face frame applied).
=============================================================================================
Renders symmetric_embryo.png from the symmetric unified-embryo movie -- three views of the
final body coloured by V_m: a top view showing the bilateral symmetry about the midline, a
side view showing the dorsal neural tube over the ventral yolk, and an oblique 3D view.

Run: python -m medic.symmetric_embryo_figure
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOVIE = Path("data/movie/zebrafish_unified_frames.json")


def main():
    d = json.load(open(MOVIE))
    f = d["frames"][-1]                                   # final body (~26 hpf)
    P = np.array(f["xyz"]); V = np.array(f["vm"])
    m = P[:, 1] > -500                                    # born cells only
    P, V = P[m], V[m]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]                   # AP, DV, ML
    vk = dict(c=V, cmap="RdBu_r", vmin=-70, vmax=-25, s=4, linewidths=0)

    # Three views stacked vertically so each spans the full width and reads large.
    fig = plt.figure(figsize=(9.5, 13.0))
    # (a) top view: AP vs ML -- bilateral symmetry about the midline
    ax = fig.add_subplot(3, 1, 1)
    ax.scatter(x, z, **vk); ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(a) top view (AP $\\times$ ML): bilaterally symmetric", fontsize=12)

    # (b) side view: AP vs DV -- dorsal neural tube over ventral yolk
    ax = fig.add_subplot(3, 1, 2)
    ax.scatter(x, y, **vk)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("(b) side view (AP $\\times$ DV): dorsal neural, ventral yolk", fontsize=12)

    # (c) oblique 3D view
    ax = fig.add_subplot(3, 1, 3, projection="3d")
    ax.scatter(x, z, y, c=V, cmap="RdBu_r", vmin=-70, vmax=-25, s=3, linewidths=0)
    ax.view_init(elev=22, azim=-60)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect((np.ptp(x), np.ptp(z), np.ptp(y)))
    except Exception:
        pass
    ax.set_title("(c) oblique 3D view", fontsize=10)

    fig.suptitle("The unified embryo with the electric-face frame applied: one grow-from-one-cell forward pass "
                 "(division, PRC2/Hox differentiation, convergent extension, neural fold, cadherin sorting + ECM "
                 "fascia),\nnow a bilaterally symmetric body---blue hyperpolarized neural tube on the dorsal midline, "
                 "depolarized yolk ventral. Colour = $V_m$; " + str(len(P)) + " cells.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for p in ("symmetric_embryo.png", "data/organ_cascade/symmetric_embryo.png"):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved symmetric_embryo.png  (", len(P), "cells )")


if __name__ == "__main__":
    main()
