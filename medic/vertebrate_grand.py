"""
A single large, grand render of the vertebrate body for Paper #8.

Reuses the Paper-6 growth engine exactly as medic/vertebrate_growth.py does (grow-from-one-cell
division, four heads, cadherin + integrin/ECM cohesion, bilateral symmetry via the electric frame
and lateral inhibition; four cadherin-compact limbs at the Hox-coded fore/hind levels; the derived
organs -- four-chambered heart, hollow gut, head sinuses -- placed at their genome-derived
addresses and built as the cavities they are). Instead of the two-panel figure, it renders ONE
large single-panel 3/4 view of the whole vertebrate, with a translucent cutaway hint of the organs,
at high resolution for a full-page figure.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.vertebrate_grand
Out: data/vertebrate_grand.png
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import simulate
from medic.vertebrate_growth import (
    symmetrize, grow_limb, relax, build_organs,
    N_BODY, N_LIMB, FORE_AP, HIND_AP,
)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    print("growing the Paper-6 body (2x cells) for the grand render ...")
    import medic.unified_embryo as ue
    ue.N_END = 13000
    frames, m = simulate(use_ecm=True, seed=0)
    P, fid = symmetrize(m["pos"], m["fid"])
    if len(P) > N_BODY:
        keep = np.random.default_rng(1).choice(len(P), N_BODY, replace=False)
        P, fid = P[keep], fid[keep]

    rng = np.random.default_rng(3)
    L = np.vstack([grow_limb(P, af, s, N_LIMB, rng)
                   for af in (FORE_AP, HIND_AP) for s in (+1, -1)])
    L, nn = relax(L, iters=30)
    organs = build_organs(P, rng)
    n_org = sum(len(v) for v in organs.values())
    total = len(P) + len(L) + n_org
    print(f"  body {len(P)} + limbs {len(L)} + organs {n_org} = {total} cells")

    # body fate colours (opaque, saturated), limbs brown, organs by cavity
    fu = np.unique(fid[fid >= 0])
    cmap = plt.cm.tab20(np.linspace(0, 1, max(1, len(fu))))
    col = np.tile(np.array([0.72, 0.75, 0.78, 1.0]), (len(P), 1))
    for j, f in enumerate(fu):
        col[fid == f] = cmap[j]
    ocol = {"heart (4 chambers)": "#d62728", "hollow gut": "#ff7f0e", "sinuses": "#17becf"}

    # ---- one grand panel -------------------------------------------------
    plt.rcParams["figure.facecolor"] = "white"
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # whole body, fate-coloured
    ax.scatter(P[:, 0], P[:, 2], P[:, 1], c=col, s=7, alpha=0.85,
               linewidths=0, depthshade=True)
    # limbs, brown, a touch larger so they read as compact buds
    ax.scatter(L[:, 0], L[:, 2], L[:, 1], c="#b5651d", s=9, alpha=0.95,
               linewidths=0, depthshade=True)
    # translucent cutaway hint: the derived organs glow through the body
    for name, pts in organs.items():
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=ocol[name], s=12,
                   alpha=0.6, linewidths=0, depthshade=True)

    ax.view_init(elev=27, azim=-60)
    ax.set_box_aspect((np.ptp(P[:, 0]),
                       np.ptp(np.r_[P[:, 2], L[:, 2]]),
                       np.ptp(P[:, 1])), zoom=1.95)
    ax.set_axis_off()

    # clean anatomical end-labels placed at the data, not on a cube frame
    xa, xp = P[:, 0].min(), P[:, 0].max()
    ymid = float(np.median(P[:, 2]))
    ztop = float(P[:, 1].max())
    zbot = float(P[:, 1].min())
    ax.text(xa - 0.04 * (xp - xa), ymid, float(np.median(P[:, 1])), "anterior",
            fontsize=12, color="0.25", ha="right", va="center", weight="bold")
    ax.text(xp + 0.04 * (xp - xa), ymid, float(np.median(P[:, 1])), "posterior",
            fontsize=12, color="0.25", ha="left", va="center", weight="bold")
    ax.text(float(np.median(P[:, 0])), ymid, ztop + 0.10 * (ztop - zbot), "dorsal",
            fontsize=11, color="0.35", ha="center", va="bottom")

    # legend for the organs and limbs
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", mfc=ocol[n], mec="none", ms=10, label=n)
               for n in organs] + [
        Line2D([0], [0], marker="o", ls="", mfc="#b5651d", mec="none", ms=10, label="limbs (cadherin-compact)")]
    ax.legend(handles=handles, fontsize=11, loc="upper left",
              bbox_to_anchor=(0.02, 0.98), framealpha=0.9)

    fig.suptitle(f"A vertebrate computed from the genome: {total:,} cells grown from one cell",
                 fontsize=18, y=0.94)
    fig.text(0.5, 0.895,
             "four limbs at the Hox-coded levels and the derived organs "
             "(four-chambered heart, hollow gut, head sinuses)",
             ha="center", fontsize=12.5, color="0.3")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.87)
    fig.savefig("data/vertebrate_grand.png", dpi=220)
    plt.close(fig)
    print("saved data/vertebrate_grand.png")


if __name__ == "__main__":
    main()
