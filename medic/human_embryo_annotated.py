"""
human_embryo_annotated.py -- a large annotated figure of the model's own EMBRYO (the curled tetrapod grown from
one cell), each structure labelled at its position: the brain vesicles, the eye and otic placodes, the heart, the
liver and gut, the somite series, the neural tube, and the four limb buds. This is the earliest frame of the
developmental movie, the phylotypic form the whole compiler folds toward.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.human_embryo_annotated
Out: data/organ_cascade/human_embryo_annotated.png
"""
from __future__ import annotations
import os
import numpy as np

from medic.adult_persistence_audit import build_base
from medic.human_dev_montage import _stage_cloud
from medic.human_movie import _ANAT, FATES
from medic.unified_embryo import FIDX

_COL = {i: _ANAT.get(f, (0.62, 0.65, 0.70)) for i, f in enumerate(FATES)}

# structures to label (fate names grouped), side = which margin the label sits on
GROUPS = [
    ("Forebrain", ["Forebrain"], +1), ("Midbrain", ["Midbrain"], +1), ("Hindbrain", ["Hindbrain"], +1),
    ("Eye", ["Eye", "Retina"], +1), ("Otic vesicle", ["Otic"], +1), ("Cranial mesenchyme", ["HeadMes"], +1),
    ("Neural tube", ["Spinal Cord", "Notochord"], -1), ("Somites", ["Somite"], -1),
    ("Heart", ["Heart", "Ventricle", "Atrium", "Outflow"], +1), ("Liver", ["Liver"], +1),
    ("Gut", ["Gut"], -1), ("Mesonephros", ["Kidney", "Nephron"], -1),
    ("Limb buds", ["Limb Bud"], -1),
]


def _stagger(anchors, lo, hi, gap):
    order = np.argsort(anchors); ys = np.array(anchors, float)[order]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    ys = np.clip(ys - max(0.0, ys[-1] - hi), lo, hi)
    out = np.zeros(len(anchors)); out[order] = ys
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S0, F = build_base()
    Q = _stage_cloud(S0, F, 0.0)                                # the embryo (movie frame 0)
    # side view shows the fetal C-curl: DV (y) horizontal, AP (x) vertical
    px, py = Q[:, 1], Q[:, 0]
    cols = np.array([_COL[int(i)] for i in F])
    H = np.ptp(py) + 1e-9; W = np.ptp(px) + 1e-9
    fig, ax = plt.subplots(figsize=(9.5, 11), facecolor="#0d1017")
    ax.set_facecolor("#0d1017"); ax.set_aspect("equal"); ax.axis("off")
    order = np.argsort(Q[:, 2])
    ax.scatter(px[order], py[order], s=7, c=cols[order], alpha=0.85, linewidths=0)
    # label anchors
    present = []
    for name, fates, side in GROUPS:
        ids = [FIDX[n] for n in fates if n in FIDX]
        m = np.isin(F, ids)
        if m.sum() < 8:
            continue
        present.append((name, float(np.median(px[m])), float(np.median(py[m])), side))
    LM = 0.95 * W
    for side in (+1, -1):
        grp = [g for g in present if g[3] == side]
        ys = _stagger([g[2] for g in grp], py.min(), py.max(), 0.075 * H)
        for (name, ax_, ay, _s), ty in zip(grp, ys):
            xtext = px.max() + 0.55 * W if side > 0 else px.min() - 0.55 * W
            ax.annotate(name, xy=(ax_, ay), xytext=(xtext, ty), color="#e2e8f0", fontsize=11,
                        ha="left" if side > 0 else "right", va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="->", color="#8a97a8", lw=0.8))
    ax.set_xlim(px.min() - 1.5 * W, px.max() + 1.5 * W)
    ax.set_ylim(py.min() - 0.06 * H, py.max() + 0.06 * H)
    ax.set_title("The model's own embryo — one cell cloud folded into the phylotypic tetrapod\n"
                 "(the first frame of the developmental movie; each structure grown, not placed)",
                 color="#cbd5e1", fontsize=12)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/human_embryo_annotated.png", dpi=150, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/human_embryo_annotated.png")


if __name__ == "__main__":
    main()
