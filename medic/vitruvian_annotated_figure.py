"""
vitruvian_annotated_figure.py -- the annotated figure for Paper 7: the model-native adult, with the identifiable
SURFACE anatomy (the muscles that read through the skin) on the left, and the corrected INTERNAL anatomy (every
organ on its Hox-addressed head-to-toe level, from the AP-address head) on the right. This is the figure that
shows the body has become nameable -- deltoid, pectoralis, the rectus six-pack, the gluteus, the nose; heart
above the kidneys, liver in the upper abdomen, bladder in the pelvis.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.vitruvian_annotated_figure
Out: data/organ_cascade/vitruvian_annotated.png
"""
from __future__ import annotations
import os
import numpy as np

from medic.adult_persistence_audit import build_base
from medic.human_movie import (mature_cloud, MATURE_SEARCHED, grow_limbs, _limb_grow_model,
                               _long_axis_len, LIMB)
from medic.flesh_surface_head import flesh_skin
from medic.fine_relief_head import body_relief
from medic.unified_embryo import FIDX


def _adult():
    """The matured, posed, scaled adult cloud (the movie's Q_adult recipe) + its fleshed, muscled skin."""
    b, F = build_base()
    Q = mature_cloud(b, F, 1.0, MATURE_SEARCHED)
    Q = grow_limbs(Q, F == LIMB, _limb_grow_model(1.0, MATURE_SEARCHED["limb_ext"]),
                   _limb_grow_model(1.0, MATURE_SEARCHED.get("leg_ext", MATURE_SEARCHED["limb_ext"])), pose=1.0)
    Q = Q * (3.2 / _long_axis_len(Q))
    sv, _sf = flesh_skin(Q, F)
    sv, rel = body_relief(sv, Q, F, amp=1.0)
    return Q, F, sv, rel


# identifiable surface muscles -> where to point the label (apf, ml-side, "front"/"side")
_MUSCLE_LABELS = [
    ("Deltoid",           0.80, 0.9, "front"),
    ("Pectoralis major",  0.73, 0.4, "front"),
    ("Rectus abdominis\n(six-pack)", 0.55, 0.15, "front"),
    ("Biceps",            0.62, 0.85, "front"),
    ("Nose",              0.93, 0.0, "front"),
    ("Trapezius",         0.79, 0.2, "side"),
    ("Latissimus dorsi",  0.64, 0.5, "side"),
    ("Gluteus maximus",   0.47, 0.4, "side"),
    ("Gastrocnemius",     0.12, 0.3, "side"),
]

# internal organs, head-to-toe, at their registered AP levels (canonical apf)
_ORGAN_LABELS = [
    ("Forebrain", 0.94), ("Eye", 0.93), ("Thymus", 0.78), ("Lung", 0.74), ("Heart", 0.71),
    ("Liver", 0.65), ("Spleen", 0.66), ("Pancreas", 0.63), ("Adrenal", 0.62), ("Kidney", 0.60),
    ("Bladder", 0.48),
]


def _emit():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from medic.human_adult_annotated import _crop
    Q, F, sv, rel = _adult()
    x = Q[:, 0]; xmin = x.min(); H = np.ptp(x) + 1e-9
    sap = (sv[:, 0] - xmin) / H
    # the movie's own adult, front and side (viewer stills), added as the first two panels
    front_img = _crop("data/organ_cascade/hview_front.png", pad=0.05)
    side_img = _crop("data/organ_cascade/hview_profile.png", pad=0.05)
    fig = plt.figure(figsize=(15, 13), facecolor="#0d1017")
    # TOP row: the movie's adult front + side (centred); BOTTOM row: the three analysis panels
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.3], hspace=0.10, wspace=0.06)
    axFront = fig.add_subplot(gs[0, 1:3]); axSide = fig.add_subplot(gs[0, 3:5])
    for a in (axFront, axSide):
        a.set_facecolor("#0d1017"); a.axis("off")
    axFront.imshow(front_img); axFront.set_title("adult — front (movie)", color="#e8c9a8", fontsize=10)
    axSide.imshow(side_img); axSide.set_title("adult — side (movie)", color="#e8c9a8", fontsize=10)
    ax = [fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]
    ylo, yhi = xmin - 0.06 * H, xmin + 1.10 * H
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off"); a.set_ylim(ylo, yhi)

    def _stagger(anchors, lo, hi, gap):
        """Assign each label a non-overlapping y (>= gap apart), as close to its anchor as possible, sorted."""
        order = np.argsort(anchors)
        ys = np.array(anchors, float)[order]
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < gap:
                ys[i] = ys[i - 1] + gap
        shift = max(0.0, ys[-1] - hi)
        ys = np.clip(ys - shift, lo, hi)
        out = np.zeros(len(anchors)); out[order] = ys
        return out

    LM = 0.42 * H                                              # fixed label-margin x (same for every panel)

    def _panel(a, sx, sy, half, labels, side_sign, txt_c="#e2e8f0", arw="#7f8ea3"):
        a.scatter(sx, sy, s=4, c=rel, cmap="inferno", vmin=0, vmax=np.percentile(rel, 99) + 1e-6)
        a.set_xlim(-0.5 * H, 0.5 * H)                          # tight, so the body reads TALL; labels spill to margin
        anch = [xmin + l[1] * H for l in labels]
        ly = _stagger(anch, ylo + 0.05 * H, yhi - 0.05 * H, 0.052 * H)
        for (name, apf, mside, view), ay, ty in zip(labels, anch, ly):
            a.annotate(name, xy=(side_sign * min(half * 0.9, 0.3 * H), ay), xytext=(side_sign * LM, ty),
                       color=txt_c, fontsize=8.5, ha="left" if side_sign > 0 else "right", va="center",
                       annotation_clip=False,
                       arrowprops=dict(arrowstyle="->", color=arw, lw=0.7, connectionstyle="arc3,rad=0.0"))

    xr = float(np.percentile(np.abs(sv[:, 2]), 98)) + 1e-6      # ML half-extent (front)
    xd = float(np.percentile(np.abs(sv[:, 1]), 98)) + 1e-6      # DV half-extent (side)
    _panel(ax[0], sv[:, 2], sv[:, 0], xr, [l for l in _MUSCLE_LABELS if l[3] == "front"], +1)
    ax[0].set_title("surface — front\ndeltoid · pectoralis · six-pack · biceps · nose", color="#e8c9a8", fontsize=9)
    _panel(ax[1], sv[:, 1], sv[:, 0], xd, [l for l in _MUSCLE_LABELS if l[3] == "side"], -1)
    ax[1].set_title("surface — side/back\ntrapezius · latissimus · gluteus · calf", color="#e8c9a8", fontsize=9)
    # ---- panel 2: internal organs, head-to-toe on their registered levels ----
    ax[2].scatter(Q[::4, 2], Q[::4, 0], s=1, c="#233042", alpha=0.5)     # faint body
    ax[2].set_xlim(-0.5 * H, 0.5 * H)
    oanch = []
    for name, apf in _ORGAN_LABELS:
        fid = FIDX.get(name)
        if fid is not None and (F == fid).sum() > 8:
            m = F == fid; ax[2].scatter(Q[m, 2], Q[m, 0], s=6, c="#e07a5a", alpha=0.7)
            oanch.append(float(np.median(Q[m, 0])))
        else:
            oanch.append(xmin + apf * H)
    sides = [1 if i % 2 == 0 else -1 for i in range(len(_ORGAN_LABELS))]
    lyR = _stagger([a for a, s in zip(oanch, sides) if s > 0], ylo + 0.05 * H, yhi - 0.05 * H, 0.05 * H)
    lyL = _stagger([a for a, s in zip(oanch, sides) if s < 0], ylo + 0.05 * H, yhi - 0.05 * H, 0.05 * H)
    iR = iL = 0
    for (name, apf), ay, side in zip(_ORGAN_LABELS, oanch, sides):
        ty = (lyR[iR] if side > 0 else lyL[iL]);  iR += side > 0;  iL += side < 0
        ax[2].annotate(name, xy=(0, ay), xytext=(side * LM, ty), color="#cbd5e1", fontsize=8.5,
                       ha="left" if side > 0 else "right", va="center", annotation_clip=False,
                       arrowprops=dict(arrowstyle="->", color="#5b7a6a", lw=0.6))
    ax[2].set_title("internal — head-to-toe\nevery organ on its Hox-addressed level", color="#8fd0b0", fontsize=9)

    fig.suptitle("The model-native adult: nameable surface muscle + every organ on its correct head-to-toe level "
                 "(the AP-address head; census offset 0.13→0.003, 22 order-inversions→0)",
                 color="#e2e8f0", fontsize=10, y=0.98)
    fig.subplots_adjust(top=0.93, bottom=0.02, left=0.04, right=0.97)
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/vitruvian_annotated.png", dpi=140, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/vitruvian_annotated.png")


if __name__ == "__main__":
    _emit()
