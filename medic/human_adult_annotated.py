"""
human_adult_annotated.py -- the model's own ADULT annotated, built from the ACTUAL viewer stills of the movie
(data/organ_cascade/hview_{front,profile}.png, captured by medic._shot_human_views from the model movie at the
standing-adult frame). The viewer's own render of the movie frame, cropped and labelled -- the same picture
Miles watches in the browser, not a matplotlib scatter.

Prereq: serve data on 8903 + run medic._shot_human_views (writes hview_front.png / hview_profile.png).
Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.human_adult_annotated
Out: data/organ_cascade/human_adult_annotated.png
"""
from __future__ import annotations
import os
import numpy as np


def _crop(path, pad=0.05):
    """Load a viewer still and crop to the body (the non-black bounding box) with a little padding."""
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"))
    lum = im.sum(2)
    ys, xs = np.where(lum > 30)
    if not len(ys):
        return im
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    py = int(pad * (y1 - y0)); px = int(pad * (x1 - x0))
    y0 = max(0, y0 - py); y1 = min(im.shape[0], y1 + py); x0 = max(0, x0 - px); x1 = min(im.shape[1], x1 + px)
    return im[y0:y1, x0:x1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    front = _crop("data/organ_cascade/hview_front.png")
    prof = _crop("data/organ_cascade/hview_profile.png")
    fig = plt.figure(figsize=(15, 10), facecolor="#0d1017")
    # front | spacer (room for the front's right-side labels) | profile
    gs = fig.add_gridspec(1, 3, width_ratios=[2.0, 0.85, 1.05], wspace=0.0)
    axF = fig.add_subplot(gs[0]); axS = fig.add_subplot(gs[2])
    for a in (axF, axS):
        a.set_facecolor("#0d1017"); a.axis("off")

    def A(a, name, fx, fy, tx, ty, img_wh, ha):
        w, h = img_wh
        a.annotate(name, xy=(fx * w, fy * h), xytext=(tx * w, ty * h), color="#f0e6da", fontsize=12,
                   ha=ha, va="center", annotation_clip=False,
                   arrowprops=dict(arrowstyle="->", color="#9aa7b4", lw=1.0))

    fh, fw = front.shape[:2]
    axF.imshow(front)
    # (name, anchor fx, fy, label tx, ty, ha). Adult is a Vitruvian T-pose: head top-centre, arms out to the
    # sides (hands at the ends), legs and feet at the bottom. Left labels spill to the left margin; the few
    # right labels (short) land in the spacer column so nothing overlaps the profile panel.
    fL = [("Head / face", 0.50, 0.07, -0.32, 0.07, "right"),
          ("Deltoid", 0.33, 0.30, -0.32, 0.25, "right"),
          ("Hand (five fingers)", 0.05, 0.31, -0.32, 0.45, "right"),
          ("Rectus abdominis (six-pack)", 0.50, 0.55, -0.32, 0.63, "right"),
          ("Foot (five toes)", 0.47, 0.97, -0.32, 0.90, "right"),
          ("Pectoralis", 0.42, 0.34, 1.30, 0.16, "left"),
          ("Heart", 0.52, 0.42, 1.30, 0.36, "left"),
          ("Knee", 0.47, 0.78, 1.30, 0.80, "left")]
    for name, fx, fy, tx, ty, ha in fL:
        A(axF, name, fx, fy, tx, ty, (fw, fh), ha)
    axF.set_title("front — the model's own adult (actual movie frame, viewer render)", color="#cbd5e1", fontsize=12)

    ph, pw = prof.shape[:2]
    axS.imshow(prof)
    # profile: head at top, face/nose points forward (to the RIGHT); back/gluteus to the left, feet at bottom.
    pL = [("Nose", 0.78, 0.07, 1.55, 0.06, "left"),
          ("Spinal curve", 0.30, 0.42, 1.55, 0.38, "left"),
          ("Gluteus maximus", 0.24, 0.55, 1.55, 0.58, "left"),
          ("Calf", 0.42, 0.84, 1.55, 0.84, "left")]
    for name, fx, fy, tx, ty, ha in pL:
        A(axS, name, fx, fy, tx, ty, (pw, ph), ha)
    axS.set_title("profile — nose forward, gluteus back", color="#8fd0b0", fontsize=12)

    fig.suptitle("The model's own adult, from the movie: a recognizable human grown from one cell, "
                 "the anatomy showing through the skin", color="#e8c9a8", fontsize=13)
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/human_adult_annotated.png", dpi=150, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/human_adult_annotated.png")


if __name__ == "__main__":
    main()
