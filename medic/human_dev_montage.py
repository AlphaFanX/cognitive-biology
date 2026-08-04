"""
human_dev_montage.py -- the developmental sequence from OUR movie as a print montage, built from the ACTUAL
viewer stills (data/organ_cascade/hdev_*.png, captured by medic._shot_human_views-style Playwright shots of the
model movie at six stages): the curled embryo, fetus, newborn, child, standing adult, and the internal-anatomy
reveal. The viewer's own render -- the same picture Miles watches -- not a matplotlib scatter.

Prereq: serve data on 8903 + capture hdev_{embryo,fetus,newborn,child,adult,anatomy}.png from the viewer.
Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.human_dev_montage
Out: data/organ_cascade/human_dev_montage.png
"""
from __future__ import annotations
import os
import numpy as np

from medic.human_adult_annotated import _crop

STAGES = [("embryo", "hdev_embryo.png"), ("fetus", "hdev_fetus.png"),
          ("newborn", "hdev_newborn.png"), ("child", "hdev_child.png"),
          ("adult", "hdev_adult.png"), ("anatomy", "hdev_anatomy.png")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ims = [(lab, _crop(f"data/organ_cascade/{fn}", pad=0.06)) for lab, fn in STAGES]
    fig, ax = plt.subplots(1, len(ims), figsize=(2.05 * len(ims), 4.8), facecolor="#0d1017")
    for a, (lab, im) in zip(ax, ims):
        a.set_facecolor("#0d1017"); a.axis("off"); a.imshow(im)
        a.set_title(lab, color="#e2e8f0", fontsize=11)
    fig.suptitle("The model's own developmental movie: one cell cloud grown from the curled embryo to the standing "
                 "adult, then its internal anatomy — the viewer's own render of the movie, no template mesh",
                 color="#cbd5e1", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/human_dev_montage.png", dpi=150, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/human_dev_montage.png")


if __name__ == "__main__":
    main()
