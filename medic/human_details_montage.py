"""
human_details_montage.py -- close-ups from the movie's own adult, cropped straight out of the ACTUAL viewer
still (data/organ_cascade/hview_front.png / hview_profile.png, the model movie's standing-adult frame): a
five-fingered HAND, the muscled TORSO, and a five-toed FOOT. The viewer's own render, not a scatter.

Prereq: medic._shot_human_views (writes hview_front.png / hview_profile.png).
Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.human_details_montage
Out: data/organ_cascade/human_details_montage.png
"""
from __future__ import annotations
import os
import numpy as np

from medic.human_adult_annotated import _crop


def _sub(im, fx0, fx1, fy0, fy1):
    h, w = im.shape[:2]
    return im[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    front = _crop("data/organ_cascade/hview_front.png")
    prof = _crop("data/organ_cascade/hview_profile.png")
    panels = [
        (_sub(front, 0.74, 1.00, 0.20, 0.42), "hand — five fingers"),
        (_sub(front, 0.28, 0.72, 0.22, 0.66), "torso — deltoid · pectoral · six-pack"),
        (_sub(prof, 0.20, 0.90, 0.82, 1.00), "foot — five toes (side)"),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="#0d1017")
    for a, (im, title) in zip(ax, panels):
        a.set_facecolor("#0d1017"); a.axis("off")
        a.imshow(im); a.set_title(title, color="#e8c9a8", fontsize=11)
    fig.suptitle("Close-ups from the model's own adult, cropped from the movie frame: five-fingered hands, the "
                 "muscled torso, five-toed feet", color="#e2e8f0", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/human_details_montage.png", dpi=145, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/human_details_montage.png")


if __name__ == "__main__":
    main()
