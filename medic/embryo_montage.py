#!/usr/bin/env python3
"""
Build the paper montage figure from the unified zebrafish embryo (medic.zebrafish_embryo).

Composes the key developmental stages of the SAME cell field into a single
multi-panel still -- the genome->embryo film as a printable strip -- plus the
emergent cell-count validation panel. No GIF rendered (this is the figure path).

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.embryo_montage
Outputs:
    zebrafish_embryo_montage.png            (6-stage strip)
    zebrafish_cellcount_validation.png      (emergent division law vs real counts)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from . import zebrafish_embryo as ze
except ImportError:  # pragma: no cover
    from medic import zebrafish_embryo as ze


# Stages chosen to span the whole film: cleavage -> epiboly -> CE/axis ->
# segmentation -> pharyngula -> larva (beating heart). hpf, short caption.
STAGES = [
    (3.0,  "blastula"),
    (7.0,  "epiboly"),
    (10.0, "bud (100% epiboly)"),
    (11.0, "axis (convergent extension)"),
    (16.0, "segmentation (somites)"),
    (48.0, "larva (beating heart)"),
]


def make_montage(path: str = "zebrafish_embryo_montage.png", cols: int = 2):
    field, clock, T = ze.build()
    rows = (len(STAGES) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11 * cols * 0.62, 5.4 * rows * 0.62))
    axes = axes.ravel()
    for ax, (hpf, cap) in zip(axes, STAGES):
        ze.render_frame(ax, field, clock, T, hpf)
        # panel caption strip along the bottom of each panel
        ax.text(0.5, -0.02, f"{hpf:.0f} hpf  -  {cap}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8.5, color="#11213a")
    for ax in axes[len(STAGES):]:
        ax.axis("off")
    fig.suptitle("One cell field from the zygote to the larva, validated against the Silic atlas",
                 fontsize=12, weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved montage: {path}  ({rows}x{cols}, {len(STAGES)} stages, her1 T={T:.1f} min)")
    return path


def main():
    print("=" * 72)
    print("ZEBRAFISH EMBRYO MONTAGE  --  paper figure builder")
    print("=" * 72)
    ze.count_validation()           # writes zebrafish_cellcount_validation.png
    print()
    make_montage()


if __name__ == "__main__":
    main()
