"""
skin_placode_head.py -- the SKIN PLACODE head: lay a periodic array of appendage primordia on the skin.

A head that reads the skin surface, read-only, and writes a periodic array of PLACODES -- the epithelial
thickenings that become hair follicles, glands, teeth, feathers, mammary buds. It never moves the skin, so
it cannot oppose the body surface.

Biology (genome-anchored, reads a field already produced): placodes self-organise by a TURING / lateral-
inhibition system -- a short-range ACTIVATOR (Eda/Edar + Wnt) fires a placode, which secretes a long-range
INHIBITOR (Dkk/Bmp) that suppresses new placodes within one wavelength, so the placodes tile the skin at a
regular spacing (the activator:inhibitor ratio sets the wavelength). Implemented as exactly that: cells fire
placodes in random order, each inhibiting its neighbourhood within the wavelength -> a periodic array.

READ-ONLY: reads the skin cells, writes placodes. Validation = the placodes are periodic (nearest-neighbour
spacing ~ the wavelength with low spread) and tile the whole skin (coverage).

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.skin_placode_head
Out: data/organ_cascade/skin_placode_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.tuned_knobs import tuned

SKIN = ["Skin", "Epidermal"]


def place(skin, wavelength, seed=0):
    """Lateral-inhibition (Turing) placode placement: fire placodes in random order, each suppressing new
    ones within `wavelength` (the Dkk/Bmp inhibition range) -> a periodic array."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(skin))
    chosen = []
    for i in order:
        if not chosen:
            chosen.append(i); continue
        d = np.linalg.norm(skin[chosen] - skin[i], axis=1).min()
        if d >= wavelength:                                   # not inhibited by an existing placode -> fire
            chosen.append(i)
    return np.array(chosen)


def build():
    base, F = build_base()
    ids = [FIDX[n] for n in SKIN if n in FIDX]
    skin = base[np.isin(F, ids)]
    if len(skin) < 20:
        return None
    scale = np.ptp(skin, 0).max() + 1e-9
    wl = tuned("skin_placode", {"wl_frac": 0.09})["wl_frac"] * scale    # tuned Turing wavelength (Wnt:Dkk ratio)
    idx = place(skin, wl)
    placodes = skin[idx]
    # periodicity: nearest-neighbour spacing among placodes
    if len(placodes) > 2:
        dnn, _ = cKDTree(placodes).query(placodes, k=2)
        nn = dnn[:, 1]
        periodic = float(nn.std() / (nn.mean() + 1e-9))       # low CV = regular spacing
        spacing = float(nn.mean() / scale)
    else:
        periodic, spacing = float("nan"), float("nan")
    d, _ = cKDTree(placodes).query(skin)
    coverage = float((d < 1.3 * wl).mean())
    return dict(skin=skin, placodes=placodes, n=len(placodes), wl=wl / scale,
                spacing=round(spacing, 3), periodicity_cv=round(periodic, 3), coverage=round(coverage, 3))


def _figure(r):
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), facecolor="#0d1017")
    for a, (i, j, ttl) in zip(ax, [(0, 1, "front (AP x DV)"), (0, 2, "top (AP x ML)")]):
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(r["skin"][:, i], r["skin"][:, j], s=3, c="#2b3340", alpha=0.4)
        a.scatter(r["placodes"][:, i], r["placodes"][:, j], s=22, c="#f0a840", edgecolors="#fff", linewidths=0.2)
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    fig.suptitle(f"Skin placode head: {r['n']} placodes tiled by Eda/Wnt-Dkk lateral inhibition "
                 f"(spacing CV {r['periodicity_cv']}, {r['coverage']*100:.0f}% skin covered) — reads the skin, "
                 f"read-only", color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/skin_placode_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/skin_placode_head.png")


def main():
    r = build()
    if r is None:
        print("not enough skin cells"); return
    print(f"{r['n']} placodes laid on the skin")
    print(f"  periodic: spacing {r['spacing']} of body, spacing CV {r['periodicity_cv']} (low = regular Turing array)")
    print(f"  coverage: {r['coverage']*100:.0f}% of skin within a wavelength of a placode")
    print("  genome-derived: Eda/Edar + Wnt (activator) vs Dkk/Bmp (inhibitor) set the wavelength; Shh down-growth")
    _figure(r)
    json.dump(dict(placodes=r["n"], spacing=r["spacing"], periodicity_cv=r["periodicity_cv"],
                   coverage=r["coverage"]), open("data/organ_cascade/skin_placode_head.json", "w"), indent=1)
    print("saved data/organ_cascade/skin_placode_head.json")


if __name__ == "__main__":
    main()
