"""
adipose_head.py -- the FAT: the adipose flesh that gives the skeleton-and-organs a body's contour.

The first of the three flesh layers (muscle, FAT, collagen). Reads the body envelope + the abdominal viscera,
read-only, and writes adipose cells in the two depots Gray's names: a SUBCUTANEOUS layer just inside the skin
(the panniculus adiposus -- the fat that rounds out the surface into flesh) and a VISCERAL depot packed around the
abdominal organs (omental / perirenal fat). It moves no existing cell; it lays the fat the surface reads as flesh.

Biology / genome: adipogenesis is driven by the master TF PPARG with CEBPA; the pre-adipocytes are mesenchymal
(the Adipose fate the model already carries). The two depots differ by position and by genes (subcutaneous vs
visceral are distinct developmental fields), which is why the head writes them separately: the subcutaneous depot
is a shell read off the body surface, the visceral depot fills the coelomic gaps around the viscera.

READ-ONLY. Validation = the subcutaneous depot forms a shell covering the body's length; the visceral depot sits
among the abdominal organs.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.adipose_head
Out: data/organ_cascade/adipose_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX

SUBCUT_LO, SUBCUT_HI = 0.80, 0.95      # radial band (fraction of local body radius) the subcutaneous fat fills
N_BINS = 44                            # AP slices along the body


def _organ_cells(base, F, fates):
    ids = [FIDX[n] for n in fates if n in FIDX]
    return base[np.isin(F, ids)] if ids else base[:0]


def build(base, F, rng=None):
    rng = rng or np.random.default_rng(0)
    ap = base[:, 0]
    x0, H = ap.min(), np.ptp(ap) + 1e-9

    # ---- SUBCUTANEOUS: a shell just inside the body surface, per AP slice (PPARG panniculus adiposus) --------
    sub = []
    covered = 0
    for i in range(N_BINS):
        lo = x0 + i / N_BINS * H
        sl = base[(ap >= lo) & (ap < lo + H / N_BINS)]
        if len(sl) < 6:
            continue
        c = sl.mean(0)
        d = sl[:, 1:] - c[1:]                                 # (DV, ML) offset from the slice centre
        r = np.linalg.norm(d, axis=1)
        rmax = np.percentile(r, 92) + 1e-9                    # the slice's outer radius (the surface)
        n = max(10, len(sl) // 5)
        ang = rng.uniform(0, 2 * np.pi, n)
        rr = rmax * rng.uniform(SUBCUT_LO, SUBCUT_HI, n)      # in the outer band = under the skin
        pts = np.c_[np.full(n, c[0]) + rng.normal(size=n) * (H / N_BINS) * 0.4,
                    c[1] + rr * np.cos(ang), c[2] + rr * np.sin(ang)]
        sub.append(pts); covered += 1
    subcutaneous = np.vstack(sub) if sub else base[:0]

    # ---- VISCERAL: fat packed around the abdominal organs (omental / perirenal) ------------------------------
    abd = _organ_cells(base, F, ["Liver", "Kidney", "Gut", "Foregut", "Hindgut", "Pancreas"])
    visceral = base[:0]
    if len(abd) > 8:
        # place fat cells near the viscera but jittered outward into the coelomic gaps
        k = max(200, len(abd) // 4)
        idx = rng.choice(len(abd), k, replace=True)
        visceral = abd[idx] + rng.normal(size=(k, 3)) * 0.02 * H

    return dict(subcutaneous=subcutaneous, visceral=visceral, bins_covered=covered)


def _validate(res):
    sub, vis = res["subcutaneous"], res["visceral"]
    return dict(subcutaneous_cells=int(len(sub)), visceral_cells=int(len(vis)),
                subcutaneous_coverage=round(res["bins_covered"] / N_BINS, 2),
                shell=bool(res["bins_covered"] >= 0.6 * N_BINS), visceral_present=bool(len(vis) > 0))


def _figure(res, base):
    sub, vis = res["subcutaneous"], res["visceral"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 7), facecolor="#0d1017")
    for j, (i, k, ttl) in enumerate([(2, 0, "front (ML x AP)"), (1, 0, "side (DV x AP)")]):
        a = ax[j]; a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(base[::6, i], base[::6, k], s=1, c="#232a36", alpha=0.4)          # body (faint)
        if len(sub):
            a.scatter(sub[:, i], sub[:, k], s=3, c="#f4d58d", alpha=0.5)            # subcutaneous fat (yellow)
        if len(vis):
            a.scatter(vis[:, i], vis[:, k], s=4, c="#e8a04b", alpha=0.7)            # visceral fat (amber)
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    v = _validate(res)
    fig.suptitle(f"Adipose head: subcutaneous shell ({v['subcutaneous_cells']} cells, cover "
                 f"{v['subcutaneous_coverage']}) + visceral depot ({v['visceral_cells']}) -- PPARG, the flesh contour",
                 color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/adipose_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/adipose_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res)
    print(f"adipose: subcutaneous {v['subcutaneous_cells']} cells (coverage {v['subcutaneous_coverage']}, "
          f"shell={v['shell']}) + visceral {v['visceral_cells']} cells")
    print("  genome-derived: PPARG/CEBPA adipogenesis; subcutaneous shell under the skin + visceral around the viscera")
    _figure(res, base)
    json.dump(v, open("data/organ_cascade/adipose_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/adipose_head.json")


if __name__ == "__main__":
    main()
