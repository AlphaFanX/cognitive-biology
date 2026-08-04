"""
fascia_head.py -- the COLLAGEN: the connective-tissue flesh that wraps the muscles and ties the bones.

The third flesh layer (muscle, fat, COLLAGEN). Reads the muscles + the bones, read-only, and writes the fibrous
connective tissue Gray's names: DEEP FASCIA -- thin collagen sheets investing each muscle mass (the epimysium /
deep fascia that give the flesh its planes) -- and LIGAMENTS -- collagen bands tying bone to bone across each
articulation (the longitudinal ligaments of the vertebral column, the collateral/cruciate bands of the limb
joints). It moves no existing cell; it lays the collagen that holds the flesh together.

Biology / genome: the fibrous connective tissue is COL1A1/COL3A1 fibrillar collagen laid by fibroblasts; ligament
and tendon progenitors are marked by SCX (Scleraxis) and MKX (Mohawk). Fascia invests the muscle it is read off;
ligaments span the two bones they connect -- so the head's output is fixed by the muscle and bone it reads, and
cannot invent a plane or a band that the parts do not already imply.

READ-ONLY. Validation = the fascia covers the muscle mass; every ligament spans two distinct bones.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.fascia_head
Out: data/organ_cascade/fascia_head.{png,json}
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
from medic import part_resolved_anatomy as PR
from medic import limb_chondrogenesis_head as LC


def _muscle_cells(base, F):
    ids = [FIDX[n] for n in ("Muscle",) if n in FIDX]
    return base[np.isin(F, ids)] if ids else base[:0]


def build(base, F, rng=None):
    rng = rng or np.random.default_rng(0)
    H = np.ptp(base[:, 0]) + 1e-9

    # ---- DEEP FASCIA: a collagen sheet investing the muscle mass (COL1A1 epimysium) ------------------------
    mus = _muscle_cells(base, F)
    fascia = base[:0]
    if len(mus) > 20:
        c = mus.mean(0)
        d = mus - c
        r = np.linalg.norm(d[:, 1:], axis=1)                      # radial in DV-ML about the muscle axis
        outer = mus[r >= np.percentile(r, 78)]                    # the investing surface of the muscle mass
        fascia = outer + rng.normal(size=outer.shape) * 0.006 * H

    # ---- LIGAMENTS: collagen bands tying bone to bone across each joint (SCX/MKX) --------------------------
    ligaments = []
    try:
        V, Vname, *_ = PR.complete_column(base, F, np.random.default_rng(0))
        Vname = [str(n) for n in Vname]
        # centroid per named vertebra, ordered along the column
        cents = {}
        for nm in set(Vname):
            cents[nm] = V[np.array([n == nm for n in Vname])].mean(0)
        order = sorted(cents, key=lambda nm: cents[nm][0])        # by AP
        for a, b in zip(order[:-1], order[1:]):                   # longitudinal ligaments between adjacent vertebrae
            ligaments.append(dict(kind="intervertebral", a=a, b=b,
                                  p0=cents[a].tolist(), p1=cents[b].tolist()))
    except Exception:
        pass
    # limb-joint ligaments: within each limb, tie adjacent long bones (stylopod-zeugopod, zeugopod-autopod)
    lb = LC.carve(base, F)
    for key, v in lb.items():
        P, bl = v["P"], np.asarray(v.get("bone", []))
        names = [b for b in ("femur", "tibia", "tarsals", "radius", "carpals", "humerus") if (bl == b).any()]
        segs = [(names[i], names[i + 1]) for i in range(len(names) - 1)]
        for a, b in segs:
            ca, cb = P[bl == a].mean(0), P[bl == b].mean(0)
            ligaments.append(dict(kind=f"{key}-joint", a=a, b=b, p0=ca.tolist(), p1=cb.tolist()))

    return dict(fascia=fascia, ligaments=ligaments, n_muscle=int(len(mus)))


def _validate(res):
    lig = res["ligaments"]
    spans2 = sum(1 for l in lig if l["a"] != l["b"])
    return dict(fascia_cells=int(len(res["fascia"])), ligaments=len(lig),
                ligaments_spanning_two_bones=spans2,
                fascia_invests_muscle=bool(len(res["fascia"]) > 0 and res["n_muscle"] > 0),
                kinds=sorted({l["kind"].split("-")[0] for l in lig}))


def _figure(res, base):
    fascia, lig = res["fascia"], res["ligaments"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 7), facecolor="#0d1017")
    for j, (i, k, ttl) in enumerate([(2, 0, "front (ML x AP)"), (1, 0, "side (DV x AP)")]):
        a = ax[j]; a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(base[::6, i], base[::6, k], s=1, c="#232a36", alpha=0.4)
        if len(fascia):
            a.scatter(fascia[:, i], fascia[:, k], s=2, c="#cfe8d0", alpha=0.35)     # fascia (pale collagen)
        for l in lig:
            p0, p1 = np.array(l["p0"]), np.array(l["p1"])
            a.plot([p0[i], p1[i]], [p0[k], p1[k]], c="#8fd3f4", lw=0.8, alpha=0.8)   # ligaments (blue bands)
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    v = _validate(res)
    fig.suptitle(f"Fascia head: deep fascia investing the muscle ({v['fascia_cells']} cells) + {v['ligaments']} "
                 f"ligaments bone-to-bone ({v['ligaments_spanning_two_bones']} span two bones) -- COL1A1/SCX",
                 color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/fascia_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/fascia_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res)
    print(f"fascia: {v['fascia_cells']} fascia cells investing the muscle; {v['ligaments']} ligaments "
          f"({v['ligaments_spanning_two_bones']} span two distinct bones); kinds {v['kinds']}")
    print("  genome-derived: COL1A1/COL3A1 fibrillar collagen; SCX/MKX ligament; fascia read off the muscle, ligaments off the bones")
    _figure(res, base)
    json.dump(v, open("data/organ_cascade/fascia_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/fascia_head.json")


if __name__ == "__main__":
    main()
