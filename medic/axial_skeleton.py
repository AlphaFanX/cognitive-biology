"""
axial_skeleton.py -- BONE FOR BONE, step 1: the axial skeleton (vertebral column).

Until now the model grows axial CARTILAGE as a tissue CLASS -- a blob. Here every axial-cartilage cell is
given an INDIVIDUAL VERTEBRA IDENTITY, so the column becomes a named series that corresponds one-for-one
with the atlas's enumerated vertebrae:

  * the somitogenesis clock ORDERS the segments along AP (medic.vertebral_column.build_column),
  * the Hox formula NAMES them -- 7 cervical + 13 thoracic + 6 lumbar + 4 sacral = C1..C7, T1..T13, L1..L6,
    S1..S4 -- exactly the atlas roster (menagerie.skeleton enumerates the same names).

That is the restriction R (cells -> named vertebra); the atlas vertebra capsule is the lifting target
P (named vertebra -> cells). This module wires R and reports the correspondence; it also LIFTS the model's
cells into the atlas vertebra spacing so the column can be drawn as the model's own cells occupying the
named skeleton.

Validation (does the model's own axis implement the atlas?): per-region vertebra COUNTS = 7/13/6/4, all 30
vertebrae populated, and the model's vertebra-centroid AP order is monotonic and matches the atlas order.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.axial_skeleton
Out: data/organ_cascade/axial_skeleton.{png,json}
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
from medic.vertebral_column import build_column, FORMULA
from menagerie.targets import reference_genome
from menagerie.skeleton import build_skeleton

REGION_PFX = {"cervical": "C", "thoracic": "T", "lumbar": "L", "sacral": "S"}
REGION_COL = {"cervical": "#38bdf8", "thoracic": "#ef4444", "lumbar": "#f59e0b", "sacral": "#a78bfa"}
N_VERT = sum(n for _, n in FORMULA)


def vertebra_names():
    """The 30 named vertebrae, anterior -> posterior, matching the atlas roster."""
    names = []
    for reg, n in FORMULA:
        names += [f"{REGION_PFX[reg]}{i + 1}" for i in range(n)]
    return names


def region_of(name):
    for reg, pfx in REGION_PFX.items():
        if name.startswith(pfx):
            return reg
    return "?"


def atlas_vertebra_ap():
    """{name: AP position} for the atlas's enumerated vertebrae (menagerie x is the body long axis)."""
    g = reference_genome("human_male")
    verts = [b for b in build_skeleton(g) if "vertebra" in b.homology]
    return {b.name: float(b.a[0]) for b in verts}


def build():
    base, F = build_base()
    zmax = np.abs(base[:, 2]).max()
    cart = (F == FIDX["Cartilage"]) & (np.abs(base[:, 2]) < 0.18 * zmax)     # axial cartilage = the centra
    P = base[cart].copy()
    P[:, 0] = -P[:, 0]                                                       # build_base has head at +x; flip so t=0 = anterior
    t, vert, region = build_column(P, n_vert=N_VERT)
    names = vertebra_names()
    cell_name = np.array([names[v] for v in vert])

    counts = {nm: int((cell_name == nm).sum()) for nm in names}
    populated = sum(1 for nm in names if counts[nm] > 0)
    region_counts = {reg: sum(counts[nm] for nm in names if region_of(nm) == reg) for reg, _ in FORMULA}
    region_verts = {reg: sum(1 for nm in names if region_of(nm) == reg and counts[nm] > 0) for reg, _ in FORMULA}
    model_ap = {nm: float(t[cell_name == nm].mean()) for nm in names if counts[nm] > 0}

    # AP-order check vs the atlas: rank-correlate the shared vertebrae's AP positions
    atlas_ap = atlas_vertebra_ap()
    shared = [nm for nm in names if nm in atlas_ap and nm in model_ap]
    if len(shared) > 3:
        ma = np.array([model_ap[nm] for nm in shared])
        aa = -np.array([atlas_ap[nm] for nm in shared])          # atlas x decreases posteriorly -> negate for anterior->posterior
        ap_corr = float(np.corrcoef(np.argsort(np.argsort(ma)), np.argsort(np.argsort(aa)))[0, 1])
    else:
        ap_corr = float("nan")
    # monotonic AP? (centroids should increase anterior->posterior along the named order)
    seq = [model_ap[nm] for nm in names if nm in model_ap]
    monotonic = float(np.mean(np.diff(seq) > 0)) if len(seq) > 1 else 0.0

    return dict(P=P, t=t, cell_name=cell_name, region=region, names=names, counts=counts,
                populated=populated, region_counts=region_counts, region_verts=region_verts,
                ap_corr=round(ap_corr, 3), monotonic=round(monotonic, 3), model_ap=model_ap,
                atlas_ap=atlas_ap, shared=shared)


def _figure(r):
    P, names, cell_name = r["P"], r["names"], r["cell_name"]
    name_i = {nm: i for i, nm in enumerate(names)}
    vi = np.array([name_i[c] for c in cell_name])
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    # A: by region
    ax[0].scatter(P[:, 0], P[:, 1], s=14, c=[REGION_COL[region_of(c)] for c in cell_name])
    ax[0].set_title("regionalised (Hox): 7C / 13T / 6L / 4S", color="#a5f3c0", fontsize=10)
    # B: by individual vertebra (rainbow), label the first of each region
    ax[1].scatter(P[:, 0], P[:, 1], s=14, c=vi, cmap="turbo")
    for nm in ("C1", "T1", "L1", "S1", names[-1]):
        if nm in r["model_ap"]:
            m = cell_name == nm
            ax[1].annotate(nm, (P[m, 0].mean(), P[m, 1].max()), color="#e2e8f0", fontsize=8, ha="center")
    ax[1].set_title(f"per-vertebra identity ({r['populated']}/{len(names)} named)\nC1..C7 T1..T13 L1..L6 S1..S4",
                    color="#7dd3fc", fontsize=10)
    # C: the atlas roster (target), same region colours, by AP
    aap = r["atlas_ap"]
    xs = [-aap[nm] for nm in names if nm in aap]
    cs = [REGION_COL[region_of(nm)] for nm in names if nm in aap]
    ax[2].scatter(xs, [0] * len(xs), s=60, c=cs)
    ax[2].set_title("atlas enumerated vertebrae (target roster)", color="#cbd5e1", fontsize=10)
    fig.suptitle("Bone for bone -- the axial skeleton: the model's own cartilage cells given individual "
                 "named-vertebra identity (C1..S4), matched to the atlas roster", color="#e2e8f0", fontsize=11)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/axial_skeleton.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/axial_skeleton.png")


def main():
    r = build()
    print(f"axial cartilage cells: {len(r['P'])}  ->  {r['populated']}/{len(r['names'])} vertebrae named")
    print(f"region cell counts : {r['region_counts']}")
    print(f"region vertebrae   : {r['region_verts']}  (target 7C/13T/6L/4S)")
    print(f"AP order vs atlas  : rank-corr {r['ap_corr']} | monotonic {r['monotonic']}")
    empty = [nm for nm in r["names"] if r["counts"][nm] == 0]
    if empty:
        print(f"UNPOPULATED vertebrae ({len(empty)}): {', '.join(empty)}")
    _figure(r)
    out = dict(populated=r["populated"], n_vertebrae=len(r["names"]),
               region_counts=r["region_counts"], region_verts=r["region_verts"],
               ap_corr=r["ap_corr"], monotonic=r["monotonic"], counts=r["counts"])
    json.dump(out, open("data/organ_cascade/axial_skeleton.json", "w"), indent=1)
    print("saved data/organ_cascade/axial_skeleton.json")


if __name__ == "__main__":
    main()
