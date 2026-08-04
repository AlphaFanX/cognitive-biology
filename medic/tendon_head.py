"""
tendon_head.py -- the TENDON (Scleraxis) head: connect each muscle to the bones it spans.

The connector head. It composes on THREE prior heads' outputs -- the axial muscles (CT-scaffold), the limb
muscles, and the bones (axial + limb skeleton) -- read-only, and writes a tendon from each muscle's ends to
its nearest bone attachments. It moves nothing, so it cannot oppose the muscle or the skeleton.

Biology (genome-anchored, reads fields already produced): tendons are the Scleraxis (Scx+) SYNDETOME, the
cell layer BETWEEN the myotome (muscle) and the sclerotome (bone), induced by FGF from the muscle tips. A
tendon is defined by exactly the two things it joins -- a muscle end and a bone -- so the head reads {muscle
groups, bones} and draws the Scx+ bridge at each muscle end to the nearest bone (its origin and insertion).

READ-ONLY: reads muscles + bones, writes tendons. Validation = every muscle gets an origin AND an insertion
tendon (2 per muscle), each landing on a real bone, so the muscle-bone loop is closed.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.tendon_head
Out: data/organ_cascade/tendon_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.ct_scaffold_head import carve as carve_axial
from medic.limb_muscle_head import carve as carve_limb
from medic.limb_chondrogenesis_head import carve as carve_bones
from medic.tuned_knobs import tuned


def _muscle_groups(base, F):
    """All muscle groups (axial CT-scaffold domains + limb muscles), each as {name, cells}."""
    groups = []
    ax = carve_axial(base, F)
    if ax is not None:
        for mid in np.unique(ax["muscle_id"]):
            m = ax["muscle_id"] == mid
            if m.sum() >= 4:
                groups.append(dict(name=str(ax["name"][m][0]), P=ax["mus"][m]))
    for lname, r in carve_limb(base, F).items():
        for mu in set(r["muscle"].tolist()):
            m = r["muscle"] == mu
            if m.sum() >= 4:
                groups.append(dict(name=f"{lname} {mu}", P=r["P"][m]))
    return groups


def _bone_points(base, F):
    """Bone anchor cloud: axial cartilage (vertebrae) + the limb-bone cells."""
    zmax = np.abs(base[:, 2]).max()
    axial = base[(F == FIDX["Cartilage"]) & (np.abs(base[:, 2]) < 0.18 * zmax)]
    limb = np.vstack([r["P"] for r in carve_bones(base, F).values()]) if carve_bones(base, F) else base[:0]
    return np.vstack([axial, limb]) if len(limb) else axial


def attach(groups, bones, attach_radius=None):
    """Draw a tendon from each muscle END to its nearest bone, rejecting any attachment longer than
    `attach_radius` (as a fraction of the bone-cloud diagonal) -- a tendon should not span the whole body.
    attach_radius=None -> unbounded (nearest bone always). Returns tendons + attachment-quality metrics."""
    bt = cKDTree(bones)
    diag = float(np.linalg.norm(np.ptp(bones, 0))) + 1e-9
    rmax = attach_radius * diag if attach_radius is not None else np.inf
    tendons, lens = [], []
    both = distinct = 0
    for g in groups:
        P = g["P"]
        C = P - P.mean(0)
        ax = C @ np.linalg.svd(C, full_matrices=False)[2][0]          # muscle long axis
        e0 = P[ax.argmin()]; e1 = P[ax.argmax()]                      # the two muscle ends
        js = []
        for e in (e0, e1):
            d, j = bt.query(e)
            if d <= rmax:
                tendons.append(np.array([e, bones[j]])); lens.append(float(d)); js.append(int(j))
        both += int(len(js) == 2)
        distinct += int(len(js) == 2 and js[0] != js[1])              # origin + insertion on DIFFERENT bones
    return dict(tendons=tendons, both_ends=both, distinct=distinct, lens=lens, diag=diag)


def build(attach_radius=None):
    base, F = build_base()
    groups = _muscle_groups(base, F)
    bones = _bone_points(base, F)
    if attach_radius is None:                                   # None -> use the tuned radius (else unbounded)
        attach_radius = tuned("tendon", {"radius": None})["radius"]
    r = attach(groups, bones, attach_radius)
    return dict(base=base, bones=bones, groups=groups, tendons=r["tendons"],
                n_muscles=len(groups), n_tendons=len(r["tendons"]), both_ends=r["both_ends"],
                distinct=r["distinct"])


def _figure(r):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor="#0d1017")
    ax.set_facecolor("#0d1017"); ax.set_aspect("equal"); ax.axis("off")
    ax.scatter(r["bones"][:, 0], r["bones"][:, 1], s=6, c="#8a8f9a", alpha=0.5, label="bone")
    for g in r["groups"]:
        ax.scatter(g["P"][:, 0], g["P"][:, 1], s=5, c="#b05050", alpha=0.4)
    segs = [t[:, [0, 1]] for t in r["tendons"]]
    ax.add_collection(LineCollection(segs, colors="#f0f0b0", linewidths=0.7, alpha=0.9))
    ax.set_title(f"Tendon (Scx) head: {r['n_tendons']} tendons join {r['n_muscles']} muscles to the bones\n"
                 f"{r['both_ends']}/{r['n_muscles']} muscles get BOTH an origin + insertion — reads muscle+bone, "
                 f"read-only", color="#e2e8f0", fontsize=10)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/tendon_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/tendon_head.png")


def main():
    r = build()
    print(f"{r['n_muscles']} muscles -> {r['n_tendons']} tendons")
    print(f"  {r['both_ends']}/{r['n_muscles']} muscles get BOTH an origin + insertion tendon on a real bone")
    print("  genome-derived: Scleraxis (Scx) syndetome between myotome + sclerotome, induced by muscle-tip FGF")
    _figure(r)
    json.dump(dict(muscles=r["n_muscles"], tendons=r["n_tendons"], both_ends=r["both_ends"]),
              open("data/organ_cascade/tendon_head.json", "w"), indent=1)
    print("saved data/organ_cascade/tendon_head.json")


if __name__ == "__main__":
    main()
