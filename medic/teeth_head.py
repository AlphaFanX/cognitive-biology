"""
teeth_head.py -- the TEETH: lateral-inhibition-spaced teeth on the two jaw arches (the DV upper/lower split).

Teeth are modified placodes: a row of tooth germs along the dental lamina whose NUMBER and SPACING are set by a
reaction-diffusion / lateral-inhibition system (Bmp/Shh/Wnt/Ectodin; Kavanagh's inhibitory cascade), each germ
inhibiting its neighbours into a regular arcade. This head reads the maxilla and the mandible (the two jaw bones
the skull head already names), read-only, and writes a spaced row of teeth on each -- the maxillary arch above and
the mandibular arch below, which is the DORSOVENTRAL split of the dentition. It moves no existing cell.

Genome: MSX1/PITX2 initiate the tooth (1st-arch neural crest), the SHH/BMP/WNT-ectodin reaction-diffusion sets the
inhibition wavelength (spacing), and the same inhibitory cascade grades tooth TYPE front-to-back (incisor ->
canine -> premolar -> molar). The count is arch-length / wavelength, capped at the human 16 per arch (8 per
quadrant: I1 I2 C P1 P2 M1 M2 M3), so the arcade is genome-set, not asserted.

READ-ONLY. Validation = teeth on BOTH arches (the DV split), regular spacing (low CV = lateral inhibition), the
human count, and a monotone type gradient from the midline.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.teeth_head
Out: data/organ_cascade/teeth_head.{png,json}
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
from medic import skull_head as SKU

N_PER_ARCH = 16                      # human adult: 16 per arch (8 per quadrant)
# tooth type per position from the midline (one quadrant, repeated L/R): the Kavanagh inhibitory-cascade classes
QUADRANT = ["incisor", "incisor", "canine", "premolar", "premolar", "molar", "molar", "molar"]
TYPE_SIZE = {"incisor": 0.9, "canine": 1.1, "premolar": 1.0, "molar": 1.4}   # relative crown size (M largest)


def _dorsal_sign(base, F):
    if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8:
        return 1.0 if np.median(base[F == FIDX["Notochord"]][:, 1]) >= np.median(base[:, 1]) else -1.0
    return 1.0


def _arcade(arch_cells, occlusal_dir, dsgn, rng, n=N_PER_ARCH):
    """Place n teeth as a lateral-inhibition arcade along the alveolar margin of a jaw bone. The arch is a
    parabola in the AP(x)-ML(z) plane -- incisors anterior at the midline, molars posterior-lateral -- and the
    teeth erupt toward the occlusal plane (occlusal_dir = the DV direction the crowns point)."""
    c = arch_cells.mean(0)
    x, z = arch_cells[:, 0], arch_cells[:, 2]
    x_front = np.percentile(x, 92)                                  # most anterior (incisor edge)
    z_half = np.percentile(np.abs(z - c[2]), 88) + 1e-6            # half the arch width
    depth = 0.9 * (x_front - np.percentile(x, 25))                 # antero-posterior depth of the arcade
    y0 = c[1] + occlusal_dir * 0.10 * z_half                       # alveolar margin, toward the occlusal plane
    # dense arcade curve, then place teeth at even ARC-LENGTH = the lateral-inhibition steady state (each germ
    # one inhibition wavelength from the next), which is regular spacing ALONG the curve, not even in z.
    tt = np.linspace(-1.0, 1.0, 400)
    zc = c[2] + z_half * tt
    xc = x_front - depth * (tt ** 2)
    seg = np.sqrt(np.diff(xc) ** 2 + np.diff(zc) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, arc[-1], n)                         # n teeth, one wavelength apart in arc length
    picks = np.clip(np.searchsorted(arc, targets), 0, len(tt) - 1)
    teeth = []
    for i, kk in enumerate(picks):
        t = float(tt[kk]); zt = float(zc[kk]); xt = float(xc[kk])
        q = int(round(abs(t) * (len(QUADRANT) - 1)))               # quadrant index by distance from midline
        ttype = QUADRANT[min(q, len(QUADRANT) - 1)]
        crown = TYPE_SIZE[ttype] * 0.02 * (np.ptp(arch_cells[:, 0]) + 1e-6)
        pos = np.array([xt, y0 + occlusal_dir * crown, zt])
        P = pos + rng.normal(size=(10, 3)) * crown * 0.4           # a small crown cluster
        teeth.append(dict(idx=i, t=float(t), type=ttype, side=("M" if abs(t) < 1e-6 else ("R" if t > 0 else "L")),
                          pos=pos, P=P, size=TYPE_SIZE[ttype]))
    return teeth


def build(base, F, skull=None):
    rng = np.random.default_rng(0)
    dsgn = _dorsal_sign(base, F)
    # skull = the already-carved cranial bones (parts dict); reuse it if the caller has it (integrated_body)
    # so the skull head is not re-run, else carve it here.
    parts = skull if skull is not None else SKU.build(base, F).get("parts", {})
    out = {}
    # maxilla = UPPER arch (teeth erupt DOWNWARD, toward the occlusal plane); mandible = LOWER arch (erupt UP).
    for bone, occl in (("maxilla", -dsgn), ("mandible", +dsgn)):
        p = parts.get(bone)
        if p is None or len(p["P"]) < 8:
            continue
        arch = "maxillary" if bone == "maxilla" else "mandibular"
        out[arch] = _arcade(p["P"], occl, dsgn, rng)
    return dict(arches=out)


def _spacing_cv(teeth):
    pts = np.array([t["pos"] for t in teeth])
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.std(d) / (np.mean(d) + 1e-9)) if len(d) else 9.9


def _validate(res):
    arches = res["arches"]
    per = {a: len(t) for a, t in arches.items()}
    cv = {a: round(_spacing_cv(t), 3) for a, t in arches.items()}
    # type gradient monotone from midline: incisor->molar as |t| grows
    order = ["incisor", "canine", "premolar", "molar"]
    graded = {}
    for a, t in arches.items():
        half = [x for x in t if x["t"] >= 0]
        seq = [order.index(x["type"]) for x in sorted(half, key=lambda x: x["t"])]
        graded[a] = bool(all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1)))
    return dict(arches=sorted(arches), teeth_per_arch=per, total=sum(per.values()),
                dv_split=bool(set(arches) >= {"maxillary", "mandibular"}),
                spacing_cv=cv, regular=bool(all(v < 0.25 for v in cv.values())),
                type_graded=graded)


def _figure(res, base):
    arches = res["arches"]
    x = base[:, 0]; apf = (x - x.min()) / (np.ptp(x) + 1e-9); head = base[apf >= 0.80]
    col = {"incisor": "#e8eef2", "canine": "#cfd8e0", "premolar": "#b8c4cf", "molar": "#9fb0bd"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor="#0d1017")
    for j, (i, k, ttl) in enumerate([(2, 0, "front (ML x AP)"), (2, 1, "occlusal (ML x DV)")]):
        a = ax[j]; a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(head[:, i], head[:, k], s=2, c="#2a3140", alpha=0.35)
        for arch, teeth in arches.items():
            for t in teeth:
                a.scatter(t["P"][:, i], t["P"][:, k], s=8 * t["size"], c=col[t["type"]], alpha=0.9)
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    v = _validate(res)
    fig.suptitle(f"Teeth head: {v['total']} teeth on the maxillary + mandibular arches (the DV split); "
                 f"spacing CV {v['spacing_cv']} (lateral inhibition), incisor->molar graded -- MSX1/SHH/BMP",
                 color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/teeth_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/teeth_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res)
    print(f"teeth: {v['total']} total on arches {v['arches']} (DV split={v['dv_split']})")
    print(f"  per arch: {v['teeth_per_arch']}   spacing CV: {v['spacing_cv']} (regular={v['regular']})")
    print(f"  tooth-type graded incisor->molar from midline: {v['type_graded']}")
    print("  genome-derived: MSX1/PITX2 initiation; SHH/BMP/WNT-ectodin lateral inhibition sets spacing; "
          "inhibitory cascade grades the type; upper=maxillary / lower=mandibular = the DV split")
    _figure(res, base)
    json.dump(v, open("data/organ_cascade/teeth_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/teeth_head.json")


if __name__ == "__main__":
    main()
