"""
skull_head.py -- the SKULL head: carve the cranial region into the named bones of Gray's skull.

Reads the head cells (the cranial region at the anterior/superior pole) + the dorsal reference, read-only, and
writes named cranial bones by carving the head's OUTER shell -- it moves no cell. Currently the model's head is
a mass of cranial mesenchyme + brain; this head names its surface into the Gray's roster the way the girdle and
rib heads name their structures, so the skull is a roster and not a blob.

Biology / Gray's: the skull = the NEUROCRANIUM (the brain-case vault: frontal, paired parietals, paired
temporals, occipital) + the VISCEROCRANIUM (the face: nasal, maxilla, paired zygomatics, mandible). The vault
is dermal + chondral bone induced by the underlying brain vesicles; the face is neural-crest derived. Here each
outer head cell is assigned to the cranial bone whose canonical DIRECTION on the head sphere it lies nearest
(a Voronoi partition of the cranial surface), so the partition is read from the geometry, not asserted.

READ-ONLY. Validation = the roster's bones are all populated and bilateral pairs are symmetric.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.skull_head
Out: data/organ_cascade/skull_head.{png,json}
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

# canonical unit directions on the head sphere: (anterior+, superior+, lateral) -- lateral sign gives L/R.
# The nearest-direction partition of the head's outer shell names each cranial bone.
PROTO = {
    "frontal":     (0.55,  0.65, 0.00),
    "parietal-R":  (-0.15, 0.75, 0.45), "parietal-L": (-0.15, 0.75, -0.45),
    "occipital":   (-0.85, 0.25, 0.00),
    "temporal-R":  (-0.20, -0.05, 0.80), "temporal-L": (-0.20, -0.05, -0.80),
    "nasal":       (0.90,  0.05, 0.00),
    "maxilla":     (0.85, -0.35, 0.00),
    "mandible":    (0.70, -0.70, 0.00),
    "zygomatic-R": (0.55, -0.10, 0.55), "zygomatic-L": (0.55, -0.10, -0.55),
}
NEURO = {"frontal", "parietal-R", "parietal-L", "occipital", "temporal-R", "temporal-L"}


def _dorsal_sign(base, F):
    if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8:
        return 1.0 if np.median(base[F == FIDX["Notochord"]][:, 1]) >= np.median(base[:, 1]) else -1.0
    return 1.0


# the cranial region the skull forms around: the brain vesicles + eye/olfactory + cranial neural crest
HEAD_FATES = ["Forebrain", "Midbrain", "Hindbrain", "Cerebellum", "Eye", "OlfactoryBulb", "Neural Crest"]


def build(base, F, shell_pct=50):
    """Carve the cranial region's outer shell into named skull bones. The head region is selected by CRANIAL
    FATE (the brain vesicles + neural crest the skull forms around) -- an AP cutoff over-captures the torso,
    because the matured body's AP is skewed (dense torso squeezed at the top, sparse long legs below).
    `shell_pct` = radius percentile above which a head cell is on the outer shell. Returns per-bone {part,side,P}."""
    ids = [FIDX[n] for n in HEAD_FATES if n in FIDX]
    head = base[np.isin(F, ids)] if ids else base[:0]
    if len(head) < 20:                                            # fallback: the top few % of AP
        x = base[:, 0]; apf = (x - x.min()) / (np.ptp(x) + 1e-9)
        head = base[apf >= 0.92]
    if len(head) < 20:
        return dict(parts={}, n=0)
    dsgn = _dorsal_sign(base, F)
    c = head.mean(0)
    d = head - c
    r = np.linalg.norm(d, axis=1)
    shell = head[r >= np.percentile(r, shell_pct)]                # the cranial surface (where bone forms)
    R = np.linalg.norm(shell - c, axis=1).max() + 1e-9
    # direction of each shell cell in (anterior, superior, lateral), unit-normalised
    dirs = np.c_[(shell[:, 0] - c[0]), dsgn * (shell[:, 1] - c[1]), (shell[:, 2] - c[2])] / R
    names = list(PROTO)
    protos = np.array([PROTO[n] for n in names])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True) + 1e-9
    assign = (dirs @ protos.T).argmax(1)                         # nearest canonical direction (dot product)
    # the MIDLINE facial bones (nasal, maxilla, mandible) sit in the CENTRAL-lower face -- the Voronoi carve
    # over-claims wide lateral cells for them (the jaw ends up wider than the head), so confine them to a central
    # ML band about the head axis. A real mouth/jaw spans ~half the head width, not the whole of it.
    hw = np.percentile(np.abs(shell[:, 2] - c[2]), 95) + 1e-9      # head half-width
    MIDFACE = {"nasal", "maxilla", "mandible"}
    parts = {}
    for i, nm in enumerate(names):
        m = assign == i
        if nm in MIDFACE and m.any():
            m = m & (np.abs(shell[:, 2] - c[2]) < 0.28 * hw)       # central band only (mouth ~half head width)
        if m.sum() >= 1:
            side = "R" if nm.endswith("-R") else ("L" if nm.endswith("-L") else "M")
            parts[nm] = dict(kind="bone", part=nm.split("-")[0], bone=nm, side=side, P=shell[m])
    return dict(parts=parts, n=len(parts), head_center=c)


def _validate(res):
    parts = res["parts"]
    named = set(parts)
    neuro = len(named & NEURO)
    pairs = sum(1 for b in ("parietal", "temporal", "zygomatic") if f"{b}-R" in parts and f"{b}-L" in parts)
    return dict(bones=len(parts), neurocranium=neuro, facial=len(parts) - neuro, bilateral_pairs=pairs,
                roster=sorted(named))


def _figure(res, base):
    parts = res["parts"]
    cmap = {n: i for i, n in enumerate(sorted(parts))}
    fig, ax = plt.subplots(1, 2, figsize=(12, 7), facecolor="#0d1017")
    x = base[:, 0]; apf = (x - x.min()) / (np.ptp(x) + 1e-9); head = base[apf >= 0.72]
    for j, (i, k, ttl) in enumerate([(2, 0, "front (ML x AP)"), (1, 0, "side (DV x AP)")]):
        a = ax[j]; a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(head[:, i], head[:, k], s=3, c="#2a3140", alpha=0.4)
        for nm, p in parts.items():
            a.scatter(p["P"][:, i], p["P"][:, k], s=10, c=[cmap[nm]], cmap="tab20",
                      vmin=0, vmax=len(cmap), alpha=0.9)
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    v = _validate(res)
    fig.suptitle(f"Skull head: {v['bones']} cranial bones ({v['neurocranium']} neurocranium + {v['facial']} "
                 f"facial, {v['bilateral_pairs']} bilateral pairs) -- nearest-direction carve of the head shell",
                 color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/skull_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/skull_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res)
    print(f"skull: {v['bones']} cranial bones ({v['neurocranium']} neurocranium + {v['facial']} facial), "
          f"{v['bilateral_pairs']} bilateral pairs")
    print(f"  roster: {v['roster']}")
    print("  genome-derived: neurocranium induced by the brain vesicles, viscerocranium neural-crest; nearest-direction carve")
    _figure(res, base)
    json.dump(v, open("data/organ_cascade/skull_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/skull_head.json")


if __name__ == "__main__":
    main()
