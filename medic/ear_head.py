"""ear_head.py -- shape the Otic (ear) primordium into a real EXTERNAL EAR (pinna), flush to the lateral head.

The Otic cells are placed by organ_sprouting as a diffuse dorsal PAIRED blob (no shape, sticks out far), so the
head reads with shapeless ears sticking out. This reshapes each side's Otic cells into a compact, thin, vertically-
oriented oval pinna sitting on the lateral head surface at ear level (roughly the eye AP level, slightly posterior),
with a shallow concha dip -- a recognisable ear that hugs the head instead of a protruding cloud. Read-only on
every non-Otic cell; cosmetic (does not touch the skeleton, so Gray's integrity is unaffected).
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX


def shape_ears(base, F):
    if "Otic" not in FIDX:
        return base
    base = np.asarray(base, float).copy()
    F = np.asarray(F)
    m = np.where(F == FIDX["Otic"])[0]
    if len(m) < 8:
        return base
    # head frame from the brain (robust; arms are excluded)
    brain = base[F == FIDX["Forebrain"]] if "Forebrain" in FIDX else base[m]
    if len(brain) < 8:
        brain = base[m]
    c = brain.mean(0)
    ap_span = float(np.ptp(brain[:, 0])) + 1e-6
    # ear level: at the eye AP if eyes exist, else mid-brain; slightly posterior in DV
    eye = base[F == FIDX["Eye"]] if "Eye" in FIDX else base[:0]
    ear_ap = float(np.median(eye[:, 0])) if len(eye) > 8 else c[0]
    ear_dv = c[1] - 0.05 * ap_span                                          # ~eye line, a touch behind
    # head half-width AT the ear level from the SKIN surface (NOT the narrow brain) -> ears on the lateral wall
    skin = base[F == FIDX["Skin"]] if "Skin" in FIDX else base[:0]
    band = skin[np.abs(skin[:, 0] - ear_ap) < 0.18 * ap_span] if len(skin) else base[:0]
    ref = band if len(band) > 20 else brain
    ml_half = float(np.percentile(np.abs(ref[:, 2] - c[2]), 80)) + 1e-6     # lateral head-wall ML
    Rap = 0.30 * ap_span                                                    # ear HEIGHT (AP) -- ears are tall
    Rdv = 0.18 * ap_span                                                    # ear width (DV, front-back)
    prot = 0.05 * ml_half                                                   # shallow protrusion beyond the head wall
    rng = np.random.default_rng(0)
    z = base[m, 2]
    for sgn in (-1.0, 1.0):                                                 # each side
        side = m[np.sign(z - c[2]) == sgn] if np.any(np.sign(z - c[2]) == sgn) else m[z * sgn >= 0]
        if len(side) == 0:
            continue
        n = len(side)
        t = rng.random(n) * 2 * np.pi
        r = np.sqrt(rng.random(n))
        a = ear_ap + r * np.cos(t) * Rap                                   # oval pinna in the AP-DV plane
        b = ear_dv + r * np.sin(t) * Rdv * 0.9
        # concha: dip the centre inward, rim slightly out -> an ear cup, thin in ML
        depth = (ml_half + prot) - (prot + 0.10 * ml_half) * (1.0 - r)      # rim out, centre in
        wall = sgn * depth + sgn * rng.normal(size=n) * 0.01 * ml_half
        base[side, 0] = a
        base[side, 1] = b
        base[side, 2] = wall
    return base


if __name__ == "__main__":
    from medic.adult_persistence_audit import build_base
    b, F = build_base(30000)
    b2 = shape_ears(b, F)
    m = F == FIDX["Otic"]
    print("otic cells:", int(m.sum()), "ML spread before %.3f after %.3f" %
          (float(np.std(b[m, 2])), float(np.std(b2[m, 2]))))
