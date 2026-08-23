"""grown_arm_head.py -- replace the sparse build_base ARM bud with a DENSE growth-activity upper limb.

When the movie was unified onto the build_base cloud, the arm lost the old simulate-path dense fore-boost bud, so
it became cell-sparse -> `grow_limbs` could only stretch a few cells into a thin, short arm. This grows the arm the
way it forms: each bone is a growth-plate ROD at its CALIBRATED human length (humerus 0.186 H, forearm 0.146,
hand 0.108 = ~0.44 H upper limb; medic.sox9_condensation_head), laid laterally from the shoulder (Vitruvian).
Genome-anchored: each bone's length = its growth-plate ACTIVITY (the calibrated per-bone knob). Dense + full-length,
so `grow_limbs` no longer has to stretch a stub.

Read-only on every non-arm cell. Called near the end of adult_persistence_audit.build_base.
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX
from medic.sox9_condensation_head import grow_bone_rod

# upper-limb bones as fractions of stature H (standard anthropometry); radius+ulna are the parallel zeugopod.
_FORE = [("humerus", 0.186), ("radius", 0.146), ("ulna", 0.146), ("hand", 0.108)]


def grow_arms(base, F, n_seg=380):
    LB = FIDX.get("Limb Bud")
    if LB is None:
        return base, F
    base = np.asarray(base, float); F = np.asarray(F)
    ap = base[:, 0]; H = float(np.ptp(ap)) + 1e-9; apf = (ap - ap.min()) / H
    ml = base[:, 2]; mid = float(np.median(ml))
    arm = (F == LB) & (apf >= 0.5) & (np.abs(ml - mid) > 0.04 * H)     # the upper-limb bud (lateral, upper half)
    if arm.sum() < 8:
        return base, F
    keep = ~arm
    r = 0.013 * H                                                     # thin appositional scaffold (flesh adds girth)
    trans = np.array([1.0, 0.0, 0.0])                                 # forearm bones offset along AP (fore/aft)
    sh_ml = float(np.percentile(np.abs(base[keep, 2] - mid), 70))     # shoulder at the trunk's own side
    newP = []
    for si, sgn in enumerate((1.0, -1.0)):
        m = arm & (np.sign(ml - mid) == sgn)
        if m.sum() < 4:
            continue
        pos = np.array([float(np.percentile(base[m, 0], 80)),         # shoulder AP (top of the bud)
                        float(np.median(base[m, 1])), mid + sgn * sh_ml])
        axis = np.array([0.0, 0.0, sgn])                             # straight out to the side = the Vitruvian arm
        for bi, (nm, frac) in enumerate(_FORE):
            L = frac * H
            off = (0.5 if nm == "radius" else -0.5) * 2.5 * r if nm in ("radius", "ulna") else 0.0
            rod = grow_bone_rod(L, r * (1.5 if nm == "hand" else 1.0), axis, pos, transverse=trans, offset=off,
                                n=n_seg, seed=10 * si + bi)
            newP.append(rod)
            if nm != "radius":                                       # radius & ulna share the zeugopod PD span
                pos = pos + axis * L
    if not newP:
        return base, F
    newP = np.vstack(newP)
    return np.vstack([base[keep], newP]), np.concatenate([F[keep], np.full(len(newP), LB)])


def grow_feet(base, F, n=260):
    """The distal leg is a bare tapering tip (no foot). Grow a proper FOOT at each ankle: a horizontal platform
    projecting ANTERIORLY (the ankle bends ~90 deg so the sole is horizontal, toes forward). Ventral (forward) =
    away from the dorsal notochord. Genome-anchored: the foot is the autopod, oriented ventrally by the DV axis."""
    LB = FIDX.get("Limb Bud")
    if LB is None:
        return base, F
    base = np.asarray(base, float); F = np.asarray(F)
    ap = base[:, 0]; H = float(np.ptp(ap)) + 1e-9; apf = (ap - ap.min()) / H
    ml = base[:, 2]; mid = float(np.median(ml))
    hind = (F == LB) & (apf < 0.45) & (np.abs(ml - mid) > 0.02 * H)
    if hind.sum() < 8:
        return base, F
    # forward (ventral) DV direction = away from the dorsal notochord
    noto = FIDX.get("Notochord")
    vsign = -1.0
    if noto is not None and (F == noto).sum() > 8:
        vsign = -1.0 if np.median(base[F == noto, 1]) > np.median(base[:, 1]) else 1.0
    newP = []
    for si, sgn in enumerate((1.0, -1.0)):
        m = hind & (np.sign(ml - mid) == sgn)
        if m.sum() < 4:
            continue
        ankle_ap = float(np.percentile(base[m, 0], 8))                # the distal end of the leg
        anc = m & (base[:, 0] < np.percentile(base[m, 0], 22))
        start = np.array([ankle_ap, float(np.median(base[anc, 1])), float(np.median(base[m, 2]))])
        axis = np.array([0.04, vsign, 0.0]); axis /= np.linalg.norm(axis)   # slightly down, strongly ANTERIOR
        newP.append(grow_bone_rod(0.15 * H, 0.022 * H, axis, start, transverse=np.array([0.0, 0.0, 1.0]),
                                  n=n, seed=13 * si + 3))
    if not newP:
        return base, F
    newP = np.vstack(newP)
    return np.vstack([base, newP]), np.concatenate([F, np.full(len(newP), LB)])
