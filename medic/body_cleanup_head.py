"""body_cleanup_head.py -- fix the obvious inspection artifacts (2026-08-08, Miles): the spinal cord reads as a
fat DV wedge, the heart is oversized + bulging, the eyes are large low blobs. All are cosmetic cell-position
reshapes (no skeleton touched -> Gray's integrity unaffected), wired read-only per fate in build_base.
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX


def thin_spinal_cord(base, F, ml_keep=0.30, dv_keep=0.30, nb=16):
    """Compress the Spinal Cord to a THIN tube around its own per-AP-band centre-line (it followed the spine but
    spread into a DV wedge). Keeps the cord's path + all cells; just thins ML+DV so it reads as a cord, not a wedge."""
    from medic.subhead_program import expand_names
    ids = [FIDX[n] for n in expand_names(["Spinal Cord"]) if n in FIDX]   # + the 4 cord-level sub-heads
    if not ids:
        return base
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    idx = np.where(np.isin(F, ids))[0]
    if len(idx) < 20:
        return base
    P = base[idx]; x = P[:, 0]
    xn = (x - x.min()) / (np.ptp(x) + 1e-9); b = np.clip((xn * nb).astype(int), 0, nb - 1)
    for k in range(nb):
        m = b == k
        if m.sum() < 3:
            continue
        cml = 0.0                                   # cord sits on the midline (ML=0)
        cdv = np.median(P[m, 1])                    # per-band DV centre = the spine's path at that level
        P[m, 2] = cml + (P[m, 2] - cml) * ml_keep
        P[m, 1] = cdv + (P[m, 1] - cdv) * dv_keep
    base[idx] = P
    return base


def shrink_heart(base, F, scale=0.72):
    """The heart is ~3x the liver (oversized) + bulges laterally. Scale the chambers toward their shared centroid,
    keeping the D-loop shape + the (lateralised) address."""
    ids = [FIDX[n] for n in ("Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow")
           if n in FIDX]                                   # incl the L/R ventricle relabels (heart_tube split)
    if not ids:
        return base
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    m = np.isin(F, ids)
    if m.sum() < 20:
        return base
    c = base[m].mean(0)
    base[m] = c + (base[m] - c) * scale
    return base


def compact_eyes(base, F, scale=0.62):
    """The eyes are large low blobs. Shrink each eyeball toward its OWN centre (split L/R), keeping position so the
    orbit still reads there -- a compact eyeball, not a big blob."""
    ids = [FIDX[n] for n in ("Eye", "Retina") if n in FIDX]
    if not ids:
        return base
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    m = np.isin(F, ids)
    if m.sum() < 20:
        return base
    z = base[m, 2]; zc = np.median(z)
    for side in (z >= zc, z < zc):
        sel = np.where(m)[0][side]
        if len(sel) < 8:
            continue
        c = base[sel].mean(0)
        base[sel] = c + (base[sel] - c) * scale
    return base


def straighten_posture(base, F):
    """Remove the net DV lean (the body stands diagonally): subtract the DV-vs-AP linear trend so the long axis
    is vertical. Excludes the lateral limbs from the FIT (arms/legs skew it) but de-shears every cell.

    SUPERSEDED (2026-08-09): the lean is now fixed at its SOURCE -- adult_persistence_audit.build_base builds the
    basis at flex=0 (straight) instead of flex=1 (a ~24 deg whole-body bow), which dropped the trunk lean from
    +16.8 deg to +1.9 deg with Gray's integrity still 13/13 (flex is a rigid arc applied before parts are
    identified, so straightening preserves rib<->vertebra articulation -- unlike this post-hoc DV shear, which
    broke it 13->12/13). Kept only as a diagnostic; NOT applied in build_base."""
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    x = base[:, 0]
    trunk = np.abs(base[:, 2]) < np.percentile(np.abs(base[:, 2]), 60)      # central trunk drives the fit
    if trunk.sum() < 50:
        trunk = np.ones(len(base), bool)
    a = np.polyfit(x[trunk], base[trunk, 1], 1)[0]
    base[:, 1] = base[:, 1] - a * (x - x.mean())
    return base


def apply(base, F):
    base = thin_spinal_cord(base, F)
    base = shrink_heart(base, F)
    base = compact_eyes(base, F)
    # NOTE: straighten_posture() is NOT applied here -- the lean is fixed at its SOURCE (build_base now builds the
    # basis at flex=0, straight, instead of flex=1's ~24 deg bow; lean +16.8->+1.9 deg, integrity still 13/13).
    # The global DV de-shear was the wrong tool (it broke rib articulation 13->12/13); the source fix is a rigid
    # arc change that keeps articulation. straighten_posture is kept only as a diagnostic.
    return base
