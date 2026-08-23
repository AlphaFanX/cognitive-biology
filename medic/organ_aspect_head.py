"""organ_aspect_head.py -- STAGE-1 shape knob, wired into the build: give each aspect-fixable organ the
PCA-axis aspect that best matches its isolated canonical mesh (from organ_shape_search).

This is the "make it a real knob, not a post-hoc reshape" step: instead of warping the final cloud in a
viewer, the build itself reads a per-organ aspect knob (ASPECT, the genome-like table produced by the D2
knob search) and redistributes each organ's cells to that aspect at generation time (after condensation).

Only ASPECT-FIXABLE organs are wired (the knob search showed a knob genuinely closes their D2 gap).
EXCLUDED: heart + its chambers (shaped by heart_tube_head), subheads folded into a parent (Nephron->Kidney,
LiverHaem->Liver), bones (Rib -> rib_cage_head), and TOPOLOGY-limited organs (gut coil / eye / hindbrain /
pancreas / spinal cord) where an aspect knob only games D2 without fixing the real (mechanism) shape gap.

Volume-preserving: the aspect (axis RATIOS) is what the scale-invariant D2 optimised, so an overall isotropic
rescale is free w.r.t. D2 -- we normalise to keep each organ's volume, so the body doesn't blow up.
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX

# fate -> (a1, a2): scale the 2nd/3rd principal axes vs the 1st. From medic.organ_shape_search (aspect-ok set).
ASPECT = {
    "Kidney":     (2.60, 2.20), "Midbrain":   (2.60, 2.20), "Lung":     (2.06, 2.20),
    "Forebrain":  (2.06, 2.20), "Cerebellum": (2.60, 2.20), "Spleen":   (0.64, 2.20),
    "Adrenal":    (1.63, 1.75), "Liver":      (1.02, 2.20), "Thymus":   (0.81, 1.39),
    "Bladder":    (0.81, 1.10),
}


def apply(base, F):
    """Redistribute each wired organ's cells to its canonical aspect (in the organ's own principal frame,
    about its centroid so the address is preserved). Read-only on every other cell."""
    base = np.asarray(base, float).copy()
    F = np.asarray(F)
    for name, (a1, a2) in ASPECT.items():
        if name not in FIDX:
            continue
        m = F == FIDX[name]
        if m.sum() < 8:
            continue
        P = base[m]
        c = P.mean(0)
        C = P - c
        Vt = np.linalg.svd(C, full_matrices=False)[2]          # principal axes (rows), largest first
        coords = C @ Vt.T                                       # into the organ's principal frame
        coords[:, 1] *= a1
        coords[:, 2] *= a2
        coords *= (a1 * a2) ** (-1.0 / 3.0)                    # volume-preserving (D2 depends only on ratios)
        base[m] = coords @ Vt + c                               # back to the body frame, centroid kept
    return base


if __name__ == "__main__":
    from medic.adult_persistence_audit import build_base
    b, F = build_base(30000)
    print("organ aspect head: wired organs present:",
          [n for n in ASPECT if n in FIDX and (F == FIDX[n]).sum() >= 8])
