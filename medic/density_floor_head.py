"""density_floor_head.py -- PER-PART DENSITY FLOORS (cycle 37, 2026-09-04; Miles: "continue the
loop with the per-part density floors").

The starvation census on build_base (86,333 cells, 71 fates) found the named small parts
cell-starved to the point of unscorability: Bladder 28 cells (its grays 0.0 / not-hollow failures
were a sampling floor, not anatomy), Gonad 16, Adrenal 32, OlfactoryBulb 40, the pancreatic parts
87-146, the renal subparts 109-130 -- while the trunk bulk swims in cells. The hands proved the
principle (302 cells was the binding constraint, fixed by populate_autopods' densification): a
part below its budget cannot carry shape at all. Detail scales as the cube root of n, so the
FIRST spend is allocation, not total (the strategic ladder, 2026-09-04).

THE FLOOR: every part on the allowlist is topped up to FLOOR cells by cloning its own cells with
a jitter of 0.6x the family's own median NN spacing -- in-place densification that adds sampling
mass without inventing new anatomy (clones land inside the part's existing envelope). Jitter is
scaled to the family spacing so the overcrowding death (kth-NN < 0.40x family median) would not
extrude the clones if it ran again. Allowlist is EXPLICIT (scoreable anatomy only) -- floors on
developmental residues (YSL/Hypoblast/Mesoderm/Blood) would amplify misplaced remnants.

Runs LAST in build_base, after the anoikis/overcrowding deaths (clones are never culled, and the
stragglers were already cleaned so we clone the honest core).
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX

FLOOR = 150

FLOOR_FATES = [
    "Bladder", "Adrenal", "OlfactoryBulb", "Gonad",
    "Pancreatic Head", "Pancreatic Body", "Pancreatic Tail",
    "Renal Pelvis", "Renal Medulla", "Renal Cortex",
    "Adipose",
]


def apply(base, F, floor=FLOOR, verbose=True, names=None):
    """Top up each allowlisted fate to `floor` cells by jittered cloning. Returns (base, F, report).
    `names` overrides the allowlist (cycle 82g: the 76 autopod bones named at maturation take a floor of
    their own on the matured body -- the ledger's 'completed' reads >= 20 cells at term)."""
    adds_P, adds_F, report = [], [], {}
    for name in (FLOOR_FATES if names is None else names):
        fid = FIDX.get(name)
        if fid is None:
            continue
        idx = np.where(F == fid)[0]
        n = len(idx)
        if n < 3 or n >= floor:
            continue
        need = floor - n
        P = base[idx]
        # family spacing: median NN distance (small n -> direct pairwise)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        spacing = float(np.median(D.min(axis=1)))
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = 0.002
        rng = np.random.default_rng(fid * 7919 + n)          # deterministic per (fate, count)
        src = idx[rng.integers(0, n, need)]
        jit = rng.normal(scale=0.6 * spacing, size=(need, 3))
        adds_P.append(base[src] + jit)
        adds_F.append(np.full(need, fid, dtype=F.dtype))
        report[name] = (n, floor)
    if adds_P:
        base = np.vstack([base] + adds_P)
        F = np.concatenate([F] + adds_F)
        if verbose:
            print("  [floor] " + "  ".join(f"{k} {a}->{b}" for k, (a, b) in report.items()))
    return base, F, report
