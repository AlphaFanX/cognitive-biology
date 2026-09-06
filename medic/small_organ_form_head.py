"""small_organ_form_head.py -- FORM for the parts the density floors just gave substrate
(cycle 38, 2026-09-04; the ladder's second rung: allocation gave cells, form shapes them).

BLADDER VESICLE: the urinary bladder forms from the urogenital sinus (cloacal partitioning,
SHH/BMP4/ISL1) as a HOLLOW endodermal sac from the start -- it is never solid in life. The
150-cell family (28 pre-floor) was a shapeless clump scoring grays hollow 0.0 (check: >=0.2).
Reshape: cells onto a spheroid SHELL (radius 0.013 of stature = the empty bladder ~4.5 cm /
165 cm; shell band 0.75-1.0 R) about the family's own centroid.

PANCREAS AXIS: the grays Pancreas check needs elong >= 1.5 (an elongated gland); the family read
1.26 -- the belly squeeze bunched Head/Body/Tail into one clump (cycle-19 note). The pancreatic
bud elongates along the dorsal mesentery (PDX1/PTF1A): arrange the three subfamily centroids
along the measured head->tail axis (head in the duodenal C on the right, tail rising toward the
spleen on the left; span ~0.075 of stature), each subfamily riding as one body.

Runs after density_floor_head in build_base -- flows into BOTH the grays/canonical assemble()
specimen and the movie (which matures the same base cloud).
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX

_BLADDER_R_FRAC = 0.013          # empty-bladder radius / stature (bp3d-scale)
_PANC_SPAN_FRAC = 0.095          # pancreas head->tail span / stature (real gland ~15/165 cm = 0.091)
_PANC_DIR = np.array([0.22, 0.05, -0.95])   # laid frame (AP up, DV, ML; model left = -ML):
#                                              tail rises up-left toward the spleen


def bladder_vesicle(base, F):
    # THE FAMILY, not the parent label (cycle 82 -- families move as families): once the bladder has
    # real cells (the cycle-82 allocation: 300+ at 120k, up from ~8), subhead_program splits them into
    # Trigone / Body / Dome BEFORE this head runs, so `F == Bladder` found nothing (< 40) and the
    # vesicle silently never formed -- grays hollow 0.00. The shell now takes every family cell.
    from medic.subhead_program import expand_names
    ids = [FIDX[n] for n in expand_names(["Bladder"]) if n in FIDX]
    if not ids:
        return base
    idx = np.where(np.isin(F, ids))[0]
    if len(idx) < 40:
        return base
    fid = FIDX["Bladder"]
    stat = float(np.ptp(base[:, 0])) + 1e-9
    R = _BLADDER_R_FRAC * stat
    c = base[idx].mean(0)
    rng = np.random.default_rng(fid * 7919 + len(idx))
    # ARRANGEMENT-PRESERVING shell: each cell keeps its own direction from the family centroid (so
    # the trigone stays at the inferior pole and the dome at the superior one -- the chain split's
    # order survives the reshaping); only cells sitting on the centroid get a seeded random direction.
    # Radii by RANK of the cell's original depth in [0.75, 1.0] R (a sac wall with thickness).
    d = base[idx] - c
    n = np.linalg.norm(d, axis=1)
    u = np.where(n[:, None] > 1e-9, d / (n[:, None] + 1e-12), 0.0)
    deg = n <= 1e-9
    if deg.any():
        v = rng.normal(size=(int(deg.sum()), 3)); u[deg] = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    rank = np.argsort(np.argsort(n)).astype(float) / (len(n) - 1 + 1e-9)
    rr = R * (0.75 + 0.25 * rank ** (1 / 3))
    base = base.copy()
    base[idx] = c + u * rr[:, None]
    return base


def pancreas_axis(base, F):
    parts = [FIDX.get(n) for n in ("Pancreatic Head", "Pancreatic Body", "Pancreatic Tail")]
    if any(p is None for p in parts):
        return base
    masks = [np.where(F == p)[0] for p in parts]
    if any(len(m) < 20 for m in masks):
        return base
    stat = float(np.ptp(base[:, 0])) + 1e-9
    d = _PANC_DIR / np.linalg.norm(_PANC_DIR)
    span = _PANC_SPAN_FRAC * stat
    allidx = np.concatenate(masks)
    c = base[allidx].mean(0)
    base = base.copy()
    for k, m in enumerate(masks):                        # head at -d (right), tail at +d (left)
        tgt = c + d * span * (k - 1) * 0.5
        base[m] += tgt - base[m].mean(0)
        # compaction so each part reads as a segment of the gland, not a loose cloud (0.55: the
        # first pass at 0.75 left family elong 1.19 -- the parts' own spread swamped the axis)
        base[m] = tgt + (base[m] - tgt) * 0.55
    return base


def apply(base, F, verbose=True):
    base = bladder_vesicle(base, F)
    base = pancreas_axis(base, F)
    if verbose:
        print("  [form] bladder vesicle shell + pancreas head->tail axis")
    return base, F
