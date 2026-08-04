"""
Volumetric body: solidify the schematic grow-from-one-cell body into a filled tissue region.
=============================================================================================
The forward NCA body (medic.unified_embryo) is a sparse CURVE-WITH-ORGANS -- correct topology and
organ placement, but it cannot FILL a real MOSTA section, which is solid tissue at high cell density.
This module solidifies it: it samples points INSIDE the body outline (within a fill radius of a grown
cell) at uniform density and gives each the fate of the nearest grown cell, turning the sparse body
into a dense filled slice. The fill is the DIVISION head taken to tissue density; the fate propagation
is the grown genome fate map; nothing about the placement/identity is changed, only the body is made
solid so it has real area to register onto the atlas.

`fill_2d(xy, fates, n_target)` -> (xy_filled, fates_filled).  `fill_3d(P, fid, n_target)` for volume.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

# Surface ectoderm is a genomic program (masters Trp63/Grhl3/Ovol1) that is CALLED TO THE OUTSIDE: the
# outermost cells see high BMP and become epidermis. So skin is correctly a surface fate placed by a
# radial/outer rule -- not a geometry artifact. On a SOLID body it is a thin 1-2 cell epithelium (~3% of
# cells); it only blew up (~27%) because our grown body was a sparse rod with a huge surface/volume ratio.
# The fill below therefore keeps skin surface-called but fills the INTERIOR with bulk tissue, so skin's
# small fraction EMERGES from a solid interior rather than being capped.
SURFACE_FATES = {"Skin", "Epidermal", "Epidermis", "Surface Ectoderm", "Surface ectoderm"}

# the compact, condensed point-organs -- protected from vote erosion so they fill as coherent lumps
# rather than dissolving into the surrounding interstitial mesenchyme at their borders.
POINT_ORGANS = {"Heart", "Atrium", "Ventricle", "Outflow", "Liver", "LiverHaem", "Lung", "Pancreas",
                "Kidney", "Nephron", "Eye", "Otic", "Gut", "Foregut", "Hindgut", "Spleen", "Adrenal",
                "Thymus", "Gonad", "Bladder", "Choroid", "Retina"}


def _wmode(query, src_pts, src_fates, k=10, protect=None, protect_boost=3.0):
    """Distance-weighted k-NN MAJORITY fate for each query point: the dominant fate of its local
    neighbourhood, not the single nearest cell. This turns the loosely-condensed grown organs into
    COHERENT filled blobs and outvotes stray foreign cells scattered inside an organ (nearest-1 Voronoi
    fill produced salt-and-pepper). Weight 1/(d+eps) keeps small dense organs from being eroded while
    still smoothing speckle. Fates in `protect` (the compact point-organs) get their vote weight
    multiplied by `protect_boost` so a small dense organ is not eroded at its border by the surrounding
    interstitium -- organs clump rather than dissolve."""
    src_fates = np.asarray(src_fates)
    if len(query) == 0:
        return src_fates[:0].copy()
    kk = min(k, len(src_pts))
    dd, ii = cKDTree(src_pts).query(query, k=kk)
    if kk == 1:
        return src_fates[ii]
    codes, inv = np.unique(src_fates, return_inverse=True)          # fate -> integer code
    nc = inv[ii]                                                    # (Q, k) neighbour codes
    w = 1.0 / (dd + 1e-6)                                           # distance weight
    if protect:
        pcode = np.array([1.0 + (protect_boost - 1.0) * (c in protect) for c in codes])
        w = w * pcode[nc]                                          # up-weight protected organ votes
    acc = np.zeros((len(query), len(codes)), np.float64)
    rows = np.repeat(np.arange(len(query)), kk)
    np.add.at(acc, (rows, nc.ravel()), w.ravel())                  # summed weight per fate per point
    return codes[acc.argmax(1)]


def _fill(pts, fates, n_target, r, seed, ndim):
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, float)
    fates = np.asarray(fates)
    tree = cKDTree(pts)
    if r is None:
        d, _ = tree.query(pts, k=min(6, len(pts)))
        nn = d[:, 1:].ravel(); nn = nn[nn > 1e-9]           # ignore duplicate points (bilateral mirror)
        base = float(np.median(nn)) if len(nn) else 0.0
        ext = float(np.max(pts.max(0) - pts.min(0)))
        r = max(base * 2.2, 0.02 * ext)                     # floor by body extent so duplicates can't zero it
    lo = pts.min(0) - r; hi = pts.max(0) + r
    out_p, out_f, tries = [], [], 0
    have = 0
    while have < n_target and tries < 80:
        cand = rng.uniform(lo, hi, (max(n_target, 4000) * 2, ndim))
        dd, ii = tree.query(cand, k=1)
        inside = dd < r                                     # inside the body = within a ball of a grown cell
        out_p.append(cand[inside]); out_f.append(fates[ii[inside]])
        have += int(inside.sum()); tries += 1
    P = np.concatenate(out_p) if out_p else pts.copy()
    F = np.concatenate(out_f) if out_f else fates.copy()
    if len(P) > n_target:
        sel = rng.choice(len(P), n_target, replace=False); P, F = P[sel], F[sel]
    return P, F


def fill_2d(xy, fates, n_target, nbins=90, smooth=4, seed=0, target_g=None, skin_depth=0.02):
    """Fill the 2D body SILHOUETTE solidly, with a SMOOTH outline. Fit a smooth DORSAL and VENTRAL
    boundary curve vs AP (per-bin DV percentiles, interpolated over gaps, moving-averaged), then fill
    uniformly between them; each point takes the nearest grown-cell fate (organs propagate). Smoothing
    the two boundary curves -- instead of independent per-bin rectangles -- gives a clean continuous
    outline that tapers at the head and tail, a solid tissue slice with a real body contour.

    If `target_g` (a dict {'a': [...0-1 AP...], 'g': [...girth as a fraction of AP length...]}, the real
    MOSTA silhouette) is given, the body is filled to the TARGET girth per AP -- centre it on the body's
    local DV midline and set the half-height to 0.5*g(a)*L -- so the generated section takes the real
    embryo's regional bulk (big head, thick trunk, thin tail) instead of the thin grown rod's own envelope.
    The organs propagate by nearest fate, so they ride the widened body; this is the migration head
    shaping the body to the measured silhouette."""
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, float); fates = np.asarray(fates)
    x, y = xy[:, 0], xy[:, 1]
    xlo, xhi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    L = xhi - xlo + 1e-9
    edges = np.linspace(xlo, xhi, nbins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    cen = np.full(nbins, np.nan); dors = np.full(nbins, np.nan); vent = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() >= 4:
            cen[i] = np.median(y[m])
            dors[i] = np.percentile(y[m], 96); vent[i] = np.percentile(y[m], 4)
    good = ~np.isnan(dors)
    if good.sum() < 3:
        return xy.copy(), fates.copy()
    if target_g is not None:                                         # fill to the REAL silhouette girth
        cen = np.interp(ctr, ctr[good], cen[good])
        ga = np.interp((ctr - xlo) / L, np.asarray(target_g["a"]), np.asarray(target_g["g"]))
        half = 0.5 * np.clip(ga, 0.02, None) * L
        dors = cen + half; vent = cen - half
    else:
        dors = np.interp(ctr, ctr[good], dors[good])                 # fill gaps
        vent = np.interp(ctr, ctr[good], vent[good])
    for _ in range(smooth):                                          # smooth the two boundary curves
        dors[1:-1] = 0.25 * dors[:-2] + 0.5 * dors[1:-1] + 0.25 * dors[2:]
        vent[1:-1] = 0.25 * vent[:-2] + 0.5 * vent[1:-1] + 0.25 * vent[2:]
    xa = rng.uniform(xlo, xhi, n_target * 2)                         # fill between the smooth boundaries
    dv_hi = np.interp(xa, ctr, dors); dv_lo = np.interp(xa, ctr, vent)
    ya = dv_lo + rng.uniform(0, 1, len(xa)) * (dv_hi - dv_lo)
    P = np.c_[xa, ya]
    # INTERIOR = BULK, SURFACE = EPITHELIUM, both filled by a k-NN MAJORITY vote (coherent organs, no
    # speckle). Surface ectoderm is genomically called to the outside, so it is correct as a surface fate;
    # on a solid body it is a thin 1-2 cell layer. The interior votes among BULK (non-surface) cells so the
    # surface-called epidermis is confined to a genuine outer RIM of thickness `skin_depth`; skin is NOT
    # capped -- its small fraction EMERGES from the filled interior (the test that the interior fill worked).
    surf = np.array([f in SURFACE_FATES for f in fates])
    bd = np.minimum(dv_hi - ya, ya - dv_lo)                         # distance to the nearest DV boundary
    half = 0.5 * (dv_hi - dv_lo)                                    # local body half-height
    rim = bd < np.minimum(skin_depth * L, 0.35 * half)             # outer epithelial rim (bounded)
    F = np.empty(len(P), dtype=fates.dtype)
    src_int = np.where(~surf)[0] if (~surf).any() else np.arange(len(fates))
    F[~rim] = _wmode(P[~rim], xy[src_int], fates[src_int], protect=POINT_ORGANS)  # interior -> dominant BULK
    F[rim] = _wmode(P[rim], xy, fates, protect=POINT_ORGANS)       # rim -> dominant fate incl. epidermis
    if len(P) > n_target:
        sel = rng.choice(len(P), n_target, replace=False); P, F = P[sel], F[sel]
    return P, F


def fill_3d(P, fid, n_target, r=None, seed=0):
    """Fill the 3D body volume to `n_target` cells at uniform density (each = nearest grown-cell fate)."""
    return _fill(P, fid, n_target, r, seed, 3)
