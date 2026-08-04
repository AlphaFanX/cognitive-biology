"""
AP-unbending: straighten a curved embryo into a canonical AP x DV frame.
========================================================================

A turned embryo (mouse E9.5+) folds, so a straight PCA axis cuts across the C and the
axial code (Hox, organ AP) gets compressed. This fits a PRINCIPAL CURVE (medial-axis
backbone) through the cell cloud and re-parameterises each cell by its ARC-LENGTH along
the backbone (= true AP) and its signed PERPENDICULAR offset (= DV). Unbending recovers
the axial reads and gives the canonical frame the shape-training + organ anchors want.

`unbend(xy) -> (ap, dv)` both in [0,1]/signed. Run on E9.5 to show it lifts the Hox
colinearity above the raw-PCA value.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.ap_unbend
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np


def _backbone(xy, ap, n_knots):
    qs = np.quantile(ap, np.linspace(0, 1, n_knots + 1))
    knots = []
    for i in range(n_knots):
        m = (ap >= qs[i]) & (ap <= qs[i + 1])
        if m.sum() >= 3:
            knots.append(xy[m].mean(0))
    K = np.array(knots)
    # smooth the backbone (moving average) so it's a gentle curve
    for _ in range(2):
        K[1:-1] = 0.25 * K[:-2] + 0.5 * K[1:-1] + 0.25 * K[2:]
    return K


def _project(xy, K):
    """Project each point onto the polyline K: return arc-length s and signed perp d."""
    seg = K[1:] - K[:-1]
    seglen = np.hypot(seg[:, 0], seg[:, 1]) + 1e-9
    cum = np.concatenate([[0], np.cumsum(seglen)])
    s = np.zeros(len(xy)); d = np.zeros(len(xy))
    for i, p in enumerate(xy):
        v = p - K[:-1]
        t = np.clip((v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seglen ** 2, 0, 1)
        proj = K[:-1] + t[:, None] * seg
        dist = np.hypot(p[0] - proj[:, 0], p[1] - proj[:, 1])
        j = int(np.argmin(dist))
        s[i] = cum[j] + t[j] * seglen[j]
        cross = seg[j, 0] * (p[1] - K[j, 1]) - seg[j, 1] * (p[0] - K[j, 0])   # signed side
        d[i] = dist[j] * np.sign(cross)
    return s, d


def _project_vec(xy, K):
    """Vectorised projection of ALL points onto the polyline K (for large clouds, e.g. 50k cells).
    Returns arc-length s and signed perpendicular offset d, identical semantics to _project."""
    seg = K[1:] - K[:-1]                                   # (S,2)
    seglen2 = (seg ** 2).sum(1) + 1e-12                    # (S,)
    seglen = np.sqrt(seglen2)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])       # (S+1,)
    v = xy[:, None, :] - K[None, :-1, :]                   # (N,S,2)
    t = np.clip((v * seg[None]).sum(2) / seglen2[None], 0.0, 1.0)   # (N,S)
    proj = K[None, :-1, :] + t[:, :, None] * seg[None]     # (N,S,2)
    dist = np.sqrt(((xy[:, None, :] - proj) ** 2).sum(2))  # (N,S)
    j = np.argmin(dist, 1)                                 # (N,)
    idx = np.arange(len(xy))
    s = cum[j] + t[idx, j] * seglen[j]
    cross = seg[j, 0] * (xy[:, 1] - K[j, 1]) - seg[j, 1] * (xy[:, 0] - K[j, 0])
    d = dist[idx, j] * np.sign(cross)
    return s, d


def unbend(xy, n_knots=20, iters=3, fast=None):
    """Fit a principal-curve backbone and re-parameterise each cell by arc-length (AP) + signed
    perpendicular offset (DV). `fast` uses the vectorised projection; auto-on above 4000 points."""
    xy = np.asarray(xy, float)
    if fast is None:
        fast = len(xy) > 4000
    proj = _project_vec if fast else _project
    c = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ap = c @ vt[0]
    ap = (ap - ap.min()) / (np.ptp(ap) + 1e-9)
    s = ap
    for _ in range(iters):
        K = _backbone(xy, s, n_knots)
        s, d = proj(xy, K)
        s = (s - s.min()) / (np.ptp(s) + 1e-9)
    dv = (d - d.min()) / (np.ptp(d) + 1e-9)
    return s, dv


def geodesic_ap(xy, k=8):
    """Intrinsic AP coordinate in [0,1] = the Fiedler vector (first non-trivial eigenvector of the
    normalised Laplacian) of the spatial kNN graph. For a bent embryo the Fiedler vector runs ALONG
    the body, around the C-curl -- the geodesic axis a straight PCA line cannot see. Empirically the
    best axis for a curled MOSTA section (beats PCA at every stage E9.5-E13.5); unlike the
    principal-curve backbone it is not misled by limb branches or the head blob. Not oriented -- the
    caller flips it anterior-to-posterior. Falls back to the PCA long axis if the eigensolve fails.
    """
    xy = np.asarray(xy, float)
    try:
        from scipy.spatial import cKDTree
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        n = len(xy)
        tree = cKDTree(xy)
        _, nb = tree.query(xy, k=k + 1)
        rows = np.repeat(np.arange(n), k); cols = nb[:, 1:].ravel()
        W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        W = W.maximum(W.T)
        d = np.asarray(W.sum(1)).ravel() + 1e-9
        Dinv2 = sp.diags(1.0 / np.sqrt(d))
        L = sp.identity(n) - Dinv2 @ W @ Dinv2
        vals, vecs = eigsh(L, k=3, sigma=0, which="LM")
        f = vecs[:, np.argsort(vals)[1]]
    except Exception:
        c = xy - xy.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False); f = c @ vt[0]
    return (f - f.min()) / (np.ptp(f) + 1e-9)


def graph_eigenmodes(xy, k=6, knn=8):
    """The first k non-trivial eigenvectors of the spatial-kNN normalised Laplacian (mode 1 = the
    geodesic AP axis; higher modes are the body's other standing waves, one of which is the DV axis).
    Returns an (n, k) array of modes ordered by ascending eigenvalue; PCA fallback if the solve fails."""
    xy = np.asarray(xy, float); n = len(xy)
    try:
        from scipy.spatial import cKDTree
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh
        _, nb = cKDTree(xy).query(xy, k=knn + 1)
        rows = np.repeat(np.arange(n), knn); cols = nb[:, 1:].ravel()
        W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)); W = W.maximum(W.T)
        d = np.asarray(W.sum(1)).ravel() + 1e-9; Dinv2 = sp.diags(1.0 / np.sqrt(d))
        L = sp.identity(n) - Dinv2 @ W @ Dinv2
        vals, vecs = eigsh(L, k=k + 1, sigma=0, which="LM")
        return vecs[:, np.argsort(vals)[1:k + 1]]
    except Exception:
        c = xy - xy.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False)
        return (c @ vt.T)[:, :min(k, xy.shape[1])]


def backbone_dv(xy, ap, dorsal_mask, n_knots=24, nbins=40):
    """Supervised dorso-ventral coordinate in [0,1] from the AP backbone. Fit the medial-axis backbone
    along the geodesic AP, take each cell's SIGNED perpendicular offset from it, then normalise that
    offset WITHIN each AP slice (so the head's girth and the tail's do not distort the scale) and orient
    the neural tissue dorsal. A per-slice signed distance from the backbone -- crisp where a single global
    perpendicular or a low eigenmode is noisy on a curled section."""
    xy = np.asarray(xy, float); ap = np.asarray(ap, float); dm = np.asarray(dorsal_mask, bool)
    K = _backbone(xy, ap, n_knots)
    _s, d = _project_vec(xy, K)                                  # signed perpendicular offset from the backbone
    dv = np.zeros(len(xy))
    edges = np.linspace(ap.min(), ap.max() + 1e-9, nbins + 1)
    for i in range(nbins):
        m = (ap >= edges[i]) & (ap < edges[i + 1])
        if m.sum() >= 4:
            dd = d[m]
            lo, hi = np.percentile(dd, [3, 97])
            dv[m] = np.clip((dd - lo) / (hi - lo + 1e-9), 0, 1)  # per-slice normalise the offset
        elif m.sum():
            dv[m] = 0.5
    if dm.sum() and dv[dm].mean() < 0.5:                         # orient: neural tissue dorsal (=1)
        dv = 1.0 - dv
    sep = float(abs(dv[dm].mean() - 0.5)) * 2 if dm.sum() else 0.0
    return dv, sep


def dv_eigenmode(xy, dorsal_mask, ventral_mask, ap=None, k=6):
    """The dorso-ventral coordinate in [0,1]: the low graph eigenmode that best SEPARATES DORSAL tissue
    (e.g. spinal cord) from VENTRAL tissue (heart/liver/gut), excluding AP-aligned modes (the AP harmonics),
    oriented dorsal=1. A dorsal-vs-ventral contrast (not neural-vs-all, which conflates AP because the neural
    axis runs the whole body) gives a clean DV on a curled section where a perpendicular-offset DV is noisy."""
    V = graph_eigenmodes(xy, k=k)
    dm = np.asarray(dorsal_mask, bool); vm = np.asarray(ventral_mask, bool)
    if ap is None:
        ap = V[:, 0]
    apc = ap - ap.mean()
    if dm.sum() < 3 or vm.sum() < 3:
        v = V[:, 1] if V.shape[1] > 1 else V[:, 0]
        return (v - v.min()) / (np.ptp(v) + 1e-9), 0.0
    best, bestsep = None, -1.0
    for m in range(V.shape[1]):
        v = V[:, m]
        if abs(np.corrcoef(v, apc)[0, 1]) > 0.5:                 # skip AP axis + its harmonics
            continue
        vn = (v - v.min()) / (np.ptp(v) + 1e-9)
        sep = abs(float(vn[dm].mean() - vn[vm].mean()))          # dorsal-vs-ventral separation
        if sep > bestsep:
            best, bestsep = v, sep
    if best is None:
        best = V[:, 1] if V.shape[1] > 1 else V[:, 0]
    dv = (best - best.min()) / (np.ptp(best) + 1e-9)
    if dv[dm].mean() < dv[vm].mean():
        dv = 1.0 - dv
    return dv, float(bestsep)


def _hox_colinearity(h5ad, ap):
    import re, anndata as ad, scipy.sparse as sp
    a = ad.read_h5ad(h5ad)
    vn = {str(g): i for i, g in enumerate(a.var_names)}
    per = {}
    for g, i in vn.items():
        m = re.match(r"Hox([abcd])(\d+)$", g)
        if not m:
            continue
        col = a.X[:, i]; e = col.toarray().ravel() if sp.issparse(col) else np.asarray(col).ravel()
        if (e > 0).sum() < 30:
            continue
        per.setdefault(int(m.group(2)), []).append(float((e / e.sum() * ap).sum()))
    ps = sorted(per); av = [np.mean(per[p]) for p in ps]
    return abs(float(np.corrcoef(ps, av)[0, 1])), ps, av


def main():
    import anndata as ad
    h5 = "data/mosta/E9.5_E2S2.MOSTA.h5ad"
    a = ad.read_h5ad(h5)
    xy = np.asarray(a.obsm["spatial"], float)
    c = xy - xy.mean(0); _, _, vt = np.linalg.svd(c, full_matrices=False)
    ap_pca = (c @ vt[0]); ap_pca = (ap_pca - ap_pca.min()) / (np.ptp(ap_pca) + 1e-9)
    ap_un, dv_un = unbend(xy)

    ap_geo = geodesic_ap(xy)
    r_pca, _, _ = _hox_colinearity(h5, ap_pca)
    r_un, ps, av = _hox_colinearity(h5, ap_un)
    r_geo, psg, avg = _hox_colinearity(h5, ap_geo)
    print(f"E9.5 Hox colinearity  |  raw PCA-AP: rho={r_pca:.2f}   principal-curve unbend: rho={r_un:.2f}"
          f"   Fiedler geodesic: rho={r_geo:.2f}")
    print(" geodesic Hox AP by paralog:", {p: round(v, 2) for p, v in zip(psg, avg)})
    best = max([("PCA", r_pca), ("principal-curve unbend", r_un), ("Fiedler geodesic", r_geo)],
               key=lambda t: t[1])
    print(f" -> best axis for recovering axial colinearity: {best[0]} (rho={best[1]:.2f}).")


if __name__ == "__main__":
    main()
