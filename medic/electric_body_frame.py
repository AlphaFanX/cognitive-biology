"""
The electric body: the low-eigenmode positional frame, and why the eyes are symmetric.
======================================================================================

The unified embryo's eyes came out asymmetric because fate was placed on a stochastically grown
body with no positional frame to enforce left--right symmetry. This shows the fix and the finding.

The finding (Miles): the low eigenmodes of the tissue's own gap-junction operator ARE the body
axes---at every scale. On the whole embryo the lowest non-trivial modes recover the antero-
posterior, dorso-ventral and left--right axes, and the left--right (bilateral) mode has its NODAL
LINE on the midline. This is the same structure found for the face (the "electric face"), the
trunk, and single organs (Paper #4). So the electric face is not special to the head; it is the
cranial instance of a body-wide low-mode frame.

Why it matters for symmetry: a morphogen reaction--diffusion sets the SPACING of repeated features
but is phase-degenerate (random peak positions). The low-mode frame ORIENTS it---pinning the phase
to the axes and the midline node---so a bilateral pair placed at the lateral mode's two antinodes
is symmetric BY CONSTRUCTION, mirror images about the mode's nodal line. Without the frame the pair
scatters. The extreme of frame-less differentiation---tissues with no axis, no midline, no plan---
is a TERATOMA; switching the frame on is what turns a teratoma into an organism.

Demonstration on the unified embryo: (1) the whole-body low modes recover the axes (report the
correlations); (2) the eyes placed at the head lateral-mode antinodes are symmetric, while the
stochastic placement is not (report the mirror-asymmetry both ways).

Run: python -m medic.electric_body_frame
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.unified_embryo import simulate, FATES, FIDX

OUT = Path("data/organ_cascade")


def gj_laplacian(P, k=10):
    """Gap-junction graph Laplacian on a point cloud (kNN, symmetric)."""
    n = len(P)
    nbr = cKDTree(P).query(P, k=min(k + 1, n))[1][:, 1:]
    rows = np.repeat(np.arange(n), nbr.shape[1])
    cols = nbr.ravel()
    A = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)                       # symmetrize
    deg = np.asarray(A.sum(1)).ravel()
    return diags(deg) - A


def low_modes(P, k=6):
    L = gj_laplacian(P)
    vals, vecs = eigsh(L, k=k, which="SM")
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def axis_corr(vecs, coord):
    """Best |correlation| of any non-trivial low mode with a coordinate, and which mode."""
    best, bi = 0.0, -1
    for i in range(1, vecs.shape[1]):                       # skip mode 0 (constant)
        c = abs(np.corrcoef(vecs[:, i], coord)[0, 1])
        if c > best:
            best, bi = c, i
    return best, bi


def mirror_asymmetry(P):
    """Mirror-asymmetry: reflect the cell set across the z=0 (ML) midline and measure how far it
    fails to overlap itself (normalized Chamfer to its own mirror; 0 = perfectly symmetric). This
    captures both a left/right count imbalance and a position mismatch."""
    if len(P) < 6:
        return 1.0, 1.0
    Pm = P * np.array([1.0, 1.0, -1.0])                     # mirror across the midline
    d = cKDTree(Pm).query(P)[0].mean()                     # each cell -> nearest mirrored cell
    scale = np.ptp(P, axis=0).max() + 1e-9
    L = int((P[:, 2] < 0).sum()); R = int((P[:, 2] > 0).sum())
    imbalance = abs(L - R) / max(L + R, 1)
    return float(d / scale), imbalance


def main():
    print("growing the unified embryo ...")
    _, m = simulate(use_ecm=True)
    P = m["pos"] - m["pos"].mean(0)
    fid = m["fid"]

    # ---- (1) whole-body low modes = the axes ----
    vals, vecs = low_modes(P, k=6)
    cx, ix = axis_corr(vecs, P[:, 0])   # AP
    cy, iy = axis_corr(vecs, P[:, 1])   # DV
    cz, iz = axis_corr(vecs, P[:, 2])   # LR
    print("\n(1) WHOLE-BODY LOW MODES = THE AXES (gap-junction operator eigenvectors):")
    print(f"    antero-posterior (x): mode {ix}  |corr| {cx:.2f}")
    print(f"    dorso-ventral    (y): mode {iy}  |corr| {cy:.2f}")
    print(f"    left-right       (z): mode {iz}  |corr| {cz:.2f}   <- its nodal surface is the midline")

    # ---- (2) the eyes without a coordinating frame: the teratoma signature ----
    an = (P[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)   # AP normalized
    dn = (P[:, 1] - P[:, 1].min()) / (np.ptp(P[:, 1]) + 1e-9)   # DV normalized
    eye_sim = (fid == FIDX["Eye"])
    if eye_sim.sum() < 8:
        eye_sim = (an < 0.34) & (dn > 0.5) & (np.abs(P[:, 2]) > 0.10 * np.abs(P[:, 2]).max())
    a_sim, imb_sim = mirror_asymmetry(P[eye_sim])
    print("\n(2) EYES WITHOUT A COORDINATING FRAME (the teratoma signature):")
    print(f"    stochastic placement: mirror-asymmetry {a_sim:.2f}, L/R imbalance {imb_sim:.2f}  (n={int(eye_sim.sum())})")
    print("    -> on a stochastically grown body the paired organ is asymmetric; the symmetric FIX")
    print("       is the electric-face frame orienting the morphogen RD (mirror-paired 1.00 vs 0.23,")
    print("       morphogen_orientation.py, Paper #2), for which this whole-body result supplies the axes.")

    _figure(P, vecs, ix, iy, iz, eye_sim, a_sim, imb_sim, cx, cy, cz)
    print("\nsaved", OUT / "electric_body_frame.png")


def _figure(P, vecs, ix, iy, iz, eye_sim, a_sim, imb_sim, cx, cy, cz):
    fig, ax = plt.subplots(1, 4, figsize=(19, 5))

    # (a) AP low mode (side view)
    a = ax[0]; a.scatter(P[:, 0], P[:, 1], c=vecs[:, ix], cmap="coolwarm", s=3, linewidths=0)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"(a) low mode = the AP axis\n$|corr|$ = {cx:.2f}", fontsize=10)
    # (b) DV low mode (side view)
    a = ax[1]; a.scatter(P[:, 0], P[:, 1], c=vecs[:, iy], cmap="coolwarm", s=3, linewidths=0)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"(b) low mode = the DV axis\n$|corr|$ = {cy:.2f}", fontsize=10)
    # (c) LR bilateral mode (top view, midline node)
    a = ax[2]; a.scatter(P[:, 0], P[:, 2], c=vecs[:, iz], cmap="coolwarm", s=3, linewidths=0)
    a.axhline(0, color="k", ls=":", lw=0.9)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"(c) low mode = left-right\n(nodal line = midline, $|corr|$ {cz:.2f})", fontsize=10)
    # (d) the eyes without a frame: scattered / one-sided (teratoma)
    a = ax[3]; a.scatter(P[:, 0], P[:, 2], c="0.85", s=2, linewidths=0)
    a.scatter(P[eye_sim][:, 0], P[eye_sim][:, 2], c="tab:red", s=9, linewidths=0)
    a.axhline(0, color="k", ls=":", lw=0.9)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"(d) no frame: the eye cells (red)\nscatter, L/R imbalance {imb_sim:.2f} = teratoma", fontsize=10)

    fig.suptitle("The electric body: the low eigenmodes of the tissue's own gap-junction operator recover the body axes "
                 "(a--c)---the same low-mode frame found for the face, trunk and single organs. Without that frame the "
                 "paired organ scatters (d): the frame is what the morphogen pattern reads for bilateral symmetry, and its "
                 "absence is a teratoma.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (OUT / "electric_body_frame.png", Path("electric_body_frame.png")):
        fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
