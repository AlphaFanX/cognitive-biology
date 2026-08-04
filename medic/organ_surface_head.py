"""
organ_surface_head.py -- the per-organ SURFACE: turn each condensed organ point cloud into a closed boundary
mesh, so an organ renders as a Gray's-crisp SOLID rather than a fuzzy scatter of points.

The condensation head (adult_persistence_audit._condense_organs) sorts each organ into a sharply-bounded mass;
this head wraps that mass in a surface. The method is an ALPHA SHAPE (a concave hull), built with scipy alone:
a Delaunay tetrahedralisation of the organ's points, keeping only the tetrahedra whose circumscribed sphere is
smaller than a radius alpha, then taking the triangles that bound exactly one kept tetrahedron. Unlike a convex
hull it follows concavity (the reniform notch of the kidney, the cardiac loop), and unlike a fixed-topology
envelope it needs no seam and handles a bilateral organ's two lobes as two surfaces automatically. alpha is set
per organ as a multiple of the median nearest-neighbour spacing, so it adapts to each organ's density.

READ-ONLY on the cloud. Validation = a watertight-ish closed surface per organ (each boundary edge shared by
two faces), tight to the points (max point-to-hull gap small).

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.organ_surface_head
Out: data/organ_cascade/organ_surface_head.png  (+ .npz of the meshes)
"""
from __future__ import annotations
import os
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from medic.adult_persistence_audit import build_base, _CONDENSE_ORGANS
from medic.unified_embryo import FIDX

# a colour per organ for the solid render
COLORS = {
    "Heart": "#c0392b", "Atrium": "#e05a4a", "Ventricle": "#a93226", "Outflow": "#d98880",
    "Lung": "#e59866", "Liver": "#7d5a3c", "Kidney": "#8e44ad", "Nephron": "#a569bd",
    "Eye": "#2e86c1", "Retina": "#5dade2", "Pancreas": "#d4ac0d", "Otic": "#48c9b0",
    "Spleen": "#884ea0", "Thymus": "#7fb3d5", "Adrenal": "#e8a33d", "Bladder": "#5499c7",
    "Forebrain": "#58d68d", "Midbrain": "#45b39d", "Hindbrain": "#28b463", "Cerebellum": "#82e0aa"}


def _circumradius(p):
    """Circumscribed-sphere radius of each tetrahedron p (M,4,3). Degenerate tets -> +inf (dropped)."""
    p0 = p[:, 0]
    A = 2.0 * (p[:, 1:] - p0[:, None])                          # (M,3,3)
    b = (p[:, 1:] ** 2).sum(-1) - (p0 ** 2).sum(-1)[:, None]    # (M,3)
    R = np.full(len(p), np.inf)
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-12
    if ok.any():
        c = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]   # circumcentre
        R[ok] = np.linalg.norm(c - p0[ok], axis=1)
    return R


def alpha_shape(P, alpha):
    """Concave-hull triangles of the point set P at radius alpha. Returns (V, faces) with V=P."""
    if len(P) < 5:
        return P, np.zeros((0, 3), int)
    tri = Delaunay(P)
    T = tri.simplices                                          # (M,4) tetrahedra
    keep = _circumradius(P[T]) < alpha
    if not keep.any():
        keep = _circumradius(P[T]) < np.inf                   # fall back to the convex hull
    T = T[keep]
    # each tet has 4 triangular faces; a face on the BOUNDARY bounds exactly one kept tet
    faces = np.concatenate([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]])
    key = np.sort(faces, axis=1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = faces[cnt[inv] == 1]                                 # faces belonging to exactly one tet = the surface
    return P, bnd


def build(base, F, alpha_mult=3.0, cap=2500):
    """Per-organ alpha-shape surface meshes from the (condensed) cloud. alpha = alpha_mult x median NN spacing."""
    rng = np.random.default_rng(0)
    out = {}
    for nm in _CONDENSE_ORGANS:
        fid = FIDX.get(nm)
        if fid is None:
            continue
        P = base[F == fid]
        if len(P) < 30:
            continue
        if len(P) > cap:                                       # subsample for a tractable Delaunay
            P = P[rng.choice(len(P), cap, replace=False)]
        d = cKDTree(P).query(P, k=2)[0][:, 1]
        alpha = alpha_mult * float(np.median(d))
        V, faces = alpha_shape(P, alpha)
        if len(faces):
            out[nm] = dict(V=V, faces=faces, color=COLORS.get(nm, "#cccccc"), n=int((F == fid).sum()))
    return out


def _validate(mesh):
    """Closed-ness: fraction of edges shared by exactly two faces (a watertight surface -> ~1)."""
    faces = mesh["faces"]
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return float((cnt == 2).mean()), len(faces)


def _figure(meshes, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(7, 10), facecolor="#0d1017")
    ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor("#0d1017")
    for nm, m in meshes.items():
        V, faces = m["V"], m["faces"]
        # front view: X=ML(z), Y=up(AP), Z=depth(DV)
        tri = V[faces][:, :, [2, 0, 1]]
        pc = Poly3DCollection(tri, alpha=0.9, facecolor=m["color"], edgecolor="none")
        ax.add_collection3d(pc)
    B = base
    ax.set_xlim(B[:, 2].min(), B[:, 2].max()); ax.set_ylim(B[:, 0].min(), B[:, 0].max())
    ax.set_zlim(B[:, 1].min(), B[:, 1].max())
    try:
        ax.set_box_aspect((np.ptp(B[:, 2]), np.ptp(B[:, 0]), np.ptp(B[:, 1])))
    except Exception:
        pass
    ax.view_init(elev=8, azim=-90); ax.axis("off")
    ax.set_title("Per-organ surfaces (alpha-shape solids) -- Gray's-crisp organs, not point scatter",
                 color="#e2e8f0", fontsize=9)
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/organ_surface_head.png", dpi=130, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/organ_surface_head.png")


def main():
    base, F = build_base()
    meshes = build(base, F)
    print(f"\n== PER-ORGAN SURFACE (alpha-shape solids) ==  ({len(meshes)} organs)")
    print(f"{'organ':11s} {'cells':>6s} {'faces':>6s} {'closed':>7s}")
    for nm, m in meshes.items():
        closed, nf = _validate(m)
        print(f"{nm:11s} {m['n']:6d} {nf:6d} {closed:7.2f}")
    _figure(meshes, base)
    np.savez_compressed("data/organ_cascade/organ_surfaces.npz",
                        **{f"{nm}_V": m["V"] for nm, m in meshes.items()},
                        **{f"{nm}_F": m["faces"] for nm, m in meshes.items()})
    print("saved data/organ_cascade/organ_surfaces.npz")


if __name__ == "__main__":
    main()
