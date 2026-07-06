#!/usr/bin/env python3
"""
The recursive primordium operator on the heart's genome-derived operator, in 3D.
================================================================================

This lifts medic.primordium_operator (a 1-D schematic) onto a real organ's
GENOME-DERIVED gap-junction operator, in a 3-D VOLUME, with an internal cavity --
the volumetric template for the morphogenesis programme.

Pieces, each already grounded:
  * the HEART is THE gap-junction organ (connexin-43); the gap-junction operator on
    the cardiac syncytium is the passive cable/bidomain operator, whose low
    eigenmodes are the cardiac activation modes (Paper #4 showed mode 1 = the
    apex-base axis on the LV surface). Here the operator is built in the VOLUME.
  * the operator is GENOME-DERIVED: the 6-neighbour graph Laplacian on the
    myocardial voxels, weighted by the heart's ABC gap-junction conductance g_gj
    (medic.bioelectric_development; a scalar connexin read from accessibility).
  * the recursive PRIMORDIUM operator (Miles): {division, differentiation,
    motility} under {telomere, Hox/PRC, motility clocks} x {field}, then repeat.
    Primordia nucleate at the operator's successive modal ANTINODES (the cymatic
    subdivision = the conduction-system / chamber regionalisation); the telomere
    clock gates the recursion depth and supplies the stop signal.
  * the internal CAVITY: the ventricular LUMEN emerges by CAVITATION -- a resorb
    (negative) attractor seeded on the endocardial core hollows the inner volume,
    leaving the myocardial wall. This is the volumetric feature a surface model
    cannot make.

Honest scope: idealised prolate-spheroid heart, uniform (scalar) g_gj, operator-
level; the fate labels are positional (apex-base) not validated cell types.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.heart_primordium_3d
Output: heart_primordium_3d.png (+ console trace)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from .bioelectric_development import _compute_organ_conductances
except ImportError:  # pragma: no cover
    from medic.bioelectric_development import _compute_organ_conductances

# grid (x, y, z); z = apex(0) -> base. Modest for a CPU volumetric eigensolve.
NX, NY, NZ = 30, 30, 44
TELOMERE0 = 16
N_MODES = 12


# ---------------------------------------------------------------------------
# 1. The 3D volumetric heart (myocardial primordium) + geometry
# ---------------------------------------------------------------------------
def heart_volume():
    """Solid truncated prolate spheroid: closed apex (z=0), open base (z=1).
    Returns the in-heart voxel coords (m,3), their (i,j,k) indices, and a t=z/NZ
    apex-base coordinate per voxel."""
    xs = (np.arange(NX) - (NX - 1) / 2) / ((NX - 1) / 2)     # -1..1
    ys = (np.arange(NY) - (NY - 1) / 2) / ((NY - 1) / 2)
    X, Y, Z = np.meshgrid(xs, ys, np.arange(NZ), indexing="ij")
    t = Z / (NZ - 1)                                         # 0 apex -> 1 base
    r = np.sqrt(X ** 2 + Y ** 2)
    Rout = np.sqrt(np.clip(t * (2 - t), 0, 1)) * 0.95        # closed apex, max at base
    inside = (r <= Rout) & (t >= 0.04)
    return inside, X, Y, Z, t, r, Rout


def connexin_map(coords, g0):
    """Spatial connexin (gap-junction conductance) map, a step beyond the scalar
    ABC read: the cardiac conduction system is not uniformly coupled. Working
    myocardium is Cx43-high (fast), the inflow/base node is Cx45-low (slow
    pacemaker), the apical Purkinje is Cx40-high (fastest). g_gj varies with the
    apex-base coordinate accordingly (Severs et al.; Jongsma & Wilders)."""
    t = coords[:, 2] / (NZ - 1)                              # 0 apex -> 1 base
    node = 0.35 * np.exp(-((t - 1.0) / 0.10) ** 2)           # base inflow node: low
    purk = 0.50 * np.exp(-((t - 0.05) / 0.12) ** 2)          # apex Purkinje: high
    work = 1.0                                               # working myocardium: Cx43
    factor = work + purk - node
    return g0 * np.clip(factor, 0.25, 1.8)


def gj_operator(inside, spatial=True):
    """Genome-derived gap-junction operator: 6-neighbour graph Laplacian on the
    myocardial voxels, edge weight = the heart's gap-junction conductance. With
    spatial=True the conductance is a per-voxel connexin map (edge = mean of its
    endpoints); otherwise the scalar ABC read."""
    g0 = _compute_organ_conductances()["heart"][4]           # scalar connexin read (ABC)
    coords = np.argwhere(inside)
    idx = -np.ones(inside.shape, int)
    idx[inside] = np.arange(coords.shape[0])
    n = coords.shape[0]
    gmap = connexin_map(coords, g0) if spatial else np.full(n, g0)
    I, J = [], []
    for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        nb = np.roll(idx, -np.array(d), axis=(0, 1, 2))
        m = inside & (nb >= 0)
        ai = idx[m]; bi = nb[m]
        I += [ai, bi]; J += [bi, ai]
    I = np.concatenate(I); J = np.concatenate(J)
    w = 0.5 * (gmap[I] + gmap[J])                            # edge = mean endpoint g_gj
    A = sp.coo_matrix((w, (I, J)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    L = (sp.diags(deg) - A).tocsr()
    return L, coords, g0, gmap


def low_modes(L, k):
    vals, vecs = spla.eigsh(L, k=k + 1, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]                    # drop constant mode


# ---------------------------------------------------------------------------
# 2. The internal cavity: ventricular lumen by cavitation (resorb attractor)
# ---------------------------------------------------------------------------
def lumen_mask(coords, inside, t_all, frac):
    """Resorb (cavitation) attractor: the endocardial core hollows. frac in [0,1]
    grows the lumen as the operator runs. Returns a boolean over in-heart voxels."""
    X = (coords[:, 0] - (NX - 1) / 2) / ((NX - 1) / 2)
    Y = (coords[:, 1] - (NY - 1) / 2) / ((NY - 1) / 2)
    t = coords[:, 2] / (NZ - 1)
    r = np.sqrt(X ** 2 + Y ** 2)
    Rout = np.sqrt(np.clip(t * (2 - t), 0, 1)) * 0.95
    Rin = (Rout - 0.30) * frac                               # inner prolate, grows
    return (r < Rin) & (t > 0.18)                            # apex stays solid


# ---------------------------------------------------------------------------
# 3. The recursive primordium operator on the heart operator
# ---------------------------------------------------------------------------
CARDIAC_FATE = [
    (0.78, 1.01, "base: inflow / pacemaker (SAN-like, Tbx3/Shox2)"),
    (0.45, 0.78, "mid wall: working myocardium (Nkx2-5/GATA4)"),
    (0.00, 0.45, "apex: fast conduction (Purkinje, Irx3/Nkx2-5)"),
]


def cardiac_fate(t):
    for lo, hi, nm in CARDIAC_FATE:
        if lo <= t < hi:
            return nm
    return "myocardium"


def run_operator(vals, vecs, coords):
    """Successive modal antinodes nucleate primordia (cymatic subdivision), gated
    by the telomere clock; each commits the cardiac fate its apex-base position
    selects. Returns the per-level trace."""
    t_all = coords[:, 2] / (NZ - 1)
    trace, telomere, level = [], TELOMERE0, 0
    prim = []
    while telomere >= 1 and level < N_MODES:
        phi = vecs[:, level]
        ap, an = int(np.argmax(phi)), int(np.argmin(phi))    # + and - antinodes
        for vi in (ap, an):
            prim.append(dict(level=level, vox=vi, t=float(t_all[vi]),
                             fate=cardiac_fate(t_all[vi]), division=int(telomere)))
        trace.append(dict(level=level, telomere=telomere, n_prim=len(prim),
                          antinodes=(ap, an), mode_val=float(vals[level])))
        telomere /= 2.0
        level += 1
    return trace, prim


def main():
    print("=" * 76)
    print("RECURSIVE PRIMORDIUM OPERATOR on the HEART's genome-derived operator, in 3D")
    print("=" * 76)
    inside, X, Y, Z, t, r, Rout = heart_volume()
    L, coords, g_gj, gmap = gj_operator(inside, spatial=True)
    print(f"\nMyocardial volume: {coords.shape[0]} voxels; genome g_gj (heart ABC) = {g_gj:.3f}")
    print(f"Spatial connexin map: g_gj in [{gmap.min():.3f}, {gmap.max():.3f}] "
          f"(base node slow, apex Purkinje fast, working myocardium Cx43)")

    vals, vecs = low_modes(L, N_MODES)
    # mode 1 = apex-base?
    t_all = coords[:, 2] / (NZ - 1)
    rho = np.corrcoef(vecs[:, 0], t_all)[0, 1]
    print(f"Operator mode 1 vs apex-base axis: |corr| = {abs(rho):.2f} "
          f"(mode 1 = the dominant cardiac activation axis)")

    trace, prim = run_operator(vals, vecs, coords)
    print(f"\nRecursion ran {len(trace)} levels (telomere gated); {len(prim)} primordia at modal antinodes:")
    for p in prim:
        print(f"   level {p['level']} t={p['t']:.2f} (div budget {p['division']:2d}) -> {p['fate']}")

    # internal cavity: lumen volume fraction at full cavitation
    lum = lumen_mask(coords, inside, t_all, frac=1.0)
    print(f"\nVentricular lumen (cavitation): {lum.mean()*100:.0f}% of the volume hollows "
          f"-> wall + chamber. (resorb attractor; apex stays solid)")

    _figure(coords, vecs, t_all, trace, prim, lum)
    ok = abs(rho) > 0.7 and len(trace) >= 4 and 0.1 < lum.mean() < 0.6
    return ok


def _figure(coords, vecs, t_all, trace, prim, lum):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    fig = plt.figure(figsize=(18, 6))

    # (a) mode 1 = apex-base activation axis, on the myocardial wall (lumen hollow)
    wall = ~lum
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    p = ax.scatter(x[wall], y[wall], z[wall], c=vecs[wall, 0], cmap="RdBu_r", s=5, alpha=0.5)
    ax.set_title("(a) genome operator mode 1 = apex-base\nactivation axis (myocardial wall)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # (b) the primordia at modal antinodes, coloured by cardiac fate
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(x[wall], y[wall], z[wall], c="lightgray", s=2, alpha=0.12)
    fates = sorted({p["fate"].split(":")[0] for p in prim})
    cmap = {f: c for f, c in zip(fates, ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])}
    for p in prim:
        vi = p["vox"]
        ax2.scatter(x[vi], y[vi], z[vi], c=cmap[p["fate"].split(":")[0]], s=70,
                    marker="v", edgecolors="k", depthshade=False)
    ax2.set_title("(b) primordia at modal antinodes\n(recursive cymatic subdivision)", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])

    # (c) cross-section: the lumen cavity hollowed out
    ax3 = fig.add_subplot(1, 3, 3)
    midx = (np.abs(coords[:, 0] - (NX - 1) / 2) <= 1)
    ax3.scatter(z[midx & wall], y[midx & wall], c="#b03030", s=10, label="myocardial wall")
    ax3.scatter(z[midx & lum], y[midx & lum], c="#cfe8ff", s=10, label="lumen (cavitated)")
    ax3.set_aspect("equal"); ax3.set_xlabel("apex -> base (z)"); ax3.set_ylabel("y")
    ax3.set_title("(c) ventricular lumen by cavitation\n(internal cavity, resorb attractor)", fontsize=9)
    ax3.legend(fontsize=7, loc="upper left")

    fig.suptitle("The recursive primordium operator on the heart's genome-derived gap-junction operator, in 3D: "
                 "mode 1 = apex-base activation,\nprimordia subdivide at successive modal antinodes, and the "
                 "ventricular lumen emerges by cavitation (the internal cavity a surface model cannot make).",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig("heart_primordium_3d.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: heart_primordium_3d.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}")
