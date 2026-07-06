#!/usr/bin/env python3
"""
The recursive primordium operator on the gut's genome-derived operator, in 3D.
==============================================================================

The gut counterpart of medic.heart_primordium_3d. The gut is the second canonical
electrical syncytium (the interstitial cells of Cajal carry the slow wave through
a gap-junction network), so its genome-derived gap-junction operator has the
oral-aboral peristaltic axis as its first mode (Paper #4 showed this on the gut
surface; here in the volume). The internal cavity is the gut LUMEN, which in
development forms by RECANALIZATION -- the gut passes through a solid stage and the
central core is hollowed by apoptosis (failure = duodenal atresia). That is a
resorb/cavitation attractor, exactly as for the heart chamber.

  * volume: a solid tapered cylinder (oral z=0 -> aboral), the gut rod.
  * genome-derived operator: 6-neighbour graph Laplacian weighted by the gut's
    gap-junction conductance, with a spatial ICC map (oral-aboral slow-wave
    frequency gradient).
  * recursive primordia: nucleate at successive modal antinodes -> oral-aboral
    regionalisation (foregut / midgut / hindgut), telomere-gated.
  * internal cavity: the lumen recanalizes -- the central core cavitates along the
    whole oral-aboral length.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.gut_primordium_3d
Output: gut_primordium_3d.png (+ console trace)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

try:
    from .bioelectric_development import _compute_organ_conductances
except ImportError:  # pragma: no cover
    from medic.bioelectric_development import _compute_organ_conductances

NX, NY, NZ = 26, 26, 64           # narrow long tube (oral-aboral = z)
TELOMERE0 = 16
N_MODES = 12


def gut_volume():
    """Solid tapered cylinder: oral (z=0, wider) -> aboral (z=1, narrower)."""
    xs = (np.arange(NX) - (NX - 1) / 2) / ((NX - 1) / 2)
    ys = (np.arange(NY) - (NY - 1) / 2) / ((NY - 1) / 2)
    X, Y, Z = np.meshgrid(xs, ys, np.arange(NZ), indexing="ij")
    t = Z / (NZ - 1)                                  # 0 oral -> 1 aboral
    r = np.sqrt(X ** 2 + Y ** 2)
    Rout = 0.92 * (1.0 - 0.25 * t)                    # gentle aboral taper
    inside = r <= Rout
    return inside, t, r, Rout


def icc_map(coords, g0):
    """Spatial ICC gap-junction map: the slow-wave coupling runs highest orally and
    decreases aborally (the frequency gradient of the gut pacemaker)."""
    t = coords[:, 2] / (NZ - 1)
    return g0 * (1.3 - 0.6 * t)                        # oral high -> aboral low


def gj_operator(inside, spatial=True):
    g0 = _compute_organ_conductances()["gut"][4]
    coords = np.argwhere(inside)
    idx = -np.ones(inside.shape, int); idx[inside] = np.arange(coords.shape[0])
    n = coords.shape[0]
    gmap = icc_map(coords, g0) if spatial else np.full(n, g0)
    I, J = [], []
    for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        nb = np.roll(idx, -np.array(d), axis=(0, 1, 2))
        m = inside & (nb >= 0)
        I += [idx[m], nb[m]]; J += [nb[m], idx[m]]
    I = np.concatenate(I); J = np.concatenate(J)
    w = 0.5 * (gmap[I] + gmap[J])
    A = sp.coo_matrix((w, (I, J)), shape=(n, n)).tocsr()
    L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
    return L, coords, g0, gmap


def low_modes(L, k):
    vals, vecs = spla.eigsh(L, k=k + 1, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]


def lumen_mask(coords, frac):
    """Recanalization: the central core cavitates -> the lumen, along the length."""
    X = (coords[:, 0] - (NX - 1) / 2) / ((NX - 1) / 2)
    Y = (coords[:, 1] - (NY - 1) / 2) / ((NY - 1) / 2)
    t = coords[:, 2] / (NZ - 1)
    r = np.sqrt(X ** 2 + Y ** 2)
    Rout = 0.92 * (1.0 - 0.25 * t)
    Rin = (Rout - 0.32) * frac
    return r < Rin


GUT_FATE = [
    (0.00, 0.33, "foregut (Sox2/Foxa2; oesophagus-stomach)"),
    (0.33, 0.72, "midgut (Cdx2; small intestine)"),
    (0.72, 1.01, "hindgut (post. Hox/Cdx2; colon)"),
]


def gut_fate(t):
    for lo, hi, nm in GUT_FATE:
        if lo <= t < hi:
            return nm
    return "gut tube"


def run_operator(vals, vecs, coords):
    t_all = coords[:, 2] / (NZ - 1)
    prim, telomere, level = [], TELOMERE0, 0
    trace = []
    while telomere >= 1 and level < N_MODES:
        phi = vecs[:, level]
        for vi in (int(np.argmax(phi)), int(np.argmin(phi))):
            prim.append(dict(level=level, vox=vi, t=float(t_all[vi]),
                             fate=gut_fate(t_all[vi]), division=int(telomere)))
        trace.append(dict(level=level, telomere=telomere, n_prim=len(prim)))
        telomere /= 2.0; level += 1
    return trace, prim


def main():
    print("=" * 76)
    print("RECURSIVE PRIMORDIUM OPERATOR on the GUT's genome-derived operator, in 3D")
    print("=" * 76)
    inside, t, r, Rout = gut_volume()
    L, coords, g0, gmap = gj_operator(inside, spatial=True)
    print(f"\nGut wall volume: {coords.shape[0]} voxels; genome g_gj (gut ABC) = {g0:.3f}")
    print(f"Spatial ICC map: g_gj in [{gmap.min():.3f}, {gmap.max():.3f}] "
          f"(oral fast slow-wave -> aboral slow)")

    vals, vecs = low_modes(L, N_MODES)
    t_all = coords[:, 2] / (NZ - 1)
    rho = np.corrcoef(vecs[:, 0], t_all)[0, 1]
    print(f"Operator mode 1 vs oral-aboral axis: |corr| = {abs(rho):.2f} "
          f"(mode 1 = the peristaltic / slow-wave axis)")

    trace, prim = run_operator(vals, vecs, coords)
    print(f"\nRecursion ran {len(trace)} levels (telomere gated); {len(prim)} primordia:")
    for p in prim:
        print(f"   level {p['level']} t={p['t']:.2f} (div {p['division']:2d}) -> {p['fate']}")

    lum = lumen_mask(coords, 1.0)
    print(f"\nGut lumen (recanalization): {lum.mean()*100:.0f}% hollows along the length "
          f"-> wall + lumen. (cavitation; failure = atresia)")

    _figure(coords, vecs, t_all, prim, lum)
    return abs(rho) > 0.7 and len(trace) >= 4 and 0.1 < lum.mean() < 0.6


def _figure(coords, vecs, t_all, prim, lum):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    wall = ~lum
    fig = plt.figure(figsize=(18, 5.5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(x[wall], y[wall], z[wall], c=vecs[wall, 0], cmap="RdBu_r", s=4, alpha=0.4)
    ax.set_title("(a) genome operator mode 1 =\noral-aboral peristaltic axis", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_box_aspect((1, 1, 3))

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.scatter(x[wall], y[wall], z[wall], c="lightgray", s=2, alpha=0.1)
    fates = sorted({p["fate"].split(" (")[0] for p in prim})
    cmap = {f: c for f, c in zip(fates, ["#d62728", "#2ca02c", "#1f77b4", "#9467bd"])}
    for p in prim:
        vi = p["vox"]
        ax2.scatter(x[vi], y[vi], z[vi], c=cmap[p["fate"].split(" (")[0]], s=60,
                    marker="v", edgecolors="k", depthshade=False)
    ax2.set_title("(b) primordia at modal antinodes\n(foregut/midgut/hindgut)", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([]); ax2.set_box_aspect((1, 1, 3))

    ax3 = fig.add_subplot(1, 3, 3)
    midx = (np.abs(coords[:, 0] - (NX - 1) / 2) <= 1)
    ax3.scatter(z[midx & wall], y[midx & wall], c="#7a5230", s=10, label="gut wall")
    ax3.scatter(z[midx & lum], y[midx & lum], c="#cfe8ff", s=10, label="lumen (recanalized)")
    ax3.set_aspect("equal"); ax3.set_xlabel("oral -> aboral (z)"); ax3.set_ylabel("y")
    ax3.set_title("(c) gut lumen by recanalization\n(internal cavity, resorb attractor)", fontsize=9)
    ax3.legend(fontsize=7, loc="upper right")

    fig.suptitle("The recursive primordium operator on the gut's genome-derived gap-junction operator, in 3D: "
                 "mode 1 = oral-aboral peristaltic axis,\nprimordia subdivide it into foregut/midgut/hindgut, "
                 "and the lumen forms by recanalization (cavitation along the length).", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig("gut_primordium_3d.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("\nSaved: gut_primordium_3d.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}")
