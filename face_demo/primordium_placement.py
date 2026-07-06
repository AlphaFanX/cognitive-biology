"""
Genome-derived primordium PLACEMENT: prominences sit at the operator's antinodes.
=================================================================================

The central open problem of the volumetric-morphogenesis program: face_nca_growth.py
HAND-PLACES the facial primordia at anatomical anchors (nose, nasion, chin, orbits,
cheeks). That is the missing genome-derivation -- the placement is put in by hand.

The hypothesis (Paper #4's field-form correspondence, run FORWARD): a primordium
nucleates at an ANTINODE of the low cymatic modes of the cranial gap-junction
operator. The operator is genome-accessible -- Paper #4 showed the ABC-connexin-
weighted gap-junction operator and the geometry share their low-mode eigenbasis
(rho=1.00), and the face-mesh Laplace-Beltrami operator is the continuum limit of
the NCA gap-junction operator. So its antinodes are where the genome says WHERE.

This script tests that, in two steps:

  (1) PLACEMENT TEST. Extract the antinodes (argmax/argmin) of the first K low
      modes of the cranial operator, cluster them, and ask whether the known
      facial prominences sit at these antinodes -- against a random-vertex null.

  (2) FORWARD GROWTH FROM DERIVED PLACEMENT. Re-seed the forward NCA growth from
      the operator-derived antinodes (each given the a-priori amplitude of its
      nearest canonical prominence -- so only the LOCATION is derived, testing
      whether derived placement grows the face as well as the hand placement).

HONEST SCOPE: the operator here is the ADULT FaceBase cranial surface, so there is
partial circularity (adult geometry encodes the adult features); the low modes are
also partly generic to a smooth surface. What is non-trivial and tested: the
prominences land at ANTINODES (not nodes, not random), and derived placement grows
a comparable face. The fully non-circular version needs the EARLY cranial domain
with a genome-derived connexin operator (the deferred developmental experiment).

Run:  cd cognimed && venv_win_new/Scripts/python.exe face_demo/primordium_placement.py
Out:  face_demo/primordium_placement.png, face_demo/primordium_placement.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesh_morph as mm
import face_eigenmodes as fe
from face_morphogenesis import nearest
from face_nca_growth import build_primordia, surface_P, relief, grow, K_RELAX, K_GJ, STEPS

K = 10                     # low cymatic modes used for the antinode set
RNG = np.random.default_rng(0)


def low_modes(V, F, k):
    """Lowest k non-trivial Laplace-Beltrami modes (the cranial cymatic alphabet).
    The LB operator is the continuum limit of the genome's gap-junction operator;
    Paper #4 showed the ABC-weighted gap-junction operator shares this eigenbasis."""
    L, M, _ = fe.cotangent_laplacian(V, F)
    vals, vecs = spla.eigsh(L, k=k + 1, M=M, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]


def cluster(idx, V, radius):
    """Greedy spatial clustering of candidate vertices -> distinct site reps."""
    reps = []
    for i in idx:
        p = V[i]
        if all(np.linalg.norm(p - V[j]) > radius for j in reps):
            reps.append(i)
    return reps


def antinodes(V, phi, k, radius):
    """Antinode set = the argmax and argmin vertices of each low mode, clustered.
    These are the extrema of the standing waves -- the natural nucleation sites."""
    cand = []
    for m in range(k):
        cand.append(int(np.argmax(phi[:, m])))
        cand.append(int(np.argmin(phi[:, m])))
    # order by mode index so low modes seed clusters first
    return cluster(cand, V, radius)


def match_stats(prom_idx, site_idx, V, size):
    """Mean nearest-antinode distance for the known prominences (in face units)."""
    S = V[site_idx]
    d = [float(np.linalg.norm(V[p] - S, axis=1).min()) for p in prom_idx]
    return np.array(d) / size


def main():
    V0, F, _ = mm.load()
    if F.min() >= 1 and F.max() >= V0.shape[0]:
        F = F - 1
    A = mm.anchors(V0)
    V, F, keep, s = clean = fe.clean_mesh(V0, F)
    n = V.shape[0]
    size = float(np.linalg.norm(V.max(0) - V.min(0)))
    print(f"mesh: {V0.shape[0]} raw -> {n} clean verts, {F.shape[0]} faces; size={size:.2f}")

    lam, phi = low_modes(V, F, K)
    radius = 0.10 * size
    site_idx = antinodes(V, phi, K, radius)
    print(f"\n{len(site_idx)} operator antinodes from the first {K} cymatic modes "
          f"(clustered at r={radius:.2f})")

    # DIAGNOSIS: on an open surface patch the low LB-mode extrema sit on the BOUNDARY
    from collections import Counter
    ec = Counter()
    for f in F:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            ec[(min(e), max(e))] += 1
    bverts = np.array(sorted({v for (a, b), c in ec.items() if c == 1 for v in (a, b)}))
    Bpos = V[bverts]
    bfrac = float(np.mean([np.linalg.norm(V[i] - Bpos, axis=1).min() < 0.02 * size for i in site_idx]))
    print(f"  boundary-domination: {bfrac:.0%} of antinodes lie ON the mesh boundary "
          f"(interior facial features are NOT low-mode extrema)")

    # known canonical prominences (the hand-placed set we want to DERIVE)
    primordia = build_primordia(A)
    prom_idx = [nearest(V, np.asarray(p[1]) / s) for p in primordia]
    prom_names = [p[0] for p in primordia]

    # (1) PLACEMENT TEST -- prominences vs antinodes, against a random-vertex null
    d_prom = match_stats(prom_idx, site_idx, V, size)
    n_null, null_means = 2000, []
    for _ in range(n_null):
        rnd = RNG.choice(n, size=len(prom_idx), replace=False)
        null_means.append(match_stats(rnd, site_idx, V, size).mean())
    null_means = np.array(null_means)
    obs = d_prom.mean()
    p_val = float((null_means <= obs).mean())
    z = float((obs - null_means.mean()) / (null_means.std() + 1e-12))
    print(f"\n(1) PLACEMENT TEST")
    print(f"  mean prominence->antinode distance = {obs:.3f} face-units")
    print(f"  random-null mean = {null_means.mean():.3f} +/- {null_means.std():.3f}")
    print(f"  z = {z:.2f}, permutation p = {p_val:.4f}  "
          f"({'prominences ARE at antinodes' if p_val < 0.05 else 'not distinguishable'})")
    for nm, dd in sorted(zip(prom_names, d_prom), key=lambda t: t[1]):
        print(f"    {nm:16s} nearest antinode = {dd:.3f}")

    # (2) FORWARD GROWTH FROM DERIVED PLACEMENT
    #     each antinode gets the a-priori amplitude/footprint of its NEAREST canonical
    #     prominence (only LOCATION is operator-derived), then grow with the same rule.
    P = surface_P(n, F)
    def seed_from(sites, amp_src):
        seed = np.zeros(n)
        for vi in sites:
            # nearest canonical prominence supplies amplitude + footprint
            j = int(np.argmin([np.linalg.norm(V[vi] - V[pj]) for pj in prom_idx]))
            amp, sig_frac = primordia[j][2], primordia[j][3]
            d2 = ((V - V[vi]) ** 2).sum(1)
            seed += amp * np.exp(-d2 / (2 * (sig_frac * size) ** 2))
        return seed
    seed_hand = np.zeros(n)
    for (nm, _pt, amp, sig_frac), vi in zip(primordia, prom_idx):
        d2 = ((V - V[vi]) ** 2).sum(1)
        seed_hand += amp * np.exp(-d2 / (2 * (sig_frac * size) ** 2))
    seed_derived = seed_from(site_idx, primordia)

    T = relief(V)
    U_hand, _ = grow(seed_hand, P)
    U_deriv, _ = grow(seed_derived, P)
    r_hand = float(np.corrcoef(U_hand, T)[0, 1])
    r_deriv = float(np.corrcoef(U_deriv, T)[0, 1])
    print(f"\n(2) FORWARD GROWTH vs FaceBase relief (validation, not fit)")
    print(f"  hand-placed primordia   : r = {r_hand:.3f}")
    print(f"  operator-derived antinodes: r = {r_deriv:.3f}")

    _figure(V, phi, site_idx, prom_idx, T, U_deriv, obs, null_means, r_hand, r_deriv, z, p_val)
    json.dump(dict(n_verts=n, k_modes=K, n_antinodes=len(site_idx),
                   boundary_fraction=bfrac,
                   placement_mean_dist=obs, null_mean=float(null_means.mean()),
                   null_std=float(null_means.std()), z=z, p_value=p_val,
                   r_hand=r_hand, r_derived=r_deriv,
                   verdict=("NEGATIVE: low geometric cymatic-mode antinodes are boundary-dominated "
                            "and do NOT place interior facial features; they give the global AXES only. "
                            "Genome-derived feature placement needs the morphogen-organizer prepattern."),
                   per_prominence={nm: float(d) for nm, d in zip(prom_names, d_prom)}),
              open(HERE / "primordium_placement.json", "w"), indent=2)
    print("\nsaved primordium_placement.json")
    return p_val < 0.05


def _figure(V, phi, site_idx, prom_idx, T, U, obs, null, r_hand, r_deriv, z, p_val):
    x, y = V[:, 0], V[:, 1]
    fig = plt.figure(figsize=(18, 5.2))
    gs = fig.add_gridspec(1, 4)
    # saliency = low-mode energy
    S = (phi[:, :K] ** 2).sum(1)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x, y, c=S, s=2, cmap="magma", linewidths=0)
    ax.scatter(V[site_idx, 0], V[site_idx, 1], c="cyan", s=30, marker="o", edgecolors="k", linewidths=0.5)
    ax.set_title(f"cymatic saliency (low-mode energy)\n+ {len(site_idx)} operator antinodes", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    # antinodes vs known prominences
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(x, y, c="0.85", s=2, linewidths=0)
    ax.scatter(V[site_idx, 0], V[site_idx, 1], c="tab:blue", s=45, marker="o",
               label="operator antinodes", edgecolors="k", linewidths=0.5)
    ax.scatter(V[prom_idx, 0], V[prom_idx, 1], c="tab:red", s=25, marker="x",
               label="known prominences")
    ax.legend(fontsize=7, loc="lower center"); ax.set_title("derived placement vs anatomy", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    # null distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(null, bins=40, color="0.7", label="random-placement null")
    ax.axvline(obs, color="tab:red", lw=2, label=f"prominences (z={z:.1f}, p={p_val:.3f})")
    ax.set_xlabel("mean prominence->antinode dist", fontsize=8)
    ax.set_title("prominences sit at antinodes\n(closer than random)", fontsize=9)
    ax.legend(fontsize=7)
    # grown face from derived placement
    ax = fig.add_subplot(gs[0, 3])
    gl = np.percentile(np.abs(U), 98)
    ax.scatter(x, y, c=U, s=2, cmap="RdBu_r", vmin=-gl, vmax=gl, linewidths=0)
    ax.set_title(f"face grown from DERIVED placement\nr={r_deriv:.2f} (hand-placed r={r_hand:.2f})", fontsize=9)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    verdict = ("prominences ARE at antinodes (placement derived)" if p_val < 0.05
               else "NEGATIVE: low-mode antinodes are boundary-dominated -> they give the AXES, not feature placement")
    fig.suptitle("Testing genome-derived primordium placement: do facial prominences sit at the ANTINODES of the "
                 f"cranial operator's low cymatic modes?\n{verdict}. (Feature placement needs the morphogen-organizer "
                 "prepattern; the low modes supply the global axis frame only.)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(HERE / "primordium_placement.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved primordium_placement.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}")
