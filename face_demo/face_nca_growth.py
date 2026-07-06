"""
Forward NCA growth of the face from morphogen primordia (Paper #2).
===================================================================

A step on the road to full volumetric, genome-derived morphogenesis. Where
face_morphogenesis.py FIT heat-kernel bumps to the observed relief (a static
decomposition), this GROWS the facial relief FORWARD with the inner perceptron's
own local rule, from morphogen PRIMORDIA placed at the facial prominences, and
then VALIDATES the emergent form against the FaceBase mean face (it is not fit to
it).

The primordia. Each canonical facial prominence (frontonasal, medial- and
lateral-nasal, maxillary, mandibular) is seeded as a morphogen source on the
cranial surface, with an a-priori outgrowth amplitude (anatomical / GWAS-grounded
ordering: nose and jaw largest), NOT tuned to the target.

The growth. The same inner-perceptron rule as the body NCA (medic.nca_vertebrate_3d),
on the cranial-surface graph:

    U <- U + k_relax (seed - U) + k_gj (P U - U)

where seed is the primordium field, P is the row-normalized surface adjacency
(gap-junction coupling), and U is the outgrowth field. The prominences grow and
spread by the rule; the relief EMERGES.

Honest scope (the open program, shared with every organ): the operator P here is
the surface GEOMETRY's adjacency, not yet the genome-derived gap-junction operator
(the connexin/ABC-weighted operator the organ-formation paper derives for heart
and gut); the model is a SURFACE, so internal/volumetric features -- paranasal
sinuses (pneumatization = internal cavitation, a resorptive negative attractor) --
cannot form and need a 3D volumetric NCA with a resorb cell-state; and there is no
prominence-fusion topology. This file does the forward growth; those three are the
road ahead.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe face_demo/face_nca_growth.py
Output: face_nca_growth.png, face_nca_growth.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesh_morph as mm
from face_eigenmodes import clean_mesh
from face_morphogenesis import prominence_sources, nearest

K_RELAX, K_GJ, STEPS = 0.02, 0.10, 300


def build_primordia(A):
    """The principled set of facial primordia as (name, 3D point, amplitude,
    footprint-fraction). Amplitudes are a-priori (anatomical ordering; nose, jaw
    protrude, the OPTIC primordia RECESS the orbits) -- NOT tuned to the target.
    Each is a competence domain, not a point."""
    s, cx = A["size"], A["cx"]
    nose, nasion, chin = A["nose"], A["nasion"], A["chin"]
    bridge = 0.5 * (nasion + nose)                              # nasal ridge midpoint
    orbitL = nasion + np.array([-0.13 * s, -0.02 * s, -0.03 * s])
    orbitR = nasion + np.array([+0.13 * s, -0.02 * s, -0.03 * s])
    cheekL = np.array([cx - 0.26 * s, nose[1], nose[2] - 0.04 * s])
    cheekR = np.array([cx + 0.26 * s, nose[1], nose[2] - 0.04 * s])
    return [
        ("frontonasal",     nasion + np.array([0, 0.08 * s, 0]), 0.50, 0.100),
        ("nasal-bridge",    bridge,            0.60, 0.060),
        ("medial-nasal",    nose,              1.00, 0.075),
        ("lateral-nasal L", A["alare_L"],      0.30, 0.050),
        ("lateral-nasal R", A["alare_R"],      0.30, 0.050),
        ("optic/orbit L",   orbitL,           -0.60, 0.070),
        ("optic/orbit R",   orbitR,           -0.60, 0.070),
        ("maxillary L",     cheekL,            0.55, 0.120),
        ("maxillary R",     cheekR,            0.55, 0.120),
        ("mandibular",      chin,              0.90, 0.130),
    ]


def surface_P(n, F):
    """Row-normalized surface adjacency (gap-junction coupling operator P)."""
    e0 = np.r_[F[:, 0], F[:, 1], F[:, 2]]
    e1 = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    A = sp.coo_matrix((np.ones(e0.size), (e0, e1)), shape=(n, n))
    A = (A + A.T).tocsr(); A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ A


def relief(V):
    """FaceBase target relief: anterior protrusion with smooth quadratic trend removed."""
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    B = np.c_[np.ones_like(x), x, y, x * x, y * y, x * y]
    coef, *_ = np.linalg.lstsq(B, z, rcond=None)
    T = z - B @ coef
    return T - T.mean()


def grow(seed, P, steps=STEPS, k_relax=K_RELAX, k_gj=K_GJ, capture=None):
    """Forward NCA: relax toward the primordium seeds + diffuse on the surface."""
    U = np.zeros_like(seed)
    frames = []
    for s in range(steps):
        U = U + k_relax * (seed - U) + k_gj * (P @ U - U)
        if capture is not None and s in capture:
            frames.append(U.copy())
    return U, frames


def main():
    V0, F, HL = mm.load()
    if F.min() >= 1 and F.max() >= V0.shape[0]:
        F = F - 1
    A = mm.anchors(V0)
    src_pts = prominence_sources(V0, A, 1.0)
    V, F, keep, s = clean_mesh(V0, F)
    n = V.shape[0]
    print(f"mesh: {V0.shape[0]} verts raw -> {n} clean, {F.shape[0]} faces")

    P = surface_P(n, F)
    size = float(np.linalg.norm(V.max(0) - V.min(0)))

    primordia = build_primordia(A)
    src_names = [p[0] for p in primordia]
    src_idx = [nearest(V, np.asarray(p[1]) / s) for p in primordia]

    # primordium seed field: each primordium is a competence DOMAIN -- a Gaussian
    # patch (footprint = its anatomical size) at signed amplitude set a priori
    # (optic primordia recess the orbits; nose and jaw protrude).
    seed = np.zeros(n)
    for (nm, _pt, amp, sig_frac), vi in zip(primordia, src_idx):
        d2 = ((V - V[vi]) ** 2).sum(1)
        seed += amp * np.exp(-d2 / (2 * (sig_frac * size) ** 2))

    cap = set(np.linspace(0, STEPS - 1, 5).astype(int))
    U, frames = grow(seed, P, capture=cap)

    # validate against the FaceBase relief (NOT fit -- amplitudes set a priori)
    T = relief(V)
    # match overall scale for a fair shape comparison (single global scalar, sign-free)
    a = float((U @ T) / (U @ U + 1e-12))
    corr = float(np.corrcoef(U, T)[0, 1])
    print(f"\nForward NCA growth: {len(src_idx)} primordia, {STEPS} steps "
          f"(k_relax={K_RELAX}, k_gj={K_GJ})")
    print(f"  emergent relief vs FaceBase: correlation = {corr:.3f}  (validation, not a fit)")
    print(f"  global scale a = {a:.3f}")
    print("\nPrimordium amplitudes (a-priori, anatomical ordering):")
    for nm, _pt, amp, sig in sorted(primordia, key=lambda p: -p[2]):
        print(f"    {nm:16s} amp={amp:+.2f}  footprint={sig:.3f}")

    _figure(V, src_idx, seed, frames + [U], a * np.ones(1), U, T, corr)
    json.dump(dict(n_primordia=len(src_idx), steps=STEPS, k_relax=K_RELAX, k_gj=K_GJ,
                   corr_vs_facebase=corr,
                   primordia={p[0]: [float(p[2]), float(p[3])] for p in primordia}),
              open(HERE / "face_nca_growth.json", "w"), indent=2)
    print("\nsaved face_nca_growth.json")
    return corr > 0.45


def _figure(V, src_idx, seed, stages, a, U, T, corr):
    x, y = V[:, 0], V[:, 1]
    fig = plt.figure(figsize=(18, 5))
    n_stage = len(stages)
    gs = fig.add_gridspec(1, n_stage + 2)
    Um = max(np.abs(U).max(), 1e-9)

    # growth stages
    for i, S in enumerate(stages):
        ax = fig.add_subplot(gs[0, i])
        ax.scatter(x, y, c=S, s=2, cmap="viridis", vmin=0, vmax=Um, linewidths=0)
        if i == 0:
            for vi in src_idx:
                ax.scatter(V[vi, 0], V[vi, 1], c="r", s=14, marker="x")
            ax.set_title("primordia (t=0)", fontsize=9)
        else:
            ax.set_title(f"NCA growth step {i}/{len(stages)-1}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    lim = np.percentile(np.abs(T), 98)
    axT = fig.add_subplot(gs[0, n_stage])
    axT.scatter(x, y, c=T, s=2, cmap="RdBu_r", vmin=-lim, vmax=lim, linewidths=0)
    axT.set_title("FaceBase relief (target)", fontsize=9)
    axT.set_aspect("equal"); axT.set_xticks([]); axT.set_yticks([])

    axG = fig.add_subplot(gs[0, n_stage + 1])
    gl = np.percentile(np.abs(U), 98)
    axG.scatter(x, y, c=U, s=2, cmap="RdBu_r", vmin=-gl, vmax=gl, linewidths=0)
    axG.set_title(f"grown relief (r={corr:.2f} vs FaceBase)", fontsize=9)
    axG.set_aspect("equal"); axG.set_xticks([]); axG.set_yticks([])

    fig.suptitle("Forward NCA growth of the face from morphogen primordia: the inner perceptron's local "
                 "rule grows the prominences\nfrom a-priori-amplitude seeds; the emergent relief is VALIDATED "
                 "(not fit) against FaceBase. (Step on the road to volumetric, genome-derived morphogenesis.)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(HERE / "face_nca_growth.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved face_nca_growth.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}")
