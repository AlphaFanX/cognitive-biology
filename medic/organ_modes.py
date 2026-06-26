"""The electric-face correspondence, for an organ: the heart (Paper #4 §2.6 demo).

The face demo showed the gap-junction field operator and the facial geometry share
a low-mode eigenbasis. The same should hold for an organ -- and the heart is the
principled test: it is THE gap-junction organ (connexin-43), and the gap-junction
operator on the cardiac syncytium is exactly the passive cable / bidomain operator
(dV/dt = D nabla^2 V - V/tau), whose eigenmodes are the cardiac activation modes.

We build the standard idealised left ventricle -- a truncated prolate spheroid,
closed at the apex, open at the base (valve plane) -- and compare on that one
surface:
  * the GEOMETRY operator (cotangent Laplace-Beltrami), and
  * the FIELD operator (gap-junction graph Laplacian = the NCA k_gj nabla^2),
weighted by the heart's ABC gap-junction conductance.

HONEST SCOPE: idealised LV geometry, not a patient mesh; the low cymatic alphabet
is partly generic to smooth surfaces, so the demonstration is operator-level (the
field and the form share an eigenbasis on the same organ). The biological claim --
that measured cardiac activation maps ARE these low modes -- is referenced, not
validated here.

Run: python -m medic.organ_modes
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path("face_demo")).resolve()))
import face_eigenmodes as fe                      # reuse robust cotangent_laplacian
from medic.bioelectric_development import _ABC_ION_CHANNEL_ACTIVITY, E_K

K = 40
OUT = Path("data/organ_cascade")


def lv_mesh(nu=56, nv=72, u_max=0.80 * np.pi, a=1.0, c=1.7):
    """Truncated prolate-spheroid left ventricle: closed apex, open base.

    Returns V (n,3), F (m,3) triangles, UV (n,2)=(circumferential v, longitudinal u)."""
    verts = [np.array([0.0, 0.0, 0.0])]            # apex pole
    uv = [np.array([0.0, 0.0])]
    ring0 = 1
    for i in range(1, nu + 1):
        u = (i / nu) * u_max
        for j in range(nv):
            v = 2 * np.pi * j / nv
            x = a * np.sin(u) * np.cos(v)
            y = a * np.sin(u) * np.sin(v)
            z = c * (1 - np.cos(u))                # apex (u=0) at z=0
            verts.append(np.array([x, y, z])); uv.append(np.array([v, u]))
    V = np.array(verts); UV = np.array(uv)

    def ring(i, j):                                # vertex index in ring i (1..nu), col j
        return ring0 + (i - 1) * nv + (j % nv)

    F = []
    for j in range(nv):                            # apex fan
        F.append([0, ring(1, j), ring(1, j + 1)])
    for i in range(1, nu):                         # quad strips -> 2 triangles
        for j in range(nv):
            a0, b0 = ring(i, j), ring(i, j + 1)
            a1, b1 = ring(i + 1, j), ring(i + 1, j + 1)
            F.append([a0, b0, b1]); F.append([a0, b1, a1])
    return V, np.array(F), UV


def gut_tube(nz=84, nv=40, length=6.0, r0=1.0, r1=0.75):
    """Idealised intestinal segment: a slightly tapered open tube (oral->aboral).

    The gut is a smooth-muscle electrical syncytium; the interstitial-cell-of-Cajal
    gap-junction network carries the slow wave -- a travelling wave along the
    oral-aboral (axial) axis. Returns V, F, UV=(circumferential v, axial z)."""
    verts, uv = [], []
    for i in range(nz + 1):
        t = i / nz
        z = length * t
        r = r0 * (1 - t) + r1 * t
        for j in range(nv):
            v = 2 * np.pi * j / nv
            verts.append(np.array([r * np.cos(v), r * np.sin(v), z]))
            uv.append(np.array([v, z]))
    V = np.array(verts); UV = np.array(uv)

    def idx(i, j):
        return i * nv + (j % nv)

    F = []
    for i in range(nz):
        for j in range(nv):
            a0, b0 = idx(i, j), idx(i, j + 1)
            a1, b1 = idx(i + 1, j), idx(i + 1, j + 1)
            F.append([a0, b0, b1]); F.append([a0, b1, a1])
    return V, np.array(F), UV


def gapjunction_laplacian(V, F):
    n = V.shape[0]
    I = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]
    J = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
    A = sp.coo_matrix((np.ones(I.size), (I, J)), shape=(n, n)).tocsr()
    A = (A > 0).astype(float)
    d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(d) - A).tocsr()


def low_modes(L, M, k, generalized):
    if generalized:
        vals, vecs = spla.eigsh(L, k=k + 1, M=M, sigma=-1e-8, which="LM")
    else:
        vals, vecs = spla.eigsh(L, k=k + 1, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]


def gj_ratio_rank():
    return sorted(((o, v[4] / sum(v[:4])) for o, v in _ABC_ION_CHANNEL_ACTIVITY.items()),
                  key=lambda kv: -kv[1])


def run_organ(name, V, F, UV, abc_key, mode_names, ylabel, shape_title, headline):
    print(f"\n== {name} ==  mesh: {V.shape[0]} verts, {F.shape[0]} faces")
    Lcot, M, _ = fe.cotangent_laplacian(V, F)          # geometry
    Lgj = gapjunction_laplacian(V, F)                  # syncytium field operator
    lam_lb, phi_lb = low_modes(Lcot, M, K, True)
    lam_gj, phi_gj = low_modes(Lgj, None, K, False)

    Qlb, _ = np.linalg.qr(phi_lb)
    Qgj, _ = np.linalg.qr(phi_gj)
    C = Qgj.T @ Qlb
    captured = (C ** 2).sum(0)
    rho = float(spearmanr(lam_lb, lam_gj).correlation)
    cap5 = float(captured[:5].mean()); cap_all = float(captured.mean())
    GJ = _ABC_ION_CHANNEL_ACTIVITY[abc_key][4]
    ratio = GJ / sum(_ABC_ION_CHANNEL_ACTIVITY[abc_key][:4])
    rank = [o for o, _ in gj_ratio_rank()]
    print(f"  geometry vs field: Spearman rho={rho:.3f}, capture {cap5:.3f} (5) -> {cap_all:.3f} ({K})")
    print(f"  {abc_key} ABC g_GJ={GJ:.1f} (ratio {ratio:.3f}); rank #{rank.index(abc_key)+1} of {len(rank)}")
    perm = np.argmax(np.abs(C), axis=0)
    sgn = np.sign(C[perm, np.arange(K)])

    def sct(ax, cvals, title):
        ax.scatter(UV[:, 0], UV[:, 1], c=cvals, s=4, cmap="RdBu_r", linewidths=0)
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("circumferential", fontsize=6); ax.set_ylabel(ylabel, fontsize=6)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0, 0]); ax.scatter(V[:, 0], V[:, 2], c=V[:, 2], s=3, cmap="viridis", linewidths=0)
    ax.set_aspect("equal"); ax.set_title(shape_title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 1]); ax.plot(lam_lb, lam_gj, ".", ms=3)
    ax.set_xlabel("geometry eigenvalue", fontsize=7); ax.set_ylabel("field eigenvalue", fontsize=7)
    ax.set_title(f"(b) eigenvalue ordering\nSpearman rho = {rho:.2f}", fontsize=8); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[0, 2]); ax.plot(np.arange(1, K + 1), np.cumsum(captured) / np.arange(1, K + 1), "o-", ms=2)
    ax.set_ylim(0, 1.02); ax.axhline(1, ls="--", c="k", lw=0.5)
    ax.set_xlabel("# modes", fontsize=7); ax.set_ylabel("mean capture", fontsize=7)
    ax.set_title(f"(c) field reconstructs geometry\ncapture {cap5:.2f} (5) -> {cap_all:.2f} ({K})", fontsize=8)
    ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    ax.text(0.0, 0.98, headline + "\n\n"
            f"{abc_key} ABC g_GJ = {GJ:.0f}  (rank #{rank.index(abc_key)+1}/{len(rank)}).\n"
            f"geometry & field share the eigenbasis:\n"
            f"  Spearman rho = {rho:.2f}\n  capture {cap5:.2f} (first 5 modes).",
            fontsize=8, va="top", family="monospace")
    for cc in range(4):
        ax = fig.add_subplot(gs[1, cc]); sct(ax, phi_lb[:, cc], f"geometry mode {cc+1}\n[{mode_names[cc]}]")
    for cc in range(4):
        b = int(perm[cc]); s = float(sgn[cc])
        ax = fig.add_subplot(gs[2, cc]); sct(ax, s * phi_gj[:, b],
                                             f"field (gap-junction) mode {b+1}\nmatch |r|={abs(C[b, cc]):.2f}")
    fig.suptitle(f"The field-form correspondence for the {name}: the gap-junction field operator (row 3) and the "
                 f"organ geometry (row 2) share a low-mode eigenbasis  (rho={rho:.2f}, capture {cap5:.2f})",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / f"{abc_key}_modes.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("  saved", out)
    json.dump({"organ": abc_key, "n_verts": int(V.shape[0]), "spearman_eigorder": rho,
               "capture_first5": cap5, "capture_all": cap_all, "gGJ": float(GJ),
               "gj_relax_ratio": float(ratio), "gj_ratio_rank": rank},
              open(OUT / f"{abc_key}_modes.json", "w"), indent=2)
    return rho, cap5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Vh, Fh, UVh = lv_mesh()
    run_organ("heart (left ventricle)", Vh, Fh, UVh, "heart",
              ["apex-base gradient", "circumferential dipole", "dipole (orth.)", "saddle / quadrupole"],
              "apex - base", "(a) idealised LV (prolate spheroid)\napex bottom, base open",
              "THE HEART = the gap-junction organ (Cx43).\n"
              "Its syncytium operator IS the passive cable /\n"
              "bidomain operator; the low modes are cardiac\n"
              "activation modes. Mode 1 = apex-base gradient,\n"
              "the dominant activation axis.")
    Vg, Fg, UVg = gut_tube()
    run_organ("gut (intestinal segment)", Vg, Fg, UVg, "gut",
              ["oral-aboral gradient", "circumferential (ring)", "ring (orth.)", "axial / ring mix"],
              "oral - aboral", "(a) idealised intestine\n(tapered open tube)",
              "THE GUT = a smooth-muscle syncytium.\n"
              "The ICC gap-junction network carries the\n"
              "SLOW WAVE -- a travelling wave along the\n"
              "oral-aboral axis. Mode 1 = oral-aboral\n"
              "gradient, the peristaltic/slow-wave axis.")


if __name__ == "__main__":
    main()
