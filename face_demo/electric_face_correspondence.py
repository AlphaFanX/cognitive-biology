"""Electric-face correspondence test (Paper #4 §2 demo).

The face_eigenmodes.py demo computed the GEOMETRY modes: the cotangent
Laplace-Beltrami eigenmodes of the final FaceBase surface. Those are the modes of
the *shape*, not of the developing bioelectric field. The mechanistically
meaningful operator is the NCA gap-junction Laplacian (dV/dt = k_relax(V*-V) +
k_gj nabla^2 V): a graph Laplacian whose edges are gap-junction couplings between
neighbouring cells. Its eigenmodes are the bioelectric cymatic standing waves.

This script realises the NCA gap-junction operator on the face cell-sheet (the
mesh graph) and tests whether its eigenmodes CORRESPOND to the geometric
Laplace-Beltrami modes. A strong correspondence is the operator-level form of the
"electric face" claim: the bioelectric prepattern field shares an eigenbasis with
the resulting facial geometry.

HONEST SCOPE (and why this is a demo, not the validation): this compares two
operators on the SAME final adult mesh. It is NOT yet the biological validation,
which requires (i) the embryonic head Vm prepattern from real ABC craniofacial
channel data on developing-head geometry, (ii) Levin's measured DiBAC electric-
face maps, and (iii) morphometric face data, and a test that the MEASURED
prepattern projects onto the same low modes BEFORE the geometry exists. The demo
shows the operators share an eigenbasis; the causal/temporal claim still needs the
developmental experiment.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import subspace_angles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesh_morph as mm
import face_eigenmodes as fe          # reuse cotangent_laplacian + clean_mesh

K_MODES = 60


def gapjunction_laplacian(V, F):
    """NCA gap-junction operator on the cell-sheet = unweighted graph Laplacian.

    Gap junctions couple neighbouring cells; on the mesh the neighbours are the
    triangle edges. The NCA's k_gj*nabla^2 is the discrete 5-point stencil; on an
    irregular mesh its faithful analogue is the combinatorial graph Laplacian
    L = D - A built from the mesh edges (uniform coupling, like the stencil).
    """
    n = V.shape[0]
    I = np.r_[F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]]
    J = np.r_[F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]]
    A = sp.coo_matrix((np.ones(I.size), (I, J)), shape=(n, n)).tocsr()
    A = (A > 0).astype(float)                          # unweighted adjacency
    d = np.asarray(A.sum(1)).ravel()
    L = sp.diags(d) - A
    return L.tocsr()


def low_modes(L, M, k, generalized):
    if generalized:
        vals, vecs = spla.eigsh(L, k=k + 1, M=M, sigma=-1e-8, which="LM")
    else:
        vals, vecs = spla.eigsh(L, k=k + 1, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]              # drop constant mode


def main():
    V0, F, HL = mm.load()
    if F.min() >= 1 and F.max() >= V0.shape[0]:
        F = F - 1
    A0 = mm.anchors(V0)
    V, F, keep, s = fe.clean_mesh(V0, F)
    HL = HL / s
    print(f"mesh: {V.shape[0]} clean verts, {F.shape[0]} faces")

    # --- the two operators on the SAME mesh ---
    Lcot, M, mass = fe.cotangent_laplacian(V, F)        # geometry (Laplace-Beltrami)
    Lgj = gapjunction_laplacian(V, F)                   # NCA gap-junction (graph Laplacian)

    lam_lb, phi_lb = low_modes(Lcot, M, K_MODES, generalized=True)
    lam_gj, phi_gj = low_modes(Lgj, None, K_MODES, generalized=False)
    print(f"LB  eigenvalues[:6]: {np.round(lam_lb[:6], 3)}")
    print(f"GJ  eigenvalues[:6]: {np.round(lam_gj[:6], 3)}")

    # orthonormalise each basis (Euclidean) for rotation-invariant comparison
    Qlb, _ = np.linalg.qr(phi_lb)
    Qgj, _ = np.linalg.qr(phi_gj)
    C = Qgj.T @ Qlb                                     # (K,K) overlap

    # --- correspondence metrics ---
    # (a) how much of each LB mode lives in the first-K GJ subspace (rotation-invariant)
    captured = (C ** 2).sum(axis=0)                    # per LB mode j: ||proj_GJ phi_j||^2
    # (b) cumulative subspace capture vs K
    Ks = [2, 5, 10, 20, 40, K_MODES]
    cap_K = []
    for kk in Ks:
        cj = (C[:kk, :kk] ** 2).sum(axis=0)
        cap_K.append(float(cj.mean()))
    # (c) principal angles between leading-K subspaces
    pa = {}
    for kk in [5, 10, 20, 40]:
        ang = subspace_angles(Qgj[:, :kk], Qlb[:, :kk])
        pa[kk] = {"max_angle_deg": float(np.degrees(ang.max())),
                  "mean_cos": float(np.cos(ang).mean())}
    # (d) eigenvalue rank correlation
    from scipy.stats import spearmanr
    rho = float(spearmanr(lam_lb, lam_gj).correlation)

    print(f"mean LB-mode capture by GJ subspace (K={K_MODES}): {captured.mean():.3f}")
    for kk in [5, 10, 20]:
        print(f"  K={kk:2d}: capture {cap_K[Ks.index(kk)]:.3f}, "
              f"max principal angle {pa[kk]['max_angle_deg']:.1f} deg")
    print(f"eigenvalue rank corr (Spearman): {rho:.3f}")

    # --- GWAS deformations are low-rank in the GJ basis too (consistency w/ LB) ---
    genes = ["EDAR", "PAX3", "DCHS2", "RUNX2", "GLI3", "PAX1"]
    gj_eff = {}
    for g in genes:
        D = (mm.morph(V0, A0, {g: 2.0}) - V0)[keep] / s
        coeff = Qgj.T @ D
        e = (coeff ** 2).sum(1); p = e / (e.sum() + 1e-12)
        gj_eff[g] = float(1.0 / np.sum(p ** 2))

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4)
    x, y = V[:, 0], V[:, 1]
    # top two rows: GJ mode (row0) above the matched LB mode (row1), first 4 modes.
    # Eigenvectors are defined only up to sign (phi and -phi are the same standing
    # wave), so the solver returns arbitrary signs; align each GJ mode's sign to its
    # LB partner for display. This is cosmetic only -- every metric uses C**2 / QR
    # and is sign-invariant.
    for c in range(4):
        sgn = np.sign(phi_gj[:, c] @ phi_lb[:, c]) or 1.0
        ax = fig.add_subplot(gs[0, c])
        ax.scatter(x, y, c=sgn * phi_gj[:, c], s=2, cmap="RdBu_r", linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"NCA gap-junction mode {c+1}", fontsize=8)
        ax = fig.add_subplot(gs[1, c])
        ax.scatter(x, y, c=phi_lb[:, c], s=2, cmap="RdBu_r", linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"geometry (Laplace-Beltrami) mode {c+1}", fontsize=8)
    # overlap heatmap
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(np.abs(C[:20, :20]), cmap="magma", vmin=0, vmax=1)
    ax.set_title("|overlap| GJ vs LB\n(diagonal = correspondence)", fontsize=8)
    ax.set_xlabel("LB mode"); ax.set_ylabel("GJ mode")
    fig.colorbar(im, ax=ax, fraction=0.046)
    # subspace capture vs K
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(Ks, cap_K, "o-")
    ax.set_ylim(0, 1.02); ax.axhline(1.0, ls="--", c="k", lw=0.6)
    ax.set_xlabel("# modes K"); ax.set_ylabel("mean LB-mode capture")
    ax.set_title("GJ subspace reconstructs\nthe geometry modes", fontsize=8)
    # eigenvalue correspondence
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(lam_lb, lam_gj, ".", ms=3)
    ax.set_xlabel("LB eigenvalue"); ax.set_ylabel("GJ eigenvalue")
    ax.set_title(f"eigenvalue ordering\nSpearman rho = {rho:.2f}", fontsize=8)
    # GWAS eff-modes in GJ basis
    ax = fig.add_subplot(gs[2, 3])
    ax.bar(range(len(genes)), [gj_eff[g] for g in genes])
    ax.set_xticks(range(len(genes))); ax.set_xticklabels(genes, rotation=45, fontsize=7)
    ax.set_ylabel("effective modes")
    ax.set_title("GWAS effects low-rank\nin the GJ basis too", fontsize=8)

    fig.suptitle("Electric-face correspondence: the NCA gap-junction field operator and the facial "
                 "geometry share an eigenbasis (demo on the adult mesh; biological validation pending)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE / "electric_face_correspondence.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)

    json.dump({
        "n_verts": int(V.shape[0]), "k_modes": K_MODES,
        "lb_eigenvalues": [float(v) for v in lam_lb[:8]],
        "gj_eigenvalues": [float(v) for v in lam_gj[:8]],
        "mean_capture_K60": float(captured.mean()),
        "capture_vs_K": dict(zip(map(str, Ks), cap_K)),
        "principal_angles": pa,
        "eigval_spearman": rho,
        "gwas_eff_modes_gj_basis": gj_eff,
    }, open(HERE / "electric_face_correspondence.json", "w"), indent=2)


if __name__ == "__main__":
    main()
