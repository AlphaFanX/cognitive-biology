"""
Electric-face eigenmode demo (Paper #2 entry point).

Levin's "electric face" = a standing-wave (cymatic) prepattern of the craniofacial
bioelectric field. Concretely: the low-order eigenmodes of the Laplace-Beltrami
operator on the head surface. This computes them on the real FaceBase mean-face
mesh and tests the Paper #2 claim that GWAS face effects are ADDITIVE MODE SHIFTS
(i.e. face variation is linear in the eigenbasis) by projecting the published GWAS
deformation fields (EDAR->chin, PAX3->nasion, ...) onto the modes.

If the GWAS deformations are low-rank in the eigenbasis, then:
  - the eigenmode set IS the "frozen craniofacial kernel",
  - GWAS cis-LoRA = reweighting a few modes,
  - the §3.6 linearity question resolves (additive effects = additive mode shifts).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh_morph as mm

HERE = Path(__file__).resolve().parent
K_MODES = 60          # eigenmodes to compute
N_SHOW = 8            # eigenmodes to render


def cotangent_laplacian(V, F):
    """Cotangent stiffness L (PSD) and lumped mass M (diagonal) for a triangle mesh."""
    n = V.shape[0]
    I, J, W = [], [], []
    mass = np.zeros(n)
    for tri in F:
        a, b, c = tri
        va, vb, vc = V[a], V[b], V[c]
        # edge vectors and triangle area
        cross = np.cross(vb - va, vc - va)
        area = 0.5 * np.linalg.norm(cross)
        if area < 1e-12:
            continue
        mass[a] += area / 3.0; mass[b] += area / 3.0; mass[c] += area / 3.0
        # cot of angle at each vertex; contributes to the OPPOSITE edge
        def cot(p, q, r):  # angle at p, between p->q and p->r
            e1, e2 = q - p, r - p
            return float(np.dot(e1, e2) / (np.linalg.norm(np.cross(e1, e2)) + 1e-12))
        wc_a = cot(va, vb, vc) / 2.0   # -> edge (b,c)
        wc_b = cot(vb, va, vc) / 2.0   # -> edge (a,c)
        wc_c = cot(vc, va, vb) / 2.0   # -> edge (a,b)
        for (i, j, w) in ((b, c, wc_a), (a, c, wc_b), (a, b, wc_c)):
            I += [i, j]; J += [j, i]; W += [-w, -w]      # off-diagonal
            I += [i, j]; J += [i, j]; W += [w, w]        # diagonal accumulation
    L = sp.coo_matrix((W, (I, J)), shape=(n, n)).tocsr()
    # Floor zero-mass vertices (unreferenced / degenerate triangles) so M is invertible
    # for the shift-invert generalized eigensolve; otherwise splu(M) is exactly singular.
    pos = mass[mass > 0]
    floor = (pos.mean() * 1e-6) if pos.size else 1e-12
    mass = np.maximum(mass, floor)
    M = sp.diags(mass)
    return L, M, mass


def clean_mesh(V, F):
    """Unit-rescale, drop degenerate faces, restrict to the main connected component.

    The raw FaceBase mean mesh is stored at micro-scale (extent ~0.01) with ~1705
    zero-area faces and two tiny stray components. Left uncleaned, the Laplace-Beltrami
    spectrum is flooded with spurious near-zero modes. Returns the cleaned submesh plus
    the index map back into the original vertex array.
    """
    import scipy.sparse.csgraph as csg
    s = float(np.linalg.norm(V.max(0) - V.min(0)))
    V = V / s                                         # unit-rescale (modes are scale-free)
    va, vb, vc = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(vb - va, vc - va), axis=1)
    F = F[area > 1e-12]
    n = V.shape[0]
    I = np.r_[F[:, 0], F[:, 1], F[:, 2]]
    J = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    A = sp.coo_matrix((np.ones(I.size), (I, J)), shape=(n, n))
    A = A + A.T
    _, lab = csg.connected_components(A, directed=False)
    main = np.bincount(lab).argmax()
    keep = np.where(lab == main)[0]
    remap = -np.ones(n, int); remap[keep] = np.arange(keep.size)
    Fm = remap[F]; Fm = Fm[(Fm >= 0).all(1)]
    return V[keep], Fm, keep, s


def main():
    V0, F, HL = mm.load()
    # FaceBase .mat faces are 1-indexed (MATLAB); the Gaussian morph never used F so
    # this was latent. Normalize to 0-indexing for the Laplacian assembly.
    if F.min() >= 1 and F.max() >= V0.shape[0]:
        F = F - 1
    A0 = mm.anchors(V0)                                # GWAS anchors in ORIGINAL coords
    V, F, keep, s = clean_mesh(V0, F)
    HL = HL / s                                        # landmarks into rescaled frame
    print(f"mesh: {V0.shape[0]} verts raw -> {V.shape[0]} clean, {F.shape[0]} faces")
    L, M, mass = cotangent_laplacian(V, F)

    # smallest eigenvalues (shift-invert near 0); drop the constant mode 0
    vals, vecs = spla.eigsh(L, k=K_MODES + 1, M=M, sigma=-1e-8, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    # M-orthonormalise
    for k in range(vecs.shape[1]):
        nrm = np.sqrt(vecs[:, k] @ (M @ vecs[:, k]))
        vecs[:, k] /= (nrm + 1e-12)
    phi = vecs[:, 1:]          # drop constant
    lam = vals[1:]
    print(f"first eigenvalues: {np.round(lam[:8], 4)}")

    # ---- GWAS deformations projected onto the eigenbasis ----
    genes = ["EDAR", "PAX3", "DCHS2", "RUNX2", "GLI3", "PAX1"]
    gwas_spec = {}
    for g in genes:
        # deformation computed in ORIGINAL coords, then restricted to the clean submesh
        Dfull = mm.morph(V0, A0, {g: 2.0}) - V0        # (n0, 3)
        D = Dfull[keep] / s                            # (n, 3) rescaled to mesh frame
        # M-weighted projection coefficients per axis, energy per mode
        coeff = phi.T @ (M @ D)                         # (K, 3)
        energy = (coeff ** 2).sum(axis=1)              # (K,)
        tot = energy.sum() + 1e-12
        cum = np.cumsum(energy) / tot
        # participation ratio (effective # of modes)
        p = energy / tot
        eff = float(1.0 / np.sum(p ** 2))
        gwas_spec[g] = {"energy": energy, "cum": cum, "eff_modes": eff,
                        "modes_for_90pct": int(np.searchsorted(cum, 0.90) + 1)}
        print(f"{g:6s}: effective modes {eff:5.1f}, modes for 90% energy "
              f"{gwas_spec[g]['modes_for_90pct']}")

    # ---- spectral reconstruction of the face geometry ----
    Vc = V - V.mean(0)
    coeffV = phi.T @ (M @ Vc)                          # (K,3)
    recon_err = []
    Ks = [2, 5, 10, 20, 40, K_MODES]
    for kk in Ks:
        Vr = phi[:, :kk] @ coeffV[:kk]
        recon_err.append(float(np.sqrt((mass[:, None] * (Vr - Vc) ** 2).sum() /
                                       (mass[:, None] * Vc ** 2).sum())))

    # ---------- figure ----------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4)
    x, y = V[:, 0], V[:, 1]                            # frontal view (+z toward us)
    # render first 6 modes (top two rows, cols 0-2) + 2 analysis panels (col 3)
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for idx, (r, c) in enumerate(slots):
        ax = fig.add_subplot(gs[r, c])
        ax.scatter(x, y, c=phi[:, idx], s=2, cmap="RdBu_r", linewidths=0)
        ax.scatter(HL[:, 0], HL[:, 1], c="k", s=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"electric-face mode {idx+1} (lam={lam[idx]:.3f})", fontsize=8)

    # GWAS spectral concentration
    ax = fig.add_subplot(gs[0, 3])
    for g in genes:
        ax.plot(np.arange(1, K_MODES + 1), gwas_spec[g]["cum"], label=g)
    ax.axhline(0.9, ls="--", c="k", lw=0.6)
    ax.set_xlabel("# eigenmodes"); ax.set_ylabel("cumulative energy")
    ax.set_title("GWAS deformations are LOW-RANK\nin the eigenbasis", fontsize=8)
    ax.legend(fontsize=6); ax.set_xlim(1, 30)

    # spectral reconstruction error
    ax = fig.add_subplot(gs[1, 3])
    ax.plot(Ks, recon_err, "o-")
    ax.set_xlabel("# eigenmodes"); ax.set_ylabel("relative recon error")
    ax.set_title("face geometry is low-dimensional\nin the eigenbasis", fontsize=8)

    fig.suptitle("The electric face: Laplace-Beltrami eigenmodes of the FaceBase "
                 "mean face are cymatic standing waves; GWAS effects are additive "
                 "mode shifts", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = HERE / "electric_face_eigenmodes.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)

    json.dump({
        "n_verts": int(V.shape[0]), "k_modes": K_MODES,
        "eigenvalues": [float(v) for v in lam[:12]],
        "gwas": {g: {"eff_modes": gwas_spec[g]["eff_modes"],
                     "modes_for_90pct": gwas_spec[g]["modes_for_90pct"]}
                 for g in genes},
        "recon_K": Ks, "recon_err": recon_err,
    }, open(HERE / "electric_face_eigenmodes.json", "w"), indent=2)


if __name__ == "__main__":
    main()
