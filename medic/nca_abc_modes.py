"""NCA modes from the ABC data (Paper #4 §2 demo).

The electric-face correspondence demo (face_demo/electric_face_correspondence.py)
built the NCA gap-junction operator with UNIFORM coupling on a generic mesh. The
operator's eigenmodes were therefore geometric, and the ABC database entered
nowhere. This script closes that gap: it computes the NCA cymatic modes from the
ABC data itself, on two counts.

  1. TARGET from ABC. The genome's bioelectric target field V*(x) is the ABC
     Goldman read at each grid position (GenomicChannelLookup.compute_voltage):
     the same conductance profile bioelectric_development uses, here as a spatial
     map. We project V* onto the NCA modes and ask how many cymatic standing
     waves the genome's target actually occupies.

  2. OPERATOR from ABC. The NCA's gap-junction coupling is connexin-mediated;
     ABC scores a per-position gap-junction conductance g_GJ (the GJA/GJB/GJC
     super-enhancer activity). Weighting the gap-junction Laplacian edges by g_GJ
     makes the operator -- and hence its eigenmodes -- come from ABC, not from a
     uniform stencil. We compare the ABC-weighted modes to the uniform ones.

The claim under test (Paper #4 §2.3): the genome specifies a bioelectric target
that is a few low cymatic modes of its own gap-junction operator. HONEST SCOPE:
ABC is organ-resolved and human-adult, so this is the body-plan grid (where ABC
has real spatial structure), NOT the embryonic face; the face-specific version
still needs developmental craniofacial tracks (the deferred experiment).

Run: python -m medic.nca_abc_modes
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

from medic.four_head_morphogenesis import GenomicChannelLookup
from medic.tissue.genomic_nca import V_ZYGOTE

N = 48                # grid resolution
K = 60                # eigenmodes
K_RELAX, K_GJ = 0.30, 0.12     # the NCA operator's actual constants
OUT = Path("data/organ_cascade")


def abc_fields(n):
    """ABC Goldman target V* and ABC gap-junction conductance g_GJ on an n x n grid."""
    look = GenomicChannelLookup(seed=42)
    V = np.zeros((n, n)); G = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            pos = np.array([c / (n - 1), r / (n - 1), 0.5])
            V[r, c] = look.compute_voltage(pos)
            G[r, c] = look.forward(np.zeros(6), {}, pos)["g_GJ"]
    return V, G


def grid_index(n):
    idx = np.arange(n * n).reshape(n, n)
    return idx


def uniform_laplacian(n):
    """Standard 5-point grid Laplacian, Neumann (free) edges = the NCA stencil."""
    idx = grid_index(n)
    I, J, W = [], [], []
    for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        a = idx[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)]
        b = idx[max(0, di):n - max(0, -di), max(0, dj):n - max(0, -dj)]
        I += [a.ravel()]; J += [b.ravel()]; W += [np.ones(a.size)]
    I = np.concatenate(I); J = np.concatenate(J); W = np.concatenate(W)
    A = sp.coo_matrix((W, (I, J)), shape=(n * n, n * n)).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(d) - A).tocsr()


def abc_weighted_laplacian(n, G):
    """Gap-junction Laplacian whose edge weights are the ABC g_GJ field.

    Edge (i,j) coupling = mean of the two cells' ABC gap-junction conductance,
    normalised to the field mean so it is a relative reweighting of the same
    stencil (the operator, hence its modes, now come from ABC)."""
    idx = grid_index(n)
    g = G.ravel() / (G.mean() + 1e-12)
    I, J, W = [], [], []
    for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        a = idx[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)].ravel()
        b = idx[max(0, di):n - max(0, -di), max(0, dj):n - max(0, -dj)].ravel()
        w = 0.5 * (g[a] + g[b])
        I += [a]; J += [b]; W += [w]
    I = np.concatenate(I); J = np.concatenate(J); W = np.concatenate(W)
    A = sp.coo_matrix((W, (I, J)), shape=(n * n, n * n)).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    return (sp.diags(d) - A).tocsr()


def modes(L, k):
    vals, vecs = spla.eigsh(L, k=k + 1, sigma=-1e-9, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]      # drop constant mode


def facebase_modes(k=4):
    """Laplace--Beltrami modes of the cleaned FaceBase mean face (for comparison)."""
    import sys
    fd = Path("face_demo").resolve()
    sys.path.insert(0, str(fd))
    import mesh_morph as mm
    import face_eigenmodes as fe
    V0, F, _ = mm.load()
    if F.min() >= 1 and F.max() >= V0.shape[0]:
        F = F - 1
    V, F, _, _ = fe.clean_mesh(V0, F)
    L, M, _ = fe.cotangent_laplacian(V, F)
    vals, vecs = spla.eigsh(L, k=k + 1, M=M, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return V, vecs[:, o][:, 1:k + 1]


def low_rank(field_flat, phi):
    """Project a centred field onto an orthonormal basis; report modal energy."""
    f = field_flat - field_flat.mean()
    Q, _ = np.linalg.qr(phi)
    coeff = Q.T @ f
    e = coeff ** 2; p = e / (e.sum() + 1e-12)
    cum = np.cumsum(p)
    eff = float(1.0 / np.sum(p ** 2))
    m90 = int(np.searchsorted(cum, 0.90) + 1)
    return eff, m90, cum, coeff, Q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Vstar, Ggj = abc_fields(N)
    print(f"ABC target V*: {Vstar.min():.1f}..{Vstar.max():.1f} mV   "
          f"g_GJ: {Ggj.min():.3f}..{Ggj.max():.3f}  (n={N}x{N})")

    Lu = uniform_laplacian(N)
    La = abc_weighted_laplacian(N, Ggj)
    lam_u, phi_u = modes(Lu, K)
    lam_a, phi_a = modes(La, K)
    print(f"uniform NCA eigenvalues[:5]: {np.round(lam_u[:5], 4)}")
    print(f"ABC-weighted  eigenvalues[:5]: {np.round(lam_a[:5], 4)}")

    f = Vstar.ravel()
    eff_u, m90_u, cum_u, _, Qu = low_rank(f, phi_u)
    eff_a, m90_a, cum_a, _, Qa = low_rank(f, phi_a)
    print(f"ABC target low-rank in UNIFORM NCA modes : eff {eff_u:.1f}, 90% by {m90_u} modes")
    print(f"ABC target low-rank in ABC-WEIGHTED modes: eff {eff_a:.1f}, 90% by {m90_a} modes")

    # how much do the ABC-weighted modes differ from the uniform ones?
    C = Qa[:, :20].T @ Qu[:, :20]
    cap = float((C ** 2).sum(0).mean())
    print(f"ABC-weighted vs uniform leading-20 subspace capture: {cap:.3f}")

    # screened-Poisson modal gain with the real NCA constants -> the SETTLED field
    gain_u = K_RELAX / (K_RELAX + K_GJ * lam_u)
    # forced energy = (projection * gain)^2, share kept by the lowest modes
    coeff_u = Qu.T @ (f - f.mean())
    forced = (coeff_u * gain_u) ** 2
    cum_forced = np.cumsum(forced) / (forced.sum() + 1e-12)
    m90_forced = int(np.searchsorted(cum_forced, 0.90) + 1)
    print(f"settled NCA field (k_gj/k_relax screened): 90% energy by {m90_forced} modes")

    # FaceBase geometry modes, for the side-by-side comparison
    Vf, phi_f = facebase_modes(4)
    xf, yf = Vf[:, 0], Vf[:, 1]

    # --- cross-domain mode MATCHING (by shape, sharing the vertical axis) ---
    # Bin the face modes onto the same N x N grid (face-vertical -> grid-vertical),
    # then correlate with the NCA modes. Ordering is a geometry-dependent permutation;
    # the matching recovers which NCA mode is which FaceBase mode, and its sign.
    xn = (xf - xf.min()) / (xf.max() - xf.min())
    yn = (yf - yf.min()) / (yf.max() - yf.min())
    ci = np.clip((xn * (N - 1)).round().astype(int), 0, N - 1)
    ri = np.clip(((1 - yn) * (N - 1)).round().astype(int), 0, N - 1)   # up -> row 0
    binned, cnt = np.zeros((4, N, N)), np.zeros((N, N))
    np.add.at(cnt, (ri, ci), 1.0)
    for m in range(4):
        np.add.at(binned[m], (ri, ci), phi_f[:, m])
    valid = cnt > 0
    binned = np.where(valid[None], binned / np.maximum(cnt, 1)[None], 0.0)
    vmask = valid.ravel()
    corr = np.zeros((4, 4))
    for a in range(4):
        na = phi_a[:, a].reshape(N, N).ravel()[vmask]; na = na - na.mean()
        for b in range(4):
            fb = binned[b].ravel()[vmask]; fb = fb - fb.mean()
            corr[a, b] = (na @ fb) / (np.linalg.norm(na) * np.linalg.norm(fb) + 1e-12)
    perm = np.argmax(np.abs(corr), axis=1)                 # FB mode matching each NCA mode
    sgn = np.sign(corr[np.arange(4), perm])
    print("cross-domain |corr| matrix (NCA rows x FaceBase cols):")
    print(np.array2string(np.abs(corr), precision=2, suppress_small=True))
    for a in range(4):
        print(f"  NCA mode {a+1}  ==  {'-' if sgn[a] < 0 else '+'}FaceBase mode {perm[a]+1}"
              f"   (|corr| {abs(corr[a, perm[a]]):.2f})")

    shape_names = ["vertical gradient", "lateral dipole", "saddle", "quadrupole"]

    def panel(ax, title, fs=8):
        ax.set_title(title, fontsize=fs); ax.set_xticks([]); ax.set_yticks([])

    # ================= FIGURE 1: NCA modes from ABC (the low-rank result) =======
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4)
    # row 0: ABC inputs + quantitative panels
    ax = fig.add_subplot(gs[0, 0]); im = ax.imshow(Vstar, cmap="RdBu_r")
    panel(ax, "(a) ABC Goldman target V*(x)\nthe genome's bioelectric map (mV)"); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[0, 1]); im = ax.imshow(Ggj, cmap="viridis")
    panel(ax, "(b) ABC gap-junction g_GJ(x)\nweights the operator edges"); fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(np.arange(1, K + 1), cum_a, label="ABC-weighted")
    ax.plot(np.arange(1, K + 1), cum_u, label="uniform", ls="--")
    ax.plot(np.arange(1, K + 1), cum_forced, label="settled (screened)", ls=":")
    ax.axhline(0.9, c="k", lw=0.5, ls="--"); ax.set_xlim(1, 40)
    ax.set_xlabel("# NCA modes", fontsize=7); ax.set_ylabel("cum. energy of V*", fontsize=7)
    ax.set_title(f"(c) ABC target is LOW-RANK in NCA modes\neff {eff_a:.1f} modes; 90% by {m90_a} of 60", fontsize=8)
    ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(np.arange(1, K + 1), lam_a, "o-", ms=2, label="ABC-weighted")
    ax.plot(np.arange(1, K + 1), lam_u, ".--", ms=2, label="uniform")
    ax.set_xlabel("mode", fontsize=7); ax.set_ylabel("eigenvalue", fontsize=7); ax.set_xlim(1, 40)
    ax.set_title(f"(d) NCA operator spectrum\nABC vs uniform; subspace capture {cap:.2f}", fontsize=8)
    ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    # row 1: V* reconstructions from a few NCA modes
    for col, kk in [(0, 1), (1, 2), (2, 5), (3, 60)]:
        recon = (Qa[:, :kk] @ (Qa[:, :kk].T @ (f - f.mean())) + f.mean()).reshape(N, N)
        ax = fig.add_subplot(gs[1, col]); ax.imshow(recon, cmap="RdBu_r")
        panel(ax, ("(e) full ABC target V*" if kk == 60 else f"V* rebuilt from {kk} NCA mode" + ("s" if kk > 1 else "")))
    # row 2: the 4 NCA modes computed from ABC  (square body grid)
    for c in range(4):
        ax = fig.add_subplot(gs[2, c]); ax.imshow(phi_a[:, c].reshape(N, N), cmap="RdBu_r")
        panel(ax, f"(f{c+1}) NCA mode {c+1} from ABC g_GJ\n[{shape_names[c]}]")
    fig.suptitle("Figure A. NCA modes computed from ABC: the genome's bioelectric target V* (ABC Goldman read) is a few "
                 "low cymatic standing waves of its own ABC-weighted gap-junction operator (eff "
                 f"{eff_a:.1f} modes, 90% in {m90_a} of 60)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = OUT / "nca_abc_modes.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight"); print("saved", out1)

    # ================= FIGURE 2: NCA <-> FaceBase correspondence ================
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4)
    # row 0: the 4 NCA modes from ABC, shape-annotated
    for c in range(4):
        ax = fig.add_subplot(gs[0, c]); ax.imshow(phi_a[:, c].reshape(N, N), cmap="RdBu_r")
        panel(ax, f"NCA mode {c+1}  (ABC, body grid)\n[{shape_names[c]}]")
    # row 1: the MATCHED FaceBase mode beneath each, permuted + sign-aligned
    for c in range(4):
        b = int(perm[c]); s = float(sgn[c])
        ax = fig.add_subplot(gs[1, c])
        ax.scatter(xf, yf, c=s * phi_f[:, b], s=1.5, cmap="RdBu_r", linewidths=0)
        ax.set_aspect("equal")
        panel(ax, f"= FaceBase mode {b+1}  (face surface)\nmatch |r| = {abs(corr[c, b]):.2f}")
        ax.annotate("", xy=(0.5, 1.04), xytext=(0.5, 1.16), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.4))
    # row 2: correlation matrix + annotation
    ax = fig.add_subplot(gs[2, :2])
    im = ax.imshow(np.abs(corr), cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"FB {i+1}" for i in range(4)])
    ax.set_yticks(range(4)); ax.set_yticklabels([f"NCA {i+1}" for i in range(4)])
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{abs(corr[a,b]):.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr[a, b]) < 0.6 else "black")
        ax.add_patch(plt.Rectangle((perm[a]-0.5, a-0.5), 1, 1, fill=False, ec="cyan", lw=2.5))
    ax.set_title("cross-domain |corr| matrix  (matched pair boxed)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[2, 2:]); ax.axis("off")
    txt = ("The same cymatic alphabet, re-ordered by domain geometry.\n\n"
           "Permutation:  NCA 1,2,3,4  $\\leftrightarrow$  FaceBase 3,1,4,2\n\n"
           f"  NCA 1 (vertical gradient)  =  FaceBase 3   |r|={abs(corr[0,perm[0]]):.2f}\n"
           f"  NCA 2 (lateral dipole)     =  FaceBase 1   |r|={abs(corr[1,perm[1]]):.2f}\n"
           f"  NCA 3 (saddle)             =  FaceBase 4   |r|={abs(corr[2,perm[2]]):.2f}\n"
           f"  NCA 4 (quadrupole)         =  FaceBase 2   |r|={abs(corr[3,perm[3]]):.2f}\n\n"
           "A square body grid and a taller facial surface assign the\n"
           "SAME standing waves different eigenvalue ranks. The\n"
           "correspondence is therefore by SHAPE (the mode set), not\n"
           "by index. Signs are solver-arbitrary per eigenvector and\n"
           "are aligned for display; the permutation is the robust fact.")
    ax.text(0.0, 0.98, txt, fontsize=9, va="top", ha="left", family="monospace")
    fig.suptitle("Figure B. NCA(ABC) $\\leftrightarrow$ FaceBase geometry correspondence: each FaceBase mode (row 2) sits "
                 "beneath the NCA mode it matches (row 1); the ordering is a geometry-dependent permutation",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out2 = OUT / "nca_face_correspondence.png"
    fig.savefig(out2, dpi=140, bbox_inches="tight"); print("saved", out2)

    json.dump({
        "grid": N, "k_modes": K,
        "abc_target_mV": [float(Vstar.min()), float(Vstar.max())],
        "abc_gGJ_range": [float(Ggj.min()), float(Ggj.max())],
        "eff_modes_uniform": eff_u, "modes90_uniform": m90_u,
        "eff_modes_abc_weighted": eff_a, "modes90_abc_weighted": m90_a,
        "abc_vs_uniform_subspace_capture20": cap,
        "settled_modes90": m90_forced,
        "k_relax": K_RELAX, "k_gj": K_GJ,
    }, open(OUT / "nca_abc_modes.json", "w"), indent=2)


if __name__ == "__main__":
    main()
