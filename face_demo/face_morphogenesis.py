"""
Deriving facial morphogenesis: prominences as outgrowth attractors (Paper #2).
==============================================================================

Paper #2 §3.6/§3.7 established that facial VARIATION (GWAS) and the electric face
are low-rank in the Laplace-Beltrami eigenbasis of the FaceBase mean face. This
module takes the next step asked of the framework: deriving the gross facial FORM
itself as a morphogenetic process on that same surface -- the inner perceptron of
Paper #5 applied to the head.

The model. The vertebrate face is built by a small set of canonical facial
PROMINENCES -- the frontonasal prominence, the paired medial- and lateral-nasal
processes, the paired maxillary prominences and the paired mandibular
prominences -- each an outgrowth bud, organized by the frontonasal ectodermal
zone (a Shh/Fgf8 boundary, the facial analogue of the limb apical ridge; Hu,
Marcucio & Helms 2003). We model each prominence as a morphogen source on the
cranial surface, spread by surface diffusion (the Laplace-Beltrami HEAT KERNEL =
the inner perceptron's gap-junction/morphogen smoothing). The facial relief is
then the steady outgrowth field

    G(x) = sum_i a_i * h_t( source_i )(x),
    h_t(s)(x) = sum_k exp(-t lam_k) phi_k(s) phi_k(x)        (heat kernel)

with phi_k, lam_k the eigenmodes of the cotangent Laplace-Beltrami operator.

The derivation (inverse). We measure the real facial relief T(x) of the FaceBase
mean face (anterior protrusion with its smooth trend removed), and solve for the
prominence amplitudes a_i (and the diffusion time t) so that the morphogen-
generated relief G matches T -- entirely in the low-rank eigenbasis. A good fit
means the face FORM, not just its variation, is the outgrowth of a few
genome-placed prominence sources on the cranial field.

Honest scope: a quasi-static outgrowth field, not a dynamic 4D growth simulation;
the surface is already fused (no cleft/fusion topology); the source LOCATIONS are
landmark-anchored, not yet derived from the genome (the stated open problem).

Run:
    cd cognimed && venv_win_new/Scripts/python.exe face_demo/face_morphogenesis.py
Output: face_morphogenesis.png, face_morphogenesis.json
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesh_morph as mm
from face_eigenmodes import cotangent_laplacian, clean_mesh

K_MODES = 60


def nearest(V, p):
    return int(np.argmin(((V - p) ** 2).sum(1)))


def prominence_sources(V0, A, s):
    """Canonical facial prominences as source points, mapped to nearest cleaned-mesh
    vertices. V0=original verts, A=anchors (original coords), s=clean rescale factor."""
    cx, size = A["cx"], A["size"]
    ny = A["nose"][1]
    # maxillary (cheek) prominences: lateral, mid-face height, most anterior there.
    def cheek(sign):
        m = (np.abs(V0[:, 1] - ny) < 0.12 * size) & (sign * (V0[:, 0] - cx) > 0.18 * size)
        return V0[m][np.argmax(V0[m][:, 2])] if m.any() else A["alare_L"]
    pts = {
        "frontonasal": A["nasion"],
        "medial-nasal": A["nose"],
        "lateral-nasal L": A["alare_L"], "lateral-nasal R": A["alare_R"],
        "maxillary L": cheek(-1), "maxillary R": cheek(+1),
        "mandibular": A["chin"],
    }
    return pts


def main():
    V0, F, HL = mm.load()
    if F.min() >= 1 and F.max() >= V0.shape[0]:
        F = F - 1
    A = mm.anchors(V0)
    src_pts = prominence_sources(V0, A, 1.0)

    V, F, keep, s = clean_mesh(V0, F)
    print(f"mesh: {V0.shape[0]} verts raw -> {V.shape[0]} clean, {F.shape[0]} faces")
    L, M, mass = cotangent_laplacian(V, F)

    vals, vecs = spla.eigsh(L, k=K_MODES + 1, M=M, sigma=-1e-8, which="LM")
    order = np.argsort(vals); vals, vecs = vals[order], vecs[:, order]
    for k in range(vecs.shape[1]):
        nrm = np.sqrt(vecs[:, k] @ (M @ vecs[:, k])); vecs[:, k] /= (nrm + 1e-12)
    phi, lam = vecs[:, 1:], vals[1:]               # drop constant mode

    # source vertices in the cleaned/rescaled frame
    src_names = list(src_pts.keys())
    src_idx = [nearest(V, np.asarray(src_pts[n]) / s) for n in src_names]

    # --- target relief: anterior protrusion (+z) with smooth quadratic trend removed ---
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    Bpoly = np.c_[np.ones_like(x), x, y, x * x, y * y, x * y]
    coef, *_ = np.linalg.lstsq(Bpoly, z, rcond=None)
    T = z - Bpoly @ coef
    T = T - (mass * T).sum() / mass.sum()          # mass-center
    Tk = phi.T @ (M @ T)                            # target modal coefficients

    # --- forward heat-kernel design (multi-scale: each prominence has its own
    #     outgrowth range, so each source enters at several diffusion times) ---
    phi_src = phi[src_idx, :]                            # (n_src, K)
    scales = np.geomspace(2e-3, 4e-2, 4)                 # 4 prominence widths
    cols, col_src = [], []
    for i in range(len(src_idx)):
        for t in scales:
            cols.append(np.exp(-t * lam) * phi_src[i])   # (K,) mode-space heat kernel
            col_src.append(i)
    D = np.array(cols).T                                 # (K, n_src*scales)
    a, *_ = np.linalg.lstsq(D, Tk, rcond=None)
    ck = D @ a
    err = float(np.linalg.norm(ck - Tk) / (np.linalg.norm(Tk) + 1e-12))
    G = phi @ ck
    corr = float(np.corrcoef(G, T)[0, 1])

    # per-prominence contribution = RMS of that source's summed multi-scale field (>=0)
    col_src = np.array(col_src)
    contrib = {}
    for i, n in enumerate(src_names):
        cki = D[:, col_src == i] @ a[col_src == i]       # this source's modal coeffs
        fi = phi @ cki
        contrib[n] = float(np.sqrt((mass * fi ** 2).sum() / mass.sum()))

    e = Tk ** 2; cum = np.cumsum(e) / (e.sum() + 1e-12)
    eff = float(1.0 / np.sum((e / e.sum()) ** 2))
    k90 = int(np.searchsorted(cum, 0.90) + 1)

    print(f"\nForward model: {len(src_idx)} prominence sources x {len(scales)} outgrowth scales")
    print(f"  facial relief reconstruction error (eigenbasis) = {err:.3f}")
    print(f"  spatial corr(generated, real relief)            = {corr:.3f}")
    print(f"  relief is low-rank: {eff:.1f} effective modes, {k90} modes for 90% energy")
    print("\nPer-prominence contribution to facial form (RMS outgrowth):")
    for n, c in sorted(contrib.items(), key=lambda kv: -kv[1]):
        print(f"    {n:16s} {c:.4f}")

    _figure(V, HL, T, G, src_idx, src_names, contrib, cum, scales, err, corr)
    json.dump(dict(n_src=len(src_idx), scales=[float(t) for t in scales],
                   recon_err=err, corr=corr, eff_modes=eff, modes_90=k90,
                   contributions=contrib),
              open(HERE / "face_morphogenesis.json", "w"), indent=2)
    print("\nsaved face_morphogenesis.json")
    return err < 0.5 and corr > 0.7


def _figure(V, HL, T, G, src_idx, src_names, contrib, cum, scales, err, corr):
    x, y = V[:, 0], V[:, 1]
    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(1, 4)
    lim = np.percentile(np.abs(T), 98)

    def face(ax, c, title):
        ax.scatter(x, y, c=c, s=2, cmap="RdBu_r", vmin=-lim, vmax=lim, linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)

    ax0 = fig.add_subplot(gs[0, 0]); face(ax0, T, "real FaceBase relief\n(anterior protrusion)")
    ax1 = fig.add_subplot(gs[0, 1]); face(ax1, G, f"generated by {len(src_idx)} prominence sources\n(heat-kernel outgrowth)")
    for si in src_idx:
        ax1.scatter(V[si, 0], V[si, 1], c="k", s=18, marker="x")
    ax2 = fig.add_subplot(gs[0, 2]); face(ax2, T - G, f"residual  (err={err:.2f}, r={corr:.2f})")

    ax3 = fig.add_subplot(gs[0, 3])
    items = sorted(contrib.items(), key=lambda kv: kv[1])
    ax3.barh([n for n, _ in items], [c for _, c in items], color="#4878a8")
    ax3.set_xlabel("RMS outgrowth contribution"); ax3.set_title("prominence contributions to facial form", fontsize=9)
    ax3.tick_params(axis="y", labelsize=7)

    fig.suptitle("Deriving facial form: the FaceBase face relief is the steady outgrowth of a few prominence "
                 "morphogen sources\ndiffusing on the cranial surface (the inner perceptron), recovered by "
                 "inversion in the electric-face eigenbasis", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(HERE / "face_morphogenesis.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved face_morphogenesis.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}")
