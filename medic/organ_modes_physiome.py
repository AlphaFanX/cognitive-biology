"""Retiring Paper #4's "idealised / generic-smooth-surface" caveat for the heart:
run the field-form mode decomposition with a REAL, VALIDATED cardiac monodomain
operator instead of an isotropic graph Laplacian.

WHAT CHANGES vs medic/organ_modes.py
------------------------------------
organ_modes.py built the "field" operator as an UNWEIGHTED binary graph Laplacian
(sp.diags(d) - A). That is isotropic and physically generic -- the caveat rightly
warned its low modes "are partly generic to smooth surfaces."

Here the field operator is the actual cardiac MONODOMAIN operator  div(D grad V),
assembled by linear FEM on the same surface, with:

  * VALIDATED CONDUCTIVITIES -- the Niederer et al. (2011) N-version cardiac
    tissue benchmark (Phil. Trans. R. Soc. A), Table 3:
        intra  sigma_il=0.17,  sigma_it=0.019  S/m
        extra  sigma_el=0.62,  sigma_et=0.24   S/m
        monodomain sigma = sigma_i sigma_e / (sigma_i + sigma_e)
        -> sigma_par (along fibre)  = 0.17*0.62/(0.17+0.62) = 0.1334 S/m
        -> sigma_trans (cross)      = 0.019*0.24/(0.019+0.24) = 0.0176 S/m
        -> anisotropy ratio         ~ 7.58 : 1
    (Physiome cardiac-modelling lineage: Niederer/Noble, Oxford.)

  * FIBRE ARCHITECTURE -- a Streeter helical field on the ventricular surface:
    fibre = cos(alpha) e_circ + sin(alpha) e_long, helix angle alpha swept over the
    physiological transmural range (endo +60 deg ... mid 0 ... epi -60 deg).

Because the fibre-weighted operator is physically SPECIFIC (not generic to smooth
surfaces), this is the honest test of the caveat's substance:
  (Q1) does mode 1 of the real cardiac operator stay the apex-base axis, or does
       fast fibre (circumferential) conduction flip it?
  (Q2) does the geometry <-> field low-mode correspondence survive real anisotropy?

Sanity check built in: with sigma_par = sigma_trans the FEM stiffness must reduce to
the cotangent Laplacian used for the geometry operator.

Run: python -m medic.organ_modes_physiome
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str((Path("face_demo")).resolve()))
import face_eigenmodes as fe
from medic.organ_modes import lv_mesh, gapjunction_laplacian, low_modes

K = 40
OUT = Path("data/organ_cascade")

# --- Niederer et al. (2011) N-version benchmark, monodomain reduction --------
SIG_IL, SIG_IT = 0.17, 0.019      # intracellular  long / trans  (S/m)
SIG_EL, SIG_ET = 0.62, 0.24       # extracellular  long / trans  (S/m)
SIG_PAR   = SIG_IL * SIG_EL / (SIG_IL + SIG_EL)      # 0.1334 S/m
SIG_TRANS = SIG_IT * SIG_ET / (SIG_IT + SIG_ET)      # 0.0176 S/m
ANISO = SIG_PAR / SIG_TRANS                          # ~7.58


def fibre_field(centroid, normal, helix_deg):
    """Streeter fibre at a surface point: helix_deg between circumferential and
    longitudinal (apex-base). Returns a unit vector in the tangent plane."""
    z = np.array([0.0, 0.0, 1.0])
    e_long = z - np.dot(z, normal) * normal            # apex->base, in tangent plane
    nl = np.linalg.norm(e_long)
    if nl < 1e-9:                                       # at apex pole: pick any tangent
        e_long = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
        nl = np.linalg.norm(e_long)
    e_long /= nl
    e_circ = np.cross(normal, e_long)                  # circumferential, in tangent plane
    e_circ /= (np.linalg.norm(e_circ) + 1e-12)
    a = np.radians(helix_deg)
    f = np.cos(a) * e_circ + np.sin(a) * e_long
    return f / (np.linalg.norm(f) + 1e-12)


def monodomain_stiffness(V, F, helix_deg, sig_par=SIG_PAR, sig_trans=SIG_TRANS):
    """Linear-FEM anisotropic stiffness K_ij = sum_T area_T grad(phi_i)^T D grad(phi_j),
    D = sig_trans I + (sig_par - sig_trans) f f^T, f = Streeter fibre on triangle T."""
    n = V.shape[0]
    rows, cols, vals = [], [], []
    for tri in F:
        p = V[tri]
        cross = np.cross(p[1] - p[0], p[2] - p[0])
        a2 = np.linalg.norm(cross)                      # 2*area
        if a2 < 1e-12:
            continue
        nrm = cross / a2
        area = 0.5 * a2
        # in-plane gradients of the linear hat functions (sum to zero)
        g = np.array([np.cross(nrm, p[2] - p[1]),
                      np.cross(nrm, p[0] - p[2]),
                      np.cross(nrm, p[1] - p[0])]) / a2
        c = p.mean(0)
        f = fibre_field(c, nrm, helix_deg)
        D = sig_trans * np.eye(3) + (sig_par - sig_trans) * np.outer(f, f)
        Ke = area * (g @ D @ g.T)                       # 3x3 local stiffness
        for a in range(3):
            for b in range(3):
                rows.append(tri[a]); cols.append(tri[b]); vals.append(Ke[a, b])
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def gen_low_modes(L, M, k):
    """Lowest k non-trivial generalized eigenmodes of (L, M)."""
    vals, vecs = spla.eigsh(L, k=k + 1, M=M, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]


def axis_alignment(mode, coord):
    """|pearson| of a mode against a coordinate axis (apex-base = z, etc.)."""
    r, _ = pearsonr(mode, coord)
    return abs(r)


def subspace(Qref, phi):
    Q, _ = np.linalg.qr(phi)
    C = Q.T @ Qref
    return (C ** 2).sum(0)                              # per-ref-mode capture in [0,1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V, F, UV = lv_mesh()
    z = V[:, 2]                                         # apex(0) -> base
    circ = np.arctan2(V[:, 1], V[:, 0])                # circumferential angle
    print(f"LV surface: {V.shape[0]} verts, {F.shape[0]} faces")
    print(f"Niederer monodomain: sig_par={SIG_PAR:.4f}  sig_trans={SIG_TRANS:.4f}  "
          f"anisotropy={ANISO:.2f}:1")

    # geometry operator (isotropic cotangent LB) -- the "form"
    Lcot, M, _ = fe.cotangent_laplacian(V, F)
    lam_g, phi_g = gen_low_modes(Lcot, M, K)
    Qg, _ = np.linalg.qr(phi_g)

    # sanity: FEM stiffness with D=I must match the cotangent stiffness
    Kiso = monodomain_stiffness(V, F, 0.0, 1.0, 1.0)
    rel = spla.norm(Kiso - Lcot) / spla.norm(Lcot)
    print(f"[sanity] ||K_iso - L_cot|| / ||L_cot|| = {rel:.2e}  "
          f"({'OK' if rel < 1e-6 else 'CHECK'})")

    # old baseline: binary graph Laplacian (what organ_modes.py used)
    Lgj = gapjunction_laplacian(V, F)
    lam_b, phi_b = low_modes(Lgj, None, K, False)
    cap_b = subspace(Qg, phi_b)
    b_apexbase = axis_alignment(phi_b[:, 0], z)
    print(f"\n[baseline isotropic graph-Laplacian]  "
          f"mode1 apex-base |r|={b_apexbase:.3f}  capture5={cap_b[:5].mean():.3f}")

    # the real cardiac operator, swept over the physiological helix range
    helices = [-60, -30, 0, 30, 60]
    results = []
    for h in helices:
        K_m = monodomain_stiffness(V, F, h)
        lam_m, phi_m = gen_low_modes(K_m, M, K)
        cap = subspace(Qg, phi_m)
        m1 = phi_m[:, 0]
        ab = axis_alignment(m1, z)
        cc = axis_alignment(m1, circ)
        rho = float(spearmanr(lam_g, lam_m).correlation)
        # effective rank of the target apex-base axis in the field eigenbasis
        proj = (phi_m.T @ (z - z.mean()))
        proj = proj / (np.linalg.norm(proj) + 1e-12)
        cum = np.cumsum(proj ** 2)
        n90 = int(np.searchsorted(cum, 0.90) + 1)
        results.append(dict(helix=h, mode1_apexbase=ab, mode1_circ=cc,
                            capture5=float(cap[:5].mean()), capture_all=float(cap.mean()),
                            spearman=rho, apexbase_modes_90=n90))
        print(f"  helix {h:+3d} deg : mode1 apex-base |r|={ab:.3f}  circ |r|={cc:.3f}  "
              f"capture5={cap[:5].mean():.3f}  rho={rho:.3f}  "
              f"apex-base in {n90} modes (90%)")

    json.dump(dict(niederer=dict(sig_par=SIG_PAR, sig_trans=SIG_TRANS, anisotropy=ANISO),
                   baseline=dict(mode1_apexbase=b_apexbase, capture5=float(cap_b[:5].mean())),
                   monodomain=results),
              open(OUT / "heart_physiome_modes.json", "w"), indent=2)
    print("\nsaved", OUT / "heart_physiome_modes.json")

    # ---- figure: mode 1 of the real operator across the fibre range ----------
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, len(helices) + 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(UV[:, 0], UV[:, 1], c=phi_g[:, 0], s=4, cmap="RdBu_r", linewidths=0)
    ax.set_title("geometry mode 1\n(apex-base form)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel("apex - base", fontsize=7)
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(UV[:, 0], UV[:, 1], c=phi_b[:, 0], s=4, cmap="RdBu_r", linewidths=0)
    ax.set_title(f"OLD isotropic field mode 1\napex-base |r|={b_apexbase:.2f}", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_ylabel("apex - base", fontsize=7)
    for c, (h, res) in enumerate(zip(helices, results), start=1):
        K_m = monodomain_stiffness(V, F, h)
        _, phi_m = gen_low_modes(K_m, M, 6)
        m1 = phi_m[:, 0]
        if pearsonr(m1, z)[0] < 0:
            m1 = -m1
        ax = fig.add_subplot(gs[0, c])
        ax.scatter(UV[:, 0], UV[:, 1], c=m1, s=4, cmap="RdBu_r", linewidths=0)
        ax.set_title(f"real cardiac op\nhelix {h:+d} deg\nmode 1", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[1, c]); ax.axis("off")
        ax.text(0.0, 0.95,
                f"apex-base |r|={res['mode1_apexbase']:.2f}\n"
                f"circ     |r|={res['mode1_circ']:.2f}\n"
                f"capture5 ={res['capture5']:.2f}\n"
                f"rho      ={res['spearman']:.2f}\n"
                f"axis in {res['apexbase_modes_90']} modes",
                fontsize=8, va="top", family="monospace")
    fig.suptitle("Real validated cardiac monodomain operator (Niederer 2011, 7.58:1 anisotropy, "
                 "Streeter fibres) vs the LV form -- retiring the idealised/generic-surface caveat",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "heart_physiome_modes.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
