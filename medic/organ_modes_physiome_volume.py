"""Definitive volumetric test: the real cardiac monodomain operator on the LV WALL
with true transmural Streeter fibre rotation (+60 endo -> -60 epi), validated
Niederer (2011) conductivities. Resolves the surface sweep's helix-angle ambiguity
by integrating over the full fibre rotation the way a real ventricular wall does.

Mesh   : prolate-spheroid myocardial shell (endo+epi, wall thickness), structured
         hexahedra split into tetrahedra. Apex hole + open base (valve plane).
Fibre  : analytic tangent frame (e_long, e_circ) per node; helix angle rotates
         linearly across the wall, alpha(t) = +60(1-t) - 60 t.
Operator: linear-FEM 3D anisotropic stiffness  K_ij = sum_T vol_T grad L_i^T D grad L_j,
         D = sig_trans I + (sig_par - sig_trans) f f^T.
Compare: geometry (isotropic D=I) shape modes vs the real anisotropic field modes.

Q1 does mode 1 of the REAL wall operator = the apex-base activation axis?
Q2 does the geometry<->field low-mode correspondence survive in 3D with real fibres?

Run: python -m medic.organ_modes_physiome_volume
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = 30
OUT = Path("data/organ_cascade")

# Niederer et al. (2011) N-version benchmark -> monodomain reduction
SIG_IL, SIG_IT, SIG_EL, SIG_ET = 0.17, 0.019, 0.62, 0.24
SIG_PAR   = SIG_IL * SIG_EL / (SIG_IL + SIG_EL)      # 0.1334 S/m
SIG_TRANS = SIG_IT * SIG_ET / (SIG_IT + SIG_ET)      # 0.0176 S/m
ANISO = SIG_PAR / SIG_TRANS

# 6-tet decomposition of a hexahedron (node order 0..7 = i j l lattice below)
HEX_TETS = [(0, 1, 3, 7), (0, 1, 7, 5), (0, 5, 7, 4),
            (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7)]


def lv_wall(nu=22, nv=36, nt=4, u_min=0.18 * np.pi, u_max=0.80 * np.pi,
            a_endo=0.75, c_endo=1.45, wall=0.30):
    """Structured tet mesh of a prolate-spheroid LV wall. Returns V, tets, fibres."""
    a_epi, c_epi = a_endo + wall, c_endo + wall
    nodes, fib = [], []
    def nid(i, j, l): return (i * nv + (j % nv)) * (nt + 1) + l
    for i in range(nu + 1):
        u = u_min + (u_max - u_min) * i / nu
        for j in range(nv):
            v = 2 * np.pi * j / nv
            for l in range(nt + 1):
                t = l / nt
                a = a_endo + (a_epi - a_endo) * t
                c = c_endo + (c_epi - c_endo) * t
                P = np.array([a * np.sin(u) * np.cos(v),
                              a * np.sin(u) * np.sin(v),
                              c * (1 - np.cos(u))])
                # analytic tangent frame (orthogonal on the spheroid)
                e_long = np.array([a * np.cos(u) * np.cos(v),
                                   a * np.cos(u) * np.sin(v), c * np.sin(u)])
                e_circ = np.array([-a * np.sin(u) * np.sin(v),
                                   a * np.sin(u) * np.cos(v), 0.0])
                e_long /= np.linalg.norm(e_long) + 1e-12
                e_circ /= np.linalg.norm(e_circ) + 1e-12
                alpha = np.radians(60.0 - 120.0 * t)          # +60 endo -> -60 epi
                f = np.cos(alpha) * e_circ + np.sin(alpha) * e_long
                nodes.append(P); fib.append(f / (np.linalg.norm(f) + 1e-12))
    V = np.array(nodes); Fb = np.array(fib)
    tets = []
    for i in range(nu):
        for j in range(nv):
            for l in range(nt):
                # 8 hex corners in lattice order matching HEX_TETS
                c = [nid(i, j, l), nid(i + 1, j, l), nid(i + 1, j + 1, l), nid(i, j + 1, l),
                     nid(i, j, l + 1), nid(i + 1, j, l + 1), nid(i + 1, j + 1, l + 1), nid(i, j + 1, l + 1)]
                for a, b, d, e in HEX_TETS:
                    tets.append([c[a], c[b], c[d], c[e]])
    return V, np.array(tets), Fb


def fem3d(V, tets, fibres, sig_par, sig_trans):
    """Anisotropic 3D linear-FEM stiffness K and lumped mass M."""
    n = V.shape[0]
    rows, cols, vals = [], [], []
    mass = np.zeros(n)
    for tet in tets:
        p = V[tet]
        J = np.array([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
        detJ = np.linalg.det(J)
        vol = abs(detJ) / 6.0
        if vol < 1e-14:
            continue
        Jinv = np.linalg.inv(J)
        grad = np.zeros((4, 3))
        grad[1:] = Jinv.T                       # grad of lambda_1,2,3 = rows of J^{-1} (as columns)
        grad[0] = -grad[1:].sum(0)
        f = fibres[tet].mean(0)
        nf = np.linalg.norm(f)
        f = f / nf if nf > 1e-9 else np.zeros(3)
        D = sig_trans * np.eye(3) + (sig_par - sig_trans) * np.outer(f, f)
        Ke = vol * (grad @ D @ grad.T)
        for a in range(4):
            mass[tet[a]] += vol / 4.0
            for b in range(4):
                rows.append(tet[a]); cols.append(tet[b]); vals.append(Ke[a, b])
    Kf = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    m = np.maximum(mass, mass[mass > 0].mean() * 1e-6)
    return Kf, sp.diags(m)


def gen_low(L, M, k):
    vals, vecs = spla.eigsh(L, k=k + 1, M=M, sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]


def align(mode, coord):
    return abs(pearsonr(mode, coord)[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V, tets, fib = lv_wall()
    z, circ = V[:, 2], np.arctan2(V[:, 1], V[:, 0])
    print(f"LV WALL: {V.shape[0]} nodes, {tets.shape[0]} tets")
    print(f"Niederer monodomain sig_par={SIG_PAR:.4f} sig_trans={SIG_TRANS:.4f} aniso={ANISO:.2f}:1")
    print("fibres: transmural Streeter rotation +60 (endo) -> -60 (epi) deg")

    # geometry (isotropic) vs real cardiac (anisotropic, transmural fibres)
    Kg, M = fem3d(V, tets, fib, 1.0, 1.0)
    Ka, _ = fem3d(V, tets, fib, SIG_PAR, SIG_TRANS)
    lam_g, phi_g = gen_low(Kg, M, K)
    lam_a, phi_a = gen_low(Ka, M, K)
    Qg, _ = np.linalg.qr(phi_g)
    Qa, _ = np.linalg.qr(phi_a)
    cap = (Qa.T @ Qg) ** 2
    cap = cap.sum(0)
    rho = float(spearmanr(lam_g, lam_a).correlation)

    m1 = phi_a[:, 0]
    ab, cc = align(m1, z), align(m1, circ)
    g1_ab = align(phi_g[:, 0], z)
    # effective rank of the apex-base axis in the real field eigenbasis
    proj = phi_a.T @ (z - z.mean()); proj /= np.linalg.norm(proj) + 1e-12
    n90 = int(np.searchsorted(np.cumsum(proj ** 2), 0.90) + 1)

    print(f"\n[geometry] mode1 apex-base |r|={g1_ab:.3f}")
    print(f"[REAL cardiac wall] mode1 apex-base |r|={ab:.3f}  circ |r|={cc:.3f}")
    print(f"[correspondence] capture5={cap[:5].mean():.3f} -> capture{K}={cap.mean():.3f}  "
          f"Spearman rho={rho:.3f}")
    print(f"[low-rank] apex-base axis captured in {n90} field modes (90%)")

    res = dict(nodes=int(V.shape[0]), tets=int(tets.shape[0]),
               niederer=dict(sig_par=SIG_PAR, sig_trans=SIG_TRANS, anisotropy=ANISO),
               geometry_mode1_apexbase=g1_ab,
               real_mode1_apexbase=ab, real_mode1_circ=cc,
               capture5=float(cap[:5].mean()), capture_all=float(cap.mean()),
               spearman=rho, apexbase_modes_90=n90)
    json.dump(res, open(OUT / "heart_physiome_volume.json", "w"), indent=2)
    print("saved", OUT / "heart_physiome_volume.json")

    # figure: epicardial slice of geometry mode1 vs real-wall mode1 + correspondence
    epi = np.abs(np.linalg.norm(V[:, :2], axis=1) - 0) >= 0  # all; colour by z for context
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    s = -1 if pearsonr(phi_a[:, 0], z)[0] < 0 else 1
    axs[0].scatter(circ, z, c=phi_g[:, 0], s=6, cmap="RdBu_r", linewidths=0)
    axs[0].set_title(f"geometry mode 1 (form)\napex-base |r|={g1_ab:.2f}", fontsize=9)
    axs[1].scatter(circ, z, c=s * phi_a[:, 0], s=6, cmap="RdBu_r", linewidths=0)
    axs[1].set_title(f"REAL cardiac wall mode 1\nNiederer 7.58:1 + transmural fibres\n"
                     f"apex-base |r|={ab:.2f}  circ |r|={cc:.2f}", fontsize=9)
    for a in axs[:2]:
        a.set_xlabel("circumferential", fontsize=8); a.set_ylabel("apex - base (z)", fontsize=8)
    axs[2].plot(np.arange(1, K + 1), np.cumsum(cap) / np.arange(1, K + 1), "o-", ms=3)
    axs[2].axhline(1, ls="--", c="k", lw=0.5); axs[2].set_ylim(0, 1.02)
    axs[2].set_title(f"field reconstructs form\ncapture5={cap[:5].mean():.2f} rho={rho:.2f}", fontsize=9)
    axs[2].set_xlabel("# modes", fontsize=8); axs[2].set_ylabel("mean capture", fontsize=8)
    fig.suptitle("LV WALL, real validated monodomain operator with transmural Streeter fibres "
                 "-- the definitive field-form test", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "heart_physiome_volume.png", dpi=140, bbox_inches="tight")
    print("saved", OUT / "heart_physiome_volume.png")


if __name__ == "__main__":
    main()
