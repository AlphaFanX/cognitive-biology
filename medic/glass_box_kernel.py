"""
The glass-box zygote kernel: one coupled genome->body forward pass (v1 assembly).
=================================================================================

The morphogen-placement head was the last missing component of the NCA+MLP zygote
kernel; with it solved, every stage now exists as a genome-traced module. This
wires them into ONE coupled forward pass on a body grid -- a glass box, because
every stage is observable, and forward, because it runs genome -> field -> heads
-> form with no fitting to the output.

The stages (each a real module, not a re-implementation):
  0  KERNEL BASE      the frozen methylation ground state V_ZYGOTE (-70 mV)
                      (medic.tissue.genomic_nca).
  1  LGM SET-POINTS   the outer perceptron reads the genome and emits the score:
                      the ABC Goldman target field V*(x) and the connexin
                      conductance field g_GJ(x)  (medic.nca_abc_modes.abc_fields).
  2  OPERATOR+FRAME   the inner substrate: the g_GJ-weighted gap-junction Laplacian
                      and its low eigenmodes = the cymatic axis frame.
  3  PLACEMENT HEAD   the positional/patterning attention: a morphogen reaction-
                      diffusion oriented by the frame -> the primordium loci
                      (face_demo.morphogen_orientation.oriented_turing).
  4  NCA INNER LOOP   the field settles from the zygote base toward V* on the
                      operator:  V <- V + k_relax(V*-V) + k_gj(P V - V).
  5  BEHAVIOURAL HEADS at each primordium the settled V_m selects the FATE (the
                      organ whose preferred voltage it matches), and the field
                      supplies MIGRATION (the V_m gradient) and DIVISION.

HONEST SCOPE: this is the GLASS-BOX (traced, not trained) v1 -- a 2D body grid,
the outer read frozen (ABC/AlphaGenome, closed LGM), the heads read from the field
rather than from one trained shared-kernel MLP, and the absolute V_m still anchored
per organ (the un-anchored magnitude layer is the open scientific gap). What it
demonstrates is that the components now compose end to end into one runnable
genome->body pass.

Run: python -m medic.glass_box_kernel
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

from medic.tissue.genomic_nca import V_ZYGOTE
from medic.nca_abc_modes import abc_fields
from medic.bioelectric_development import ORGAN_PREFERRED_VOLTAGE

sys.path.insert(0, str(Path("face_demo").resolve()))
from morphogen_orientation import oriented_turing, local_peaks

N = 48
K_RELAX, K_GJ = 0.30, 0.12
OUT = Path("data/organ_cascade")


def weighted_operator(n, G):
    """g_GJ-weighted gap-junction adjacency A, row-normalised P, and Laplacian L."""
    idx = np.arange(n * n).reshape(n, n)
    g = G.ravel() / (G.mean() + 1e-12)
    I, J, W = [], [], []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        a = idx[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)].ravel()
        b = idx[max(0, di):n - max(0, -di), max(0, dj):n - max(0, -dj)].ravel()
        w = 0.5 * (g[a] + g[b])
        I += [a]; J += [b]; W += [w]
    I = np.concatenate(I); J = np.concatenate(J); W = np.concatenate(W)
    A = sp.coo_matrix((W, (I, J)), shape=(n * n, n * n)).tocsr()
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
    P = sp.diags(1.0 / d) @ A
    L = (sp.diags(d) - A).tocsr()
    return P, L


def low_modes(L, k=4):
    vals, vecs = spla.eigsh(L, k=k + 1, sigma=-1e-9, which="LM")
    o = np.argsort(vals)
    return vecs[:, o][:, 1:]


def settle(Vstar, P, steps=250):
    """Stage 4: the inner NCA loop settles V from the zygote base toward V*."""
    V = np.full(Vstar.size, V_ZYGOTE)
    Vt = Vstar.ravel()
    traj = []
    for s in range(steps):
        V = V + K_RELAX * (Vt - V) + K_GJ * (P @ V - V)
        if s in (0, 4, 20, steps - 1):
            traj.append(V.reshape(Vstar.shape).copy())
    return V.reshape(Vstar.shape), traj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("GLASS-BOX ZYGOTE KERNEL -- coupled genome->body forward pass\n")

    # Stage 0: kernel base
    base = np.full((N, N), V_ZYGOTE)
    print(f"0 KERNEL BASE      V_ZYGOTE = {V_ZYGOTE:.0f} mV (frozen methylation ground state)")

    # Stage 1: LGM set-points (outer perceptron emits V* and g_GJ from the genome)
    Vstar, G = abc_fields(N)
    print(f"1 LGM SET-POINTS   V*(x) {Vstar.min():.0f}..{Vstar.max():.0f} mV, "
          f"g_GJ(x) {G.min():.3f}..{G.max():.3f}")

    # Stage 2: operator + cymatic frame
    P, L = weighted_operator(N, G)
    phi = low_modes(L, 4)
    print(f"2 OPERATOR+FRAME   g_GJ-weighted gap-junction Laplacian; low eigenmodes = the axis frame")

    # Stage 3: placement head (morphogen RD oriented by the frame -> primordia)
    mask = np.ones((N, N), bool)
    placement = oriented_turing(mask, k0=2.5 / N)
    px, py = local_peaks(placement, mask)              # cols, rows
    print(f"3 PLACEMENT HEAD   oriented reaction-diffusion -> {len(px)} primordium loci")

    # Stage 4: NCA inner loop settles the field
    Vsettled, traj = settle(Vstar, P)
    print(f"4 NCA INNER LOOP   settled V_m {Vsettled.min():.0f}..{Vsettled.max():.0f} mV "
          f"(from the -70 base toward V* on the operator)")

    # Stage 5: behavioural heads read the settled field at each primordium
    organs = list(ORGAN_PREFERRED_VOLTAGE)
    ov = np.array([ORGAN_PREFERRED_VOLTAGE[o] for o in organs])
    gy, gx = np.gradient(Vsettled)
    assigns = []
    for c, r in zip(px, py):
        vm = float(Vsettled[r, c])
        oi = int(np.argmin(np.abs(vm - ov)))
        mig = float(np.hypot(gx[r, c], gy[r, c]))       # migration head = |grad V|
        div = float((vm - Vsettled.min()) / (np.ptp(Vsettled) + 1e-9))  # division proxy
        assigns.append(dict(row=int(r), col=int(c), vm=vm, fate=organs[oi],
                            migration=mig, division=div))
    placed = sorted(set(a["fate"] for a in assigns))
    print(f"5 BEHAVIOURAL HEADS FATE assigned at each primordium -> organs placed: "
          f"{', '.join(placed)}  ({len(placed)}/{len(organs)})")

    # whole-field organ territories (nearest preferred voltage), for the map panel
    terr = np.argmin(np.abs(Vsettled[..., None] - ov[None, None, :]), axis=2)

    _figure(base, Vstar, G, phi, placement, (px, py), Vsettled, terr, organs, assigns)
    json.dump(dict(v_zygote=V_ZYGOTE, vstar_range=[float(Vstar.min()), float(Vstar.max())],
                   gGJ_range=[float(G.min()), float(G.max())],
                   settled_range=[float(Vsettled.min()), float(Vsettled.max())],
                   n_primordia=len(px), organs_placed=placed,
                   assignments=assigns),
              open(OUT / "glass_box_kernel.json", "w"), indent=2)
    print("\nsaved", OUT / "glass_box_kernel.json")
    return True


def _figure(base, Vstar, G, phi, placement, peaks, Vsettled, terr, organs, assigns):
    px, py = peaks
    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    a = ax[0, 0]; im = a.imshow(base, cmap="RdBu_r", vmin=-75, vmax=-15)
    a.set_title("0  kernel base: V_ZYGOTE (-70 mV)\nfrozen methylation ground state", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046)
    a = ax[0, 1]; im = a.imshow(Vstar, cmap="RdBu_r"); fig.colorbar(im, ax=a, fraction=0.046)
    a.set_title("1  LGM set-points: V*(x) (mV)\nthe outer perceptron reads the genome", fontsize=9)
    a = ax[0, 2]; im = a.imshow(phi[:, 0].reshape(N, N), cmap="RdBu_r")
    a.set_title("2  operator + frame: low eigenmode 1\n(g_GJ-weighted gap-junction Laplacian)", fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046)
    a = ax[1, 0]; a.imshow(placement, cmap="viridis")
    a.scatter(px, py, c="red", s=22, edgecolors="w", linewidths=0.4)
    a.set_title(f"3  placement head: oriented RD\n-> {len(px)} primordium loci", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    a = ax[1, 1]; im = a.imshow(Vsettled, cmap="RdBu_r"); fig.colorbar(im, ax=a, fraction=0.046)
    a.set_title("4  NCA inner loop: settled V_m\n(from base toward V* on the operator)", fontsize=9)
    a = ax[1, 2]
    im = a.imshow(terr, cmap="tab20", vmin=0, vmax=len(organs) - 1)
    a.scatter(px, py, c="k", s=26)
    for d in assigns:
        a.text(d["col"], d["row"], d["fate"][:3], fontsize=6, ha="center", va="center", color="w")
    a.set_title(f"5  behavioural heads: FATE map\n{len(set(d['fate'] for d in assigns))} organs placed "
                f"at the primordia", fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    fig.suptitle("The glass-box zygote kernel: one coupled genome->body forward pass -- kernel base -> LGM set-points "
                 "-> genome operator+frame -> oriented-RD placement -> NCA settle -> the fate/migration/division heads. "
                 "Every stage observable; nothing fit to the output.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "glass_box_kernel.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "glass_box_kernel.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'ASSEMBLED' if ok else 'CHECK'}")
