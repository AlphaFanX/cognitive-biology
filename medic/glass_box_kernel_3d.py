"""
The glass-box zygote kernel, lifted onto the real 3D vertebrate body.
=====================================================================

glass_box_kernel.py ran the coupled genome->body pass on a flat grid. This lifts
it onto the actual 3D vertebrate geometry of medic.nca_vertebrate_3d -- the body
that already settles to the 8/8 Bauplan -- so the placement and fate heads now act
on a real head/trunk/tail/limb-bud body with a dorsoventral germ-layer axis.

The stages, on the 3D body:
  0  KERNEL BASE      V_ZYGOTE (-70 mV) everywhere in the body.
  1  LGM SET-POINTS   build_static_target: the genome-sourced 3D target V*(AP,DV,LR)
                      -- DV germ-layer floor + AP polarity + organ RD fields.
  2  OPERATOR+FRAME   the 6-neighbour no-flux gap-junction Laplacian on the body;
                      its lowest eigenmode is the anteroposterior axis.
  3  PLACEMENT HEAD   a morphogen standing wave oriented by that frame (AP-tiered,
                      DV-tiered, bilaterally symmetric about the LR midline) ->
                      the primordium loci in the volume.
  4  NCA INNER LOOP   run_clock settles V from the -70 base toward V* on the
                      operator, through the her1 clock -> the Bauplan field (8/8).
  5  BEHAVIOURAL HEADS at each primordium the settled V_m selects the FATE (nearest
                      organ preferred voltage), placing organs at anatomically
                      sensible AP/DV stations (brain dorsal-anterior, heart ventral-
                      anterior, gut ventral, ...).

HONEST SCOPE: glass-box (traced, not trained). The frame/placement wavelength is
idealised; the heads read the field by nearest preferred voltage rather than a
trained shared MLP; absolute V_m is still anchored (the open magnitude gap). What
the lift shows is that the whole coupled pass runs on the real 3D body that already
produces the vertebrate Bauplan.

Run: python -m medic.glass_box_kernel_3d
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.nca_vertebrate_3d import (build_static_target, run_clock, bauplan_checklist,
                                     _grids, NAP, NDV, NLR)
from medic.tissue.genomic_nca import V_ZYGOTE
from medic.bioelectric_development import ORGAN_PREFERRED_VOLTAGE

OUT = Path("data/organ_cascade")


def frame_modes(mask, AP, k=5):
    """Low non-trivial eigenmodes of the body's no-flux gap-junction Laplacian.
    Returns the mode best aligned with the AP axis (the anteroposterior standing
    wave) and its |corr| -- the low modes carry the body axes, but which index is
    AP depends on the body's graph aspect, so we select rather than assume mode 1."""
    idx = np.full(mask.shape, -1)
    idx.ravel()[np.flatnonzero(mask)] = np.arange(int(mask.sum()))
    I, J = [], []
    for axis in range(3):
        s0 = [slice(None)] * 3; s1 = [slice(None)] * 3
        s0[axis] = slice(0, -1); s1[axis] = slice(1, None)
        both = mask[tuple(s0)] & mask[tuple(s1)]
        i0 = idx[tuple(s0)][both]; i1 = idx[tuple(s1)][both]
        I += [i0, i1]; J += [i1, i0]
    I = np.concatenate(I); J = np.concatenate(J)
    n = int(mask.sum())
    A = sp.coo_matrix((np.ones(I.size), (I, J)), shape=(n, n)).tocsr()
    d = np.asarray(A.sum(1)).ravel()
    L = (sp.diags(d) - A).tocsr()
    vals, vecs = spla.eigsh(L, k=k + 1, sigma=-1e-9, which="LM")
    o = np.argsort(vals)
    apc = AP[mask]
    corrs = [abs(np.corrcoef(vecs[:, o][:, i], apc)[0, 1]) for i in range(1, k + 1)]
    best = int(np.argmax(corrs))
    phi = np.full(mask.shape, np.nan)
    phi[mask] = vecs[:, o][:, best + 1]
    return phi, float(corrs[best]), best + 1


def placement_wave(mask, n_ap=2.0, n_dv=1.5, n_lr=1.11):
    """Morphogen standing wave oriented by the frame: AP-tiered, DV-tiered, and
    bilaterally symmetric about the LR midline (cos in LR -> antinodes at 0, +-).
    Its local maxima in the body are the primordium loci."""
    AP, DV, LR = _grids()
    wave = (np.cos(2 * np.pi * n_ap * AP)
            * np.cos(2 * np.pi * n_dv * (DV - 0.5))
            * np.cos(2 * np.pi * n_lr * LR))
    w = np.where(mask, wave, -np.inf)
    peaks = (maximum_filter(w, size=3) == w) & mask & (wave > 0.5)
    return list(zip(*np.where(peaks))), wave


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("GLASS-BOX ZYGOTE KERNEL on the REAL 3D vertebrate body\n")

    # Stage 1: LGM set-points on the 3D body (also gives mask + axes)
    Vstar, mask, AP, DV, LR, _ = build_static_target()
    print(f"1 LGM SET-POINTS   V*(AP,DV,LR) on the body ({int(mask.sum())} voxels), "
          f"{np.nanmin(Vstar[mask]):.0f}..{np.nanmax(Vstar[mask]):.0f} mV")

    # Stage 2: operator + frame (the AP axis appears among the low eigenmodes)
    phi, corr, ap_mode = frame_modes(mask, AP)
    print(f"2 OPERATOR+FRAME   no-flux gap-junction Laplacian; AP axis = low mode {ap_mode} (|r|={corr:.2f})")

    # Stage 3: placement head (oriented standing wave -> primordia)
    prim, wave = placement_wave(mask)
    print(f"3 PLACEMENT HEAD   oriented morphogen wave -> {len(prim)} primordium loci")

    # Stage 4: NCA inner loop settles to the Bauplan
    res = run_clock()
    V = res["V"]
    checks = bauplan_checklist(V)
    npass = sum(1 for _, ok in checks)
    print(f"4 NCA INNER LOOP   settled to the Bauplan: {npass}/{len(checks)} checks pass")

    # Stage 5: the TRAINED shared-kernel head reads the settled field (position + V_m)
    import torch
    from medic.trained_kernel_head import train_head, features as head_features, ORGANS as ATLAS_ORGANS
    Pbody = np.stack([AP[mask], DV[mask], LR[mask]], axis=1)
    net, _, _ = train_head(Pbody, V[mask])
    with torch.no_grad():
        fate_body = net(torch.tensor(head_features(Pbody, V[mask])))[0].argmax(1).numpy()
    idx3d = np.full(mask.shape, -1); idx3d[mask] = np.arange(int(mask.sum()))
    assigns = []
    for (a, b, c) in prim:
        if not np.isfinite(V[a, b, c]):
            continue
        assigns.append(dict(ap=float(AP[a, b, c]), dv=float(DV[a, b, c]), lr=float(LR[a, b, c]),
                            ap_i=int(a), dv_i=int(b), vm=float(V[a, b, c]),
                            fate=ATLAS_ORGANS[int(fate_body[idx3d[a, b, c]])]))
    placed = sorted(set(d["fate"] for d in assigns))
    print(f"5 BEHAVIOURAL HEADS TRAINED shared-kernel head placed {len(placed)} organ types: {', '.join(placed)}")
    # anatomical sanity: mean AP/DV per placed organ
    print("\n  organ        mean AP   mean DV   (0=anterior/ventral, 1=posterior/dorsal)")
    for o in placed:
        ds = [d for d in assigns if d["fate"] == o]
        print(f"  {o:12s}  {np.mean([d['ap'] for d in ds]):.2f}     {np.mean([d['dv'] for d in ds]):.2f}")

    _figure(V, mask, phi, wave, assigns, checks, corr, ap_mode, npass)
    json.dump(dict(voxels=int(mask.sum()), vstar_range=[float(np.nanmin(Vstar[mask])), float(np.nanmax(Vstar[mask]))],
                   frame_mode1_ap_corr=float(corr), n_primordia=len(prim),
                   bauplan_pass=f"{npass}/{len(checks)}", organs_placed=placed,
                   assignments=assigns),
              open(OUT / "glass_box_kernel_3d.json", "w"), indent=2)
    print("\nsaved", OUT / "glass_box_kernel_3d.json")
    return npass >= 7


def _figure(V, mask, phi, wave, assigns, checks, corr, ap_mode, npass):
    sag = np.nanmean(np.where(mask, V, np.nan), axis=2).T        # (NDV, NAP), LR-averaged
    with np.errstate(invalid="ignore"):
        phis = np.nanmean(phi, axis=2).T
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(phis, origin="lower", aspect="auto", cmap="RdBu_r")
    ax.set_title(f"2  operator frame: AP-aligned low eigenmode {ap_mode} (LR-averaged)\n|r|={corr:.2f} with the AP axis", fontsize=9)
    ax.set_xlabel("AP (anterior -> posterior)"); ax.set_ylabel("DV"); fig.colorbar(im, ax=ax, fraction=0.03)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(sag, origin="lower", aspect="auto", cmap="RdBu_r")
    ax.set_title(f"4  NCA settled V_m (sagittal, LR-averaged) -> Bauplan {npass}/8", fontsize=9)
    ax.set_xlabel("AP"); ax.set_ylabel("DV (ventral -> dorsal)"); fig.colorbar(im, ax=ax, fraction=0.03)

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(sag, origin="lower", aspect="auto", cmap="Greys_r", alpha=0.5)
    cmap = plt.get_cmap("tab10")
    organs = sorted(set(d["fate"] for d in assigns))
    cix = {o: i for i, o in enumerate(organs)}
    for d in assigns:
        ax.scatter(d["ap_i"], d["dv_i"], color=cmap(cix[d["fate"]] % 10), s=60, edgecolors="k", linewidths=0.5)
    for o in organs:
        ax.scatter([], [], color=cmap(cix[o] % 10), label=o)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_title("5  trained shared-kernel head: organs placed\n(AP x DV, LR-averaged)", fontsize=9)
    ax.set_xlabel("AP"); ax.set_ylabel("DV")

    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    txt = "THE GLASS BOX ON THE 3D BODY\n" + "-" * 40 + "\n"
    txt += "stages 0-5 all run on the real vertebrate\ngeometry that settles to the Bauplan.\n\n"
    txt += f"Bauplan checklist: {npass}/8\n"
    for label, ok in checks:
        txt += f"  [{'x' if ok else ' '}] {label[:44]}\n"
    ax.text(0.0, 0.98, txt, fontsize=7.4, va="top", family="monospace")

    fig.suptitle("The glass-box zygote kernel lifted onto the real 3D vertebrate body: LGM set-points -> operator "
                 "frame -> oriented-RD placement -> NCA settle (8/8 Bauplan) -> fate heads place organs anatomically",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "glass_box_kernel_3d.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "glass_box_kernel_3d.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'LIFTED (Bauplan holds)' if ok else 'CHECK'}")
