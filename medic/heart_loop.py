"""
Phase 3 (organ program #1) -- HEART LOOPING: the straight heart tube loops rightward into the D-loop.
============================================================================================================
The linear heart tube (cranio-caudal, at the ventral midline) does not stay straight: it LOOPS. The
central (ventricular) segment bulges VENTRALLY and swings to the RIGHT (dextral / D-loop), so the tube
becomes a C then an S, bringing the outflow cranio-ventral and the atria dorso-caudal -- the looped
topology the mouse heart has by ~E10.5 and keeps at E12.5.

Genome anchor (each behaviour -> master gene):
  - HANDEDNESS (which way it loops) = the LEFT-RIGHT axis: Nodal on the left -> Pitx2 (left) -> the tube
    loops to the RIGHT (dextral). Flip Pitx2 -> L-loop / situs inversus. Here `pitx2 = +1` dextral.
  - CHAMBER identity ALONG the tube (the second heart field adds at the arterial pole):
    inflow/sinus venosus (caudal) -> ATRIUM (Tbx5) -> atrioventricular canal -> VENTRICLE (Hand1/Hand2,
    the ventral bulge apex) -> OUTFLOW tract (Isl1, second heart field, cranial).
  - The looping bend itself = actomyosin-driven differential growth + rightward cell movement.

The straight tube is parameterised t in [0,1] (inflow->outflow) by projection on the heart's long axis;
each cell keeps its radial offset from the axis and is carried on the looped centreline's local frame,
so the tube stays a tube (no smearing). This is the organ analogue of the whole-body body_fold.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.heart_loop
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.tps_register import STAGES, STAGE_FRAC, _our_slab, _silhouette_from
from medic.cell_level_mouse import read_stage, COL
from medic.unified_embryo import simulate, FATES

HEART_FATES = ("Heart", "Outflow", "Atrium", "Ventricle")
CHAMBER_COL = {"inflow": "#8b5cf6", "Atrium": "#38bdf8", "Ventricle": "#ef4444",
               "Outflow": "#f59e0b"}
CHAMBER_GENE = {"inflow": "Sinus venosus", "Atrium": "Tbx5", "Ventricle": "Hand1/2", "Outflow": "Isl1"}


def loop_centerline(t, pitx2=+1.0, ventral=0.55, right=0.42, coil=0.85):
    """The D-loop centreline as a function of tube parameter t in [0,1] (inflow->outflow). A C/S loop:
    the ventricular midsegment bulges VENTRALLY (-y) and the tube swings RIGHT (+z * pitx2), with a mild
    second inflection so the outflow rises cranially = the S. Returns an (N,3) curve in heart-local axes
    (x = original cranio-caudal, y = dorso-ventral, z = left-right)."""
    x = t - 0.5                                                   # cranio-caudal (outflow cranial +)
    y = -ventral * np.sin(np.pi * t) - 0.10 * np.sin(2 * np.pi * t)   # ventral bulge (ventricle) + S kink
    z = pitx2 * right * np.sin(coil * np.pi * t)                  # dextral swing (Pitx2)
    return np.stack([x, y, z], 1)


def chamber_of(t):
    """Chamber identity along the tube (caudal inflow -> cranial outflow), each anchored to its gene."""
    lab = np.empty(t.shape, dtype=object)
    lab[t < 0.22] = "inflow"                                      # sinus venosus (venous pole)
    lab[(t >= 0.22) & (t < 0.45)] = "Atrium"                      # Tbx5
    lab[(t >= 0.45) & (t < 0.78)] = "Ventricle"                   # Hand1/2 (the ventral bulge)
    lab[t >= 0.78] = "Outflow"                                    # Isl1 (second heart field)
    return lab


def loop_heart(hpos, pitx2=+1.0):
    """Loop a straight heart-cell cloud. hpos = (N,3) heart cells. Fit the tube axis (PCA long axis),
    parameterise cells t in [0,1] along it (caudal->cranial), keep each cell's radial offset, and place
    it on the looped centreline's local Frenet frame. Returns (looped (N,3), t, chamber labels)."""
    c = hpos - hpos.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    axis = vt[0]
    # orient axis caudal(inflow)->cranial(outflow): outflow is the more anterior (higher original x)
    proj = c @ axis
    if np.corrcoef(proj, hpos[:, 0])[0, 1] < 0:
        axis = -axis; proj = -proj
    t = (proj - proj.min()) / (np.ptp(proj) + 1e-9)              # 0 inflow .. 1 outflow
    radial = c - np.outer(proj, axis)                            # offset perpendicular to the axis
    scale = np.ptp(proj)                                         # keep the loop the tube's own size
    # dense centreline + local frame
    ts = np.linspace(0, 1, 120)
    C = loop_centerline(ts, pitx2) * scale
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    up = np.array([0, 0, 1.0])
    N = np.cross(up, T); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    B = np.cross(T, N)
    idx = np.clip((t * (len(ts) - 1)).astype(int), 0, len(ts) - 1)
    # radial offset expressed in the axis-perpendicular plane -> carry via the local (N,B) frame
    r2 = radial - np.outer(radial @ axis, axis)
    e1 = np.cross(up, axis); e1 /= (np.linalg.norm(e1) + 1e-9); e2 = np.cross(axis, e1)
    u = r2 @ e1; v = r2 @ e2
    looped = C[idx] + u[:, None] * N[idx] + v[:, None] * B[idx] + hpos.mean(0)
    return looped, t, chamber_of(t)


def main():
    st = "E12.5"; rxy, rann = read_stage(STAGES[st])
    sil = _silhouette_from(rxy, rann); sil = {"a": sil["a"], "g": [max(0.14, 1.6 * g) for g in sil["g"]]}
    frames, _ = simulate(use_ecm=True, limb_buds=True, convergent_ext=1.0,
                         n_end=int(30000 * STAGE_FRAC[st]), shape_target=sil, flexure=0.0,
                         seed=0, head_expand=True, regional_growth=True, head_shape_relax=0.3)
    P = frames[-1][3]; fid = frames[-1][5]
    names = np.array([FATES[f] if f >= 0 else "" for f in fid])
    hm = np.isin(names, HEART_FATES)
    hpos = P[hm].astype(float)
    print(f"  heart cells: {hm.sum()}")
    looped, t, cham = loop_heart(hpos)
    ccol = [CHAMBER_COL[c] for c in cham]

    fig, ax = plt.subplots(1, 4, figsize=(19, 6), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(hpos[:, 0], hpos[:, 1], s=8, c=ccol); ax[0].set_title("straight heart tube (sagittal)\ncaudal→cranial", color="#7dd3fc", fontsize=9)
    ax[1].scatter(looped[:, 0], looped[:, 1], s=8, c=ccol); ax[1].set_title("D-loop (sagittal x-y)\nventricle bulges ventral", color="#a5f3c0", fontsize=9)
    ax[2].scatter(looped[:, 0], looped[:, 2], s=8, c=ccol); ax[2].set_title("D-loop (top x-z) — dextral\nPitx2 → rightward", color="#a5f3c0", fontsize=9)
    hh = np.array([("Heart" in a) for a in rann])
    ax[3].scatter(rxy[:, 0], -rxy[:, 1], s=3, c=["#ef4444" if h else "#2a2f3a" for h in hh]); ax[3].set_title(f"real mouse {st}\n(heart red)", color="#cbd5e1", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=CHAMBER_COL[k], mec="none", label=f"{k} ({CHAMBER_GENE[k]})") for k in CHAMBER_COL]
    ax[1].legend(handles=handles, loc="lower left", fontsize=7, facecolor="#0d1017", labelcolor="#cbd5e1", framealpha=0.3)
    fig.suptitle("Phase 3 — Heart looping: the straight tube loops rightward (D-loop) into the looped chambers; "
                 "handedness = Nodal→Pitx2, chamber identity = Isl1/Hand/Tbx5", color="#e2e8f0", fontsize=11)
    fig.tight_layout(); os.makedirs("data/organ_cascade", exist_ok=True)
    out = "data/organ_cascade/heart_loop.png"
    fig.savefig(out, dpi=125, facecolor="#0d1017"); print(f"saved {out}")


if __name__ == "__main__":
    main()
