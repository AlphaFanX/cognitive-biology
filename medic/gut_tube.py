"""
Phase 3 (organ program #2) -- GUT TUBE + ROTATION: the endoderm strand regionalises and the midgut coils
into the primary intestinal loop, rotating counterclockwise (herniation).
============================================================================================================
The definitive gut is a cranio-caudal endoderm TUBE, regionalised foregut -> midgut -> hindgut. Our model
already has a thin ventral gut strand (probe: foregut/midgut/hindgut split by AP). What it lacks is the
E12.5 morphogenesis: the STOMACH dilation + rotation, and the MIDGUT primary loop, which herniates
ventrally into the umbilicus and ROTATES counterclockwise (~270 deg total) about the superior-mesenteric-
artery axis. Here we build that on the gut pool: cranio-caudal tube -> stomach dilation -> midgut coil,
handed by Pitx2.

Genome anchor (behaviour -> gene):
  - AP regionalisation: Sox2 (foregut: pharynx/oesophagus/STOMACH) -> ... -> Cdx2 / Hoxa13,Hoxd13 (hindgut).
  - tube epithelium <-> mesenchyme signalling: FoxA2 (endoderm), Shh (gut epithelium) -> mesenchymal Bmp4.
  - ROTATION handedness (counterclockwise) = the left-right axis Nodal -> Pitx2 (left dorsal mesentery);
    Pitx2 loss randomises gut rotation (heterotaxy) -- the assay for this step.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.gut_tube
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.tps_register import STAGES, STAGE_FRAC, _silhouette_from
from medic.cell_level_mouse import read_stage, COL
from medic.unified_embryo import simulate, FATES

GUT_FATES = ("Gut", "Foregut", "Hindgut")
REGION_COL = {"Foregut": "#34d399", "Stomach": "#10b981", "Midgut": "#f59e0b", "Hindgut": "#a78bfa"}
REGION_GENE = {"Foregut": "Sox2", "Stomach": "Sox2/Barx1", "Midgut": "small intestine", "Hindgut": "Cdx2/Hoxa13"}


def region_of_gut(t):
    lab = np.empty(t.shape, dtype=object)
    lab[t < 0.20] = "Foregut"
    lab[(t >= 0.20) & (t < 0.38)] = "Stomach"                    # foregut dilation
    lab[(t >= 0.38) & (t < 0.82)] = "Midgut"                     # the primary loop
    lab[t >= 0.82] = "Hindgut"
    return lab


def gut_centerline(t, pitx2=+1.0, n_coils=1.6, herniate=0.55):
    """Gut-tube centreline vs cranio-caudal t (0 foregut..1 hindgut). Foregut/hindgut ~straight midline;
    the MIDGUT (0.38-0.82) coils VENTRALLY (herniation) with a lateral component (rotation, Pitx2 sign)."""
    x = t - 0.5
    y = np.zeros_like(t); z = np.zeros_like(t)
    m = (t >= 0.38) & (t <= 0.82)
    u = np.clip((t - 0.38) / 0.44, 0, 1)                          # 0..1 across the midgut
    bump = np.sin(np.pi * u)                                      # 0 at loop ends, max mid-loop
    phase = u * n_coils * 2 * np.pi
    y[m] = -herniate * bump[m] * (1.0 + 0.25 * np.cos(phase[m]))  # ventral loop (herniation)
    z[m] = pitx2 * herniate * 0.7 * bump[m] * np.sin(phase[m])    # counterclockwise rotation
    # gentle overall ventral bow so foregut/hindgut sit at the body's ventral line
    y += -0.10 * np.sin(np.pi * t)
    return np.stack([x, y, z], 1)


def build_gut(gpos):
    """Order the gut pool cranio-caudally (t), keep radial offset, place on the coiling centreline. The
    STOMACH region is dilated (larger radius). Returns (tube (N,3), t, region labels)."""
    c = gpos - gpos.mean(0)
    t = (gpos[:, 0] - gpos[:, 0].min()) / (np.ptp(gpos[:, 0]) + 1e-9)   # cranio-caudal
    reg = region_of_gut(t)
    rad = c[:, 1] - c[:, 1].mean()
    L = np.ptp(gpos[:, 0])
    ts = np.linspace(0, 1, 200)
    C = gut_centerline(ts) * np.array([L, L, L])
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    up = np.array([0, 0, 1.0]); N = np.cross(up, T); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    B = np.cross(T, N)
    idx = np.clip((t * (len(ts) - 1)).astype(int), 0, len(ts) - 1)
    girth = np.where(reg == "Stomach", 0.16, 0.06) * L                 # stomach dilation
    ang = t * 40.0                                                     # wrap cells round the tube section
    tube = C[idx] + (girth * np.cos(ang) + 0.4 * rad)[:, None] * N[idx] + (girth * np.sin(ang))[:, None] * B[idx]
    tube += gpos.mean(0)
    return tube, t, reg


def main():
    st = "E12.5"; rxy, rann = read_stage(STAGES[st])
    sil = _silhouette_from(rxy, rann); sil = {"a": sil["a"], "g": [max(0.14, 1.6 * g) for g in sil["g"]]}
    frames, _ = simulate(use_ecm=True, limb_buds=True, convergent_ext=1.0,
                         n_end=int(30000 * STAGE_FRAC[st]), shape_target=sil, flexure=0.0,
                         seed=0, head_expand=True, regional_growth=True, head_shape_relax=0.3)
    P = frames[-1][3].astype(float); fid = frames[-1][5]
    names = np.array([FATES[f] if f >= 0 else "" for f in fid])
    gpos = P[np.isin(names, GUT_FATES)]
    print(f"  gut cells: {len(gpos)}")
    tube, t, reg = build_gut(gpos)
    rcol = [REGION_COL[r] for r in reg]

    fig, ax = plt.subplots(1, 4, figsize=(19, 6), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(gpos[:, 0], gpos[:, 1], s=12, c=rcol); ax[0].set_title("straight gut strand (sagittal)\nregionalised foregut→hindgut", color="#7dd3fc", fontsize=9)
    ax[1].scatter(tube[:, 0], tube[:, 1], s=12, c=rcol); ax[1].set_title("gut TUBE (sagittal x-y)\nstomach dilates, midgut loops ventral", color="#a5f3c0", fontsize=9)
    ax[2].scatter(tube[:, 0], tube[:, 2], s=12, c=rcol); ax[2].set_title("gut (top x-z) — CCW rotation\nmidgut coil (Pitx2)", color="#a5f3c0", fontsize=9)
    gg = np.array([("GI" in a) or ("Gut" in a) for a in rann])
    ax[3].scatter(rxy[:, 0], -rxy[:, 1], s=3, c=["#f59e0b" if g else "#2a2f3a" for g in gg]); ax[3].set_title(f"real mouse {st}\n(GI tract orange)", color="#cbd5e1", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=REGION_COL[k], mec="none", label=f"{k} ({REGION_GENE[k]})") for k in REGION_COL]
    ax[1].legend(handles=handles, loc="lower left", fontsize=7, facecolor="#0d1017", labelcolor="#cbd5e1", framealpha=0.3)
    fig.suptitle("Phase 3 — Gut tube + rotation: regionalised endoderm tube; stomach dilates, the midgut coils into the "
                 "primary loop and rotates CCW (Sox2/Cdx2 · FoxA2/Shh · Pitx2)", color="#e2e8f0", fontsize=10)
    fig.tight_layout(); os.makedirs("data/organ_cascade", exist_ok=True)
    out = "data/organ_cascade/gut_tube.png"
    fig.savefig(out, dpi=125, facecolor="#0d1017"); print(f"saved {out}")


if __name__ == "__main__":
    main()
