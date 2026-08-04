"""
Phase 3 (organ program #6) -- VERTEBRAL COLUMN: the somite-derived sclerotome resegments into a series of
vertebrae around the notochord, regionalised by the Hox code into the mouse vertebral formula.
============================================================================================================
The paraxial somites (her1 segmentation clock) split: SCLEROTOME (ventral, Pax1/Pax9, induced by notochord
Shh) -> vertebrae + ribs; DERMOMYOTOME (dorsal) -> dermis + myotome(muscle). The sclerotome RESEGMENTS
(Remak): the caudal half of one somite fuses with the rostral half of the NEXT, so each vertebra straddles a
somite boundary (vertebrae are intersegmental -> the myotome muscle can span a joint). The Hox code sets AP
identity, giving the mouse formula 7 cervical + 13 thoracic + 6 lumbar (+ sacral/caudal).

Genome anchor (behaviour -> gene):
  - sclerotome vs dermomyotome = Shh (notochord/floor plate, ventral) -> Pax1/Pax9 sclerotome; Wnt (dorsal)
    -> dermomyotome. Chondrogenesis = Sox9.
  - resegmentation boundary = Uncx4.1 / Tbx18 mark the caudal/rostral somite halves.
  - vertebral IDENTITY (cervical/thoracic/lumbar) = the nested Hox code (Hox5/6 = C/T boundary, Hox9/10 =
    T/L, Hox10/11 = L/S). Shifting a Hox boundary shifts the rib-bearing region -> a homeotic transformation.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.vertebral_column
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medic.tps_register import STAGES, STAGE_FRAC, _silhouette_from
from medic.cell_level_mouse import read_stage, COL
from medic.unified_embryo import simulate, FATES

REGION_COL = {"cervical": "#38bdf8", "thoracic": "#ef4444", "lumbar": "#f59e0b", "sacral": "#a78bfa"}
# HUMAN vertebral formula per Gray's Anatomy: 7 cervical + 12 thoracic + 5 lumbar + 5 sacral (fused sacrum)
# (+ 4 coccygeal, fused, omitted). Was the mouse formula 7C/13T/6L/4S; switched to follow Gray's (human).
# 12 thoracic => 12 rib pairs; C1..C7, T1..T12, L1..L5, S1..S5.
FORMULA = [("cervical", 7), ("thoracic", 12), ("lumbar", 5), ("sacral", 5)]


def build_column(cpos, n_vert=30):
    """Segment the axial cartilage column into vertebrae along AP, resegment (half-shift the boundaries off
    the somite grid), and assign Hox regional identity by the mouse formula. Returns (t, vertebra, region)."""
    x = cpos[:, 0]
    t = (x - x.min()) / (np.ptp(x) + 1e-9)                          # 0 anterior(C1) .. 1 posterior
    # RESEGMENTATION: vertebra boundaries sit at the somite half-boundary -> shift the grid by half a segment
    vert = np.clip(((t * n_vert) + 0.5).astype(int), 0, n_vert - 1)
    # Hox regional identity by the mouse formula (cumulative vertebra counts -> region)
    total = sum(n for _, n in FORMULA)
    bounds = np.cumsum([n for _, n in FORMULA]) / total
    region = np.empty(len(t), dtype=object)
    for i, (name, _) in enumerate(FORMULA):
        lo = 0.0 if i == 0 else bounds[i - 1]
        region[(t >= lo) & (t <= bounds[i] + 1e-9)] = name
    return t, vert, region


def main():
    st = "E12.5"; rxy, rann = read_stage(STAGES[st])
    sil = _silhouette_from(rxy, rann); sil = {"a": sil["a"], "g": [max(0.14, 1.6 * g) for g in sil["g"]]}
    frames, _ = simulate(use_ecm=True, limb_buds=True, convergent_ext=1.0,
                         n_end=int(30000 * STAGE_FRAC[st]), shape_target=sil, flexure=0.0,
                         seed=0, head_expand=True, regional_growth=True, head_shape_relax=0.3)
    P = frames[-1][3].astype(float); fid = frames[-1][5]
    names = np.array([FATES[f] if f >= 0 else "" for f in fid])
    # the axial cartilage column = the vertebral centra (near the midline)
    cmask = (names == "Cartilage") & (np.abs(P[:, 2]) < 0.18 * np.abs(P[:, 2]).max())
    cpos = P[cmask]
    print(f"  vertebral (axial cartilage) cells: {cmask.sum()}")
    t, vert, region = build_column(cpos)
    rcol = [REGION_COL[r] for r in region]
    # per-vertebra alternating shade to show the segmentation
    seg_shade = np.where(vert % 2 == 0, 0.9, 0.55)

    fig, ax = plt.subplots(1, 3, figsize=(16, 6), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(cpos[:, 0], cpos[:, 1], s=12, c=[(s, s, s) for s in seg_shade])
    ax[0].set_title("SEGMENTED vertebrae (resegmented sclerotome)\none block per vertebra, Sox9/Pax1", color="#7dd3fc", fontsize=9)
    ax[1].scatter(cpos[:, 0], cpos[:, 1], s=12, c=rcol)
    ax[1].set_title("Hox identity: 7C · 13T · 6L · S\n(mouse vertebral formula)", color="#a5f3c0", fontsize=9)
    cc = np.array([("Cartilage" in t2) for t2 in rann])
    ax[2].scatter(rxy[:, 0], -rxy[:, 1], s=2, c=["#e5e7eb" if c else "#2a2f3a" for c in cc])
    ax[2].set_title(f"real mouse {st} (skeleton)", color="#cbd5e1", fontsize=9)
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=REGION_COL[k], mec="none", label=k) for k in REGION_COL]
    ax[1].legend(handles=handles, loc="lower left", fontsize=7, facecolor="#0d1017", labelcolor="#cbd5e1", framealpha=0.3, ncol=2)
    fig.suptitle("Phase 3 — Vertebral column: resegmented sclerotome -> per-somite vertebrae around the notochord, "
                 "Hox-patterned into the mouse formula (7C/13T/6L)", color="#e2e8f0", fontsize=10)
    fig.tight_layout(); os.makedirs("data/organ_cascade", exist_ok=True)
    out_p = "data/organ_cascade/vertebral_column.png"
    fig.savefig(out_p, dpi=125, facecolor="#0d1017"); print(f"saved {out_p}")


if __name__ == "__main__":
    main()
