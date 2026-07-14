"""
The genome-grounded limb model, extended to the menagerie.
==========================================================

Each animal is grown from one cell by the SAME unified NCA+LGM embryo (medic.unified_embryo),
now with the GENOME-GROUNDED limb frame active (medic.limb_genome_frame):

  * fore/hind limb AP levels  <-  real Hox chromosomal colinearity (Hox6 / Hox10);
  * body width (convergent extension) <- the Wnt-PCP module: human ABC accessibility for the
      tetrapod base, and the MEASURED zebrafish/mouse Wnt-PCP enrichment ratio (ZESTA vs MOSTA)
      for the finned fish.

The width decides whether the body is wide enough for the LEFT-RIGHT electric-body eigenmode to
enter the low spectrum. A tetrapod (convergent_ext ~ 1.0) is wide -> the LR mode is present ->
four limbs emerge on its antinodes. The finned fish (convergent_ext ~ 3.6) tapers -> no LR mode
-> no limbs. Same genome, one knob, deep homology: the fish->tetrapod (amphibian) threshold now
runs the whole menagerie.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m menagerie.grow_limbed
Out:  data/menagerie_limbed.png
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX
from .grow_nca import nca_body, species_deform, species_convergent_ext
from .targets import reference_genome

LIMB = FIDX["Limb Bud"]
SHOWN = ["giraffe", "elephant", "human_male", "zebrafish"]


def grow(sp, seed=0):
    import medic.unified_embryo as ue
    from medic.limb_shape import shape
    from medic.limb_genome_frame import genome_limb_frame
    g = reference_genome(sp)
    ce = species_convergent_ext(g)
    frames, _ = ue.simulate(use_ecm=True, seed=seed, limb_buds=True, convergent_ext=ce)
    b, t, pr, P, V, F = frames[-1]
    Ps, Vs, Fs = ue._symmetrize(P, V, F)        # bilateral fold -> four symmetric limbs
    pos = species_deform(Ps, g)                 # von Baer: late species deformation (proportions)
    gf = genome_limb_frame(ce)
    pos, seg = shape(pos, Fs, LIMB, gf["fore_ap"], gf["hind_ap"])   # tighten + PD segments
    return pos, Fs, seg, g


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.3 * len(SHOWN), 5.2))
    rows = []
    SEGC = ["#1f73eb", "#29d15c", "#f7a828"]     # stylopod / zeugopod / autopod
    for i, sp in enumerate(SHOWN):
        pos, fid, seg, g = grow(sp)
        ce = species_convergent_ext(g)
        islimb = fid == LIMB
        nlimb = int(islimb.sum())
        ml, ap = np.ptp(pos[:, 2]), np.ptp(pos[:, 0])
        bp = getattr(g, "body_plan", "tetrapod")
        rows.append((sp, bp, ce, ml / ap, nlimb))

        ax = fig.add_subplot(1, len(SHOWN), i + 1, projection="3d")
        body = ~islimb
        ax.scatter(pos[body, 0], pos[body, 1], pos[body, 2], c="0.72", s=3, alpha=0.5, linewidths=0)
        for si in range(3):
            mm = islimb & (seg == si)
            if mm.any():
                ax.scatter(pos[mm, 0], pos[mm, 1], pos[mm, 2], c=SEGC[si], s=9, alpha=0.95, linewidths=0)
        tag = "finned — NO limbs" if bp == "finned" else f"{nlimb} limb-bud cells"
        ax.set_title(f"{sp.replace('_', ' ')}\n{bp} · conv_ext {ce:.2f} · {tag}", fontsize=10)
        ax.set_box_aspect((np.ptp(pos[:, 0]), np.ptp(pos[:, 1]) + 1e-3, np.ptp(pos[:, 2]) + 1e-3), zoom=1.5)
        ax.view_init(elev=16, azim=-72); ax.set_axis_off()

    fig.suptitle("The genome-grounded limb model across the menagerie: one knob (Wnt-PCP convergent "
                 "extension) sets the body width;\nwide tetrapods grow four limbs on the electric-body "
                 "antinodes, the finned fish stays limbless — deep homology, same genome", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("data/menagerie_limbed.png", dpi=150, bbox_inches="tight")
    print(f"  {'species':14s} {'body_plan':9s} {'conv_ext':>8s} {'ML/AP':>7s} {'limbs':>6s}")
    for sp, bp, ce, asp, n in rows:
        print(f"  {sp:14s} {bp:9s} {ce:8.2f} {asp:7.3f} {n:6d}")
    print("saved data/menagerie_limbed.png")


if __name__ == "__main__":
    main()
