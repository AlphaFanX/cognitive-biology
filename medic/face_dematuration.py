"""Paper #7: the allometric maturation operator run BACKWARD on the standard (kernel) face.

The developmental forward map (Layer 3) matures the embryo toward the adult. Its inverse is an
ALLOMETRIC operator: each facial region scales against the whole by its own power-law exponent
(Huxley's law, y=b*x^alpha), so running it backward contracts the adult face toward the fetal one.
The dominant, well-measured gradient is neurocranium-vs-face: the neurocranium is negatively
allometric (large early, grows little) and the lower face/jaw is positively allometric (small early,
grows most) -- so de-maturation recovers the "baby-schema" face (large forehead + eyes, small flat
nose, small receding jaw).

Applied to the FaceBase STANDARD (mean) face = the additive intercept = the frozen kernel's output
(subtract the GWAS adapter and you are here). Rendering adult -> child -> late-fetal shows the operator.
The exponents here are the qualitative neurocranium/face gradient; the exact per-landmark alpha come
from the longitudinal craniofacial databases (AAOF Craniofacial Growth Legacy Collection; INTERGROWTH-21st
fetal biometry; the 3D Atlas of Human Embryology) -- the "developmental series" the paper's Layer 3 is
built to be fit to. The final hand-off from the fetal face to the embryonic PROMINENCES is topological
(process fusion), where allometry gives way to the electric-frame morphogenetic map (Fig. electric-face).

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.face_dematuration
Out: data/organ_cascade/face_dematuration.png
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "face_demo"))
import mesh_morph
import medic.allele_faces_bodies as afb

ALPHA_LOWER = 1.5   # lower face / jaw: strong positive allometry (grows most postnatally)
ALPHA_CRAN  = 0.9   # neurocranium: negative allometry (large early, grows little)


def dematurate(V, f):
    """Inverse maturation to age-fraction f (1=adult -> 0 fetal): allometric contraction about the
    orbital line. Below-eye face scales by f**(ALPHA_LOWER-1) (small jaw); cranium relatively enlarges;
    the lower face flattens in protrusion (had not yet grown forward)."""
    y = V[:, 1]; ymin, ymax = y.min(), y.max(); H = ymax - ymin
    eye = ymin + 0.60 * H
    # SMOOTH face(below)->cranium(above) weight across a band at the eye line (no hard seam).
    band = 0.22 * H
    w = np.clip((eye + 0.5 * band - y) / band, 0.0, 1.0)      # 1 = face, 0 = cranium
    w = w * w * (3.0 - 2.0 * w)                                # smoothstep (C1-continuous)
    sc_below = f ** (ALPHA_LOWER - 1.0)                        # lower face contracts toward the eye line
    sc_above = 1.0 + (1.0 - f) * (1.0 - ALPHA_CRAN) * 1.3      # cranium relatively enlarges
    sy = w * sc_below + (1.0 - w) * sc_above
    zref = np.median(V[:, 2]); cx = np.median(V[:, 0])
    fz = w * (0.5 + 0.5 * f) + (1.0 - w) * 1.0                 # lower face flattens in protrusion; cranium unchanged
    fx = w * 1.0 + (1.0 - w) * (1.0 + 0.10 * (1.0 - f))        # cranium widens a little; face unchanged
    Vd = V.copy()
    Vd[:, 1] = eye + (y - eye) * sy
    Vd[:, 2] = zref + (V[:, 2] - zref) * fz
    Vd[:, 0] = cx + (V[:, 0] - cx) * fx
    return Vd


def main():
    V0, F, HL = mesh_morph.load()
    if F.max() >= len(V0):
        F = F - 1

    ages = [(1.00, "standard adult\n(kernel face)"),
            (0.55, "child ≈ 6 yr"),
            (0.30, "infant"),
            (0.12, "late fetal")]
    allV = np.vstack([dematurate(V0, f)[:, :2] for f, _ in ages])
    mn = allV.min(0); mx = allV.max(0); pad = 0.06 * (mx - mn)
    box = (mn[0] - pad[0], mx[0] + pad[0], mn[1] - pad[1], mx[1] + pad[1])

    fig = plt.figure(figsize=(12.4, 5.3))
    fig.suptitle("The maturation operator run backward: the standard (kernel) face de-matured by allometry",
                 fontsize=12.3, color=afb.INK, y=0.975)
    fig.text(0.5, 0.895, "Each region scales by its own power-law exponent (Huxley); the face is positively "
             "allometric to the neurocranium, so reversing recovers the baby-schema face. "
             "Exact per-landmark α ← AAOF / INTERGROWTH-21st / 3D-embryo-atlas longitudinal data.",
             ha="center", fontsize=8.8, color=afb.INK)
    for i, (f, lab) in enumerate(ages):
        ax = fig.add_axes([0.02 + i * 0.245, 0.09, 0.235, 0.74])
        afb.draw_mesh(ax, dematurate(V0, f), F, afb.SKIN, box=box, ambient=0.42)
        fig.text(0.02 + i * 0.245 + 0.117, 0.11, lab, ha="center", va="top", fontsize=9, color=afb.INK)
    out = "data/organ_cascade/face_dematuration.png"
    fig.savefig(out, dpi=150); print("saved", out)


if __name__ == "__main__":
    main()
