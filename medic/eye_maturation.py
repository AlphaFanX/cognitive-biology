"""The developmental loop on a SECOND organ: the eye (Paper #7 Section 6).

The heart closed the loop through a looping tube. The eye closes it through a different embryonic behaviour --
apical constriction folding the optic vesicle into a cup (medic/optic_cup_forward.py) -- and a different adult
trait: ocular AXIAL LENGTH, the antero-posterior dimension of the globe, the morphological axis of myopia and
one of the most heavily associated traits in human genetics. That the same machinery closes on a different
organ, a different behaviour, and a different trait is the point of doing a second organ.

Same three pieces as heart_maturation.py:
  1. FORWARD MAP theta -> adult morphometrics.  theta = the eye's embryonic knobs: cup_size (the eye-field /
     optic-vesicle scale, set by eye-field growth, RSPO1/Wnt-PAX6-SIX3-RAX) -> adult axial length; and
     constriction (apical constriction amplitude, Shroom3/Rock1/Myh9) -> corneal curvature. Maturation is
     approximated as endpoint-anchored allometric growth (Section 2): the constants are set so the baseline
     optic cup matures onto the adult reference eye.
  2. JACOBIAN d(adult)/d(knob), central differences.
  3. INVERSE the loop: a real per-allele axial-length effect back-propagates to the embryonic eye-size knob.

The axial-length effect sizes here are REAL and, unusually, reported directly in millimetres (not SD units):
RSPO1 and ZC3H11B from the CREAM axial-length GWAS via the GWAS Catalog (trait EFO_0005318). HONEST SCOPE:
axial length has a large POST-natal growth component (emmetropization), so its genetics partition across the
embryonic and maturation layers exactly as the loop's Section-5 limit describes; RSPO1 (Wnt eye growth) sits
toward the embryonic eye-size end. The maturation map is a calibrated allometric first cut, not a fitted
developmental trajectory.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.eye_maturation
"""
import json
import numpy as np

# Adult eye endpoint anchor (population-typical adult values; standard ophthalmic ranges -- VERIFY before paper).
REF = {"axial_length_mm": 23.5, "corneal_radius_mm": 7.8}
TRAIT_SD = {"axial_length_mm": 1.2, "corneal_radius_mm": 0.27}

# Baseline embryonic knobs: cup_size (eye-field/optic-vesicle scale) and apical-constriction amplitude
# (0.82 from optic_cup_forward.constriction_profile).
TH0 = {"cup_size": 1.0, "constriction": 0.82}
KNOB_GENE = {
    "cup_size":     ("RSPO1/Wnt eye-field", "optic-vesicle / eye-field scale"),
    "constriction": ("SHROOM3/ROCK1/MYH9", "apical constriction, cup curvature"),
}
TRAITS = ["axial_length_mm", "corneal_radius_mm"]
KNOBS = ["cup_size", "constriction"]

# REAL per-allele axial-length GWAS effects (GWAS Catalog trait EFO_0005318), reported directly in mm.
REAL_BETA = [
    dict(gene="RSPO1", snp="rs4074961", allele="T", eaf=0.436, trait="axial_length_mm", knob="cup_size",
         beta_mm=+0.0728, pval="4e-13",
         source="CREAM axial-length GWAS (GWAS Catalog EFO_0005318): +0.0728 mm axial length per T allele"),
    dict(gene="ZC3H11B", snp="rs994767", allele="A", eaf=0.409, trait="axial_length_mm", knob="cup_size",
         beta_mm=-0.0716, pval="1e-11",
         source="CREAM axial-length GWAS (GWAS Catalog EFO_0005318): -0.0716 mm axial length per A allele"),
]


def predict_adult(theta=None):
    """Endpoint-anchored allometry: eye knobs -> adult eye morphometrics (TH0 -> REF)."""
    th = dict(TH0)
    if theta: th.update(theta)
    cup = th["cup_size"] / TH0["cup_size"]                 # eye-field scale -> globe size
    con = th["constriction"] / TH0["constriction"]         # apical constriction -> cup steepness
    return {
        "axial_length_mm":  REF["axial_length_mm"] * cup,          # globe grows with the eye-field/vesicle scale
        "corneal_radius_mm": REF["corneal_radius_mm"] / con,       # more constriction -> steeper -> smaller radius
    }


def jacobian(eps=0.1):
    J = np.zeros((len(TRAITS), len(KNOBS)))
    for j, k in enumerate(KNOBS):
        h = eps * abs(TH0[k])
        yp, ym = predict_adult({k: TH0[k] + h}), predict_adult({k: TH0[k] - h})
        for i, t in enumerate(TRAITS):
            J[i, j] = (yp[t] - ym[t]) / (2 * h)
    return J


def back_propagate(trait, delta_mm, knob):
    J = jacobian()
    s = J[TRAITS.index(trait), KNOBS.index(knob)]
    return None if abs(s) < 1e-9 else delta_mm / s


def main():
    base = predict_adult()
    print("Baseline adult eye (optic cup -> maturation) vs anchor REF:")
    for t in TRAITS:
        print(f"  {t:17s} {base[t]:6.2f}   (ref {REF[t]})")

    J = jacobian()
    print("\nMaturation Jacobian  d(adult)/d(embryonic knob):")
    print(f"{'':18s}" + "".join(f"{k[:11]:>13s}" for k in KNOBS))
    for i, t in enumerate(TRAITS):
        print(f"{t:18s}" + "".join(f"{J[i, j]:13.3f}" for j in range(len(KNOBS))))

    print("\nQuantitative loop with REAL per-allele axial-length beta (mm):")
    real = []
    for b in REAL_BETA:
        dtheta = back_propagate(b["trait"], b["beta_mm"], b["knob"])
        rel = 100.0 * dtheta / TH0[b["knob"]]
        print(f"  {b['gene']:8s} {b['snp']}-{b['allele']}  {b['beta_mm']:+.4f} mm/allele  ->  "
              f"d{b['knob']} = {dtheta:+.5f} ({rel:+.2f}% baseline/allele)")
        real.append(dict(gene=b["gene"], snp=b["snp"], allele=b["allele"], trait=b["trait"], knob=b["knob"],
                         beta_mm=b["beta_mm"], per_allele_dtheta=round(float(dtheta), 6),
                         pct_baseline_per_allele=round(float(rel), 3), source=b["source"]))

    out = dict(organ="eye", behaviour="apical constriction (optic cup)", reference=REF, baseline=base,
               traits=TRAITS, knobs=KNOBS, jacobian=[[round(float(v), 4) for v in r] for r in J],
               real_beta_loop=real,
               scope="Second organ, different behaviour (apical constriction) and trait (axial length). "
                     "Axial-length betas REAL and in mm (CREAM/GWAS Catalog EFO_0005318). Axial length has a "
                     "large post-natal component; RSPO1 (Wnt eye growth) sits toward the embryonic eye-size end. "
                     "Maturation = calibrated allometric first cut, not a fitted developmental trajectory.")
    path = "data/organ_cascade/eye_maturation.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
