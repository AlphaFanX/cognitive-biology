"""The developmental loop on a THIRD organ: the kidney (Paper #7 Section 6).

Heart = looping tube; eye = apical-constriction cup; kidney = a THIRD embryonic behaviour, BRANCHING
MORPHOGENESIS. The ureteric bud branches into the collecting-duct tree and each tip induces a nephron, so
the branching programme sets the nephron endowment and, with it, the volume of the kidney cortex. The adult
morphological trait is kidney cortex volume, measured by abdominal MRI in the UK Biobank.

Same three pieces as heart_maturation.py / eye_maturation.py:
  1. FORWARD MAP theta -> adult morphometrics. theta = the kidney's embryonic knobs: branching (nephron
     endowment, MYCN/SIX2/GDNF-RET) -> cortex volume; elongation (convergent extension) -> reniform length.
     Maturation is endpoint-anchored allometry: the constants set the baseline metanephros onto the adult kidney.
  2. JACOBIAN d(adult)/d(knob).
  3. INVERSE the loop: a real per-allele kidney-cortex-volume effect back-propagates to the branching knob.

THE LOOP LOCUS (real data both ends): rs807624 (chr2, MYCN/DDX1). Its cortex-volume-INCREASING allele (A)
is the SAME allele that raises nephroblastoma (Wilms tumour) risk, OR 1.33 -- and a Wilms tumour is a
developmental tumour of PERSISTENT NEPHRON PROGENITORS. So the adult kidney-volume signal has a documented
embryonic origin (nephron-progenitor proliferation), read back to the branching/endowment knob. beta from the
UK Biobank abdominal-MRI organ-volume GWAS (Liu 2021 eLife) via the GWAS Catalog.

HONEST SCOPE: kidney cortex volume also tracks post-natal growth and filtration demand; the branching knob is
the embryonic-endowment end of that. Maturation = calibrated allometric first cut, not a fitted trajectory.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.kidney_maturation
"""
import json
import numpy as np

# Adult kidney endpoint anchor (population-typical adult single-kidney values -- VERIFY exact means before paper).
REF = {"kidney_cortex_vol_mL": 110.0, "kidney_length_mm": 110.0}
TRAIT_SD = {"kidney_cortex_vol_mL": 25.0, "kidney_length_mm": 12.0}

TH0 = {"branching": 1.0, "elongation": 1.0}
KNOB_GENE = {
    "branching":  ("MYCN/SIX2/GDNF-RET", "ureteric branching / nephron endowment"),
    "elongation": ("Wnt-PCP / CE", "reniform elongation"),
}
TRAITS = ["kidney_cortex_vol_mL", "kidney_length_mm"]
KNOBS = ["branching", "elongation"]

# REAL per-allele kidney-cortex-volume GWAS effect (GWAS Catalog; Liu 2021 eLife abdominal-MRI organ volumes),
# in SD units of the normalized volume. rs807624-G = -0.06 SD cortex volume; the INCREASING allele (A, the
# Wilms-tumour-risk allele, OR 1.33) is +0.06 SD/allele.
REAL_BETA = [
    dict(gene="MYCN/DDX1", snp="rs807624", allele="A", eaf=0.35, trait="kidney_cortex_vol_mL", knob="branching",
         beta_sd=0.06, pval="2e-18",
         source="Liu 2021 eLife 65554 (GWAS Catalog): +0.06 SD kidney cortex volume per A allele; same allele "
                "raises nephroblastoma risk OR 1.33 (embryonic nephron-progenitor origin)"),
]


def predict_adult(theta=None):
    th = dict(TH0)
    if theta: th.update(theta)
    br = th["branching"] / TH0["branching"]                 # nephron endowment -> cortex mass
    el = th["elongation"] / TH0["elongation"]               # convergent extension -> reniform length
    return {
        "kidney_cortex_vol_mL": REF["kidney_cortex_vol_mL"] * br,
        "kidney_length_mm":     REF["kidney_length_mm"] * el,
    }


def jacobian(eps=0.1):
    J = np.zeros((len(TRAITS), len(KNOBS)))
    for j, k in enumerate(KNOBS):
        h = eps * abs(TH0[k])
        yp, ym = predict_adult({k: TH0[k] + h}), predict_adult({k: TH0[k] - h})
        for i, t in enumerate(TRAITS):
            J[i, j] = (yp[t] - ym[t]) / (2 * h)
    return J


def back_propagate(trait, delta_units, knob):
    J = jacobian()
    s = J[TRAITS.index(trait), KNOBS.index(knob)]
    return None if abs(s) < 1e-9 else delta_units / s


def main():
    base = predict_adult()
    print("Baseline adult kidney (metanephros -> maturation) vs anchor REF:")
    for t in TRAITS:
        print(f"  {t:22s} {base[t]:7.1f}   (ref {REF[t]})")

    J = jacobian()
    print("\nMaturation Jacobian  d(adult)/d(embryonic knob):")
    print(f"{'':23s}" + "".join(f"{k[:11]:>13s}" for k in KNOBS))
    for i, t in enumerate(TRAITS):
        print(f"{t:23s}" + "".join(f"{J[i, j]:13.2f}" for j in range(len(KNOBS))))

    print("\nQuantitative loop with REAL per-allele beta:")
    real = []
    for b in REAL_BETA:
        d_units = b["beta_sd"] * TRAIT_SD[b["trait"]]
        dtheta = back_propagate(b["trait"], d_units, b["knob"])
        rel = 100.0 * dtheta / TH0[b["knob"]]
        print(f"  {b['gene']:10s} {b['snp']}-{b['allele']}  beta={b['beta_sd']:+.3f} SD -> {d_units:+.2f} "
              f"{b['trait']}/allele  ->  d{b['knob']} = {dtheta:+.5f} ({rel:+.2f}% baseline/allele)")
        real.append(dict(gene=b["gene"], snp=b["snp"], allele=b["allele"], trait=b["trait"], knob=b["knob"],
                         beta_sd=b["beta_sd"], delta_trait_units=round(float(d_units), 3),
                         per_allele_dtheta=round(float(dtheta), 6), pct_baseline_per_allele=round(float(rel), 3),
                         source=b["source"]))

    out = dict(organ="kidney", behaviour="branching morphogenesis (nephron endowment)", reference=REF,
               baseline=base, traits=TRAITS, knobs=KNOBS,
               jacobian=[[round(float(v), 4) for v in r] for r in J], real_beta_loop=real,
               scope="Third organ, third embryonic behaviour (branching morphogenesis). rs807624 cortex-volume-"
                     "raising allele = the Wilms-tumour-risk allele (embryonic nephron-progenitor origin). beta "
                     "from Liu 2021 eLife via GWAS Catalog. Maturation = calibrated allometric first cut.")
    path = "data/organ_cascade/kidney_maturation.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
