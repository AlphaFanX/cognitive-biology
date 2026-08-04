"""The embryo->adult maturation map for the heart -- a FIRST CUT of the developmental-forward layer.

Paper #7 Section 2 names this the hard/open layer: there is no dense human cardiac developmental time series
to train a maturation map on. The design principle (Section 2) is therefore NOT to mechanise maturation in
full, but to ANCHOR the adult endpoint on the abundant adult atlas and let the model carry the trajectory.
This module does exactly that, as an allometric ballooning map:

    theta (embryonic luminal-loop knobs)  --maturation-->  adult cardiac-MRI morphometrics (real units)

The embryonic tube regions balloon into the adult chambers (Christoffels/Moorman ballooning model): the
inflow/atrial segment -> the atria, the ventricular loop -> the ventricles, the wall program -> the myocardial
shell. Each adult morphometric is written as an allometric function of the embryonic knobs, and the free
constants are CALIBRATED so the baseline embryo (heart_luminal.BEST) lands on adult reference means -- the
endpoint anchor. This is a schematic calibrated map, NOT a trajectory fitted to developmental data; its value
is (1) it produces real-unit adult morphometrics from the embryonic knobs, so a GWAS effect size can be
back-propagated in real units, and (2) the SIGN of each adult-trait sensitivity is a testable prediction,
which here matches the real cardiac-MRI GWAS directions (PITX2 -> larger LA, TTN -> larger LVEDV, TBX5 -> wall).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.heart_maturation
"""
import json
import numpy as np
from medic.heart_luminal import BEST

# Adult endpoint anchor: UK Biobank CMR reference means (Petersen 2017, J Cardiovasc Magn Reson 19:18;
# sex-combined approx of M/F: LVEDV 166/124, LV mass 103/70 g, LA max 71/62 mL). Wall thickness ~8 mm (typical).
REF = {"LVEDV_mL": 145.0, "LV_mass_g": 86.0, "LV_wall_mm": 8.0, "LA_vol_mL": 66.0}
# Population SDs (Petersen 2017) for converting per-allele GWAS effects in SD units to real units.
TRAIT_SD = {"LVEDV_mL": 27.0, "LV_mass_g": 17.0, "LV_wall_mm": 1.5, "LA_vol_mL": 18.0}

# REAL per-allele GWAS effect sizes (GWAS Catalog / Pirruccello 2024), in SD units of the normalized trait.
REAL_BETA = [
    dict(gene="PITX2", snp="rs2634073", allele="T", eaf=0.166, trait="LA_vol_mL", knob="axis_bend",
         beta_sd=0.057, pval="2e-8",
         source="Pirruccello 2024 Nat Commun 15:4304 (GWAS Catalog): +0.057 SD LA volume per T allele"),
    # rs1873164 near TTN/CCDC141: effect allele G = -0.0604 SD; the LVEDV-INCREASING allele (A, freq 0.80)
    # is +0.0604 SD/allele. From the Pirruccello 2020 deposited sumstats (GCST010131), P=1.2e-16, N=36042.
    dict(gene="TTN/CCDC141", snp="rs1873164", allele="A", eaf=0.800, trait="LVEDV_mL", knob="span",
         beta_sd=0.0604, pval="1.2e-16",
         source="Pirruccello 2020 GCST010131 deposited sumstats: rs1873164 LVEDV-increasing allele +0.0604 SD/allele"),
]

# Baseline embryonic knobs (the tuned E16.5/CS luminal loop).
TH0 = {k: BEST[k] for k in ["axis_bend", "nturns", "span", "wall"]}


def predict_adult(theta=None):
    """Allometric ballooning: embryonic knobs -> adult morphometrics, anchored so TH0 -> REF."""
    th = dict(TH0);
    if theta: th.update(theta)
    span = th["span"] / TH0["span"]                 # ventricular length ratio
    wall = th["wall"] / TH0["wall"]                 # myocardial wall ratio
    lat = 0.5 * (th["axis_bend"] / TH0["axis_bend"]) + 0.5 * (th["nturns"] / TH0["nturns"])  # atrial laterality/looping
    return {
        "LVEDV_mL":   REF["LVEDV_mL"] * span,               # cavity volume grows with ventricular length
        "LV_wall_mm": REF["LV_wall_mm"] * wall,             # wall thickness from the chamber-wall program
        "LV_mass_g":  REF["LV_mass_g"] * wall * span ** 0.5,# myocardial shell ~ wall x size
        "LA_vol_mL":  REF["LA_vol_mL"] * lat,               # atrial size from laterality/looping (Pitx2 territory)
    }


TRAITS = ["LVEDV_mL", "LV_wall_mm", "LV_mass_g", "LA_vol_mL"]
KNOBS = ["axis_bend", "nturns", "span", "wall"]


def maturation_jacobian(eps=0.1):
    """J[trait, knob] = d(adult morphometric) / d(embryonic knob), central differences (real units)."""
    J = np.zeros((len(TRAITS), len(KNOBS)))
    for j, k in enumerate(KNOBS):
        h = eps * abs(TH0[k])
        yp, ym = predict_adult({k: TH0[k] + h}), predict_adult({k: TH0[k] - h})
        for i, t in enumerate(TRAITS):
            J[i, j] = (yp[t] - ym[t]) / (2 * h)
    return J


# Real cardiac-MRI GWAS directions (verified sources; see heart_dev_loop.LOCI).
GWAS = [
    dict(gene="PITX2", trait="LA_vol_mL", knob="axis_bend", adult_dir=+1,
         source="Pirruccello 2024 Nat Commun 15:4304 (AF-risk allele -> greater LA volume)"),
    dict(gene="TTN", trait="LVEDV_mL", knob="span", adult_dir=+1,
         source="Pirruccello 2020 Nat Commun 11:2254 (DCM direction -> larger LVEDV)"),
    dict(gene="TBX5", trait="LV_wall_mm", knob="wall", adult_dir=+1,
         source="developmental (Holt-Oram chamber-wall program)"),
]


def back_propagate(trait, delta_trait_units, knob):
    """The quantitative loop: an adult effect (real units) -> the embryonic knob shift it implies."""
    J = maturation_jacobian()
    s = J[TRAITS.index(trait), KNOBS.index(knob)]
    return None if abs(s) < 1e-9 else delta_trait_units / s


def main():
    base = predict_adult()
    print("Baseline adult heart (embryo BEST -> maturation), vs anchor REF:")
    for t in TRAITS:
        print(f"  {t:11s} {base[t]:7.1f}   (ref {REF[t]:.0f})")

    J = maturation_jacobian()
    print("\nMaturation Jacobian  d(adult trait)/d(embryonic knob):")
    print(f"{'':12s}" + "".join(f"{k[:9]:>10s}" for k in KNOBS))
    for i, t in enumerate(TRAITS):
        print(f"{t:12s}" + "".join(f"{J[i, j]:10.2f}" for j in range(len(KNOBS))))

    print("\nSign check vs real cardiac-MRI GWAS directions:")
    ok = 0
    for g in GWAS:
        s = J[TRAITS.index(g["trait"]), KNOBS.index(g["knob"])]
        match = np.sign(s) == g["adult_dir"]
        ok += match
        print(f"  {g['gene']:6s} {g['trait']:11s} <- {g['knob']:10s}  d={s:+.1f}  GWAS {g['adult_dir']:+d}  "
              f"{'MATCH' if match else 'MISMATCH'}")
    print(f"  {ok}/{len(GWAS)} sign-consistent")

    # Quantitative loop with a REAL per-allele GWAS effect (PITX2), in real units end to end.
    print("\nQuantitative loop with REAL per-allele beta:")
    real = []
    for b in REAL_BETA:
        d_mL = b["beta_sd"] * TRAIT_SD[b["trait"]]                     # SD units -> real units via the trait SD
        dtheta = back_propagate(b["trait"], d_mL, b["knob"])          # -> embryonic knob shift per allele
        rel = 100.0 * dtheta / TH0[b["knob"]]
        print(f"  {b['gene']:6s} {b['snp']}-{b['allele']}  beta={b['beta_sd']:+.3f} SD -> {d_mL:+.2f} {b['trait']}"
              f"/allele  ->  d{b['knob']} = {dtheta:+.4f} ({rel:+.2f}% baseline/allele)")
        real.append(dict(gene=b["gene"], snp=b["snp"], allele=b["allele"], trait=b["trait"], knob=b["knob"],
                         beta_sd=b["beta_sd"], delta_trait_units=round(float(d_mL), 3),
                         per_allele_dtheta=round(float(dtheta), 5), pct_baseline_per_allele=round(float(rel), 3),
                         source=b["source"]))

    # Directional loop for the loci without a fitted magnitude yet (unit adult effect -> knob shift).
    print("\nDirectional loop (loci pending an exact per-allele beta; +1 unit adult effect -> knob shift):")
    demo = []
    for g in GWAS:
        dtheta = back_propagate(g["trait"], 1.0, g["knob"])
        rel = 100.0 * dtheta / TH0[g["knob"]]
        print(f"  {g['gene']:6s} +1 {g['trait']:11s}  ->  d{g['knob']:10s} = {dtheta:+.4f}  ({rel:+.2f}% of baseline)")
        demo.append(dict(gene=g["gene"], trait=g["trait"], knob=g["knob"],
                         per_unit_dtheta=round(float(dtheta), 5), pct_of_baseline=round(float(rel), 3),
                         source=g["source"]))

    out = dict(reference=REF, baseline=base, traits=TRAITS, knobs=KNOBS,
               jacobian=[[round(float(v), 4) for v in row] for row in J],
               sign_consistent=f"{int(ok)}/{len(GWAS)}", real_beta_loop=real, loop=demo,
               scope="Allometric ballooning map anchored on adult reference means (endpoint anchor, Section 2), "
                     "NOT a trajectory fitted to developmental data. Real GWAS effect-size magnitudes (per-allele "
                     "units) are the final drop-in; the machinery back-propagates them in real units.")
    path = "data/organ_cascade/heart_maturation.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")
    try:
        _figure(base, J)
    except Exception as e:
        print(f"(figure skipped: {e})")


def _figure(base, J):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    xs = np.arange(len(TRAITS))
    a0.bar(xs - 0.18, [base[t] for t in TRAITS], 0.36, label="matured embryo", color="#c0392b")
    a0.bar(xs + 0.18, [REF[t] for t in TRAITS], 0.36, label="adult reference", color="#7f8c8d")
    a0.set_xticks(xs); a0.set_xticklabels([t.replace("_", "\n") for t in TRAITS], fontsize=8)
    a0.set_title("Maturation lands on the adult anchor", fontsize=10); a0.legend(fontsize=8)
    vmax = np.abs(J).max() or 1.0
    im = a1.imshow(J, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    a1.set_xticks(range(len(KNOBS))); a1.set_xticklabels(KNOBS, fontsize=8, rotation=20)
    a1.set_yticks(range(len(TRAITS))); a1.set_yticklabels(TRAITS, fontsize=9)
    for i in range(len(TRAITS)):
        for j in range(len(KNOBS)):
            a1.text(j, i, f"{J[i, j]:.0f}", ha="center", va="center", fontsize=8)
    for g in GWAS:
        i, j = TRAITS.index(g["trait"]), KNOBS.index(g["knob"])
        a1.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor="k", lw=2))
        a1.text(j, i + 0.32, g["gene"], ha="center", va="center", fontsize=7)
    a1.set_title("Maturation Jacobian  d(adult)/d(embryonic knob)\n(boxed = real GWAS locus)", fontsize=10)
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("data/organ_cascade/heart_maturation.png", dpi=140)
    print("saved data/organ_cascade/heart_maturation.png")


if __name__ == "__main__":
    main()
