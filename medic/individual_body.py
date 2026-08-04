"""Individual genotype -> knobs -> surface: A computable human, not THE computable human (Miles 2026-07-26).

Everything so far used the population MEAN. An individual is a vector of polygenic scores on the GWAS-covered
body traits: a genotype gives, per trait, a PGS z-score (deviation from the population mean in SD units), and
the model maps those z-scores to the body-plan knobs, knob = mean + z * sd_in_proportion_units. The genome
thus specifies a SPECIFIC body. The surface metric is scale-invariant, so the shape-relevant individual
variation is in the RATIOS -- leg-to-height (sitting-height ratio), girth (BMI/waist-hip), head fraction --
not overall height (which sets size only); the model builds each individual and we measure how far apart the
individuals are (genome-driven individuality) and how each sits relative to the reference human.

Genotypes here are drawn from the standard-normal PGS distribution (each trait's polygenic score is ~normal by
the many-small-loci architecture) plus two named extremes; each z is genome-grounded in the sense that it is
the polygenic-score axis the loci in the column define (sitting-height ratio 1231, BMI/WHR 2338, head circ 112,
height 7452), and a real genotype would supply the specific z via its allele dosages.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.individual_body
Out:  data/organ_cascade/individual_body.{json,png}
"""
import os, json
import numpy as np
from medic.genome_set_body import fit_proportions, build_labeled
from medic.surface_metric import pca_align, chamfer

HERE = os.path.dirname(__file__)
MH = os.path.join(HERE, "..", "data", "bodybase", "makehuman_decimated.npz")
OUT = os.path.join(HERE, "..", "data", "organ_cascade", "individual_body.json")
FIG = os.path.join(HERE, "..", "data", "organ_cascade", "individual_body.png")

# trait -> (population mean proportion, SD in proportion units, GWAS source, loci)  [armspan = canon, 0 loci]
TRAITS = {
    "leg_fraction":  (0.48, 0.028, "sitting-height ratio / body proportion", 1231),
    "girth_ratio":   (0.13, 0.022, "BMI / waist-hip", 2338),
    "head_fraction": (0.13, 0.009, "head circumference / ICV", 112),
    "armspan_ratio": (1.00, 0.020, "(canon; no GWAS)", 0),
}
# individuals as PGS z-vectors (leg, girth, head, armspan); named extremes + genome-drawn samples
NAMED = {
    "population mean": dict(leg_fraction=0, girth_ratio=0, head_fraction=0, armspan_ratio=0),
    "tall, long-legged, slim": dict(leg_fraction=+1.8, girth_ratio=-1.5, head_fraction=-0.8, armspan_ratio=+0.6),
    "stocky, short-legged": dict(leg_fraction=-1.6, girth_ratio=+1.8, head_fraction=+0.6, armspan_ratio=-0.4),
}


def knobs_from_z(z):
    targets = {t: (TRAITS[t][0] + z.get(t, 0.0) * TRAITS[t][1], "", 0) for t in TRAITS}
    knob, got = fit_proportions(targets, n_iter=35)
    return knob, got


def run():
    rng = np.random.default_rng(7)
    za = np.load(MH, allow_pickle=True); Va = np.asarray(za["V"], float); Aa = pca_align(Va[rng.choice(len(Va), 3000, replace=False)])

    # named + 3 genome-drawn individuals (each trait z ~ N(0,1) = a polygenic-score genotype)
    people = dict(NAMED)
    for i in range(3):
        people[f"genotype {i+1}"] = {t: float(rng.standard_normal()) for t in TRAITS}

    built = {}
    for name, z in people.items():
        knob, got = knobs_from_z(z)
        pts, lab = build_labeled(knob)
        built[name] = dict(z=z, knobs=knob, proportions={k: round(got[k], 3) for k in TRAITS},
                           chamfer_to_ref=round(chamfer(pca_align(pts), Aa, pre_aligned=True), 2), pts=pts, lab=lab)

    names = list(built)
    print("INDIVIDUAL genotype -> body (PGS z-scores on the GWAS-covered traits):\n")
    print(f"{'individual':26s} {'leg':>6s} {'girth':>6s} {'head':>6s}  Chamfer-to-ref")
    for n in names:
        p = built[n]["proportions"]
        print(f"  {n:26s} {p['leg_fraction']:6.2f} {p['girth_ratio']:6.2f} {p['head_fraction']:6.2f}   "
              f"{built[n]['chamfer_to_ref']:.1f}%")
    # pairwise shape distance between individuals = genome-driven individuality
    D = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                D[i, j] = D[j, i] = chamfer(pca_align(built[a]["pts"]), pca_align(built[b]["pts"]))
    mean_between = float(D[np.triu_indices(len(names), 1)].mean())
    print(f"\nmean pairwise surface distance BETWEEN individuals = {mean_between:.1f}% "
          f"(genome-driven individuality); each within ~{np.mean([built[n]['chamfer_to_ref'] for n in names]):.0f}% of the reference.")
    print("So a genotype -- a vector of polygenic scores on the GWAS-covered ratios -- specifies a distinct "
          "individual body: the model computes A human, not only THE human.")

    out = dict(traits={t: dict(mean=TRAITS[t][0], sd=TRAITS[t][1], source=TRAITS[t][2], loci=TRAITS[t][3]) for t in TRAITS},
               individuals={n: dict(z=built[n]["z"], proportions=built[n]["proportions"],
                                    chamfer_to_ref=built[n]["chamfer_to_ref"]) for n in names},
               mean_between_individuals=round(mean_between, 2),
               note="individual = PGS z-vector on the GWAS-covered body ratios -> knobs -> body; height is "
                    "scale-only (Chamfer scale-invariant) so shape individuality is in the ratios; a real "
                    "genotype supplies each z via allele dosages.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(built, names, mean_between)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _figure(built, names, mean_between):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = {0: "#3aa869", 1: "#d64545", 2: "#2a6fb0", 3: "#c98a2b"}
    n = len(names)
    fig, ax = plt.subplots(1, n, figsize=(2.4 * n, 5.2), facecolor="white")
    for a, name in zip(ax, names):
        Y = pca_align(built[name]["pts"]); Y = Y / (np.ptp(Y[:, 0]) + 1e-9)
        for L in (0, 3, 2, 1):
            m = built[name]["lab"] == L
            a.scatter(Y[m, 2], Y[m, 0], s=2, color=cols[L], alpha=0.5)
        a.set_aspect("equal"); a.axis("off")
        p = built[name]["proportions"]
        a.set_title(f"{name}\nleg {p['leg_fraction']:.2f} girth {p['girth_ratio']:.2f}", fontsize=8)
    fig.suptitle(f"Individual genotype -> body: a vector of polygenic scores on the GWAS-covered ratios (leg / "
                 f"sitting-height 1231 loci, girth / BMI 2338, head 112) specifies a distinct person.\n"
                 f"Mean surface distance between individuals {mean_between:.0f}%% -- genome-driven individuality; "
                 f"the model computes A human, not only THE human.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG, dpi=135, facecolor="white")


if __name__ == "__main__":
    run()
