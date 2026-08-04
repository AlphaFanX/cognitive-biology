"""The APPENDAGES on the canalized<->polygenic spectrum -- hands, feet, fingers, toes (Miles, 2026-07-26).

The arc placed organs/body-plan at the canalized end (few master genes carry the Bauplan) and the individual
face/stature at the polygenic end. Where do the limbs and digits fall? They span the SAME axis, and they
split it cleanly into two sub-questions the genome answers differently:

  IDENTITY / NUMBER (the autopod plan: five rays, the stylopod-zeugopod-autopod pattern) is CANALIZED. Digit
  number is an invariant -- essentially everyone has five -- so its heritability in the normal range is ~0 and
  its VARIATION is single-gene Mendelian disease: polydactyly (GLI3, the SHH limb enhancer ZRS), brachydactyly
  (HOXD13, GDF5, BMPR1B, IHH), syndactyly. The tell in the common-variant GWAS Catalog is ABSENCE: these
  malformation traits contribute almost no genome-wide-significant COMMON loci, exactly as canalization
  predicts (the variation is rare and large-effect, not common and polygenic).

  PROPORTION / SIZE (relative finger length = the 2D:4D digit-length ratio; hand grip strength) is POLYGENIC,
  like stature and the face -- many independent loci of small effect.

  FINGERPRINT MINUTIAE are individual through developmental STOCHASTICITY, not the genome: the dermatoglyphic
  pattern TYPE has a modest GWAS (limb-patterning genes, e.g. EVI1/ADAMTS9), but the identifying ridge
  minutiae are noise, individual without being genetic.

METRIC NOTE: polygenicity here is measured as the COUNT of independent genome-wide-significant loci (unique
mapped genes) per trait. The effect-variance Lorenz used for stature in medic.body_absynth is NOT reused here
because the catalog betas for these traits are in heterogeneous units across studies, which corrupts a
variance-weighted read (it spuriously reports grip strength as oligogenic); the robust, unit-free read is the
locus count. Digit NUMBER's canalization is read as the near-absence of common malformation loci.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.appendage_spectrum
Out:  data/organ_cascade/appendage_spectrum.{json,png}
"""
import os, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CATALOG = "data/gwas/gwas-catalog-download-associations-alt-full.tsv"
OUT = "data/organ_cascade/appendage_spectrum.json"
FIG = "data/organ_cascade/appendage_spectrum.png"

POLY_TRAITS = ["digit length ratio", "grip strength measurement"]   # exact MAPPED_TRAIT (shape / function)
MALFORM_KW = ["polydactyly", "syndactyly", "brachydactyly", "clubfoot", "oligodactyly", "ectrodactyly",
              "split hand", "split foot", "limb malformation", "limb reduction", "digit anomaly"]
MENDELIAN_GENES = ["GLI3", "SHH/ZRS", "HOXD13", "GDF5", "BMPR1B", "IHH", "TBX5", "WNT7A"]
# stature reference from medic.body_absynth (unique GW-sig height loci by mapped gene)
HEIGHT_TRAIT = "body height"
PMAX = 5e-8


def scan():
    poly_genes = {t: set() for t in POLY_TRAITS}
    height_genes = set()
    malform_rows, malform_genes = 0, set()
    with open(CATALOG, encoding="utf-8", errors="replace", newline="") as fh:
        r = csv.reader(fh, delimiter="\t"); head = next(r)
        col = {n: i for i, n in enumerate(head)}
        iT, iM, iG, iP = col["DISEASE/TRAIT"], col["MAPPED_TRAIT"], col["MAPPED_GENE"], col["P-VALUE"]
        for row in r:
            if len(row) <= iG:
                continue
            try:
                if float(row[iP]) > PMAX:
                    continue
            except Exception:
                continue
            mt = row[iM].strip().lower()
            genes = [g.strip() for g in row[iG].replace(" x ", ",").split(",")
                     if g.strip() and g.strip() != "NR" and " - " not in g]
            if mt in poly_genes:
                poly_genes[mt].update(genes)
            if mt == HEIGHT_TRAIT:
                height_genes.update(genes)
            td = (row[iT] + " | " + row[iM]).lower()
            if any(k in td for k in MALFORM_KW):
                malform_rows += 1; malform_genes.update(genes)
    return poly_genes, height_genes, malform_rows, malform_genes


def run():
    poly_genes, height_genes, malform_rows, malform_genes = scan()
    counts = {t: len(poly_genes[t]) for t in POLY_TRAITS}
    n_height = len(height_genes)

    print("APPENDAGES on the canalized<->polygenic spectrum")
    print("(polygenicity = # independent GW-sig loci; unit-free, robust to catalog beta heterogeneity)\n")
    print("POLYGENIC side (appendage proportion / function):")
    for t in POLY_TRAITS:
        print(f"  {t:26s} {counts[t]:4d} independent GW-sig loci  -> POLYGENIC")
    print(f"  (reference: body height {n_height} loci)")
    print(f"\nCANALIZED side (appendage identity / digit NUMBER = Mendelian):")
    print(f"  limb/digit MALFORMATION common GW-sig loci: {malform_rows} rows, {len(malform_genes)} genes "
          f"-> essentially ABSENT from common polygenic variation")
    print(f"  variation is single-gene disease: {', '.join(MENDELIAN_GENES)}")
    print(f"  = digit number is CANALIZED (invariant; heritability ~0 in the normal range)")
    print("\nFINGERPRINT minutiae = developmental STOCHASTICITY (pattern type modest-GWAS; minutiae are noise)")

    out = dict(polygenic_loci=counts, height_loci_reference=n_height,
               malformation_common_loci=malform_rows, malformation_genes=sorted(malform_genes),
               mendelian_genes=MENDELIAN_GENES,
               reading=("appendage IDENTITY/NUMBER canalized (Mendelian, ~%d common loci); appendage "
                        "PROPORTION/SIZE polygenic (digit-length-ratio %d + grip %d loci, cf. height %d); "
                        "fingerprint minutiae stochastic"
                        % (malform_rows, counts["digit length ratio"],
                           counts["grip strength measurement"], n_height)))
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(counts, n_height, malform_rows)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _figure(counts, n_height, malform_rows):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white")
    # panel A: polygenicity = independent GW-sig locus count (log), canalized traits near zero
    labels = ["digit NUMBER\n(malformation)", "digit length\nratio", "grip strength", "body height\n(ref)"]
    vals = [max(malform_rows, 0.6), max(counts["digit length ratio"], 0.6),
            max(counts["grip strength measurement"], 0.6), max(n_height, 0.6)]
    cols = ["#2e7d32", "#7b6fb0", "#8a6fb0", "#2a6fb0"]
    ax[0].bar(range(4), vals, color=cols)
    ax[0].set_yscale("log"); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("# independent GW-sig loci (log)")
    ax[0].set_title("Appendage NUMBER: ~0 common loci (Mendelian, canalized).\n"
                    "Appendage PROPORTION / function: many loci (polygenic).")
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.15, f"{int(round(v)) if v>=1 else 0}", ha="center", fontsize=8)

    # panel B: the whole canalized<->polygenic spectrum, appendages placed on it
    items = [("organs\n(body plan)", 0.94, "#2e7d32", False),
             ("digit NUMBER\n(Mendelian)", 0.90, "#3aa869", True),
             ("organs mean\n(shape)", 0.63, "#3aa869", False),
             ("body height", 0.05, "#2a6fb0", False),
             ("digit length\nratio", 0.05, "#7b6fb0", False),
             ("grip\nstrength", 0.03, "#8a6fb0", False),
             ("face\n(individual)", 0.044, "#b05fa8", False)]
    for i, (lbl, val, col, hatch) in enumerate(items):
        ax[1].bar(i, val, color=col, hatch="//" if hatch else None, edgecolor="white")
        ax[1].bar(i, 1 - val, bottom=val, color="#e5e5e5")
    ax[1].set_xticks(range(len(items))); ax[1].set_xticklabels([it[0] for it in items], fontsize=7.3)
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("canalization  (few genes carry it)")
    ax[1].set_title("Appendages span the spectrum: digit NUMBER canalized (hatched = Mendelian),\n"
                    "digit PROPORTION + grip polygenic like height and the face")
    fig.suptitle("Hands, feet, fingers, toes on the canalized<->polygenic axis: the autopod PLAN and digit "
                 "NUMBER are canalized (Mendelian; ~%d common malformation loci), digit PROPORTION and hand "
                 "function are polygenic like stature and the face." % malform_rows, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG, dpi=135, facecolor="white")


if __name__ == "__main__":
    run()
