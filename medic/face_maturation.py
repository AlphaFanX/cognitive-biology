"""The developmental loop on the FACE -- the archetype, and the extension from a scalar to a SHAPE (Paper #7 Sec 6).

The face is where the developmental loop was first documented: adult facial-shape association signals are
enriched in EMBRYONIC cranial-neural-crest enhancers (Claes 2018), so the adult signal tracing back to embryonic
regulation is the loop itself. It also extends the loop in two ways beyond the four organs:

  1. FROM A SCALAR TO A SHAPE. The organs gave one number each (a volume, a length). The face is a multi-part
     SHAPE measured by geometric morphometrics -- landmark displacements from the mean face -- so the loop here
     runs on a shape coordinate, not a scalar.
  2. THE EMBRYONIC KNOB IS THE ELECTRIC-FACE FRAME. The facial prominences (frontonasal, maxillary, mandibular)
     sit on the antinodes of the face's own gap-junction eigenmodes -- the RECURSION, the same frame machinery
     one scale down (prominences on the face frame as organs sit on the body frame; Papers #2/#4/#9). Cranial
     neural crest migrates into the prominences and their outgrowth sculpts the face.

The loop: a facial-shape association effect (a landmark displacement in mm) back-propagates through the
craniofacial-growth map to the embryonic prominence displacement it implies. THE LOOP LOCUS: PAX3, a neural-crest
gene, contributes +0.388 mm to a nasal (nasion) facial distance per G allele (facial-shape GWAS via the GWAS
Catalog); its embryonic origin is documented both as a neural-crest gene and by the CNC-enhancer enrichment of
facial-shape signals as a class (Claes 2018).

HONEST SCOPE: the face is a shape, and here the loop is run on ONE landmark (nasion) with a single-locus mm
effect; the prominence knob is the electric-face-frame antinode it sits on; the craniofacial-growth factor
(embryonic prominence -> adult landmark) is a schematic allometric scaling, not a fitted trajectory.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.face_maturation
"""
import json

# The face works in DISPLACEMENTS from the mean face (geometric morphometrics), not absolute organ volumes.
# Craniofacial growth scales an embryonic prominence displacement to the adult landmark displacement; the face
# grows ~3x linearly from the prominence stage to the adult (schematic allometric factor).
GROWTH = 3.0

# Embryonic knob: the frontonasal-prominence position on the electric-face frame (neural-crest territory,
# PAX3). Adult trait: nasion landmark displacement (mm from the mean face).
KNOB_GENE = {"frontonasal_prominence": ("PAX3 / cranial neural crest", "frontonasal antinode on the electric-face frame")}

# REAL per-allele facial-shape effect (GWAS Catalog): PAX3 rs7559271-G, +0.388 mm on a nasion facial distance.
REAL_BETA = [
    dict(gene="PAX3", snp="rs7559271", allele="G", trait="nasion_disp_mm", knob="frontonasal_prominence",
         beta_mm=+0.388, pval="4e-16",
         source="facial-shape GWAS (GWAS Catalog): +0.388 mm nasion per G allele; PAX3 = neural crest; "
                "facial-shape signals enriched in embryonic cranial-neural-crest enhancers (Claes 2018)"),
]


def adult_from_prominence(prominence_disp_mm):
    """Craniofacial growth: embryonic prominence displacement -> adult landmark displacement."""
    return GROWTH * prominence_disp_mm


def back_propagate(adult_disp_mm):
    """The loop: adult facial-landmark displacement -> the embryonic prominence displacement it implies."""
    return adult_disp_mm / GROWTH


def main():
    print("The face: the loop on a SHAPE landmark (displacement from the mean face), not a scalar volume.")
    print(f"Embryonic knob: {KNOB_GENE['frontonasal_prominence'][0]} "
          f"({KNOB_GENE['frontonasal_prominence'][1]})")
    print(f"Craniofacial growth factor (prominence -> adult landmark): x{GROWTH:.0f}\n")

    print("Quantitative loop with REAL per-allele facial-shape beta (mm):")
    real = []
    for b in REAL_BETA:
        d_prom = back_propagate(b["beta_mm"])
        print(f"  {b['gene']:5s} {b['snp']}-{b['allele']}  {b['beta_mm']:+.3f} mm adult {b['trait']}/allele  ->  "
              f"embryonic {b['knob']} displacement {d_prom:+.4f} mm/allele")
        real.append(dict(gene=b["gene"], snp=b["snp"], allele=b["allele"], trait=b["trait"], knob=b["knob"],
                         adult_beta_mm=b["beta_mm"], embryonic_prominence_disp_mm=round(float(d_prom), 4),
                         source=b["source"]))

    out = dict(structure="face", behaviour="cranial-neural-crest migration into facial prominences",
               extends="scalar -> multi-part SHAPE (geometric morphometrics); knob = prominence on the "
                       "electric-face eigenframe (recursion)", growth_factor=GROWTH, real_beta_loop=real,
               scope="Archetype of the loop (facial-shape signals enriched in embryonic CNC enhancers, Claes "
                     "2018). Run on one landmark (nasion) with a single-locus mm effect; prominence knob = its "
                     "electric-face antinode; craniofacial-growth factor is a schematic allometric scaling.")
    path = "data/organ_cascade/face_maturation.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")
    print("\nThe adult facial-shape signal reads back to an embryonic neural-crest/prominence displacement:")
    print("  the loop, on the structure where it was first documented, now extended from a scalar to a shape.")


if __name__ == "__main__":
    main()
