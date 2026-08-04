"""The developmental loop on a FOURTH organ: the liver -- and the PARTITION shown the other way (Paper #7 Sec 6).

Heart (looping), eye (optic-cup fold), kidney (ureteric branching) each had an adult morphology signal that
traced BACK to an embryonic behaviour. The liver is the deliberate CONTRAST, and it tests the honest limit the
loop is built to respect (Section 5): not every adult-morphology variant acts in the embryo.

The liver's embryonic behaviour is a fourth kind: hepatoblasts delaminate from the hepatic diverticulum and
MIGRATE as cords into the septum transversum (a PROX1-dependent collective migration/EMT), then expand. That
embryonic expansion sets a base liver volume. But the adult trait -- liver volume by abdominal MRI -- is
dominated by a different, POST-natal layer: its genome-wide-significant loci are PPP1R3B (hepatic glycogen
synthesis) and GCKR (metabolic), not developmental genes (Liu 2021 eLife 65554; the developmental hepatoblast
regulators do NOT reach significance for volume). Liver volume is largely GLYCOGEN + LIPID storage, a
maturation-layer / metabolic quantity.

So the loop's partition, applied to the liver, assigns the adult volume signal to the MATURATION layer, not the
embryonic knob -- the same rule that put the kidney's signal in the embryo puts the liver's in maturation. That
is the ontogenetic-timing separation working in both directions, and it is why the loop does not over-claim
that every organ's adult association is embryonic.

This module encodes that partition (it is a classification result, not a knob magnitude: the liver-volume loci
are metabolic-layer parameters, so the loop does not back-propagate them to the embryonic expansion knob).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.liver_maturation
"""
import json

# Adult liver endpoint anchor (population-typical adult liver volume -- VERIFY exact mean before paper).
REF = {"liver_vol_mL": 1500.0}

# Two layers of the liver's adult volume: an EMBRYONIC base (hepatoblast migration/expansion) and a
# MATURATION component (glycogen + lipid storage). The knob names carry their genes.
KNOBS = {
    "hepatoblast_expansion": dict(layer="embryonic", genes="PROX1/hepatoblast migration+proliferation",
                                  role="delamination/cord migration into septum transversum, then expansion"),
    "metabolic_storage":     dict(layer="maturation", genes="PPP1R3B (glycogen) / GCKR (metabolic)",
                                  role="post-natal glycogen + lipid storage sets the bulk of adult volume"),
}

# REAL genome-wide-significant liver-VOLUME loci (Liu 2021 eLife 65554), with the mechanism that decides their
# layer. Both are metabolic-layer parameters -> the partition assigns them to maturation, NOT the embryo.
LIVER_VOLUME_LOCI = [
    dict(gene="PPP1R3B", snp="rs4240624", mechanism="hepatic glycogen synthesis", layer="maturation"),
    dict(gene="GCKR",    snp="rs1260326", mechanism="glucokinase regulation / metabolic", layer="maturation"),
]


def partition(locus):
    """The loop's rule: a locus whose mechanism is a morphogenesis parameter -> embryonic layer; a locus whose
    mechanism is a metabolic/storage parameter -> maturation layer. Returns the assigned knob."""
    metabolic = any(w in locus["mechanism"].lower() for w in ("glycogen", "metabolic", "lipid", "glucokinase"))
    return "metabolic_storage" if metabolic else "hepatoblast_expansion"


def main():
    print(f"Liver adult anchor: {REF['liver_vol_mL']:.0f} mL; two layers of volume:")
    for k, v in KNOBS.items():
        print(f"  {k:22s} [{v['layer']:10s}]  {v['genes']}")

    print("\nPartition of the REAL liver-volume loci (Liu 2021) by mechanism:")
    embryonic, maturation = 0, 0
    for L in LIVER_VOLUME_LOCI:
        knob = partition(L)
        layer = KNOBS[knob]["layer"]
        maturation += layer == "maturation"; embryonic += layer == "embryonic"
        print(f"  {L['gene']:8s} {L['snp']:12s} {L['mechanism']:34s} -> {knob} [{layer}]")
    print(f"\n  {maturation}/{len(LIVER_VOLUME_LOCI)} liver-volume loci -> MATURATION layer; "
          f"{embryonic} -> embryonic.")
    print("  => the loop assigns the adult liver-volume signal to MATURATION, not the embryonic knob.")
    print("  Contrast: the kidney's cortex-volume locus (Wilms/nephron progenitors) partitioned to the EMBRYO.")

    out = dict(organ="liver", behaviour="hepatoblast delamination + cord migration (PROX1), then expansion",
               reference=REF, knobs=KNOBS, liver_volume_loci=[dict(L, assigned_knob=partition(L))
                                                              for L in LIVER_VOLUME_LOCI],
               result="all real liver-volume loci are metabolic-layer -> partitioned to MATURATION, not embryo",
               scope="Fourth organ = the CONTRAST case. Liver volume is dominated by glycogen+lipid storage "
                     "(PPP1R3B/GCKR, Liu 2021), a maturation-layer quantity; no developmental hepatoblast locus "
                     "reaches significance for volume. The loop's ontogeny-vs-maturation partition assigns the "
                     "signal to maturation -- the same rule that put the kidney in the embryo -- so the loop does "
                     "not over-claim that every adult association is embryonic. Classification result, honest.")
    path = "data/organ_cascade/liver_maturation.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
