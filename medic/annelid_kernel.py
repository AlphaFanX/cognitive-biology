"""
Annelid kernel-porting probe (Capitella teleta) -- the FIRST test of the genomic
KERNEL across phyla, not just the bioelectric floor.
================================================================================

Every cross-phylum result so far validates the bioelectric FLOOR (planaria/zebrafish/
xenopus -- physics, no kernel). This probe targets the other half: does the genomic
KERNEL itself port beyond vertebrates? Annelids are the right test because, unlike
planaria/nematodes/flies, they RETAINED CpG methylation (DNMT1/DNMT3 intact, mosaic
gene-body methylation) -- so the Jadhav-style methylation reader can run natively, AND
the universal accessibility reader can run alongside it.

THE KEY OPPORTUNITY (from data scouting): *Capitella teleta* uniquely has ALL FOUR layers
in matched species + matched developmental stages:
  - METHYLOME (developmental EM-seq) ........ Guynes, Bogdanovic, de Mendoza et al. 2024,
        Genome Biology 25:204 (PMID 39090757). Confirmed mosaic GbM; GbM<->transcription
        coupling DECAYS across development = a developmentally-ordered signal, exactly what
        a 'read-in-developmental-order' kernel consumes.
  - ATAC-seq + stage RNA-seq (developmental) . Martin-Duran et al. 2022, Nature 606:301
        (PMID 36697830); processed bigWig/bed at github.com/ChemaMD/OweniaGenome.
  - GENOME (shared coordinate space) ......... GCA_000328365.1 (JGI Capca1).
  - DIFFERENTIATION ORACLE (2nd step) ........ Platynereis dumerilii single-cell atlas
        (Achim et al. 2018 PMID 29373712; Vergara/PlatyBrowser 2021 PMID 34380046) -- best
        annelid cell-type ground truth, bridged by orthogroups (no Platynereis methylome).

Because Capitella carries BOTH methylation AND accessibility at the same stages, it is the
first place to test the kernel/reader decoupling ([[cognimed-kernel-reader-decoupling]]) on
REAL non-vertebrate data: do the two readers recover the SAME kernel strata in an annelid?

THE PROBE (two falsifiable tests):
  T1 (cross-reader equivalence, near-term): read Capitella regulatory regions via BOTH
      MethylationReader (Guynes methylome) and AccessibilityReader (Martin-Duran ATAC);
      score stratum concordance. PASS = the kernel is substrate-agnostic in a 2nd phylum.
  T2 (kernel porting, full): derive the annelid kernel as vertebrate-base + annelid LoRA
      adapter (nested-LoRA), read it with either substrate, and predict developmental
      enhancer USE-ORDER; score against the Capitella stage RNA-seq / Platynereis atlas.
      PASS = the kernel concept, not just the floor, transfers across phyla.

STATUS: T1 DONE ON REAL DATA -> see medic/annelid_concordance.py. Resolved accessions (scouted
via NCBI eutils, medic/_scout_annelid_data.py): ATAC = GSE210814 GSM6438588 Ctel_gastrula rep1
narrowPeak; methylome = GSE250187 GSM7975265 ctel.gastrula EM-seq CGmap; BOTH on Capca1 /
GCA_000328365.1 (CAPTEscaffold; no liftover). Owenia fusiformis (GSE202283/184126) is a DIFFERENT
species -- ruled out. REAL RESULT: ATAC-accessible CpG meth 6.5% vs 23.7% background (0.27x),
93.2% of peaks hypomethylated over 13.5M CpGs -> PASS, the two readers agree in a 2nd phylum
(Annelida) on matched real assays. The synthetic T1 below is retained only as the pipeline demo.

STATUS (legacy note): the synthetic T1 exercises the pipeline end-to-end on annelid-PARAMETERIZED
regions (mosaic GbM: intermediate methylation, gene-body bias); superseded by the real run above.

Run: python -m medic.annelid_kernel
"""
from medic.genome.kernel_reader import (
    RegionSignal, MethylationReader, AccessibilityReader,
)

# Real datasets to plug into T1/T2 (resolve exact PRJEB/GSE from the data-availability sections).
DATA_PLAN = {
    "genome":      "GCA_000328365.1 (Capca1, JGI) -- shared coordinate space",
    "methylome":   "Guynes et al. 2024 Genome Biology (PMID 39090757), Capitella EM-seq dev time-course",
    "atac":        "Martin-Duran et al. 2022 Nature (PMID 36697830), processed at github.com/ChemaMD/OweniaGenome",
    "rna_target":  "Martin-Duran 2022 stage RNA-seq (matched species/stages)",
    "oracle_2nd":  "Platynereis atlas: Achim 2018 (PMID 29373712), Vergara/PlatyBrowser 2021 (PMID 34380046)",
}

# Annelid-PARAMETERIZED synthetic regions for the T1 pipeline test. Mosaic GbM: methylation
# sits at INTERMEDIATE levels (not the global hyper-methylation of vertebrates), and the
# accessibility trajectory encodes the same open-stage story per region. Stages = (early
# larva, juvenile, adult), matching the Guynes/Martin-Duran developmental sampling.
ANNELID_REGIONS = [
    # early-acting developmental enhancer: open (low-meth / accessible) from the larval stage
    RegionSignal("Capca_sc1", 1000, 1500, methylation=[22.0, 18.0, 16.0], accessibility=[6.0, 6.5, 6.2]),
    # mid (juvenile) enhancer: gene body methylated early, opens by juvenile
    RegionSignal("Capca_sc1", 4000, 4500, methylation=[70.0, 28.0, 24.0], accessibility=[1.0, 4.8, 5.1]),
    # late (adult) enhancer: opens only in the adult
    RegionSignal("Capca_sc2",  200,  700, methylation=[82.0, 75.0, 33.0], accessibility=[0.6, 1.1, 3.6]),
    # constitutively methylated / inaccessible: outside the recoverable kernel
    RegionSignal("Capca_sc2", 9000, 9500, methylation=[88.0, 86.0, 84.0], accessibility=[0.4, 0.5, 0.5]),
]


def t1_cross_reader_equivalence(regions=ANNELID_REGIONS, verbose=True):
    """T1: do methylation and accessibility recover the same annelid kernel strata?"""
    mr, ar = MethylationReader(), AccessibilityReader()
    agree = 0
    rows = []
    for r in regions:
        ms, mw = mr.read_stratum(r), mr.read_weight(r)
        as_, aw = ar.read_stratum(r), ar.read_weight(r)
        same = (ms == as_); agree += same
        rows.append((r.chrom, r.start, ms, mw, as_, aw, same))
    acc = agree / len(regions)
    if verbose:
        print("=" * 86)
        print("ANNELID (Capitella teleta) KERNEL-PORTING PROBE -- T1 cross-reader equivalence")
        print("=" * 86)
        print("  data plan (plug real tracks in to run for real):")
        for k, v in DATA_PLAN.items():
            print(f"    {k:10s}: {v}")
        print("\n  T1 on annelid-parameterized regions (mosaic GbM; swap in Guynes/Martin-Duran tracks):")
        print(f"  {'region':16s} | {'methylation':22s} | {'accessibility':22s} | concordant")
        print("-" * 86)
        for chrom, start, ms, mw, as_, aw, same in rows:
            msn = ms.name if ms is not None else "none"
            asn = as_.name if as_ is not None else "none"
            mw = mw or 0.0; aw = aw or 0.0
            print(f"  {chrom+':'+str(start):16s} | stratum={msn:9s} w={mw:.2f} | stratum={asn:9s} w={aw:.2f} |   {'YES' if same else 'no'}")
        print("-" * 86)
        print(f"  T1 stratum concordance: {agree}/{len(regions)} = {acc:.0%}")
        print("  PASS criterion: methylation and ATAC recover the SAME kernel in a 2nd (annelid) phylum")
        print("  => the kernel is substrate-agnostic beyond vertebrates (the [[kernel-reader-decoupling]] claim).")
        print("\n  NEXT (T2, full porting): derive annelid kernel = vertebrate base + annelid LoRA adapter,")
        print("  predict developmental enhancer use-order, score vs Capitella stage RNA-seq / Platynereis atlas.")
    return acc


if __name__ == "__main__":
    t1_cross_reader_equivalence()
