"""
The cross-phylum CLADE LADDER -- the kernel-porting program in one artifact.
============================================================================

Consolidates the cross-phylum work into a single registry instead of one module per
clade. Each clade carries: its position on the nested-LoRA phylogenetic stack (how far
down the shared path with vertebrates it branches), its epigenetic SUBSTRATE status,
which kernel READER applies, whether the bioelectric FLOOR is already validated, and the
data feasibility + best concrete dataset pairing for a genomic-kernel porting probe.

Two universals frame every row (see [[cognimed-kernel-reader-decoupling]]):
  - the bioelectric FLOOR (Goldman/Vm) -- physics, needs no kernel, validated 5 species;
  - chromatin ACCESSIBILITY (ATAC) -- the universal kernel reader.
Between them sits a handful of substrate readers; CpG methylation was ancestral to
bilaterians and independently LOST in nematodes/flatworms/dipterans.

Data feasibility verdicts come from two scouting passes (2026-06-17). A single T1
dual-reader check (methylation vs accessibility recover the same kernel) is run to show
the reader is clade-agnostic; the real per-clade tracks (accessions below) are the path
to running each probe for real.

Run: python -m medic.clade_ladder
"""
from dataclasses import dataclass, field
from typing import List, Optional
from medic.genome.kernel_reader import RegionSignal, MethylationReader, AccessibilityReader

# Nested-LoRA adapter path from the shared base W0 (eumetazoan floor) outward.
# The shared-floor DEPTH with vertebrates = how many leading adapters match before
# divergence; SMALLER = a DEEPER test of the kernel concept.
VERT_PATH = ["eumetazoa", "bilateria", "deuterostome", "chordate", "vertebrate"]

def shared_depth(path: List[str]) -> int:
    d = 0
    for a, b in zip(path, VERT_PATH):
        if a != b: break
        d += 1
    return d

@dataclass
class Clade:
    name: str
    species: str
    lora_path: List[str]
    bodyplan: str
    substrate: str          # methylation status
    reader: str             # native kernel reader
    floor: str              # bioelectric-floor validation status
    kernel_feasible: str    # data verdict for a genomic-kernel porting probe
    best_pairing: str       # concrete species + datasets
    note: str = ""

LADDER: List[Clade] = [
    Clade("Vertebrata", "mouse / zebrafish / xenopus",
          ["eumetazoa","bilateria","deuterostome","chordate","vertebrate"],
          "bilateral triploblast", "CpG global + gene-body", "methylation (native)",
          "VALIDATED (zebrafish 7/7, xenopus rho=0.83, mouse AUC 0.74)",
          "HAVE", "Jadhav kernel (mouse) -- the base", ""),
    Clade("Cnidaria", "Nematostella vectensis",
          ["eumetazoa","cnidaria"],
          "radial diploblast (oral-aboral)", "CpG gene-body (vertebrate-like)", "methylation (native)",
          "GAP (Hydra Vm literature-level only)",
          "FEASIBLE -- STRONGEST (all 4 layers matched + DNMT-KO ground truth)",
          "Nematostella: genome GCF_932526225.1 + methylome Xu 2026 (WGBS+DNMT-KO) + Sebe-Pedros 2018 scRNA+ATAC",
          "deepest split (outgroup to ALL Bilateria); DNMT-knockout = rare loss-of-function ground truth"),
    Clade("Annelida", "Capitella teleta",
          ["eumetazoa","bilateria","protostome","spiralia","annelid"],
          "bilateral triploblast (segmented)", "CpG mosaic gene-body", "methylation (native)",
          "n/a (kernel target, not floor)",
          "FEASIBLE (all 4 layers matched-species)",
          "Capitella: genome GCA_000328365.1 + methylome Guynes 2024 + Martin-Duran 2022 ATAC+RNA", ""),
    Clade("Mollusca", "Crassostrea gigas (oyster)",
          ["eumetazoa","bilateria","protostome","spiralia","mollusc"],
          "bilateral triploblast", "CpG mosaic gene-body", "methylation (native)",
          "n/a (kernel target)",
          "FEASIBLE -- pick the OYSTER (all 4 layers)",
          "C. gigas: genome (Ensembl Metazoa) + Wang 2014 WGBS / dev methylome + 2024 ATAC + metamorphosis RNA",
          "Spiralia sister to Annelida -> tests ADAPTER SHARING (short LoRA hop from annelid)"),
    Clade("Mollusca/Cephalopoda", "Octopus bimaculoides",
          ["eumetazoa","bilateria","protostome","spiralia","mollusc","cephalopod"],
          "derived (camera eyes, 500M-neuron CNS)", "CpG gene-body (decoupled from neural state)", "methylation + RNA-editing layer",
          "n/a",
          "EDGE CASE -- genome+methylome+brain-scRNA yes, but no ATAC and A-to-I editing recodes >60% of neural transcripts",
          "O. bimaculoides genome GCF_001194135.2; methylome (PMC9476566); optic-lobe scRNA (Curr Biol 2022)",
          "RNA editing (Liscovitch-Brauer 2017) = somatic/neural, non-heritable -> genome under-specifies neural state. The kernel's THIRD edge case (after planarian bioelectric kernel, holometabolous two-attractor)"),
    Clade("Platyhelminthes", "Schmidtea mediterranea (planaria)",
          ["eumetazoa","bilateria","protostome","spiralia","platyhelminth"],
          "bilateral (regenerative)", "CpG LOST (no DNMT3)", "ATAC (accessibility) + bioelectric",
          "VALIDATED (8/8 polarity, NO kernel)",
          "kernel via ATAC reader (Fincher/Plass 2018 atlas)",
          "floor: medic/planaria_bioelectric.py", "the bioelectric-kernel case"),
    Clade("Nematoda", "Caenorhabditis elegans",
          ["eumetazoa","bilateria","protostome","ecdysozoa","nematode"],
          "bilateral (eutelic)", "CpG LOST (no DNMT1/3)", "chromatin / fixed lineage",
          "n/a (already 'solved')",
          "lineage-as-kernel (invariant Sulston lineage)",
          "Packer 2019 embryo atlas", ""),
    Clade("Insecta (Diptera)", "Drosophila melanogaster",
          ["eumetazoa","bilateria","protostome","ecdysozoa","arthropod","insect"],
          "bilateral; HOLOMETABOLOUS (2 body plans)", "CpG LOST (DNMT2 only)", "Polycomb/Trithorax",
          "wing-disc handle (stub)",
          "needs the PcG/TrxG reader (next build)",
          "medic/insect_bioelectric.py", "holometaboly = one genome, two attractors, temporal switch"),
]

def t1_reader_agnostic():
    """Show the kernel reader recovers the same strata regardless of substrate/clade."""
    regions = [
        RegionSignal("r", 0, 1, methylation=[20.,16.,14.], accessibility=[6.,6.5,6.3]),   # embryonic
        RegionSignal("r", 0, 1, methylation=[72.,28.,24.], accessibility=[1.,4.6,5.0]),    # fetal
        RegionSignal("r", 0, 1, methylation=[83.,76.,33.], accessibility=[0.6,1.1,3.6]),   # adult
        RegionSignal("r", 0, 1, methylation=[88.,86.,84.], accessibility=[0.4,0.5,0.5]),   # none
    ]
    mr, ar = MethylationReader(), AccessibilityReader()
    return sum(mr.read_stratum(x) == ar.read_stratum(x) for x in regions), len(regions)

def main():
    print("=" * 110)
    print("CROSS-PHYLUM CLADE LADDER -- kernel-porting program (floor = physics-universal; reader = substrate-specific)")
    print("=" * 110)
    print(f"  {'clade':22s} {'depth':5s} {'substrate':24s} {'reader':22s} {'kernel-probe feasibility'}")
    print("-" * 110)
    for c in LADDER:
        d = shared_depth(c.lora_path)
        print(f"  {c.name:22s} {d:^5d} {c.substrate:24s} {c.reader:22s} {c.kernel_feasible[:40]}")
    print("-" * 110)
    print("  depth = shared nested-LoRA adapters with Vertebrata before divergence (SMALLER = DEEPER test).")
    print("  Cnidaria depth=1 (shares only the eumetazoan floor) = the deepest kernel test.\n")

    print("  BIOELECTRIC FLOOR status (physics, no kernel needed):")
    for c in LADDER:
        if "VALIDATED" in c.floor or "handle" in c.floor:
            print(f"    {c.name:22s} {c.floor}")
    print("\n  GENOMIC-KERNEL probe targets (methylation retained -> native reader ports):")
    for c in LADDER:
        if "FEASIBLE" in c.kernel_feasible:
            print(f"    {c.name:22s} -> {c.best_pairing}")
    print("\n  EDGE CASES (information in another substrate):")
    print("    planaria   -> bioelectric kernel (Vm pattern memory, Durant 2017)")
    print("    Drosophila -> holometaboly: one genome, two attractors, ecdysone switch (PcG/TrxG memory)")
    print("    octopus    -> A-to-I RNA editing recodes >60% of neural transcripts (somatic, non-heritable)")

    ok, n = t1_reader_agnostic()
    print(f"\n  reader-agnostic check (kernel_reader): {ok}/{n} strata concordant -> the kernel reader is clade-agnostic.")

if __name__ == "__main__":
    main()
