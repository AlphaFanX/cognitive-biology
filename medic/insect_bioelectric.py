"""
Insect (Drosophila) Bioelectric Stub -- holometaboly + the wing-disc Vm handle
==============================================================================

STATUS: PRELIMINARY STUB (cross-phylum paper, Insecta section). Two purposes:
 1. Encode holometaboly as the framework's 'one genome, two attractors' case.
 2. Give a first falsifiable bioelectric handle on the wing imaginal disc.

Holometaboly (complete metamorphosis): one genome builds a LARVAL body, then -- after
histolysis of most larval tissue in the pupa -- an ADULT body from imaginal discs. In
Cognitive Biology terms this is the cleanest demonstration that morphology is a
CONTROLLABLE ATTRACTOR selected by a TEMPORAL switch (the ecdysone / juvenile-hormone
clock), with Polycomb/Trithorax as the chromatin memory that survives the transition.
NB: Diptera LOST CpG methylation -> the kernel reader here is PcG/TrxG, not methylation
(see cross-phylum substrate map). That reader is the NEXT required build; this file does
not implement it.

Wing-disc bioelectric handle (the testable part): inwardly-rectifying K+ channels
(Irk1/Irk2) influence wing morphogenesis by gating Dpp/BMP release -- a wing-disc analog
of the zebrafish fin-growth axis (K+ -> hyperpolarize -> altered growth/patterning).

Run: python -m medic.insect_bioelectric
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

# ---------------------------------------------------------------------------
# 1. Holometaboly as a two-attractor / temporal-switch scaffold
# ---------------------------------------------------------------------------
@dataclass
class Attractor:
    name: str
    body_plan: str
    source_cells: str
    hormonal_state: str       # ecdysone / JH context that holds this attractor

METAMORPHIC_ATTRACTORS = [
    Attractor("larva", "segmented feeding grub/maggot", "larval differentiated tissue",
              "high juvenile hormone, low ecdysone pulses -> hold larval program"),
    Attractor("pupa", "histolysis + remodeling", "imaginal discs activate; larval tissue lysed",
              "ecdysone pulse with falling JH -> switch"),
    Attractor("adult", "winged reproductive imago", "imaginal discs (set-aside cells)",
              "ecdysone-driven metamorphic program, JH absent"),
]

SWITCH = {
    "control_input": "ecdysone (20E) pulses gated by juvenile hormone (JH)",
    "memory_substrate": "Polycomb/Trithorax (H3K27me3 / H3K4me3) -- NOT methylation (lost in Diptera)",
    "framework_reading": "same frozen kernel, two attractors; JH/ecdysone clock selects which; "
                         "PcG/TrxG carries Hox/positional identity across histolysis",
}

# ---------------------------------------------------------------------------
# 2. Wing-disc bioelectric handle (falsifiable target; same Goldman operator)
# ---------------------------------------------------------------------------
E = {"Na": 60.0, "K": -90.0, "Ca": 120.0, "Cl": -65.0}
G_BASE = {"Na": 1.0, "K": 4.0, "Ca": 0.5, "Cl": 1.5}
GOF, LOF = 3.0, 1.0/3.0

def goldman(g):
    num = g["Na"]*E["Na"] + g["K"]*E["K"] + g["Ca"]*0.1*E["Ca"] + g["Cl"]*E["Cl"]
    den = g["Na"] + g["K"] + g["Ca"]*0.1 + g["Cl"] + 1e-9
    return num/den
V_BASE = goldman(G_BASE)

@dataclass
class WingDiscPerturbation:
    gene: str
    family: str
    mutation_type: str
    expected_effect: str       # published direction, if known
    confidence: str
    source: str

WING_DISC = [
    WingDiscPerturbation("Irk1", "K", "LOF", "depolarization -> altered Dpp release / wing defects",
                         "established", "Dahal et al. 2017 (Development) -- Irk channels gate Dpp"),
    WingDiscPerturbation("Irk2", "K", "LOF", "depolarization -> wing patterning defects",
                         "established", "Dahal et al. 2017"),
    WingDiscPerturbation("Ork1", "K", "GOF", "hyperpolarization -> growth/patterning shift",
                         "related", "open-rectifier K+; wing-disc Vm handle"),
]

def operator_dvm(p: WingDiscPerturbation):
    mult = GOF if p.mutation_type == "GOF" else LOF
    g = dict(G_BASE); g[p.family] = g[p.family]*mult
    return goldman(g) - V_BASE

def report():
    print("=" * 78)
    print("INSECT (Drosophila) BIOELECTRIC STUB -- holometaboly + wing-disc handle")
    print("=" * 78)
    print("\nHOLOMETABOLY: one genome -> two attractors (temporal switch)")
    for a in METAMORPHIC_ATTRACTORS:
        print(f"  [{a.name:5s}] {a.body_plan:34s} <- {a.source_cells}")
        print(f"          hold: {a.hormonal_state}")
    print(f"\n  switch input : {SWITCH['control_input']}")
    print(f"  memory       : {SWITCH['memory_substrate']}")
    print(f"  reading      : {SWITCH['framework_reading']}")

    print("\nWING-DISC Vm HANDLE (same Goldman operator; falsifiable target):")
    print(f"  {'gene':6s} {'fam':3s} {'mut':4s} {'dVm':>7s}  predicted direction")
    print("-" * 78)
    for p in WING_DISC:
        dvm = operator_dvm(p)
        direction = "hyperpolarization" if dvm < 0 else "depolarization"
        print(f"  {p.gene:6s} {p.family:3s} {p.mutation_type:4s} {dvm:+6.1f}  {direction:18s} [{p.confidence}] {p.source}")
    print("\nNEXT BUILD (gating): the Polycomb/Trithorax reader -- required before any insect")
    print("kernel can be derived (methylation absent in Diptera). This stub does not implement it.")

if __name__ == "__main__":
    report()
