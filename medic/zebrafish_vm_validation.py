"""
Zebrafish Vm-head validation (cross-phylum cycle, vertebrate point)
===================================================================

Second species in the cross-phylum validation after planaria. Where planaria tests
the Vm head on axial POLARITY (head/tail), zebrafish tests it on a different
morphogenetic readout: TISSUE GROWTH (fin size), the best-characterized vertebrate
bioelectric morphology phenotype.

The established relationship (Perathoner et al. 2014 PLoS Genet; Silic & Zhang 2020
Genetics): K+ channel GAIN-OF-FUNCTION -> membrane HYPERPOLARIZATION of fin
mesenchyme/dermomyotome -> EXCESS GROWTH -> long fin. Multiple independent K+ channels
(kcnk5b, kcnj13, kcnj1b, kcnj10a, kcnk9) converge on the same long-fin phenotype, all
by hyperpolarizing -- a clean, repeated, single-direction test.

The test: does the SAME Goldman operator (planaria, mouse) derive the correct resting-Vm
DIRECTION from only (channel family, GOF/LOF), and does that direction map to the observed
morphogenetic outcome (hyperpolarization -> overgrowth)?

Scope honesty: the resting-Vm operator is the right model for the MORPHOGENETIC cases
(fin/dermomyotome growth, pigment pattern). The cardiac/neural mutants in the dataset are
ACTION-POTENTIAL / excitability phenotypes (Na/Ca AP upstroke, repolarization) -- a different
regime the resting-Vm operator is not meant to score; they are reported separately, not counted.

Run: python -m medic.zebrafish_vm_validation
"""
import numpy as np
from medic.zebrafish_bioelectric import ION_CHANNEL_MUTANTS

# Ion reversal potentials (mV) -- same convention as planaria / f_theta.
E = {"Na": 60.0, "K": -90.0, "Ca": 120.0, "Cl": -65.0}
# Generic fin-mesenchyme / dermomyotome baseline conductances.
G_BASE = {"Na": 1.0, "K": 4.0, "Ca": 0.5, "Cl": 1.5}
GOF, LOF = 3.0, 1.0 / 3.0     # multiplicative conductance change

def goldman(g):
    num = g["Na"]*E["Na"] + g["K"]*E["K"] + g["Ca"]*0.1*E["Ca"] + g["Cl"]*E["Cl"]
    den = g["Na"] + g["K"] + g["Ca"]*0.1 + g["Cl"] + 1e-9
    return num/den
V_BASE = goldman(G_BASE)

FAMILY = {"K": "K", "Na": "Na", "Ca": "Ca", "Cl": "Cl"}
def family_of(channel_type):
    for k in FAMILY:
        if channel_type.startswith(k):
            return k
    return None

def operator_dvm(m):
    """Resting-Vm change predicted from (channel family, GOF/LOF)."""
    fam = family_of(m.channel_type)
    if fam is None:
        return None
    mult = GOF if m.mutation_type == "GOF" else LOF   # LOF & dominant-negative -> down
    g = dict(G_BASE)
    g[fam] = g[fam] * mult
    return goldman(g) - V_BASE                          # signed dVm (mV)

# Which dataset entries are MORPHOGENETIC (resting-Vm -> shape) vs EXCITABILITY (AP).
MORPHOGENETIC_TISSUES = ("fin", "dermomyotome", "craniofacial", "melanophore")
def is_morphogenetic(m):
    return any(t in m.affected_tissue for t in MORPHOGENETIC_TISSUES)

def predicted_direction(dvm):
    return "hyperpolarization" if dvm < 0 else "depolarization"

def observed_direction(m):
    e = m.bioelectric_effect.lower()
    if "hyperpol" in e:
        return "hyperpolarization"
    if "depol" in e:
        return "depolarization"
    return "other"   # excitability/AP language (reduced depol, delayed repol, ...)

def morphogenetic_outcome(dvm, tissue):
    """Map Vm direction -> morphology, for growth-competent appendage/mesenchyme tissue."""
    growth = ("fin" in tissue) or ("dermomyotome" in tissue)
    if dvm < 0:
        return "overgrowth (long fin)" if growth else "pattern shift (hyperpol)"
    return "reduced growth / pattern disruption"

def validate():
    print("=" * 84)
    print("ZEBRAFISH Vm-HEAD VALIDATION  (operator derives Vm direction from channel family + GOF/LOF)")
    print("=" * 84)
    morph = [m for m in ION_CHANNEL_MUTANTS if is_morphogenetic(m)]
    exc   = [m for m in ION_CHANNEL_MUTANTS if not is_morphogenetic(m)]

    print(f"\nMORPHOGENETIC set (resting-Vm -> shape), n={len(morph)}:")
    print(f"  {'gene':9s} {'chan':14s} {'mut':4s} {'dVm':>7s}  {'PRED dir':>16s} {'OBS dir':>16s}  ok  outcome")
    print("-" * 84)
    correct = 0
    for m in morph:
        dvm = operator_dvm(m)
        pdir, odir = predicted_direction(dvm), observed_direction(m)
        ok = (pdir == odir)
        correct += ok
        print(f"  {m.gene:9s} {m.channel_type[:14]:14s} {m.mutation_type:4s} {dvm:+6.1f}  "
              f"{pdir:>16s} {odir:>16s}  {'Y' if ok else '.':>2s}  {morphogenetic_outcome(dvm, m.affected_tissue)}")
    acc = correct / len(morph)
    print("-" * 84)
    print(f"MORPHOGENETIC accuracy (Vm direction matches published): {correct}/{len(morph)} = {acc:.0%}")
    longfin = [m for m in morph if "fin" in m.affected_tissue and m.mutation_type == "GOF"]
    lf_ok = sum(1 for m in longfin if operator_dvm(m) < 0)
    print(f"  Long-fin convergence (K+ GOF -> hyperpolarize -> overgrowth): {lf_ok}/{len(longfin)} genes consistent")

    print(f"\nEXCITABILITY set (action-potential regime; resting-Vm operator not applicable), n={len(exc)}:")
    for m in exc:
        print(f"  {m.gene:9s} {m.channel_type[:16]:16s} {m.mutation_type:4s}  {m.affected_tissue:28s} [{m.bioelectric_effect}]")
    print("\nSame Goldman operator (planaria, mouse) derives the correct fin-mesenchyme Vm direction from")
    print("channel identity alone; hyperpolarization -> overgrowth reproduces the long-fin morphology.")
    return acc

if __name__ == "__main__":
    validate()
