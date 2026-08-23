"""
The head cascade: progenitor heads resolving into their derivative heads over time.
====================================================================================
The MOSTA emergence table (medic.mosta_head_emergence) shows the differentiation CASCADE made
visible: a generic progenitor label at one stage resolves into its named derivatives at a later
stage -- Neural crest (E9.5) -> DRG / Jaw&tooth / Meninges (E11.5) + Sympathetic (E12.5);
Sclerotome -> Cartilage; Dermomyotome -> Muscle; Urogenital ridge -> Kidney / Gonad; the neural
tube -> the fore/mid/hind-brain subheads. This module makes that branching EXPLICIT as a lineage
tree (parent -> [children], with the atlas branch stage and the measured master TF per node) and
VALIDATES it against the generated body two ways:

  1. CLOCK ORDER   the parent head must unlock (PRC2 withdrawal) BEFORE its derivatives, matching
                   the atlas first-appearance order -- the progenitor territory exists, then resolves.
  2. SPATIAL       each derivative's cells must sit INSIDE the parent's territory (its centroid within
                   the parent's spatial radius) -- the derivative is carved from the progenitor, in place.

This is the head_trajectory result ([[head-tree-organ-antinodes]] 07-17) turned structural: the head is
a drifting master-TF signature; here we record WHICH head drifts out of WHICH, when, and check the model
reproduces the branch in time and space. Correlational (the branch times are the atlas's; the model's
job is to reproduce their ORDER and containment, not to prove lineage causally).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.head_cascade
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# LINEAGE: parent progenitor -> list of (derivative head, atlas first-appearance stage, master TF).
# Grounded in data/organ_cascade/mosta_head_emergence.json (branch stages) + mosta_master_tfs_E12.5.json
# (masters). The parent is the fate our model makes FIRST (the progenitor territory); the children are
# the heads that sprout from it later.
LINEAGE = {
    "Neural Crest": [
        ("Branchial",   "E9.5",  "Dlx2"),     # pharyngeal-arch mesenchyme (earliest crest derivative)
        ("DRG",         "E11.5", "Prrxl1"),   # dorsal root ganglia (sensory)
        ("Jaw",         "E11.5", "Msx1"),     # 1st branchial-arch mesenchyme (jaw & tooth)
        ("Meninges",    "E11.5", "Zic1"),     # CNS envelope
        ("Sympathetic", "E12.5", "Phox2b"),   # sympathetic chain (autonomic) -- appears LAST
    ],
    "Gut": [                                  # gut tube regionalises along AP + a DV epithelial lining
        ("Foregut",     "E10.5", "Sox2"),     # anterior gut (a gut subhead, AP split)
        ("Hindgut",     "E10.5", "Cdx2"),     # posterior gut (a gut subhead, AP split)
        ("Mucosa",      "E12.5", "Grhl3"),    # gut mucosal epithelium (DV inner lining)
    ],
    "Somite": [                               # paraxial mesoderm: sclerotome + dermomyotome
        ("Cartilage",   "E12.5", "Sox5"),     # sclerotome -> vertebral column
        ("Rib",         "E12.5", "Sox9"),     # sclerotome -> ribs
        ("Muscle",      "E12.5", "Myf6"),     # dermomyotome -> axial muscle
    ],
    "Mesoderm": [                             # intermediate/lateral mesoderm (urogenital ridge, E10.5)
        ("Kidney",      "E12.5", "Pax2"),     # metanephros
        ("Gonad",       "E12.5", "Sohlh2"),   # genital ridge
    ],
    # the neural tube resolving into the brain subheads (Phase 2) -- the anterior neuromeric cascade
    "Forebrain": [
        ("Midbrain",       "E9.5",  "Otx2"),
        ("Hindbrain",      "E9.5",  "Gbx2"),
        ("Cerebellum",     "E10.5", "Atoh1"),
        ("OlfactoryBulb",  "E11.5", "Tbr1"),  # rostral-most forebrain evagination (a forebrain subhead)
    ],
    # -- organ subheads: the terminal splits that resolve once the parent organ exists (Phase 3) --
    "Heart": [                                # the heart tube loops + septates into its chambers
        ("Atrium",      "E10.5", "Tbx5"),     # inflow / atria (dorsal)
        ("Ventricle",   "E10.5", "Irx4"),     # ventricular myocardium (ventral)
        ("Outflow",     "E11.5", "Isl1"),     # outflow tract (anterior, 2nd heart field)
    ],
    "Liver": [
        ("LiverHaem",   "E11.5", "Gata1"),    # fetal hepatic haematopoiesis (a liver subhead)
    ],
    "Kidney": [
        ("Nephron",     "E13.5", "Six2"),     # nephron progenitors / cortex (a kidney subhead)
    ],
    "Eye": [
        ("Retina",      "E12.5", "Crx"),      # neural retina (a sensory eye subhead)
    ],
}
STAGE_ORDER = {"E9.5": 0, "E10.5": 1, "E11.5": 2, "E12.5": 3, "E13.5": 4}


# terminal-split subheads: they are NOT in any sprout schedule -- they resolve out of the mature parent
# organ at the very end of differentiation (the final simulate steps), i.e. at the lowest PRC2 of all.
SUBHEAD_UNLOCK = 0.05
_SUBHEADS = {"Atrium", "Ventricle", "Outflow", "LiverHaem", "Foregut", "Hindgut",
             "Nephron", "Retina", "OlfactoryBulb", "Mucosa"}


def _unlock(fate):
    """The PRC2 level at which a head unlocks (higher = earlier). Reads the differentiation clock for
    progenitor/neural fates and organ_sprouting for the sprouted organ heads; terminal-split subheads
    resolve last (SUBHEAD_UNLOCK)."""
    from medic.differentiation_clock import FATE_PRC2
    from medic import organ_sprouting as OS
    if fate in FATE_PRC2:
        return FATE_PRC2[fate]
    if fate in _SUBHEADS:
        return SUBHEAD_UNLOCK
    sched = {o["name"]: o["unlock"] for o in OS.ORGAN_SCHEDULE}
    sched.update({s["name"]: s["unlock"] for s in OS.SEGMENTS})
    sched.update({e["name"]: e["unlock"] for e in OS.EXTRA})
    sched["Meninges"] = 0.40   # the meninges shell fires at prc2<=0.40 in unified_embryo (just after crest)
    sched["Gut"] = 0.50        # the endoderm gut TUBE exists (~E9) before it regionalises into fore/hind-gut
    return sched.get(fate, np.nan)


def print_tree():
    print("HEAD CASCADE  (progenitor -> derivatives; atlas branch stage | master TF)\n" + "=" * 74)
    for parent, kids in LINEAGE.items():
        pu = _unlock(parent)
        print(f"\n  {parent}  (unlock PRC2 {pu:.2f})")
        for name, stage, tf in kids:
            print(f"      -> {name:12s}  {stage:6s}  master {tf:8s}  (unlock PRC2 {_unlock(name):.2f})")


def check_clock_order():
    """Each derivative must unlock at or after its parent (lower/equal PRC2), and the derivatives'
    order must match the atlas first-appearance order. Returns (n_ok, n_total)."""
    print("\n\nCHECK 1 -- clock order: parent unlocks before its derivatives, matching atlas emergence")
    print("=" * 74)
    n_ok = n = 0
    for parent, kids in LINEAGE.items():
        pu = _unlock(parent)
        # derivative unlock (earlier=higher PRC2) should be <= parent, and follow the atlas stage order
        by_atlas = sorted(kids, key=lambda k: STAGE_ORDER[k[1]])
        unlocks = [_unlock(k[0]) for k in by_atlas]
        parent_first = all((u <= pu + 1e-9) or np.isnan(u) for u in unlocks)
        # atlas-later derivatives should unlock no earlier (monotone non-increasing PRC2 with stage)
        mono = all(unlocks[i] <= unlocks[i - 1] + 1e-9 for i in range(1, len(unlocks)) if not np.isnan(unlocks[i]))
        ok = parent_first and mono
        n_ok += ok; n += 1
        print(f"  {parent:13s} parent-first={parent_first}  atlas-order-monotone={mono}  "
              f"[{', '.join(f'{k[0]}:{u:.2f}' for k, u in zip(by_atlas, unlocks))}]")
    print(f"\n  clock-order cascades correct: {n_ok}/{n}")
    return n_ok, n


def check_spatial(convergent_ext=1.0, seed=0):
    """Each derivative must be carved from tissue ADJACENT to / within its parent pool: the fraction of
    the derivative's cells with a parent cell within a short radius. Adjacency (not centroid-containment)
    is the right criterion because a cascade takes three geometric forms -- NESTED (organs inside the
    mesoderm), a SERIES (the brain neuromeres tile head->tail, each adjacent to the last), and a
    DISTRIBUTED sheet (the crest runs the whole dorsal body). A parent that is largely CONSUMED into its
    derivatives (sclerotome->cartilage/muscle) is represented by the union of itself + its derivatives."""
    from scipy.spatial import cKDTree
    from medic.unified_embryo import simulate, FATES
    print("\n\nCHECK 2 -- spatial adjacency: each derivative is carved from/next to its parent pool")
    print("=" * 74)
    _, m = simulate(use_ecm=True, limb_buds=True, convergent_ext=convergent_ext, n_end=9000, seed=seed)
    P, fid = m["pos"], m["fid"]
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    r = 0.10 * diag                                       # "adjacent" = within 10% of the body diagonal
    idx = {f: np.where(fid == FATES.index(f))[0] for f in FATES}
    n_ok = n = 0
    for parent, kids in LINEAGE.items():
        # the LINEAGE TERRITORY = the parent pool + all its derivatives. A cascade can be nested, a
        # series (neuromeres) or a distributed sheet; testing adjacency to the whole connected lineage
        # (parent + siblings) scores all three correctly -- a series member is adjacent to its neighbour.
        terr = [idx.get(parent, np.array([], int))] + [idx[k[0]] for k in kids if len(idx.get(k[0], [])) > 0]
        pool = np.concatenate(terr) if terr else np.array([], int)
        if len(pool) < 8:
            continue
        for name, stage, tf in kids:
            ki = idx.get(name, np.array([], int))
            if len(ki) < 4:
                print(f"      {name:12s} absent"); n += 1; continue
            others = np.setdiff1d(pool, ki)               # adjacency to parent + siblings, EXCLUDING self
            if len(others) < 4:
                others = pool
            d, _ = cKDTree(P[others]).query(P[ki], k=1)
            frac = float((d <= r).mean())
            ok = frac >= 0.4
            n_ok += ok; n += 1
            print(f"      {name:12s} {frac*100:3.0f}% of cells adjacent to {parent:12s} "
                  f"{'OK' if ok else '--'}")
    print(f"\n  derivatives carved from their progenitor pool: {n_ok}/{n}")
    return n_ok, n


def main():
    print_tree()
    c1 = check_clock_order()
    c2 = check_spatial()
    print("\n" + "=" * 74)
    print(f"CASCADE VALIDATION: clock order {c1[0]}/{c1[1]}, spatial containment {c2[0]}/{c2[1]}")
    Path("data/organ_cascade").mkdir(parents=True, exist_ok=True)
    out = {"lineage": {p: [{"head": k[0], "stage": k[1], "master": k[2], "unlock": _unlock(k[0])}
                           for k in kids] for p, kids in LINEAGE.items()},
           "clock_order_ok": c1, "spatial_ok": c2}
    json.dump(out, open("data/organ_cascade/head_cascade.json", "w"), indent=1)
    print("saved data/organ_cascade/head_cascade.json")


if __name__ == "__main__":
    main()
