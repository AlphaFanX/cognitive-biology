"""
The head tree: the ordered schedule of morphogenetic heads and their clocks.
============================================================================

Every head is a super-enhancer cluster read out at a time on a clock. The tree is
a COMPOSITION hierarchy whose depth is developmental time: axis heads at the root,
the shared effector heads in the middle, one organ head per SE cluster at the leaves
-- each leaf just calls the effectors at an address on the body-electric frame.

  TIER 0  CLOCKS      the sequencers -- decide WHICH SE cluster is read WHEN
  TIER 1  AXIS        make the body's length/width and its AP/DV/LR electric frame
  TIER 2  ADDRESS     carve the frame into named coordinates (Hox-AP, DV, segments)
  TIER 3  EFFECTORS   the shared library: how tissue moves / sticks / dies / folds
  TIER 4  ORGANS      leaves: each fills an antinode of the body electric, in clock
                      order, using the Tier-3 effectors (medic.organ_sprouting)

check_organ_order() validates Miles's law: the order organs unlock (the master clock)
matches the anterior->posterior order of the antinodes they land on, and the antinode
frame predicts organ position BETTER than Vm (Vm is the readout, not the placer).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.head_tree
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import numpy as np

from medic.organ_sprouting import ORGAN_SCHEDULE

# tier -> list of (head, clock/unlock, what it modifies, effectors/notes)
HEAD_TREE = [
    ("0 CLOCKS", [
        ("telomere -> TERRA -> PRC2", "reads the methylation strata in order",
         "gates every SE cluster (reverse-developmental de-repression)"),
        ("segmentation clock (her1/Hes)", "T = 29 min oscillator", "somite boundaries"),
        ("cell-cycle clock", "division count", "growth timing"),
    ]),
    ("1 AXIS / GEOMETRY", [
        ("elongation (Wnt-PCP / conv-ext)", "earliest", "body length + WIDTH (fish<->tetrapod)"),
        ("division", "continuous", "cell count / growth"),
        ("electric frame (connexin/GJ)", "once body has width", "AP + DV + LR eigenmode axes"),
    ]),
    ("2 ADDRESS", [
        ("Hox-AP (fossil-record enhancers)", "Hox colinearity", "antero-posterior address (fore/hind)"),
        ("DV / germ layer (BMP-Chordin, Vm)", "gastrulation", "dorso-ventral + germ layer"),
        ("segmentation", "clock x wavefront", "metameric somite address"),
    ]),
    ("3 EFFECTORS (shared library)", [
        ("migration", "-", "cell position (convergent extension, ingression)"),
        ("cadherin (adherens)", "-", "topology: same-fate sorting into compartments"),
        ("integrin / ECM (fascia)", "-", "cohesion: binds compartments into one continuum"),
        ("apoptosis", "-", "removal: sculpting, cavitation, digit separation"),
        ("fusion", "-", "merge: neural tube, palate, limb seams"),
        ("folding / buckling", "-", "curvature: neural fold, gut looping"),
        ("lateral inhibition (Delta-Notch)", "-", "spacing: one bud per antinode (exclusivity)"),
    ]),
    ("4 ORGANS (leaves: fill antinodes in clock order)",
     [(o["name"], f"PRC2<={o['unlock']:.2f}", f"{o['place']} organ; recruits {sorted(o['germ'])[:2]}...")
      for o in ORGAN_SCHEDULE]),
]

# MEASURED master TF per head (SCENIC regulon specificity on real E12.5 MOSTA, medic.mosta_master_tfs)
# -- the head IS this super-enhancer / master-TF cluster. Each leaf's identity, read from the atlas.
MASTER_TF = {
    "Heart": "Nkx2-5", "Liver": "Foxa3", "Lung": "Elf5", "Pancreas": "Pdx1", "Kidney": "Pax2",
    "Gut": "Cdx1", "Eye": "Rax", "Otic": "Six1", "Notochord": "T/Brachyury", "Cartilage": "Sox5/6",
    "Rib": "Sox9", "Muscle": "Myf6", "DRG": "Prrxl1", "Sympathetic": "Phox2b", "Vessel": "Etv2/Tal1",
    "Jaw": "Msx1", "Meninges": "Zic1", "Choroid": "Rfx2", "Gonad": "Sohlh2", "Connective": "Twist2",
    "Forebrain": "Arx", "Midbrain": "Otx2", "Hindbrain": "Gbx2", "Cerebellum": "Atoh1",
    "Spinal Cord": "Hoxb9", "Skin": "Ovol1",
}


def print_schedule():
    print("HEAD TREE  (root -> leaves = developmental time)\n" + "=" * 74)
    for tier, heads in HEAD_TREE:
        print(f"\nTIER {tier}")
        for name, clock, modifies in heads:
            print(f"   {name:34s} | {clock:26s} | {modifies}")
    print("\nThe Tier-4 organ leaves fill the body-electric antinodes in this clock order:")
    print("   " + "  ->  ".join(f"{o['name']}({MASTER_TF.get(o['name'],'?')})" for o in ORGAN_SCHEDULE))
    print_cascade()


def print_cascade():
    """The Tier-5 SUBHEAD / lineage layer: progenitor heads resolving into derivative heads over
    developmental time (medic.head_cascade), each derivative tagged with its measured master TF."""
    from medic.head_cascade import LINEAGE
    print("\nTIER 5 CASCADE (progenitor head -> derivative subheads, in atlas emergence order)")
    for parent, kids in LINEAGE.items():
        chain = ", ".join(f"{k[0]}({k[2]},{k[1]})" for k in kids)
        print(f"   {parent:13s} -> {chain}")


def _auc(score, label):
    """Rank-AUC of `score` predicting boolean `label` (0.5 = chance)."""
    order = np.argsort(score)
    ranks = np.empty_like(order, float); ranks[order] = np.arange(len(score))
    pos = label.astype(bool); npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[pos].sum() - npos * (npos - 1) / 2) / (npos * nneg)


def check_organ_order():
    from medic.unified_embryo import simulate, FIDX
    from medic.body_electric_antinodes import body_electric_antinodes
    print("\n\nCHECK -- organ clock order vs antinode order, and antinode-vs-Vm placement")
    print("=" * 74)
    frames, m = simulate(use_ecm=True, limb_buds=True, convergent_ext=1.0)
    P, fid = m["pos"], m["fid"]
    vm = frames[-1][4]                                    # (born, t, prc2, pos, VM, fid) of the last frame
    a = (P[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)
    lad = np.array(body_electric_antinodes(P)["ap_levels"])

    print(f"antinode ladder (AP): {[round(x,2) for x in lad]}")
    print(f"\n {'organ':7s} {'clock rank':10s} {'AP centroid':11s} {'nearest antinode':16s}")
    ap_cent, clk_rank = [], []
    organ_mask = np.zeros(len(fid), bool)
    for k, org in enumerate(ORGAN_SCHEDULE):
        sel = fid == FIDX[org["name"]]
        if not sel.any():
            continue
        organ_mask |= sel
        c = a[sel].mean(); nn = lad[np.argmin(np.abs(lad - c))]
        ap_cent.append(c); clk_rank.append(k)
        print(f" {org['name']:7s} {k:^10d} {c:^11.2f} {nn:^16.2f}")

    # order check: does clock rank increase monotonically with AP centroid?
    order_ok = np.all(np.diff(ap_cent) > 0)
    from numpy import corrcoef
    rho = corrcoef(clk_rank, ap_cent)[0, 1]
    print(f"\n clock order == anterior->posterior antinode order?  {order_ok}  (rho={rho:.3f})")

    # Vm cross-check: does the antinode frame predict organ location better than Vm?
    if vm is not None:
        anti_score = np.exp(-((a[:, None] - lad[None, :]) / 0.05) ** 2).max(1)  # closeness to any antinode
        vm_score = np.abs(vm - np.median(vm))                                   # Vm extremeness
        auc_anti = _auc(anti_score, organ_mask)
        auc_vm = _auc(vm_score, organ_mask)
        print(f"\n predicting organ location:  antinode-frame AUC = {auc_anti:.3f}   "
              f"Vm AUC = {auc_vm:.3f}")
        print("   -> the electric-body antinodes place the organs; Vm is the weaker (readout) signal,"
              if auc_anti > auc_vm else "   -> (unexpected: Vm >= antinode here)")
        print("      as predicted." if auc_anti > auc_vm else "")


def main():
    print_schedule()
    check_organ_order()


if __name__ == "__main__":
    main()
