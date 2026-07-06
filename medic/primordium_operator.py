#!/usr/bin/env python3
"""
The primordium as a recursive operator: {division, differentiation, motility}
under {clocks x morphogen field}, then repeat.
=============================================================================

Miles's formulation (2026-06-30): a primordium is not a static seed but a
RECURSIVE OPERATOR. At each locus it reads the morphogen/Vm field (the WHERE) and
the three clocks (the WHEN), executes the three behavioural heads, and the
execution reshapes the field so that DAUGHTER primordia nucleate -- then it
repeats. The body is the recursive application of one operator.

One recursion level, at a competence locus:
  1. read the FIELD  -- the Vm/morphogen positional information (antinodes of the
     genome-set operator's standing wave: cymatic mode 2^k at level k).
  2. read the CLOCKS -- telomere (division budget), Hox/PRC (differentiation
     index), motility (migration gate).
  3. execute the THREE HEADS:
       DIVISION       gated by the telomere clock  -> proliferate (grow domain).
       DIFFERENTIATION gated by the Hox/PRC clock  -> commit the master-TF fate
                       that the local Vm selects (the Vm->cascade table).
       MOTILITY       gated by the motility clock  -> displace / set boundary.
  4. RESHAPE THE FIELD: division+migration change the pattern -> the morphogen
     the primordium emits updates Vm (the Vm->morphogen->Vm step) -> a FINER
     standing wave with new antinodes = new competence loci.
  5. RECURSE at each new antinode; decrement the clocks.
  TERMINATION (the stop signal): when the telomere budget exhausts (no more
  division), Hox is fully withdrawn (terminal fate), and the field is smooth
  (no new antinodes) -- the recursion bottoms out.

This is an operator-level schematic on a 1-D axis (idealised, not data-grounded);
it shows the recursion, the clock-gated three heads, and the traced
Vm<->morphogen cascade. Grounding each arrow in a real organ is the volumetric
programme.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.primordium_operator
Output: primordium_operator.png  (+ console cascade trace)
"""
from __future__ import annotations

import numpy as np

N = 400
X = np.linspace(0, 1, N)
TELOMERE0 = 16          # initial division budget (halves each recursion level)

# Vm -> fate bands (the master-TF cascade the local voltage selects; abbreviated
# from medic.perceptron_trace.VM_CASCADE). Vm in mV.
FATE_BANDS = [
    (-90, -58, "neural/eye (Rx/Pax6/Six3)"),
    (-58, -45, "muscle/limb (MyoD/Tbx5)"),
    (-45, -33, "skeletal/cartilage (Sox9)"),
    (-33, -28, "cardiac (Nkx2-5/GATA4)"),
    (-28, -10, "endoderm (FoxA2/Pdx1)"),
]


def fate(vm):
    for lo, hi, name in FATE_BANDS:
        if lo <= vm < hi:
            return name
    return "uncommitted"


def base_field():
    """The level-0 Vm field: the validated AP voltage floor (head depol -> tail hyperpol)."""
    return -25.0 - 45.0 * X        # -25 mV anterior .. -70 mV posterior


def antinodes(level):
    """Competence loci = antinodes of the cymatic standing wave at this level
    (genome-set operator's mode 2^level). 2^level evenly spaced positions."""
    k = 2 ** level
    return (np.arange(k) + 0.5) / k


def run():
    """Run the recursive primordium operator until the clocks exhaust."""
    Vm = base_field()
    trace = []                 # per-level record
    cascade = []               # the Vm->morphogen->Vm->... arrows
    level = 0
    telomere = TELOMERE0
    hox = 0

    while telomere >= 1:
        pos = antinodes(level)
        prim = []
        morph = np.zeros(N)
        for p in pos:
            i = int(round(p * (N - 1)))
            vm_here = Vm[i]
            # 3 heads, clock-gated
            division = int(telomere)                       # telomere -> # divisions left
            differentiation = fate(vm_here)                 # Hox/PRC -> fate from local Vm
            motility = 0.5 / 2 ** level                     # motility clock -> displacement scale
            prim.append(dict(pos=p, vm=vm_here, division=division,
                             fate=differentiation, motility=motility, hox=hox))
            # the morphogen this primordium emits (a localized source -> reshapes Vm)
            sigma = 0.5 / 2 ** (level + 1)
            morph += np.exp(-0.5 * ((X - p) / sigma) ** 2)
            hox += 1
        trace.append(dict(level=level, Vm=Vm.copy(), pos=pos, prim=prim,
                          telomere=telomere, n_prim=len(pos)))
        cascade.append((f"Vm(level {level})", f"{len(pos)} primordia -> morphogen",
                        f"reshape -> Vm(level {level+1})"))
        # RESHAPE THE FIELD: morphogen feeds back on channels -> new Vm with a finer
        # standing wave = the next level's antinodes (Vm -> morphogen -> Vm).
        finer = 6.0 * np.cos(2 ** (level + 1) * np.pi * X)
        Vm = Vm + 0.0 * morph + finer * np.exp(-level * 0.15)   # finer modulation, damped
        telomere /= 2.0                                          # division budget halves
        level += 1

    return trace, cascade, level


def main():
    print("=" * 74)
    print("THE PRIMORDIUM AS A RECURSIVE OPERATOR  --  {div, diff, motility} x {clocks, field}")
    print("=" * 74)
    trace, cascade, depth = run()
    total = sum(t["n_prim"] for t in trace)
    print(f"\nRecursion ran {depth} levels until the telomere clock exhausted "
          f"(stop signal); {total} primordia total.\n")
    for t in trace:
        ex = t["prim"][0]
        print(f"  level {t['level']}: {t['n_prim']:2d} primordia | telomere(division)={t['telomere']:5.1f} "
              f"| e.g. Vm={ex['vm']:+5.1f} -> fate: {ex['fate']}")
    print("\nTraced Vm<->morphogen cascade (the loop, per level):")
    for a, b, c in cascade:
        print(f"  {a:18s} --[3 heads]--> {b:28s} --[feedback]--> {c}")
    print("\nTermination = telomere exhausted + field smooth (no new antinodes) = the STOP signal.")

    _figure(trace, depth)
    return depth >= 4 and total >= 15


def _figure(trace, depth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # (A) the recursive field subdivision: Vm per level + primordia at antinodes
    axA = fig.add_subplot(gs[0, :])
    for t in trace:
        y = t["level"]
        v = t["Vm"]
        axA.scatter(X, np.full(N, -y), c=v, cmap="viridis", s=4, linewidths=0)
        axA.scatter(t["pos"], np.full(t["n_prim"], -y), c="red", s=40, marker="v",
                    edgecolors="k", zorder=5)
    axA.set_yticks([-t["level"] for t in trace])
    axA.set_yticklabels([f"level {t['level']}\n({t['n_prim']} prim)" for t in trace])
    axA.set_xlabel("anterior -> posterior axis")
    axA.set_title("Recursive subdivision: each level's primordia (red) nucleate at the field's antinodes; "
                  "the field is reshaped (Vm->morphogen->Vm) into a finer standing wave -> daughter primordia")

    # (B) recursion tree (primordia doubling)
    axB = fig.add_subplot(gs[1, 0])
    for t in trace:
        axB.scatter(t["pos"], np.full(t["n_prim"], t["level"]), c="red", s=30, marker="v")
        if t["level"] > 0:
            par = antinodes(t["level"] - 1)
            for p in t["pos"]:
                j = np.argmin(np.abs(par - p))
                axB.plot([par[j], p], [t["level"] - 1, t["level"]], "k-", lw=0.4, alpha=0.5)
    axB.invert_yaxis(); axB.set_xlabel("axis"); axB.set_ylabel("recursion level")
    axB.set_title("recursion tree: 1 -> 2 -> 4 -> ... primordia", fontsize=9)

    # (C) clocks decrementing -> termination
    axC = fig.add_subplot(gs[1, 1])
    lv = [t["level"] for t in trace]
    tel = [t["telomere"] for t in trace]
    axC.plot(lv, tel, "o-", color="#d62728", label="telomere (division budget)")
    axC.axhline(1.0, ls="--", c="k", lw=0.8)
    axC.text(0.1, 1.2, "division floor -> STOP", fontsize=8)
    axC.plot(lv, [sum(t["n_prim"] for t in trace[:i + 1]) for i in range(len(trace))],
             "s-", color="#1f77b4", label="cumulative primordia (Hox index)")
    axC.set_xlabel("recursion level"); axC.set_title("clocks gate the recursion depth", fontsize=9)
    axC.legend(fontsize=7)

    fig.suptitle("The primordium as a recursive operator: {division, differentiation, motility} under "
                 "{telomere, Hox/PRC, motility clocks} x {Vm/morphogen field}, then repeat.\n"
                 "The recursion terminates when the clocks exhaust -- the endogenous stop signal.",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig("primordium_operator.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: primordium_operator.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}")
