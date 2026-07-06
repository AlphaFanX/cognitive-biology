#!/usr/bin/env python3
"""
The ectopic eye: an organ is a RELOCATABLE attractor, not a genomic address.
============================================================================

This is the central demonstration of Paper #5 ("The Inner Perceptron versus the
Outer Perceptron"). It reproduces, in the model's own machinery, Levin's result
that an eye can be induced OUTSIDE the anterior eye field by a local bioelectric
instruction (Pai, Aw, Shomrat, Lemire & Levin, "Transmembrane voltage potential
controls embryonic eye patterning in Xenopus laevis", Development 139:313, 2012):
a specific hyperpolarized membrane voltage is INSTRUCTIVE for eye formation, and
imposing that voltage in non-eye regions -- including the tail -- triggers an
ectopic eye.

The mechanism here is the paper's thesis made operational:

  * the OUTER perceptron writes a resting-voltage (Vm) set-point field;
  * a cell becomes EYE-COMPETENT where its Vm sits in the eye window
    (hyperpolarized, ~ -65 mV) -- this is the Vm -> cascade trigger (Table:
    vm-cascade). Competence is NOT an anteroposterior address;
  * the INNER perceptron runs the SAME Gierer-Meinhardt eye activator-inhibitor
    everywhere; wherever competence is open, a peak self-organizes and KICKS OFF
    the eye differentiation cascade (Rx/Pax6/Six3/Lhx2).

So the eye's position is a free boundary condition on the field. We show three
conditions on one dorsal sheet (LR x AP), with IDENTICAL eye dynamics:

  WT          : anterior Vm hyperpolarized -> 2 head eyes (Shh splits the field).
  INSTRUCTED  : + a hyperpolarizing Vm patch in the tail -> a 3rd, ectopic eye.
  CONTROL     : the tail location WITHOUT the instruction -> no eye there.

The only difference between INSTRUCTED and CONTROL is the local voltage. The eye
GRN, the diffusion constants, the production rates are unchanged. That is the
relocatable attractor.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.ectopic_eye
Output: ectopic_eye.png  (+ console eye counts)
"""
from __future__ import annotations

import numpy as np

try:
    from . import morphogen_rd as mrd
except ImportError:  # pragma: no cover
    from medic import morphogen_rd as mrd

# Reuse the morphogen sheet geometry and helpers (LR rows, AP cols).
NLR, NAP = mrd.NLR, mrd.NAP
LR, AP = mrd.LR, mrd.AP
g = mrd.g

# Eye bioelectric window (Pai-Levin 2012): eyes form at a hyperpolarized Vm.
V_EYE = -65.0       # eye-permissive set-point (mV)
SIGMA_V = 9.0       # width of the competence window (mV)
TAIL_AP = 0.88      # where we impose the ectopic instruction


def baseline_vm():
    """The outer perceptron's resting-voltage set-point over the sheet: anterior/
    head hyperpolarized (ectoderm, eye-permissive) grading to a depolarized tail.
    This is the validated AP voltage floor (head depol vs tail is encoded as the
    eye-permissive *hyperpolarized* anterior here, matching the craniofacial
    electric-face convention that the eye field is hyperpolarized)."""
    return -68.0 + 46.0 * AP        # -68 mV anterior -> -22 mV tail


def eye_competence(Vm, shh):
    """Eye competence = a Gaussian window on Vm (the bioelectric trigger), carved
    at the midline by Shh. Cells in the hyperpolarized window are competent; Shh
    represses competence at the midline so the anterior field splits into two."""
    window = np.exp(-0.5 * ((Vm - V_EYE) / SIGMA_V) ** 2)
    midline_cut = np.clip(1.0 - shh / (shh.max() + 1e-9), 0.0, 1.0)
    return window * midline_cut


def tail_instruction():
    """The Levin instruction: a local hyperpolarizing Vm patch in the tail
    (mimicking targeted ion-channel misexpression), at the midline so it forms a
    single ectopic eye. Returns a negative (hyperpolarizing) Vm offset field."""
    return -47.0 * g(AP, TAIL_AP, 0.035) * g(LR, 0.0, 0.22)


def run_condition(instructed):
    """Build Vm + competence, run the SAME eye activator-inhibitor, return fields."""
    Vm = baseline_vm()
    # Shh from the prechordal plate (anterior midline only) -> splits the head field.
    shh_head = 2.0 * g(LR, 0.0, 0.14) * g(AP, 0.10, 0.10)
    if instructed:
        Vm = Vm + tail_instruction()
    comp = eye_competence(Vm, shh_head)
    # the eye cascade: identical Gierer-Meinhardt dynamics in every condition.
    eyes = mrd.gierer_meinhardt(comp, shh=0.3 * shh_head, seed=1)
    return Vm, comp, eyes


def count_eyes(eyes, ap_lo, ap_hi, thresh=0.5):
    """Count distinct eye spots in an AP band. Normalize against the GLOBAL eye
    amplitude (not the band-local max) so an eyeless band stays near zero instead
    of having its noise floor rescaled to 1."""
    m = (AP[0] >= ap_lo) & (AP[0] <= ap_hi)
    col = eyes[:, m].max(axis=1)
    col = col / (eyes.max() + 1e-9)
    on = col > thresh
    return int(np.sum(np.diff(on.astype(int)) == 1) + (1 if on[0] else 0))


def main():
    print("=" * 74)
    print("THE ECTOPIC EYE  --  organ = relocatable attractor (Pai-Levin 2012)")
    print("=" * 74)
    print("Same eye GRN dynamics in every condition; only the Vm field differs.\n")

    Vm_wt, comp_wt, eyes_wt = run_condition(instructed=False)
    Vm_in, comp_in, eyes_in = run_condition(instructed=True)

    head_wt = count_eyes(eyes_wt, 0.03, 0.18)
    tail_wt = count_eyes(eyes_wt, 0.80, 0.96)
    head_in = count_eyes(eyes_in, 0.03, 0.18)
    tail_in = count_eyes(eyes_in, 0.80, 0.96)

    print(f"  WT (no instruction)        : head eyes = {head_wt} (expect 2), "
          f"tail eyes = {tail_wt} (expect 0)")
    print(f"  INSTRUCTED (tail Vm patch) : head eyes = {head_in} (expect 2), "
          f"tail eyes = {tail_in} (expect 1)")
    # the tail Vm at the instruction site, both conditions, to make the trigger plain
    j = int(round(TAIL_AP * (NAP - 1)))
    i = NLR // 2
    print(f"\n  Vm at the tail site: WT = {Vm_wt[i, j]:+.1f} mV (depolarized, out of "
          f"eye window) -> CONTROL: no eye")
    print(f"                       instructed = {Vm_in[i, j]:+.1f} mV "
          f"(driven into the eye window ~{V_EYE:+.0f}) -> ectopic eye")

    ok = (head_wt == 2 and tail_wt == 0 and head_in == 2 and tail_in == 1)
    print(f"\n  RESULT: {'PASS' if ok else 'CHECK'} -- the eye cascade relocates to "
          f"wherever the field opens the window.")

    _figure(Vm_wt, comp_wt, eyes_wt, Vm_in, comp_in, eyes_in,
            head_wt, tail_wt, head_in, tail_in)
    return ok


def _figure(Vm_wt, comp_wt, eyes_wt, Vm_in, comp_in, eyes_in,
            head_wt, tail_wt, head_in, tail_in):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [0, 1, -1, 1]
    fig, ax = plt.subplots(2, 3, figsize=(16, 7))

    def show(a, Z, title, cmap="viridis", vlim=None):
        kw = {} if vlim is None else dict(vmin=vlim[0], vmax=vlim[1])
        im = a.imshow(Z, origin="lower", extent=ext, aspect="auto", cmap=cmap, **kw)
        a.set_title(title, fontsize=10)
        a.set_xlabel("anterior -> posterior (AP)"); a.set_ylabel("L <- LR -> R")
        fig.colorbar(im, ax=a, fraction=0.035, pad=0.02)

    vlim = (-70, -20)
    show(ax[0, 0], Vm_wt, "WT  --  outer-perceptron Vm field\n(anterior hyperpolarized = eye-permissive)",
         "magma", vlim)
    show(ax[0, 1], comp_wt, "WT  --  eye competence = Vm window x (Shh midline cut)", "cividis")
    show(ax[0, 2], eyes_wt, f"WT  --  eye cascade fires\n{head_wt} head eyes, {tail_wt} tail")
    ax[0, 2].text(0.86, 0.0, "tail\n(no eye)", color="white", fontsize=8, ha="center")

    show(ax[1, 0], Vm_in, "INSTRUCTED  --  + hyperpolarizing tail Vm patch\n(Levin bioelectric instruction)",
         "magma", vlim)
    ax[1, 0].text(TAIL_AP, 0.0, "instruction", color="cyan", fontsize=8, ha="center")
    show(ax[1, 1], comp_in, "INSTRUCTED  --  competence window now OPEN in the tail", "cividis")
    show(ax[1, 2], eyes_in, f"INSTRUCTED  --  ECTOPIC eye in the tail\n{head_in} head eyes, {tail_in} tail eye")
    ax[1, 2].text(TAIL_AP, 0.0, "ectopic\neye", color="white", fontsize=8, ha="center")

    fig.suptitle("An organ is a relocatable attractor: the eye cascade fires wherever the Vm field opens "
                 "its competence window\n(Pai, Aw, Shomrat, Lemire & Levin, Development 2012 -- ectopic "
                 "eyes by membrane-voltage instruction)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("ectopic_eye.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: ectopic_eye.png")


if __name__ == "__main__":
    main()
