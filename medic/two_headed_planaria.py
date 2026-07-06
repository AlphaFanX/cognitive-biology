#!/usr/bin/env python3
"""
The two-headed planarian: a GLOBAL-axis ectopic organ, purely inner-loop.
=========================================================================

Companion to medic.ectopic_eye. The ectopic eye relocates a LOCAL organ attractor
by a local Vm peak while the vertebrate axis (run by its outer genomic kernel)
stays intact. The two-headed planarian is the GLOBAL version: the whole
anteroposterior axis attractor is flipped, and -- because flatworms lost the
genomic (CpG-methylation) kernel and pattern the body from the BIOELECTRIC field
alone (Cognitive Biology Paper #3) -- it isolates the INNER loop completely. No
outer genomic kernel is involved.

Two classic results (Levin lab) are reproduced as the model's own inner-loop
behaviour:

  (A) INDUCTION BY GAP-JUNCTION BLOCK (Oviedo, Levin et al. 2010; Beane et al.
      2011). A regenerating head produces a long-range, gap-junction-propagated
      inhibitor that tells the far (posterior) wound to make a tail. The signal
      reaches the posterior wound only if the gap-junction coupling is high
      enough: the decay length is lambda = sqrt(D_gj/mu). Intact coupling ->
      lambda >> body length -> posterior wound inhibited -> TAIL (one head, one
      tail). Block the gap junctions (octanol / innexin RNAi) -> lambda << body
      length -> the inhibitor never arrives -> the posterior wound also makes a
      HEAD -> two heads. This is a bifurcation in a single inner-loop parameter.

  (B) HERITABLE BIOELECTRIC MEMORY (Durant, Levin et al. 2017). Once two-headed,
      the worm's stored Vm pattern is BIPOLAR (depolarized at both ends,
      hyperpolarized in the middle) rather than monopolar. Re-cut in plain water
      WITHOUT any further manipulation, a middle fragment inherits that bipolar
      pattern and regenerates two heads again. The genome never changed; the
      target morphology is stored in the bioelectric field -- the inner loop.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.two_headed_planaria
Output: two_headed_planaria.png  (+ console PASS lines)
"""
from __future__ import annotations

import numpy as np

L = 1.0                 # body-axis length (fragment, head wound at x=0)
MU = 1.0                # head-inhibitor decay rate
S0 = 1.0                # inhibitor source strength at the regenerating head
THETA_S = 0.35          # posterior wound makes a tail iff inhibitor here > THETA_S
D_GJ_INTACT = 2.0       # gap-junction coupling, normal worm
D_GJ_BLOCKED = 0.15     # gap junctions blocked (octanol / innexin RNAi)

# Bioelectric identity readout: a wound becomes a HEAD where it is depolarized
# relative to the fragment interior, a tail where hyperpolarized.
V_HEAD = -25.0          # depolarized head pole (mV)
V_TAIL = -65.0          # hyperpolarized tail pole (mV)


# ---------------------------------------------------------------------------
# (A) Induction: the head inhibitor reaching the posterior wound via gap junctions
# ---------------------------------------------------------------------------
def inhibitor_at_posterior(D_gj):
    """Steady head-inhibitor concentration at the posterior wound (x=L). The
    inhibitor is produced at the anterior head and decays as it spreads through
    gap junctions: S(x) = S0 exp(-x/lambda), lambda = sqrt(D_gj/mu)."""
    lam = np.sqrt(D_gj / MU)
    return S0 * np.exp(-L / lam), lam


def regenerate_poles(D_gj):
    """Returns (anterior_identity, posterior_identity) for a trunk fragment after
    amputation at the given gap-junction coupling. The anterior wound makes a head
    by default; the posterior wound makes a tail only if the head's inhibitor
    reaches it."""
    s_post, lam = inhibitor_at_posterior(D_gj)
    posterior = "tail" if s_post > THETA_S else "head"
    return ("head", posterior), s_post, lam


# ---------------------------------------------------------------------------
# (B) Memory: stored Vm pattern -> re-cut outcome (no manipulation)
# ---------------------------------------------------------------------------
def stored_vm(kind, x):
    """The worm's stored bioelectric pattern along the axis.
    monopolar : depolarized anterior head -> hyperpolarized posterior tail.
    bipolar   : depolarized at BOTH ends, hyperpolarized middle (a two-head worm)."""
    if kind == "monopolar":
        return V_HEAD + (V_TAIL - V_HEAD) * x                 # -25 -> -65
    # bipolar: -45 + 20 cos(2 pi x) -> -25 at x=0,1 ; -65 at x=0.5
    mid = 0.5 * (V_HEAD + V_TAIL)
    amp = 0.5 * (V_HEAD - V_TAIL)
    return mid + amp * np.cos(2 * np.pi * x)


def recut_outcome(kind, frag=(0.30, 0.70), n=400):
    """Cut a middle fragment out of a worm with the given stored Vm pattern, in
    plain water (no manipulation). Each new wound regenerates a head if its
    inherited Vm is depolarized relative to the fragment interior, else a tail."""
    x = np.linspace(0, 1, n)
    V = stored_vm(kind, x)
    lo, hi = frag
    m = (x >= lo) & (x <= hi)
    interior = V[m].mean()
    v_ant, v_post = stored_vm(kind, np.array([lo]))[0], stored_vm(kind, np.array([hi]))[0]
    ant = "head" if v_ant > interior else "tail"
    post = "head" if v_post > interior else "tail"
    return (ant, post), (v_ant, v_post, interior)


def main():
    print("=" * 74)
    print("THE TWO-HEADED PLANARIAN  --  global-axis ectopic organ, purely inner-loop")
    print("=" * 74)
    print("Flatworms lost the genomic (methylation) kernel -> body plan = the")
    print("bioelectric field = the INNER loop. So this is inner-loop only.\n")

    print("(A) Induction by gap-junction block (Oviedo/Beane/Levin):")
    (a_i, p_i), s_i, lam_i = regenerate_poles(D_GJ_INTACT)
    (a_b, p_b), s_b, lam_b = regenerate_poles(D_GJ_BLOCKED)
    print(f"    intact GJ  (D={D_GJ_INTACT}): lambda={lam_i:.2f} >> L  -> inhibitor at "
          f"posterior = {s_i:.2f} > {THETA_S} -> {a_i} + {p_i}")
    print(f"    blocked GJ (D={D_GJ_BLOCKED}): lambda={lam_b:.2f} << L  -> inhibitor at "
          f"posterior = {s_b:.2f} < {THETA_S} -> {a_b} + {p_b}")
    okA = (a_i, p_i) == ("head", "tail") and (a_b, p_b) == ("head", "head")
    print(f"    => {'PASS' if okA else 'CHECK'}: blocking one inner-loop parameter "
          f"(gap-junction coupling) flips 1 head to 2 heads.\n")

    print("(B) Heritable bioelectric memory (Durant/Levin): re-cut in plain water:")
    (m_a, m_p), (mva, mvp, mint) = recut_outcome("monopolar")
    (b_a, b_p), (bva, bvp, bint) = recut_outcome("bipolar")
    print(f"    monopolar worm -> fragment faces Vm ({mva:+.0f},{mvp:+.0f}) vs interior "
          f"{mint:+.0f} -> {m_a} + {m_p}")
    print(f"    bipolar  worm  -> fragment faces Vm ({bva:+.0f},{bvp:+.0f}) vs interior "
          f"{bint:+.0f} -> {b_a} + {b_p}")
    okB = (m_a, m_p) == ("head", "tail") and (b_a, b_p) == ("head", "head")
    print(f"    => {'PASS' if okB else 'CHECK'}: the bipolar Vm pattern regenerates 2 "
          f"heads with NO genome change and NO further manipulation.\n")

    print(f"RESULT: {'PASS' if (okA and okB) else 'CHECK'} -- the anteroposterior axis "
          f"is an inner-loop bioelectric attractor; it can be flipped to a second head\n"
          f"        and the flipped pattern is heritable through cuts. No outer "
          f"(genomic) kernel is involved.")

    _figure(okA, okB)
    return okA and okB


def _figure(okA, okB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))

    # (1) inhibitor profiles + bifurcation vs gap-junction coupling
    x = np.linspace(0, 1, 200)
    for D, lab, c in [(D_GJ_INTACT, "intact GJ", "#1f77b4"),
                      (D_GJ_BLOCKED, "blocked GJ (octanol)", "#d62728")]:
        lam = np.sqrt(D / MU)
        ax[0].plot(x, S0 * np.exp(-x / lam), color=c, lw=2, label=f"{lab} ($\\lambda$={lam:.2f})")
    ax[0].axhline(THETA_S, color="k", ls="--", lw=1)
    ax[0].text(0.02, THETA_S + 0.02, "tail-induction threshold", fontsize=8)
    ax[0].annotate("posterior\nwound", xy=(1.0, 0.0), xytext=(0.75, 0.25), fontsize=8,
                   ha="center")
    ax[0].set_xlabel("position along axis (head wound at 0 -> posterior wound at 1)")
    ax[0].set_ylabel("head-inhibitor concentration")
    ax[0].set_title("(A) Head inhibitor reaches the posterior wound\nonly if gap junctions are open")
    ax[0].legend(fontsize=8, loc="upper right")

    # bifurcation curve
    Dg = np.linspace(0.05, 3.0, 300)
    s_post = S0 * np.exp(-L / np.sqrt(Dg / MU))
    axb = ax[1]
    axb.plot(Dg, s_post, color="#555", lw=2)
    axb.axhline(THETA_S, color="k", ls="--", lw=1)
    Dcrit = MU * (L / (-np.log(THETA_S))) ** 2
    axb.axvline(Dcrit, color="purple", ls=":", lw=1.5)
    axb.fill_between(Dg, 0, THETA_S, where=(Dg < Dcrit), color="#d62728", alpha=0.12)
    axb.fill_between(Dg, THETA_S, 1.0, where=(Dg > Dcrit), color="#1f77b4", alpha=0.12)
    axb.text(Dcrit * 0.5, 0.06, "2 HEADS", color="#d62728", fontsize=10, ha="center")
    axb.text(min(2.4, Dcrit * 2.2), 0.7, "1 head\n1 tail", color="#1f77b4", fontsize=10, ha="center")
    axb.text(Dcrit, 0.92, f"$D_{{crit}}$={Dcrit:.2f}", color="purple", fontsize=8, ha="center")
    axb.set_xlabel("gap-junction coupling  $D_{gj}$")
    axb.set_ylabel("inhibitor at posterior wound")
    axb.set_title("(A) Bifurcation in one inner-loop parameter\n(coupling) -> 1 vs 2 heads")

    # (3) stored Vm patterns + re-cut outcome
    xx = np.linspace(0, 1, 400)
    ax[2].plot(xx, stored_vm("monopolar", xx), color="#2ca02c", lw=2, label="1-head worm (monopolar)")
    ax[2].plot(xx, stored_vm("bipolar", xx), color="#d62728", lw=2, label="2-head worm (bipolar)")
    for f in (0.30, 0.70):
        ax[2].axvline(f, color="gray", ls=":", lw=1)
    ax[2].text(0.5, V_TAIL - 1, "re-cut fragment", fontsize=8, ha="center", va="top")
    ax[2].annotate("H", xy=(0.30, stored_vm("bipolar", np.array([0.30]))[0]), color="#d62728",
                   fontsize=11, ha="center", fontweight="bold")
    ax[2].annotate("H", xy=(0.70, stored_vm("bipolar", np.array([0.70]))[0]), color="#d62728",
                   fontsize=11, ha="center", fontweight="bold")
    ax[2].annotate("H", xy=(0.30, stored_vm("monopolar", np.array([0.30]))[0]), color="#2ca02c",
                   fontsize=11, ha="center", fontweight="bold")
    ax[2].annotate("T", xy=(0.70, stored_vm("monopolar", np.array([0.70]))[0]), color="#2ca02c",
                   fontsize=11, ha="center", fontweight="bold")
    ax[2].set_xlabel("position along axis")
    ax[2].set_ylabel("stored $V_m$ (mV)")
    ax[2].set_title("(B) Heritable memory: re-cut in plain water\nbipolar pattern -> 2 heads again (no genome change)")
    ax[2].legend(fontsize=8, loc="lower center")

    fig.suptitle("The two-headed planarian: the anteroposterior axis is an inner-loop bioelectric attractor "
                 "(flatworms have no genomic kernel)\n(A) Oviedo/Beane/Levin 2010-11 gap-junction induction   "
                 "(B) Durant/Levin 2017 heritable bioelectric memory", fontsize=11.5, y=1.04)
    fig.tight_layout()
    fig.savefig("two_headed_planaria.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved: two_headed_planaria.png")


if __name__ == "__main__":
    main()
