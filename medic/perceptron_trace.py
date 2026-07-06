#!/usr/bin/env python3
"""
Tracing the two perceptrons to genomic parameters and literature equations.
===========================================================================

Paper #5 ("The Inner Perceptron versus the Outer Perceptron") claims that the
developmental controller is a GLASS BOX: every value in either perceptron is
derivable from a genomic parameter through a named literature equation -- none
is a trained-opaque weight. This module makes that concrete. It

  (1) recomputes the OUTER perceptron's emitted set-points -- the per-organ
      ion-channel conductances from the ABC activity table and the Goldman
      resting voltage they produce -- and prints the genomic-parameter ->
      Goldman -> V_target chain (medic.bioelectric_development);

  (2) recomputes the INNER perceptron's coefficients -- the NCA relaxation
      rule weights (k_relax, k_gj), the her1 delayed-feedback period from the
      Lewis DDE, the Gierer-Meinhardt activator-inhibitor rates, and the
      FitzHugh-Nagumo ERK-wave excitability -- each tagged with its genomic
      referent and source equation;

  (3) tabulates which morphogen / differentiation cascade each Vm set-point
      triggers (the Gaussian peak as an inductive seed, not a label);

  (4) writes three LaTeX tables for the paper:
        paper5_tab_outer_setpoints.tex  (ABC conductances -> Goldman V)
        paper5_tab_perceptron_trace.tex (every coefficient -> param x equation)
        paper5_tab_vm_cascade.tex       (Vm band -> organ -> cascade)

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.perceptron_trace
"""
from __future__ import annotations

import numpy as np

try:
    from . import bioelectric_development as bd
    from .zebrafish_somitogenesis import Her1Oscillator
    from . import morphogen_rd as mrd
    from . import motility_clock as mc
except ImportError:  # pragma: no cover
    from medic import bioelectric_development as bd
    from medic.zebrafish_somitogenesis import Her1Oscillator
    from medic import morphogen_rd as mrd
    from medic import motility_clock as mc


# ---------------------------------------------------------------------------
# (1) OUTER perceptron: ABC activity -> conductances -> Goldman V_target
# ---------------------------------------------------------------------------
def goldman(g_Na, g_K, g_Ca, g_Cl):
    """GHK steady state with the model's resting-Ca fraction (g_Ca_rest=0.10 g_Ca)."""
    g_Ca_rest = g_Ca * 0.10
    num = (g_Na * bd.E_NA + g_K * bd.E_K + g_Ca_rest * bd.E_CA + g_Cl * bd.E_CL)
    den = g_Na + g_K + g_Ca_rest + g_Cl
    return num / (den + 1e-10)


def outer_setpoints():
    """The outer perceptron's emitted per-organ set-points: the conductances it
    writes (from the ABC accessibility read, g_K solved to the Levin target) and
    the Goldman voltage they produce. Returns a list of row dicts."""
    profiles = bd._compute_organ_conductances()
    rows = []
    for organ in bd._ABC_ION_CHANNEL_ACTIVITY:
        # kidney is IMPUTED (no kidney biosample in ABC); excluded as in Papers #1/#4.
        if organ == "kidney":
            continue
        # use the key that carries the real Levin target: organs whose target lives
        # on a paired key (lung -> lung_left, etc.) take the paired profile, which is
        # the one re-solved to that target; otherwise the organ's own profile.
        if organ in bd.ORGAN_PREFERRED_VOLTAGE:
            key = organ
        elif organ + "_left" in profiles:
            key = organ + "_left"
        else:
            continue
        g_Na, g_K, g_Ca, g_Cl, g_gj = profiles[key]
        v = goldman(g_Na, g_K, g_Ca, g_Cl)
        v_target = bd.ORGAN_PREFERRED_VOLTAGE.get(
            organ, bd.ORGAN_PREFERRED_VOLTAGE.get(organ + "_left", -50.0))
        rows.append(dict(organ=organ, g_Na=g_Na, g_K=g_K, g_Ca=g_Ca, g_Cl=g_Cl,
                         g_gj=g_gj, V=v, V_target=v_target))
    return rows


# ---------------------------------------------------------------------------
# (2) INNER perceptron: relaxation rule + clocks + RD rates, with provenance
# ---------------------------------------------------------------------------
def inner_coefficients():
    """Every inner-perceptron coefficient with its value, genomic referent and
    literature equation. Values are read from the live modules where possible."""
    # her1 period from the Lewis 2003 delayed-feedback DDE (computed, not assumed)
    osc = Her1Oscillator()
    T_her1, _ = osc.period()

    # Gierer-Meinhardt defaults (medic.morphogen_rd.gierer_meinhardt signature)
    gm = dict(D_a=0.4, D_h=8.0, rho=0.04, mu_a=0.06, mu_h=0.08, rho0=0.04, kS=0.4)
    # FitzHugh-Nagumo ERK-wave defaults (medic.motility_clock.run signature)
    fhn = dict(D_u=0.9, eps=0.08, a=0.7, b=0.8, D_max=1.0)

    C = []  # (block, symbol, value, genomic referent, literature equation)
    C.append(("Relaxation rule", "k_relax", "0.14",
              "channel turnover / drive toward the genomic set-point",
              "dV/dt = k_relax(V_target-V) + k_gj nabla^2 V  (Mordvintsev 2020 NCA)"))
    C.append(("Relaxation rule", "k_gj", "0.045",
              "gap-junction conductance g_gj (connexin ABC read)",
              "diffusion term of the same local rule"))
    C.append(("Relaxation rule", "V_ZYGOTE", "-70 mV",
              "inherited zygote base (Jadhav methylation fossil record)",
              "V_target = V_ZYGOTE + lora_scale * dV_lora"))
    C.append(("her1 clock", "tau_p", "6.0 min",
              "her1 intron length / transcriptional delay",
              "delayed autorepression DDE (Lewis, Curr Biol 2003)"))
    C.append(("her1 clock", "tau_m", "2.0 min",
              "her1 translational delay",
              "delayed autorepression DDE (Lewis 2003)"))
    C.append(("her1 clock", "Hill n", "2",
              "her1 dimer cooperativity (autorepression)",
              "dm/dt = k_m/(1+(p_del/p0)^n) - a_m m"))
    C.append(("her1 clock", "T (emergent)", f"{T_her1:.1f} min",
              "= f(tau_p, tau_m, decay) -- emergent, not set",
              "peak-to-peak of the integrated DDE; somite S = v*T (Cooke-Zeeman 1976)"))
    C.append(("Eye/heart/limb RD", "D_h/D_a", f"{gm['D_h']/gm['D_a']:.0f}",
              "inhibitor vs activator mobility (diffusible repressor range)",
              "Gierer-Meinhardt activator-inhibitor (Gierer & Meinhardt 1972; Turing 1952)"))
    C.append(("Eye/heart/limb RD", "rho, mu_a, mu_h",
              f"{gm['rho']}, {gm['mu_a']}, {gm['mu_h']}",
              "activator/inhibitor production and decay rates (transcription/degradation)",
              "da/dt = D_a nabla^2 a + rho a^2/h - mu_a a + rho0*comp - kS*Shh*a"))
    C.append(("Eye/heart/limb RD", "rho0*comp", f"{gm['rho0']}",
              "the COMPETENCE input = the Vm-gated trigger (where the cascade may fire)",
              "basal activator source gated by the upstream Vm field"))
    C.append(("Motility (ERK waves)", "eps", f"{fhn['eps']}",
              "ERK activation/inactivation time-scale separation",
              "FitzHugh-Nagumo excitable medium (FitzHugh 1961; Nagumo 1962; Hiratsuka 2015)"))
    C.append(("Motility (FGF gradient)", "D_max, theta", "1.0, 0.40",
              "FGF8-set random-cell-motility gradient (tailbud high)",
              "D_cell = D_max sigmoid((FGF-theta)/w) (Benazeraf et al., Nature 2010)"))
    return C, T_her1


# ---------------------------------------------------------------------------
# (3) Vm -> cascade: which morphogen/differentiation program each set-point fires
# ---------------------------------------------------------------------------
# Each row: the Vm set-point emitted by the outer perceptron opens a COMPETENCE
# window; a local RD peak at that voltage nucleates the named master-TF cascade
# (the Fate head of Paper #4). The Vm is INSTRUCTIVE (Levin bioelectric prepattern),
# not merely correlated -- the ectopic-eye demo (medic.ectopic_eye) tests exactly
# this by firing the eye cascade at an off-target voltage in the tail.
VM_CASCADE = [
    # (Vm band mV, territory / germ layer, master-TF cascade triggered, citation)
    ("-72..-60", "neural plate / eye field (ectoderm, hyperpolarized)",
     "Rx, Pax6, Six3, Lhx2 eye GRN; SoxB1/proneural neural",
     "Pai-Levin 2012 (Vm instructs eye); Zuber 2003 (eye GRN)"),
    ("-45..-38", "lateral-plate / paraxial mesoderm (muscle, limb)",
     "Tbx5/Fgf10 limb bud -> Fgf8-Shh AER/ZPA loop; MyoD/Myf5 myogenic",
     "Zeller 2009; Tabin/Olson"),
    ("-33..-28", "cardiac mesoderm (anterior-ventral)",
     "Nkx2-5, GATA4, Tbx5, Mef2c cardiac CRC",
     "Olson 2006"),
    ("-27..-18", "endoderm (gut, liver, pancreas; depolarized)",
     "FoxA2/GATA4 competence -> Pdx1/Ptf1a (pancreas), Hnf4a (liver)",
     "Zaret 2008"),
    ("oscillatory", "presomitic mesoderm (her1 clock + wavefront)",
     "Mesp2, Tbx6, Ripply segmentation cascade",
     "Pourquie 2011; Cooke-Zeeman 1976"),
]


# ---------------------------------------------------------------------------
# LaTeX emitters
# ---------------------------------------------------------------------------
def _tex_outer(rows, path):
    L = [r"\begin{table}[H]\centering\small",
         r"\caption{The outer perceptron's emitted set-points. Per-organ ion-channel "
         r"conductances (mS\,cm$^{-2}$) from the ABC accessibility read, with $g_K$ "
         r"solved to the Levin target; the Goldman voltage $V$ they produce reproduces "
         r"the target $V^{*}$. Every column is a genomic read or a closed-form solve, "
         r"not a fitted weight.}",
         r"\label{tab:outer-setpoints}",
         r"\begin{tabular}{lrrrrrrr}",
         r"\toprule",
         r"organ & $g_{Na}$ & $g_K$ & $g_{Ca}$ & $g_{Cl}$ & $g_{gj}$ & $V$ & $V^{*}$\\",
         r"\midrule"]
    for r in rows:
        L.append(f"{r['organ']} & {r['g_Na']:.2f} & {r['g_K']:.2f} & {r['g_Ca']:.2f} "
                 f"& {r['g_Cl']:.2f} & {r['g_gj']:.3f} & {r['V']:+.1f} & {r['V_target']:+.0f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"  wrote {path}")


def _tex_trace(coeffs, path):
    L = [r"\begin{table}[H]\centering\small",
         r"\caption{Tracing the inner perceptron. Each coefficient of the inner "
         r"(spatial NCA) perceptron with its value, the genomic parameter it stands "
         r"for, and the literature equation that fixes it. No coefficient is a trained "
         r"weight; the her1 period is emergent from the delay equation.}",
         r"\label{tab:perceptron-trace}",
         r"\begin{tabular}{p{2.4cm}p{1.7cm}p{1.6cm}p{3.4cm}p{4.6cm}}",
         r"\toprule",
         r"block & symbol & value & genomic referent & literature equation\\",
         r"\midrule"]
    for block, sym, val, ref, eq in coeffs:
        esc = lambda s: (s.replace("&", r"\&").replace("_", r"\_")
                         .replace("^", r"\^{}").replace("%", r"\%"))
        L.append(f"{esc(block)} & {esc(sym)} & {esc(val)} & {esc(ref)} & {esc(eq)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"  wrote {path}")


def connexin_matrix():
    """The gap-junction (connexin) conductance per organ, read from the ABC GJ
    accessibility column -- the operator weighting, organ by organ, COMPUTED not
    fitted. Returns rows (organ, gj_raw_ABC, g_gj)."""
    profiles = bd._compute_organ_conductances()
    rows = []
    for organ, acts in bd._ABC_ION_CHANNEL_ACTIVITY.items():
        if organ == "kidney":
            continue
        key = organ if organ in profiles else organ + "_left"
        if key not in profiles:
            continue
        rows.append((organ, acts[4], profiles[key][4]))      # GJ_total raw, g_gj
    return rows


def _tex_connexin(rows, path):
    L = [r"\begin{table}[H]\centering\small",
         r"\caption{The connexin matrix, organ by organ. The gap-junction conductance "
         r"$g_{gj}$ that weights each organ's field operator is read from that organ's "
         r"ABC gap-junction accessibility (the GJ family of \texttt{\_ABC\_ION\_CHANNEL"
         r"\_ACTIVITY}), not fitted. The operator is therefore genome-derived per organ; "
         r"the within-organ (cell-type) connexin map is the same read at finer "
         r"resolution.}",
         r"\label{tab:connexin}",
         r"\begin{tabular}{lrr}",
         r"\toprule",
         r"organ & ABC GJ activity & $g_{gj}$ (mS\,cm$^{-2}$)\\",
         r"\midrule"]
    for organ, raw, g in rows:
        L.append(f"{organ} & {raw:.1f} & {g:.3f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"  wrote {path}")


def _tex_cascade(path):
    L = [r"\begin{table}[H]\centering\small",
         r"\caption{Which morphogen/differentiation cascade each $V_m$ set-point "
         r"triggers. The outer perceptron writes a resting voltage; that voltage opens "
         r"a competence window in which a local reaction-diffusion peak nucleates the "
         r"named master-transcription-factor cascade. The voltage is instructive (a "
         r"bioelectric prepattern), so the cascade is relocatable to any cell driven to "
         r"that voltage -- tested by the ectopic-eye demonstration.}",
         r"\label{tab:vm-cascade}",
         r"\begin{tabular}{p{1.7cm}p{3.6cm}p{4.3cm}p{3.2cm}}",
         r"\toprule",
         r"$V_m$ (mV) & territory / germ layer & master-TF cascade triggered & source\\",
         r"\midrule"]
    for vm, terr, casc, cite in VM_CASCADE:
        esc = lambda s: s.replace("&", r"\&").replace("_", r"\_")
        L.append(f"{esc(vm)} & {esc(terr)} & {esc(casc)} & {esc(cite)}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L) + "\n")
    print(f"  wrote {path}")


def main():
    print("=" * 74)
    print("TRACING THE TWO PERCEPTRONS  --  genomic parameters x literature equations")
    print("=" * 74)

    rows = outer_setpoints()
    print("\n(1) OUTER perceptron set-points (ABC conductances -> Goldman V):")
    print(f"    {'organ':9} {'gNa':>5} {'gK':>6} {'gCa':>5} {'gCl':>5} {'ggj':>6} "
          f"{'V':>7} {'V*':>5}")
    for r in rows:
        print(f"    {r['organ']:9} {r['g_Na']:5.2f} {r['g_K']:6.2f} {r['g_Ca']:5.2f} "
              f"{r['g_Cl']:5.2f} {r['g_gj']:6.3f} {r['V']:+7.1f} {r['V_target']:+5.0f}")
    err = np.mean([abs(r['V'] - r['V_target']) for r in rows])
    print(f"    mean |V - V*| = {err:.2f} mV  (g_K solved to target => near-exact)")

    coeffs, T_her1 = inner_coefficients()
    print("\n(2) INNER perceptron coefficients (each = genomic param x equation):")
    for block, sym, val, ref, _eq in coeffs:
        print(f"    {block:22} {sym:14} = {val:12}  <- {ref}")
    print(f"    her1 period T = {T_her1:.1f} min  (emergent from the Lewis DDE; "
          f"real zebrafish ~30 min)")

    print("\n(3) Vm -> cascade (the Gaussian peak as an inductive trigger):")
    for vm, terr, casc, _c in VM_CASCADE:
        print(f"    {vm:12} {terr:42} -> {casc}")

    cx = connexin_matrix()
    print("\n(4) The connexin matrix (organ by organ, read from ABC GJ accessibility -- COMPUTED):")
    print(f"    {'organ':9} {'ABC GJ':>8} {'g_gj':>7}")
    for organ, raw, g in sorted(cx, key=lambda r: -r[2]):
        print(f"    {organ:9} {raw:8.1f} {g:7.3f}")

    print("\n(5) Writing LaTeX tables:")
    _tex_outer(rows, "paper5_tab_outer_setpoints.tex")
    _tex_trace(coeffs, "paper5_tab_perceptron_trace.tex")
    _tex_cascade("paper5_tab_vm_cascade.tex")
    _tex_connexin(cx, "paper2_tab_connexin.tex")
    print("\nDone.")


if __name__ == "__main__":
    main()
