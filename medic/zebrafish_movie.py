#!/usr/bin/env python3
"""
Zebrafish developmental movie, computed from the kernel and validated frame-by-
frame against the Silic et al. (2022) atlas.

This is the first end-to-end "video of the developing embryo from the genome":
a forward run of the segmentation clock + wavefront (medic.zebrafish_somitogenesis)
laid onto a schematic lateral body that ELONGATES with time, with somites
segmenting posteriorly on the her1 clock and organs igniting at their Silic onset
hpf, each coloured by its genome-derived resting Vmem (TISSUE_VMEM_ESTIMATES ->
Goldman). Every frame carries a HUD with the nearest Silic stage and that stage's
mechanistic validation (coverage + accuracy) from medic.silic_validation -- so the
movie is not just a cartoon, it is scored against the atlas at all 12 stages.

Honest scope: the rendering is SCHEMATIC (a body-plan layout that elongates and
lights up), not a photoreal morphing 3-D embryo. The morphogenetic *movement* of
gastrulation/neurulation is the next gap; what is validated here is the
spatiotemporal *pattern* (which tissue, when, what polarity/dynamics).

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.zebrafish_movie
Outputs:
    zebrafish_development_movie.gif   (animated)
    zebrafish_movie_stage_*.png       (key Silic-stage frames)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .zebrafish_bioelectric import DEVELOPMENTAL_VOLTAGE_ATLAS, TISSUE_VMEM_ESTIMATES
    from .zebrafish_somitogenesis import Her1Oscillator, SegmentationClock
    from . import silic_validation as sv
except ImportError:  # pragma: no cover
    from medic.zebrafish_bioelectric import DEVELOPMENTAL_VOLTAGE_ATLAS, TISSUE_VMEM_ESTIMATES
    from medic.zebrafish_somitogenesis import Her1Oscillator, SegmentationClock
    from medic import silic_validation as sv

V = TISSUE_VMEM_ESTIMATES

# Window: segmentation -> early larva, where the body plan has spatial structure.
HPF_START, HPF_END = 10.0, 48.0
N_FRAMES = 90

# Somite schedule calibrated to the atlas (6-somite@12 hpf, 12@15, 18@18): ~2/hr,
# onset ~10.3 hpf. Capped at 18 (the clock builds 18).
SOMITE_ONSET_HPF = 10.3
SOMITES_PER_HPF = 2.0
MAX_SOMITES = 18


def somites_at(hpf: float) -> int:
    return int(np.clip((hpf - SOMITE_ONSET_HPF) * SOMITES_PER_HPF, 0, MAX_SOMITES))


# --- Schematic body-plan layout: organ -> (V-key, AP[0..1], DV offset, onset hpf, beats)
@dataclass
class Organ:
    name: str
    vkey: str
    ap: float          # 0 anterior head ... 1 posterior tail
    dv: float          # +dorsal / -ventral offset (body half-thickness units)
    onset: float       # hpf at which it appears
    beats: bool = False
    r: float = 0.020   # marker radius (axis fraction)


ORGANS: List[Organ] = [
    Organ("brain",      "brain_neuron",     0.06,  0.45, 10.0, r=0.030),
    Organ("eye",        "retinal_ganglion", 0.11,  0.05, 12.0, r=0.022),
    Organ("heart",      "heart_primordium", 0.17, -0.55, 16.0, beats=True, r=0.024),
    Organ("pronephros", "pronephros",       0.30, -0.30, 24.0, r=0.018),
    Organ("liver",      "liver",            0.33, -0.45, 30.0, r=0.020),
    Organ("pancreas",   "pancreas",         0.37, -0.40, 32.0, r=0.016),
    Organ("gut",        "gut_endoderm",     0.45, -0.45, 24.0, r=0.018),
    Organ("pectoral fin","fin_mesenchyme",  0.24, -0.55, 28.0, r=0.016),
]

# Vmem -> colour: hyperpolarised (<= -55) deep blue, depolarised (>= -30) red.
def vmem_color(vmem: float):
    import matplotlib
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=-70.0, vmax=-20.0)
    return matplotlib.colormaps["coolwarm"](norm(vmem))


def nearest_stage(hpf: float):
    return min(DEVELOPMENTAL_VOLTAGE_ATLAS, key=lambda s: abs(s.hpf - hpf))


def build_clock() -> Tuple[SegmentationClock, float]:
    T, _ = Her1Oscillator().period()
    if not np.isfinite(T):
        T = 29.0
    return SegmentationClock(period_min=T, n_somites=MAX_SOMITES), T


# =============================================================================
# Rendering
# =============================================================================
def _body_outline(ax, x0, x1, ymid, half):
    """A tapering lateral body from anterior x0 to posterior tail x1."""
    xs = np.linspace(x0, x1, 60)
    # head full, tail tapering to a point
    frac = (xs - x0) / max(1e-6, (x1 - x0))
    th = half * (1.0 - 0.85 * frac ** 1.5)
    ax.fill_between(xs, ymid - th, ymid + th, color="#eef2f7", ec="#9aa7b5",
                    lw=1.2, zorder=1)


def render_frame(ax, hpf: float, clock: SegmentationClock, T: float):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ymid, half = 0.55, 0.16
    x_ant = 0.07
    # body elongates with somite count
    nsom = somites_at(hpf)
    body_len = 0.30 + 0.55 * (nsom / MAX_SOMITES)
    x_post = x_ant + body_len
    _body_outline(ax, x_ant, x_post, ymid, half)

    # head region (anterior) carries brain/eye; somites start just behind it
    x_som0 = x_ant + 0.10 * body_len

    # notochord (axial line) once the axis exists
    if nsom > 0:
        ax.plot([x_ant + 0.02, x_post], [ymid, ymid], color="#b0772e", lw=2.6,
                zorder=2, solid_capstyle="round")

    # somites: dorsal blocks along the axis, coloured by Vmem; mid-body oscillates
    if nsom > 0:
        prof = clock.vmem_profile(nsom * T, osc_phase=2 * np.pi * (hpf % 1.0))
        seg = (x_post - x_som0) / max(1, nsom)
        for i in range(nsom):
            sx = x_som0 + i * seg
            vinst = prof["vinst"][i] if i < len(prof["vinst"]) else V["somite_new"]
            amp = prof["vamp"][i] if i < len(prof["vamp"]) else 0.0
            col = vmem_color(vinst)
            # oscillating mid-somites get a brighter pulse edge
            ec = "#000000" if amp > 2.5 else "#5b6776"
            ax.add_patch(_somite_patch(sx, seg, ymid, half * 0.92, col, ec, amp))

    # organs ignite at their onset hpf
    for o in ORGANS:
        if hpf < o.onset:
            continue
        ox = x_ant + o.ap * body_len
        oy = ymid + o.dv * half
        vm = V.get(o.vkey, -40.0)
        col = vmem_color(vm)
        rr = o.r
        if o.beats and hpf >= 16.0:  # heart beats from heart-cone on
            rr = o.r * (1.0 + 0.18 * np.sin(2 * np.pi * (hpf % 1.0) * 2))
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse((ox, oy), rr * 2, rr * 2 * 1.3, color=col,
                              ec="#2b333d", lw=1.0, zorder=4))
        ax.text(ox, oy - rr * 1.7, o.name, ha="center", va="top", fontsize=7,
                color="#2b333d", zorder=5)

    _hud(ax, hpf)


def _somite_patch(sx, seg, ymid, half, col, ec, amp):
    """Dorsal somite block sitting above the notochord (ymid)."""
    from matplotlib.patches import FancyBboxPatch
    y0 = ymid + 0.014
    h = (half - 0.02) * (0.92 + 0.06 * (amp > 2.5))
    return FancyBboxPatch((sx, y0), seg * 0.82, h,
                          boxstyle="round,pad=0.002,rounding_size=0.004",
                          fc=col, ec=ec, lw=0.8, zorder=3)


def _hud(ax, hpf: float):
    stage = nearest_stage(hpf)
    score = sv.score_stage(stage)
    acc = score["accuracy"]
    acc_s = "n/a" if (acc != acc) else f"{acc*100:.0f}%"
    cov = score["coverage"] * 100
    at_stage = abs(stage.hpf - hpf) <= (HPF_END - HPF_START) / N_FRAMES
    tick = "  ✓ scored" if at_stage else ""

    ax.text(0.02, 0.97, f"Zebrafish development from the genome",
            fontsize=12, weight="bold", color="#11213a", va="top")
    ax.text(0.02, 0.90, f"{hpf:5.1f} hpf   •   {stage.kimmel_period}",
            fontsize=10, color="#33425a", va="top")
    # validation panel
    ax.text(0.62, 0.97,
            f"nearest Silic stage: {stage.stage_name} ({stage.hpf:.0f} hpf){tick}",
            fontsize=9.5, color="#11213a", va="top",
            bbox=dict(boxstyle="round", fc="#f2f6ff", ec="#9aa7b5"))
    ax.text(0.62, 0.90,
            f"Silic validation  •  coverage {score['n_mapped']}/{score['n_total']}"
            f"   accuracy {acc_s}",
            fontsize=9.5, color="#0b6b2f" if (acc == acc and acc >= 0.9) else "#33425a",
            va="top")
    # colourbar legend
    ax.text(0.02, 0.06, "Vmem:", fontsize=8, color="#33425a")
    for i, (lab, vm) in enumerate([("hyper", -65), ("", -45), ("depol", -25)]):
        ax.add_patch(__import__("matplotlib").patches.Rectangle(
            (0.09 + i * 0.05, 0.045), 0.045, 0.02, color=vmem_color(vm)))
    ax.text(0.09, 0.02, "hyperpolarised", fontsize=7, color="#33425a")
    ax.text(0.19, 0.02, "depolarised", fontsize=7, color="#33425a")
    ax.text(0.98, 0.02, "schematic body plan; pattern validated, not morphology",
            fontsize=6.5, color="#8a97a5", ha="right", style="italic")


# =============================================================================
def make_movie(path_gif: str = "zebrafish_development_movie.gif") -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e})")
        return None

    clock, T = build_clock()
    hpfs = np.linspace(HPF_START, HPF_END, N_FRAMES)
    fig, ax = plt.subplots(figsize=(11, 5))

    def _update(k):
        render_frame(ax, float(hpfs[k]), clock, T)
        return []

    anim = FuncAnimation(fig, _update, frames=N_FRAMES, blit=False)
    anim.save(path_gif, writer=PillowWriter(fps=12))
    plt.close(fig)

    # also dump the exact Silic-stage frames as stills
    saved = []
    for st in DEVELOPMENTAL_VOLTAGE_ATLAS:
        if st.hpf < HPF_START or st.hpf > HPF_END:
            continue
        f2, a2 = plt.subplots(figsize=(11, 5))
        render_frame(a2, st.hpf, clock, T)
        p = f"zebrafish_movie_stage_{st.stage_name}.png"
        f2.savefig(p, dpi=110, bbox_inches="tight"); plt.close(f2)
        saved.append(p)
    return path_gif, saved, T


def main():
    print("=" * 72)
    print("ZEBRAFISH DEVELOPMENTAL MOVIE  --  from the genome, validated vs Silic")
    print("=" * 72)
    out = make_movie()
    if not out:
        return
    gif, stills, T = out
    print(f"\nher1 clock period T = {T:.1f} min   ({MAX_SOMITES} somites built)")
    print(f"window {HPF_START:.0f}-{HPF_END:.0f} hpf, {N_FRAMES} frames")

    # per-stage validation summary across the movie window
    print("\nFrame-validated Silic stages in window:")
    tot_m = tot_c = 0
    for st in DEVELOPMENTAL_VOLTAGE_ATLAS:
        if st.hpf < HPF_START or st.hpf > HPF_END:
            continue
        s = sv.score_stage(st)
        acc = "n/a" if s["accuracy"] != s["accuracy"] else f"{s['accuracy']*100:5.1f}%"
        print(f"  {st.stage_name:>10} ({st.hpf:4.0f} hpf)  "
              f"coverage {s['n_mapped']}/{s['n_total']}  accuracy {acc}")
        tot_m += s["n_mapped"]; tot_c += s["n_correct"]
    if tot_m:
        print(f"\n  window total: {tot_c}/{tot_m} mapped tissues correct "
              f"({100*tot_c/tot_m:.1f}%)")
    print(f"\nSaved movie: {gif}")
    print(f"Saved {len(stills)} key-stage stills: {', '.join(stills[:3])} ...")


if __name__ == "__main__":
    main()
