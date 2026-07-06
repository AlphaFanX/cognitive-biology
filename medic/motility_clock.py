#!/usr/bin/env python3
"""
The MOTILITY clock -- the Migration head's timer (third of the three heads).
============================================================================

Cognimed's NCA/TRM kernel has three readout HEADS: Migration (WHERE cells move),
Fate (WHAT they become), Division (HOW MANY / proliferation). By the symmetry of
the heads, each head has a temporal driver:

    Division  head  <- the TELOMERE / cell-cycle clock  (counts divisions)
    Fate      head  <- the HOX / Polycomb (PRC2) clock  (collinear identity)
    Migration head  <- the MOTILITY clock   (THIS MODULE)

The her1 segmentation clock is NOT a division clock; it is the patterning
oscillator that reads the regressing wavefront to set somite periodicity. All
four are registered to the SAME posteriorly-regressing FGF/Wnt wavefront, which
is the embryo's shared developmental-time reference.

The motility clock has two coupled parts, both real and sourced:

(1) FGF MOTILITY GRADIENT (Benazeraf et al., Nature 2010).
    FGF8 is high in the tailbud and decays anteriorly; it sets a posterior->
    anterior GRADIENT of random cell motility (effective diffusivity D_cell).
    Cells "gel" (motility collapses) as the front passes them -- exactly where
    the her1 clock freezes a somite. So motility falls and a somite forms at the
    same AP level: the migration and segmentation heads meet at the wavefront.

        D_cell(x,t) = D_max * sigmoid( (FGF(x,t) - theta) / w )
        FGF(x,t)    = exp( -(x_tail(t) - x)_+ / lambda )   (tailbud source, regresses)

(2) ERK PULSATILE WAVES (FitzHugh-Nagumo excitable medium; ERK waves: Hiratsuka
    et al., eLife 2015; Aoki et al.). Periodic ERK activation at the tailbud
    launches travelling pulses that sweep anteriorly through the PSM; each pulse
    is a synchronized burst of actomyosin-driven motility -- the clock's "tick".

        du/dt = D_u lap(u) + u - u^3/3 - v + I_tailbud(t)
        dv/dt = eps (u + a - b v)

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.motility_clock
Output: motility_clock.png
"""
from __future__ import annotations

import numpy as np

try:
    from .nca_vertebrate_3d import get_clock, front_ap, somite_ap_center, TRUNK_START, TRUNK_END
except ImportError:  # pragma: no cover
    from medic.nca_vertebrate_3d import get_clock, front_ap, somite_ap_center, TRUNK_START, TRUNK_END

NX = 220
DT = 0.1


def lap1d(u):
    up = np.pad(u, 1, mode="edge")
    return up[2:] + up[:-2] - 2 * up[1:-1]


def run(D_u=0.9, eps=0.08, a=0.7, b=0.8, D_max=1.0, theta=0.40, w=0.05,
        lam=0.14, n_rows=260):
    """Integrate the FGF motility gradient + ERK FitzHugh-Nagumo waves over the
    same developmental time as the somitogenesis run."""
    period, clock = get_clock()
    t_max = clock.n * period
    n_steps = int(t_max / DT)
    x = np.linspace(0, 1, NX)

    u = np.full(NX, -1.2)            # ERK resting (excitable down-state)
    v = np.full(NX, -0.62)
    erk_kymo, mot_kymo, fgf_kymo, t_axis, front_axis, tail_axis = [], [], [], [], [], []
    cap_every = max(1, n_steps // n_rows)
    T_erk = period                   # ERK tick registered to the her1 period

    for s in range(n_steps + 1):
        t = s * DT
        # tailbud regresses posteriorly as the embryo elongates.
        x_tail = 0.30 + 0.66 * (t / t_max)
        fgf = np.exp(-np.clip(x_tail - x, 0, None) / lam)          # tailbud source
        fgf *= (x <= x_tail + 0.02)
        motility = D_max / (1.0 + np.exp(-(fgf - theta) / w))      # Benazeraf gradient

        # ERK excitable medium with periodic tailbud stimulation (the tick).
        phase = (t % T_erk) / T_erk
        I = np.zeros(NX)
        if phase < 0.12:                                          # stimulate near tailbud
            I += 0.9 * np.exp(-((x - x_tail) / 0.04) ** 2)
        u += DT * (D_u * lap1d(u) + u - (u ** 3) / 3.0 - v + I)
        v += DT * (eps * (u + a - b * v))

        if s % cap_every == 0:
            erk_kymo.append(u.copy()); mot_kymo.append(motility.copy())
            fgf_kymo.append(fgf.copy()); t_axis.append(t)
            front_axis.append(front_ap(t, clock)); tail_axis.append(x_tail)

    return dict(x=x, erk=np.array(erk_kymo), mot=np.array(mot_kymo),
                fgf=np.array(fgf_kymo), t=np.array(t_axis), front=np.array(front_axis),
                tail=np.array(tail_axis), clock=clock, period=period, t_max=t_max)


def main():
    print("=" * 74)
    print("MOTILITY CLOCK  --  the Migration head's timer (FGF gradient + ERK waves)")
    print("=" * 74)
    print("Three heads -> three clocks, all read the same regressing wavefront:")
    print("  Division  head : telomere / cell-cycle clock")
    print("  Fate      head : Hox / Polycomb (PRC2) clock")
    print("  Migration head : MOTILITY clock (this module)")
    print("Equations & sources:")
    print("  FGF motility gradient: Benazeraf et al., Nature 2010 (random-motility gradient)")
    print("  ERK pulsatile waves : FitzHugh 1961 / Nagumo 1962; ERK waves Hiratsuka et al. eLife 2015")

    r = run()
    # quick wave count near mid-PSM
    mid = r["erk"][:, NX // 2]
    ticks = int(np.sum((mid[1:-1] > 0.0) & (mid[1:-1] >= mid[:-2]) & (mid[1:-1] > mid[2:])))
    print(f"\nher1 period T = {r['period']:.1f} min; developmental time 0 -> {r['t_max']:.0f} min")
    print(f"ERK waves launched (registered to T): ~{ticks} ticks past mid-axis")
    print(f"motility: high in PSM (posterior, near tailbud), collapses at the gelling front")

    _figure(r)


def _figure(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, t = r["x"], r["t"]
    ext = [0, 1, t[0], t[-1]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))

    # (1) FGF motility gradient kymograph
    im0 = ax[0].imshow(r["mot"], origin="lower", aspect="auto", extent=ext, cmap="inferno")
    ax[0].plot(r["front"], t, "w-", lw=2, label="her1 determination front")
    ax[0].plot(r["tail"], t, "c--", lw=1.5, label="tailbud (elongation)")
    ax[0].set_title("(1) Motility gradient D_cell(x,t)\n(FGF high posterior; gels at front)")
    ax[0].set_xlabel("anterior -> posterior (AP)"); ax[0].set_ylabel("developmental time (min)")
    ax[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(im0, ax=ax[0], fraction=0.04, label="cell motility D_cell")

    # (2) ERK pulsatile-wave kymograph + somite freeze registration
    im1 = ax[1].imshow(r["erk"], origin="lower", aspect="auto", extent=ext, cmap="viridis")
    clock = r["clock"]
    for k in range(clock.n):
        ax[1].plot(somite_ap_center(k, clock), (k + 1) * r["period"], "o",
                   color="#ff5555", ms=4)
    ax[1].set_title("(2) ERK pulsatile waves u(x,t)\n(travelling motility ticks; red = somite freezes)")
    ax[1].set_xlabel("anterior -> posterior (AP)"); ax[1].set_ylabel("developmental time (min)")
    fig.colorbar(im1, ax=ax[1], fraction=0.04, label="ERK activity u")

    # (3) snapshot profiles at mid-run
    j = len(t) // 2
    ax[2].plot(x, r["fgf"][j], label="FGF", color="#e3b341")
    ax[2].plot(x, r["mot"][j], label="motility D_cell", color="#fb8500")
    ax[2].plot(x, (r["erk"][j] - r["erk"][j].min()) / (np.ptp(r["erk"][j]) + 1e-9),
               label="ERK (norm)", color="#3a86ff")
    ax[2].axvline(r["front"][j], color="k", ls=":", lw=1, label="front")
    ax[2].set_title(f"(3) Profiles at t = {t[j]:.0f} min")
    ax[2].set_xlabel("anterior -> posterior (AP)"); ax[2].set_ylabel("level")
    ax[2].legend(fontsize=8)

    fig.suptitle("Motility clock: FGF random-motility gradient + ERK pulsatile waves, "
                 "registered to the her1 wavefront", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig("motility_clock.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: motility_clock.png")


if __name__ == "__main__":
    main()
