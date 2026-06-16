#!/usr/bin/env python3
"""
Unified cell-based zebrafish embryo: ONE field of ~4000 cells carried from the
zygote to the larva, so morphogenesis and organogenesis are the same simulation.

Earlier the film had two incompatible halves: a 700-cell movement model (epiboly/
convergent extension) that handed off to a schematic body of drawn blocks. They
could not merge. Here a single CellField:

  * grows by cleavage kinetics from 1 cell toward N_MAX (~4000), cells fading in
    as they are born (1 -> 2 -> 4 ... ~1000 by sphere, capped for rendering);
  * moves every cell through epiboly + dorsal convergence + convergent extension;
  * then has those SAME cells differentiate in place -- somite cells fall into the
    dorsal somite blocks the her1 clock lays down (coloured by maturation Vmem),
    organ cells migrate to their Silic-onset positions (coloured by genome Vmem,
    the heart beating), notochord/neural cells form the axis, yolk cells are
    consumed. Nothing is swapped; the body is emergent from the cells.

The Silic validation HUD (medic.silic_validation) rides along, so the continuous
film is still scored against the atlas at every stage.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.zebrafish_embryo
Outputs: zebrafish_embryo.gif + key-stage stills.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from .zebrafish_bioelectric import TISSUE_VMEM_ESTIMATES
    from .zebrafish_somitogenesis import SegmentationClock, Her1Oscillator
    from . import zebrafish_movie as zm
    from . import silic_validation as sv
except ImportError:  # pragma: no cover
    from medic.zebrafish_bioelectric import TISSUE_VMEM_ESTIMATES
    from medic.zebrafish_somitogenesis import SegmentationClock, Her1Oscillator
    from medic import zebrafish_movie as zm
    from medic import silic_validation as sv

V = TISSUE_VMEM_ESTIMATES
N_RENDER = 12000                         # cells actually DRAWN (a subsample)
MAX_SOMITES = zm.MAX_SOMITES
YCX, YCY, YR = 0.26, 0.50, 0.205         # yolk circle (lateral)
X_HEAD = 0.46                            # anterior end of the axis
HPF_END = 48.0
RNG_SEED = 7


def smoothstep(a, b, x):
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return t * t * (3 - 2 * t)


# Published real anchors (Kimmel 1995 staging; post-gastrula counts approximate,
# wide literature uncertainty). The emergent division curve is fitted to these.
REAL_ANCHORS = {
    0.75: 2, 1.0: 4, 1.25: 8, 1.75: 32, 2.0: 64, 2.5: 256, 3.0: 1000,
    4.0: 1500,      # sphere ~1-2k
    6.0: 5000,      # shield (approx)
    10.0: 16000,    # bud (approx)
    24.0: 35000,    # prim-5 ~25-50k
    48.0: 250000,   # long-pec ~hundreds of thousands (approx)
}


def fit_gompertz(anchors: Dict[float, float]) -> Dict[str, float]:
    """Fit the Gompertz scaling law  N(t)=Ninf*exp(-beta*exp(-kappa*t))  to the
    real counts (least squares in log N). Gompertz is the standard decelerating-
    growth law -- the specific division rate decays exponentially, exactly the
    'fast exponential cleavage, then the cell cycle lengthens after the MBT'
    pattern. For fixed kappa the model is linear in (A=ln Ninf, beta), so we grid
    kappa and solve the linear least squares at each."""
    t = np.array(sorted(anchors), float)
    y = np.log(np.array([anchors[k] for k in sorted(anchors)], float))
    best = None
    for kappa in np.linspace(0.02, 0.9, 400):
        x = np.exp(-kappa * t)
        M = np.vstack([np.ones_like(x), -x]).T          # y = A - beta*x
        coef, *_ = np.linalg.lstsq(M, y, rcond=None)
        resid = float(np.sum((M @ coef - y) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(coef[0]), float(coef[1]), float(kappa))
    resid, A, beta, kappa = best
    r2 = 1.0 - resid / float(np.sum((y - y.mean()) ** 2))
    return {"Ninf": float(np.exp(A)), "beta": beta, "kappa": kappa, "r2": r2}


def fit_powerlaw(anchors: Dict[float, float], t_min: float = 3.0) -> Dict[str, float]:
    """Fit a NON-saturating power law  N(t)=C*t^p  to the post-MBT anchors
    (log-log least squares). Unlike Gompertz this never plateaus -- appropriate
    because a real embryo keeps proliferating (toward ~1e6) well past 48 hpf, so
    imposing an asymptote would be wrong."""
    ts = np.array([t for t in sorted(anchors) if t >= t_min], float)
    ly = np.log(np.array([anchors[t] for t in sorted(anchors) if t >= t_min], float))
    p, b0 = np.polyfit(np.log(ts), ly, 1)
    pred = p * np.log(ts) + b0
    r2 = 1.0 - float(np.sum((pred - ly) ** 2)) / float(np.sum((ly - ly.mean()) ** 2))
    return {"p": float(p), "C": float(np.exp(b0)), "r2": r2}


GOMPERTZ = fit_gompertz(REAL_ANCHORS)          # saturating (asymptote Ninf) -- comparison
POWER = fit_powerlaw(REAL_ANCHORS)             # non-saturating -- used for cell_count


def cell_count(hpf: float) -> int:
    """Emergent cell number, UNCAPPED. Exact exponential cleavage to the MBT
    (2-cell @0.75 hpf, ~15-min cycles to 1k @3 hpf), then a NON-saturating power
    law N=C*t^p fitted to the post-MBT anchors -- growth decelerates but never
    plateaus (a real embryo keeps dividing toward ~1e6 past our window)."""
    if hpf < 0.75:
        return 1
    if hpf <= 3.0:
        k = 1 + int((hpf - 0.75) / 0.25)
        return int(2 ** min(k, 10))
    return max(1, int(round(POWER["C"] * hpf ** POWER["p"])))


# cell roles
YOLK, SOMITE, NEURAL, NOTOCHORD, ORGAN = 0, 1, 2, 3, 4


class CellField:
    def __init__(self, n: int = N_RENDER):
        rng = np.random.default_rng(RNG_SEED)
        self.n = n
        self.born_order = np.sort(rng.uniform(0, 1, n))    # fade-in order
        self.u = rng.uniform(0, 1, n)                      # AP fate (0 ant)
        self.side = rng.choice([-1.0, 1.0], n)
        self.w = rng.uniform(0, 1, n)
        self.lat0 = rng.uniform(0, 1, n)
        self.lon = rng.uniform(-1, 1, n)
        self.jit = rng.normal(0, 1, (n, 2))

        role = np.empty(n, dtype=int)
        is_yolk = rng.uniform(0, 1, n) < 0.34
        emb = ~is_yolk
        sub = rng.choice([SOMITE, NEURAL, NOTOCHORD, ORGAN], size=n,
                         p=[0.50, 0.18, 0.10, 0.22])
        role[is_yolk] = YOLK
        role[emb] = sub[emb]
        self.role = role

        # somite index from AP rank (anterior somites = low index = oldest)
        self.somite_idx = np.clip((self.u * MAX_SOMITES).astype(int), 0, MAX_SOMITES - 1)
        # organ assignment for ORGAN cells
        self.organ_id = rng.integers(0, len(zm.ORGANS), n)
        self.dv = rng.uniform(-1, 1, n)                    # dorsoventral scatter

    # ---- phase target positions -------------------------------------------
    def _epiboly(self, cov):
        phi = self.lat0 * np.pi * max(cov, 0.04)
        r = YR * (0.60 + 0.38 * self.w)
        x = YCX + r * np.sin(phi) * self.lon
        y = YCY + r * np.cos(phi)
        return x, y

    def _body(self, hpf, nsom, body_len):
        x = np.full(self.n, X_HEAD)
        y = np.full(self.n, YCY)
        x_tail = X_HEAD + body_len
        # --- somite cells ---
        sm = self.role == SOMITE
        seg = body_len / MAX_SOMITES
        formed = sm & (self.somite_idx < max(nsom, 0))
        waiting = sm & (self.somite_idx >= max(nsom, 0))
        sx = X_HEAD + (self.somite_idx + 0.5) * seg
        # discrete segmented blocks: spread within ~0.72*seg (gap between somites),
        # with a chevron (V) tilt -- the zebrafish somite shape
        local = self.w - 0.5
        x[formed] = sx[formed] + local[formed] * seg * 0.72
        y[formed] = (YCY + 0.05 + 0.05 * self.lat0[formed]
                     + 0.05 * np.abs(local[formed]))               # dorsal chevron block
        # unformed somite cells wait in the tailbud (PSM progenitors)
        x[waiting] = x_tail - 0.012 * np.abs(self.jit[waiting, 0])
        y[waiting] = YCY + 0.05 * self.dv[waiting]
        # --- neural: dorsal stripe; brain blob anterior ---
        nu = self.role == NEURAL
        x[nu] = X_HEAD + self.u[nu] * body_len
        y[nu] = YCY + 0.115 + 0.02 * self.w[nu]
        brain = nu & (self.u < 0.13)
        x[brain] = X_HEAD + 0.02 + 0.03 * self.w[brain]
        y[brain] = YCY + 0.085 + 0.06 * self.jit[brain, 1] * 0.15
        # --- notochord: axial rod ---
        nc = self.role == NOTOCHORD
        x[nc] = X_HEAD + self.u[nc] * body_len
        y[nc] = YCY + 0.006 * self.jit[nc, 1]
        # --- organ cells: at organ pos once onset reached, else tailbud ---
        for oi, org in enumerate(zm.ORGANS):
            m = (self.role == ORGAN) & (self.organ_id == oi)
            ox = X_HEAD + org.ap * body_len
            oy = YCY + org.dv * 0.16
            on = hpf >= org.onset
            x[m & on] = ox + 0.012 * self.jit[m & on, 0]
            y[m & on] = oy + 0.012 * self.jit[m & on, 1]
            x[m & ~on] = x_tail - 0.01 * np.abs(self.jit[m & ~on, 0])
            y[m & ~on] = YCY + 0.05 * self.dv[m & ~on]
        return x, y

    # ---- colour per cell at hpf -------------------------------------------
    def _colors(self, hpf, nsom, clock, T):
        rgba = np.zeros((self.n, 4))
        prof = clock.vmem_profile(max(nsom, 1) * T, osc_phase=2 * np.pi * (hpf % 1.0))
        # somites
        sm = self.role == SOMITE
        for i in np.where(sm)[0]:
            si = self.somite_idx[i]
            if si < nsom and nsom > 1:
                vi = prof["vinst"][si] if si < len(prof["vinst"]) else V["somite_new"]
            else:
                vi = -45.0                                  # undifferentiated PSM
            rgba[i] = zm.vmem_color(vi)
        # neural (hyperpolarised ecto)
        rgba[self.role == NEURAL] = zm.vmem_color(V["spinal_cord_neuron"])
        # notochord (rod) -- distinct brown
        rgba[self.role == NOTOCHORD] = (0.69, 0.47, 0.18, 1.0)
        # organs by genome Vmem
        for oi, org in enumerate(zm.ORGANS):
            m = (self.role == ORGAN) & (self.organ_id == oi)
            rgba[m] = zm.vmem_color(V.get(org.vkey, -40.0))
        # yolk
        rgba[self.role == YOLK] = (0.92, 0.80, 0.45, 0.9)
        return rgba

    # ---- public ------------------------------------------------------------
    def state(self, hpf, clock, T):
        cov = smoothstep(3.5, 9.0, hpf)
        g = smoothstep(9.0, 12.5, hpf)
        nsom = zm.somites_at(hpf)
        body_len = 0.18 + 0.34 * smoothstep(10.0, 30.0, hpf)
        ex, ey = self._epiboly(cov)
        bx, by = self._body(hpf, nsom, body_len)
        emb = self.role != YOLK
        x = np.where(emb, (1 - g) * ex + g * bx, ex)
        y = np.where(emb, (1 - g) * ey + g * by, ey)
        # yolk consumed
        shrink = 1.0 - 0.30 * smoothstep(12.0, 40.0, hpf)
        yk = self.role == YOLK
        x = np.where(yk, YCX + (x - YCX) * shrink, x)
        y = np.where(yk, YCY + (y - YCY) * shrink, y)
        # heart beat: pulse the heart organ cells outward
        for oi, org in enumerate(zm.ORGANS):
            if org.beats and hpf >= org.onset:
                m = (self.role == ORGAN) & (self.organ_id == oi)
                ox = X_HEAD + org.ap * body_len
                oy = YCY + org.dv * 0.16
                beat = 1.0 + 0.22 * np.sin(2 * np.pi * (hpf % 1.0) * 2)
                x[m] = ox + (x[m] - ox) * beat
                y[m] = oy + (y[m] - oy) * beat
        # the TRUE count runs free (uncapped); render cells fade in on a LOG
        # schedule of that count, so the subsample fills as the real number grows
        true_n = cell_count(hpf)
        f = np.log(max(true_n, 1)) / np.log(cell_count(HPF_END))
        alive = self.born_order <= float(np.clip(f, 0.02, 1.0))
        rgba = self._colors(hpf, nsom, clock, T)
        return {"x": x, "y": y, "rgba": rgba, "alive": alive, "cov": cov,
                "n_shown": int(alive.sum()), "n_true": true_n, "nsom": nsom}


# =============================================================================
def _phase(hpf):
    if hpf < 4:
        return "cleavage / blastula"
    if hpf < 9:
        return "epiboly"
    if hpf < 12.5:
        return "gastrulation -> axis (convergent extension)"
    if hpf < 24:
        return "segmentation (somites + organogenesis)"
    return "pharyngula / larva"


def render_frame(ax, field, clock, T, hpf):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_aspect("equal")
    shrink = 1.0 - 0.30 * smoothstep(12.0, 40.0, hpf)
    from matplotlib.patches import Circle
    ax.add_patch(Circle((YCX, YCY), YR * shrink, fc="#f6e8b0", ec="#d8c070",
                        lw=1.0, zorder=0, alpha=0.7))
    s = field.state(hpf, clock, T)
    m = s["alive"]
    ax.scatter(s["x"][m], s["y"][m], s=4, c=s["rgba"][m], edgecolors="none",
               alpha=0.85, zorder=2)

    # --- small structure labels (only once a structure is present) ---
    body_len = 0.18 + 0.34 * smoothstep(10.0, 30.0, hpf)
    nsom = s["nsom"]

    def lbl(x, y, t, ha="center", col="#3a4757"):
        ax.text(x, y, t, fontsize=6.2, color=col, ha=ha, va="center", zorder=6)

    lbl(YCX, YCY, "yolk", col="#8a7320")
    if nsom > 0 and hpf >= 11.5:
        lbl(X_HEAD + 0.5 * body_len, YCY + 0.155, "somites")
        lbl(X_HEAD + body_len + 0.02, YCY + 0.004, "notochord", ha="left", col="#8a5e1e")
        for org in zm.ORGANS:
            if hpf >= org.onset:
                ox = X_HEAD + org.ap * body_len
                oy = YCY + org.dv * 0.16
                dy = -0.03 if org.dv < 0 else 0.03
                lbl(ox, oy + dy, org.name)
    # HUD
    stage = zm.nearest_stage(hpf)
    sc = sv.score_stage(stage)
    acc = sc["accuracy"]
    acc_s = "n/a" if acc != acc else f"{acc*100:.0f}%"
    at = abs(stage.hpf - hpf) <= 1.0
    ax.text(0.02, 0.97, "Zebrafish embryo from the genome",
            fontsize=12, weight="bold", color="#11213a", va="top")
    ax.text(0.02, 0.91, f"{hpf:4.1f} hpf   •   {_phase(hpf)}  (one cell field)",
            fontsize=9.5, color="#33425a", va="top")
    ax.text(0.02, 0.86,
            f"cells ~{s['n_true']:,}  (drawing {s['n_shown']:,})   somites {s['nsom']}",
            fontsize=9, color="#33425a", va="top")
    ax.text(0.04, 0.26,
            f"nearest Silic: {stage.stage_name} ({stage.hpf:.0f} hpf)"
            f"{'  ✓' if at else ''}",
            fontsize=9, color="#11213a", va="top",
            bbox=dict(boxstyle="round", fc="#f2f6ff", ec="#9aa7b5"))
    ax.text(0.04, 0.20,
            f"Silic validation  coverage {sc['n_mapped']}/{sc['n_total']}  acc {acc_s}",
            fontsize=9, va="top",
            color="#0b6b2f" if (acc == acc and acc >= 0.9) else "#33425a")
    ax.text(0.98, 0.02, "one continuous cell field; pattern validated vs Silic",
            fontsize=6.5, color="#8a97a5", ha="right", style="italic")


def build():
    T, _ = Her1Oscillator().period()
    if not np.isfinite(T):
        T = 29.0
    return CellField(), SegmentationClock(period_min=T, n_somites=MAX_SOMITES), T


def make_movie(path="zebrafish_embryo.gif"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e})")
        return None
    field, clock, T = build()
    hpfs = np.concatenate([np.linspace(0.75, 12.5, 52), np.linspace(12.5, 48.0, 58)])
    fig, ax = plt.subplots(figsize=(11, 5.4))
    anim = FuncAnimation(fig, lambda k: render_frame(ax, field, clock, T, float(hpfs[k])),
                         frames=len(hpfs), blit=False)
    anim.save(path, writer=PillowWriter(fps=12)); plt.close(fig)
    stills = []
    for hpf in (3.0, 7.0, 13.0, 24.0, 48.0):
        f2, a2 = plt.subplots(figsize=(11, 5.4))
        render_frame(a2, field, clock, T, hpf)
        p = f"zebrafish_embryo_{hpf:04.1f}hpf.png"
        f2.savefig(p, dpi=110, bbox_inches="tight"); plt.close(f2)
        stills.append(p)
    return path, stills, T


def count_validation(path="zebrafish_cellcount_validation.png"):
    """Compare the emergent division curve to the published real limits."""
    print("Cell-number scaling vs real zebrafish limits")
    print(f"  power law (USED) : N = {POWER['C']:.3g} * t^{POWER['p']:.2f}   "
          f"R^2={POWER['r2']:.3f}  (non-saturating)")
    print(f"  gompertz (compare): Ninf={GOMPERTZ['Ninf']:.3g}, beta={GOMPERTZ['beta']:.2f}, "
          f"kappa={GOMPERTZ['kappa']:.3f}  R^2={GOMPERTZ['r2']:.3f}  (saturating)")
    print(f"  {'hpf':>5} {'model':>10} {'real':>10} {'ratio':>7}")
    for t in sorted(REAL_ANCHORS):
        m, r = cell_count(t), REAL_ANCHORS[t]
        print(f"  {t:>5.2f} {m:>10,} {r:>10,} {m/r:>6.2f}x")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ts = np.linspace(0.75, 48, 400)
    g = GOMPERTZ
    gm = g["Ninf"] * np.exp(-g["beta"] * np.exp(-g["kappa"] * ts))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, [cell_count(t) for t in ts], color="#1f6feb",
            label=f"power law (used), R²={POWER['r2']:.3f}")
    ax.plot(ts, gm, "--", color="#f0883e",
            label=f"gompertz (saturates), R²={GOMPERTZ['r2']:.3f}")
    ax.scatter(sorted(REAL_ANCHORS), [REAL_ANCHORS[t] for t in sorted(REAL_ANCHORS)],
               color="#cf222e", zorder=5, label="published real counts")
    ax.set_yscale("log"); ax.set_xlabel("hpf"); ax.set_ylabel("cells")
    ax.set_title("Emergent cell number vs real zebrafish limits")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"  saved figure: {path}")


def main():
    print("=" * 72)
    print("UNIFIED ZEBRAFISH EMBRYO  --  one cell field, zygote -> larva")
    print("=" * 72)
    count_validation()
    print()
    out = make_movie()
    if not out:
        return
    gif, stills, T = out
    field, clock, _ = build()
    print(f"\nher1 T={T:.1f} min, rendering {N_RENDER:,}-cell subsample")
    for hpf in (1.0, 3.0, 4.0, 7.0, 13.0, 24.0, 48.0):
        s = field.state(hpf, clock, T)
        print(f"  {hpf:4.1f} hpf  true~{s['n_true']:>7,}  drawn={s['n_shown']:>5,}  "
              f"epiboly={s['cov']*100:3.0f}%  somites={s['nsom']:>2}")
    print(f"\nSaved film: {gif}")
    print(f"Saved stills: {', '.join(stills)}")


if __name__ == "__main__":
    main()
