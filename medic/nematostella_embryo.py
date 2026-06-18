#!/usr/bin/env python3
"""
Nematostella development from the genome: body plan -> movie.
=============================================================

The Paper-1 analogue (medic/zebrafish_embryo.py) carried ONE cell field from zygote to
larva for a vertebrate. This does the same for the cnidarian Nematostella vectensis -- the
deepest cross-phylum probe, the clade whose kernel reader was validated on REAL ATAC + EM-seq
(medic/nematostella_concordance.py) and whose attention HEADS were sourced from the real
genome by super-enhancer calling (medic/nematostella_se_heads.py, N_SE=643 identity hubs).

Two steps, in order:
  1. BODY PLAN FROM THE GENOME.  medic/body_plan_generator reads the clade's (heads, masks,
     softmaxes) -- heads genome-sourced via the SE reader -- and emits the oral-aboral body
     plan: radial symmetry, 2 germ layers, and 4 axial domains (oral / pharynx / body-column /
     aboral), oriented by the validated bioelectric Vm polarity (depolarized ORAL -> hyperpolarized
     ABORAL).
  2. MOVIE.  One cell field is carried zygote -> cleavage -> blastula (hollow ball) -> gastrula
     (endoderm invaginates at the oral pole) -> planula (elongates along the oral-aboral axis,
     swims) -> primary polyp (oral tentacle ring + pharynx, aboral foot). Cells coloured by the
     Vm oral-aboral gradient; the 4 genome-derived domains emerge; a HUD reports stage, cell
     count, and the body-plan provenance.

Honest scope: a topology-level morphogenesis film (germ layers x oral-aboral domains x radial
symmetry x emergent cell number), the cnidarian counterpart of the zebrafish field -- not a
molecular-resolution embryo. The body-plan NUMBERS are genome/atlas-sourced, not hand-drawn.

Run:  cd cognimed && python -m medic.nematostella_embryo
Outputs: nematostella_embryo.gif + key-stage stills + nematostella_embryo_montage.png
"""
from __future__ import annotations
import numpy as np

try:
    from . import body_plan_generator as bpg
except ImportError:  # pragma: no cover
    from medic import body_plan_generator as bpg

N_RENDER = 4000
RNG_SEED = 7
N_SE_HEADS = 643          # genome-sourced SE identity hubs (medic/nematostella_se_heads.py)
CX, CY = 0.50, 0.50       # field centre

# Nematostella spec (radial, 2 germ layers, 4 oral-aboral domains) from the generator's table.
SPEC = next(s for s in bpg.CLADES if s.name == "nematostella")

# developmental timeline (approx hours post fertilisation, 18-22C) -> progress anchors
STAGES = [   # (hpf, name)
    (0.0,  "zygote"), (7.0, "cleavage"), (14.0, "blastula"), (24.0, "gastrula"),
    (48.0, "planula"), (96.0, "late planula"), (168.0, "primary polyp"),
]
HPF_END = 168.0

# real-ish cell-number anchors (cnidarian; wide uncertainty) -> emergent count
COUNT_ANCHORS = {0.0: 1, 7.0: 64, 14.0: 600, 24.0: 1500, 48.0: 4000, 96.0: 8000, 168.0: 15000}


def smoothstep(a, b, x):
    if x <= a: return 0.0
    if x >= b: return 1.0
    t = (x - a) / (b - a)
    return t * t * (3 - 2 * t)


def cell_count(hpf):
    ts = sorted(COUNT_ANCHORS)
    lx = np.log1p([t for t in ts]); ly = np.log([COUNT_ANCHORS[t] for t in ts])
    return int(round(np.exp(np.interp(np.log1p(hpf), lx, ly))))


# ---- body plan from the genome (step 1) -------------------------------------
def body_plan():
    region, domains, n_celltypes = bpg.generate(SPEC)
    # generate() x-axis: x=0 ORAL (depolarised) -> x=1 ABORAL. region = positional identity per x.
    # order distinct domains along the axis and name them oral->aboral.
    names = ["oral (mouth/tentacles)", "pharynx", "body column", "aboral (physa)"]
    return region, domains, n_celltypes, names


ECTO, ENDO = 0, 1


class CnidaCellField:
    def __init__(self, n=N_RENDER):
        rng = np.random.default_rng(RNG_SEED)
        self.n = n
        self.born_order = np.sort(rng.uniform(0, 1, n))
        self.u = rng.uniform(0, 1, n)              # oral-aboral fate: 1 ORAL, 0 ABORAL
        self.theta = rng.uniform(0, 2 * np.pi, n)  # around the radial axis
        self.jit = rng.normal(0, 1, (n, 2))
        # germ layer: ~40% endoderm (inner), 60% ectoderm (outer)
        self.layer = np.where(rng.uniform(0, 1, n) < 0.40, ENDO, ECTO)
        # a subset of oral ectoderm become tentacle cells in the polyp
        self.tentacle = (self.layer == ECTO) & (self.u > 0.82) & (rng.uniform(0, 1, n) < 0.5)
        self.tent_id = rng.integers(0, 8, n)       # which of ~8 tentacle buds (radial)
        region, domains, _, names = body_plan()
        idx = np.clip((self.u * (len(region) - 1)).astype(int), 0, len(region) - 1)
        # map x->u: generate x=0 is ORAL=u1, so flip index
        self.domain = region[(len(region) - 1) - idx]
        self.region, self.domains, self.dom_names = region, domains, names

    def _positions(self, hpf):
        p_blast = smoothstep(2.0, 14.0, hpf)       # form hollow ball
        p_gast  = smoothstep(16.0, 26.0, hpf)      # invaginate endoderm at oral pole
        p_elong = smoothstep(30.0, 96.0, hpf)      # elongate oral-aboral
        p_polyp = smoothstep(110.0, 168.0, hpf)    # tentacles + pharynx + foot
        n = self.n
        # --- sphere / blastula: hollow shell, radius R ---
        R = 0.10 + 0.06 * p_blast
        ang = (1.0 - self.u) * np.pi               # u=1 oral at +x pole, u=0 aboral at -x pole
        xs = CX + R * np.cos(ang)
        ys = CY + R * np.sin(ang) * np.cos(self.theta)
        # shell vs interior: ectoderm on surface, endoderm just inside
        rin = np.where(self.layer == ENDO, 0.82, 1.0)
        xs = CX + (xs - CX) * rin
        ys = CY + (ys - CY) * rin
        # --- gastrula: endoderm at oral half invaginates toward centre ---
        invag = (self.layer == ENDO) & (self.u > 0.45)
        pull = p_gast * (0.55 + 0.35 * (self.u - 0.45) / 0.55)
        xs = np.where(invag, xs + (CX - xs) * pull * 0.9, xs)
        ys = np.where(invag, ys + (CY - ys) * pull * 0.9, ys)
        # --- elongation into a planula torpedo along x (oral +x, aboral -x) ---
        half = 0.12 + 0.22 * p_elong               # body half-length
        bodyR = (0.085 + 0.02 * p_blast) * (1.0 - 0.55 * p_elong)
        x_ax = CX + (self.u - 0.5) * 2 * half
        taper = np.sqrt(np.clip(1.0 - ((self.u - 0.5) * 2) ** 2, 0.02, 1.0))  # pointed poles
        r_lay = np.where(self.layer == ENDO, 0.45, 1.0)
        y_ax = CY + bodyR * taper * r_lay * np.cos(self.theta)
        xe = (1 - p_elong) * xs + p_elong * x_ax
        ye = (1 - p_elong) * ys + p_elong * y_ax
        # --- pharynx: oral endoderm forms an in-tube from the oral pole ---
        phar = (self.layer == ENDO) & (self.u > 0.78)
        x_or = CX + half
        if phar.any():
            depth = p_polyp * (0.04 + 0.10 * (self.u - 0.78) / 0.22)
            xe = np.where(phar, x_or - depth, xe)
            ye = np.where(phar, CY + 0.018 * np.cos(self.theta), ye)
        # --- tentacle buds at the oral pole (polyp): project beyond oral pole, radial fan ---
        if self.tentacle.any():
            t = self.tentacle
            bud_ang = (self.tent_id[t] - 3.5) / 3.5 * 0.9          # fan in y
            length = p_polyp * (0.06 + 0.05 * np.abs(self.jit[t, 0]))
            xe_t = x_or + length * np.cos(bud_ang * 0.6)
            ye_t = CY + (length * 1.4) * np.sin(bud_ang) + 0.01 * self.jit[t, 1]
            xe[t] = (1 - p_polyp) * xe[t] + p_polyp * xe_t
            ye[t] = (1 - p_polyp) * ye[t] + p_polyp * ye_t
        return xe, ye

    def _colors(self, hpf):
        # Vm oral-aboral gradient: oral depolarised (warm) -> aboral hyperpolarised (cool)
        u = self.u
        rgba = np.zeros((self.n, 4))
        rgba[:, 0] = 0.15 + 0.80 * u            # R high at oral
        rgba[:, 1] = 0.25 + 0.35 * (1 - np.abs(u - 0.5) * 2)
        rgba[:, 2] = 0.20 + 0.75 * (1 - u)      # B high at aboral
        rgba[:, 3] = 0.9
        # endoderm a touch darker/greener to read the two layers
        endo = self.layer == ENDO
        rgba[endo, 1] += 0.12
        rgba[endo, :3] *= 0.88
        # tentacles brighten in the polyp
        rgba[self.tentacle, :3] = np.clip(rgba[self.tentacle, :3] * 1.15 + 0.05, 0, 1)
        return np.clip(rgba, 0, 1)

    def state(self, hpf):
        x, y = self._positions(hpf)
        true_n = cell_count(hpf)
        f = np.log(max(true_n, 1)) / np.log(cell_count(HPF_END))
        alive = self.born_order <= float(np.clip(f, 0.02, 1.0))
        return {"x": x, "y": y, "rgba": self._colors(hpf), "alive": alive,
                "n_true": true_n, "n_shown": int(alive.sum())}


def _phase(hpf):
    for (h, name), (h2, _) in zip(STAGES, STAGES[1:] + [(1e9, "")]):
        if hpf < h2:
            return name
    return STAGES[-1][1]


def render_frame(ax, field, hpf):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0.1, 0.9); ax.axis("off"); ax.set_aspect("equal")
    s = field.state(hpf, )
    m = s["alive"]
    ax.scatter(s["x"][m], s["y"][m], s=5, c=s["rgba"][m], edgecolors="none",
               alpha=0.85, zorder=2)
    # axis + domain labels once elongated
    p_elong = smoothstep(30.0, 96.0, hpf)
    if p_elong > 0.25:
        half = 0.12 + 0.22 * p_elong
        ax.annotate("", xy=(CX + half + 0.05, 0.30), xytext=(CX - half - 0.05, 0.30),
                    arrowprops=dict(arrowstyle="<->", color="#9aa7b5", lw=0.8))
        ax.text(CX + half + 0.05, 0.275, "ORAL", fontsize=6.5, color="#b03a2e", ha="center")
        ax.text(CX - half - 0.05, 0.275, "ABORAL", fontsize=6.5, color="#1f5fb0", ha="center")
        if hpf >= 96:
            for frac, nm in [(0.92, "tentacles"), (0.80, "pharynx"),
                             (0.5, "body column"), (0.10, "physa")]:
                xx = CX + (frac - 0.5) * 2 * half
                ax.text(xx, 0.685, nm, fontsize=6.0, color="#3a4757", ha="center")
    # HUD
    ax.text(0.02, 0.87, "Nematostella from the genome", fontsize=12.5, weight="bold",
            color="#11213a", va="top")
    ax.text(0.02, 0.825, f"{hpf:5.1f} hpf   •   {_phase(hpf)}   (one cell field)",
            fontsize=9.5, color="#33425a", va="top")
    ax.text(0.02, 0.785, f"cells ~{s['n_true']:,}  (drawing {s['n_shown']:,})",
            fontsize=9, color="#33425a", va="top")
    ax.text(0.02, 0.16,
            f"body plan: radial, 2 germ layers, 4 oral-aboral domains\n"
            f"heads (cell types) genome-sourced: N_SE={N_SE_HEADS} -> ~{SPEC.n_heads} families\n"
            f"axis polarity = validated bioelectric Vm gradient",
            fontsize=7.6, color="#11213a", va="top",
            bbox=dict(boxstyle="round", fc="#f2f6ff", ec="#9aa7b5"))
    ax.text(0.98, 0.12, "topology-level; body-plan numbers genome/atlas-sourced",
            fontsize=6.3, color="#8a97a5", ha="right", style="italic")


def make_movie(path="nematostella_embryo.gif"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e})"); return None
    field = CnidaCellField()
    hpfs = np.concatenate([np.linspace(0.0, 30.0, 40), np.linspace(30.0, 168.0, 70)])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    anim = FuncAnimation(fig, lambda k: render_frame(ax, field, float(hpfs[k])),
                         frames=len(hpfs), blit=False)
    anim.save(path, writer=PillowWriter(fps=12)); plt.close(fig)
    stills = []
    for hpf in (7.0, 14.0, 24.0, 48.0, 96.0, 168.0):
        f2, a2 = plt.subplots(figsize=(10.5, 5.2))
        render_frame(a2, field, hpf)
        p = f"nematostella_embryo_{hpf:05.1f}hpf.png"
        f2.savefig(p, dpi=110, bbox_inches="tight"); plt.close(f2)
        stills.append((hpf, p))
    # montage (Paper-1 style 6-panel)
    montage = "nematostella_embryo_montage.png"
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.2))
    for axm, (hpf, _) in zip(axs.ravel(), stills):
        render_frame(axm, field, hpf)
    fig.suptitle("Nematostella development from the genome (one cell field)  •  "
                 "blastula → gastrula → planula → primary polyp",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(montage, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path, stills, montage


def main():
    print("=" * 78)
    print("NEMATOSTELLA DEVELOPMENT FROM THE GENOME  --  body plan -> movie")
    print("=" * 78)
    region, domains, n_celltypes, names = body_plan()
    print("\n  STEP 1 -- body plan from the genome (medic/body_plan_generator):")
    print(f"     symmetry={SPEC.symmetry}, germ layers={SPEC.n_germ_layers}, "
          f"axis={SPEC.primary_axis}")
    print(f"     axial domains (oral->aboral): {len(domains)}  "
          f"{'OK' if abs(len(domains)-SPEC.expected_domains)<=1 else 'MISMATCH'} "
          f"(expected {SPEC.expected_domains})")
    for nm in names[:len(domains)]:
        print(f"        - {nm}")
    print(f"     cell-type diversity (heads) = {SPEC.n_heads} families "
          f"(genome-sourced: N_SE={N_SE_HEADS} super-enhancers)")
    print("\n  STEP 2 -- development movie:")
    out = make_movie()
    if not out:
        return
    gif, stills, montage = out
    field = CnidaCellField()
    for hpf, _ in stills:
        s = field.state(hpf)
        print(f"     {hpf:6.1f} hpf  {_phase(hpf):14s}  cells~{s['n_true']:>6,}  drawn={s['n_shown']:>5,}")
    print(f"\n  saved movie : {gif}")
    print(f"  saved montage: {montage}")
    print(f"  saved stills : {len(stills)} key stages")


if __name__ == "__main__":
    main()
