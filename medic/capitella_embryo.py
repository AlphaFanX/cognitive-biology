#!/usr/bin/env python3
"""
Capitella teleta development from the genome: body plan -> movie.
=================================================================

The bilateral, segmented counterpart of medic/nematostella_embryo.py, for the annelid
whose kernel was tested most directly: real matched developmental ATAC + EM-seq + RNA
(medic/annelid_concordance.py reader concordance; medic/annelid_kernel_porting.py T2
preconditions PASS -- the program is low-rank and its accessibility use-order leads
transcription). So Capitella is the most genome-grounded of the development films: its
5 developmental MASKS are literally the 5 sampled stages of that data.

Two steps:
  1. BODY PLAN FROM THE GENOME (medic/body_plan_generator): bilateral, 3 germ layers,
     anterior-posterior axis, 5 axial domains (prostomium / peristomium / anterior trunk /
     posterior trunk / pygidium), oriented by the validated bioelectric Vm polarity
     (depolarized ANTERIOR/head -> hyperpolarized POSTERIOR/tail).
  2. MOVIE: one cell field carried zygote -> spiral cleavage -> blastula -> gastrula ->
     trochophore larva (episphere + prototroch ciliary band + hyposphere) ->
     metatrochophore -> segmented juvenile worm (segments added from a posterior growth
     zone). Cells coloured by the AP Vm gradient; the segments emerge as discrete blocks;
     a HUD reports stage, cell count, and the body-plan provenance.

Honest scope: topology-level (germ layers x AP domains x segmentation x bilateral symmetry x
emergent cell number); body-plan NUMBERS are genome/atlas-sourced, morphogenesis is a
constructive reconstruction.

Run:  cd cognimed && python -m medic.capitella_embryo
"""
from __future__ import annotations
import numpy as np

try:
    from . import body_plan_generator as bpg
except ImportError:  # pragma: no cover
    from medic import body_plan_generator as bpg

N_RENDER = 4000
RNG_SEED = 7
CX, CY = 0.50, 0.50
MAX_SEG = 9                      # major trunk segments by the juvenile stage
SPEC = next(s for s in bpg.CLADES if s.name == "capitella")

STAGES = [(0.0, "zygote"), (5.0, "spiral cleavage"), (10.0, "blastula"), (16.0, "gastrula"),
          (24.0, "trochophore (st4)"), (40.0, "metatrochophore (st5)"),
          (72.0, "segmented juvenile (st7)")]
HPF_END = 72.0
COUNT_ANCHORS = {0.0: 1, 5.0: 32, 10.0: 300, 16.0: 800, 24.0: 2000, 40.0: 5000, 72.0: 12000}


def smoothstep(a, b, x):
    if x <= a: return 0.0
    if x >= b: return 1.0
    t = (x - a) / (b - a)
    return t * t * (3 - 2 * t)


def cell_count(hpf):
    ts = sorted(COUNT_ANCHORS)
    lx = np.log1p(ts); ly = np.log([COUNT_ANCHORS[t] for t in ts])
    return int(round(np.exp(np.interp(np.log1p(hpf), lx, ly))))


def n_segments(hpf):
    return int(round(MAX_SEG * smoothstep(34.0, 72.0, hpf)))


def body_plan():
    region, domains, n_celltypes = bpg.generate(SPEC)
    names = ["prostomium (head)", "peristomium", "anterior trunk", "posterior trunk", "pygidium (tail)"]
    return region, domains, n_celltypes, names


ECTO, ENDO, MESO = 0, 1, 2


class CapitellaCellField:
    def __init__(self, n=N_RENDER):
        rng = np.random.default_rng(RNG_SEED)
        self.n = n
        self.born_order = np.sort(rng.uniform(0, 1, n))
        self.u = rng.uniform(0, 1, n)              # AP fate: 1 ANTERIOR/head, 0 POSTERIOR/tail
        self.theta = rng.uniform(0, 2 * np.pi, n)  # around the body
        self.side = rng.choice([-1.0, 1.0], n)     # bilateral left/right
        self.jit = rng.normal(0, 1, (n, 2))
        # germ layers: ecto (epidermis/neural, outer), endo (gut, inner), meso (muscle, between)
        r = rng.uniform(0, 1, n)
        self.layer = np.where(r < 0.50, ECTO, np.where(r < 0.78, ENDO, MESO))
        # prototroch (ciliary band) = an equatorial ring of ectoderm cells (u ~ 0.5)
        self.proto = (self.layer == ECTO) & (np.abs(self.u - 0.5) < 0.06)
        # segment index along the trunk (anterior trunk = low index = older)
        self.seg = np.clip(((1 - self.u) * MAX_SEG).astype(int), 0, MAX_SEG - 1)
        region, domains, _, names = body_plan()
        idx = np.clip((self.u * (len(region) - 1)).astype(int), 0, len(region) - 1)
        self.domain = region[(len(region) - 1) - idx]   # x=0 is anterior in generate
        self.region, self.domains, self.dom_names = region, domains, names

    def _positions(self, hpf):
        p_blast = smoothstep(2.0, 10.0, hpf)
        p_gast  = smoothstep(12.0, 18.0, hpf)
        p_elong = smoothstep(28.0, 72.0, hpf)
        nseg = n_segments(hpf)
        # --- sphere / blastula (hollow ball) ---
        R = 0.085 + 0.05 * p_blast
        ang = (1.0 - self.u) * np.pi
        xs = CX + R * np.cos(ang)
        ys = CY + R * np.sin(ang) * np.cos(self.theta)
        rin = np.where(self.layer == ENDO, 0.78, np.where(self.layer == MESO, 0.9, 1.0))
        xs = CX + (xs - CX) * rin; ys = CY + (ys - CY) * rin
        # --- gastrula: posterior endoderm invaginates (annelid gastrulation at vegetal/posterior) ---
        invag = (self.layer == ENDO) & (self.u < 0.55)
        pull = p_gast * (0.5 + 0.4 * (0.55 - self.u) / 0.55)
        xs = np.where(invag, xs + (CX - xs) * pull * 0.9, xs)
        ys = np.where(invag, ys + (CY - ys) * pull * 0.9, ys)
        # --- elongation into a worm along AP (anterior/head LEFT, posterior/tail RIGHT) ---
        half = 0.11 + 0.26 * p_elong
        bodyR = (0.075 + 0.02 * p_blast) * (1.0 - 0.5 * p_elong)
        x_ax = CX + (0.5 - self.u) * 2 * half          # u=1 head -> left
        taper = np.sqrt(np.clip(1.0 - ((self.u - 0.5) * 2) ** 2 * 0.8, 0.05, 1.0))
        r_lay = np.where(self.layer == ENDO, 0.4, np.where(self.layer == MESO, 0.72, 1.0))
        y_ax = CY + bodyR * taper * r_lay * np.cos(self.theta)
        xe = (1 - p_elong) * xs + p_elong * x_ax
        ye = (1 - p_elong) * ys + p_elong * y_ax
        # --- segments: trunk ecto/meso cells snap into discrete bands once formed ---
        seg_w = (2 * half) / (MAX_SEG + 2)
        trunk = (self.layer != ENDO) & (self.u < 0.78) & (self.u > 0.08)
        formed = trunk & (self.seg < max(nseg, 0))
        if formed.any():
            # band centre by segment index, measured back from just behind the head
            sx = (CX - half) + (1.5 + self.seg) * seg_w
            local = (self.theta / np.pi - 1.0)              # -1..1 around body -> within-band spread
            xe[formed] = sx[formed] + local[formed] * seg_w * 0.34
            ye[formed] = CY + (bodyR * 1.1) * np.cos(self.theta[formed]) + 0.012 * np.abs(local[formed])
        return xe, ye, p_elong

    def _colors(self, hpf):
        u = self.u
        rgba = np.zeros((self.n, 4))
        rgba[:, 0] = 0.15 + 0.80 * u            # warm at anterior/head
        rgba[:, 1] = 0.25 + 0.30 * (1 - np.abs(u - 0.5) * 2)
        rgba[:, 2] = 0.20 + 0.75 * (1 - u)      # cool at posterior/tail
        rgba[:, 3] = 0.9
        endo = self.layer == ENDO; meso = self.layer == MESO
        rgba[endo, 1] += 0.12; rgba[endo, :3] *= 0.86
        rgba[meso, 0] += 0.06                    # muscle slightly redder
        # prototroch ciliary band: bright when it is the trochophore
        if smoothstep(18.0, 28.0, hpf) > 0.2 and smoothstep(34.0, 60.0, hpf) < 0.8:
            rgba[self.proto, :3] = np.array([1.0, 0.95, 0.55])
        return np.clip(rgba, 0, 1)

    def state(self, hpf):
        x, y, p_elong = self._positions(hpf)
        true_n = cell_count(hpf)
        f = np.log(max(true_n, 1)) / np.log(cell_count(HPF_END))
        alive = self.born_order <= float(np.clip(f, 0.02, 1.0))
        return {"x": x, "y": y, "rgba": self._colors(hpf), "alive": alive,
                "n_true": true_n, "n_shown": int(alive.sum()), "nseg": n_segments(hpf),
                "p_elong": p_elong}


def _phase(hpf):
    for (h, name), (h2, _) in zip(STAGES, STAGES[1:] + [(1e9, "")]):
        if hpf < h2:
            return name
    return STAGES[-1][1]


def render_frame(ax, field, hpf):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0.12, 0.88); ax.axis("off"); ax.set_aspect("equal")
    s = field.state(hpf)
    m = s["alive"]
    ax.scatter(s["x"][m], s["y"][m], s=5, c=s["rgba"][m], edgecolors="none",
               alpha=0.85, zorder=2)
    if s["p_elong"] > 0.25:
        half = 0.11 + 0.26 * s["p_elong"]
        ax.annotate("", xy=(CX - half - 0.05, 0.30), xytext=(CX + half + 0.05, 0.30),
                    arrowprops=dict(arrowstyle="<->", color="#9aa7b5", lw=0.8))
        ax.text(CX - half - 0.05, 0.275, "ANTERIOR (head)", fontsize=6.3, color="#b03a2e", ha="center")
        ax.text(CX + half + 0.05, 0.275, "POSTERIOR (tail)", fontsize=6.3, color="#1f5fb0", ha="center")
        if hpf >= 45:
            for frac, nm in [(0.95, "prostomium"), (0.78, "peristomium"),
                             (0.5, "trunk segments"), (0.06, "pygidium")]:
                xx = CX + (0.5 - frac) * 2 * half
                ax.text(xx, 0.70, nm, fontsize=6.0, color="#3a4757", ha="center")
            ax.text(CX, 0.665, f"({s['nseg']} segments)", fontsize=5.8, color="#7a8696", ha="center")
    ax.text(0.02, 0.85, "Capitella from the genome", fontsize=12.5, weight="bold",
            color="#11213a", va="top")
    ax.text(0.02, 0.808, f"{hpf:5.1f} hpf   •   {_phase(hpf)}   (one cell field)",
            fontsize=9.5, color="#33425a", va="top")
    ax.text(0.02, 0.77, f"cells ~{s['n_true']:,}  (drawing {s['n_shown']:,})",
            fontsize=9, color="#33425a", va="top")
    ax.text(0.02, 0.18,
            "body plan: bilateral, 3 germ layers, 5 AP domains, segmented\n"
            "masks = 5 sampled dev stages (real ATAC+RNA); heads atlas-sourced\n"
            "axis polarity = validated bioelectric Vm gradient",
            fontsize=7.4, color="#11213a", va="top",
            bbox=dict(boxstyle="round", fc="#f2f6ff", ec="#9aa7b5"))
    ax.text(0.98, 0.14, "topology-level; body-plan numbers genome/atlas-sourced",
            fontsize=6.2, color="#8a97a5", ha="right", style="italic")


def make_movie(path="capitella_embryo.gif"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e})"); return None
    field = CapitellaCellField()
    hpfs = np.concatenate([np.linspace(0.0, 28.0, 40), np.linspace(28.0, 72.0, 70)])
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    anim = FuncAnimation(fig, lambda k: render_frame(ax, field, float(hpfs[k])),
                         frames=len(hpfs), blit=False)
    anim.save(path, writer=PillowWriter(fps=12)); plt.close(fig)
    stills = []
    for hpf in (5.0, 16.0, 24.0, 40.0, 56.0, 72.0):
        f2, a2 = plt.subplots(figsize=(10.5, 5.2))
        render_frame(a2, field, hpf)
        p = f"capitella_embryo_{hpf:05.1f}hpf.png"
        f2.savefig(p, dpi=110, bbox_inches="tight"); plt.close(f2)
        stills.append((hpf, p))
    montage = "capitella_embryo_montage.png"
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.2))
    for axm, (hpf, _) in zip(axs.ravel(), stills):
        render_frame(axm, field, hpf)
    fig.suptitle("Capitella development from the genome (one cell field)  •  "
                 "blastula → gastrula → trochophore → segmented juvenile worm",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(montage, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path, stills, montage


def main():
    print("=" * 78)
    print("CAPITELLA DEVELOPMENT FROM THE GENOME  --  body plan -> movie")
    print("=" * 78)
    region, domains, n_celltypes, names = body_plan()
    print("\n  STEP 1 -- body plan from the genome (medic/body_plan_generator):")
    print(f"     symmetry={SPEC.symmetry}, germ layers={SPEC.n_germ_layers}, axis={SPEC.primary_axis}")
    print(f"     axial domains (anterior->posterior): {len(domains)}  "
          f"{'OK' if abs(len(domains)-SPEC.expected_domains)<=1 else 'MISMATCH'} "
          f"(expected {SPEC.expected_domains})")
    for nm in names[:len(domains)]:
        print(f"        - {nm}")
    print(f"     cell-type diversity (heads) = {SPEC.n_heads}; masks = {SPEC.n_masks} "
          f"(= the 5 sampled developmental stages with real ATAC+RNA)")
    print("\n  STEP 2 -- development movie:")
    out = make_movie()
    if not out:
        return
    gif, stills, montage = out
    field = CapitellaCellField()
    for hpf, _ in stills:
        s = field.state(hpf)
        print(f"     {hpf:6.1f} hpf  {_phase(hpf):26s}  cells~{s['n_true']:>6,}  "
              f"drawn={s['n_shown']:>5,}  segments={s['nseg']}")
    print(f"\n  saved movie : {gif}")
    print(f"  saved montage: {montage}")


if __name__ == "__main__":
    main()
