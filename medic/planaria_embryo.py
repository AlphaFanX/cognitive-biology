#!/usr/bin/env python3
"""
Planaria body plan from the BIOELECTRIC kernel: regeneration movie.
===================================================================

The honest counterpart to the Nematostella and Capitella films. Those two derive the body
plan from the GENOMIC kernel (methylation/accessibility reader). Planaria is the exception that
proves the kernel/reader story: flatworms LOST CpG methylation (no DNMT3), so there is no
methylation kernel to read -- and yet planaria patterns its whole AP axis reliably. The prior
lives in the BIOELECTRIC field instead (medic/planaria_bioelectric.py: the same Goldman/GJ Vm
operator predicts head / tail / two-headed regeneration 8/8). So planaria's "kernel" is a
bioelectric kernel, and the natural film is REGENERATION, the process that bioelectricity drives.

Two steps:
  1. BODY PLAN (medic/body_plan_generator): bilateral, 3 germ layers, anterior-posterior axis,
     4 domains (head / pre-pharyngeal / pharynx-trunk / tail). Heads (cell types) are atlas-sourced
     (Fincher/Plass 2018 accessibility, NOT methylation); the axis is set by the Vm polarity
     (depolarized HEAD -> hyperpolarized TAIL -- the validated floor).
  2. MOVIE: a trunk FRAGMENT (no head, no tail) regenerates. Blastemas grow at both wounds; the
     anterior-facing wound depolarizes and regrows the head (with eyespots + brain), the posterior
     wound hyperpolarizes and regrows the tail, and the pharynx reforms mid-body. Cells coloured by
     the AP Vm gradient; new blastema tissue starts pale and fills in. A HUD notes that no genomic
     kernel is used -- the body plan is restored from the bioelectric field.

Honest scope: topology-level morphogenesis; the body-plan numbers are atlas-sourced and the
polarity is the validated bioelectric operator; the regeneration trajectory is a constructive
reconstruction.

Run:  cd cognimed && python -m medic.planaria_embryo
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
SPEC = next(s for s in bpg.CLADES if s.name == "planaria")

# regeneration timeline (days post amputation)
STAGES = [(0.0, "trunk fragment"), (0.5, "wound healing"), (1.0, "blastema"),
          (2.0, "blastema growth"), (3.0, "head/tail repatterning"),
          (5.0, "eyespots + pharynx"), (7.0, "regenerated worm")]
DPA_END = 7.0


def smoothstep(a, b, x):
    if x <= a: return 0.0
    if x >= b: return 1.0
    t = (x - a) / (b - a)
    return t * t * (3 - 2 * t)


def body_plan():
    region, domains, n_celltypes = bpg.generate(SPEC)
    names = ["head (cephalic ganglia/eyes)", "pre-pharyngeal", "pharynx / trunk", "tail"]
    return region, domains, n_celltypes, names


class PlanariaCellField:
    """Flat worm, DORSAL view: x = AP (head LEFT), y = mediolateral width. u=1 head, u=0 tail."""
    def __init__(self, n=N_RENDER):
        rng = np.random.default_rng(RNG_SEED)
        self.n = n
        self.u = rng.uniform(0, 1, n)              # final AP fate
        self.ml = rng.uniform(-1, 1, n)            # mediolateral position (-1..1)
        self.born = rng.uniform(0, 1, n)
        self.jit = rng.normal(0, 1, (n, 2))
        # eyespot + pharynx marker cells
        self.eye = (self.u > 0.86) & (np.abs(self.ml) > 0.45) & (np.abs(self.ml) < 0.8) \
                   & (rng.uniform(0, 1, n) < 0.5)
        self.phar = (np.abs(self.u - 0.45) < 0.05) & (np.abs(self.ml) < 0.35)
        region, domains, _, names = body_plan()
        idx = np.clip((self.u * (len(region) - 1)).astype(int), 0, len(region) - 1)
        self.domain = region[(len(region) - 1) - idx]
        self.region, self.domains, self.dom_names = region, domains, names

    def _width(self, u):
        # leaf shape: widest pre-pharynx, tapering to a rounded head and a pointed tail
        uc = np.clip(u, 0, 1)
        ss = uc * uc * (3 - 2 * uc)                     # vectorized smoothstep(0,1,u)
        return 0.085 * (0.55 + 0.9 * np.sin(uc * np.pi) ** 0.7) * (0.7 + 0.3 * ss)

    def visible_range(self, dpa):
        # fragment spans u in [0.30,0.70]; blastemas extend it to [0,1] by day ~3
        g = smoothstep(0.5, 3.0, dpa)
        lo = 0.30 * (1 - g)
        hi = 0.70 + 0.30 * g
        return lo, hi

    def state(self, dpa):
        lo, hi = self.visible_range(dpa)
        alive = (self.u >= lo) & (self.u <= hi)
        elong = 0.20 + 0.16 * smoothstep(0.0, 3.0, dpa)     # half-length grows as ends regrow
        x = CX + (0.5 - self.u) * 2 * elong                  # head (u=1) -> left
        y = CY + self._width(self.u) * self.ml
        # blastema = newly-regrown tissue near the frontiers: pale until repatterned
        new_ant = (self.u > 0.70) & (self.u <= hi)
        new_post = (self.u < 0.30) & (self.u >= lo)
        repat = smoothstep(2.0, 5.0, dpa)
        # colours: AP Vm gradient (head warm/depol -> tail cool/hyperpol)
        u = self.u
        rgba = np.zeros((self.n, 4))
        rgba[:, 0] = 0.15 + 0.80 * u
        rgba[:, 1] = 0.30 + 0.25 * (1 - np.abs(u - 0.5) * 2)
        rgba[:, 2] = 0.20 + 0.75 * (1 - u)
        rgba[:, 3] = 0.9
        # blastema pale (unpigmented) then fills in with repatterning
        pale = (new_ant | new_post)
        for ch in range(3):
            rgba[pale, ch] = (1 - repat) * 0.85 + repat * rgba[pale, ch]
        # eyespots (dark) once the head is repatterned
        if repat > 0.4:
            rgba[self.eye] = np.array([0.05, 0.05, 0.08, 1.0])
        # pharynx (pale tube) once reformed
        if smoothstep(4.0, 6.0, dpa) > 0.3:
            rgba[self.phar] = np.array([0.95, 0.9, 0.8, 1.0])
        return {"x": x, "y": y, "rgba": np.clip(rgba, 0, 1), "alive": alive,
                "elong": elong, "repat": repat, "lo": lo, "hi": hi}


def _phase(dpa):
    for (h, name), (h2, _) in zip(STAGES, STAGES[1:] + [(1e9, "")]):
        if dpa < h2:
            return name
    return STAGES[-1][1]


def render_frame(ax, field, dpa):
    ax.clear()
    ax.set_xlim(0, 1); ax.set_ylim(0.2, 0.8); ax.axis("off"); ax.set_aspect("equal")
    s = field.state(dpa)
    m = s["alive"]
    ax.scatter(s["x"][m], s["y"][m], s=6, c=s["rgba"][m], edgecolors="none", alpha=0.9, zorder=2)
    elong = s["elong"]
    ax.annotate("", xy=(CX - elong - 0.05, 0.34), xytext=(CX + elong + 0.05, 0.34),
                arrowprops=dict(arrowstyle="<->", color="#9aa7b5", lw=0.8))
    ax.text(CX - elong - 0.05, 0.315, "HEAD", fontsize=6.5, color="#b03a2e", ha="center")
    ax.text(CX + elong + 0.05, 0.315, "TAIL", fontsize=6.5, color="#1f5fb0", ha="center")
    if s["repat"] > 0.45:
        for frac, nm in [(0.92, "eyes/brain"), (0.45, "pharynx"), (0.06, "tail tip")]:
            ax.text(CX + (0.5 - frac) * 2 * elong, 0.66, nm, fontsize=6.0, color="#3a4757", ha="center")
    ax.text(0.02, 0.78, "Planaria from the bioelectric kernel", fontsize=12.5, weight="bold",
            color="#11213a", va="top")
    ax.text(0.02, 0.745, f"{dpa:4.1f} days post-amputation   •   {_phase(dpa)}",
            fontsize=9.5, color="#33425a", va="top")
    ax.text(0.02, 0.245,
            "no genomic (methylation) kernel -- flatworms lost CpG 5mC\n"
            "body plan restored from the bioelectric field (Vm polarity)\n"
            "head/tail/two-headed Vm operator validated 8/8 (planaria_bioelectric.py)",
            fontsize=7.3, color="#11213a", va="top",
            bbox=dict(boxstyle="round", fc="#fff3f0", ec="#d9a79a"))
    ax.text(0.98, 0.22, "topology-level; polarity = validated bioelectric operator",
            fontsize=6.2, color="#8a97a5", ha="right", style="italic")


def make_movie(path="planaria_embryo.gif"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable: {e})"); return None
    field = PlanariaCellField()
    dpas = np.linspace(0.0, 7.0, 100)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    anim = FuncAnimation(fig, lambda k: render_frame(ax, field, float(dpas[k])),
                         frames=len(dpas), blit=False)
    anim.save(path, writer=PillowWriter(fps=12)); plt.close(fig)
    stills = []
    for dpa in (0.0, 1.0, 2.0, 3.0, 5.0, 7.0):
        f2, a2 = plt.subplots(figsize=(10.5, 4.8))
        render_frame(a2, field, dpa)
        p = f"planaria_embryo_{dpa:04.1f}dpa.png"
        f2.savefig(p, dpi=110, bbox_inches="tight"); plt.close(f2)
        stills.append((dpa, p))
    montage = "planaria_embryo_montage.png"
    fig, axs = plt.subplots(2, 3, figsize=(15, 6.6))
    for axm, (dpa, _) in zip(axs.ravel(), stills):
        render_frame(axm, field, dpa)
    fig.suptitle("Planaria body plan from the bioelectric kernel (regeneration)  •  "
                 "fragment → blastema → repatterned head + tail",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(montage, dpi=120, bbox_inches="tight"); plt.close(fig)
    return path, stills, montage


def main():
    print("=" * 78)
    print("PLANARIA BODY PLAN FROM THE BIOELECTRIC KERNEL  --  regeneration movie")
    print("=" * 78)
    region, domains, n_celltypes, names = body_plan()
    print("\n  STEP 1 -- body plan (medic/body_plan_generator):")
    print(f"     symmetry={SPEC.symmetry}, germ layers={SPEC.n_germ_layers}, axis={SPEC.primary_axis}")
    print(f"     axial domains (head->tail): {len(domains)}  "
          f"{'OK' if abs(len(domains)-SPEC.expected_domains)<=1 else 'MISMATCH'} "
          f"(expected {SPEC.expected_domains})")
    for nm in names[:len(domains)]:
        print(f"        - {nm}")
    print(f"     cell-type diversity (heads) = {SPEC.n_heads} (atlas: Fincher/Plass 2018, ACCESSIBILITY)")
    print("     NOTE: no methylation kernel (flatworms lost CpG 5mC); prior is BIOELECTRIC.")
    print("\n  STEP 2 -- regeneration movie:")
    out = make_movie()
    if not out:
        return
    gif, stills, montage = out
    field = PlanariaCellField()
    for dpa, _ in stills:
        s = field.state(dpa)
        vis = int(s["alive"].sum())
        print(f"     {dpa:4.1f} dpa  {_phase(dpa):22s}  AP extent u=[{s['lo']:.2f},{s['hi']:.2f}]  "
              f"cells drawn={vis:>4,}")
    print(f"\n  saved movie : {gif}")
    print(f"  saved montage: {montage}")


if __name__ == "__main__":
    main()
