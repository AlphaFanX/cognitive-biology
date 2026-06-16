#!/usr/bin/env python3
"""
Genomic Neural Cellular Automata: the NCA kernel made literal.

This module wires the abstract NCA (medic/tissue/nca.py) to the genomic
substrate so the conceptual correspondence becomes executable code:

TWO-MLP / LoRA STRUCTURE (corrected 2026-06-04, after re-reading Jadhav 2019):

    Zygote MLP (frozen base, W0)
        == the inherited fossil record == the UNDIFFERENTIATED zygote target
           (V_ZYGOTE). What every cell starts from. Read-out from Jadhav
           hypomethylation only (see medic/genome/real_kernel.py).

    External ABC/SEdb LoRA (DeltaW, the adapter)
        == GenomicChannelLookup (ABC-derived position -> conductances -> Vmem),
           re-expressed as the ADULT DIFFERENTIATION DELTA it adds on top of the
           frozen base:  dV_lora(pos) = compute_voltage(pos) - V_ZYGOTE.
           ABC/SEdb are measured in ADULT/differentiated cells, so they ARE the
           differentiated endpoint == base + full LoRA. Jadhav's H3K4me1/H3K27ac
           are an *activity* layer ("remnants of fetal gene activity"), NOT the
           archive -- so the adult activity signal lives here, on the LoRA, not
           in the frozen base.

    The telomere/PRC2 clock (the mask)
        == sets the LoRA SCALE in [0,1]: full telomere -> only ADULT stratum
           live -> lora_scale = 1 -> fully differentiated (base + full LoRA).
           As the clock runs down, earlier strata reactivate and lora_scale
           falls -> the differentiation delta is withdrawn -> the field relaxes
           back toward the frozen zygote base (de-differentiation).

    NCA perception (identity + Sobel gradients)
        == gap-junction / morphogen gradient sensing (the WHERE axis).

Effective per-cell target = V_ZYGOTE + lora_scale * dV_lora  (= base + scaled
LoRA). Algebraically this equals the previous (1-d)*V_adult + d*V_zygote blend,
so the validated behaviour is preserved EXACTLY -- the change is structural:
ABC is now an external adapter on a frozen fossil base, not the base itself. A
clean ablation falls out: lora_scale = 0 recovers the pure zygote.

Demo semantics (faithful to zygote_kernel.py's own clock):
    Start at full telomere -> PRC2 fully engaged -> only the ADULT stratum is
    live -> the tissue expresses its fully differentiated genomic voltage map.
    As the telomere clock runs down, PRC2 withdraws and the FETAL then EMBRYONIC
    strata reactivate IN REVERSE developmental order. Re-exposing that fossil
    record partially DE-differentiates the cells: the sharp organ voltage map
    relaxes back toward the undifferentiated zygote value. The simulation makes
    "the clock indexes the kernel" visible as a flattening voltage field and a
    rising active-stratum count.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.tissue.genomic_nca
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:  # package-relative (normal use)
    from ..four_head_morphogenesis import GenomicChannelLookup, CHANNEL_NAMES
    from ..genome.zygote_kernel import ZygoteKernel, KernelStratum
    from ..genome.real_kernel import load_real_zygote_kernel
except ImportError:  # pragma: no cover - direct execution fallback
    from medic.four_head_morphogenesis import GenomicChannelLookup, CHANNEL_NAMES
    from medic.genome.zygote_kernel import ZygoteKernel, KernelStratum
    from medic.genome.real_kernel import load_real_zygote_kernel


# Zygote resting potential (medic/developmental_trm.py initialize_zygote).
V_ZYGOTE = -70.0

# Voltage band used to read out a coarse germ-layer identity from Vmem
# (consistent with the Levin mapping in developmental_trm: hyperpolarized ->
# ectoderm/neural, depolarized -> endoderm). Diagnostic only.
GERM_BANDS = [
    ("ectoderm (hyperpol)", -1e9, -55.0),
    ("mesoderm (mid)",       -55.0, -40.0),
    ("endoderm (depol)",     -40.0,  1e9),
]

# How strongly each reactivated earlier stratum de-differentiates a cell
# (pulls its target back toward the zygote value). ADULT alone = no pull;
# FETAL reactivation = partial; EMBRYONIC reactivation = strong.
DEDIFF_WEIGHT = {
    KernelStratum.ADULT: 0.0,
    KernelStratum.FETAL: 0.4,
    KernelStratum.EMBRYONIC: 0.4,  # additive on top of fetal -> 0.8 when both live
}


# ---------------------------------------------------------------------------
# Perception (identity + Sobel gradients) -- numpy mirror of nca.py
# ---------------------------------------------------------------------------
_SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64) / 8.0
_SOBEL_Y = _SOBEL_X.T
_LAPLACIAN = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float64)


def _conv2d_same(field2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """3x3 convolution with edge (no-flux) padding -- a closed tissue boundary."""
    padded = np.pad(field2d, 1, mode="edge")
    H, W = field2d.shape
    out = np.zeros_like(field2d)
    for di in range(3):
        for dj in range(3):
            k = kernel[di, dj]
            if k != 0.0:
                out += k * padded[di:di + H, dj:dj + W]
    return out


@dataclass
class GenomicNCAState:
    """Snapshot of the tissue at one step (for history / a future viewer)."""
    step: int
    telomere_bp: float
    prc2: float
    live_strata: List[str]
    voltage: np.ndarray            # (H, W) mV
    target: np.ndarray             # (H, W) mV (strata-gated genome target)
    germ_fractions: Dict[str, float]
    v_mean: float
    v_std: float
    grad_mag_mean: float           # mean perceived gradient magnitude (boundary sharpness)


class ABCLoRAAdapter:
    """The external ABC/SEdb model, re-expressed as a LoRA adapter on the kernel.

    ABC/SEdb are measured in ADULT/differentiated cells, so they encode the
    differentiated ENDPOINT, not the inherited base. This adapter therefore
    exposes the ABC GenomicChannelLookup not as an absolute target but as the
    DELTA it adds on top of the frozen zygote base:

        delta(pos) = scale * (GenomicChannelLookup.compute_voltage(pos) - V_base)

    `scale` is the LoRA strength (the clock drives it at runtime); `V_base` is
    the frozen-base reference (the zygote potential). This is the seam where a
    trained ABC/SEdb transformer's low-rank LoRA would later plug in -- for now
    the adapter is derived directly from the ABC conductance lookup.
    """

    def __init__(self, seed: int = 42, v_base: float = V_ZYGOTE):
        self.lookup = GenomicChannelLookup(seed=seed)
        self.v_base = float(v_base)

    def adult_voltage(self, pos: np.ndarray) -> float:
        """The fully-differentiated (full-LoRA) ABC target at a position."""
        return float(self.lookup.compute_voltage(pos))

    def delta(self, pos: np.ndarray, scale: float = 1.0) -> float:
        """The differentiation delta this adapter adds over the frozen base."""
        return scale * (self.adult_voltage(pos) - self.v_base)


class GenomicNCA:
    """An NCA = frozen zygote MLP base + external ABC/SEdb LoRA, clock-gated.

    The update rule applied identically to every cell is:

        dV/dt = k_relax * (V_target - V) + k_gj * laplacian(V)
        V_target = V_ZYGOTE + lora_scale * dV_lora        (base + scaled LoRA)

    where V_ZYGOTE is the frozen zygote MLP base (the inherited, undifferentiated
    fossil base), dV_lora is the per-cell ADULT differentiation delta supplied by
    the external ABCLoRAAdapter (ABC/SEdb), and lora_scale in [0,1] is set by the
    telomere/PRC2 clock (1 = fully differentiated adult; 0 = pure zygote). The
    laplacian term is gap-junction coupling; identity + Sobel perceptions are the
    cell's gradient sense (the WHERE axis).
    """

    def __init__(
        self,
        height: int = 24,
        width: int = 24,
        z_slice: float = 0.5,
        k_relax: float = 0.30,
        k_gj: float = 0.12,
        dt: float = 1.0,
        telomere_bp: float = 10000.0,
        seed: int = 42,
        zygote_kernel: Optional[ZygoteKernel] = None,
        use_real_strata: bool = False,
        conserved_base: bool = False,
        lora_rank: Optional[int] = None,
    ):
        self.H, self.W = height, width
        self.dt = dt
        self.k_relax = k_relax
        self.k_gj = k_gj
        self.telomere_bp = float(telomere_bp)
        # conserved_base: place the conserved DV prepattern (the smooth, clade-
        # shared, axis-organizing part of the ABC field) into the FROZEN BASE W0,
        # leaving only the species/organ residual as the LoRA -- the placement
        # correction identified by phylotypic_form.py. Default False preserves the
        # validated uniform-zygote base (and the paper's published numbers).
        self.conserved_base = conserved_base

        # --- the external ABC/SEdb LoRA adapter (adult differentiation delta) ---
        self.lora = ABCLoRAAdapter(seed=seed, v_base=V_ZYGOTE)

        # --- the GATE: one stratified fossil record + the telomere clock ---
        # Prefer a caller-supplied kernel; else optionally the real mark-resolved
        # Jadhav kernel (GEO GSE111024); else the synthetic demo kernel.
        self.strata_source = "demo"
        if zygote_kernel is not None:
            self.zygote_kernel = zygote_kernel
            self.strata_source = "supplied"
        elif use_real_strata:
            real = load_real_zygote_kernel()
            if real is not None:
                self.zygote_kernel = real
                self.strata_source = "real (Jadhav GSE111024)"
            else:
                self.zygote_kernel = ZygoteKernel.demo()
        else:
            self.zygote_kernel = ZygoteKernel.demo()
        self.zygote_kernel.set_from_telomere_length(self.telomere_bp)

        # Precompute each grid cell's body position, the per-cell ABC/SEdb LoRA
        # delta (dV_lora), and the full-LoRA adult target. The frozen zygote base
        # is the uniform V_ZYGOTE; the adapter supplies the position-dependent
        # differentiation delta added on top.
        self.positions = np.zeros((self.H, self.W, 3))
        F = np.zeros((self.H, self.W))                      # full ABC adult field
        for r in range(self.H):
            for c in range(self.W):
                pos = np.array([
                    c / max(1, self.W - 1),
                    r / max(1, self.H - 1),
                    z_slice,
                ])
                self.positions[r, c] = pos
                F[r, c] = self.lora.adult_voltage(pos)

        if self.conserved_base:
            # Conserved DV prepattern = the smooth, axis-organizing part of the
            # ABC field (its AP-average per DV row). This is the clade-shared
            # body-plan signal -> it belongs in the frozen base. The LoRA keeps
            # only the AP/organ-specific residual (the species-specific part).
            prepattern = F.mean(axis=1, keepdims=True) * np.ones((1, self.W))
            self.V_base = prepattern
            self.dV_lora = F - prepattern
        else:
            self.V_base = np.full((self.H, self.W), V_ZYGOTE)  # uniform zygote base
            self.dV_lora = F - V_ZYGOTE                         # full ABC delta

        # (3) Trained low-rank LoRA: replace the full-rank ABC delta with its
        # optimal rank-r factorization B @ A (SVD = the Eckart-Young best fit to
        # the measured ABC/SEdb field). This is the genuine LoRA Delta_W = B*A of
        # the paper, fit to data, in place of the full-rank direct lookup.
        self.lora_rank = None
        if lora_rank is not None:
            U, S, Vt = np.linalg.svd(self.dV_lora, full_matrices=False)
            r = int(min(lora_rank, S.size))
            self.lora_B = U[:, :r] * S[:r]      # H x r  (left factor)
            self.lora_A = Vt[:r, :]             # r x W  (right factor)
            approx = self.lora_B @ self.lora_A
            self.lora_rank_rmse = float(np.sqrt(np.mean((approx - self.dV_lora) ** 2)))
            self.lora_energy = float(np.sum(S[:r] ** 2) / np.sum(S ** 2))
            self.dV_lora = approx
            self.lora_rank = r

        # Full-LoRA (fully differentiated) adult target = base + delta.
        # Retained under this name for the viewer / figure / scenario consumers.
        self.V_adult_target = self.V_base + self.dV_lora

        # Initial state: tissue sits at its differentiated genomic target
        # (full telomere -> only ADULT live -> lora_scale = 1 -> full delta).
        self.V = self.V_adult_target.copy()
        self.step_count = 0

    # ---- the clock gates the kernel ----------------------------------------
    def _live_strata(self) -> List[KernelStratum]:
        return sorted({e.stratum for e in self.zygote_kernel.active_enhancers()
                       if e.stratum is not None}, key=lambda s: int(s))

    def _dediff_fraction(self) -> float:
        """0 when only ADULT is live; grows as FETAL then EMBRYONIC reactivate."""
        frac = 0.0
        for s in self._live_strata():
            frac += DEDIFF_WEIGHT.get(s, 0.0)
        return float(np.clip(frac, 0.0, 0.95))

    def lora_scale(self) -> float:
        """How much of the ABC/SEdb LoRA the clock currently applies, in [0,1].

        1 when only ADULT is live (fully differentiated); falls as earlier strata
        reactivate (the differentiation delta is withdrawn)."""
        return 1.0 - self._dediff_fraction()

    def _gated_target(self) -> np.ndarray:
        """Frozen zygote base + clock-scaled ABC/SEdb LoRA delta.

        target = V_base + lora_scale * dV_lora. Algebraically identical to the
        old (1-d)*V_adult + d*V_zygote blend, but now expressed as the diagram
        intends: a frozen fossil base with the adult adapter scaled on top."""
        return self.V_base + self.lora_scale() * self.dV_lora

    # ---- one local update (the shared rule, applied everywhere) ------------
    def step(self) -> GenomicNCAState:
        target = self._gated_target()

        # Perception (mirror of nca.py): identity + Sobel gradients.
        gx = _conv2d_same(self.V, _SOBEL_X)
        gy = _conv2d_same(self.V, _SOBEL_Y)
        grad_mag = np.sqrt(gx * gx + gy * gy)

        # Gap-junction coupling = discrete Laplacian (divergence of gradient).
        lap = _conv2d_same(self.V, _LAPLACIAN)

        # Genome-grounded local update, identical coefficients in every cell.
        dV = self.k_relax * (target - self.V) + self.k_gj * lap
        self.V = self.V + self.dt * dV
        self.step_count += 1

        return self._snapshot(target, grad_mag)

    def advance_clock(self, delta_bp: float) -> None:
        """Run the telomere clock down (or up) and re-gate the strata."""
        self.set_telomere(self.telomere_bp - delta_bp)

    def set_telomere(self, telomere_bp: float) -> None:
        """Set telomere length directly and re-gate the strata."""
        self.telomere_bp = max(0.0, float(telomere_bp))
        self.zygote_kernel.set_from_telomere_length(self.telomere_bp)

    # ---- diagnostics --------------------------------------------------------
    def _germ_fractions(self) -> Dict[str, float]:
        flat = self.V.ravel()
        n = flat.size
        out = {}
        for name, lo, hi in GERM_BANDS:
            out[name] = float(np.mean((flat >= lo) & (flat < hi)))
        return out

    def _snapshot(self, target: np.ndarray, grad_mag: np.ndarray) -> GenomicNCAState:
        return GenomicNCAState(
            step=self.step_count,
            telomere_bp=self.telomere_bp,
            prc2=self.zygote_kernel.prc2_level,
            live_strata=[s.name for s in self._live_strata()],
            voltage=self.V.copy(),
            target=target.copy(),
            germ_fractions=self._germ_fractions(),
            v_mean=float(self.V.mean()),
            v_std=float(self.V.std()),
            grad_mag_mean=float(grad_mag.mean()),
        )

    # ---- a full run: settle, then run the clock down -----------------------
    def simulate(
        self,
        settle_steps: int = 20,
        clock_steps: int = 8,
        steps_per_clock: int = 6,
        delta_bp_per_clock: float = 1300.0,
    ) -> List[GenomicNCAState]:
        """Settle at the differentiated target, then run the telomere clock down.

        Returns the full per-step history.
        """
        history: List[GenomicNCAState] = []

        # 1) Settle: tissue relaxes to its differentiated genomic voltage map.
        for _ in range(settle_steps):
            history.append(self.step())

        # 2) Age: shorten telomere in stages; each stage re-gates the strata and
        #    lets the tissue re-equilibrate toward the (now de-differentiating)
        #    target. Watch v_std collapse and live-strata count rise.
        for _ in range(clock_steps):
            self.advance_clock(delta_bp_per_clock)
            for _ in range(steps_per_clock):
                history.append(self.step())

        return history


    def _frame(self, snap, progress: float) -> dict:
        return {
            "progress": round(progress, 3),
            "telomere_bp": round(snap.telomere_bp, 1),
            "prc2": round(snap.prc2, 3),
            "live_strata": snap.live_strata,
            "v_mean": round(snap.v_mean, 2),
            "v_std": round(snap.v_std, 2),
            "germ_fractions": {k: round(v, 3) for k, v in snap.germ_fractions.items()},
            "voltage": [round(float(x), 2) for x in snap.voltage.ravel()],
        }

    def export_frames(
        self,
        direction: str = "age",
        n_frames: int = 41,
        settle_steps: int = 30,
        equilibrate_steps: int = 12,
        develop_steps_per_frame: int = 1,
    ) -> dict:
        """Build per-frame voltage grids for the scrub viewer.

        direction="age":     start fully differentiated at full telomere, then
                             run the clock DOWN; strata reactivate in reverse
                             order and the map de-differentiates (Vstd collapses).
        direction="develop": hold full telomere (adult enhancers available),
                             start from a UNIFORM zygote potential, and let the
                             conserved kernel pattern the tissue FORWARD --
                             differentiation as Vstd grows from ~0 to its target.

        Returns a JSON-serializable dict (grid + frames + metadata).
        """
        frames = []
        if direction == "age":
            axis_label = "Telomere clock  (full → 0 bp)"
            self.set_telomere(10000.0)
            self.V = self.V_adult_target.copy()
            for _ in range(settle_steps):
                self.step()
            for telo in np.linspace(10000.0, 0.0, n_frames):
                self.set_telomere(float(telo))
                for _ in range(equilibrate_steps):
                    snap = self.step()
                frames.append(self._frame(snap, 1.0 - telo / 10000.0))
        elif direction == "develop":
            axis_label = "Developmental time  (zygote → differentiated)"
            self.set_telomere(10000.0)            # adult enhancers patterning
            self.V = np.full((self.H, self.W), V_ZYGOTE, dtype=float)
            # Gentle the relaxation so the differentiation spreads across the
            # whole scrub rather than converging in the first few frames.
            k_relax_saved = self.k_relax
            self.k_relax = min(self.k_relax, 0.06)
            try:
                snap = self._snapshot(self._gated_target(),
                                      np.zeros((self.H, self.W)))
                frames.append(self._frame(snap, 0.0))  # pure uniform zygote
                for i in range(1, n_frames):
                    for _ in range(develop_steps_per_frame):
                        snap = self.step()
                    frames.append(self._frame(snap, i / (n_frames - 1)))
            finally:
                self.k_relax = k_relax_saved
        else:
            raise ValueError(f"unknown direction {direction!r}")

        return {
            "direction": direction,
            "axis_label": axis_label,
            "height": self.H,
            "width": self.W,
            "strata_source": self.strata_source,
            "stratum_counts": {s.name: int(n) for s, n in
                               self.zygote_kernel.stratum_counts().items()},
            "vmin": -75.0,
            "vmax": -20.0,
            "target": [round(float(x), 2) for x in self.V_adult_target.ravel()],
            "frames": frames,
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def _print_history(history: List[GenomicNCAState]) -> None:
    print(f"\n{'step':>4} {'telomere':>9} {'PRC2':>5} {'live strata':<24} "
          f"{'Vmean':>7} {'Vstd':>6} {'gradMag':>8}")
    print("-" * 72)
    last_key = None
    for s in history:
        key = (round(s.telomere_bp), tuple(s.live_strata))
        # Only print the first and last step of each clock stage to stay compact.
        mark = " <-" if key != last_key else "   "
        if key != last_key or s is history[-1]:
            print(f"{s.step:>4} {s.telomere_bp:>9.0f} {s.prc2:>5.2f} "
                  f"{'+'.join(s.live_strata):<24} {s.v_mean:>7.1f} {s.v_std:>6.2f} "
                  f"{s.grad_mag_mean:>8.3f}{mark}")
        last_key = key


def _try_save_png(nca: GenomicNCA, history: List[GenomicNCAState], path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(matplotlib unavailable, skipping PNG: {e})")
        return None

    first = history[0]
    last = history[-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vmin, vmax = -75, -20
    im0 = axes[0].imshow(nca.V_adult_target, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Base + full LoRA\n(ADULT, full telomere)")
    im1 = axes[1].imshow(first.voltage, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Differentiated tissue\nstep {first.step}, std={first.v_std:.1f}")
    im2 = axes[2].imshow(last.voltage, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"After clock run-down\n{'+'.join(last.live_strata)}, std={last.v_std:.1f}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im2, ax=axes, fraction=0.025, label="Vmem (mV)")
    fig.suptitle("Genomic NCA: frozen zygote base + ABC/SEdb LoRA, clock-gated", y=1.02)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    print("=" * 72)
    print("GENOMIC NCA  --  frozen zygote MLP base (V_zygote)")
    print("              + external ABC/SEdb LoRA adapter (GenomicChannelLookup)")
    print("              + telomere-gated zygote-kernel strata (ZygoteKernel)")
    print("=" * 72)

    import sys
    use_real = "--demo-strata" not in sys.argv
    nca = GenomicNCA(height=24, width=24, use_real_strata=use_real)
    print(f"Frozen zygote base: uniform V_zygote = {V_ZYGOTE:+.1f} mV over "
          f"{nca.H}x{nca.W} cells")
    print(f"ABC/SEdb LoRA delta range: "
          f"[{nca.dV_lora.min():+.1f}, {nca.dV_lora.max():+.1f}] mV "
          f"(channels {CHANNEL_NAMES})")
    print(f"Full-LoRA adult target Vmem range: "
          f"[{nca.V_adult_target.min():+.1f}, {nca.V_adult_target.max():+.1f}] mV")
    print(f"Strata source: {nca.strata_source}")
    print(f"Initial zygote kernel: {nca.zygote_kernel.summary()}")

    history = nca.simulate()
    _print_history(history)

    first, last = history[0], history[-1]
    print("\nDifferentiation read-out (germ-layer-like voltage bands):")
    print(f"  settled  (step {first.step:>3}, {'+'.join(first.live_strata)}): "
          + ", ".join(f"{k}={v:.0%}" for k, v in first.germ_fractions.items()))
    print(f"  aged     (step {last.step:>3}, {'+'.join(last.live_strata)}): "
          + ", ".join(f"{k}={v:.0%}" for k, v in last.germ_fractions.items()))

    dv = last.v_std - first.v_std
    print(f"\nVoltage-map heterogeneity (std): {first.v_std:.2f} -> {last.v_std:.2f} mV "
          f"({dv:+.2f}); a collapse toward zero = de-differentiation as the")
    print("fossil record reactivates in reverse order (FETAL then EMBRYONIC).")

    png = _try_save_png(nca, history, "genomic_nca_demo.png")
    if png:
        print(f"\nSaved figure: {png}")


if __name__ == "__main__":
    main()
