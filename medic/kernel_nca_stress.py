"""
Stress test of the zygote-kernel NCA: growth waves, screening length (cymatics),
eigenmodes, robustness, and numerical stability.

The kernel NCA update (tissue/genomic_nca.py) is pure bioelectric physics:

    dV/dt = k_relax (V_target - V) + k_gj * laplacian(V)

The genome enters ONLY as the target field V_target (the ABC LoRA); the dynamics
here are gap-junction electrophysics, so this section is deliberately almost free
of the LLM/MLP. We characterise:

  (1) Settling wave  : from the uniform zygote base, the field relaxes+diffuses to
                       the genomic target -- a morphogenetic growth/settling wave.
  (2) Robustness     : ablate a patch, watch it heal back to target.
  (3) Screening length: a point source decays as exp(-r/lambda),
                       lambda = sqrt(k_gj / k_relax) -- THIS sets the spatial
                       pattern scale (the "cymatic wavelength").
  (4) Cymatic modes  : eigenfunctions of the gap-junction Laplacian on the domain
                       ARE Chladni/standing-wave patterns; the NCA operator
                       (k_gj * Lap - k_relax * I) shares this eigenbasis.
  (5) Stability bound: explicit scheme is stable for dt(k_relax + 8 k_gj) <= 2.
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.tissue.genomic_nca import GenomicNCA, _conv2d_same, _LAPLACIAN

OUT = __import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "organ_cascade"


# ---------- (1) settling wave from the zygote base ----------
def settling_wave(steps=60):
    nca = GenomicNCA(height=24, width=24)
    target = nca.V_adult_target.copy()
    nca.V = nca.V_base.copy()                      # start at uniform zygote
    res, vstd, frames = [], [], []
    for s in range(steps):
        nca.step()
        res.append(float(np.sqrt(np.mean((nca.V - target) ** 2))))
        vstd.append(float(nca.V.std()))
        if s in (0, 3, 10, steps - 1):
            frames.append((s, nca.V.copy()))
    return np.array(res), np.array(vstd), frames, target


# ---------- (2) ablation / robustness ----------
def ablation_recovery(settle=40, ablate_after=40, recover=60):
    nca = GenomicNCA(height=24, width=24)
    target = nca.V_adult_target.copy()
    for _ in range(settle):
        nca.step()
    pre = float(np.sqrt(np.mean((nca.V - target) ** 2)))
    # ablate: clamp a 6x6 corner patch far from its target value
    nca.V[2:8, 2:8] = +20.0
    wound = float(np.sqrt(np.mean((nca.V - target) ** 2)))
    heal = []
    for _ in range(recover):
        nca.step()
        heal.append(float(np.sqrt(np.mean((nca.V - target) ** 2))))
    return pre, wound, np.array(heal)


# ---------- (3) screening length from a point-source Green's function ----------
def screening_length(k_gj, k_relax=0.30, N=49, dt=0.1, steps=4000):
    """Uniform target 0, clamp center to +1, settle; fit V(r) ~ exp(-r/lambda)."""
    nca = GenomicNCA(height=N, width=N, k_relax=k_relax, k_gj=k_gj, dt=dt)
    nca.V_base[:] = 0.0
    nca.dV_lora[:] = 0.0
    nca.V[:] = 0.0
    c = N // 2
    for _ in range(steps):
        nca.V[c, c] = 1.0
        nca.step()
    nca.V[c, c] = 1.0
    # radial profile (along +x from center, avoid the clamped node)
    rs = np.arange(2, N // 2 - 1)
    vals = np.array([nca.V[c, c + r] for r in rs])
    good = vals > 1e-4
    rs, vals = rs[good], vals[good]
    lam_meas = float(-1.0 / np.polyfit(rs, np.log(vals), 1)[0]) if len(rs) > 3 else np.nan
    lam_pred = float(np.sqrt(k_gj / k_relax))
    return lam_pred, lam_meas


# ---------- (4) cymatic eigenmodes of the gap-junction Laplacian ----------
def laplacian_eigenmodes(N=24, n_modes=4):
    """Neumann (free-edge) discrete Laplacian on an NxN grid; its eigenvectors are
    the Chladni / standing-wave (cymatic) modes."""
    idx = lambda r, c: r * N + c
    L = np.zeros((N * N, N * N))
    for r in range(N):
        for c in range(N):
            i = idx(r, c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < N and 0 <= cc < N:
                    L[i, idx(rr, cc)] += 1.0
                    L[i, i] -= 1.0
    w, V = np.linalg.eigh(L)                       # ascending; w<=0
    order = np.argsort(-w)                         # 0, then most negative... take low-|w|
    modes = []
    for k in order[1:1 + n_modes]:                # skip the flat (w=0) mode
        modes.append((float(-w[k]), V[:, k].reshape(N, N)))
    return modes


# ---------- (5) numerical stability bound ----------
def stability_sweep(k_relax=0.30, dt=1.0):
    bound = (2.0 / dt - k_relax) / 8.0            # k_gj_max from dt(k_relax+8k_gj)<=2
    kgjs = np.linspace(0.02, 0.40, 20)
    maxabs = []
    for kg in kgjs:
        nca = GenomicNCA(height=24, width=24, k_relax=k_relax, k_gj=kg, dt=dt)
        nca.V += np.random.default_rng(0).standard_normal(nca.V.shape) * 1.0
        ok = True
        for _ in range(80):
            nca.step()
            if not np.isfinite(nca.V).all() or np.abs(nca.V).max() > 1e6:
                ok = False; break
        maxabs.append(float(np.abs(nca.V).max()) if ok else 1e6)
    return kgjs, np.array(maxabs), bound


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res, vstd, frames, target = settling_wave()
    pre, wound, heal = ablation_recovery()
    kgj_list = [0.05, 0.1, 0.2, 0.45, 0.8, 1.4]
    scr = [screening_length(k) for k in kgj_list]
    modes = laplacian_eigenmodes()
    kgjs, maxabs, bound = stability_sweep()

    report = {
        "settling": {"res_start": float(res[0]), "res_end": float(res[-1]),
                     "convergence_ratio": float(res[-1] / res[0])},
        "ablation": {"pre_rmse": pre, "wound_rmse": wound,
                     "post_rmse": float(heal[-1]),
                     "recovered_frac": float(1 - (heal[-1] - pre) / (wound - pre))},
        "screening": [{"k_gj": k, "lambda_pred": p, "lambda_meas": m}
                      for k, (p, m) in zip(kgj_list, scr)],
        "stability_bound_kgj": float(bound),
    }
    json.dump(report, open(OUT / "kernel_nca_stress.json", "w"), indent=2)

    # ---- figure ----
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(res, label="RMSE to target"); ax.plot(vstd, label="V std")
    ax.set_title("(1) Settling wave from zygote base"); ax.set_xlabel("NCA step")
    ax.set_ylabel("mV"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.concatenate([[wound], heal]))
    ax.axhline(pre, ls="--", c="g", label=f"pre-ablation {pre:.1f}")
    ax.set_title(f"(2) Ablation recovery\n{report['ablation']['recovered_frac']:.0%} healed")
    ax.set_xlabel("step after wound"); ax.set_ylabel("RMSE to target"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 2])
    pr = [p for p, m in scr]; me = [m for p, m in scr]
    ax.plot(pr, me, "o-")
    lim = [0, max(pr + me) * 1.1]; ax.plot(lim, lim, "k--", lw=0.7, label="y=x")
    ax.set_title("(3) Screening length sets scale\n"
                 r"$\lambda=\sqrt{k_{gj}/k_{relax}}$"); ax.set_xlabel("predicted lambda (cells)")
    ax.set_ylabel("measured lambda (cells)"); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(kgjs, maxabs, "o-"); ax.axvline(bound, c="r", ls="--",
            label=f"predicted bound {bound:.2f}")
    ax.set_yscale("log"); ax.set_title("(5) Numerical stability")
    ax.set_xlabel("k_gj"); ax.set_ylabel("max|V| after 80 steps"); ax.legend(fontsize=7)

    # (4) cymatic eigenmodes
    for j, (lam, mode) in enumerate(modes):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(mode, cmap="RdBu_r"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"(4) cymatic mode {j+1}\nLaplacian eig {lam:.2f}", fontsize=8)

    fig.suptitle("Zygote-kernel NCA stress test: growth/settling wave, robustness, "
                 "screening length (cymatic scale), Chladni eigenmodes, stability",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "kernel_nca_stress.png", dpi=140, bbox_inches="tight")
    print("saved", OUT / "kernel_nca_stress.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
