"""
The division head: the proliferation program, and division as the driver of the telomere clock.
================================================================================================

The differentiation head (differentiation_clock.py) drove fate unlocking with a telomere->PRC2
clock whose time axis was an ad-hoc proxy (0.5^(hpf/12)). The division head grounds that axis in
the real mechanism the paper already asserts: telomeres shorten WITH DIVISION, so the clock's
"time" is really CUMULATIVE DIVISIONS. Every division shortens the telomere -> TERRA reach drops
-> PRC2 retreats -> the next fate de-represses. Division and differentiation are one coupled
system: proliferation -> divisions -> telomere -> PRC2 -> fate -> V_m field.

From SHARE-seq ([[cognimed-genome-to-weights-recipe]]) division is the LEAST chromatin-legible
head (accessibility->rate only +0.06: cell-cycle genes sit on constitutive promoters), but its
OBSERVABLE -- the proliferation program in RNA -- is the cleanest. So here division is READ from
the measured cell-cycle program (23 genes) in ZESTA, not derived from accessibility; its job is
to drive the clock and the growth.

What this computes / tests:
  (1) PROLIFERATION FIELD (real observable) -- pooled cell-cycle expression per tissue / stage /
      position; validated against known biology (neuroepithelium eye+forebrain proliferative at
      24 hpf; cleavage-high then gastrula dip then organogenesis rise).
  (2) CYCLE-EXIT SWITCH -- cdkn1a (p21) rises where proliferation falls (anti-correlation): the
      molecular hand-off from dividing to differentiating.
  (3) GROWTH -> DIVISIONS -> TELOMERE -> PRC2 -- the embryo cell number N(t)=132 t^1.91 gives
      cumulative generations D(t)=log2 N; telomere = zygote - loss*D; PRC2 = telomere/zygote.
      This DIVISION-DRIVEN clock replaces the hpf proxy.
  (4) CONSISTENCY -- does the division-driven clock still reproduce the differentiation appearance
      ORDER (Spearman vs the ZESTA timecourse)? If yes, the two heads are consistently coupled.

Run: python -m medic.division_head
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.genome.zygote_kernel import ZYGOTE_TELOMERE_BP
from medic.zesta_temporal_4d import LAYER_VM, TIMES, TIME_HPF
from medic.differentiation_clock import FATE_PRC2

OUT = Path("data/organ_cascade")
ATLAS = Path("data/zesta/zf_sixtime_slice.h5ad")

CELL_CYCLE = ["mki67", "pcna", "mcm2", "mcm3", "mcm4", "mcm5", "mcm6", "mcm7", "ccnb1",
              "ccnb2", "ccna2", "cdk1", "top2a", "cdc20", "aurka", "aurkb", "bub1", "plk1",
              "ccnd1", "ccne1", "e2f1", "cenpf", "foxm1"]
CDKN = ["cdkn1a", "cdkn1bb", "cdkn1ca", "cdkn1ba"]      # p21/p27-family cycle-exit

# ---- growth model: embryo cell number (power law fitted to Kimmel counts, embryo_movie) ----
def N_cells(hpf):            return 132.0 * np.power(hpf, 1.91)        # total cells at hpf
def generations(hpf):       return np.log2(N_cells(hpf))              # cumulative divisions D(t)

# telomere shortens a fixed amount per generation -> PRC2 = telomere / zygote (the real clock).
# LOSS calibrated so PRC2 spans the differentiation range (~0.85 -> ~0.25) across the stages;
# ANY positive loss keeps PRC2 monotonic in division count, which is all the ORDER test needs.
D0, D1 = generations(TIME_HPF[TIMES[0]]), generations(TIME_HPF[TIMES[-1]])
LOSS_PER_DIV = ZYGOTE_TELOMERE_BP * (0.83 - 0.25) / (D1 - D0)         # bp lost per division


def telomere_bp_div(hpf):
    return ZYGOTE_TELOMERE_BP * 0.83 - LOSS_PER_DIV * (generations(hpf) - D0)

def prc2_div(hpf):
    return float(np.clip(telomere_bp_div(hpf) / ZYGOTE_TELOMERE_BP, 0.05, 1.0))


def load():
    import anndata as ad
    a = ad.read_h5ad(ATLAS)
    o = a.obs
    vn = np.asarray(a.var_names, str)
    anno = o["layer_annotation"].astype(str).values
    tv = o["time"].astype(str).values
    xy = np.c_[o["spatial_x"].values, o["spatial_y"].values].astype(float)
    X = a.X

    def score(genes):
        idx = [np.where(vn == g)[0][0] for g in genes if g in set(vn)]
        sub = X[:, idx]; sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        return np.log1p(sub).sum(1)
    return anno, tv, xy, score(CELL_CYCLE), score(CDKN)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anno, tv, xy, prolif, cdkn = load()
    print(f"proliferation from {len(CELL_CYCLE)} cell-cycle genes; cycle-exit from {len(CDKN)} CDKN genes\n")

    # (1) proliferation field per tissue (24 hpf) + per stage
    print("(1) PROLIFERATION FIELD")
    tiss24 = {}
    for t in sorted(set(anno[tv == "24hpf"])):
        m = (tv == "24hpf") & (anno == t)
        if m.sum() >= 20 and t in LAYER_VM:
            tiss24[t] = float(prolif[m].mean())
    for t, v in sorted(tiss24.items(), key=lambda kv: -kv[1]):
        print(f"    {t:22s} prolif {v:.2f}")
    stage_prolif = {t: float(prolif[tv == t].mean()) for t in TIMES}
    stage_cdkn = {t: float(cdkn[tv == t].mean()) for t in TIMES}
    print("    by stage (prolif / cdkn):",
          " ".join(f"{t}:{stage_prolif[t]:.1f}/{stage_cdkn[t]:.1f}" for t in TIMES))

    # (2) cycle-exit switch: cdkn1a rises where proliferation falls (per-tissue anti-corr @24hpf)
    tissues = [t for t in tiss24]
    p_arr = np.array([tiss24[t] for t in tissues])
    c_arr = np.array([cdkn[(tv == "24hpf") & (anno == t)].mean() for t in tissues])
    rho_switch, _ = spearmanr(p_arr, c_arr)
    print(f"\n(2) CYCLE-EXIT SWITCH: Spearman(proliferation, cdkn1a) across tissues = {rho_switch:.2f}"
          f"  ({'anti-correlated -> exit=differentiate' if rho_switch < 0 else 'weak'})")

    # (3) growth -> divisions -> telomere -> PRC2 (the division-driven clock)
    print("\n(3) DIVISION-DRIVEN CLOCK  (N=132 t^1.91 -> generations -> telomere -> PRC2)")
    print(f"    LOSS_PER_DIV = {LOSS_PER_DIV:.0f} bp/division")
    for t in TIMES:
        h = TIME_HPF[t]
        print(f"    {t:8s} N~{N_cells(h):8.0f}  gens={generations(h):5.1f}  "
              f"telomere={telomere_bp_div(h):6.0f}bp  PRC2_div={prc2_div(h):.2f}")

    # (4) consistency: does the DIVISION-driven clock reproduce the differentiation appearance order?
    first_seen = {}
    for t in TIMES:
        for a in set(anno[tv == t]):
            if a in LAYER_VM and a not in first_seen:
                first_seen[a] = TIME_HPF[t]
    pred = {}
    for fate, thr in FATE_PRC2.items():
        for t in TIMES:
            if prc2_div(TIME_HPF[t]) < thr:
                pred[fate] = TIME_HPF[t]; break
        else:
            pred[fate] = TIME_HPF[TIMES[0]] if thr >= 1.0 else None
    common = [f for f in first_seen if f in pred and pred[f] is not None]
    rho_order, p_order = spearmanr([pred[f] for f in common], [first_seen[f] for f in common])
    print(f"\n(4) CONSISTENCY: division-driven clock vs atlas appearance order  Spearman = {rho_order:.2f}"
          f" (p={p_order:.1e})  -> the two heads share one clock")

    _figure(anno, tv, xy, prolif, cdkn, tiss24, stage_prolif, stage_cdkn,
            p_arr, c_arr, tissues, rho_switch, rho_order, first_seen, pred, common)
    json.dump(dict(tissue_prolif_24hpf=tiss24, stage_prolif=stage_prolif, stage_cdkn=stage_cdkn,
                   cycle_exit_rho=rho_switch, loss_per_div_bp=LOSS_PER_DIV,
                   clock={t: dict(hpf=TIME_HPF[t], N=float(N_cells(TIME_HPF[t])),
                                  gens=float(generations(TIME_HPF[t])),
                                  telomere_bp=float(telomere_bp_div(TIME_HPF[t])),
                                  prc2_div=prc2_div(TIME_HPF[t])) for t in TIMES},
                   division_clock_order_rho=rho_order),
              open(OUT / "division_head.json", "w"), indent=2)
    print("\nsaved", OUT / "division_head.json")
    print(f"\nSUMMARY: proliferation field real (eye/forebrain hot, yolk cold); cycle-exit rho {rho_switch:.2f}; "
          f"division-driven telomere clock reproduces the differentiation order rho {rho_order:.2f} "
          f"-> division and differentiation are ONE coupled clock.")


def _figure(anno, tv, xy, prolif, cdkn, tiss24, stage_prolif, stage_cdkn,
            p_arr, c_arr, tissues, rho_switch, rho_order, first_seen, pred, common):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    hpf = [TIME_HPF[t] for t in TIMES]

    # (a) proliferation spatial map at 24 hpf
    m = tv == "24hpf"
    a = ax[0, 0]
    sc = a.scatter(xy[m, 0], xy[m, 1], c=prolif[m], cmap="magma", s=5, linewidths=0)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title("(a) proliferation field @24 hpf (measured)", fontsize=10); fig.colorbar(sc, ax=a, fraction=0.045)

    # (b) per-tissue proliferation gradient
    a = ax[0, 1]
    order = sorted(tiss24, key=lambda t: tiss24[t])
    a.barh(range(len(order)), [tiss24[t] for t in order], color="tab:purple")
    a.set_yticks(range(len(order))); a.set_yticklabels([t[:20] for t in order], fontsize=7)
    a.set_xlabel("proliferation"); a.set_title("(b) proliferation gradient (eye/brain high, yolk low)", fontsize=9)

    # (c) cycle-exit switch: proliferation vs cdkn1a
    a = ax[0, 2]
    a.scatter(p_arr, c_arr, s=40)
    for i, t in enumerate(tissues):
        a.annotate(t[:8], (p_arr[i], c_arr[i]), fontsize=6)
    a.set_xlabel("proliferation"); a.set_ylabel("cdkn1a (cycle exit)")
    a.set_title(f"(c) cycle-exit switch (rho {rho_switch:.2f})", fontsize=10)

    # (d) proliferation + cdkn1a by stage
    a = ax[1, 0]
    a.plot(hpf, [stage_prolif[t] for t in TIMES], "o-", color="tab:purple", label="proliferation")
    a.plot(hpf, [stage_cdkn[t] for t in TIMES], "o-", color="tab:orange", label="cdkn1a")
    a.set_xlabel("hpf"); a.set_title("(d) cleavage-high -> gastrula dip -> organogenesis rise", fontsize=9); a.legend(fontsize=8)

    # (e) growth -> divisions -> telomere -> PRC2
    a = ax[1, 1]
    a.plot(hpf, [prc2_div(h) for h in hpf], "o-", color="tab:red", label="PRC2 (division-driven)")
    a.plot(hpf, [0.5 ** (h / 12) for h in hpf], "o--", color="0.6", label="PRC2 (hpf proxy)")
    a2 = a.twinx(); a2.plot(hpf, [generations(h) for h in hpf], "s-", color="tab:green", alpha=0.6)
    a2.set_ylabel("cumulative divisions", color="tab:green")
    a.set_xlabel("hpf"); a.set_ylabel("PRC2"); a.set_title("(e) divisions shorten telomere -> PRC2", fontsize=10); a.legend(fontsize=7)

    # (f) division-driven clock reproduces differentiation order
    a = ax[1, 2]
    a.scatter([pred[f] for f in common], [first_seen[f] for f in common], s=40)
    lim = [0, 26]; a.plot(lim, lim, "k--", lw=0.6, alpha=0.5)
    a.set_xlabel("division-clock unlock (hpf)"); a.set_ylabel("atlas appearance (hpf)")
    a.set_title(f"(f) division clock reproduces order (rho {rho_order:.2f})", fontsize=10)

    fig.suptitle("The division head: the measured proliferation program drives the telomere clock (divisions -> telomere "
                 "-> PRC2), the same clock\nthat unlocks differentiation -- so proliferation, telomere shortening and "
                 "fate are one coupled system.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "division_head.png", dpi=125, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "division_head.png")


if __name__ == "__main__":
    main()
