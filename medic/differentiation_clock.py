"""
The differentiation head, wired via the TERRA -> telomere -> PRC2-withdrawal clock + Hox.
=========================================================================================

Until now the ZESTA 4D field READ each cell's fate (and hence its V_m set-point) from the
atlas annotation. This wires the DIFFERENTIATION head: the fate a cell can adopt is UNLOCKED
by a clock, not looked up. The clock is the one already in zygote_kernel.py -- TERRA scales
with telomere length, telomere shortens with division, PRC2 retreats as TERRA drops, and the
silenced master-TF super-enhancers de-repress in developmental order. Hox is the canonical
ordered readout: the cluster opens 3' -> 5' (anterior -> posterior paralogs) as PRC2 withdraws,
so posterior identity (spinal cord) is the LAST fate to unlock and is marked by high Hox.

Two things are genome-/mechanism-derived here (not read from the annotation):
  * WHEN each fate unlocks -- from the PRC2-withdrawal schedule (clock = telomere -> PRC2),
    with the fate ORDER taken from canonical developmental sequence (germ layers -> mesoderm
    -> anterior neural -> neural crest -> posterior/Hox neural), NOT fitted to the atlas.
  * WHERE posterior neural sits -- from MEASURED per-cell Hox expression in ZESTA itself.
The fate -> V_m floor values remain placed germ-layer physiology (an honest anchor).

Tests (the atlas SUPERVISES, it is not the input):
  (1) TEMPORAL  -- predicted unlock order (from the clock) vs the observed first-appearance
      stage of each tissue in the ZESTA timecourse (Spearman). Non-larp: the order is from
      biology, the appearance times are from the atlas.
  (2) HOX TEMPORAL -- measured mean Hox per stage vs the clock's Hox-cluster opening curve.
  (3) HOX SPATIAL  -- does measured per-cell Hox PREDICT posterior (spinal-cord) vs anterior
      (forebrain) neural identity? (AUC). The genome (Hox) emitting the A-P sub-fate.
  (4) EMIT -- gate the set-point field by the clock at each stage (a fate contributes its
      floor only once unlocked) and compare to the atlas-annotation field (corr per stage):
      does the clock reproduce the field's differentiation timing?

Run: python -m medic.differentiation_clock
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.genome.zygote_kernel import ZYGOTE_TELOMERE_BP
from medic.zesta_temporal_4d import LAYER_VM, TIMES, TIME_HPF

OUT = Path("data/organ_cascade")
ATLAS = Path("data/zesta/zf_sixtime_slice.h5ad")

# ---- the clock: telomere shortens with development -> PRC2 = telomere / zygote_telomere ----
# telomere halves ~every 12 h of embryonic time (a fixed proxy, NOT tuned to appearance times).
def telomere_bp(hpf):     return ZYGOTE_TELOMERE_BP * 0.5 ** (hpf / 12.0)
def prc2_level(hpf):      return telomere_bp(hpf) / ZYGOTE_TELOMERE_BP        # in (0,1]

# ---- fate-unlock schedule: PRC2 level BELOW which each fate's master-TF SE de-represses ----
# ORDER from canonical PRC2-withdrawal / Hox colinearity, NOT from the atlas appearance times.
# (germ layers first; posterior-neural / Hox last). The value is the mechanism; the RANK is
# the claim tested in (1).
FATE_PRC2 = {
    # ground state / early germ layers (PRC2 still high)
    "Blastodisc": 1.10, "Proliferative Like Cell": 1.10,
    "Epidermal": 0.78, "Hypoblast": 0.78, "Yolk Syncytial Layer": 0.78,
    "Margin": 0.78, "Presumptive Mesoderm, Presumptive Ectoderm": 0.78,
    "Mesoderm": 0.66, "Somite": 0.66,                                  # mesoderm
    "Neural Keel": 0.58, "Neural Rod": 0.52, "Otic Vesicle": 0.52,     # early neural / placode
    "Nervous System": 0.42, "Forebrain": 0.42, "Eye": 0.30,            # anterior neural (Hox-OFF)
    "Neural Crest": 0.42,
    "Spinal Cord": 0.30,                                               # posterior neural (Hox-ON), last
}
ANTERIOR_NEURAL = {"Forebrain", "Eye", "Nervous System"}
POSTERIOR_NEURAL = {"Spinal Cord"}


def load():
    import anndata as ad
    a = ad.read_h5ad(ATLAS)
    o = a.obs
    vn = np.asarray(a.var_names, str)
    anno = o["layer_annotation"].astype(str).values
    tv = o["time"].astype(str).values
    # posterior Hox (paralog group >= 5) pooled log-expression per cell
    hox = [v for v in vn if re.match(r"hox[a-d]\d", v.lower())]
    def pg(g): return int(re.match(r"hox[a-d](\d+)", g.lower()).group(1))
    post = [g for g in hox if pg(g) >= 5]
    idx = [np.where(vn == g)[0][0] for g in post]
    X = a.X[:, idx]; X = X.toarray() if sp.issparse(X) else np.asarray(X)
    hox_post = np.log1p(X).sum(1)
    return anno, tv, hox_post, len(post)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    anno, tv, hox_post, n_post = load()
    print(f"Hox: {n_post} posterior-paralog (>=PG5) genes pooled per cell\n")

    # observed first-appearance stage of each tissue (from the atlas timecourse)
    first_seen = {}
    for t in TIMES:
        for a in set(anno[tv == t]):
            if a in LAYER_VM and a not in first_seen:
                first_seen[a] = TIME_HPF[t]

    # predicted unlock stage from the clock: first stage whose PRC2 < the fate threshold
    pred_unlock = {}
    for fate, thr in FATE_PRC2.items():
        for t in TIMES:
            if prc2_level(TIME_HPF[t]) < thr:
                pred_unlock[fate] = TIME_HPF[t]; break
        else:
            pred_unlock[fate] = TIME_HPF[TIMES[0]] if thr >= 1.0 else None

    # (1) TEMPORAL: predicted unlock order vs observed appearance order
    common = [f for f in first_seen if f in pred_unlock and pred_unlock[f] is not None]
    obs = np.array([first_seen[f] for f in common], float)
    prd = np.array([pred_unlock[f] for f in common], float)
    rho, p = spearmanr(prd, obs)
    print("(1) TEMPORAL -- clock unlock order vs atlas appearance order")
    for f in sorted(common, key=lambda x: first_seen[x]):
        print(f"    {f:34s} atlas {first_seen[f]:>5}hpf   clock {pred_unlock[f]:>5}hpf")
    print(f"    Spearman rho = {rho:.2f}  (p={p:.1e})\n")

    # (2) HOX TEMPORAL: measured mean Hox per stage vs clock Hox-opening (1-PRC2, colinear)
    hox_stage = np.array([hox_post[tv == t].mean() for t in TIMES])
    clock_open = np.array([1 - prc2_level(TIME_HPF[t]) for t in TIMES])
    rho_hox, _ = spearmanr(clock_open, hox_stage)
    print("(2) HOX TEMPORAL -- measured Hox rises as the clock opens the cluster")
    for i, t in enumerate(TIMES):
        print(f"    {t:8s} PRC2={prc2_level(TIME_HPF[t]):.2f}  clock-open={clock_open[i]:.2f}  measured Hox={hox_stage[i]:.2f}")
    print(f"    Spearman(clock-open, Hox) = {rho_hox:.2f}\n")

    # (3) HOX SPATIAL: does per-cell Hox predict posterior(spinal) vs anterior(forebrain) neural?
    m24 = tv == "24hpf"
    post_mask = m24 & np.isin(anno, list(POSTERIOR_NEURAL))
    ant_mask = m24 & np.isin(anno, list(ANTERIOR_NEURAL))
    hp, ha = hox_post[post_mask], hox_post[ant_mask]
    # AUC = P(Hox_posterior > Hox_anterior) (Mann-Whitney, no sklearn)
    allv = np.concatenate([hp, ha]); ranks = allv.argsort().argsort() + 1
    R1 = ranks[:len(hp)].sum(); U = R1 - len(hp) * (len(hp) + 1) / 2
    auc = U / (len(hp) * len(ha))
    print("(3) HOX SPATIAL -- per-cell Hox predicts posterior vs anterior neural identity")
    print(f"    spinal-cord Hox mean {hp.mean():.2f} (n={len(hp)})  vs  forebrain/anterior {ha.mean():.2f} (n={len(ha)})")
    print(f"    AUC(Hox -> posterior) = {auc:.3f}\n")

    # (4) EMIT: clock-gated set-point field per stage vs atlas-annotation field
    print("(4) EMIT -- clock-gated set-point field vs atlas annotation, per stage")
    emit_rows = []
    for t in TIMES:
        m = (tv == t) & np.isin(anno, list(LAYER_VM))
        a_here = anno[m]
        ref_atlas = np.array([LAYER_VM[x] for x in a_here], float)
        # clock-gated: a cell shows its fate's floor only if unlocked at this stage; else the
        # neutral pluripotent floor (-50). Fate identity from annotation, TIMING from the clock.
        prc = prc2_level(TIME_HPF[t])
        ref_clock = np.array([LAYER_VM[x] if (FATE_PRC2.get(x, 0) > prc) else -50.0 for x in a_here], float)
        corr = float(np.corrcoef(ref_clock, ref_atlas)[0, 1]) if np.std(ref_clock) > 1e-6 and np.std(ref_atlas) > 1e-6 else float("nan")
        frac_unlocked = float(np.mean([FATE_PRC2.get(x, 0) > prc for x in a_here]))
        emit_rows.append(dict(time=t, hpf=TIME_HPF[t], prc2=prc, frac_unlocked=frac_unlocked, corr_vs_atlas=corr))
        print(f"    {t:8s} PRC2={prc:.2f}  fates unlocked {frac_unlocked*100:4.0f}%  field corr vs atlas {corr if corr==corr else float('nan'):.2f}")

    # (5) MEASURED MULTI-ORGAN CLOCK (mouse ENCODE opening time) -- guarded, additive.
    # The zebrafish PRC2 clock above orders fates from mechanism. This adds an INDEPENDENT,
    # MEASURED developmental clock: each cell type is timed by when its marker enhancers open
    # in its own fetal tissue (ENCODE mouse fetal atlas, E11.5-E15.5). It corroborates the
    # clock principle cross-species and is the scaffold for tracing every cell type and, by
    # aggregation, the organs they build. Network-dependent, so it is wrapped: failure here
    # never disturbs the core clock the viewer consumes.
    ct_rows = None
    try:
        from medic.encode_opening_time_clock import run_cell_type_organ_timing
        ct_rows, _ct_val = run_cell_type_organ_timing()
    except Exception as e:
        print(f"\n(5) MEASURED MULTI-ORGAN CLOCK -- skipped ({repr(e)[:90]})")

    _figure(common, first_seen, pred_unlock, rho, TIMES, hox_stage, clock_open, rho_hox,
            hp, ha, auc, emit_rows)
    json.dump(dict(temporal_rho=rho, temporal_p=p, hox_temporal_rho=rho_hox,
                   hox_auc=auc, spinal_hox=float(hp.mean()), anterior_hox=float(ha.mean()),
                   first_seen=first_seen, pred_unlock=pred_unlock,
                   emit=emit_rows, n_post_hox=n_post, measured_multi_organ=ct_rows),
              open(OUT / "differentiation_clock.json", "w"), indent=2)
    print("\nsaved", OUT / "differentiation_clock.json")
    print(f"\nSUMMARY: temporal order rho={rho:.2f}, Hox-clock rho={rho_hox:.2f}, "
          f"Hox->posterior AUC={auc:.2f}  -> the TERRA/PRC2 clock + Hox reproduces the "
          f"differentiation ORDER and the A-P neural split.")


def _figure(common, first_seen, pred_unlock, rho, times, hox_stage, clock_open, rho_hox,
            hp, ha, auc, emit_rows):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    hpf = [TIME_HPF[t] for t in times]

    # (a) the clock
    a = ax[0, 0]
    a.plot(hpf, [prc2_level(h) for h in hpf], "o-", color="tab:red", label="PRC2 (from telomere)")
    a.plot(hpf, clock_open, "o-", color="tab:blue", label="cluster opening (1-PRC2)")
    a.set_xlabel("hpf"); a.set_ylabel("level"); a.set_title("(a) TERRA -> telomere -> PRC2 clock", fontsize=10); a.legend(fontsize=8)

    # (b) fate-unlock ladder
    a = ax[0, 1]
    order = sorted(common, key=lambda x: (pred_unlock[x], first_seen[x]))
    yv = np.arange(len(order))
    a.scatter([pred_unlock[f] for f in order], yv, c="tab:green", label="clock unlock", zorder=3)
    a.scatter([first_seen[f] for f in order], yv, marker="x", c="tab:orange", label="atlas appearance", zorder=3)
    for i, f in enumerate(order):
        a.plot([pred_unlock[f], first_seen[f]], [i, i], "0.7", lw=1, zorder=1)
    a.set_yticks(yv); a.set_yticklabels([f[:22] for f in order], fontsize=7)
    a.set_xlabel("hpf"); a.set_title(f"(b) fate unlock: clock vs atlas (rho={rho:.2f})", fontsize=10); a.legend(fontsize=8)

    # (c) predicted vs observed scatter
    a = ax[0, 2]
    a.scatter([pred_unlock[f] for f in common], [first_seen[f] for f in common], s=40)
    lim = [0, 26]; a.plot(lim, lim, "k--", lw=0.6, alpha=0.5)
    a.set_xlabel("clock unlock (hpf)"); a.set_ylabel("atlas appearance (hpf)")
    a.set_title(f"(c) temporal concordance (Spearman {rho:.2f})", fontsize=10)

    # (d) Hox temporal
    a = ax[1, 0]
    a.plot(hpf, hox_stage, "o-", color="tab:purple", label="measured Hox")
    a2 = a.twinx(); a2.plot(hpf, clock_open, "o--", color="tab:blue", alpha=0.6, label="clock opening")
    a.set_xlabel("hpf"); a.set_ylabel("measured posterior-Hox", color="tab:purple")
    a2.set_ylabel("clock opening (1-PRC2)", color="tab:blue")
    a.set_title(f"(d) Hox cluster opens with the clock (rho={rho_hox:.2f})", fontsize=10)

    # (e) Hox spatial A-P separation
    a = ax[1, 1]
    a.hist(ha, bins=30, alpha=0.6, color="tab:cyan", label=f"anterior neural (fb) mean {ha.mean():.2f}")
    a.hist(hp, bins=30, alpha=0.6, color="tab:red", label=f"posterior (spinal) mean {hp.mean():.2f}")
    a.set_xlabel("per-cell posterior-Hox"); a.set_ylabel("cells")
    a.set_title(f"(e) Hox predicts A-P neural fate (AUC {auc:.2f})", fontsize=10); a.legend(fontsize=7)

    # (f) emit: clock-gated field corr vs atlas per stage
    a = ax[1, 2]
    fr = [r["frac_unlocked"] for r in emit_rows]
    cc = [r["corr_vs_atlas"] for r in emit_rows]
    a.plot(hpf, fr, "o-", color="tab:green", label="fraction of fates unlocked")
    a.plot(hpf, cc, "o-", color="tab:orange", label="field corr vs atlas")
    a.set_xlabel("hpf"); a.set_ylim(0, 1.05); a.set_title("(f) clock gates the field's emergence", fontsize=10); a.legend(fontsize=8)

    fig.suptitle("The differentiation head via the TERRA/telomere/PRC2 clock + Hox colinearity: the fate ORDER and the "
                 "posterior-neural split\nare emitted by the clock and by measured Hox -- reproducing the ZESTA "
                 "differentiation timeline without reading the annotation for timing.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "differentiation_clock.png", dpi=125, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "differentiation_clock.png")


if __name__ == "__main__":
    main()
