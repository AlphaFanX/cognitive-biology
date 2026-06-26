"""
Organ Cascade Stage 2 v2 — ENRICHMENT-corrected wiring.

Stage 2 v1 found naive motif PRESENCE is non-specific (super-enhancers are long
and motif-dense, so almost every TF motif occurs in almost every SE -> saturated
graphs, autoreg ~= null, flat cross-organ confusion). The standard fix: an edge
TF_a -> TF_b counts only when TF_a's motif is ENRICHED (concentrated above
genomic background) in TF_b's organ super-enhancers -- homotypic clustering, the
real core-regulatory-circuit signal.

Reuses the cached scan (region_binders.json) -- no re-scanning.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from medic.organ_cascade_wiring import WIRE_DIR, organ_head_tfs, KNOWN_EDGES

FOLD = 2.0          # enrichment fold over background to call a TF organ-active
TOP_K = 18


def load():
    rb_raw = json.load(open(WIRE_DIR / "region_binders.json"))
    region_binders = {k: set(v) for k, v in rb_raw.items()}
    ot_raw = json.load(open(WIRE_DIR / "organ_targets.json"))
    # organ -> tf_b -> list of "c:s-e" keys
    organ_targets = {}
    for o, d in ot_raw.items():
        organ_targets[o] = {tf: [f"{c}:{s}-{e}" for (c, s, e) in regs]
                            for tf, regs in d.items()}
    return region_binders, organ_targets


def main():
    region_binders, organ_targets = load()
    organs = list(organ_targets.keys())
    heads = organ_head_tfs(top_k=TOP_K)

    all_regions = list(region_binders.keys())
    n_all = len(all_regions)
    regulators = sorted(set().union(*region_binders.values())) if region_binders else []

    # background hit-rate per regulator across ALL SEs
    bg = {tf: sum(tf in region_binders[r] for r in all_regions) / n_all
          for tf in regulators}

    # per-organ unique SE regions
    organ_regions = {o: sorted({r for tf in organ_targets[o]
                                for r in organ_targets[o][tf]}) for o in organs}

    # rate[tf, organ] = fraction of organ's SEs the TF binds
    rate = np.zeros((len(regulators), len(organs)))
    tfi = {t: i for i, t in enumerate(regulators)}
    for j, o in enumerate(organs):
        regs = organ_regions[o]
        if not regs:
            continue
        for tf in regulators:
            rate[tfi[tf], j] = sum(tf in region_binders[r] for r in regs) / len(regs)

    # enrichment over background, and z-score across organs (per TF)
    bg_vec = np.array([bg[t] for t in regulators])[:, None] + 1e-9
    enrich = rate / bg_vec
    mu = rate.mean(axis=1, keepdims=True)
    sd = rate.std(axis=1, keepdims=True) + 1e-9
    z = (rate - mu) / sd

    # cross-organ specificity confusion using z-scored motif concentration:
    # conf[i,j] = mean over organ-i head TFs (with scan data) of z[TF, organ_j]
    conf = np.full((len(organs), len(organs)), np.nan)
    for i, oi in enumerate(organs):
        htfs = [t for t in heads[oi] if t in tfi]
        if not htfs:
            continue
        for j in range(len(organs)):
            conf[i, j] = np.mean([z[tfi[t], j] for t in htfs])

    # enrichment-corrected edges: TF_a -> TF_b in organ o iff TF_a binds one of
    # TF_b's SEs AND TF_a is >=FOLD enriched in organ o's SE set
    crc = {}
    for j, o in enumerate(organs):
        enr_tfs = {t for t in heads[o] if t in tfi and enrich[tfi[t], j] >= FOLD}
        edges = set(); autoreg = set()
        for tf_b, regs in organ_targets[o].items():
            for r in regs:
                for tf_a in (region_binders.get(r, set()) & enr_tfs):
                    if tf_a in heads[o]:
                        edges.add((tf_a, tf_b))
                        if tf_a == tf_b:
                            autoreg.add(tf_a)
        nodes = [t for t in heads[o] if t in tfi]
        crc[o] = {"nodes": nodes, "edges": sorted(edges),
                  "autoreg": sorted(autoreg), "n_enriched": len(enr_tfs)}

    # report
    print("=" * 76)
    print("ORGAN CASCADE Stage 2 v2 — ENRICHMENT-corrected CRC "
          f"(fold>={FOLD})")
    print("=" * 76)
    for o in organs:
        c = crc[o]
        ke = KNOWN_EDGES.get(o)
        kstr = ""
        if ke:
            rec = [e for e in ke if tuple(e) in set(c["edges"])]
            kstr = f"  known {len(rec)}/{len(ke)}: {['->'.join(e) for e in rec]}"
        print(f"[{o:16s}] enriched-TFs={c['n_enriched']:2d} edges={len(c['edges']):3d} "
              f"autoreg={len(c['autoreg']):2d}{kstr}")
    diag = np.nanmean(np.diag(conf))
    offmask = ~np.eye(len(organs), dtype=bool)
    off = np.nanmean(conf[offmask])
    correct = sum(int(np.nanargmax(conf[i]) == i)
                  for i in range(len(organs)) if not np.all(np.isnan(conf[i])))
    print("-" * 76)
    print(f"cross-organ motif-concentration specificity (z): "
          f"mean diagonal {diag:+.2f} vs mean off-diagonal {off:+.2f}")
    print(f"diagonal-dominant organs (own TFs' motifs concentrate in own SEs): "
          f"{correct}/{len(organs)}")
    print("-" * 76)

    json.dump({"organs": organs, "confusion": conf.tolist(),
               "crc": crc, "fold": FOLD,
               "diag": float(diag), "off": float(off), "diag_correct": correct},
              open(WIRE_DIR / "crc_v2.json", "w"), indent=2)
    np.save(WIRE_DIR / "z_matrix.npy", z)
    json.dump({"regulators": regulators, "organs": organs},
              open(WIRE_DIR / "z_axes.json", "w"))


if __name__ == "__main__":
    main()
