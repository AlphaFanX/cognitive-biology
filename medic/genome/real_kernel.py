#!/usr/bin/env python3
"""
Build a *real* ZygoteKernel from the Jadhav mouse data (GEO GSE111024).

This is the production counterpart to ZygoteKernel.demo(): it stratifies the
~45k adult LMRs by their surviving histone marks (the mark-based Jadhav rule,
same as scripts/stratify_by_marks.py) into EMBRYONIC / FETAL / ADULT, then
caches the (chrom, start, end, stratum) assignment so later loads are instant.

    adult LMR + H3K27ac            -> ADULT     (active now)
    adult LMR + H3K4me1, no K27ac  -> FETAL      (primed)
    adult LMR + neither            -> EMBRYONIC  (LMR-only memory)

Used by medic/tissue/genomic_nca.py to gate the shared genomic kernel with the
real mark-resolved strata instead of the 1-per-stratum synthetic demo kernel.

Run once to build the cache (needs pybigtools + the GEO bigwigs):
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.genome.real_kernel
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

try:
    from .zygote_kernel import ZygoteKernel, TaggedEnhancer, KernelStratum
    from .embryonic_methylation import (attach_embryonic_methylation,
                                        attach_stage_methylation)
except ImportError:  # pragma: no cover
    from medic.genome.zygote_kernel import ZygoteKernel, TaggedEnhancer, KernelStratum
    from medic.genome.embryonic_methylation import (attach_embryonic_methylation,
                                                    attach_stage_methylation)

DATA_DIR = "data/jadhav_mouse"
CACHE_PATH = os.path.join(DATA_DIR, "strata_marks_cache.json")
MARK_THRESHOLD = 1.0  # fold-enrichment over background to call a mark "present"


def _load_lmrs(path: str):
    out = []
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 3:
            continue
        out.append((p[0], int(p[1]), int(p[2])))
    return out


def _build_from_bigwigs(data_dir: str) -> List[TaggedEnhancer]:
    """Score every adult LMR against the H3K4me1/H3K27ac bigwigs (slow path)."""
    import numpy as np
    import pybigtools

    k4 = pybigtools.open(os.path.join(data_dir, "Adult_H3K4me1.bw"))
    k27 = pybigtools.open(os.path.join(data_dir, "Adult_H3K27ac.bw"))
    lmrs = _load_lmrs(os.path.join(data_dir, "Adult_LMRs.bed"))

    def mean_signal(bw, c, s, e):
        try:
            v = np.asarray(bw.values(c, s, e), dtype=float)
            v = v[~np.isnan(v)]
            return float(v.mean()) if v.size else 0.0
        except Exception:
            return 0.0

    tagged: List[TaggedEnhancer] = []
    for c, s, e in lmrs:
        sig_k4 = mean_signal(k4, c, s, e)
        sig_k27 = mean_signal(k27, c, s, e)
        tagged.append(TaggedEnhancer(
            chrom=c, start=s, end=e, methylation_adult=30.0,
            has_h3k4me1=(sig_k4 > MARK_THRESHOLD),
            has_h3k27ac=(sig_k27 > MARK_THRESHOLD),
            # Retain the CONTINUOUS measured intensities (previously discarded).
            h3k4me1_signal=sig_k4,
            h3k27ac_signal=sig_k27,
        ))
    return tagged


def _derive_and_normalize_weights(kernel: ZygoteKernel) -> None:
    """Derive each enhancer's weight from measured signal, then min-max normalize
    WITHIN each stratum to [0,1] (strata use different measured sources, so a
    global scale would not be comparable)."""
    import numpy as np
    for e in kernel.enhancers:
        e.derive_weight()
    by_stratum = {}
    for e in kernel.enhancers:
        if e.weight is not None and e.stratum is not None:
            by_stratum.setdefault(e.stratum, []).append(e.weight)
    norm = {}
    for strat, ws in by_stratum.items():
        lo, hi = float(min(ws)), float(max(ws))
        norm[strat] = (lo, hi if hi > lo else lo + 1.0)
    for e in kernel.enhancers:
        if e.weight is not None and e.stratum in norm:
            lo, hi = norm[e.stratum]
            e.weight = (e.weight - lo) / (hi - lo)


CACHE_VERSION = 4  # v4: ALL strata weighted by real WGBS methylation depth;
                   #     histone marks no longer feed weights (moved to ABC/SEdb
                   #     LoRA side -- two-MLP separation, 2026-06-04).


def _attach_real_methylation(kernel: ZygoteKernel, data_dir: str) -> dict:
    """Populate real per-region methylation on EVERY retained LMR, AFTER
    mark-based stratification (so strata stay authoritative), so the weight of
    each stratum derives from its OWN measured methylation depth -- never from
    a histone mark:

        ADULT     <- Adult_WGBS.txt.gz   (methylation_adult, overwrites the
                     archival placeholder with the true LMR depth)
        FETAL     <- E16.5_WGBS.txt.gz   (methylation_e16; mid-stage depth)
        EMBRYONIC <- E12.5 + E16.5 WGBS  (methylation_e12 / _e16; earliest depth)

    Histone-mark ChIP (H3K4me1/H3K27ac) is intentionally untouched here -- it
    classified the strata and is kept only for the adult-overlap validation
    against ABC/SEdb. Returns a small stats dict for reporting.
    """
    retained = [e for e in kernel.enhancers if e.stratum is not None]
    embryonic = [e for e in kernel.enhancers if e.stratum == KernelStratum.EMBRYONIC]
    fetal = [e for e in kernel.enhancers if e.stratum == KernelStratum.FETAL]

    # Adult depth for every retained LMR (real value replaces the 30.0 gate).
    a = attach_stage_methylation(retained, data_dir, "Adult_WGBS.txt.gz",
                                 "methylation_adult")
    # Embryonic stratum: E12.5 + E16.5 trajectory.
    emb = attach_embryonic_methylation(embryonic, data_dir)
    # Fetal stratum: its own E16.5 depth.
    f = attach_stage_methylation(fetal, data_dir, "E16.5_WGBS.txt.gz",
                                 "methylation_e16")

    print(f"  real methylation attached: adult WGBS {a['covered']:,}/{a['n']:,} "
          f"retained LMRs; embryonic {emb['any_covered']:,}/{emb['n']:,} "
          f"(E12.5={emb['e12_covered']:,}, E16.5={emb['e16_covered']:,}); "
          f"fetal {f['covered']:,}/{f['n']:,}")
    return {"adult": a, "embryonic": emb, "fetal": f}


def _write_cache(kernel: ZygoteKernel, path: str) -> None:
    rows = [[e.chrom, e.start, e.end,
             int(e.stratum) if e.stratum is not None else -1,
             e.weight, e.h3k4me1_signal, e.h3k27ac_signal]
            for e in kernel.enhancers]
    with open(path, "w") as f:
        json.dump({"version": CACHE_VERSION, "threshold": MARK_THRESHOLD,
                   "enhancers": rows}, f)


def _read_cache(path: str) -> Optional[List[TaggedEnhancer]]:
    with open(path) as f:
        data = json.load(f)
    # v1 caches lack weights -> signal a rebuild so weights get derived.
    if data.get("version", 1) < CACHE_VERSION:
        return None
    tagged: List[TaggedEnhancer] = []
    for chrom, start, end, strat, weight, k4, k27 in data["enhancers"]:
        te = TaggedEnhancer(chrom=chrom, start=start, end=end, methylation_adult=30.0,
                            h3k4me1_signal=k4, h3k27ac_signal=k27)
        te.stratum = None if strat < 0 else KernelStratum(strat)
        te.weight = weight
        tagged.append(te)
    return tagged


def load_real_zygote_kernel(
    data_dir: str = DATA_DIR,
    cache_path: str = CACHE_PATH,
    prefer_cache: bool = True,
) -> Optional[ZygoteKernel]:
    """Return a mark-resolved ZygoteKernel, or None if the data is unavailable.

    Uses the cache when present; otherwise scores the bigwigs and writes one.
    Strata are preserved verbatim from the cache (restratify=False), so the
    mark-based assignment is authoritative.
    """
    # Fast path: cached strata.
    if prefer_cache and os.path.exists(cache_path):
        try:
            tagged = _read_cache(cache_path)
            if tagged is not None:  # None => stale (pre-weight) cache, rebuild
                return ZygoteKernel(tagged, restratify=False)
        except Exception:
            pass  # fall through to rebuild

    # Slow path: rebuild from bigwigs (requires pybigtools + GEO files).
    bed = os.path.join(data_dir, "Adult_LMRs.bed")
    if not os.path.exists(bed):
        return None
    try:
        tagged = _build_from_bigwigs(data_dir)
    except Exception:
        return None

    kernel = ZygoteKernel(tagged, restratify=True)  # mark-based assign_stratum
    _attach_real_methylation(kernel, data_dir)       # real WGBS depth -> all strata
    _derive_and_normalize_weights(kernel)            # methylation-only weights
    try:
        _write_cache(kernel, cache_path)
    except Exception:
        pass
    return kernel


def main():
    import time
    t = time.time()
    # A rebuild happens when the cache is absent OR a stale (older-version) cache
    # is present -- both route through the bigwig path inside the loader.
    rebuilt = True
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as _f:
                rebuilt = json.load(_f).get("version", 1) < CACHE_VERSION
        except Exception:
            rebuilt = True
    kernel = load_real_zygote_kernel(prefer_cache=True)
    if kernel is None:
        print("Real kernel unavailable (missing Jadhav data or pybigtools).")
        return
    c = kernel.stratum_counts()
    n = sum(c.values())
    print(f"{'Rebuilt from bigwigs' if rebuilt else 'Loaded from cache'} "
          f"in {time.time()-t:.1f}s  ({CACHE_PATH})")
    print(f"Mark-resolved strata (n={n:,}):")
    print(f"  EMBRYONIC (LMR only)         : {c[KernelStratum.EMBRYONIC]:>6,} "
          f"({100*c[KernelStratum.EMBRYONIC]/n:.1f}%)")
    print(f"  FETAL     (LMR+H3K4me1)      : {c[KernelStratum.FETAL]:>6,} "
          f"({100*c[KernelStratum.FETAL]/n:.1f}%)")
    print(f"  ADULT     (LMR+H3K4me1+K27ac): {c[KernelStratum.ADULT]:>6,} "
          f"({100*c[KernelStratum.ADULT]/n:.1f}%)")
    # Derived-weight summary (the answer to "can we derive kernel weights?")
    import numpy as np
    print("\nDERIVED regulatory weights (methylation depth only, not fitted; "
          "marks excluded -> ABC/SEdb LoRA side), per stratum:")
    for strat in (KernelStratum.ADULT, KernelStratum.FETAL, KernelStratum.EMBRYONIC):
        ws = [e.weight for e in kernel.enhancers
              if e.stratum == strat and e.weight is not None]
        src = {"ADULT": "Adult WGBS methylation depth",
               "FETAL": "E16.5 WGBS methylation depth",
               "EMBRYONIC": "E12.5/E16.5 WGBS methylation depth"}[strat.name]
        if ws:
            a = np.asarray(ws)
            print(f"  {strat.name:<10} n={len(ws):>6,}  weight mean={a.mean():.3f} "
                  f"sd={a.std():.3f}  [{a.min():.2f}, {a.max():.2f}]  <- {src}")
        else:
            print(f"  {strat.name:<10} no weights")
    nz = sum(1 for e in kernel.enhancers if e.weight is not None)
    print(f"  -> {nz:,}/{n:,} enhancers carry a data-derived weight "
          f"(was 0 before: kernel was categorical only)")

    print("\nReverse-order reactivation as telomeres shorten:")
    for telo in [10000, 4000, 1000]:
        kernel.set_from_telomere_length(telo)
        live = sorted({s.name for s in (e.stratum for e in kernel.active_enhancers())
                       if s is not None})
        print(f"  telo={telo:>6} bp | PRC2={kernel.prc2_level:.2f} | "
              f"active={len(kernel.active_enhancers()):,} | {live}")


if __name__ == "__main__":
    main()
