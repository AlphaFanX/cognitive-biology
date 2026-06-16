#!/usr/bin/env python3
"""
Embryonic methylation readout for the EMBRYONIC kernel stratum.

The embryonic enhancers (LMR-only, no surviving H3K4me1/H3K27ac) have no adult
ChIP signal by definition, so their regulatory weight cannot come from the
acetylation/monomethyl marks that score the ADULT/FETAL strata. The proper
embryonic readout is the methylation *depth* at the developmental stage when the
enhancer was in use: how far BELOW the LMR ceiling the region sits at E12.5 /
E16.5 (more hypomethylated = more open / more active).

This module computes the mean per-region methylation (% methylated C = freqC)
over a set of LMR regions, streaming the GEO GSE111024 per-CpG WGBS tables:

    data/jadhav_mouse/E12.5_WGBS.txt.gz
    data/jadhav_mouse/E16.5_WGBS.txt.gz

File format (tab-separated, one CpG per line, leading header row):
    chr   base   freqC   freqT
    chr10 3000186 100.00 0.00

freqC is the methylation percentage at that CpG; the region value is the mean
freqC of all CpGs falling inside [start, end). TaggedEnhancer.derive_weight()
then converts that to a weight via (LMR_MAX - meth) / LMR_MAX.

The intersection is a per-chromosome binary search (regions are non-overlapping
LMR calls), so the WGBS table does NOT need to be position-sorted -- only the
region arrays are sorted. One pass per file; suitable for a one-time cache build.
"""

from __future__ import annotations

import gzip
import os
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

Region = Tuple[str, int, int]  # (chrom, start, end)


def _index_regions(regions: List[Region]):
    """Group regions per chromosome into parallel sorted (start, end, idx) arrays."""
    by_chrom: Dict[str, List[Tuple[int, int, int]]] = {}
    for i, (c, s, e) in enumerate(regions):
        by_chrom.setdefault(c, []).append((s, e, i))
    starts: Dict[str, List[int]] = {}
    ends: Dict[str, List[int]] = {}
    idxs: Dict[str, List[int]] = {}
    for c, lst in by_chrom.items():
        lst.sort()
        starts[c] = [x[0] for x in lst]
        ends[c] = [x[1] for x in lst]
        idxs[c] = [x[2] for x in lst]
    return starts, ends, idxs


def region_methylation(wgbs_path: str, regions: List[Region]) -> List[Optional[float]]:
    """Mean methylation (freqC %) per region from a per-CpG WGBS table.

    Returns a list aligned to `regions`; entries are None where no CpG in the
    file fell inside the region (sparse WGBS coverage).
    """
    starts, ends, idxs = _index_regions(regions)
    sums = [0.0] * len(regions)
    counts = [0] * len(regions)

    opener = gzip.open if wgbs_path.endswith(".gz") else open
    with opener(wgbs_path, "rt") as f:
        for line in f:
            # Fast reject: most CpGs are not in any LMR.
            tab1 = line.find("\t")
            if tab1 <= 0:
                continue
            c = line[:tab1]
            st = starts.get(c)
            if st is None:
                continue  # header ("chr") and chroms with no regions land here
            tab2 = line.find("\t", tab1 + 1)
            if tab2 < 0:
                continue
            try:
                pos = int(line[tab1 + 1:tab2])
            except ValueError:
                continue  # header row's "base" etc.
            j = bisect_right(st, pos) - 1
            if j < 0 or pos >= ends[c][j]:
                continue
            tab3 = line.find("\t", tab2 + 1)
            freq_str = line[tab2 + 1:tab3] if tab3 > 0 else line[tab2 + 1:]
            try:
                fc = float(freq_str)
            except ValueError:
                continue
            ri = idxs[c][j]
            sums[ri] += fc
            counts[ri] += 1

    return [sums[i] / counts[i] if counts[i] else None for i in range(len(regions))]


def attach_stage_methylation(enhancers, data_dir: str, wgbs_name: str,
                             attr: str) -> Dict[str, int]:
    """Set `attr` (e.g. 'methylation_adult') on every enhancer from one WGBS file.

    Generalizes attach_embryonic_methylation to any developmental stage so the
    ADULT and FETAL strata can be weighted by their own real methylation depth
    (Adult_WGBS / E16.5_WGBS), not by a placeholder or by histone marks. Only
    overwrites where the WGBS actually covers the region (m is not None), so
    uncovered LMRs keep whatever value they already had.
    """
    regions = [(e.chrom, e.start, e.end) for e in enhancers]
    stats = {"n": len(regions), "covered": 0}
    path = os.path.join(data_dir, wgbs_name)
    if not regions or not os.path.exists(path):
        return stats
    vals = region_methylation(path, regions)
    for e, m in zip(enhancers, vals):
        if m is not None:
            setattr(e, attr, m)
    stats["covered"] = sum(1 for m in vals if m is not None)
    return stats


def attach_embryonic_methylation(enhancers, data_dir: str,
                                 e12_name: str = "E12.5_WGBS.txt.gz",
                                 e16_name: str = "E16.5_WGBS.txt.gz") -> Dict[str, int]:
    """Set methylation_e12 / methylation_e16 on the given embryonic enhancers.

    `enhancers` is the EMBRYONIC subset (caller filters by stratum so this never
    perturbs ADULT/FETAL strata). Returns a small stats dict for reporting.
    """
    regions = [(e.chrom, e.start, e.end) for e in enhancers]
    stats = {"n": len(regions), "e12_covered": 0, "e16_covered": 0, "any_covered": 0}
    if not regions:
        return stats

    e12_path = os.path.join(data_dir, e12_name)
    e16_path = os.path.join(data_dir, e16_name)

    if os.path.exists(e12_path):
        m12 = region_methylation(e12_path, regions)
        for e, m in zip(enhancers, m12):
            e.methylation_e12 = m
        stats["e12_covered"] = sum(1 for m in m12 if m is not None)

    if os.path.exists(e16_path):
        m16 = region_methylation(e16_path, regions)
        for e, m in zip(enhancers, m16):
            e.methylation_e16 = m
        stats["e16_covered"] = sum(1 for m in m16 if m is not None)

    stats["any_covered"] = sum(
        1 for e in enhancers
        if e.methylation_e12 is not None or e.methylation_e16 is not None
    )
    return stats


def _selftest(data_dir: str = "data/jadhav_mouse") -> None:
    """Quick correctness check on the small *partial* table (no full pass)."""
    partial = os.path.join(data_dir, "E12.5_WGBS_partial.txt")
    bed = os.path.join(data_dir, "Adult_LMRs.bed")
    if not (os.path.exists(partial) and os.path.exists(bed)):
        print("selftest: partial WGBS or LMR bed missing, skipping")
        return
    regions: List[Region] = []
    for line in open(bed):
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) >= 3:
            regions.append((p[0], int(p[1]), int(p[2])))
    vals = region_methylation(partial, regions)
    covered = [v for v in vals if v is not None]
    print(f"selftest on {os.path.basename(partial)}: {len(regions):,} LMRs, "
          f"{len(covered):,} covered by partial table")
    if covered:
        import statistics
        print(f"  methylation%% over covered LMRs: "
              f"min={min(covered):.1f} mean={statistics.mean(covered):.1f} "
              f"max={max(covered):.1f}")


if __name__ == "__main__":
    _selftest()
