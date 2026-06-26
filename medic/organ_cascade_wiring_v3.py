"""
Organ Cascade Stage 2 v3 — homotypic CLUSTER-density vs RANDOM-GENOME background.

v1 (presence) and v2 (enrichment vs pooled SEs) failed on specificity because
SEs are motif-saturated (71%) and the SE pool is a biased background. v3 fixes
both:
  - measure motif SITE COUNT per SE per TF (connected runs above threshold, both
    strands) = homotypic cluster density (master TFs mark targets with CLUSTERS),
    not mere presence;
  - compare to a GC/length-matched RANDOM-GENOME background (not the SE pool).

Edge TF_a -> TF_b in organ o iff TF_a's cluster density in TF_b's organ-o SEs is
>= FOLD over random-genome background AND TF_a is a head TF of organ o.
Specificity = cross-organ z of cluster density (own TFs cluster in own SEs?).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from medic.organ_cascade_wiring import (
    WIRE_DIR, organ_head_tfs, KNOWN_EDGES, parse_jaspar, onehot,
    fetch_seq, _save_seq_cache, _load_seq_cache,
)
from medic.organ_cascade import load_tfs

FOLD = 2.0
TOP_K = 18
N_BG = 500
BG_LEN = 500
COUNT_CACHE = WIRE_DIR / "se_counts.npz"
BG_CACHE = WIRE_DIR / "bg_counts.npz"

HG38 = {  # chrom: length (bp)
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895,
}


def pwm_count(oh: np.ndarray, pwm: np.ndarray, thresh: float = 0.85) -> int:
    """Number of distinct motif SITES (connected runs above rel-score) on both
    strands = homotypic cluster size."""
    w = pwm.shape[0]
    if oh.shape[0] < w:
        return 0
    smax = pwm.max(axis=1).sum(); smin = pwm.min(axis=1).sum()
    rng = (smax - smin) + 1e-9
    total = 0
    for strand_oh in (oh, oh[::-1, ::-1]):
        win = np.lib.stride_tricks.sliding_window_view(strand_oh, (w, 4)).reshape(-1, w, 4)
        rel = (np.tensordot(win, pwm, axes=([1, 2], [0, 1])) - smin) / rng
        passes = rel >= thresh
        if passes.any():
            total += int(np.sum(np.diff(passes.astype(np.int8)) == 1) + (1 if passes[0] else 0))
    return total


def count_matrix(regions, regulators, pwms, seq_lookup, thresh=0.85):
    """regions: list of (key, oh). returns array [n_regions x n_reg] of site counts."""
    M = np.zeros((len(regions), len(regulators)), dtype=np.float32)
    for ri, (key, oh) in enumerate(regions):
        if oh is None:
            continue
        for ti, tf in enumerate(regulators):
            c = 0
            for p in pwms[tf]:
                c += pwm_count(oh, p, thresh)
            M[ri, ti] = c
    return M


def random_bg_regions(n, length, seed=1):
    rng = np.random.default_rng(seed)
    cache = _load_seq_cache()
    chroms = list(HG38.keys())
    weights = np.array([HG38[c] for c in chroms], float); weights /= weights.sum()
    out = []
    tries = 0
    while len(out) < n and tries < n * 6:
        tries += 1
        c = chroms[rng.choice(len(chroms), p=weights)]
        start = int(rng.integers(3_000_000, HG38[c] - 3_000_000))
        key = f"{c}:{start+1}-{start+length}"
        seq = cache.get(key)
        if seq is None:
            seq = fetch_seq(c, start, start + length)
            if len(out) % 25 == 0:
                _save_seq_cache()
        if seq and seq.upper().count("N") / max(len(seq), 1) < 0.1:
            out.append((key, onehot(seq)))
    _save_seq_cache()
    return out


def main():
    heads = organ_head_tfs(top_k=TOP_K)
    organ_targets = {o: {tf: [f"{c}:{s}-{e}" for (c, s, e) in regs]
                         for tf, regs in d.items()}
                     for o, d in json.load(open(WIRE_DIR / "organ_targets.json")).items()}
    organs = list(organ_targets.keys())
    regulators = sorted({t for o in organs for t in heads[o]})
    pwms = parse_jaspar(restrict=set(regulators))
    regulators = [t for t in regulators if t in pwms]

    seq_cache = _load_seq_cache()

    def oh_for(key):
        seq = seq_cache.get(key)
        return onehot(seq) if seq else None

    # SE regions (unique) + counts
    se_keys = sorted({k for o in organs for tf in organ_targets[o]
                      for k in organ_targets[o][tf]})
    if COUNT_CACHE.exists():
        d = np.load(COUNT_CACHE, allow_pickle=True)
        se_M = d["M"]; se_keys = list(d["keys"]); reg_saved = list(d["regs"])
        assert reg_saved == regulators, "regulator set changed; delete se_counts.npz"
        print(f"loaded SE count cache {se_M.shape}")
    else:
        regions = [(k, oh_for(k)) for k in se_keys]
        print(f"scanning {len(regions)} SEs x {len(regulators)} TFs for site counts...")
        se_M = count_matrix(regions, regulators, pwms, seq_cache)
        np.savez(COUNT_CACHE, M=se_M, keys=np.array(se_keys, object),
                 regs=np.array(regulators, object))

    # random-genome background counts
    if BG_CACHE.exists():
        d = np.load(BG_CACHE, allow_pickle=True)
        bg_M = d["M"]
        print(f"loaded BG count cache {bg_M.shape}")
    else:
        print(f"fetching {N_BG} random {BG_LEN}bp background regions...")
        bg_regions = random_bg_regions(N_BG, BG_LEN)
        print(f"scanning {len(bg_regions)} background regions...")
        bg_M = count_matrix(bg_regions, regulators, pwms, seq_cache)
        np.savez(BG_CACHE, M=bg_M)

    tfi = {t: i for i, t in enumerate(regulators)}
    bg_density = bg_M.mean(axis=0) + 1e-6  # per-TF mean site count in random genome
    se_idx = {k: i for i, k in enumerate(se_keys)}

    # per (TF, organ) cluster density = mean site count over organ's SEs
    dens = np.zeros((len(regulators), len(organs)))
    for j, o in enumerate(organs):
        keys = sorted({k for tf in organ_targets[o] for k in organ_targets[o][tf]})
        rows = [se_idx[k] for k in keys]
        dens[:, j] = se_M[rows].mean(axis=0)
    enrich = dens / bg_density[:, None]
    mu = dens.mean(axis=1, keepdims=True); sd = dens.std(axis=1, keepdims=True) + 1e-9
    z = (dens - mu) / sd

    # cross-organ specificity confusion (z of cluster density)
    conf = np.full((len(organs), len(organs)), np.nan)
    for i, oi in enumerate(organs):
        htfs = [t for t in heads[oi] if t in tfi]
        if htfs:
            for j in range(len(organs)):
                conf[i, j] = np.mean([z[tfi[t], j] for t in htfs])

    # enriched edges
    crc = {}
    for j, o in enumerate(organs):
        enr = {t for t in heads[o] if t in tfi and enrich[tfi[t], j] >= FOLD}
        edges = set(); autoreg = set()
        for tf_b, keys in organ_targets[o].items():
            for k in keys:
                ri = se_idx[k]
                for tf_a in enr:
                    if se_M[ri, tfi[tf_a]] > 0:
                        edges.add((tf_a, tf_b))
                        if tf_a == tf_b:
                            autoreg.add(tf_a)
        crc[o] = {"nodes": [t for t in heads[o] if t in tfi],
                  "edges": sorted(edges), "autoreg": sorted(autoreg),
                  "n_enriched": len(enr), "enriched": sorted(enr)}

    # report
    print("=" * 78)
    print(f"ORGAN CASCADE Stage 2 v3 — homotypic CLUSTER density vs RANDOM genome "
          f"(fold>={FOLD})")
    print("=" * 78)
    print(f"background: {bg_M.shape[0]} random {BG_LEN}bp regions; "
          f"mean bg site-count/TF = {bg_density.mean():.2f}")
    for o in organs:
        c = crc[o]
        ke = KNOWN_EDGES.get(o)
        kstr = ""
        if ke:
            rec = [e for e in ke if tuple(e) in set(c["edges"])]
            kstr = f"  known {len(rec)}/{len(ke)}: {['->'.join(e) for e in rec]}"
        print(f"[{o:16s}] enrichedTFs={c['n_enriched']:2d} {c['enriched'][:6]}")
        print(f"{'':18s} edges={len(c['edges']):3d} autoreg={len(c['autoreg']):2d}{kstr}")
    diag = np.nanmean(np.diag(conf))
    off = np.nanmean(conf[~np.eye(len(organs), dtype=bool)])
    correct = sum(int(np.nanargmax(conf[i]) == i)
                  for i in range(len(organs)) if not np.all(np.isnan(conf[i])))
    print("-" * 78)
    print(f"cross-organ cluster-density specificity (z): diag {diag:+.2f} vs off {off:+.2f}")
    print(f"diagonal-dominant organs: {correct}/{len(organs)}")
    print("-" * 78)

    json.dump({"organs": organs, "confusion": conf.tolist(), "crc": crc,
               "fold": FOLD, "diag": float(diag), "off": float(off),
               "diag_correct": int(correct),
               "bg_mean": float(bg_density.mean())},
              open(WIRE_DIR / "crc_v3.json", "w"), indent=2)
    print("saved", WIRE_DIR / "crc_v3.json")


if __name__ == "__main__":
    main()
