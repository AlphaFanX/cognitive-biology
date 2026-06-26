"""
Organ Cascade — Stage 2: cascade WIRING (TF -> SE -> TF core regulatory circuit)
================================================================================

Stage 1 (`organ_cascade.py`) showed each organ's super-enhancers drive a distinct
set of master TFs (the "head" membership). Stage 2 draws the directed edges that
make it a CASCADE / auto-regulating core regulatory circuit (Young-lab CRC,
Davidson GRN kernel):

    edge  TF_a --> TF_b   iff   TF_a's motif occurs in a super-enhancer that
                                ABC-assigns to TF_b, in that organ's cell types.

A self-edge (TF binds its own SE) = autoregulation; reciprocal edges = the
interlocking CRC clique. This is Miles's "TF activates an SE, which makes a new
TF" made concrete and falsifiable.

Pieces (all light — no genome download):
  - ABC enhancer coords + targets (local, Nasser 2021)        [SE -> TF edge]
  - JASPAR2024 CORE vertebrate PWMs (data/JASPAR2024_*.txt)   [TF -> motif]
  - Ensembl REST region endpoint (fetch only SE sequences)    [SE sequence]
  - numpy PWM scanner (MOODS/Bio unavailable)                 [TF -> SE edge]

ABC/SE remains the PLACEHOLDER readout of the zygote kernel + LLM-MLP, per
agreement; real networks later.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

from medic.organ_cascade import (
    ROOT, ABC_FILE, MIN_ABC, ORGAN_BIOSAMPLES, KNOWN_KERNEL,
    load_tfs, OUT_DIR,
)

CA = r"C:\Users\jacobsme\netskope-ca-bundle.pem"
JASPAR_FILE = ROOT / "data" / "JASPAR2024_CORE_vertebrates_nr_pfms.txt"
COORD_CACHE = ROOT / "data" / "abc_reduced_cascade_coords.parquet"
SEQ_CACHE = ROOT / "data" / "se_sequences.json"
WIRE_DIR = ROOT / "data" / "organ_cascade_wiring"

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}


# --------------------------------------------------------------------------- #
# JASPAR PWMs
# --------------------------------------------------------------------------- #
def parse_jaspar(restrict: set | None = None) -> Dict[str, List[np.ndarray]]:
    """Parse JASPAR raw format -> {TF_symbol: [log-odds PWM (w x 4), ...]}.

    Handles ::dimers (both partners get the matrix). Symbols uppercased.
    """
    text = open(JASPAR_FILE).read().splitlines()
    pwms: Dict[str, List[np.ndarray]] = {}
    i = 0
    while i < len(text):
        line = text[i]
        if line.startswith(">"):
            parts = line[1:].split()
            name = parts[1] if len(parts) > 1 else parts[0]
            rows = []
            for k in range(1, 5):
                row = text[i + k]
                lb, rb = row.find("["), row.find("]")
                nums = row[lb + 1:rb].split()
                rows.append([float(x) for x in nums])
            i += 5
            counts = np.array(rows, dtype=float)  # 4 x w (A,C,G,T)
            # ppm with background-distributed pseudocount, then log-odds vs 0.25
            colsum = counts.sum(axis=0, keepdims=True)
            ppm = (counts + 0.25 * 0.8) / (colsum + 0.8)
            lo = np.log2(ppm / 0.25).T  # w x 4
            syms = name.upper().replace("(VAR.2)", "").replace("(VAR.3)", "")
            for sym in syms.split("::"):
                sym = sym.strip()
                if restrict is not None and sym not in restrict:
                    continue
                pwms.setdefault(sym, []).append(lo)
        else:
            i += 1
    return pwms


def _revcomp_onehot(oh: np.ndarray) -> np.ndarray:
    # reverse positions and complement A<->T (0<->3), C<->G (1<->2)
    return oh[::-1, ::-1]


def onehot(seq: str) -> np.ndarray:
    oh = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        j = B2I.get(b)
        if j is not None:
            oh[i, j] = 1.0
    return oh


def pwm_hit(oh: np.ndarray, pwm: np.ndarray, thresh: float = 0.85) -> bool:
    """True if PWM matches anywhere on either strand at relative-score>=thresh."""
    w = pwm.shape[0]
    if oh.shape[0] < w:
        return False
    smax = pwm.max(axis=1).sum()
    smin = pwm.min(axis=1).sum()
    rng = (smax - smin) + 1e-9
    for strand_oh in (oh, _revcomp_onehot(oh)):
        win = np.lib.stride_tricks.sliding_window_view(strand_oh, (w, 4))
        win = win.reshape(-1, w, 4)
        scores = np.tensordot(win, pwm, axes=([1, 2], [0, 1]))
        if scores.size and ((scores.max() - smin) / rng) >= thresh:
            return True
    return False


def binds(seq_oh: np.ndarray, pwms: List[np.ndarray], thresh: float = 0.85) -> bool:
    return any(pwm_hit(seq_oh, p, thresh) for p in pwms)


# --------------------------------------------------------------------------- #
# ABC coords + sequence fetch
# --------------------------------------------------------------------------- #
def build_coord_cache(force: bool = False) -> pd.DataFrame:
    if COORD_CACHE.exists() and not force:
        return pd.read_parquet(COORD_CACHE)
    keep = set()
    for bs in ORGAN_BIOSAMPLES.values():
        keep.update(bs)
    cols = ["chr", "start", "end", "TargetGene", "CellType", "activity_base", "ABC.Score"]
    parts = []
    for chunk in pd.read_csv(ABC_FILE, sep="\t", usecols=cols, chunksize=500_000):
        m = chunk["CellType"].isin(keep) & (chunk["ABC.Score"] >= MIN_ABC)
        if m.any():
            parts.append(chunk.loc[m, cols])
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(COORD_CACHE)
    return df


_seq_cache: Dict[str, str] | None = None


def _load_seq_cache() -> Dict[str, str]:
    global _seq_cache
    if _seq_cache is None:
        _seq_cache = json.load(open(SEQ_CACHE)) if SEQ_CACHE.exists() else {}
    return _seq_cache


def _save_seq_cache():
    if _seq_cache is not None:
        json.dump(_seq_cache, open(SEQ_CACHE, "w"))


def fetch_seq(chrom: str, start: int, end: int, retries: int = 4) -> str | None:
    cache = _load_seq_cache()
    key = f"{chrom}:{start}-{end}"
    if key in cache:
        return cache[key] or None
    c = chrom[3:] if chrom.startswith("chr") else chrom
    url = f"https://rest.ensembl.org/sequence/region/human/{c}:{start+1}..{end}:1"
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"content-type": "text/plain"},
                             verify=CA, timeout=40)
            if r.status_code == 200:
                seq = r.text.strip()
                cache[key] = seq
                return seq
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 1.0)) + 0.5)
                continue
            time.sleep(0.5)
        except Exception:
            time.sleep(1.0)
    cache[key] = ""  # negative-cache failures
    return None


# --------------------------------------------------------------------------- #
# Build per-organ super-enhancer set targeting head TFs
# --------------------------------------------------------------------------- #
def organ_head_tfs(top_k: int = 18) -> Dict[str, List[str]]:
    """Read Stage-1 head tables for the head TF list per organ."""
    heads = {}
    for organ in ORGAN_BIOSAMPLES:
        f = OUT_DIR / f"head_{organ}.csv"
        if f.exists():
            heads[organ] = pd.read_csv(f)["tf"].head(top_k).tolist()
    return heads


def organ_se_targets(df: pd.DataFrame, organ: str, head_tfs: List[str],
                     max_enh_per_tf: int = 5) -> Dict[str, List[Tuple[str, int, int, float]]]:
    """For each head TF_b, the top super-enhancer constituents (by activity)
    that ABC-assigns to it in this organ's biosamples."""
    bs = ORGAN_BIOSAMPLES[organ]
    sub = df[(df["CellType"].isin(bs)) & (df["TargetGene"].isin(head_tfs))]
    out: Dict[str, List[Tuple[str, int, int, float]]] = {}
    for tf in head_tfs:
        rows = sub[sub["TargetGene"] == tf]
        if rows.empty:
            out[tf] = []
            continue
        # dedup enhancer coords, keep max activity, take top N
        rows = (rows.groupby(["chr", "start", "end"])["activity_base"]
                    .max().reset_index()
                    .sort_values("activity_base", ascending=False)
                    .head(max_enh_per_tf))
        out[tf] = [(r["chr"], int(r["start"]), int(r["end"]), float(r["activity_base"]))
                   for _, r in rows.iterrows()]
    return out


# --------------------------------------------------------------------------- #
# Build the per-organ CRC (TF -> SE -> TF) and run the tests
# --------------------------------------------------------------------------- #
DEFAULT_ORGANS = ["heart", "liver", "skeletal_muscle", "pancreas",
                  "brain", "thyroid", "intestine", "adrenal"]

# A few canonical CRC cross-regulatory edges to look for (literature answer key)
KNOWN_EDGES = {
    "heart": [("GATA4", "NKX2-5"), ("NKX2-5", "GATA4"), ("TBX5", "NKX2-5"),
              ("GATA4", "TBX5"), ("NKX2-5", "NKX2-5"), ("GATA4", "GATA4")],
    "liver": [("FOXA2", "HNF4A"), ("HNF4A", "HNF4A"), ("FOXA1", "FOXA1"),
              ("HNF4A", "FOXA1"), ("CEBPA", "CEBPA")],
    "skeletal_muscle": [("MYOD1", "MYOG"), ("MYOD1", "MYOD1"), ("MYF5", "MYOD1"),
                        ("MYOG", "MYOG"), ("MYOD1", "MYF5")],
    "pancreas": [("PDX1", "PDX1"), ("PTF1A", "PTF1A"), ("PDX1", "NKX6-1"),
                 ("FOXA2", "PDX1")],
}


def build_crc(organs=DEFAULT_ORGANS, top_k: int = 18, max_enh: int = 5,
              thresh: float = 0.85, verbose: bool = True) -> dict:
    tfs = load_tfs()
    heads = organ_head_tfs(top_k=top_k)
    organs = [o for o in organs if o in heads]
    # union of head TFs across organs = regulator motif set we need
    reg_union = sorted(set(t for o in organs for t in heads[o]))
    pwms = parse_jaspar(restrict=set(reg_union))
    df = build_coord_cache()

    # gather SE targets + fetch all sequences (cached)
    organ_targets = {o: organ_se_targets(df, o, heads[o], max_enh) for o in organs}
    all_regions = {(c, s, e) for o in organs for tf in organ_targets[o]
                   for (c, s, e, a) in organ_targets[o][tf]}
    if verbose:
        print(f"organs={len(organs)} regulators(with PWM)="
              f"{sum(1 for t in reg_union if t in pwms)}/{len(reg_union)} "
              f"unique SE regions={len(all_regions)}")
    seq_oh: Dict[Tuple[str, int, int], np.ndarray] = {}
    for n, (c, s, e) in enumerate(sorted(all_regions)):
        seq = fetch_seq(c, s, min(e, s + 3000))
        if seq:
            seq_oh[(c, s, e)] = onehot(seq)
        if n % 25 == 0:
            _save_seq_cache()
            if verbose:
                print(f"  fetched {n+1}/{len(all_regions)}")
    _save_seq_cache()

    # cache which regulators bind which SE region (region -> set of TFs)
    region_binders: Dict[Tuple[str, int, int], set] = {}
    regs_with_pwm = [t for t in reg_union if t in pwms]
    for reg in all_regions:
        oh = seq_oh.get(reg)
        if oh is None:
            region_binders[reg] = set()
            continue
        region_binders[reg] = {tf for tf in regs_with_pwm
                               if binds(oh, pwms[tf], thresh)}

    # build directed CRC edges per organ: TF_a -> TF_b if TF_a binds an SE of TF_b
    crc = {}
    for o in organs:
        nodes = [t for t in heads[o] if t in pwms]  # need a motif to be a regulator
        edges = set()
        autoreg = set()
        for tf_b, ses in organ_targets[o].items():
            for (c, s, e, a) in ses:
                binders = region_binders.get((c, s, e), set())
                for tf_a in binders:
                    if tf_a in heads[o]:
                        edges.add((tf_a, tf_b))
                        if tf_a == tf_b:
                            autoreg.add(tf_a)
        crc[o] = {"nodes": nodes, "edges": sorted(edges), "autoreg": sorted(autoreg),
                  "targets": {tf: [(c, s, e) for (c, s, e, a) in v]
                              for tf, v in organ_targets[o].items()}}
    return {"organs": organs, "heads": heads, "pwms_present": regs_with_pwm,
            "organ_targets": organ_targets, "region_binders": region_binders,
            "crc": crc, "thresh": thresh}


def analyze(res: dict, n_null: int = 200, seed: int = 0) -> dict:
    organs = res["organs"]
    heads = res["heads"]
    crc = res["crc"]
    ot = res["organ_targets"]
    rb = res["region_binders"]
    pwm_pool = res["pwms_present"]
    rng = np.random.default_rng(seed)
    tfs = load_tfs()
    all_pwm_tfs = sorted(set(parse_jaspar(restrict=tfs).keys()))

    # 1. autoregulation: fraction of head TFs that bind at least one of their own SEs
    auto = {}
    for o in organs:
        nodes = crc[o]["nodes"]
        tfs_with_own_se = [t for t in nodes if ot[o].get(t)]
        if not tfs_with_own_se:
            auto[o] = float("nan"); continue
        auto[o] = len(crc[o]["autoreg"]) / len(tfs_with_own_se)

    # null for autoregulation: random TFs (with PWM) scanned against the SAME SEs
    pwms_all = parse_jaspar(restrict=set(all_pwm_tfs))
    auto_null = _auto_null(organs, crc, ot, all_pwm_tfs, pwms_all,
                           res["thresh"], n_null, rng)

    # 2. CRC graph stats
    stats = {}
    for o in organs:
        n = len(crc[o]["nodes"])
        E = set(crc[o]["edges"])
        possible = n * n if n else 1
        recip = sum(1 for (a, b) in E if a != b and (b, a) in E)
        ne = len(E)
        stats[o] = {
            "n_nodes": n, "n_edges": ne,
            "density": ne / possible,
            "autoreg_frac": auto[o],
            "autoreg_null": auto_null[o],
            "reciprocity": (recip / ne) if ne else 0.0,
        }

    # 3. cross-organ specificity confusion: organ i's head-TF motifs vs organ j's SEs
    conf = np.zeros((len(organs), len(organs)))
    for i, oi in enumerate(organs):
        motif_tfs = [t for t in heads[oi] if t in res["pwms_present"]]
        for j, oj in enumerate(organs):
            ses = {(c, s, e) for tf in ot[oj] for (c, s, e, a) in ot[oj][tf]}
            if not ses or not motif_tfs:
                continue
            # mean over SEs of fraction of organ-i TFs that bind that SE
            vals = []
            mset = set(motif_tfs)
            for se in ses:
                b = rb.get(se, set())
                vals.append(len(b & mset) / len(motif_tfs))
            conf[i, j] = float(np.mean(vals))

    # 4. recovered known CRC edges
    known = {}
    for o, edges in KNOWN_EDGES.items():
        if o not in crc:
            continue
        E = set(crc[o]["edges"])
        known[o] = {"recovered": [e for e in edges if e in E],
                    "total": len(edges)}

    return {"organs": organs, "stats": stats, "confusion": conf.tolist(),
            "known_edges": known}


def _auto_null(organs, crc, ot, all_pwm_tfs, pwms_all, thresh, n_null, rng):
    """For each organ, expected autoregulation fraction if each target's own SEs
    were probed by a RANDOM TF (with PWM) instead of the target itself."""
    from collections import defaultdict
    out = {}
    # we need region one-hots again; reload from seq cache
    cache = _load_seq_cache()

    def region_oh(c, s, e):
        key = f"{c}:{s}-{min(e, s+3000)}"
        seq = cache.get(key)
        return onehot(seq) if seq else None

    for o in organs:
        nodes = crc[o]["nodes"]
        targets = [t for t in nodes if ot[o].get(t)]
        if not targets:
            out[o] = float("nan"); continue
        # precompute oh per target's SEs
        tgt_ohs = {t: [region_oh(c, s, e) for (c, s, e, a) in ot[o][t]] for t in targets}
        fracs = np.empty(n_null)
        for k in range(n_null):
            rand = rng.choice(all_pwm_tfs, size=len(targets), replace=False)
            hits = 0
            for rt, t in zip(rand, targets):
                for oh in tgt_ohs[t]:
                    if oh is not None and binds(oh, pwms_all[rt], thresh):
                        hits += 1
                        break
            fracs[k] = hits / len(targets)
        out[o] = float(fracs.mean())
    return out


def run(organs=DEFAULT_ORGANS, top_k: int = 18, max_enh: int = 5,
        thresh: float = 0.85, n_null: int = 200) -> dict:
    WIRE_DIR.mkdir(parents=True, exist_ok=True)
    res = build_crc(organs, top_k, max_enh, thresh)
    rep = analyze(res, n_null=n_null)
    # persist
    json.dump({"crc": {o: {k: v for k, v in res["crc"][o].items() if k != "targets"}
                       for o in res["organs"]},
               "report": rep}, open(WIRE_DIR / "crc.json", "w"), indent=2)
    # also dump region_binders compactly for the figure
    json.dump({f"{c}:{s}-{e}": sorted(v) for (c, s, e), v in res["region_binders"].items()},
              open(WIRE_DIR / "region_binders.json", "w"))
    json.dump({o: res["crc"][o]["targets"] for o in res["organs"]},
              open(WIRE_DIR / "organ_targets.json", "w"), default=list)
    return {"res": res, "rep": rep}


def _print(rep: dict):
    print("=" * 76)
    print("ORGAN CASCADE — Stage 2: TF -> SE -> TF core regulatory circuits")
    print("=" * 76)
    for o in rep["organs"]:
        st = rep["stats"][o]
        ke = rep["known_edges"].get(o)
        kstr = ""
        if ke:
            kstr = f"  known-edges {len(ke['recovered'])}/{ke['total']}: " \
                   f"{['->'.join(e) for e in ke['recovered']]}"
        af = st["autoreg_frac"]; an = st["autoreg_null"]
        afs = "n/a" if af != af else f"{af:.0%}"
        ans = "n/a" if an != an else f"{an:.0%}"
        print(f"[{o:16s}] nodes={st['n_nodes']:2d} edges={st['n_edges']:3d} "
              f"density={st['density']:.2f} recip={st['reciprocity']:.0%} "
              f"autoreg={afs} (null {ans}){kstr}")
    conf = np.array(rep["confusion"])
    diag = np.diag(conf)
    off = conf[~np.eye(len(conf), dtype=bool)]
    print("-" * 76)
    print(f"cross-organ binding specificity: mean diagonal {diag.mean():.3f} "
          f"vs mean off-diagonal {off.mean():.3f}")
    correct = sum(int(np.argmax(conf[i]) == i) for i in range(len(conf)))
    print(f"diagonal-dominant organs (own TFs bind own SEs most): "
          f"{correct}/{len(conf)}")
    print("-" * 76)


if __name__ == "__main__":
    import sys
    nn = 200
    out = run(n_null=nn)
    _print(out["rep"])
