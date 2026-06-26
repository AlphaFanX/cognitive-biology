"""
Organ Cascade — Stage 1: organ "head" membership from ABC super-enhancers
==========================================================================

Tests the hypothesis (Miles, 2026-06-24): *each organ is an attention HEAD*,
i.e. an organ's identity is carried by a distinct set of master transcription
factors driven by that organ's super-enhancers (Whyte/Hnisz core regulatory
circuitry; Davidson GRN kernels).

Pure local data, no new downloads beyond the unbiased TF list:
  - ABC enhancer->gene predictions (Nasser 2021, data/enhancer_promoter/), the
    placeholder readout of the zygote kernel + LLM-MLP promoters (per our
    agreement to keep ABC/SE as the stand-in and build the real networks later).
  - Lambert 2018 human-TF list (data/human_tfs_lambert2018.txt, 1639 TFs).

Method per organ:
  1. Take the organ's ABC biosamples (curated against the real 131 cell types).
  2. For each target gene, the super-enhancer signal = total enhancer activity
     (sum of activity_base over all enhancers ABC-assigned to it), averaged over
     the organ's biosamples (so organs with more biosamples are not inflated).
  3. Keep only genes that are TFs.
  4. Rank TFs by specificity-weighted activity -> the organ's master-TF HEAD.

The known literature kernels (KNOWN_KERNEL) are used ONLY as an answer key for
validation (recall, confusion matrix, permutation null) — never as input.

IMPORTANT data note: the Nasser 2021 ABC set has NO kidney biosample, so kidney
(a flagship in the original framing) cannot be tested here; we test heart, liver,
brain, skeletal muscle, intestine, stomach, pancreas, lung, thyroid, adrenal,
skin, bone, breast, and a blood/immune contrast head.
"""

from __future__ import annotations

import os
import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ABC_FILE = ROOT / "data" / "enhancer_promoter" / "AllPredictions.ABC.txt.gz"
TF_FILE = ROOT / "data" / "human_tfs_lambert2018.txt"
CACHE = ROOT / "data" / "abc_reduced_organ_cascade.parquet"
OUT_DIR = ROOT / "data" / "organ_cascade"

MIN_ABC = 0.02  # functional E-G threshold (Nasser 2021)

# Organ -> real ABC biosamples (curated against the 131 cell types actually
# present in AllPredictions.ABC.txt.gz). Generic immune/blood and pluripotent
# lines are kept out of identity organs and given their own contrast head.
ORGAN_BIOSAMPLES: Dict[str, List[str]] = {
    "heart": [
        "cardiac_muscle_cell-ENCODE", "heart_ventricle-ENCODE",
        "coronary_artery-ENCODE", "coronary_artery_smooth_muscle_cell-Miller2016",
    ],
    "liver": ["hepatocyte-ENCODE", "liver-ENCODE", "HepG2-Roadmap"],
    "brain": [
        "astrocyte-ENCODE", "bipolar_neuron_from_iPSC-ENCODE",
        "H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap",
        "SK-N-SH-ENCODE", "spinal_cord_fetal-ENCODE",
    ],
    "skeletal_muscle": [
        "skeletal_muscle_myoblast-Roadmap",
        "myotube_originated_from_skeletal_muscle_myoblast-Roadmap",
        "psoas_muscle-Roadmap", "gastrocnemius_medialis-ENCODE",
        "muscle_of_leg_fetal-Roadmap", "muscle_of_trunk_fetal-Roadmap",
    ],
    "intestine": [
        "small_intestine_fetal-Roadmap", "large_intestine_fetal-Roadmap",
        "sigmoid_colon-ENCODE", "transverse_colon-ENCODE",
        "HT29", "HCT116-ENCODE", "LoVo",
    ],
    "stomach": ["stomach-Roadmap", "stomach_fetal-Roadmap"],
    "pancreas": ["pancreas-Roadmap", "body_of_pancreas-ENCODE", "Panc1-ENCODE"],
    "lung": [
        "fibroblast_of_lung-Roadmap",
        "A549_treated_with_ethanol_0.02_percent_for_1_hour-Roadmap",
        "PC-9-ENCODE",
    ],
    "thyroid": ["thyroid_gland-ENCODE"],
    "adrenal": ["adrenal_gland-ENCODE", "adrenal_gland_fetal-ENCODE"],
    "skin": [
        "keratinocyte-Roadmap", "foreskin_fibroblast-Roadmap",
        "fibroblast_of_dermis-Roadmap",
    ],
    "bone": ["osteoblast-ENCODE"],
    "breast": [
        "breast_epithelium-ENCODE", "mammary_epithelial_cell-Roadmap",
        "MCF10A-Ji2017",
    ],
    "blood_immune": [
        "erythroblast-Corces2016", "CD8-positive_alpha-beta_T_cell-ENCODE",
        "CD4-positive_helper_T_cell-ENCODE", "B_cell-ENCODE",
        "natural_killer_cell-Corces2016", "CD14-positive_monocyte-ENCODE",
    ],
}

# Literature core-regulatory-circuitry / GRN-kernel TFs per organ.
# ANSWER KEY ONLY — never fed into head computation. Canonical, citable master
# regulators (Davidson kernels; Young-lab CRC; standard developmental biology).
KNOWN_KERNEL: Dict[str, List[str]] = {
    "heart": ["NKX2-5", "GATA4", "TBX5", "MEF2C", "ISL1", "HAND1", "HAND2",
              "TBX20", "GATA6", "SRF"],
    "liver": ["HNF4A", "FOXA2", "FOXA1", "CEBPA", "PROX1", "HNF1A", "ONECUT1",
              "NR1H4"],
    "brain": ["SOX2", "PAX6", "NEUROG1", "NEUROG2", "OTX2", "FOXG1", "EMX2",
              "NEUROD1", "POU3F2", "ASCL1"],
    "skeletal_muscle": ["MYOD1", "MYF5", "MYOG", "PAX7", "PAX3", "MEF2A",
                        "SIX1", "TCF21"],
    "intestine": ["CDX2", "CDX1", "KLF5", "HNF4A", "SOX9", "ATOH1", "GATA4"],
    "stomach": ["SOX2", "GATA4", "GATA6", "FOXA2", "GATA5"],
    "pancreas": ["PDX1", "NKX6-1", "PTF1A", "NEUROG3", "PAX4", "PAX6",
                 "NKX2-2", "MNX1", "RFX6"],
    "lung": ["NKX2-1", "FOXA2", "SOX9", "SOX2", "GATA6", "FOXA1"],
    "thyroid": ["NKX2-1", "PAX8", "FOXE1", "HHEX"],
    "adrenal": ["NR5A1", "GATA4", "GATA6", "NR0B1", "WT1", "PBX1"],
    "skin": ["TP63", "KLF4", "GRHL3", "SOX9", "RUNX1"],
    "bone": ["RUNX2", "SP7", "DLX5", "MSX2", "SOX9"],
    "breast": ["GATA3", "FOXA1", "ESR1", "TFAP2C", "ELF5"],
    "blood_immune": ["GATA1", "TAL1", "SPI1", "RUNX1", "GATA2", "IKZF1",
                     "EBF1", "PAX5", "TCF7", "GATA3"],
}


def load_tfs() -> set:
    return {x.strip() for x in open(TF_FILE) if x.strip()}


def build_reduced_abc(force: bool = False) -> pd.DataFrame:
    """Stream the 7.7M-row ABC file once, keep only our biosamples' functional
    E-G rows, cache a small parquet of (TargetGene, CellType, activity_base)."""
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)

    keep = set()
    for bs in ORGAN_BIOSAMPLES.values():
        keep.update(bs)

    cols = ["TargetGene", "CellType", "activity_base", "ABC.Score"]
    parts = []
    for chunk in pd.read_csv(ABC_FILE, sep="\t", usecols=cols, chunksize=500_000):
        m = chunk["CellType"].isin(keep) & (chunk["ABC.Score"] >= MIN_ABC)
        if m.any():
            parts.append(chunk.loc[m, ["TargetGene", "CellType", "activity_base"]])
    df = pd.concat(parts, ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    return df


def organ_tf_matrix(df: pd.DataFrame, tfs: set) -> pd.DataFrame:
    """organ x TF matrix of super-enhancer signal.

    Per (biosample, gene): SE signal = sum of enhancer activity_base over all
    enhancers ABC-assigned to the gene (the ROSE/Whyte super-enhancer quantity).
    Per (organ, gene): mean of that over the organ's biosamples (debiases organs
    with more biosamples). Restricted to TF genes.
    """
    df = df[df["TargetGene"].isin(tfs)].copy()
    # SE signal per biosample x gene
    per_bs = (df.groupby(["CellType", "TargetGene"])["activity_base"]
                .sum().rename("se").reset_index())
    rows = {}
    for organ, biosamples in ORGAN_BIOSAMPLES.items():
        sub = per_bs[per_bs["CellType"].isin(biosamples)]
        # mean across biosamples present for this organ
        g = sub.groupby("TargetGene")["se"].sum() / max(len(biosamples), 1)
        rows[organ] = g
    mat = pd.DataFrame(rows).fillna(0.0)  # genes x organs
    return mat.T  # organs x genes


def heads(mat: pd.DataFrame, top_k: int = 25) -> Dict[str, pd.DataFrame]:
    """Rank each organ's TFs by specificity-weighted activity.

    specificity[o,g] = score[o,g] / sum_o score[:,g]      (organ share, 0..1)
    metric[o,g]      = score[o,g] * specificity[o,g]       (loud AND specific)
    """
    score = mat.values.astype(float)                      # organs x genes
    col_sum = score.sum(axis=0, keepdims=True) + 1e-9
    spec = score / col_sum
    metric = score * spec
    out = {}
    organs = list(mat.index)
    genes = np.array(mat.columns)
    for i, organ in enumerate(organs):
        order = np.argsort(metric[i])[::-1][:top_k]
        out[organ] = pd.DataFrame({
            "tf": genes[order],
            "se_activity": score[i, order],
            "specificity": spec[i, order],
            "metric": metric[i, order],
        })
    return out


def _rank_series(mat: pd.DataFrame) -> Dict[str, pd.Series]:
    """Per organ, full ranking (descending) of TFs by specificity-weighted metric.
    Returns organ -> Series(index=tf, value=rank 1..N)."""
    score = mat.values.astype(float)
    spec = score / (score.sum(axis=0, keepdims=True) + 1e-9)
    metric = score * spec
    genes = np.array(mat.columns)
    ranks = {}
    for i, organ in enumerate(mat.index):
        order = np.argsort(metric[i])[::-1]
        ranked = genes[order]
        ranks[organ] = pd.Series(np.arange(1, len(ranked) + 1), index=ranked)
    return ranks


def validate(mat: pd.DataFrame, head_tables: Dict[str, pd.DataFrame],
             top_k: int = 25, n_perm: int = 2000, seed: int = 0) -> dict:
    """Recall@K of known kernels, confusion matrix, and a permutation null."""
    organs = list(mat.index)
    ranks = _rank_series(mat)
    expressed = set(mat.columns)  # TFs with any ABC signal in our biosamples

    # restrict each kernel to TFs actually present in the matrix
    kernel = {o: [g for g in KNOWN_KERNEL.get(o, []) if g in expressed]
              for o in organs}

    head_sets = {o: set(head_tables[o]["tf"].tolist()) for o in organs}

    # recall@K on the diagonal
    recall = {}
    found = {}
    for o in organs:
        ks = kernel[o]
        if not ks:
            recall[o] = float("nan"); found[o] = []
            continue
        hit = [g for g in ks if g in head_sets[o]]
        found[o] = hit
        recall[o] = len(hit) / len(ks)

    # confusion: kernel(i) recall against head(j)
    conf = np.zeros((len(organs), len(organs)))
    for i, oi in enumerate(organs):
        ks = kernel[oi]
        if not ks:
            continue
        for j, oj in enumerate(organs):
            conf[i, j] = sum(g in head_sets[oj] for g in ks) / len(ks)
    # argmax assignment per kernel row
    assign_correct = sum(int(np.argmax(conf[i]) == i)
                         for i in range(len(organs)) if kernel[organs[i]])

    # permutation null: shuffle which organ each kernel belongs to, recompute
    # mean diagonal recall@K. p = P(null mean >= observed).
    rng = np.random.default_rng(seed)
    valid_idx = [i for i, o in enumerate(organs) if kernel[o]]
    obs_mean = float(np.nanmean([recall[organs[i]] for i in valid_idx]))
    null = np.empty(n_perm)
    kernels_list = [kernel[organs[i]] for i in valid_idx]
    head_list = [head_sets[organs[i]] for i in valid_idx]
    for p in range(n_perm):
        perm = rng.permutation(len(valid_idx))
        vals = []
        for a, b in enumerate(perm):
            ks = kernels_list[a]
            vals.append(sum(g in head_list[b] for g in ks) / len(ks))
        null[p] = np.mean(vals)
    p_value = float((np.sum(null >= obs_mean) + 1) / (n_perm + 1))

    # median rank percentile of known TFs within their own organ
    rank_pct = {}
    for o in organs:
        ks = kernel[o]
        if not ks:
            rank_pct[o] = float("nan"); continue
        n = len(ranks[o])
        pcts = [1.0 - (ranks[o][g] - 1) / n for g in ks]  # 1.0 = top
        rank_pct[o] = float(np.median(pcts))

    return {
        "organs": organs,
        "kernel_present": kernel,
        "recall_at_k": recall,
        "found": found,
        "confusion": conf.tolist(),
        "diag_assign_correct": assign_correct,
        "n_kernels": len(valid_idx),
        "obs_mean_recall": obs_mean,
        "null_mean": float(null.mean()),
        "p_value": p_value,
        "median_rank_pct": rank_pct,
        "top_k": top_k,
    }


def run(top_k: int = 25, force: bool = False) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tfs = load_tfs()
    df = build_reduced_abc(force=force)
    mat = organ_tf_matrix(df, tfs)
    head_tables = heads(mat, top_k=top_k)
    report = validate(mat, head_tables, top_k=top_k)

    # persist
    mat.to_csv(OUT_DIR / "organ_tf_matrix.csv")
    for o, t in head_tables.items():
        t.to_csv(OUT_DIR / f"head_{o}.csv", index=False)
    with open(OUT_DIR / "validation.json", "w") as fh:
        json.dump(report, fh, indent=2)

    return {"matrix": mat, "heads": head_tables, "report": report}


def _print_report(res: dict):
    mat, head_tables, rep = res["matrix"], res["heads"], res["report"]
    print("=" * 74)
    print("ORGAN CASCADE — Stage 1: organ heads from ABC super-enhancers")
    print("=" * 74)
    print(f"organs: {len(rep['organs'])}   TFs with ABC signal: {mat.shape[1]}   "
          f"top_k={rep['top_k']}")
    print()
    for o in rep["organs"]:
        ht = head_tables[o]
        top = ", ".join(ht["tf"].head(10).tolist())
        kn = rep["kernel_present"][o]
        r = rep["recall_at_k"][o]
        hit = rep["found"][o]
        rp = rep["median_rank_pct"][o]
        rstr = "n/a" if (isinstance(r, float) and np.isnan(r)) else f"{r:.0%}"
        rpstr = "n/a" if (isinstance(rp, float) and np.isnan(rp)) else f"{rp:.0%}"
        print(f"[{o}]  head: {top}")
        print(f"     known-kernel recall@{rep['top_k']}: {rstr}  "
              f"(hits: {', '.join(hit) if hit else '-'})   "
              f"median rank-pct of kernel TFs: {rpstr}")
    print()
    print("-" * 74)
    print(f"Confusion diagonal correct (kernel assigned to right head): "
          f"{rep['diag_assign_correct']}/{rep['n_kernels']}")
    print(f"Mean diagonal recall@{rep['top_k']}: {rep['obs_mean_recall']:.1%}   "
          f"null mean: {rep['null_mean']:.1%}   "
          f"permutation p = {rep['p_value']:.4f}")
    print("-" * 74)


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    res = run(force=force)
    _print_report(res)
