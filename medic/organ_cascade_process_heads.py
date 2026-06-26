"""
Organ Cascade — the MIGRATION and DIVISION kernel heads.

The shared NCA/TRM kernel has three readout heads (developmental_trm.py):
Fate (differentiation), Migration (movement), Division (proliferation). Stage 1
derived the Fate head's TFs from ABC super-enhancers grouped by ORGAN (cell
type). Here we derive the other two heads the same way, but grouped by CELL
STATE / process:

  Migration head: mesenchymal/migratory biosamples (FG) vs epithelial (BG)
  Division head : proliferative cancer/stem biosamples (FG) vs post-mitotic (BG)

Head TF = SE-driven TF specific to the foreground state. Validated vs the
literature EMT and cell-cycle master-TF programs (answer key only).

Biological prediction: super-enhancers mark IDENTITY genes, so EMT TFs
(ZEB/TWIST/SNAI) should recover well; generic cell-cycle TFs (E2F/FOXM1) are
typical-enhancer (not SE) genes, so Division may recover weakly -- itself an
informative result about what the SE readout is tuned to.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from medic.organ_cascade import ABC_FILE, MIN_ABC, load_tfs

ROOT = ABC_FILE.parents[2]            # cognimed root
OUT = ROOT / "data" / "organ_cascade"
CACHE = ROOT / "data" / "abc_reduced_process.parquet"

# --- cell-state biosample groups (curated from the real 131 ABC cell types) ---
MIGRATION_FG = [  # mesenchymal / migratory
    "fibroblast_of_arm-ENCODE", "fibroblast_of_dermis-Roadmap",
    "fibroblast_of_lung-Roadmap", "foreskin_fibroblast-Roadmap", "IMR90-Roadmap",
    "H1_Derived_Mesenchymal_Stem_Cells-Roadmap",
    "coronary_artery_smooth_muscle_cell-Miller2016",
    "endothelial_cell_of_umbilical_vein-Roadmap", "MDA-MB-231", "osteoblast-ENCODE",
]
MIGRATION_BG = [  # epithelial / sessile
    "breast_epithelium-ENCODE", "mammary_epithelial_cell-Roadmap", "MCF10A-Ji2017",
    "keratinocyte-Roadmap", "hepatocyte-ENCODE", "epithelial_cell_of_prostate-ENCODE",
    "small_intestine_fetal-Roadmap", "large_intestine_fetal-Roadmap",
    "sigmoid_colon-ENCODE", "transverse_colon-ENCODE", "HT29", "HCT116-ENCODE",
]
DIVISION_FG = [  # highly proliferative cancer + pluripotent/stem
    "K562-Roadmap", "HepG2-Roadmap", "HCT116-ENCODE", "HeLa-S3-Roadmap",
    "MCF-7-ENCODE", "A549_treated_with_ethanol_0.02_percent_for_1_hour-Roadmap",
    "Panc1-ENCODE", "PC-9-ENCODE", "LNCAP", "SK-N-SH-ENCODE", "HAP1", "LoVo",
    "MDA-MB-231", "A673-ENCODE", "MM.1S-ENCODE", "Karpas-422-ENCODE",
    "OCI-LY7-ENCODE", "BJAB-Engreitz", "Jurkat-Engreitz", "H1-hESC-Roadmap",
    "H9-Roadmap", "H7", "induced_pluripotent_stem_cell-ENCODE",
    "iPS_DF_19.11_Cell_Line-Roadmap", "NCCIT", "GM12878-Roadmap",
]
DIVISION_BG = [  # post-mitotic / differentiated
    "cardiac_muscle_cell-ENCODE", "heart_ventricle-ENCODE", "astrocyte-ENCODE",
    "spinal_cord_fetal-ENCODE", "bipolar_neuron_from_iPSC-ENCODE",
    "myotube_originated_from_skeletal_muscle_myoblast-Roadmap",
    "gastrocnemius_medialis-ENCODE", "psoas_muscle-Roadmap",
    "hepatocyte-ENCODE", "liver-ENCODE", "thyroid_gland-ENCODE",
]

# --- literature answer keys (used ONLY for validation) ---
EMT_TFS = ["SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2", "PRRX1",
           "PRRX2", "FOXC2", "SOX10", "FOXD3", "ETS1", "TFAP2A", "TFAP2C",
           "GSC", "TCF3", "TCF4", "SRF", "RUNX2", "TEAD1"]
CELLCYCLE_TFS = ["E2F1", "E2F2", "E2F3", "E2F4", "E2F5", "E2F6", "E2F7", "E2F8",
                 "TFDP1", "TFDP2", "MYBL2", "FOXM1", "MYC", "MYCN", "MYB"]


def build_reduced(force=False):
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    keep = set(MIGRATION_FG + MIGRATION_BG + DIVISION_FG + DIVISION_BG)
    cols = ["TargetGene", "CellType", "activity_base", "ABC.Score"]
    parts = []
    for chunk in pd.read_csv(ABC_FILE, sep="\t", usecols=cols, chunksize=500_000):
        m = chunk["CellType"].isin(keep) & (chunk["ABC.Score"] >= MIN_ABC)
        if m.any():
            parts.append(chunk.loc[m, ["TargetGene", "CellType", "activity_base"]])
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(CACHE)
    return df


def se_signal_per_biosample(df, tfs):
    df = df[df["TargetGene"].isin(tfs)]
    return (df.groupby(["CellType", "TargetGene"])["activity_base"]
              .sum().rename("se").reset_index())


def group_score(per_bs, biosamples):
    present = [b for b in biosamples if b in set(per_bs["CellType"])]
    sub = per_bs[per_bs["CellType"].isin(present)]
    g = sub.groupby("TargetGene")["se"].sum() / max(len(present), 1)
    return g, present


def derive_head(per_bs, fg, bg, name, answer_key, tfs, top_k=30):
    fg_s, fg_present = group_score(per_bs, fg)
    bg_s, bg_present = group_score(per_bs, bg)
    genes = sorted(set(fg_s.index) | set(bg_s.index))
    fg_v = fg_s.reindex(genes).fillna(0.0).values
    bg_v = bg_s.reindex(genes).fillna(0.0).values
    spec = fg_v / (fg_v + bg_v + 1e-9)          # FG share (0..1)
    metric = fg_v * spec                         # loud in FG and FG-specific
    order = np.argsort(metric)[::-1]
    ranked = [genes[i] for i in order]
    head = ranked[:top_k]

    key = [g for g in answer_key if g in set(genes)]
    rankpos = {g: ranked.index(g) for g in key}
    recall = sum(g in set(head) for g in key) / max(len(key), 1)
    # rank percentile (1.0 = top) of known TFs
    N = len(ranked)
    pct = {g: 1.0 - rankpos[g] / N for g in key}
    med_pct = float(np.median(list(pct.values()))) if key else float("nan")
    # Mann-Whitney: are known TFs' metric > the rest?
    known_mask = np.array([g in set(key) for g in genes])
    if known_mask.sum() and (~known_mask).sum():
        U, p = mannwhitneyu(metric[known_mask], metric[~known_mask],
                            alternative="greater")
    else:
        p = float("nan")
    return {
        "name": name, "fg_present": fg_present, "bg_present": bg_present,
        "head": head, "metric": {g: float(metric[order[i]]) for i, g in enumerate(head)},
        "key_present": key, "recall_at_k": recall, "median_rank_pct": med_pct,
        "key_ranks": {g: rankpos[g] + 1 for g in key}, "mwu_p": float(p),
        "top_k": top_k, "ranked": ranked, "metric_full": metric.tolist(),
        "genes": genes,
    }


def main(force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    tfs = load_tfs()
    df = build_reduced(force=force)
    per_bs = se_signal_per_biosample(df, tfs)

    mig = derive_head(per_bs, MIGRATION_FG, MIGRATION_BG, "migration", EMT_TFS, tfs)
    div = derive_head(per_bs, DIVISION_FG, DIVISION_BG, "division", CELLCYCLE_TFS, tfs)

    for h in (mig, div):
        print("=" * 74)
        print(f"{h['name'].upper()} HEAD  (FG {len(h['fg_present'])} biosamples, "
              f"BG {len(h['bg_present'])})")
        print("=" * 74)
        print("top-15 derived TFs:", ", ".join(h["head"][:15]))
        kp = h["key_present"]
        hits = [g for g in kp if g in set(h["head"])]
        print(f"literature recall@{h['top_k']}: {h['recall_at_k']:.0%}  "
              f"hits: {hits}")
        print(f"median rank-pct of literature TFs: {h['median_rank_pct']:.0%}   "
              f"Mann-Whitney p (known>rest): {h['mwu_p']:.4f}")
        print("literature-TF ranks:", {g: h["key_ranks"][g] for g in kp})
        print()

    json.dump({k: {kk: vv for kk, vv in h.items()
                   if kk not in ("metric_full", "genes", "ranked")}
               for k, h in [("migration", mig), ("division", div)]},
              open(OUT / "process_heads.json", "w"), indent=2)
    # save full for figure
    np.savez(OUT / "process_heads_full.npz",
             mig_genes=np.array(mig["genes"], object),
             mig_metric=np.array(mig["metric_full"]),
             div_genes=np.array(div["genes"], object),
             div_metric=np.array(div["metric_full"]),
             emt=np.array(EMT_TFS, object), cc=np.array(CELLCYCLE_TFS, object))
    return mig, div


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
