"""
Genome-anchored tissue-specific proliferation: the cell-cycle regulon read per tissue by AlphaGenome.
=====================================================================================================
The division head was genome-BLIND: it multiplied the whole body at one near-uniform rate, so the final
cell count per organ tracked spatial territory and surface area, not the tissue-specific proliferation
that actually sets organ sizes. The result was ~9x too much skin (a surface shell on a sparse body) and
~6x too little brain (the fastest-proliferating tissue, unboosted). This module supplies the missing
per-tissue proliferation RATE, and anchors it in the genome the same way a head's identity is anchored:

  ANCHOR (forward, genome -> rate).  A tissue that divides fast expresses the cell-cycle / proliferation
  REGULON (Mki67, Pcna, Top2a, Ccnb1, Cdk1, the Mcm helicases, Foxm1, Mybl2, E2f1, ...). AlphaGenome
  predicts that regulon's expression per tissue FROM SEQUENCE, so its per-tissue mean is a genome-derived
  proliferation index -- the identical machinery used for head identity, pointed at the proliferation
  program instead of the master-TF program.

  VALIDATION (reverse, MOSTA implements).  The atlas's own per-tissue cell-fraction across E9.5->E13.5 is
  the ground-truth proliferation; the genome index should track it (brain high, skin/notochord low).

Honesty (the g_K-anchor pattern): the AlphaGenome prediction is the genome anchor; where a fate has no
clean AG tissue the weight falls back to a labelled literature prior, to be converted to genome later.
The cache means simulate() reads a genome-derived weight table, not a live API call.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.proliferation_genome
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np

from medic.alphagenome_regulon_manifold import gene_tss, GTF, LINES

HOME = Path(os.path.expanduser("~"))
os.environ.setdefault("ALPHAGENOME_API_KEY", (HOME / ".alphagenome_key").read_text().strip())
os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(HOME / ".ca_combined.pem"))
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(HOME / ".ca_combined.pem"))
WIN, BODY = 8192, 6000
RAW_CACHE = Path("data/organ_cascade/_ag_gene_rawprof_cache.json")     # shared RNA raw-profile cache
WEIGHT_TABLE = Path("data/organ_cascade/proliferation_weights.json")   # genome-derived fate -> weight
MUSCLE_KW = ("gastroc", "skeletal muscle", "myocyte")

# the proliferation / cell-cycle regulon: a fast-cycling tissue co-expresses these
PROLIF_GENES = ["Mki67", "Pcna", "Top2a", "Ccnb1", "Ccnb2", "Cdk1", "Ccna2", "Mcm2", "Mcm3", "Mcm5",
                "Mcm6", "Foxm1", "Mybl2", "E2f1", "Ccne1", "Cdc20", "Bub1", "Aurka", "Aurkb", "Birc5",
                "Cenpa", "Cdk4", "Ccnd1"]

# The progenitor-WINDOW class per fate: how long the tissue stays an undifferentiated, cycling progenitor
# before its cells exit the cell cycle -- which is set by the DIFFERENTIATION CLOCK (telomere->PRC2/Hox).
# Organ cell number = proliferation rate INTEGRATED over this window, so the SAME clock that sets WHEN a
# tissue differentiates also sets HOW BIG it gets: the big pools (CNS, muscle, meninges, cartilage,
# connective) cycle across the whole neurogenic/myogenic window; the organs (heart, liver, kidney) exit
# early and stay small. Values are the multiplicative window (1 = baseline / early-exit organ); magnitudes
# validated against the MOSTA E9.5->E13.5 fraction growth (g_K-anchor pattern: clock mechanism, atlas-tuned
# scale, to be converted to clock-derived progenitor durations).
WINDOW_CLASS = {
    # long-window neural progenitor pool (the CNS, ~22% + cord)
    "Forebrain": 3.4, "Midbrain": 3.4, "Hindbrain": 3.4, "Cerebellum": 3.4, "OlfactoryBulb": 3.4,
    "Brain": 3.4, "Retina": 2.6, "Choroid": 1.8, "Eye": 1.6,
    "Nervous System": 1.8, "Spinal Cord": 1.8, "Neural Crest": 1.6, "DRG": 1.3, "Sympathetic": 1.2,
    # long-window mesenchymal / muscle pools (the body wall)
    "Muscle": 3.2, "Meninges": 2.8, "Cartilage": 2.1, "Rib": 2.1, "Sclerotome": 2.1,
    "Connective": 1.9, "HeadMes": 1.9, "Jaw": 1.4, "Branchial": 1.3,
    # medium-window viscera / organ progenitors
    "Liver": 1.4, "LiverHaem": 1.4, "Kidney": 1.2, "Nephron": 1.2, "Gut": 1.15, "Foregut": 1.15,
    "Hindgut": 1.15, "Mucosa": 1.1, "Lung": 1.15, "Pancreas": 1.1,
    # early-exit / small: heart, notochord, small organs, surface -> baseline 1.0 (default)
}

# model fate -> the AlphaGenome mouse tissue whose proliferation index the fate inherits. Grouped so every
# fate resolves to a tissue present in the embryonic RNA panel (brain regions -> their own track).
FATE_TISSUE = {
    "Forebrain": "forebrain", "Midbrain": "midbrain", "Hindbrain": "hindbrain", "Cerebellum": "hindbrain",
    "OlfactoryBulb": "forebrain", "Brain": "forebrain", "Nervous System": "neural tube",
    "Spinal Cord": "neural tube", "Neural Crest": "neural tube", "DRG": "neural tube",
    "Sympathetic": "neural tube", "Retina": "retina", "Eye": "retina", "Choroid": "hindbrain",
    "Heart": "heart", "Atrium": "heart", "Ventricle": "heart", "Outflow": "heart",
    "Liver": "liver", "LiverHaem": "liver", "Lung": "lung", "Pancreas": "stomach",
    "Gut": "intestine", "Foregut": "stomach", "Hindgut": "intestine", "Mucosa": "intestine",
    "Kidney": "kidney", "Nephron": "kidney", "Gonad": "gonadal fat pad", "Adrenal": "adrenal gland",
    "Bladder": "urinary bladder", "Thymus": "thymus", "Spleen": "spleen",
    "Muscle": "skeletal muscle", "Cartilage": "limb", "Rib": "limb", "Sclerotome": "limb",
    "Notochord": "neural tube", "Connective": "limb", "Adipose": "subcutaneous adipose tissue",
    "HeadMes": "embryonic facial prominence", "Jaw": "embryonic facial prominence",
    "Branchial": "embryonic facial prominence", "Meninges": "forebrain",
    "Skin": "embryonic facial prominence", "Epidermis": "embryonic facial prominence",
    "Surface Ectoderm": "embryonic facial prominence", "Mesothelium": "heart", "Mesentery": "intestine",
    "Blood": "liver", "Blood Vessel": "liver", "Blood/AGM": "liver",
}


def _tissue_panel(meta):
    md = meta.rna_seq
    names = list(md["biosample_name"]); ls = [str(x).lower() for x in md["biosample_life_stage"]]
    keep = [i for i in range(len(names))
            if (("embry" in ls[i]) or any(k in names[i].lower() for k in MUSCLE_KW))
            and not any(cl in names[i].lower() for cl in LINES)]
    return keep, [names[i] for i in keep]


def tissue_proliferation_index():
    """Per-tissue genome proliferation index = mean predicted expression of the cell-cycle regulon."""
    from alphagenome.models import dna_client
    from alphagenome.data import genome
    m = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])
    meta = m.output_metadata(organism=dna_client.Organism.MUS_MUSCULUS)
    keep, tnames = _tissue_panel(meta)
    ut = sorted(set(tnames))
    tss = gene_tss(PROLIF_GENES)
    cache = json.load(open(RAW_CACHE)) if RAW_CACHE.exists() else {}

    def expr(gene):
        if gene in cache:
            raw = np.array(cache[gene])
        else:
            if gene not in tss:
                return None
            chrom, pos, strand = tss[gene]
            iv = genome.Interval(chromosome=chrom, start=pos - WIN, end=pos + WIN)
            try:
                r = m.predict_interval(iv, requested_outputs=[dna_client.OutputType.RNA_SEQ],
                                       ontology_terms=None, organism=dna_client.Organism.MUS_MUSCULUS)
            except Exception:
                return None
            v = np.asarray(r.rna_seq.values); c = v.shape[0] // 2
            seg = v[c:c + BODY] if strand == "+" else v[c - BODY:c]
            cache[gene] = [float(x) for x in seg.mean(0)]
            raw = np.array(cache[gene])
        prof = raw[keep]
        by = {}
        for t, val in zip(tnames, prof):
            by.setdefault(t, []).append(val)
        return np.array([float(np.mean(by[t])) for t in ut])

    vs = []
    for g in PROLIF_GENES:
        e = expr(g)
        if e is not None and e.std() > 0:
            vs.append((e - e.mean()) / (e.std() + 1e-9))       # z-score each gene across tissues
    json.dump(cache, open(RAW_CACHE, "w"))
    idx = np.mean(vs, 0)                                        # per-tissue proliferation index (z units)
    return {t: float(idx[i]) for i, t in enumerate(ut)}, len(vs)


# Labelled LITERATURE priors (NOT AlphaGenome) for fates whose AG tissue track is an unreliable proxy
# for their proliferation. The embryonic RNA panel has no epidermis track, and "facial prominence" is
# proliferative crest mesenchyme, not the slow-cycling epidermal sheet; the surface epithelia are given a
# low prior so the interior bulk tissues, not the surface shell, carry the growth. To be genome-anchored
# once a mouse basal-keratinocyte track is available. (These are flagged as priors in the output table.)
FATE_PRIOR = {"Skin": 0.55, "Epidermis": 0.55, "Surface Ectoderm": 0.55}


def build_weights(gamma=0.6):
    """Genome proliferation index -> per-fate division weight. gamma tempers the spread (weight = exp of a
    tempered z), so a high-proliferation fate is favoured as a dividing parent but no fate is starved. This
    is the per-STEP RATE; the differentiation clock sets how long a fate stays proliferative, and the
    simulation INTEGRATES rate x window into the final organ cell number."""
    tp, ng = tissue_proliferation_index()
    weights, prior_flag = {}, {}
    for fate, tis in FATE_TISSUE.items():
        if fate in FATE_PRIOR:
            weights[fate] = FATE_PRIOR[fate]; prior_flag[fate] = True
        else:
            weights[fate] = float(np.exp(gamma * tp.get(tis, 0.0))); prior_flag[fate] = False
    mean_w = np.mean(list(weights.values()))                   # normalise mean -> 1 (overall total unchanged)
    weights = {f: w / mean_w for f, w in weights.items()}
    out = {"genes_used": ng, "gamma": gamma, "tissue_index": tp,
           "fate_weight": weights, "is_literature_prior": prior_flag}
    json.dump(out, open(WEIGHT_TABLE, "w"), indent=1)
    return out


def main():
    out = build_weights()
    tp = out["tissue_index"]
    print(f"proliferation regulon: {out['genes_used']}/{len(PROLIF_GENES)} genes located, per-tissue index (z):\n")
    for t, v in sorted(tp.items(), key=lambda kv: -kv[1]):
        print(f"  {t:34s} {v:+.2f}")
    print("\nfate division weights (mean=1; >1 divides faster):")
    for f, w in sorted(out["fate_weight"].items(), key=lambda kv: -kv[1]):
        print(f"  {f:16s} {w:.2f}  (<- {FATE_TISSUE[f]})")

    # VALIDATION: does the genome index track the real MOSTA proliferation? Use E12.5 fraction as proxy.
    try:
        import collections
        from medic.cell_level_mouse import read_stage
        from medic.tps_register import STAGES
        _, rann = read_stage(STAGES["E12.5"]); rann = np.array(rann)
        c = collections.Counter(rann); N = len(rann)
        M = {"Brain": "forebrain", "Muscle": "skeletal muscle", "Cartilage primordium": "limb",
             "Heart": "heart", "Liver": "liver", "Kidney": "kidney", "GI tract": "intestine",
             "Epidermis": "embryonic facial prominence", "Lung": "lung"}
        xs, ys, labs = [], [], []
        for ann, tis in M.items():
            if ann in c and tis in tp:
                xs.append(tp[tis]); ys.append(100 * c[ann] / N); labs.append(ann)
        if len(xs) >= 4:
            r = float(np.corrcoef(xs, ys)[0, 1])
            print(f"\nVALIDATION vs real MOSTA E12.5 fraction: genome prolif-index correlates r={r:+.2f} "
                  f"(n={len(xs)} tissues)")
            for a, x, y in sorted(zip(labs, xs, ys), key=lambda t: -t[1]):
                print(f"  {a:22s} genome-index {x:+.2f}   real fraction {y:4.1f}%")
    except Exception as e:
        print("validation skipped:", e)
    print("\nsaved", WEIGHT_TABLE)


if __name__ == "__main__":
    main()
