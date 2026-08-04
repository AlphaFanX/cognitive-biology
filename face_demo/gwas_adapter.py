"""The cis-LoRA adapter in shape space.

face(individual) = kernel + sum_k dosage_k * beta_k * e_k        (a genotype)
face(population) = kernel + sum_k (2*p_k) * beta_k * e_k          (a population mean)

e_k = the facial deformation direction of locus k (face_model.GENE_DEFORM, grounded in
published GWAS). dosage_k from a genotype (0/1/2) or 2*allele-frequency for a population.
beta_k magnitudes are folded into mm_per_allele placeholders for now.
"""
import os, csv, json
import numpy as np
import face_model

HERE = os.path.dirname(__file__)
HITS = os.path.join(HERE, "data", "facial_gwas_hits.csv")
FREQS = os.path.join(HERE, "data", "population_freqs.json")

def load_hits():
    with open(HITS) as f:
        return list(csv.DictReader(f))

def load_freqs():
    return json.load(open(FREQS)) if os.path.exists(FREQS) else {}

def gene_to_rsid():
    """Map gene -> lead SNP rsid from the hit table (where known)."""
    m = {}
    for row in load_hits():
        snp = row.get("lead_snp", "").strip()
        if snp and snp != "TODO":
            m[row["gene"]] = snp
    return m

def population_dosages(pop):
    """Expected diploid alt-allele dosage (2*p) per gene, for genes with real per-pop freq.
    Genes without per-population frequency data contribute equally across populations and
    therefore cancel in any population *difference* -- so we omit them here (honest)."""
    freqs = load_freqs(); g2r = gene_to_rsid()
    dosages = {}
    for gene, rsid in g2r.items():
        rec = freqs.get(rsid)
        if rec and pop in rec["freq"]:
            dosages[gene] = 2.0 * rec["freq"][pop]
    return dosages

def population_face(pop):
    return face_model.deform(population_dosages(pop))

def genotype_face(genotype):
    """genotype: dict gene -> allele count in {0,1,2}."""
    return face_model.deform({g: float(d) for g, d in genotype.items()})

def face_distance(P, Q):
    """RMS landmark distance between two faces (a simple distinguishability metric)."""
    return float(np.sqrt(((P - Q) ** 2).sum(axis=1).mean()))
