"""Classify each facial-GWAS variant on the KERNEL <-> ADAPTER axis (paper Section 3.5).

The framework predicts allele-frequency x effect x pleiotropy x constraint co-vary along a
selection gradient from the zygote kernel (deeply-inherited, high-pleiotropy developmental
enhancers / machinery, strong purifying selection) to the individual cis-MLP adapter
(common, low-pleiotropy, tissue-specific, segregating). We score OUR six loci on that axis.

Data: curated annotations (well-established literature; sourced) + REAL gnomAD per-population
frequency divergence (data/population_freqs.json) as a selection-divergence proxy.
NB: a precise coordinate-level overlap against the Jadhav hypomethylated kernel would need
human developmental-enhancer tracks + liftover (refinement); here the kernel proxy is the
gene's developmental/neural-crest master-regulator role, which is what makes it constrained.
"""
import os, json, csv

HERE = os.path.dirname(__file__)
FREQS = json.load(open(os.path.join(HERE, "data", "population_freqs.json")))

# --- curated knowledge base (sourced) ---
KB = {
 "EDAR": dict(variant="rs3827760 (V370A)", consequence="coding (missense)",
    dev_role="ectodysplasin/NF-kB pathway (ectodermal appendage development)",
    pleiotropy_systems=["hair","teeth (shovel incisors)","eccrine sweat glands","ear","chin","mammary"],
    selection="strong positive sweep in East Asians (very high Fst)",
    gene_constraint="developmental pathway, dosage-sensitive",
    src="Kamberov2013; Adhikari2016; Sabeti2007"),
 "PAX3": dict(variant="rs7559271 (nasion)", consequence="regulatory (non-coding)",
    dev_role="master neural-crest paired-box TF",
    pleiotropy_systems=["pigment","hearing","craniofacial","limb (Waardenburg)"],
    selection="common, modest divergence",
    gene_constraint="high (haploinsufficient: Waardenburg)",
    src="Paternoster2012; Claes2018; OMIM Waardenburg"),
 "RUNX2": dict(variant="6p21 lead (nose bridge breadth)", consequence="regulatory (non-coding)",
    dev_role="master osteoblast/skeletal TF",
    pleiotropy_systems=["skull","clavicle","teeth","bone (cleidocranial dysplasia)"],
    selection="common", gene_constraint="high (haploinsufficient: CCD)",
    src="Adhikari2016; OMIM CCD"),
 "GLI3": dict(variant="7p13 lead (nose wing breadth)", consequence="regulatory (non-coding)",
    dev_role="Hedgehog-pathway zinc-finger TF",
    pleiotropy_systems=["digits/limb","craniofacial","brain (Greig/Pallister-Hall)"],
    selection="common", gene_constraint="high (haploinsufficient developmental)",
    src="Adhikari2016; OMIM GCPS"),
 "PAX1": dict(variant="20p11 lead (nose wing breadth)", consequence="regulatory (non-coding)",
    dev_role="paired-box skeletal/pharyngeal TF",
    pleiotropy_systems=["vertebrae","thymus","craniofacial (otofaciocervical)"],
    selection="common", gene_constraint="high (developmental TF)",
    src="Adhikari2016; OMIM OTFCS2"),
 "DCHS2": dict(variant="4q31 lead (nasal tip)", consequence="regulatory (non-coding)",
    dev_role="protocadherin, planar cell polarity / cartilage",
    pleiotropy_systems=["cartilage/PCP"],
    selection="common", gene_constraint="moderate",
    src="Adhikari2016"),
}

def freq_divergence(gene):
    """max-min per-population allele-freq spread = selection-divergence proxy (real gnomAD)."""
    for rs, rec in FREQS.items():
        if gene in rec.get("desc", ""):
            f = list(rec["freq"].values())
            return max(f) - min(f), max(f), min(f)
    return None, None, None

def classify():
    rows = []
    for gene, k in KB.items():
        div, fmax, fmin = freq_divergence(gene)
        coding = k["consequence"].startswith("coding")
        npleio = len(k["pleiotropy_systems"])
        swept = "sweep" in k["selection"]
        ks = 0
        ks += 2 if coding else 0
        ks += 1 if npleio >= 4 else 0
        ks += 2 if (swept or (div is not None and div > 0.5)) else (1 if (div is not None and div > 0.2) else 0)
        ks += 1 if "high" in k["gene_constraint"] else 0
        verdict = "KERNEL-level variant" if (coding or swept) else "ADAPTER variant on a kernel gene"
        rows.append(dict(gene=gene, variant=k["variant"], consequence=k["consequence"],
            gene_role=k["dev_role"], pleiotropy_systems=npleio,
            freq_divergence=None if div is None else round(div, 3),
            selection=k["selection"], kernel_score=ks, verdict=verdict, source=k["src"]))
    return rows

if __name__ == "__main__":
    rows = classify()
    print(f"{'gene':6} {'consequence':22} {'pleio':5} {'fdiv':5} {'Kscore':6} verdict")
    for r in sorted(rows, key=lambda x: -x["kernel_score"]):
        fd = "-" if r["freq_divergence"] is None else f"{r['freq_divergence']:.2f}"
        print(f"{r['gene']:6} {r['consequence']:22} {r['pleiotropy_systems']:<5} {fd:5} {r['kernel_score']:<6} {r['verdict']}")
    p = os.path.join(HERE, "data", "kernel_adapter_classification.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nsaved", p)
    print("KERNEL end:", [r["gene"] for r in rows if "KERNEL" in r["verdict"]])
    print("ADAPTER end:", [r["gene"] for r in rows if "ADAPTER" in r["verdict"]])
