"""Ground the face-morph deformer scales in REAL Xiong et al. (2025) C-GWAS effect sizes.

For each facial locus we take its lead SNP (present in the Xiong SnpInfo), read that SNP's
per-allele Beta across the within-region landmark-distance files of the region it shapes, and
use max|Beta| as the locus's real standardized effect magnitude. EDAR / RUNX2 use a proxy SNP
in the same locus (their canonical lead SNP is not in the Xiong imputation set).

Emits data/xiong_cgwas/real_effects.json = {gene: {snp, region, beta, a1}}.
Source: Xiong et al., Nat Commun 16:6562 (2025); Zenodo 10.5281/zenodo.13730680 (CC-BY).

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.ground_face_betas
"""
import os, json, bz2

BASE = "data/xiong_cgwas"

# gene -> (lead rsID present in Xiong, region archive folder)   [EDAR/RUNX2 = same-locus proxies]
LOCI = {
    "EDAR":  ("rs11676729", "Chin"),        # EDAR-region max-effect SNP on chin distances
    "PAX3":  ("rs7559271",  "Uppernose"),   # nasion
    "RUNX2": ("rs227833",   "Uppernose"),   # SUPT3H/RUNX2 bridge/root proxy
    "DCHS2": ("rs2045323",  "Lowernose"),   # nasal tip
    "GLI3":  ("rs11170624", "Lowernose"),   # alae
    "PAX1":  ("rs2724626",  "Lowernose"),   # alae
}


def snp_rows(wanted):
    """rsID -> (data-row-index, A1) for the wanted set."""
    out = {}
    with open(os.path.join(BASE, "SnpInfo.tsv")) as f:
        next(f)
        for i, line in enumerate(f, start=1):
            p = line.rstrip("\n").split("\t")
            if p[2] in wanted:
                out[p[2]] = (i, p[3])
                if len(out) == len(wanted):
                    break
    return out


def beta_at(region, row):
    """max|Beta| for data-row `row` across all within-region distance files."""
    best = 0.0
    d = os.path.join(BASE, region)
    for fn in os.listdir(d):
        if not fn.endswith(".tsv"):
            continue
        with open(os.path.join(d, fn)) as f:
            next(f)                                   # header
            for i, line in enumerate(f, start=1):
                if i == row:
                    b = float(line.split("\t")[0])
                    if abs(b) > abs(best):
                        best = b
                    break
    return best


def main():
    rows = snp_rows({rs for rs, _ in LOCI.values()})
    out = {}
    for gene, (rs, region) in LOCI.items():
        if rs not in rows:
            print(f"WARN {gene} {rs} absent"); continue
        row, a1 = rows[rs]
        b = beta_at(region, row)
        out[gene] = {"snp": rs, "region": region, "beta": round(b, 5), "a1": a1}
        print(f"{gene:6s} {rs:12s} {region:10s} beta={b:+.5f} a1={a1}")
    json.dump(out, open(os.path.join(BASE, "real_effects.json"), "w"), indent=2)
    print("saved", os.path.join(BASE, "real_effects.json"))


if __name__ == "__main__":
    main()
