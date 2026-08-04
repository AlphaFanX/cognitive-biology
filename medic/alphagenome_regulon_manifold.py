"""
AlphaGenome head-manifold test, strengthened: the REGULON TARGET-GENE PROGRAM as the head signature.
=====================================================================================================
The single-master-TF version (alphagenome_head_manifold) was noisy: a TF's own expression is a crude
stand-in for its head. A head is a master-TF SUPER-ENHANCER cluster driving a PROGRAM of target genes,
so here the head signature is the AlphaGenome-predicted expression profile of the head's REGULON -- its
canonical marker/target genes -- averaged over the program (each gene z-scored across tissues so all
contribute equally). This is a much more robust head signature. Each unique gene is queried once (mm10
locus from GENCODE vM25) and cached. We then compare the AlphaGenome head-by-head correlation (the
sequence-model Gram matrix) to the MOSTA head-signature matrix: Mantel r, effective rank, principal angle.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.alphagenome_regulon_manifold
"""
from __future__ import annotations
import os, gzip, json, re
from pathlib import Path
import numpy as np

HOME = Path(os.path.expanduser("~"))
os.environ.setdefault("ALPHAGENOME_API_KEY", (HOME / ".alphagenome_key").read_text().strip())
_ca = HOME / ".ca_combined.pem"
os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_ca))
os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_ca))
GTF = "data/mm10/gencode.vM25.gtf.gz"
WIN, BODY = 8192, 6000
CACHE = Path("data/organ_cascade/_ag_gene_rawprof_cache.json")   # RAW per-track mean coverage (re-aggregatable)
EMBRYONIC_ONLY = True                                            # restrict to embryonic-life-stage tissues (match MOSTA)

# MOSTA head -> (regulon program = canonical marker/target genes, expected top tissue for the sanity check)
HEAD_MARKERS = {
    "Heart": (["Nkx2-5", "Myh6", "Myh7", "Myl7", "Myl2", "Tnnt2", "Tnni3", "Actc1", "Gata4", "Gata6",
               "Tbx5", "Tbx20", "Mef2c", "Nppa", "Ryr2", "Pln", "Ttn"], "heart"),
    "Liver": (["Foxa3", "Alb", "Apoa1", "Apoa2", "Apob", "Ttr", "Hnf4a", "Serpina1a", "Fga", "Fgb",
               "Ahsg", "Apoc3", "Trf", "Cyp2e1", "Hp"], "liver"),
    "Lung": (["Nkx2-1", "Sftpc", "Sftpb", "Sftpa1", "Scgb1a1", "Foxj1", "Ager", "Aqp5", "Muc5b",
              "Elf5", "Foxp2"], "lung"),
    "Kidney": (["Pax2", "Pax8", "Wt1", "Six2", "Slc12a1", "Umod", "Cdh16", "Slc34a1", "Aqp2", "Nphs1",
                "Nphs2", "Podxl", "Lrp2"], "kidney"),
    "GI tract": (["Cdx2", "Vil1", "Fabp2", "Krt20", "Muc2", "Lgr5", "Cdx1", "Apoa4", "Fabp1", "Guca2a"], "intestine"),
    "Muscle": (["Myod1", "Myog", "Myf5", "Myf6", "Myh3", "Myh8", "Actn2", "Actn3", "Des", "Ckm",
                "Tnnt3", "Tnni2", "Acta1", "Mb", "Neb"], "skeletal muscle"),
    "Cartilage primordium": (["Sox9", "Sox5", "Sox6", "Col2a1", "Col9a1", "Col11a1", "Acan", "Comp",
                              "Matn1", "Matn3", "Hapln1", "Ucma"], "limb"),
    "Dorsal root ganglion": (["Prrxl1", "Ntrk1", "Six1", "Pou4f1", "Isl1", "Isl2", "Drgx", "Tlx3",
                              "Ntrk2", "Ntrk3", "Pvalb"], "neural tube"),
    "Sympathetic nerve": (["Phox2b", "Phox2a", "Th", "Dbh", "Ddc", "Chga", "Chgb", "Gata3", "Hand2",
                           "Ascl1"], "neural tube"),
    "Brain": (["Arx", "Foxg1", "Emx1", "Emx2", "Neurod2", "Neurod6", "Tbr1", "Sox2", "Pax6", "Dlx2",
               "Gad1", "Lhx2"], "forebrain"),
    "Notochord": (["T", "Noto", "Shh", "Foxa2", "Chrd", "Nog", "Ntn1"], "neural tube"),
    "Blood vessel": (["Pecam1", "Cdh5", "Kdr", "Tek", "Cldn5", "Etv2", "Flt1", "Tie1", "Emcn", "Cd34",
                      "Erg", "Sox17", "Sox18"], "liver"),
    "Epidermis": (["Krt14", "Krt5", "Krt15", "Krt10", "Krt1", "Grhl3", "Cdh1", "Ovol1", "Trp63",
                   "Sfn", "Dsp", "Perp"], "embryonic facial prominence"),
    "Ovary": (["Foxl2", "Figla", "Nobox", "Gdf9", "Sohlh2", "Sohlh1", "Zp3", "Zp2", "Bmp15", "Lhx8"], "ovary"),
    "Pancreas": (["Pdx1", "Ins1", "Ins2", "Gcg", "Nkx6-1", "Ptf1a", "Sst", "Neurog3", "Pax4", "Cpa1"], "stomach"),
    "Meninges": (["Zic1", "Zic2", "Foxc1", "Foxc2", "Cldn11", "Aldh1a2", "Lum", "Cxcl12"], "forebrain"),
    "Connective tissue": (["Twist2", "Twist1", "Col1a1", "Col1a2", "Col3a1", "Col5a1", "Pdgfra", "Lum",
                           "Dcn", "Fbn1", "Postn", "Loxl1"], "limb"),
    "Spinal cord": (["Hoxb9", "Olig2", "Nkx6-1", "Sox2", "Mnx1", "Isl1", "Chat", "Slc17a6", "Nkx2-2",
                     "Sim1"], "neural tube"),
    "Sclerotome": (["Pax1", "Pax9", "Nkx3-2", "Sox9", "Foxc2", "Meox1", "Meox2"], "limb"),
    "Surface ectoderm": (["Krt8", "Krt18", "Cdh1", "Grhl3", "Grhl2", "Trp63", "Krt5", "Perp", "Cldn6"],
                         "embryonic facial prominence"),
    "Choroid plexus": (["Rfx2", "Ttr", "Otx2", "Aqp1", "Folr1", "Clic6", "Kcnj13"], "hindbrain"),
    "Jaw and tooth": (["Msx1", "Msx2", "Dlx1", "Dlx2", "Barx1", "Prrx1", "Prrx2", "Lhx6", "Pitx2",
                       "Bmp4"], "embryonic facial prominence"),
}
LINES = ("patski", "c3h10t1/2", "es-e14", "es-bruce4", "e14tg2a", "mel", "3t3", "nih3t3", "a20",
         "ch12", "g1e", "r1", "ww6", "zhbtc4", "mn1", "416b", "3134", "c2c12", "embryonic fibroblast")


def gene_tss(symbols):
    want = set(symbols); out = {}
    for line in gzip.open(GTF, "rt"):
        if line.startswith("#"):
            continue
        f = line.split("\t")
        if f[2] != "gene":
            continue
        mm = re.search(r'gene_name "([^"]+)"', f[8])
        if mm and mm.group(1) in want and mm.group(1) not in out:
            out[mm.group(1)] = (f[0], int(f[3]) if f[6] == "+" else int(f[4]), f[6])
    return out


def main():
    from alphagenome.models import dna_client
    from alphagenome.data import genome
    m = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])
    meta = m.output_metadata(organism=dna_client.Organism.MUS_MUSCULUS)
    md = meta.rna_seq
    rna_bs = list(md["biosample_name"])
    ls = [str(x).lower() for x in md["biosample_life_stage"]]
    # embryonic panel + the muscle tracks (gastrocnemius/skeletal muscle/myocyte), which the embryonic
    # set lacks, so the Muscle head localises to muscle and separates from Cartilage (both otherwise
    # collapse onto limb -- the source of the 3rd-mode residual). No cartilage track exists in the model.
    MUSCLE_KW = ("gastroc", "skeletal muscle", "myocyte")
    keep_t = [i for i in range(len(rna_bs))
              if ((not EMBRYONIC_ONLY or "embry" in ls[i]) or any(k in rna_bs[i].lower() for k in MUSCLE_KW))
              and not any(cl in rna_bs[i].lower() for cl in LINES)]
    tissue_names = [rna_bs[i] for i in keep_t]
    ut = sorted(set(tissue_names))
    print(f"context: {'EMBRYONIC' if EMBRYONIC_ONLY else 'all'} RNA -> {len(ut)} tissues: {ut}\n")

    genes = sorted({g for gs, _ in HEAD_MARKERS.values() for g in gs})
    tss = gene_tss(genes)
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    def expr(gene):                                                    # per-tissue expression vector for a gene
        if gene in cache:
            raw = np.array(cache[gene])                                # cached RAW per-track profile (all tracks)
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
            raw = seg.mean(0)                                          # ALL tracks (re-aggregatable)
            cache[gene] = [float(x) for x in raw]
        prof = raw[keep_t]                                             # subset to the chosen tissue context
        by = {}
        for t, val in zip(tissue_names, prof):
            by.setdefault(t, []).append(val)
        return np.array([float(np.mean(by[t])) for t in ut])

    print(f"querying {len(genes)} unique regulon genes ({len(tss)} located in mm10) x {len(ut)} tissues ...\n")
    sig = {}
    for head, (gs, expect) in HEAD_MARKERS.items():
        vs = []
        for g in gs:
            e = expr(g)
            if e is not None and e.std() > 0:
                vs.append((e - e.mean()) / (e.std() + 1e-9))          # z-score each gene across tissues
        if len(vs) < 2:
            print(f"  {head:22s} -- too few genes, skipped"); continue
        prog = np.mean(vs, 0)                                          # the regulon PROGRAM signature
        sig[head] = prog
        top = ut[int(np.argmax(prog))]
        ok = "OK" if expect.lower() in top.lower() or top.lower() in expect.lower() else "  "
        print(f"  {head:22s} ({len(vs)} genes) top tissue: {top:32s} (expect {expect}) {ok}")
    json.dump(cache, open(CACHE, "w"))

    heads = list(sig); A = np.array([sig[h] for h in heads])
    AG = np.corrcoef(A)
    d = np.load("data/organ_cascade/head_trajectory.npz", allow_pickle=True)
    mheads = list(d["heads"]); C = d["centroids"].astype(float); present = d["present"]
    msig = {h: C[mheads.index(h)][present[mheads.index(h)]].mean(0)
            for h in heads if h in mheads and present[mheads.index(h)].any()}
    common = [h for h in heads if h in msig]
    print(f"\ncommon heads: {len(common)}")
    idx = [heads.index(h) for h in common]
    AGc = np.corrcoef(A[idx])
    M = np.array([msig[h] for h in common]); M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    MO = np.corrcoef(M)
    iu = np.triu_indices(len(common), 1)
    mantel = float(np.corrcoef(AGc[iu], MO[iu])[0, 1])
    k = 3
    eA = np.linalg.eigh(AGc)[1][:, -k:]; eM = np.linalg.eigh(MO)[1][:, -k:]
    ang = np.degrees(np.arccos(np.clip(np.linalg.svd(eA.T @ eM, compute_uv=False), -1, 1)))
    ev = np.abs(np.linalg.eigvalsh(AGc)); pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    print(f"\nAlphaGenome regulon head-matrix effective rank = {pr:.1f} of {len(common)}")
    print(f"MANTEL correlation (AlphaGenome-regulon vs MOSTA head matrices) r = {mantel:+.2f}")
    print(f"principal angles top-{k} subspaces = {', '.join(f'{a:.0f}' for a in sorted(ang))} deg "
          f"(0=aligned; smallest is the best-aligned direction)")
    json.dump(dict(heads=common, mantel=mantel, ag_eff_rank=pr,
                   principal_angles_deg=[float(a) for a in sorted(ang)]),
              open("data/organ_cascade/alphagenome_regulon_manifold.json", "w"), indent=1)
    print("\nsaved data/organ_cascade/alphagenome_regulon_manifold.json")


if __name__ == "__main__":
    main()
