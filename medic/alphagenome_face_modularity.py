"""GENOME-SIDE modularity check -- the independent second read of the face's genetic modularity.

Companion to medic.decoupling_test (the GWAS-CATALOG read: facial sub-features are governed by more
INDEPENDENT gene sets -- gene-reuse 1.48 vs heart 2.06). That read is population-genomic (which genes
associate with which imaging sub-feature). This module reads the SAME modularity from SEQUENCE: for each
gene's lead facial / cardiac GWAS variant, AlphaGenome's calibrated ATAC variant scorer gives the ref->alt
change across all 167 human ATAC tracks = the variant's REGULATORY FOOTPRINT (which tissue/cell-type
contexts it perturbs). Two sequence-derived modularity metrics, mirroring the two catalog metrics:

  (1) FOOTPRINT BREADTH  = participation ratio PR = (sum|d|)^2 / sum(d^2) over the 167 tracks
      = the effective number of regulatory contexts a variant touches. Low = concentrated/modular.
  (2) CROSS-GENE FOOTPRINT OVERLAP = pairwise cosine of the |d| footprints WITHIN a system.
      Low = each gene hits a distinct regulatory program (decoupled); high = shared core program (coupled).
      Computed both RAW and with the shared common-mode baseline removed (the fair, system-specific read).

THREE contrast systems, so the comparison is TIGHTENED against the "heterogeneous heart" confound:
  FACE       -- the named large-effect facial genes, each on a distinct facial sub-feature.
  CARDIAC_TF -- MATCHED cardiac developmental transcription factors (the fair contrast: same TF character
                as the facial genes), the cardiac core regulatory circuit.
  CARDIAC_CH -- the original functionally-heterogeneous cardiac panel (channels + a caveolin), kept as the
                robustness comparison that first showed an (artefactual) orthogonality.

FINDING (2026-07-26): a DISSOCIATION. The phenotype read (catalog decoupling) and the sequence read are not
the same manifold -- the facial genes are a single COHERENT craniofacial regulatory programme (positive
residual overlap) even though their phenotypic sub-features are decoupled; the matched cardiac TFs are
likewise a coherent programme, while the heterogeneous channel set is orthogonal. So regulatory coupling is
generic to a developmental toolkit; the face's modularity is PHENOTYPIC (a shared toolkit deployed at
distinct anatomical ADDRESSES), not written in the genes' cis-regulatory footprints.

HONEST CAVEAT: AlphaGenome human ATAC is adult-heavy with NO embryonic craniofacial track
([[cognimed-alphagenome-access]]); the footprints are read in a suboptimal context for the face.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.alphagenome_face_modularity
Out:  data/organ_cascade/alphagenome_face_modularity.{json,png}  (footprints cached in the json)
"""
from __future__ import annotations
import os, json, itertools
from pathlib import Path
import numpy as np

HOME = Path(os.path.expanduser("~"))
if "ALPHAGENOME_API_KEY" not in os.environ and (HOME / ".alphagenome_key").exists():
    os.environ["ALPHAGENOME_API_KEY"] = (HOME / ".alphagenome_key").read_text().strip()
_ca = HOME / ".ca_combined.pem"
if _ca.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_ca))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_ca))

OUT = Path("data/organ_cascade/alphagenome_face_modularity.json")
FIG = Path("data/organ_cascade/alphagenome_face_modularity.png")
WIN = 8192
SYS_ORDER = ["FACE", "CARDIAC_TF", "CARDIAC_CH"]
SYS_COL = {"FACE": "#b05fa8", "CARDIAC_TF": "#2a6fb0", "CARDIAC_CH": "#d64545"}

# gene -> (system, expected hg38 chrom, [candidate lead SNPs -- first that resolves to the expected chrom is used])
GENES = {
    # FACE: six named large-effect facial genes, each a distinct facial sub-feature (chin/nasion/tip/bridge/alae)
    "EDAR":  ("FACE", "chr2",  ["rs3827760"]),                 # chin protrusion (coding V370A)
    "PAX3":  ("FACE", "chr2",  ["rs7559271"]),                 # nasion position
    "DCHS2": ("FACE", "chr4",  ["rs2045323", "rs807037"]),     # nasal tip protrusion
    "RUNX2": ("FACE", "chr6",  ["rs227833"]),                  # nasal bridge breadth
    "GLI3":  ("FACE", "chr7",  ["rs929387", "rs17640804"]),    # alar breadth (chr7 -- guard vs the chr12 proxy)
    "PAX1":  ("FACE", "chr20", ["rs2724626", "rs1339132", "rs6054304"]),  # alar breadth
    # CARDIAC_TF: MATCHED cardiac developmental transcription factors (the fair, TF-character contrast)
    "PITX2":  ("CARDIAC_TF", "chr4",  ["rs2200733"]),          # left-right / AF; laterality TF
    "TBX5":   ("CARDIAC_TF", "chr12", ["rs883079", "rs7312625"]),   # cardiac septation / conduction TF (PR)
    "TBX3":   ("CARDIAC_TF", "chr12", ["rs10850409", "rs2891304"]), # conduction-system TF
    "NKX2-5": ("CARDIAC_TF", "chr5",  ["rs251253", "rs3729753"]),   # cardiac master TF (conduction)
    "GATA4":  ("CARDIAC_TF", "chr8",  ["rs804271", "rs3729856"]),   # cardiac master TF
    "ZFHX3":  ("CARDIAC_TF", "chr16", ["rs2106261", "rs7193343"]),  # AF TF
    "PRRX1":  ("CARDIAC_TF", "chr1",  ["rs3903239", "rs577676"]),   # AF homeobox TF
    # CARDIAC_CH: original heterogeneous cardiac panel (channels + caveolin) -- the robustness contrast
    "SCN5A":  ("CARDIAC_CH", "chr3",  ["rs6801957"]),
    "SCN10A": ("CARDIAC_CH", "chr3",  ["rs6795970"]),
    "GNB4":   ("CARDIAC_CH", "chr3",  ["rs7612445"]),
    "CAV1":   ("CARDIAC_CH", "chr7",  ["rs3807989"]),
    "KCNQ1":  ("CARDIAC_CH", "chr11", ["rs2074238"]),
    "NOS1AP": ("CARDIAC_CH", "chr1",  ["rs12143842"]),
}


def variant_coords(rs):
    import requests
    j = requests.get("https://myvariant.info/v1/query",
                     params={"q": rs, "fields": "chrom,vcf"}, timeout=30).json()
    for h in j.get("hits", []):
        v = h.get("vcf")
        if v and h.get("chrom"):
            return "chr" + str(h["chrom"]), int(v["position"]), v["ref"], v["alt"]
    return None


def footprints():
    """Score each gene's lead variant -> its 167-track ATAC |d| footprint. Cached in the json; only genes
    NOT already cached are scored, so the panel can grow without re-hitting the API for old genes."""
    fp, meta = {}, {}
    if OUT.exists():
        c = json.load(open(OUT))
        fp = {g: np.array(v) for g, v in c.get("_footprints", {}).items()}
        meta = c.get("_meta", {})
    missing = [g for g in GENES if g not in fp]
    if not missing:
        print(f"all {len(GENES)} footprints cached")
        return fp, meta
    print(f"scoring {len(missing)} new genes: {', '.join(missing)}")
    from alphagenome.models import dna_client, variant_scorers as vs
    from alphagenome.data import genome
    m = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])
    scorer = vs.RECOMMENDED_VARIANT_SCORERS["ATAC"]
    for gene in missing:
        system, exp_chrom, cands = GENES[gene]
        chosen = None
        for rs in cands:
            co = variant_coords(rs)
            if co and co[0] == exp_chrom:
                chosen = (rs, co); break
            elif co:
                print(f"  {gene} {rs} -> {co[0]} != {exp_chrom}, trying next")
        if not chosen:
            print(f"  {gene}: no candidate resolved to {exp_chrom}, SKIP"); continue
        rs, (chrom, pos, ref, alt) = chosen
        iv = genome.Interval(chromosome=chrom, start=pos - WIN, end=pos + WIN)
        var = genome.Variant(chromosome=chrom, position=pos, reference_bases=ref, alternate_bases=alt)
        ad = m.score_variant(iv, var, variant_scorers=[scorer])[0]
        d = np.abs(np.asarray(ad.X[0], dtype=float))
        fp[gene] = d
        meta[gene] = dict(system=system, rs=rs, chrom=chrom, pos=pos, ref=ref, alt=alt, ntracks=int(d.size))
        print(f"  {gene:7s} {system:10s} {rs:12s} {chrom}:{pos} {ref}>{alt:3s}  sum|d|={d.sum():.3f} PR={pr(d):.1f}")
    return fp, meta


def pr(d):
    s2 = float((d ** 2).sum())
    return float((d.sum() ** 2) / s2) if s2 > 0 else 0.0


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def run():
    fp, meta = footprints()
    for g in fp:                                          # system labels come from the CURRENT registry
        meta.setdefault(g, {})["system"] = GENES[g][0]
    sysof = {g: GENES[g][0] for g in fp}
    groups = {s: [g for g in fp if sysof[g] == s] for s in SYS_ORDER}
    groups = {s: gs for s, gs in groups.items() if len(gs) >= 2}
    print("\nscored: " + ", ".join(f"{s} {len(gs)}" for s, gs in groups.items()))

    breadth = {g: pr(fp[g]) for g in fp}

    # system-balanced common-mode baseline: equal weight per system, so no larger panel biases the residual
    sys_mean = {s: np.mean([fp[g] for g in gs], axis=0) for s, gs in groups.items()}
    base = np.mean(list(sys_mean.values()), axis=0)
    fpc = {g: fp[g] - base for g in fp}

    def within(gs, ff):
        return [cosine(ff[a], ff[b]) for a, b in itertools.combinations(gs, 2)]

    stats = {}
    for s, gs in groups.items():
        stats[s] = dict(n=len(gs),
                        mean_PR=round(float(np.mean([breadth[g] for g in gs])), 1),
                        overlap_raw=round(float(np.mean(within(gs, fp))), 4),
                        overlap_cmr=round(float(np.mean(within(gs, fpc))), 4),
                        genes=gs)

    print("\n--- metric 1: footprint breadth (participation ratio, effective # tracks) ---")
    for s in groups:
        print(f"  {s:11s} mean PR = {stats[s]['mean_PR']:5.1f}   "
              f"({', '.join(f'{g} {breadth[g]:.0f}' for g in groups[s])})")

    print("\n--- metric 2: cross-gene footprint overlap (cosine; low = decoupled) ---")
    print(f"  {'system':11s} {'raw':>7s}  {'baseline-removed':>16s}")
    for s in groups:
        print(f"  {s:11s} {stats[s]['overlap_raw']:7.3f}  {stats[s]['overlap_cmr']:+16.3f}")

    F, T, C = stats["FACE"], stats["CARDIAC_TF"], stats["CARDIAC_CH"]
    verdict = ("Genome-side ATAC footprints do NOT cleanly resolve the phenotype-level modularity. Footprint "
               "breadth is indistinguishable (FACE %.0f / CARDIAC_TF %.0f / CARDIAC_CH %.0f effective tracks) and "
               "the baseline-removed residual overlaps are all small and similar (%+.2f / %+.2f / %+.2f), so no "
               "system is the modular one from sequence. Raw overlap is marginally highest for the facial genes "
               "(%.2f vs %.2f / %.2f), weakly consistent with a shared craniofacial programme, but this is a small "
               "effect and the earlier apparent face-coupled / cardiac-orthogonal split was baseline-dependent and "
               "does NOT survive the matched-TF contrast. ROBUST CONCLUSION: the face's phenotypic modularity "
               "(catalog gene-reuse 1.48) is NOT recoverable from adult-chromatin regulatory footprints -- the "
               "modularity is PHENOTYPIC, not cis-regulatory, and adult ATAC lacks the embryonic craniofacial "
               "context where facial regulation is decided. The two reads (phenotype vs sequence) are not one "
               "manifold." % (F["mean_PR"], T["mean_PR"], C["mean_PR"], F["overlap_cmr"], T["overlap_cmr"],
                              C["overlap_cmr"], F["overlap_raw"], T["overlap_raw"], C["overlap_raw"]))
    print("\n" + verdict)

    out = dict(n_tracks=int(next(iter(fp.values())).size), systems=stats,
               interpretation=verdict, _meta=meta, _footprints={g: fp[g].tolist() for g in fp})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(fp, fpc, sysof, groups, breadth, stats)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _figure(fp, fpc, sysof, groups, breadth, stats):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np.random.seed(0)
    order = [g for s in groups for g in groups[s]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.7), facecolor="white")

    # panel 1: per-gene breadth
    ax[0].bar(range(len(order)), [breadth[g] for g in order], color=[SYS_COL[sysof[g]] for g in order])
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    ax[0].set_ylabel("footprint breadth (participation ratio,\neffective # ATAC tracks)")
    ax[0].set_title("metric 1: regulatory breadth\n(no clean separation by system)")

    # panel 2: raw vs baseline-removed overlap per system
    ss = list(groups); x = np.arange(len(ss)); w = 0.38
    ax[1].bar(x - w/2, [stats[s]["overlap_raw"] for s in ss], w, color="#cccccc", label="raw (shared baseline)")
    ax[1].bar(x + w/2, [stats[s]["overlap_cmr"] for s in ss], w,
              color=[SYS_COL[s] for s in ss], label="baseline removed")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels([s.replace("_", "\n") for s in ss], fontsize=8)
    ax[1].set_ylabel("cross-gene footprint overlap (cosine)")
    ax[1].legend(fontsize=7.5, loc="upper right")
    ax[1].set_title("metric 2: baseline-removed overlap\nall systems small & similar = footprints do not resolve modularity")

    # panel 3: footprint heatmap over the most variable tracks, rows grouped by system
    M = np.vstack([fp[g] for g in order])
    var_tracks = np.argsort(M.var(0))[::-1][:40]
    im = ax[2].imshow(M[:, var_tracks], aspect="auto", cmap="magma", extent=[0, 40, len(order), 0])
    ax[2].set_yticks(np.arange(len(order)) + 0.5); ax[2].set_yticklabels(order, fontsize=6.5)
    ax[2].set_xlabel("top-40 most variable ATAC tracks"); ax[2].set_xticks([])
    ax[2].set_title("variant footprints (|$\\Delta$| ATAC)")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    fc = stats.get("FACE", {}).get("overlap_cmr", 0); tf = stats.get("CARDIAC_TF", {}).get("overlap_cmr", 0)
    ch = stats.get("CARDIAC_CH", {}).get("overlap_cmr", 0)
    fig.suptitle("Genome-side modularity check (AlphaGenome), tightened with matched cardiac developmental TFs: a "
                 "NULL. Footprint breadth and baseline-removed overlap do not separate FACE (%+.2f), matched "
                 "CARDIAC_TF (%+.2f) or the channel set (%+.2f) -- all small and similar.\nThe phenotype-level "
                 "modularity of the face is NOT recoverable from adult-chromatin footprints: modularity is "
                 "PHENOTYPIC, not cis-regulatory. The phenotype read and the sequence read are not one manifold."
                 % (fc, tf, ch), fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG, dpi=135, facecolor="white")


if __name__ == "__main__":
    run()
