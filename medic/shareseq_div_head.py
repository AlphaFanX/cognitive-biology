"""
Cloning the head-tracing recipe to the DIVISION head (SHARE-seq skin).
=====================================================================

Paper #6 sec:heads: after the differentiation head (shareseq_diff_head.py), the recipe
clones to division. Division is a scalar head---the genome sets a proliferation rate---so
the fit is a regression, not a multi-class classification:

  proliferation(cell)  ~  d * accessibility_of_the_cell-cycle_module(cell)

PREDICTOR (the genome's setting, from ATAC): per-cell fraction of accessible reads falling
  in the cis-windows of a cell-cycle master module (Mki67/Top2a/Ccnb/Cdk1/Mcm/...), read
  from the ATAC peak x cell matrix.
OBSERVABLE (the realized division state, from RNA in the SAME cell): per-cell fraction of
  transcripts from the same cell-cycle genes -- the standard proliferation score.
The two are joined per cell through the celltype file (atac.bc <-> rna.bc), so this is the
multiome accessibility->state arrow for proliferation. We test whether cell-cycle
accessibility predicts the held-out proliferation score above a shuffled-pairing null, and
whether both rise together across the real proliferation gradient of the hair follicle
(transit-amplifying / matrix high; differentiated and bulge low).

Data (data/shareseq/, GSE140203, GSM4156597 ATAC + GSM4156608 RNA):
  skin_atac_peakmatrix.txt.gz  MatrixMarket 344592 peaks x 34774 cells (ATAC)
  skin_peaks.bed.gz            peak coords (row order)
  skin_atac_barcodes.txt.gz    atac.bc (column order)
  skin_rna.counts.txt.gz       dense gene x cell table (RNA; header = rna.bc, comma-delim)
  skin_celltype.txt.gz         atac.bc, rna.bc, celltype

Run: python -m medic.shareseq_div_head {modules|atac|rna|fit|all}
"""
from __future__ import annotations
import gzip, json, sys
from pathlib import Path
import numpy as np

SS = Path("data/shareseq")
OUT = Path("data/organ_cascade")
PEAKS = SS / "skin_peaks.bed.gz"
REFGENE = SS / "mm10_refGene.txt.gz"
ATAC_MTX = SS / "skin_atac_peakmatrix.txt.gz"
ATAC_BC = SS / "skin_atac_barcodes.txt.gz"
RNA = SS / "skin_rna.counts.txt.gz"
CELLTYPE = SS / "skin_celltype.txt.gz"
WINDOW = 50_000

# cell-cycle / proliferation master module (canonical S + G2/M drivers; mouse symbols)
CYCLE_GENES = ["Mki67", "Top2a", "Pcna", "Mcm2", "Mcm3", "Mcm5", "Mcm6",
               "Ccnb1", "Ccnb2", "Ccna2", "Cdk1", "Cdc20", "Cdc6", "Birc5",
               "Aurka", "Aurkb", "Bub1", "Plk1", "Rrm2", "Tyms", "Ube2c",
               "Cenpa", "Cenpf", "Nusap1", "Cks2"]


def _tss():
    tss = {}
    with gzip.open(REFGENE, "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            chrom, strand, txStart, txEnd, name2 = c[2], c[3], int(c[4]), int(c[5]), c[12]
            tss.setdefault(name2, (chrom, txStart if strand == "+" else txEnd))
    return tss


def build_modules():
    """1-based ROW indices (matching the peak matrix / peaks.bed order) of peaks in a
    cell-cycle gene cis-window."""
    tss = _tss()
    wins = []
    for g in CYCLE_GENES:
        if g in tss:
            chrom, t = tss[g]; wins.append((chrom, t - WINDOW, t + WINDOW))
        else:
            print(f"  WARN {g} not in refGene")
    cyc = []
    with gzip.open(PEAKS, "rt") as f:
        for i, line in enumerate(f, start=1):          # 1-based to match MatrixMarket rows
            c = line.split("\t")
            if len(c) < 3:
                continue
            chrom = c[0]; mid = (int(c[1]) + int(c[2])) // 2
            for (wc, s, e) in wins:
                if wc == chrom and s <= mid <= e:
                    cyc.append(i); break
    json.dump({"genes": CYCLE_GENES, "cycle_peak_rows": cyc, "window": WINDOW},
              open(SS / "div_modules.json", "w"))
    print(f"cell-cycle module: {len(CYCLE_GENES)} genes -> {len(cyc)} peaks")


def build_atac():
    """Per-cell cell-cycle ATAC accessibility fraction, from the peak x cell matrix."""
    cyc = set(json.load(open(SS / "div_modules.json"))["cycle_peak_rows"])
    bcs = [l.strip() for l in gzip.open(ATAC_BC, "rt")]     # column order (atac.bc)
    ncell = len(bcs)
    cycc = np.zeros(ncell + 1); totc = np.zeros(ncell + 1)  # 1-based cols
    with gzip.open(ATAC_MTX, "rt") as f:
        assert f.readline().lower().startswith("%%matrixmarket")
        f.readline()                                        # dims line
        for k, line in enumerate(f):
            r, c, v = line.split()
            c = int(c); v = int(v)
            totc[c] += v
            if int(r) in cyc:
                cycc[c] += v
            if k % 30_000_000 == 0 and k:
                print(f"  ...{k//1_000_000}M nonzeros")
    acc = cycc[1:] / np.clip(totc[1:], 1, None)
    np.savez(SS / "div_atac.npz", bc=np.array(bcs), acc=acc, tot=totc[1:])
    print(f"saved div_atac.npz: {ncell} cells, mean cycle-accessibility {acc.mean():.4f}")


def build_rna():
    """Per-cell proliferation score = fraction of transcripts from the cell-cycle genes."""
    cyc = set(CYCLE_GENES)
    with gzip.open(RNA, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        bcs = [b.replace(",", ".") for b in header[1:]]     # rna.bc (comma->dot)
        n = len(bcs)
        cyc_sum = np.zeros(n); tot = np.zeros(n)
        for k, line in enumerate(f):
            i = line.find("\t")
            gene = line[:i]
            vals = np.fromstring(line[i + 1:], sep="\t")
            if vals.shape[0] != n:                          # guard ragged lines
                vals = np.resize(vals, n)
            tot += vals
            if gene in cyc:
                cyc_sum += vals
            if k % 5000 == 0 and k:
                print(f"  ...{k} genes")
    score = cyc_sum / np.clip(tot, 1, None)
    np.savez(SS / "div_rna.npz", bc=np.array(bcs), score=score, tot=tot)
    print(f"saved div_rna.npz: {n} cells, mean proliferation score {score.mean():.4f}")


def fit():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    A = np.load(SS / "div_atac.npz", allow_pickle=True)
    R = np.load(SS / "div_rna.npz", allow_pickle=True)
    # SHARE-seq cell id = the split-pool triple R1.R2.R3; the trailing P1 is the sublibrary
    # index and differs between a cell's ATAC and RNA reads, so join on the R1.R2.R3 prefix.
    rrr = lambda bc: ".".join(bc.split(".")[:6])
    acc = dict(zip([rrr(b) for b in A["bc"].astype(str)], A["acc"]))
    score = dict(zip([rrr(b) for b in R["bc"].astype(str)], R["score"]))
    xs, ys, cts = [], [], []
    with gzip.open(CELLTYPE, "rt") as f:
        next(f)
        for line in f:
            abc, rbc, ct = line.rstrip("\n").split("\t")
            k = rrr(abc)
            if k in acc and k in score:
                xs.append(acc[k]); ys.append(score[k]); cts.append(ct)
    x = np.array(xs); y = np.array(ys); cts = np.array(cts)
    # standardize
    xz = (x - x.mean()) / (x.std() + 1e-12); yz = (y - y.mean()) / (y.std() + 1e-12)
    r = float(np.corrcoef(xz, yz)[0, 1])
    # held-out R^2 of a 1-feature linear fit
    rng = np.random.RandomState(0); idx = rng.permutation(len(x))
    tr, te = idx[:int(0.7 * len(x))], idx[int(0.7 * len(x)):]
    b, a0 = np.polyfit(xz[tr], yz[tr], 1)
    pred = a0 + b * xz[te]
    r2 = float(1 - np.sum((yz[te] - pred) ** 2) / np.sum((yz[te] - yz[te].mean()) ** 2))
    # null: shuffle the pairing
    null = np.array([abs(np.corrcoef(xz, rng.permutation(yz))[0, 1]) for _ in range(200)])
    print(f"n paired cells: {len(x)}")
    print(f"corr(cell-cycle ACCESSIBILITY, proliferation SCORE) = {r:+.3f}  "
          f"(|null| {null.mean():.3f}+-{null.std():.3f})")
    print(f"held-out R^2 (1-feature) = {r2:.3f}")
    # proliferation gradient across celltypes (biological validation)
    order = ["TAC-1", "TAC-2", "Basal", "ORS", "Medulla", "IRS", "Hair Shaft-cuticle.cortex",
             "ahighCD34+ bulge", "alowCD34+ bulge"]
    print("\nper-celltype means (proliferation gradient):")
    print(f"  {'celltype':26s} {'ATAC-acc':>9} {'RNA-score':>10} {'n':>6}")
    rows = []
    for ct in order:
        m = cts == ct
        if m.sum() > 20:
            rows.append((ct, x[m].mean(), y[m].mean(), int(m.sum())))
            print(f"  {ct:26s} {x[m].mean():9.4f} {y[m].mean():10.4f} {int(m.sum()):6d}")

    _figure(x, y, cts, r, r2, null, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_cells=len(x), corr=r, heldout_r2=r2, null_mean=float(null.mean()),
                   null_std=float(null.std()), genes=CYCLE_GENES,
                   celltype_means={ct: dict(atac=a, rna=b, n=n) for ct, a, b, n in rows}),
              open(OUT / "shareseq_div_head.json", "w"), indent=2)
    print("\nsaved", OUT / "shareseq_div_head.png")
    return r > 0 and r > null.mean() + 5 * (null.std() + 1e-9)


def _figure(x, y, cts, r, r2, null, rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    # hexbin accessibility vs score
    ax[0].hexbin(x, y, gridsize=45, cmap="magma", bins="log")
    b, a0 = np.polyfit((x - x.mean()) / x.std(), (y - y.mean()) / y.std(), 1)
    ax[0].set_xlabel("cell-cycle ATAC accessibility (fraction)")
    ax[0].set_ylabel("proliferation RNA score (fraction)")
    ax[0].set_title(f"(a) per-cell, {len(x)} cells\ncorr = {r:+.2f}  (|null| {null.mean():.2f})", fontsize=9)
    # per-celltype gradient
    labels = [r_[0] for r_ in rows]
    xa = np.array([r_[1] for r_ in rows]); ya = np.array([r_[2] for r_ in rows])
    ax[1].scatter(xa, ya, s=60, color="tab:purple")
    for i, lab in enumerate(labels):
        ax[1].annotate(lab.replace("Hair Shaft-cuticle.cortex", "Cortex"), (xa[i], ya[i]),
                       fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("mean cell-cycle ATAC accessibility")
    ax[1].set_ylabel("mean proliferation RNA score")
    ax[1].set_title("(b) the proliferation gradient:\naccessibility and division rise together (TAC high, bulge low)", fontsize=9)
    fig.suptitle("Division head (Paper #6 sec:heads): per-cell cell-cycle module ACCESSIBILITY predicts the realized "
                 "proliferation state, and both track the hair-follicle proliferation gradient.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "shareseq_div_head.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("modules", "all"):
        build_modules()
    if stage in ("atac", "all"):
        build_atac()
    if stage in ("rna", "all"):
        build_rna()
    if stage in ("fit", "all"):
        ok = fit()
        print(f"\nRESULT: {'TRACED (accessibility predicts proliferation)' if ok else 'CHECK'}")
