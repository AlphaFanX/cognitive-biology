"""
Cloning the head-tracing recipe to the MIGRATION head (SHARE-seq skin).
=======================================================================

Paper #6 sec:heads, third clone. Migration is the head where a dissociated atlas runs out
of observable: SHARE-seq measures molecules, not movement, so it cannot record displacement
or velocity. What it can record is the molecular MOTILITY PROGRAM---the mesenchymal/
migratory transcriptional and accessibility state (Snai2/Twist1/Zeb1, vimentin, the ECM-
remodelling and contractility genes). So we fit, exactly as for division,

  motility-program expression(cell)  ~  d * accessibility_of_the_motility_module(cell)

and report it honestly for what it is: the CAPACITY to migrate (the program), not the
migration itself. The actual movement is the missing observable this paper's title names---
it is not in the atlas and must be supplied by the mechanical simulation, or by live imaging.

Same data and machinery as shareseq_div_head.py (ATAC peak matrix + RNA table + the
R1.R2.R3 split-pool join). Predictor = motility-module ATAC accessibility fraction;
observable = motility-module RNA fraction (a migratory-state score). Expectation, stated
before the run: because the motility program is tied to mesenchymal IDENTITY (dermal
fibroblast/sheath/papilla, endothelium, neural-crest derivatives) and identity is chromatin-
encoded, this may be MORE accessibility-legible than division was---but it is still the
program, not the movement.

Run: python -m medic.shareseq_mig_head {modules|atac|rna|fit|all}
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

# motility / mesenchymal-migration module (EMT TFs, cytoskeleton/contractility, ECM remodelling)
MIG_GENES = ["Snai2", "Twist1", "Zeb1", "Prrx1", "Prrx2", "Vim", "Acta2", "Tagln",
             "Fn1", "Mmp2", "Mmp14", "Postn", "Sparc", "Lox", "Col1a1", "Col3a1",
             "Pdgfra", "Pdgfrb", "Thy1", "Cdh2"]

# migratory/mesenchymal (high) -> epithelial/sessile (low)
GRAD = ["Dermal Sheath", "Dermal Fibroblast", "Dermal Papilla", "Endothelial",
        "Schwann Cell", "Melanocyte", "ORS", "Basal", "IRS", "Medulla",
        "Hair Shaft-cuticle.cortex", "ahighCD34+ bulge"]


def _tss():
    tss = {}
    with gzip.open(REFGENE, "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            chrom, strand, txStart, txEnd, name2 = c[2], c[3], int(c[4]), int(c[5]), c[12]
            tss.setdefault(name2, (chrom, txStart if strand == "+" else txEnd))
    return tss


def build_modules():
    tss = _tss(); wins = []
    for g in MIG_GENES:
        if g in tss:
            chrom, t = tss[g]; wins.append((chrom, t - WINDOW, t + WINDOW))
        else:
            print(f"  WARN {g} not in refGene")
    mig = []
    with gzip.open(PEAKS, "rt") as f:
        for i, line in enumerate(f, start=1):
            c = line.split("\t")
            if len(c) < 3:
                continue
            chrom = c[0]; mid = (int(c[1]) + int(c[2])) // 2
            for (wc, s, e) in wins:
                if wc == chrom and s <= mid <= e:
                    mig.append(i); break
    json.dump({"genes": MIG_GENES, "module_peak_rows": mig, "window": WINDOW},
              open(SS / "mig_modules.json", "w"))
    print(f"motility module: {len(MIG_GENES)} genes -> {len(mig)} peaks")


def build_atac():
    mig = set(json.load(open(SS / "mig_modules.json"))["module_peak_rows"])
    bcs = [l.strip() for l in gzip.open(ATAC_BC, "rt")]
    ncell = len(bcs)
    modc = np.zeros(ncell + 1); totc = np.zeros(ncell + 1)
    with gzip.open(ATAC_MTX, "rt") as f:
        assert f.readline().lower().startswith("%%matrixmarket")
        f.readline()
        for k, line in enumerate(f):
            r, c, v = line.split(); c = int(c); v = int(v)
            totc[c] += v
            if int(r) in mig:
                modc[c] += v
            if k % 30_000_000 == 0 and k:
                print(f"  ...{k//1_000_000}M nonzeros")
    acc = modc[1:] / np.clip(totc[1:], 1, None)
    np.savez(SS / "mig_atac.npz", bc=np.array(bcs), acc=acc)
    print(f"saved mig_atac.npz: {ncell} cells, mean motility-accessibility {acc.mean():.4f}")


def build_rna():
    mig = set(MIG_GENES)
    with gzip.open(RNA, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        bcs = [b.replace(",", ".") for b in header[1:]]; n = len(bcs)
        mod = np.zeros(n); tot = np.zeros(n)
        for k, line in enumerate(f):
            i = line.find("\t"); gene = line[:i]
            vals = np.fromstring(line[i + 1:], sep="\t")
            if vals.shape[0] != n:
                vals = np.resize(vals, n)
            tot += vals
            if gene in mig:
                mod += vals
            if k % 5000 == 0 and k:
                print(f"  ...{k} genes")
    score = mod / np.clip(tot, 1, None)
    np.savez(SS / "mig_rna.npz", bc=np.array(bcs), score=score)
    print(f"saved mig_rna.npz: {n} cells, mean motility score {score.mean():.4f}")


def fit():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    A = np.load(SS / "mig_atac.npz", allow_pickle=True); R = np.load(SS / "mig_rna.npz", allow_pickle=True)
    rrr = lambda bc: ".".join(bc.split(".")[:6])
    acc = dict(zip([rrr(b) for b in A["bc"].astype(str)], A["acc"]))
    score = dict(zip([rrr(b) for b in R["bc"].astype(str)], R["score"]))
    xs, ys, cts = [], [], []
    with gzip.open(CELLTYPE, "rt") as f:
        next(f)
        for line in f:
            abc, rbc, ct = line.rstrip("\n").split("\t"); k = rrr(abc)
            if k in acc and k in score:
                xs.append(acc[k]); ys.append(score[k]); cts.append(ct)
    x = np.array(xs); y = np.array(ys); cts = np.array(cts)
    xz = (x - x.mean()) / (x.std() + 1e-12); yz = (y - y.mean()) / (y.std() + 1e-12)
    r = float(np.corrcoef(xz, yz)[0, 1])
    rng = np.random.RandomState(0); idx = rng.permutation(len(x))
    tr, te = idx[:int(0.7 * len(x))], idx[int(0.7 * len(x)):]
    b, a0 = np.polyfit(xz[tr], yz[tr], 1); pred = a0 + b * xz[te]
    r2 = float(1 - np.sum((yz[te] - pred) ** 2) / np.sum((yz[te] - yz[te].mean()) ** 2))
    null = np.array([abs(np.corrcoef(xz, rng.permutation(yz))[0, 1]) for _ in range(200)])
    print(f"n paired cells: {len(x)}")
    print(f"corr(motility ACCESSIBILITY, motility-program SCORE) = {r:+.3f}  (|null| {null.mean():.3f}+-{null.std():.3f})")
    print(f"held-out R^2 (1-feature) = {r2:.3f}")
    print("\nper-celltype means (migratory -> sessile):")
    print(f"  {'celltype':26s} {'ATAC-acc':>9} {'RNA-score':>10} {'n':>6}")
    rows = []
    for ct in GRAD:
        m = cts == ct
        if m.sum() > 20:
            rows.append((ct, x[m].mean(), y[m].mean(), int(m.sum())))
            print(f"  {ct:26s} {x[m].mean():9.4f} {y[m].mean():10.4f} {int(m.sum()):6d}")
    _figure(x, y, r, r2, null, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_cells=len(x), corr=r, heldout_r2=r2, null_mean=float(null.mean()),
                   null_std=float(null.std()), genes=MIG_GENES,
                   celltype_means={ct: dict(atac=a, rna=b, n=n) for ct, a, b, n in rows}),
              open(OUT / "shareseq_mig_head.json", "w"), indent=2)
    print("\nsaved", OUT / "shareseq_mig_head.png")
    return r, r2, null


def _figure(x, y, r, r2, null, rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
    ax[0].hexbin(x, y, gridsize=45, cmap="viridis", bins="log")
    ax[0].set_xlabel("motility-module ATAC accessibility (fraction)")
    ax[0].set_ylabel("motility-program RNA score (fraction)")
    ax[0].set_title(f"(a) per-cell, {len(x)} cells\ncorr = {r:+.2f}  (|null| {null.mean():.2f})", fontsize=9)
    labels = [r_[0] for r_ in rows]
    xa = np.array([r_[1] for r_ in rows]); ya = np.array([r_[2] for r_ in rows])
    ax[1].scatter(xa, ya, s=60, color="tab:green")
    for i, lab in enumerate(labels):
        ax[1].annotate(lab, (xa[i], ya[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("mean motility-module ATAC accessibility")
    ax[1].set_ylabel("mean motility-program RNA score")
    ax[1].set_title("(b) the motility PROGRAM across cell types\n(mesenchymal high, epithelial low) -- program, not movement", fontsize=9)
    fig.suptitle("Migration head (Paper #6 sec:heads): the molecular motility PROGRAM is traceable from accessibility, "
                 "but the movement itself is not in a dissociated atlas -- the missing observable.", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "shareseq_mig_head.png", dpi=140, bbox_inches="tight")
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
        r, r2, null = fit()
        print(f"\nRESULT: program corr {r:+.3f} (movement itself = the missing observable, absent here)")
