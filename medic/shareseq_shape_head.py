"""
Cloning the head-tracing recipe to the SHAPE / FOLDING head (SHARE-seq + the physics).
=======================================================================================

Paper #6 sec:heads, fourth and last clone -- and the one that realizes the paper's title.
The shape head has TWO parts, at two scales, because unlike the other three its observable
is in no molecular atlas at all:

  PART A  the genome sets the mechanical PARAMETER.  Exactly as for the other heads, an
          adhesion module (E-cadherin, P-cadherin, the desmosomal plaque, tight junctions,
          alpha/delta catenins) has an accessibility that predicts its own expression in the
          same cell -- the genome-set cell-cell adhesion. Epithelial cells high, mesenchymal
          low; measured on SHARE-seq like fate/division/migration.

  PART B  the parameter -> the SHAPE is COMPUTED, not measured.  The adhesion parameter is
          fed to the fixed mechanical operator (medic.mechanical_fusion): two growing fronts
          fuse into a lip or persist as a cleft, and which one is read off the Fiedler
          eigenvalue lambda_2 of the tissue graph (0 = two pieces / cleft; >0 = one piece /
          fused). Sweeping the genome-set adhesion traverses the cleft<->fuse basin. This
          observable -- did it fold, did it fuse -- is in NO transcriptomic or multiome
          atlas; it is computed by the physics. The shape head is therefore the full chain:
          genome -> adhesion parameter (measured) -> fold/cleft (computed).

Run: python -m medic.shareseq_shape_head {modules|atac|rna|fitA|partB|all}
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

# cell-cell adhesion module (the mechanical parameter the fold/fuse operator consumes)
ADH_GENES = ["Cdh1", "Cdh3", "Dsp", "Pkp1", "Pkp3", "Dsg3", "Dsc3", "Jup",
             "Cldn4", "Ocln", "Ctnna1", "Ctnnd1", "Perp", "Krt5", "Krt14"]
# epithelial (high adhesion) -> mesenchymal (low)
GRAD = ["Basal", "Spinous", "Granular", "IRS", "Medulla", "Hair Shaft-cuticle.cortex",
        "ORS", "ahighCD34+ bulge", "Dermal Papilla", "Endothelial", "Dermal Sheath",
        "Dermal Fibroblast"]


def _tss():
    tss = {}
    with gzip.open(REFGENE, "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            tss.setdefault(c[12], (c[2], int(c[4]) if c[3] == "+" else int(c[5])))
    return tss


def build_modules():
    tss = _tss(); wins = []
    for g in ADH_GENES:
        if g in tss:
            chrom, t = tss[g]; wins.append((chrom, t - WINDOW, t + WINDOW))
        else:
            print(f"  WARN {g} not in refGene")
    adh = []
    with gzip.open(PEAKS, "rt") as f:
        for i, line in enumerate(f, start=1):
            c = line.split("\t")
            if len(c) < 3:
                continue
            chrom = c[0]; mid = (int(c[1]) + int(c[2])) // 2
            for (wc, s, e) in wins:
                if wc == chrom and s <= mid <= e:
                    adh.append(i); break
    json.dump({"genes": ADH_GENES, "module_peak_rows": adh}, open(SS / "shape_modules.json", "w"))
    print(f"adhesion module: {len(ADH_GENES)} genes -> {len(adh)} peaks")


def build_atac():
    adh = set(json.load(open(SS / "shape_modules.json"))["module_peak_rows"])
    bcs = [l.strip() for l in gzip.open(ATAC_BC, "rt")]; n = len(bcs)
    modc = np.zeros(n + 1); totc = np.zeros(n + 1)
    with gzip.open(ATAC_MTX, "rt") as f:
        assert f.readline().lower().startswith("%%matrixmarket"); f.readline()
        for k, line in enumerate(f):
            r, c, v = line.split(); c = int(c); v = int(v)
            totc[c] += v
            if int(r) in adh:
                modc[c] += v
    np.savez(SS / "shape_atac.npz", bc=np.array(bcs), acc=modc[1:] / np.clip(totc[1:], 1, None))
    print(f"saved shape_atac.npz: {n} cells")


def build_rna():
    adh = set(ADH_GENES)
    with gzip.open(RNA, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        bcs = [b.replace(",", ".") for b in header[1:]]; n = len(bcs)
        mod = np.zeros(n); tot = np.zeros(n)
        for line in f:
            i = line.find("\t"); gene = line[:i]
            vals = np.fromstring(line[i + 1:], sep="\t")
            if vals.shape[0] != n:
                vals = np.resize(vals, n)
            tot += vals
            if gene in adh:
                mod += vals
    np.savez(SS / "shape_rna.npz", bc=np.array(bcs), score=mod / np.clip(tot, 1, None))
    print(f"saved shape_rna.npz: {n} cells")


def part_A():
    """Genome sets the adhesion parameter: accessibility -> adhesion expression (SHARE-seq)."""
    A = np.load(SS / "shape_atac.npz", allow_pickle=True); R = np.load(SS / "shape_rna.npz", allow_pickle=True)
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
    rng = np.random.RandomState(0)
    null = np.array([abs(np.corrcoef(xz, rng.permutation(yz))[0, 1]) for _ in range(200)])
    rows = [(ct, x[cts == ct].mean(), y[cts == ct].mean(), int((cts == ct).sum()))
            for ct in GRAD if (cts == ct).sum() > 20]
    print(f"PART A  genome sets adhesion: corr(accessibility, adhesion expression) = {r:+.3f} "
          f"(|null| {null.mean():.3f})  n={len(x)}")
    for ct, a, b, nn in rows:
        print(f"    {ct:26s} acc {a:.4f}  expr {b:.4f}  n {nn}")
    return dict(x=x, y=y, r=r, null=float(null.mean()), rows=rows)


def part_B():
    """The parameter -> shape is COMPUTED: adhesion -> fold/cleft via Fiedler lambda_2."""
    from medic.mechanical_fusion import cleft_sweep, prominence_fusion
    sw = cleft_sweep(n=11, steps=800)
    # at fixed sufficient outgrowth, the fusion outcome vs the genome-set adhesion
    lo = prominence_fusion(outgrowth=1.0, adhesion=0.15, steps=1000)
    hi = prominence_fusion(outgrowth=1.0, adhesion=1.0, steps=1000)
    print(f"PART B  adhesion -> shape (computed by Fiedler): "
          f"low adhesion -> lambda_2={lo['lam2']:.3f} ncomp={lo['ncomp']} ({'cleft' if not lo['fused'] else 'fused'}); "
          f"high adhesion -> lambda_2={hi['lam2']:.3f} ncomp={hi['ncomp']} ({'fused' if hi['fused'] else 'cleft'})")
    return dict(sweep=sw, lo=lo, hi=hi)


def figure(A, B):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # (a) part A per-cell
    ax[0].hexbin(A["x"], A["y"], gridsize=45, cmap="cividis", bins="log")
    ax[0].set_xlabel("adhesion-module ATAC accessibility"); ax[0].set_ylabel("adhesion expression")
    ax[0].set_title(f"(a) genome sets the adhesion parameter\ncorr {A['r']:+.2f} (|null| {A['null']:.2f}), {len(A['x'])} cells", fontsize=9)
    # (b) part A gradient
    labs = [r[0] for r in A["rows"]]; xa = np.array([r[1] for r in A["rows"]]); ya = np.array([r[2] for r in A["rows"]])
    ax[1].scatter(xa, ya, s=55, color="tab:blue")
    for i, l in enumerate(labs):
        ax[1].annotate(l.replace("Hair Shaft-cuticle.cortex", "Cortex"), (xa[i], ya[i]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[1].set_xlabel("mean adhesion accessibility"); ax[1].set_ylabel("mean adhesion expression")
    ax[1].set_title("(b) epithelial high, mesenchymal low\n(adhesion = identity, chromatin-encoded)", fontsize=9)
    # (c) part B: the computed shape observable -- the cleft/fuse basin
    sw = B["sweep"]
    im = ax[2].imshow(sw["lam2"], origin="lower", aspect="auto", cmap="RdYlGn",
                      extent=[sw["as_"][0], sw["as_"][-1], sw["gs"][0], sw["gs"][-1]])
    ax[2].set_xlabel("genome-set ADHESION (part A axis)"); ax[2].set_ylabel("outgrowth")
    ax[2].set_title("(c) COMPUTED observable: fold/cleft\nFiedler $\\lambda_2$ (0=cleft, >0=fused)", fontsize=9)
    fig.colorbar(im, ax=ax[2], fraction=0.045, label="$\\lambda_2$")
    fig.suptitle("Shape/folding head: the genome sets the adhesion parameter (a,b, measured on SHARE-seq); the fold-or-cleft "
                 "SHAPE it produces (c) is COMPUTED by the mechanical operator -- an observable in no molecular atlas.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "shareseq_shape_head.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(partA_corr=A["r"], partA_null=A["null"], n_cells=len(A["x"]),
                   celltype=[(r[0], r[1], r[2], r[3]) for r in A["rows"]],
                   partB_low_adh_lam2=B["lo"]["lam2"], partB_low_fused=bool(B["lo"]["fused"]),
                   partB_high_adh_lam2=B["hi"]["lam2"], partB_high_fused=bool(B["hi"]["fused"])),
              open(OUT / "shareseq_shape_head.json", "w"), indent=2)
    print("saved", OUT / "shareseq_shape_head.png")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("modules", "all"):
        build_modules()
    if stage in ("atac", "all"):
        build_atac()
    if stage in ("rna", "all"):
        build_rna()
    if stage in ("fitA", "partB", "all"):
        A = part_A(); B = part_B(); figure(A, B)
        print(f"\nRESULT: genome->adhesion corr {A['r']:+.3f} (measured); adhesion->fold/cleft = COMPUTED (Fiedler), in no atlas")
