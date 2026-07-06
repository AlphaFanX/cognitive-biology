"""
First fit of the DIFFERENTIATION head from a multiome trajectory (SHARE-seq skin).
==================================================================================

Paper #6 sec:heads, made concrete. Trace one head from the genome as a low-rank map:
  fate_logit_k = sum_m  a_m * d_{m,k}
where a_m = per-cell ACCESSIBILITY of master-TF module m (the LGM's scalar, measured here
directly from scATAC), the module->peak structure is FROZEN (master-TF cis windows), and the
direction d is FIT so accessibility predicts fate. Held-out cells test whether accessibility
predicts fate through the low-rank map -- i.e. whether one head's weights are traceable from
the genome. Data: GSE140203 GSM4156597 skin.late.anagen (hair-follicle lineage).

Stages (cached):
  1 build_modules  -> data/shareseq/modules.json         (needs skin_peaks.bed.gz + mm10_refGene.txt.gz)
  2 build_access   -> data/shareseq/access.npz            (needs skin_atac_fragments.bed.gz + celltype)
  3 fit            -> data/organ_cascade/shareseq_diff_head.{png,json}

Run: python -m medic.shareseq_diff_head {modules|access|fit|all}
"""
from __future__ import annotations
import gzip, json, sys
from pathlib import Path
import numpy as np

SS = Path("data/shareseq")
OUT = Path("data/organ_cascade")
PEAKS = SS / "skin_peaks.bed.gz"
REFGENE = SS / "mm10_refGene.txt.gz"
FRAG = SS / "skin_atac_fragments.bed.gz"
CELLTYPE = SS / "skin_celltype.txt.gz"
WINDOW = 50_000                       # +/- cis window around each master-TF TSS

# Master-TF modules per differentiated fate (hair-follicle lineage; well-established TFs).
MODULES = {
    "Cortex":  ["Lef1", "Hoxc13", "Foxn1"],     # hair shaft / cortex
    "IRS":     ["Gata3", "Cux1", "Dlx3"],        # inner root sheath (Gata3 = master)
    "Medulla": ["Foxq1", "Msx2"],                # medulla
    "ORS":     ["Sox9", "Lhx2", "Nfatc1"],       # outer root sheath / bulge stem
}
# map the atlas celltype label -> our fate module (differentiated lineage only, v1)
FATE_OF = {
    "Hair Shaft-cuticle.cortex": "Cortex",
    "IRS": "IRS",
    "Medulla": "Medulla",
    "ORS": "ORS",
}
MODNAMES = list(MODULES)


def _tss_table():
    tss = {}
    with gzip.open(REFGENE, "rt") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            chrom, strand, txStart, txEnd, name2 = c[2], c[3], int(c[4]), int(c[5]), c[12]
            t = txStart if strand == "+" else txEnd
            tss.setdefault(name2, (chrom, t))     # first occurrence
    return tss


def build_modules():
    SS.mkdir(parents=True, exist_ok=True)
    tss = _tss_table()
    # module cis windows per chromosome
    windows = {m: [] for m in MODULES}
    for m, genes in MODULES.items():
        for g in genes:
            if g not in tss:
                print(f"  WARN gene {g} not in refGene"); continue
            chrom, t = tss[g]
            windows[m].append((chrom, t - WINDOW, t + WINDOW, g))
    # assign each peak to the modules whose window it falls in
    modpeaks = {m: [] for m in MODULES}
    npk = 0
    with gzip.open(PEAKS, "rt") as f:
        for i, line in enumerate(f):
            c = line.split("\t")
            if len(c) < 3:
                continue
            chrom, s, e = c[0], int(c[1]), int(c[2])
            mid = (s + e) // 2
            npk += 1
            for m, ws in windows.items():
                for (wc, ws0, we0, g) in ws:
                    if wc == chrom and ws0 <= mid <= we0:
                        modpeaks[m].append([chrom, s, e]); break
    for m in MODULES:
        print(f"  module {m:8s} genes {MODULES[m]}  -> {len(modpeaks[m])} peaks")
    json.dump({"window": WINDOW, "modules": MODULES, "modpeaks": modpeaks, "n_peaks_total": npk},
              open(SS / "modules.json", "w"))
    print(f"total peaks scanned {npk}; saved {SS/'modules.json'}")


def _load_lineage_cells():
    """cell atac-barcode -> fate (only lineage cells we model)."""
    bc2fate = {}
    with gzip.open(CELLTYPE, "rt") as f:
        next(f)
        for line in f:
            atac_bc, rna_bc, ct = line.rstrip("\n").split("\t")
            if ct in FATE_OF:
                bc2fate[atac_bc] = FATE_OF[ct]
    return bc2fate


def build_access():
    mods = json.load(open(SS / "modules.json"))["modpeaks"]
    # per-chromosome sorted peak arrays per module (for fast overlap)
    chr_mod = {}                                  # chrom -> list of (start,end,module_idx)
    for mi, m in enumerate(MODNAMES):
        for chrom, s, e in mods[m]:
            chr_mod.setdefault(chrom, []).append((s, e, mi))
    for chrom in chr_mod:
        chr_mod[chrom].sort()
    bc2fate = _load_lineage_cells()
    cells = {}                                    # atac_bc -> [total_frags, per-module frags...]
    n_line = len(bc2fate)
    print(f"lineage cells to track: {n_line}")

    op = gzip.open(FRAG, "rt")
    for k, line in enumerate(op):
        c = line.split("\t")
        if len(c) < 4:
            continue
        bc = c[3].strip().replace(",", ".")     # fragments use commas; celltype file uses dots
        fate = bc2fate.get(bc)
        if fate is None:
            continue
        chrom = c[0]; s = int(c[1]); e = int(c[2])
        rec = cells.get(bc)
        if rec is None:
            rec = [0] + [0] * len(MODNAMES); cells[bc] = rec
        rec[0] += 1                               # total fragments (library size)
        segs = chr_mod.get(chrom)
        if segs:
            mid = (s + e) // 2
            for (ps, pe, mi) in segs:
                if ps <= mid <= pe:
                    rec[1 + mi] += 1; break
        if k % 20_000_000 == 0 and k:
            print(f"  ...{k//1_000_000}M fragments, {len(cells)} lineage cells hit")
    op.close()

    bcs = [b for b in cells if cells[b][0] >= 200]     # min library size
    X = np.array([[cells[b][1 + mi] / cells[b][0] for mi in range(len(MODNAMES))] for b in bcs])
    y = np.array([bc2fate[b] for b in bcs])
    np.savez(SS / "access.npz", X=X, y=y, cells=np.array(bcs), modules=np.array(MODNAMES))
    print(f"saved {SS/'access.npz'}: {X.shape[0]} cells x {X.shape[1]} modules; fates {dict(zip(*np.unique(y, return_counts=True)))}")


def fit():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import torch, torch.nn as nn
    d = np.load(SS / "access.npz", allow_pickle=True)
    X, y, mods = d["X"].astype(np.float32), d["y"].astype(str), list(d["modules"].astype(str))
    classes = sorted(set(y))
    yi = np.array([classes.index(t) for t in y])
    Xs = ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)
    # stratified 70/30 split
    rng = np.random.RandomState(0)
    tr = np.zeros(len(yi), bool)
    for k in range(len(classes)):
        idx = np.where(yi == k)[0]; rng.shuffle(idx); tr[idx[:int(0.7 * len(idx))]] = True
    te = ~tr

    def _train(Xtr, ytr, Xte, yte, epochs=400):
        torch.manual_seed(0)
        net = nn.Linear(Xtr.shape[1], len(classes))
        opt = torch.optim.Adam(net.parameters(), lr=0.05, weight_decay=1e-3)
        lf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xtr); yt = torch.tensor(ytr, dtype=torch.long)
        for _ in range(epochs):
            opt.zero_grad(); lf(net(Xt), yt).backward(); opt.step()
        with torch.no_grad():
            pred = net(torch.tensor(Xte)).argmax(1).numpy()
        return float((pred == yte).mean()), net.weight.detach().numpy()

    acc, Wd = _train(Xs[tr], yi[tr], Xs[te], yi[te])   # Wd (n_fate, n_module)
    null = []
    for s in range(20):
        rr = np.random.RandomState(s + 1); yp = yi[tr].copy(); rr.shuffle(yp)
        null.append(_train(Xs[tr], yp, Xs[te], yi[te], epochs=250)[0])
    null = np.array(null)
    print(f"held-out fate accuracy = {acc:.3f}  vs shuffled-label null {null.mean():.3f}+-{null.std():.3f} "
          f"(chance={1/len(classes):.3f})")
    # the learned direction matrix (module -> fate): does each module point at its own fate?
    print("\nlearned direction d[module->fate] (rows=fate, cols=module):")
    print("            " + "".join(f"{m:>9}" for m in mods))
    diag_ok = 0
    for i, fate in enumerate(classes):
        row = Wd[i]
        print(f"  {fate:9s} " + "".join(f"{row[j]:+9.2f}" for j in range(len(mods))))
        # is the strongest module for this fate its OWN module?
        if fate in mods and int(np.argmax(row)) == mods.index(fate):
            diag_ok += 1
    print(f"\nmodules whose top fate is their own: {diag_ok}/{len(classes)} "
          f"(the accessibility->fate directions are self-consistent)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    im = ax[0].imshow(Wd, cmap="RdBu_r", vmin=-abs(Wd).max(), vmax=abs(Wd).max(), aspect="auto")
    ax[0].set_xticks(range(len(mods))); ax[0].set_xticklabels(mods, rotation=45, fontsize=8)
    ax[0].set_yticks(range(len(classes))); ax[0].set_yticklabels(classes, fontsize=8)
    ax[0].set_title("learned direction d[module->fate]\n(diagonal = accessibility predicts own fate)", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=0.045)
    ax[1].bar(["accessibility\nmodel", "shuffled-label\nnull", "chance"],
              [acc, null.mean(), 1/len(set(y))], color=["tab:green", "0.6", "0.8"],
              yerr=[0, null.std(), 0], capsize=4)
    ax[1].set_ylabel("held-out fate accuracy"); ax[1].set_ylim(0, 1)
    ax[1].set_title(f"differentiation head from genome accessibility\n{X.shape[0]} SHARE-seq hair-follicle cells", fontsize=9)
    fig.suptitle("First fit of the differentiation head (Paper #6 sec:heads): per-cell master-TF module accessibility "
                 "predicts held-out hair-follicle fate through a low-rank map.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "shareseq_diff_head.png", dpi=140, bbox_inches="tight")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(heldout_acc=float(acc), null_mean=float(null.mean()), null_std=float(null.std()),
                   chance=1/len(set(y)), n_cells=int(X.shape[0]), modules=mods, fates=classes,
                   direction=Wd.tolist(), diag_selfconsistent=int(diag_ok)),
              open(OUT / "shareseq_diff_head.json", "w"), indent=2)
    print("\nsaved", OUT / "shareseq_diff_head.png")
    return acc > null.mean() + 3 * (null.std() + 1e-6)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("modules", "all"):
        build_modules()
    if stage in ("access", "all"):
        build_access()
    if stage in ("fit", "all"):
        ok = fit()
        print(f"\nRESULT: {'TRACED (accessibility predicts held-out fate)' if ok else 'CHECK'}")
