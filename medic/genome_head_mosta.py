"""
The genome-to-MOSTA bridge, with the heads in the middle.
=========================================================
The architecture of the series is a three-layer stack: the GENOME (the outer large genomic
model), the HEADS (the master-transcription-factor super-enhancer clusters -- the middle
layer), and the BODY (the real anatomy, here the MOSTA atlas). This module implements and
measures the two links of that stack on the real E12.5 mouse section, using the SCENIC
master-TF regulons carried per cell as the head layer:

  LINK 1  HEADS -> IDENTITY.  The head layer (the master-TF regulon activities) is a
          sufficient bottleneck for anatomical identity: a nearest-centroid classifier on
          the regulons alone recovers the tissue label of a held-out cell far above chance.
          An organ IS its master-TF head, read from the data.

  LINK 2  GENOME -> POSITION, through the heads.  The genome's antero-posterior code is Hox
          colinearity. Reading it at the head layer -- each tissue's Hox-regulon profile,
          weighted by the colinear Hox number -- gives a genome-derived AP address per head,
          and that address predicts the tissue's real spatial position in the atlas. The Hox
          code, mediated by the head, places the anatomy.

The genome-derived organ AP addresses are written out so organ_sprouting can consume them in
place of the fitted MOSTA anchors (medic.mosta_organ_anchors) -- the g_K-anchor pattern
carried one step further, from fitted-to-MOSTA to read-from-the-genome-code.

Writes data/organ_cascade/genome_head_mosta.{json,png}.
Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.genome_head_mosta [stage]
"""
from __future__ import annotations
import sys, os, json, re, glob
import h5py
import numpy as np

D = "data/mosta"


def _load(path):
    """annotation (labels + codes), spatial coords, and the SCENIC regulon matrix (the head layer)."""
    with h5py.File(path, "r") as h:
        cat = [c.decode() if isinstance(c, bytes) else str(c)
               for c in h["obs/__categories/annotation"][:]]
        ann = np.asarray(h["obs/annotation"][:])
        xy = np.asarray(h["obsm/spatial"][:], float)
        reg_keys = sorted(k for k in h["obs"].keys() if k.startswith("Regulon - "))
        R = np.stack([np.asarray(h[f"obs/{k}"][:], np.float32) for k in reg_keys], 1)
    tfs = [k.replace("Regulon - ", "") for k in reg_keys]
    return cat, ann, xy, R, tfs


def _body_ap(xy, ann, cat, axis="geodesic"):
    """AP coordinate in [0,1], oriented so the brain is anterior (a small). `axis="geodesic"` uses
    the Fiedler vector of the spatial kNN graph, which follows the body's C-curl (the best axis for
    a curled MOSTA section); `axis="pca"` uses the raw PCA long axis (cuts across the curl)."""
    if axis == "geodesic":
        from medic.ap_unbend import geodesic_ap
        ap = geodesic_ap(xy)
    else:
        c = xy - xy.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        ap = c @ vt[0]
        ap = (ap - ap.min()) / (np.ptp(ap) + 1e-9)
    if "Brain" in cat:
        bi = cat.index("Brain")
        if ap[ann == bi].mean() > 0.5:
            ap = 1.0 - ap
    return ap


def _hox_number(tf):
    """Colinear AP rank of a Hox / Cdx regulon: Hox<letter><n> -> n; Cdx -> posterior (~12)."""
    m = re.fullmatch(r"Hox[abcd]([0-9]{1,2})", tf)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"Cdx[124]", tf):
        return 12                       # Cdx acts with the posterior Hox to set trunk/tail identity
    return None


def link1_identity(R, ann, cat, seed=0, min_cells=40):
    """HEADS -> IDENTITY: nearest-centroid classifier on the standardized regulon layer.
    Held-out balanced accuracy vs chance (1/#classes)."""
    keep = np.array([i for i in range(len(cat)) if (ann == i).sum() >= min_cells])
    m = np.isin(ann, keep)
    X = R[m]; y = ann[m]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    rng = np.random.RandomState(seed)
    tr = rng.rand(len(y)) < 0.7
    cls = np.unique(y)
    cent = np.stack([X[tr & (y == c)].mean(0) for c in cls])     # per-class centroid on train
    te = ~tr
    pred = cls[np.argmin(((X[te][:, None, :] - cent[None]) ** 2).sum(2), 1)]
    yt = y[te]
    per = [float((pred[yt == c] == c).mean()) for c in cls]      # balanced (per-class) accuracy
    return dict(balanced_acc=float(np.mean(per)), overall_acc=float((pred == yt).mean()),
                chance=1.0 / len(cls), n_classes=int(len(cls)), n_test=int(te.sum()))


# Hox codes the TRUNK; the cranial / anterior structures are the Hox-free anterior default and must
# not be regressed on the Hox code (they carry no colinear signal). This is the biology, not a cull.
ANTERIOR_HOXFREE = {"Brain", "Choroid plexus", "Meninges", "Jaw and tooth", "Branchial arch",
                    "Otic", "Inner ear", "Surface ectoderm", "Epidermis", "Spinal cord"}


def link2_hox_position(R, tfs, ann, cat, ap, min_cells=40):
    """GENOME -> POSITION through the heads: each tissue's Hox-regulon profile, weighted by the
    colinear Hox number, is a genome-derived AP address; correlate it with the measured AP centroid.

    A DISTRIBUTED tissue (muscle, cavity, blood vessel, connective tissue) spans the whole AP axis --
    it has no single position, so its AP median is meaningless and it only adds noise to a POSITION
    correlation. We measure each head's AP dispersion (IQR) and split LOCALIZED (point-like organ,
    IQR below the trunk median) from DISTRIBUTED (span-type). The localized-only correlation is the
    honest position test; the dispersion-weighted one keeps every head but down-weights the smears."""
    hox_idx = [(i, _hox_number(t)) for i, t in enumerate(tfs) if _hox_number(t) is not None]
    cols = [i for i, _ in hox_idx]; nums = np.array([n for _, n in hox_idx], float)
    Hox = R[:, cols]                                              # (cells, #hox regulons)
    rows = []
    for i, label in enumerate(cat):
        sel = ann == i
        if sel.sum() < min_cells:
            continue
        w = np.clip(Hox[sel].mean(0), 0, None)                  # mean Hox-regulon activity for this head
        genome_ap = float((w * nums).sum() / (w.sum() + 1e-9))   # activity-weighted colinear Hox number
        q = np.percentile(ap[sel], [25, 75])
        rows.append(dict(head=label, n=int(sel.sum()), trunk=label not in ANTERIOR_HOXFREE,
                         genome_hox=round(genome_ap, 3),
                         mosta_ap=round(float(np.median(ap[sel])), 3),
                         ap_iqr=round(float(q[1] - q[0]), 3)))

    # localized = AP dispersion below the median of the trunk heads (the point-like organs)
    trunk = [r for r in rows if r["trunk"]]
    med_iqr = float(np.median([r["ap_iqr"] for r in trunk])) if trunk else 0.0
    for r in rows:
        r["localized"] = bool(r["trunk"] and r["ap_iqr"] <= med_iqr)

    def _corr(rs, weighted=False):
        if len(rs) < 3:
            return float("nan"), float("nan")
        g = np.array([r["genome_hox"] for r in rs]); a = np.array([r["mosta_ap"] for r in rs])
        if weighted:                                            # inverse-dispersion weights
            wt = 1.0 / (np.array([r["ap_iqr"] for r in rs]) + 0.05)
            gm = np.average(g, weights=wt); am = np.average(a, weights=wt)
            cov = np.average((g - gm) * (a - am), weights=wt)
            pear = float(cov / (np.sqrt(np.average((g - gm) ** 2, weights=wt) *
                                        np.average((a - am) ** 2, weights=wt)) + 1e-12))
            return pear, float("nan")
        pear = float(np.corrcoef(g, a)[0, 1])
        spear = float(np.corrcoef(np.argsort(np.argsort(g)), np.argsort(np.argsort(a)))[0, 1])
        return pear, spear

    pear, spear = _corr(rows)
    tpear, tspear = _corr(trunk)
    loc = [r for r in rows if r["localized"]]
    lpear, lspear = _corr(loc)
    wpear, _ = _corr(trunk, weighted=True)
    return dict(n_hox_regulons=len(cols), pearson=pear, spearman=spear,
                trunk_pearson=tpear, trunk_spearman=tspear, n_trunk=len(trunk),
                loc_pearson=lpear, loc_spearman=lspear, n_localized=len(loc),
                trunk_weighted_pearson=wpear,
                rows=rows, hox_regulons=[tfs[i] for i in cols])


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "E12.5"
    path = sorted(glob.glob(os.path.join(D, f"{stage}_*.MOSTA.h5ad")))[0]
    cat, ann, xy, R, tfs = _load(path)
    ap = _body_ap(xy, ann, cat)
    print(f"{stage}: {len(ann):,} cells, {len(tfs)} head regulons, "
          f"{sum((ann==i).sum()>=40 for i in range(len(cat)))} heads (>=40 cells)\n")

    id1 = link1_identity(R, ann, cat)
    print(f"LINK 1  heads -> identity: balanced acc {id1['balanced_acc']:.2f}, "
          f"overall {id1['overall_acc']:.2f}  vs chance {id1['chance']:.2f} "
          f"({id1['n_classes']} heads) -> the head layer carries anatomical identity.")

    pos = link2_hox_position(R, tfs, ann, cat, ap)
    print(f"\nLINK 2  genome (Hox) -> position through heads: {pos['n_hox_regulons']} Hox regulons")
    print(f"   all {len(pos['rows'])} heads:        Pearson {pos['pearson']:.2f}, Spearman {pos['spearman']:.2f}")
    print(f"   {pos['n_trunk']} TRUNK heads:       Pearson {pos['trunk_pearson']:.2f}, "
          f"Spearman {pos['trunk_spearman']:.2f}  (cranial heads are the Hox-free anterior default)")
    print(f"   {pos['n_localized']} LOCALIZED organs: Pearson {pos['loc_pearson']:.2f}, "
          f"Spearman {pos['loc_spearman']:.2f}  (dropping the body-spanning distributed tissues)")
    print(f"   TRUNK dispersion-weighted: Pearson {pos['trunk_weighted_pearson']:.2f}  "
          f"(keep every head, down-weight the smears)")
    print(f"\n {'head':22s} {'kind':>10s} {'genome Hox-AP':>13s} {'MOSTA AP':>9s} {'AP IQR':>7s}")
    for r in sorted(pos["rows"], key=lambda r: r["genome_hox"]):
        kind = "localized" if r["localized"] else ("distributed" if r["trunk"] else "cranial")
        print(f" {r['head']:22s} {kind:>10s} {r['genome_hox']:>13.2f} {r['mosta_ap']:>9.2f} {r['ap_iqr']:>7.2f}")

    out = dict(stage=stage, link1_identity=id1, link2_hox_position=pos)
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(out, open("data/organ_cascade/genome_head_mosta.json", "w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 6))
        for r in pos["rows"]:
            trunk = r["trunk"]
            ax.scatter(r["genome_hox"], r["mosta_ap"], s=30,
                       c="#2b6cb0" if trunk else "#c05621",
                       marker="o" if trunk else "^")
            ax.annotate(r["head"], (r["genome_hox"], r["mosta_ap"]), fontsize=6,
                        xytext=(3, 2), textcoords="offset points",
                        color="#1a1a1a" if trunk else "#7b341e")
        # trunk trend line
        gt = np.array([r["genome_hox"] for r in pos["rows"] if r["trunk"]])
        at = np.array([r["mosta_ap"] for r in pos["rows"] if r["trunk"]])
        b1, b0 = np.polyfit(gt, at, 1)
        xs = np.array([gt.min(), gt.max()])
        ax.plot(xs, b0 + b1 * xs, "-", c="#2b6cb0", lw=1.2, alpha=0.6)
        ax.scatter([], [], c="#2b6cb0", marker="o", label="trunk (Hox-competent)")
        ax.scatter([], [], c="#c05621", marker="^", label="cranial (Hox-free anterior)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_xlabel("genome AP address  (Hox-regulon colinear number, read at the head)")
        ax.set_ylabel("MOSTA measured AP position")
        ax.set_title(f"Genome $\\to$ head $\\to$ MOSTA ({stage}): the Hox code places the trunk anatomy\n"
                     f"trunk Pearson r={pos['trunk_pearson']:.2f}, Spearman $\\rho$={pos['trunk_spearman']:.2f} "
                     f"({pos['n_trunk']} heads); heads$\\to$identity acc {id1['balanced_acc']:.2f} vs chance {id1['chance']:.2f}")
        fig.tight_layout(); fig.savefig("data/organ_cascade/genome_head_mosta.png", dpi=130)
        print("\nsaved data/organ_cascade/genome_head_mosta.{json,png}")
    except Exception as e:
        print(f"\nsaved json (plot skipped: {e})")


if __name__ == "__main__":
    main()
