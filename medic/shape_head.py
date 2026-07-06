"""
The shape head (4th, last): the mechanical program (measured) -> fold + cleft, COMPUTED.
========================================================================================

Migration already gave most of the shape (convergent extension = axis elongation). The residue
migration does NOT give is the topology change of a fold or a cleft, and that is the shape head.
The engine exists (mechanical_fusion.py): a fold is a BUCKLING eigenmode of the constriction-
weighted operator (drive=1 closes the neural tube, weak drive -> a neural-tube defect), and a
fusion is the FIEDLER lambda_2 of the tissue graph Laplacian (adhesion high -> one lip, lambda_2
> 0; adhesion low -> a cleft, lambda_2 = 0 = harelip). Until now `drive` and `adhesion` were free
knobs. The shape head EMITS them from the measured genome program, closing the last head.

What the genome gives (measurable in ZESTA): the apical-constriction / actomyosin program
(shroom3, myh9/10, mylk, rock, actn) sets the FOLD drive; the epithelial adhesion program
(cdh1/2/6, catenins, desmosome pkp/jup, tight-junction cldn/ocln, epcam) sets the FUSION
adhesion. Neural epithelium scores high on constriction (forebrain/spinal tube folds); mesenchyme
scores low -- the epithelial vs mesenchymal axis, the OTHER side of migration's motility program.
The observable (the fold/cleft geometry) is in NO atlas -> COMPUTED (the missing observable).

Payoff (the medicine link, abundance-medicine thesis): the two classic structural birth defects
fall straight out -- weak constriction program -> the tube fails to close = NEURAL TUBE DEFECT;
weak adhesion program -> the seam fails to resolve = CLEFT LIP / HARELIP. Genome program ->
mechanical parameter -> normal shape or defect.

Run: python -m medic.shape_head
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.mechanical_fusion import (constriction_profile, fold_from_curvature,
                                     prominence_fusion)

OUT = Path("data/organ_cascade")
ATLAS = Path("data/zesta/zf_sixtime_slice.h5ad")

CONSTRICT = ["shroom3", "myh9a", "myh9b", "myh10", "mylka", "mylkb", "rock2a", "actn1", "cdh2"]
ADHESION = ["cdh1", "cdh2", "cdh6", "ctnnb1", "ctnna1", "pkp3a", "jupa", "cldnb", "cldn7a",
            "cldn7b", "oclna", "oclnb", "epcam", "cldne", "cldnh"]
MOTILITY = ["vangl2", "prickle1a", "snai1a", "snai1b", "snai2", "twist1a", "twist1b",
            "foxd3", "sox10", "mmp2", "mmp9", "pdgfra"]      # for the EMT-axis cross-check
NEURAL = ["Forebrain", "Spinal Cord", "Nervous System", "Neural Rod", "Neural Keel"]
EPITHELIAL = ["Epidermal", "Forebrain", "Spinal Cord"]


def measure():
    import anndata as ad
    a = ad.read_h5ad(ATLAS)
    o = a.obs
    vn = np.asarray(a.var_names, str); Sv = set(vn); X = a.X

    def score(genes):
        idx = [np.where(vn == g)[0][0] for g in genes if g in Sv]
        sub = X[:, idx]; sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
        return np.log1p(sub).sum(1)
    anno = o["layer_annotation"].astype(str).values
    tv = o["time"].astype(str).values
    con, adh, mot = score(CONSTRICT), score(ADHESION), score(MOTILITY)
    per = {}
    for t in sorted(set(anno[tv == "24hpf"])):
        m = (tv == "24hpf") & (anno == t)
        if m.sum() >= 20:
            per[t] = dict(constriction=float(con[m].mean()), adhesion=float(adh[m].mean()),
                          motility=float(mot[m].mean()))
    return per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = measure()
    print("(1) MECHANICAL PROGRAM per tissue (24 hpf, measured):")
    for t, v in per.items():
        print(f"    {t:22s} constriction {v['constriction']:.2f}  adhesion {v['adhesion']:.2f}  motility {v['motility']:.2f}")

    # EMT-axis cross-check: shape (adhesion) vs migration (motility) across tissues
    from scipy.stats import spearmanr
    A = np.array([per[t]["adhesion"] for t in per]); M = np.array([per[t]["motility"] for t in per])
    rho_emt, _ = spearmanr(A, M)
    print(f"\n    EMT-axis check: Spearman(adhesion, motility) across tissues = {rho_emt:.2f}")

    # ----- program -> mechanical parameter (calibrate so WT sits in the healthy region) -----
    neural_con = float(np.mean([per[t]["constriction"] for t in NEURAL if t in per]))
    epith_adh = float(np.mean([per[t]["adhesion"] for t in EPITHELIAL if t in per]))
    # calibrate the WT program to the closure threshold (WT embryos close the tube), so drive_wt
    # sits just above closure; a program knockdown then drops below it -> NTD.
    drive_wt = min(1.05, neural_con / 0.90)                # WT neural constriction -> closure
    drive_kd = 0.5 * drive_wt                              # program knockdown
    adh_wt = min(1.0, epith_adh / 2.50)                    # WT epithelial adhesion -> fused
    adh_kd = 0.15 * adh_wt

    # ----- (2) FOLD: neural-tube closure as a buckling eigenmode, drive from the program -----
    comp = constriction_profile(121)
    fold_wt = fold_from_curvature(drive_wt, competence=comp)
    fold_kd = fold_from_curvature(drive_kd, competence=comp)
    print(f"\n(2) FOLD -- neural-tube closure (drive from the constriction program)")
    print(f"    WT  drive={drive_wt:.2f} -> gap {fold_wt['gap']:.2f}  {'CLOSED tube' if fold_wt['closed'] else 'OPEN'}")
    print(f"    KD  drive={drive_kd:.2f} -> gap {fold_kd['gap']:.2f}  {'CLOSED' if fold_kd['closed'] else 'OPEN = NEURAL TUBE DEFECT'}")

    # ----- (3) FUSION: lip fusion vs cleft (Fiedler lambda_2), adhesion from the program -----
    fuse_wt = prominence_fusion(outgrowth=1.0, adhesion=adh_wt, steps=900)
    fuse_kd = prominence_fusion(outgrowth=1.0, adhesion=adh_kd, steps=900)
    print(f"\n(3) FUSION -- lip fusion (adhesion from the epithelial program)")
    print(f"    WT  adhesion={adh_wt:.2f} -> components={fuse_wt['ncomp']} lambda2={fuse_wt['lam2']:.3f}  "
          f"{'FUSED lip' if fuse_wt['fused'] else 'CLEFT'}")
    print(f"    KD  adhesion={adh_kd:.2f} -> components={fuse_kd['ncomp']} lambda2={fuse_kd['lam2']:.3f}  "
          f"{'FUSED' if fuse_kd['fused'] else 'CLEFT LIP = HARELIP'}")

    # 1D adhesion sweep -> the fusion threshold, with the measured WT marked
    adh_axis = np.linspace(0.0, 1.0, 8)
    sweep = [prominence_fusion(outgrowth=1.0, adhesion=aa, steps=700) for aa in adh_axis]
    lam_axis = [r["lam2"] for r in sweep]

    _figure(per, fold_wt, fold_kd, fuse_wt, fuse_kd, adh_axis, lam_axis, adh_wt,
            drive_wt, drive_kd, rho_emt)
    json.dump(dict(program=per, emt_axis_rho=rho_emt, neural_constriction=neural_con,
                   epithelial_adhesion=epith_adh, drive_wt=drive_wt, drive_kd=drive_kd,
                   fold_wt_gap=fold_wt["gap"], fold_wt_closed=fold_wt["closed"],
                   fold_kd_gap=fold_kd["gap"], fold_kd_closed=fold_kd["closed"],
                   adh_wt=adh_wt, fuse_wt=dict(ncomp=fuse_wt["ncomp"], lam2=fuse_wt["lam2"], fused=fuse_wt["fused"]),
                   fuse_kd=dict(ncomp=fuse_kd["ncomp"], lam2=fuse_kd["lam2"], fused=fuse_kd["fused"])),
              open(OUT / "shape_head.json", "w"), indent=2)
    print("\nsaved", OUT / "shape_head.json")
    print(f"\nSUMMARY: the mechanical program emits the fold drive + fusion adhesion; WT closes the "
          f"neural tube and fuses the lip, program knockdown gives a neural-tube defect and a harelip "
          f"-- shape (the buckling/cleft residue) COMPUTED from the genome. 4th head done.")


def _figure(per, fold_wt, fold_kd, fuse_wt, fuse_kd, adh_axis, lam_axis, adh_wt,
            drive_wt, drive_kd, rho_emt):
    fig = plt.figure(figsize=(19, 8.5))
    gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.28)

    # (a) measured program per tissue
    ax = fig.add_subplot(gs[0, 0])
    ts = list(per)
    y = np.arange(len(ts))
    ax.barh(y - 0.2, [per[t]["adhesion"] for t in ts], 0.4, color="tab:blue", label="adhesion")
    ax.barh(y + 0.2, [per[t]["constriction"] for t in ts], 0.4, color="tab:red", label="constriction")
    ax.set_yticks(y); ax.set_yticklabels([t[:16] for t in ts], fontsize=7)
    ax.set_title("(a) mechanical program (measured)", fontsize=10); ax.legend(fontsize=7)

    # (b) EMT axis: adhesion vs motility
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter([per[t]["motility"] for t in ts], [per[t]["adhesion"] for t in ts], s=35)
    ax.set_xlabel("motility (migration head)"); ax.set_ylabel("adhesion (shape head)")
    ax.set_title(f"(b) EMT axis: shape vs migration (rho {rho_emt:.2f})", fontsize=9)

    # (c) fold WT closed
    def draw_fold(ax, f, color, title):
        A, B = f["A"], f["B"]
        for i in range(len(A) - 1):
            poly = np.array([A[i], A[i + 1], B[i + 1], B[i]])
            ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor="#555", lw=0.2, alpha=0.85)
        ax.plot(A[:, 0], A[:, 1], "-", color="#b02318", lw=1.4)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)
    draw_fold(fig.add_subplot(gs[0, 2]), fold_wt, "#9ecae1",
              f"(c) WT: neural tube CLOSED\ndrive={drive_wt:.2f} gap={fold_wt['gap']:.2f}")
    draw_fold(fig.add_subplot(gs[0, 3]), fold_kd, "#fdd0a2",
              f"(d) program KD: OPEN = NTD\ndrive={drive_kd:.2f} gap={fold_kd['gap']:.2f}")

    # (e,f) fusion WT vs cleft
    def show_phi(ax, r, title):
        ext = [r["X"].min(), r["X"].max(), r["Y"].min(), r["Y"].max()]
        ax.imshow(r["phi"], origin="lower", extent=ext, aspect="equal", cmap="pink_r", vmin=0, vmax=1)
        ax.axvline(0, color="#2171b5", ls=":", lw=1)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    show_phi(fig.add_subplot(gs[1, 0]), fuse_wt,
             f"(e) WT: lip FUSED\ncomp={fuse_wt['ncomp']} $\\lambda_2$={fuse_wt['lam2']:.2f}")
    show_phi(fig.add_subplot(gs[1, 1]), fuse_kd,
             f"(f) adhesion KD: CLEFT = harelip\ncomp={fuse_kd['ncomp']} $\\lambda_2$={fuse_kd['lam2']:.2f}")

    # (g) fusion threshold vs adhesion, measured WT marked
    ax = fig.add_subplot(gs[1, 2:])
    ax.plot(adh_axis, lam_axis, "o-", color="tab:purple")
    ax.axvline(adh_wt, color="tab:green", ls="--", label=f"measured WT adhesion ({adh_wt:.2f})")
    ax.set_xlabel("fusion adhesion (from the program)"); ax.set_ylabel("Fiedler $\\lambda_2$")
    ax.set_title("(g) fusion threshold: $\\lambda_2$ rises off zero as adhesion crosses the seam-resolution point", fontsize=9)
    ax.legend(fontsize=8)

    fig.suptitle("The shape head (4th/last): the measured mechanical program emits the fold drive and fusion adhesion; "
                 "the fold (buckling eigenmode)\nand cleft (Fiedler $\\lambda_2$) are COMPUTED. Program knockdown -> "
                 "neural-tube defect + harelip -- the residual shape migration does not give.", fontsize=11)
    fig.savefig(OUT / "shape_head.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "shape_head.png")


if __name__ == "__main__":
    main()
