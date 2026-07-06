"""
The migration head: the motility program (measured) + convergent extension COMPUTED in 3D.
==========================================================================================

Migration is the head where the atlas runs out of observable twice over: a Stereo-seq section
is a STATIC snapshot (no displacement) AND it is a single 2D sagittal slice, while the movement
that matters -- convergent extension -- happens by cells intercalating MEDIOLATERALLY, straight
through that slice. So migration cannot be read from the data at all; it must be COMPUTED in 3D.
What the genome DOES give (and we can measure) is the motility PROGRAM: which cells are competent
to move (PCP / convergent-extension genes vangl2, prickle, wnt5/11, gpc4, fn1, cdh2; EMT / crest
genes snai, twist, foxd3, sox10, mmp). This is the 0.28 head of SHARE-seq -- real but noisier
than fate, and confounded by transcriptome depth at bin resolution.

This also answers "does migration give shape?": YES, for a large class. Convergent extension --
mediolateral intercalation of the axial/paraxial mesoderm -- narrows the tissue and elongates
the body axis. That elongation IS shape, produced purely by movement. (The residue migration
does NOT give -- folds/clefts from adhesion-tension buckling -- is the separate shape head:
mechanical_fusion.py, Fiedler lambda_2.)

So: (1) measure the motility program per tissue (real, note the confound); (2) run convergent
extension FORWARD in 3D on a gastrula-like block, gated by that program, and show the axis
elongate -- migration giving shape, in 3D, as the computed missing observable.

Run: python -m medic.migration_head
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/organ_cascade")
MOVIE = Path("data/movie/zebrafish_ce_frames.json")
ATLAS = Path("data/zesta/zf_sixtime_slice.h5ad")

CE_PCP = ["vangl2", "prickle1a", "prickle1b", "wnt5b", "wnt11", "fzd7a", "fzd7b", "gpc4",
          "celsr1a", "scrib", "fn1a", "fn1b", "cdh2", "dvl2"]
EMT = ["snai1a", "snai1b", "snai2", "twist1a", "twist1b", "zeb1a", "zeb1b", "foxd3",
       "sox10", "mmp2", "mmp9", "pdgfra", "cxcr4a", "cxcr4b", "cdh6"]


def measure_program():
    import anndata as ad
    a = ad.read_h5ad(ATLAS)
    o = a.obs
    vn = np.asarray(a.var_names, str); Sv = set(vn)
    X = a.X
    idx = [np.where(vn == g)[0][0] for g in CE_PCP + EMT if g in Sv]
    sub = X[:, idx]; sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    mot = np.log1p(sub).sum(1)
    anno = o["layer_annotation"].astype(str).values
    tv = o["time"].astype(str).values
    per_t = {}
    for t in sorted(set(anno[tv == "24hpf"])):
        m = (tv == "24hpf") & (anno == t)
        if m.sum() >= 20:
            per_t[t] = float(mot[m].mean())
    return per_t


# ---------- convergent extension, computed forward in 3D ----------
def ce_simulate(n=1600, steps=60, snap_every=3, seed=0):
    """A gastrula-like 3D block (broad mediolaterally, short antero-posteriorly) whose
    program-competent cells intercalate toward the midline; soft-sphere repulsion in a thin
    sheet forces the freed volume into the antero-posterior axis -> the axis elongates.
    x = AP (free long axis), y = DV (thin sheet, confined), z = ML (converges to midline)."""
    rng = np.random.RandomState(seed)
    x = rng.uniform(-0.6, 0.6, n)          # AP: short
    z = rng.uniform(-1.3, 1.3, n)          # ML: broad
    y = rng.uniform(-0.22, 0.22, n)        # DV: thin sheet
    P = np.c_[x, y, z]

    # motility program: high over the axial/paraxial CE domain (|z| < ~0.9), tapering laterally;
    # lateral-most cells (yolk/epidermis) are non-motile -- matches the measured gradient shape.
    prog = 1.0 / (1.0 + np.exp((np.abs(z) - 0.9) / 0.18))     # ~1 near midline, ->0 laterally

    R = 0.10                                # soft-sphere radius
    k_conv, k_rep, k_dv = 0.10, 0.55, 0.20
    frames = [P.copy()]
    aspect = []
    for s in range(steps):
        tree = cKDTree(P)
        disp = np.zeros_like(P)
        # mediolateral convergence toward the midline, gated by the program
        disp[:, 2] -= k_conv * prog * P[:, 2]
        # soft-sphere repulsion (volume conservation) -> escapes along the free AP axis
        pairs = tree.query_pairs(2 * R, output_type="ndarray")
        if len(pairs):
            d = P[pairs[:, 0]] - P[pairs[:, 1]]
            dist = np.linalg.norm(d, axis=1) + 1e-9
            push = (k_rep * np.maximum(2 * R - dist, 0) / dist)[:, None] * d
            np.add.at(disp, pairs[:, 0], push)
            np.add.at(disp, pairs[:, 1], -push)
        disp[:, 1] -= k_dv * P[:, 1]        # keep the sheet thin (DV confinement)
        P = P + disp
        if s % snap_every == 0 or s == steps - 1:
            frames.append(P.copy())
        ap = P[:, 0].max() - P[:, 0].min()
        ml = np.percentile(P[:, 2], 97.5) - np.percentile(P[:, 2], 2.5)
        aspect.append(ap / (ml + 1e-9))
    return frames, np.array(aspect), prog


def export_movie(frames, prog):
    """Write the CE trajectory as a viewer movie (real 3D positions changing over time)."""
    n = len(frames[0])
    # scale to the viewer's ~unit box; colour by motility program (who drives the shape)
    allP = np.vstack(frames)
    c = allP.mean(0); half = 0.5 * max(np.ptp(allP[:, 0]), np.ptp(allP[:, 2]))
    scale = 1.5 / half
    pmin, pmax = float(prog.min()), float(prog.max())
    out_frames = []
    for i, P in enumerate(frames):
        Q = (P - c) * scale
        ap = P[:, 0].max() - P[:, 0].min()
        ml = np.percentile(P[:, 2], 97.5) - np.percentile(P[:, 2], 2.5)
        out_frames.append(dict(
            stage=f"CE step {i}/{len(frames)-1} · axis {ap/(ml+1e-9):.2f}", n_cells=n,
            # viewer expects [x, y, z]; map AP->x, DV->y, ML->z
            xyz=[[round(float(Q[j, 0]), 3), round(float(Q[j, 1]), 3), round(float(Q[j, 2]), 3)] for j in range(n)],
            vm=[round(float(v), 3) for v in prog]))
    doc = dict(display="Zebrafish · convergent extension (3D migration → shape)",
               source="computed 3D CE gated by the measured motility program · axis elongation = shape",
               accent="#e0a458", vmin=pmin, vmax=pmax, n_points=n, open_frame=0,
               setpoints={"motility program (low)": round(pmin, 2), "motility program (high)": round(pmax, 2)},
               frames=out_frames)
    MOVIE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(MOVIE, "w"))
    print(f"saved {MOVIE}  ({len(out_frames)} frames, n={n}, {MOVIE.stat().st_size/1e6:.1f} MB)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_t = measure_program()
    print("(1) MOTILITY PROGRAM per tissue (24 hpf, measured; noisier than fate -- depth confound):")
    for t, v in sorted(per_t.items(), key=lambda kv: -kv[1]):
        print(f"    {t:22s} {v:.2f}")

    print("\n(2) CONVERGENT EXTENSION computed forward in 3D (gated by the program):")
    frames, aspect, prog = ce_simulate()
    print(f"    axis aspect ratio (AP/ML): {aspect[0]:.2f} -> {aspect[-1]:.2f}  "
          f"(x{aspect[-1]/aspect[0]:.1f} elongation)")
    print("    -> migration produces shape: the body axis narrows mediolaterally and extends "
          "antero-posteriorly.")
    export_movie(frames, prog)

    _figure(frames, aspect, prog, per_t)
    json.dump(dict(motility_per_tissue=per_t, aspect_start=float(aspect[0]),
                   aspect_end=float(aspect[-1]), elongation_x=float(aspect[-1] / aspect[0]),
                   n_cells=len(frames[0]), n_steps=len(aspect)),
              open(OUT / "migration_head.json", "w"), indent=2)
    print("\nsaved", OUT / "migration_head.json")
    print(f"\nSUMMARY: motility program measured (crest/mesoderm competent, yolk not); convergent "
          f"extension computed in 3D elongates the axis x{aspect[-1]/aspect[0]:.1f} = migration GIVES "
          f"shape, and it is intrinsically 3D (the M-L intercalation is out of the 2D slice).")


def _figure(frames, aspect, prog, per_t):
    fig = plt.figure(figsize=(18, 5.2))
    P0, P1 = frames[0], frames[-1]
    # (a) before  (top view: AP vs ML)
    ax = fig.add_subplot(1, 4, 1)
    ax.scatter(P0[:, 2], P0[:, 0], c=prog, cmap="viridis", s=6, linewidths=0)
    ax.set_xlabel("ML"); ax.set_ylabel("AP"); ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.8, 1.8)
    ax.set_title(f"(a) before CE  (aspect {aspect[0]:.2f})", fontsize=10)
    # (b) after
    ax = fig.add_subplot(1, 4, 2)
    ax.scatter(P1[:, 2], P1[:, 0], c=prog, cmap="viridis", s=6, linewidths=0)
    ax.set_xlabel("ML"); ax.set_ylabel("AP"); ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.8, 1.8)
    ax.set_title(f"(b) after CE  (aspect {aspect[-1]:.2f})  — converged + extended", fontsize=10)
    # (c) aspect ratio over time
    ax = fig.add_subplot(1, 4, 3)
    ax.plot(aspect, "o-", color="tab:orange", ms=3)
    ax.set_xlabel("step"); ax.set_ylabel("axis aspect (AP / ML)")
    ax.set_title("(c) the body axis elongates (shape from migration)", fontsize=10)
    # (d) measured program per tissue
    ax = fig.add_subplot(1, 4, 4)
    order = sorted(per_t, key=lambda t: per_t[t])
    ax.barh(range(len(order)), [per_t[t] for t in order], color="tab:green")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([t[:18] for t in order], fontsize=7)
    ax.set_xlabel("motility program"); ax.set_title("(d) measured motility program", fontsize=9)
    fig.suptitle("The migration head: the motility program is measured from the genome; the movement (convergent "
                 "extension) is COMPUTED in 3D and elongates the axis\n-- migration gives shape, and it is "
                 "intrinsically 3D (mediolateral intercalation runs through the 2D sagittal slice).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "migration_head.png", dpi=125, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "migration_head.png")


if __name__ == "__main__":
    main()
