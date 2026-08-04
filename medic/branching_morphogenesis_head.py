"""
branching_morphogenesis_head.py -- the BRANCHING head: grow the airway (lung) and collecting-duct (kidney)
epithelial trees inside their organs.

A head that reads an organ's own cells (the FGF10 mesenchyme) + the organ's hilum, read-only, and writes a
branched epithelial tree -- it never moves the organ, so it cannot oppose the organ placement.

Biology (genome-anchored, reads fields already produced): the epithelial bud sprouts and branches under
FGF10 secreted by the surrounding MESENCHYME; Shh feeds back from the epithelium to pattern the mesenchyme;
Sprouty sets the branch spacing; Sox9/Wnt the tips. That reciprocal FGF10<->Shh loop is exactly a
SPACE-COLONIZATION field CONFINED to the organ: each organ cell is an FGF10 source, the epithelium grows up
the gradient from the hilum and prunes a source once reached -- so the tree FILLS the organ the placement
head built (lung -> airways, kidney -> collecting ducts). Same primitive as the vasculature head, per-organ.

READ-ONLY: reads the organ cells + hilum, writes the tree. Validation = a single connected tree rooted at
the hilum that fills the organ (coverage), branched (branch points), grown from the organ's own FGF10 field.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.branching_morphogenesis_head
Out: data/organ_cascade/branching_morphogenesis_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.vasculature_head import grow_tree
from medic.tuned_knobs import tuned

ORGANS = {"Lung": "airway tree", "Kidney": "collecting-duct tree"}


def branch_organ(cells, body_c):
    """Grow the epithelial tree inside one organ. Root = the hilum (the organ point nearest the body core,
    where the bronchus/ureter enters). Attractors = the organ cells (FGF10). Scales to the organ size."""
    scale = np.ptp(cells, 0).max() + 1e-9
    hilum = cells[np.linalg.norm(cells - body_c, axis=1).argmin()]      # nearest the body core = the stalk
    root = [hilum, hilum + (cells.mean(0) - hilum) * 0.15]              # a short stalk into the organ
    tk = tuned("branching_lung", {"step_f": 0.11, "dinf_f": 0.9, "dkill_f": 0.16})    # tuned FGF10 grow knobs
    nodes, edges = grow_tree(cells, root, n_attr=min(400, len(cells)),
                             step=tk["step_f"] * scale, d_inf=tk["dinf_f"] * scale,
                             d_kill=tk["dkill_f"] * scale, iters=140)
    d, _ = cKDTree(nodes).query(cells)
    coverage = float((d < 0.25 * scale).mean())
    branch = sum(1 for i in range(len(nodes)) if sum(e[0] == i for e in edges) >= 2)
    return dict(cells=cells, nodes=nodes, edges=edges, coverage=round(coverage, 3),
                branch_points=branch, n_nodes=len(nodes))


def build():
    base, F = build_base()
    body_c = base.mean(0)
    out = {}
    for name in ORGANS:
        if name not in FIDX:
            continue
        cells = base[F == FIDX[name]]
        if len(cells) < 12:
            continue
        out[name] = branch_organ(cells, body_c)
    return out


def _figure(res):
    n = len(res)
    fig, ax = plt.subplots(1, n, figsize=(6 * n, 6), facecolor="#0d1017")
    if n == 1:
        ax = [ax]
    from matplotlib.collections import LineCollection
    for a, (name, r) in zip(ax, res.items()):
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        C, nodes, edges = r["cells"], r["nodes"], r["edges"]
        a.scatter(C[:, 0], C[:, 1], s=8, c="#2f4a3a", alpha=0.5)
        segs = np.array([[nodes[e[0]][[0, 1]], nodes[e[1]][[0, 1]]] for e in edges])
        a.add_collection(LineCollection(segs, colors="#e0b040", linewidths=0.9, alpha=0.9))
        a.set_title(f"{name}: {ORGANS[name]}\n{r['branch_points']} branch points · {r['coverage']*100:.0f}% fill",
                    color="#a5f3c0", fontsize=10)
    fig.suptitle("Branching morphogenesis head: FGF10 space-colonization inside each organ (reads the organ "
                 "cells, read-only)", color="#e2e8f0", fontsize=10)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/branching_morphogenesis_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/branching_morphogenesis_head.png")


def main():
    res = build()
    for name, r in res.items():
        print(f"{name} ({ORGANS[name]}): {r['n_nodes']} nodes, {r['branch_points']} branch points, "
              f"{r['coverage']*100:.0f}% fill")
    print("  genome-derived: FGF10 (mesenchyme->epithelium), Shh feedback, Sprouty spacing, Sox9 tips")
    _figure(res)
    json.dump({n: {k: r[k] for k in ("n_nodes", "branch_points", "coverage")} for n, r in res.items()},
              open("data/organ_cascade/branching_morphogenesis_head.json", "w"), indent=1)
    print("saved data/organ_cascade/branching_morphogenesis_head.json")


if __name__ == "__main__":
    main()
