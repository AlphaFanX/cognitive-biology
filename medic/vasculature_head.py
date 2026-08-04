"""
vasculature_head.py -- the VASCULATURE head: grow a branching blood-vessel tree that perfuses the body.

A head that reads the BODY (the tissue that needs perfusion) + the midline frame, read-only, and writes a
vascular tree -- it never moves a tissue cell, so it cannot oppose the body plan.

Biology (all genome-anchored, all fields already produced):
  * ROOT = the DORSAL AORTA on the body MIDLINE, just ventral to the notochord -- the same midline the
    electric-body frame + notochord already define (Hedgehog/Shh + VEGF specify it).
  * DRIVE = VEGF-A. Tissue that is far from a vessel is HYPOXIC (HIF1a) and secretes VEGF, which pulls a
    sprout toward it; once perfused, the VEGF signal stops. That is exactly a SPACE-COLONIZATION field:
    each tissue cell is a VEGF attractor, the tree grows up the VEGF gradient and prunes an attractor when
    it is reached. Tip/stalk selection = Notch/Dll4; arterial identity = Notch/EphrinB2.
So the head reads {tissue positions = the VEGF sources, midline = the aorta} and writes the tree. The tree
FILLS whatever body the upstream heads built -- widen the body or add a limb and the vessels follow, because
the attractors are the tissue itself.

READ-ONLY: reads the body, writes vessels. Validation = the tree is a single connected network rooted on
the aorta that PERFUSES most of the tissue (coverage), grown from the tissue's own VEGF field, not an
imposed vessel map.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.vasculature_head
Out: data/organ_cascade/vasculature_head.{png,json}
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
from medic.tuned_knobs import tuned


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-9)


def grow_tree(body, root, n_attr=1100, step=0.028, d_inf=0.22, d_kill=0.05, iters=260, seed=0):
    """VEGF space colonization: attractors = a sample of tissue cells; grow the tree from the aorta root up
    the VEGF gradient, pruning each attractor once a vessel reaches it. Returns (nodes, edges)."""
    rng = np.random.default_rng(seed)
    A = body[rng.choice(len(body), min(n_attr, len(body)), replace=False)].astype(float)
    nodes = [p.astype(float) for p in root]
    edges = list(zip(range(len(root) - 1), range(1, len(root))))     # the aorta is a chain
    for _ in range(iters):
        if not len(A):
            break
        N = np.array(nodes)
        d, idx = cKDTree(N).query(A)
        act = d < d_inf
        if not act.any():
            break
        grow = {}
        for ai in np.where(act)[0]:
            ni = int(idx[ai])
            grow.setdefault(ni, np.zeros(3))
            grow[ni] += _unit(A[ai] - N[ni])
        for ni, g in grow.items():
            newp = N[ni] + step * _unit(g)
            nodes.append(newp); edges.append((ni, len(nodes) - 1))
        d2, _ = cKDTree(np.array(nodes)).query(A)
        A = A[d2 > d_kill]                                            # prune perfused (reached) attractors
    return np.array(nodes), edges


def build():
    base, F = build_base()
    body = base
    # DORSAL AORTA root: a chain of nodes along the AP midline, just ventral to the notochord
    if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8:
        noto = base[F == FIDX["Notochord"]]
        ay = np.median(noto[:, 1]) + 0.02 * np.ptp(base[:, 1])       # a touch ventral of the notochord
    else:
        ay = np.median(base[:, 1])
    xs = np.linspace(np.percentile(base[:, 0], 4), np.percentile(base[:, 0], 96), 16)
    root = [np.array([x, ay, 0.0]) for x in xs]                      # z=0 midline
    tk = tuned("vasculature", {"step": 0.028, "d_inf": 0.22, "d_kill": 0.05})     # tuned sprouting knobs
    nodes, edges = grow_tree(body, root, step=tk["step"], d_inf=tk["d_inf"], d_kill=tk["d_kill"])
    # perfusion coverage: fraction of tissue within d_inf of a vessel node
    d, _ = cKDTree(nodes).query(body)
    coverage = float((d < 0.22).mean())
    tips = sum(1 for i in range(len(nodes)) if all(e[0] != i for e in edges))   # nodes that are no one's parent
    branch = sum(1 for i in range(len(nodes)) if sum(e[0] == i for e in edges) >= 2)
    return dict(body=body, nodes=nodes, edges=edges, root_n=len(root),
                coverage=round(coverage, 3), n_nodes=len(nodes), tips=tips, branch_points=branch)


def _figure(r):
    body, nodes, edges = r["body"], r["nodes"], r["edges"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    for a, (i, j, ttl) in zip(ax, [(0, 1, "front (AP x DV)"), (0, 2, "top (AP x ML)")]):
        a.scatter(body[:, i], body[:, j], s=3, c="#2a3140", alpha=0.5)
        segs = np.array([[nodes[e[0]][[i, j]], nodes[e[1]][[i, j]]] for e in edges])
        from matplotlib.collections import LineCollection
        a.add_collection(LineCollection(segs, colors="#e0403a", linewidths=0.6, alpha=0.85))
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    fig.suptitle(f"Vasculature head: dorsal-aorta-rooted tree grown by VEGF space-colonization on the tissue "
                 f"-- {r['coverage']*100:.0f}% perfusion coverage, {r['branch_points']} branch points, reads the "
                 f"body read-only", color="#e2e8f0", fontsize=10)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/vasculature_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/vasculature_head.png")


def main():
    r = build()
    print(f"vascular tree: {r['n_nodes']} nodes (root aorta {r['root_n']}), {r['branch_points']} branch points, "
          f"{r['tips']} tips")
    print(f"perfusion coverage: {r['coverage']*100:.0f}% of tissue within reach of a vessel")
    print("  genome-derived: aorta=midline (Shh+VEGF), sprouting=VEGF-A/HIF1a space-colonization, tip/stalk=Notch/Dll4")
    print("  reads the tissue (VEGF sources) + midline, read-only -> cannot oppose the body plan")
    _figure(r)
    json.dump(dict(n_nodes=r["n_nodes"], root_aorta=r["root_n"], branch_points=r["branch_points"],
                   tips=r["tips"], coverage=r["coverage"]),
              open("data/organ_cascade/vasculature_head.json", "w"), indent=1)
    print("saved data/organ_cascade/vasculature_head.json")


if __name__ == "__main__":
    main()
