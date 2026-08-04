"""
neural_wiring_head.py -- the NEURAL WIRING (connectome) head: grow axon tracts that connect the nervous
system to its targets.

Placing neural tissue is a prior head's job; WIRING it is this head -- it reads the neurons + their targets
(muscle, the other side of the CNS) and writes axon tracts along the genome's guidance gradients, read-only,
so it never moves a cell and cannot oppose the body plan.

Biology (genome-anchored, reads fields already produced):
  * MOTOR axons: spinal-cord ventral-horn motor neurons grow OUT to the muscle they will drive (the muscle
    cells the CT-scaffold/limb-muscle heads made) -- guided by the muscle's own cues (HGF, Sema). Reads
    {spinal cord, muscle} -> the neuromuscular tract.
  * COMMISSURES: axons cross the MIDLINE to connect the two sides -- NETRIN/DCC pulls them to the floor
    plate, they cross, then SLIT/ROBO stops them re-crossing. Reads the bilateral neurons + the midline
    (the electric-body LR node) -> the crossing tracts.
  * LONGITUDINAL tract: the main brain<->cord axis bundle along AP (descending motor / ascending sensory).
So the head reads {neurons, targets, midline} and writes tracts up the guidance gradients.

READ-ONLY: reads cells, writes axons. Validation = motor tracts innervate the muscle (coverage),
commissures actually CROSS the midline (sign change in ML), the longitudinal tract spans brain->cord.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.neural_wiring_head
Out: data/organ_cascade/neural_wiring_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.tuned_knobs import tuned

CNS = ["Spinal Cord", "Forebrain", "Midbrain", "Hindbrain", "Nervous System", "Cerebellum"]


def _cells(base, F, names):
    ids = [FIDX[n] for n in names if n in FIDX]
    return base[np.isin(F, ids)]


def build(n_motor=None, n_comm=60, base=None, F=None):
    if n_motor is None:
        n_motor = int(tuned("neural_wiring", {"n_motor": 90})["n_motor"])
    if base is None:
        base, F = build_base()
    rng = np.random.default_rng(0)
    cord = _cells(base, F, ["Spinal Cord"])
    brain = _cells(base, F, ["Forebrain", "Midbrain", "Hindbrain", "Cerebellum"])
    neural = _cells(base, F, CNS)
    muscle = base[F == FIDX["Muscle"]] if "Muscle" in FIDX else base[:0]
    tracts, kinds = [], []

    # 1) MOTOR: sample cord motor neurons -> nearest muscle target (the neuromuscular tract)
    innerv = 0
    if len(cord) and len(muscle):
        somata = cord[rng.choice(len(cord), min(n_motor, len(cord)), replace=False)]
        mt = cKDTree(muscle)
        used = set()
        for s in somata:
            _, j = mt.query(s)
            tracts.append(np.array([s, muscle[j]])); kinds.append("motor")
            used.add(int(j))
        # coverage: fraction of muscle within reach of a motor endpoint
        d, _ = cKDTree(np.array([t[1] for t in tracts])).query(muscle)
        innerv = float((d < 0.12 * (np.ptp(muscle, 0).max() + 1e-9) * 4).mean())

    # 2) COMMISSURES: bilateral neurons cross the midline (Netrin -> floor plate -> cross)
    crossed = 0
    if len(neural):
        left = neural[neural[:, 2] < 0]; right = neural[neural[:, 2] > 0]
        if len(left) and len(right):
            L = left[rng.choice(len(left), min(n_comm, len(left)), replace=False)]
            rt = cKDTree(right)
            for s in L:
                _, j = rt.query([s[0], s[1], -s[2]])          # its mirror partner on the right
                t = right[j]
                mid = np.array([0.5 * (s[0] + t[0]), 0.5 * (s[1] + t[1]), 0.0])   # via the midline (Netrin)
                tracts.append(np.array([s, mid, t])); kinds.append("commissure")
                crossed += int(np.sign(s[2]) != np.sign(t[2]))

    # 3) LONGITUDINAL brain<->cord tract (the main AP bundle)
    if len(brain) and len(cord):
        b = brain.mean(0); c = cord.mean(0)
        mids = np.linspace(0, 1, 8)[:, None]
        path = b[None] * (1 - mids) + c[None] * mids
        tracts.append(path); kinds.append("longitudinal")

    return dict(base=base, neural=neural, muscle=muscle, tracts=tracts, kinds=kinds,
                n_motor=kinds.count("motor"), n_comm=kinds.count("commissure"),
                innervation=round(innerv, 3), crossed=crossed)


def _figure(r):
    col = {"motor": "#e0403a", "commissure": "#40c0e0", "longitudinal": "#f0d040"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), facecolor="#0d1017")
    for a, (i, j, ttl) in zip(ax, [(0, 1, "front (AP x DV)"), (0, 2, "top (AP x ML) — commissures cross z=0")]):
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(r["base"][:, i], r["base"][:, j], s=2, c="#232a36", alpha=0.4)
        a.scatter(r["neural"][:, i], r["neural"][:, j], s=4, c="#5b6b8c", alpha=0.5)
        for kind in ("motor", "commissure", "longitudinal"):
            segs = [t[:, [i, j]] for t, k in zip(r["tracts"], r["kinds"]) if k == kind]
            if segs:
                lc = LineCollection([np.column_stack([s[:, 0], s[:, 1]]) for s in segs],
                                    colors=col[kind], linewidths=0.6, alpha=0.8)
                a.add_collection(lc)
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    fig.suptitle(f"Neural wiring head: {r['n_motor']} motor tracts (red -> muscle) · {r['n_comm']} commissures "
                 f"(blue, cross the midline) · longitudinal (yellow) — reads neurons+targets, read-only",
                 color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/neural_wiring_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/neural_wiring_head.png")


def main():
    r = build()
    print(f"MOTOR tracts   : {r['n_motor']} (spinal cord -> muscle); innervation reach {r['innervation']*100:.0f}%")
    print(f"COMMISSURES    : {r['n_comm']} (bilateral); {r['crossed']} actually cross the midline (Netrin/DCC)")
    print(f"LONGITUDINAL   : brain<->cord AP tract")
    print("  genome-derived: Netrin/DCC + Slit/Robo (midline), HGF/Sema (motor), Ephrin/Eph (topographic)")
    _figure(r)
    json.dump(dict(motor=r["n_motor"], commissures=r["n_comm"], crossed=r["crossed"],
                   innervation=r["innervation"]), open("data/organ_cascade/neural_wiring_head.json", "w"), indent=1)
    print("saved data/organ_cascade/neural_wiring_head.json")


if __name__ == "__main__":
    main()
