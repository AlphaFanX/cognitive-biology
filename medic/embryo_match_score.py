"""A standing MATCH-SCORE benchmark: how close the computed embryo is to the real, per organ / tissue.

The dense E12.5 reconstruction is the fixed real target. For every tissue we store a shape FINGERPRINT
(size + the scale/rotation-invariant descriptors elongation, flatness, axis-bend), and a scoring function
returns a 0..1 match for any computed version of that tissue. This file:

  1. builds the benchmark (real fingerprint per tissue) and saves it, so future work scores against a fixed target;
  2. sets the CURRENT baseline -- each organ modelled as a shapeless blob (position right, shape null), which is
     what the cloud model effectively is -- giving a per-organ shape-match now and ranking the organs by how
     much SHAPE matters (elongated/curved organs score low as blobs = the ones forward models must fix);
  3. where a forward organ model exists (heart, gut, neural tube, optic cup), scores it too, to show the shape
     gain (or, honestly, the current overshoot).

The overall embryo score is the cell-weighted mean per-organ match. Re-run it after any model change to track
progress. Extends to subtissues by passing subtissue point clouds to score_tissue().

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.embryo_match_score
Out:  data/organ_cascade/embryo_match_score.{json,png}
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.organ_3d_vs_real import desc3d, blob
from medic.topology_metric import topo3d
from medic.forward_organs_solid import (heart_solid, gut_solid, brain_solid, spinal_cord_solid,
                                         urogenital_solid, mucosal_solid, liver_solid)

KEYS = ["elongation", "flatness", "bend", "tortuosity", "hollowness"]   # outer moments + topology


def full_desc(P):
    d = desc3d(P); d.update(topo3d(P)); return d


def rel(d1, d2):
    return float(np.mean([abs(d1[k] - d2[k]) / (abs(d1[k]) + abs(d2[k]) + 1e-6) for k in KEYS]))

TARGET = "data/mosta/mouse_e125_3d.npz"
MIN_CELLS = 300
RNG = np.random.default_rng(0)
# tissues we currently have a (solid-filled, magnitude-calibrated) forward organ model for
FORWARD = {"Heart": heart_solid, "GI tract": gut_solid, "Brain": brain_solid, "Spinal cord": spinal_cord_solid,
           "Urogenital ridge": urogenital_solid, "Mucosal epithelium": mucosal_solid, "Liver": liver_solid}


def fingerprint(P):
    P = np.asarray(P, float)
    d = full_desc(P)
    return {"n": int(len(P)), "centroid": [round(float(v), 3) for v in P.mean(0)],
            "size": round(float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())), 3),
            **{k: round(float(d[k]), 3) for k in d}}


def shape_match(desc_computed, fp):
    """0..1 match over outer-moment + topology descriptors: 1 = identical, decaying with relative distance."""
    ref = {k: fp[k] for k in KEYS}
    return float(np.exp(-2.0 * rel(desc_computed, ref)))


def score_tissue(points, fp):
    """Score any computed point cloud for a tissue against its benchmark fingerprint (moments + topology)."""
    return shape_match(full_desc(points), fp)


def main():
    d = np.load(TARGET, allow_pickle=True)
    xyz, tissue = d["xyz"], d["tissue"]
    tissues = [t for t in np.unique(tissue) if t != "?" and (tissue == t).sum() >= MIN_CELLS]

    bench, rows = {}, {}
    for t in tissues:
        R = xyz[tissue == t]
        fp = fingerprint(R); bench[t] = fp
        # current baseline: a shapeless blob (the cloud model) at the organ's size
        s_blob = shape_match(full_desc(blob(len(R), fp["size"] * 2)), fp)
        row = {"n": fp["n"], "elong": fp["elongation"], "bend": fp["bend"],
               "shape_match_blob": round(s_blob, 3)}
        if t in FORWARD:
            row["shape_match_forward"] = round(score_tissue(FORWARD[t](), fp), 3)
        rows[t] = row

    # per-organ (unweighted) mean: every organ counts once, so a big near-round organ (brain) does not mask
    # the gains on the smaller shaped organs
    overall_blob = float(np.mean([rows[t]["shape_match_blob"] for t in tissues]))
    fwd_t = [t for t in tissues if "shape_match_forward" in rows[t]]
    overall_fwd = float(np.mean([rows[t]["shape_match_forward"] for t in fwd_t])) if fwd_t else None

    out = {"stage": "E12.5", "n_tissues": len(tissues), "benchmark_fingerprints": bench,
           "scores": rows, "overall_shape_match_blob": round(overall_blob, 3),
           "overall_shape_match_forward_organs": round(overall_fwd, 3) if overall_fwd else None,
           "note": "shape-only; position/identity are the model's strength and a separate future axis"}
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(out, open("data/organ_cascade/embryo_match_score.json", "w"), indent=1)

    order = sorted(tissues, key=lambda t: rows[t]["shape_match_blob"])
    print(f"E12.5 match-score benchmark ({len(tissues)} tissues). Overall shape-match (cloud/blob baseline) "
          f"= {overall_blob:.2f}")
    print(f"{'tissue':22s} {'n':>6s} {'elong':>6s} {'bend':>6s} {'blob':>6s} {'forward':>8s}  (low blob = shape matters)")
    for t in order:
        r = rows[t]
        fw = f"{r['shape_match_forward']:.2f}" if "shape_match_forward" in r else "  -"
        print(f"  {t:20s} {r['n']:6d} {r['elong']:6.2f} {r['bend']:6.2f} {r['shape_match_blob']:6.2f} {fw:>8s}")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0d1017")
    ax.set_facecolor("#0d1017")
    y = np.arange(len(order))
    ax.barh(y, [rows[t]["shape_match_blob"] for t in order], color="#64748b", label="cloud/blob (current)")
    for i, t in enumerate(order):
        if "shape_match_forward" in rows[t]:
            ax.scatter(rows[t]["shape_match_forward"], i, s=70, color="#7dd3fc", zorder=3,
                       label="forward model" if t == fwd_t[0] else None)
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8, color="#cbd5e1")
    ax.set_xlabel("shape match to real (1 = perfect)", color="#94a3b8")
    ax.set_title(f"E12.5 per-tissue shape-match benchmark  (overall cloud baseline {overall_blob:.2f})\n"
                 "low bars = organs whose SHAPE matters most (elongated/curved) -> where forward models must help",
                 color="#e2e8f0", fontsize=11)
    ax.tick_params(colors="#94a3b8"); ax.legend(fontsize=8, facecolor="#0d1017", labelcolor="#cbd5e1", loc="lower right")
    for s in ax.spines.values(): s.set_color("#334155")
    fig.tight_layout()
    fig.savefig("data/organ_cascade/embryo_match_score.png", dpi=130, facecolor="#0d1017")
    print(f"\nsaved data/organ_cascade/embryo_match_score.{{json,png}}  (benchmark set for future work)")


if __name__ == "__main__":
    main()
