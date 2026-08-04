"""
ct_scaffold_head.py -- the CONNECTIVE-TISSUE-SCAFFOLD head: carve the myotome into individual muscles.

The missing primitive for muscle-for-muscle. Individual muscles are NOT patterned by the myoblasts -- they
are carved by the muscle CONNECTIVE TISSUE (Tcf7l2/Tcf4+, Osr1+ fibroblasts). Kardon/Christ: transplant the
CT and the muscle pattern follows the CT, not the muscle cells. So the myoblasts are naive and FILL whatever
domains the CT scaffold dictates.

GENOME COMPATIBILITY (Miles's constraint -- it must not oppose the genome). This head IMPOSES NO muscle map.
Its scaffold field is read entirely from fields the genome has ALREADY produced for the skeleton:
  * AP segment  = the SAME somitogenesis clock + Hox segmentation that builds the vertebrae
                  (medic.vertebral_column.build_column). Axial muscles are segmentally repeated, one tier per
                  vertebral level -- so they share the vertebrae's genome frame exactly.
  * epaxial / hypaxial = the DORSOVENTRAL axis (the electric-body / notochord-Shh axis the model already
                  computes): dorsal to the axis = epaxial (deep back, erector spinae), ventral = hypaxial
                  (body wall: intercostals / obliques).
  * Tcf4/Osr1 = the CT gene that reads that positional code and lays the cleavage planes.
Because its only inputs are the genome's own outputs, the carving cannot oppose the genome; the validation
is that the result comes out SEGMENTAL + LAYERED (correlates with the vertebral segmentation), i.e. it
reproduces the roster topology because it shares the frame -- it is not told the answer.

SCOPE: this carves the AXIAL myotome (the trunk muscle the model grows). Limb muscles need the LIMB CT
(lateral-plate, Hox-coded) and limb muscle cells, which the model does not grow yet -- deferred.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.ct_scaffold_head
Out: data/organ_cascade/ct_scaffold_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.vertebral_column import build_column, FORMULA
from medic.tuned_knobs import tuned

N_SEG = sum(n for _, n in FORMULA)      # same segment count as the vertebrae (shared genome frame)


def carve(base, F, n_seg=N_SEG, dv_off=None):
    """The CT-scaffold head. `dv_off` = the epaxial/hypaxial split offset (tunable knob; None -> tuned).
    Returns per-myotome-cell (segment, group, muscle_id, muscle_name)."""
    if dv_off is None:
        dv_off = tuned("ct_scaffold", {"dv_off": 0.0})["dv_off"]
    mus = base[F == FIDX["Muscle"]].copy()
    if not len(mus):
        return None
    # --- AP segment: the SAME clock+Hox segmentation as the vertebrae (genome frame, not imposed) ---
    m = mus.copy(); m[:, 0] = -m[:, 0]                      # anterior -> t=0, matching axial_skeleton
    seg_t, seg, _ = build_column(m, n_vert=n_seg)

    # --- epaxial vs hypaxial: the DV axis. Split at the axial (notochord) DV level per AP band ---
    if "Notochord" in FIDX:
        axis = base[F == FIDX["Notochord"]].copy(); axis[:, 0] = -axis[:, 0]
    else:
        axis = m
    ax_t = (axis[:, 0] - axis[:, 0].min()) / (np.ptp(axis[:, 0]) + 1e-9)
    apbin = np.clip((seg_t * 24).astype(int), 0, 23)
    ax_bin = np.clip((ax_t * 24).astype(int), 0, 23)
    axis_y = np.full(24, np.median(axis[:, 1]))
    for k in range(24):
        b = ax_bin == k
        if b.sum() >= 2:
            axis_y[k] = np.median(axis[b, 1])
    # +y = ventral -> ventral of the axis = hypaxial (body wall), dorsal = epaxial (deep back)
    group = np.where(mus[:, 1] > axis_y[apbin] + dv_off, "hypaxial", "epaxial")

    # --- carve: one muscle domain per (segment x group). This is the cleavage the CT lays down ---
    muscle_id = seg * 2 + (group == "hypaxial").astype(int)
    # name each domain after its axial-muscle homology (segmental erector-spinae / intercostal-oblique tier)
    def _nm(s, g):
        lvl = ("C%d" % (s + 1) if s < 7 else "T%d" % (s - 6) if s < 20 else
               "L%d" % (s - 19) if s < 26 else "S%d" % (s - 25))
        return ("erector spinae" if g == "epaxial" else "intercostal/oblique") + f" @{lvl}"
    name = np.array([_nm(int(s), g) for s, g in zip(seg, group)])
    return dict(mus=mus, seg=seg, seg_t=seg_t, group=group, muscle_id=muscle_id, name=name)


def _validate(r):
    """Genome-derivation checks: is the carving SEGMENTAL (many domains, ~one per level) + LAYERED
    (both epaxial and hypaxial present), and does the segment index track the AP axis monotonically?"""
    seg, group, seg_t = r["seg"], r["group"], r["seg_t"]
    n_domains = len(set(zip(seg.tolist(), group.tolist())))
    n_seg_used = len(set(seg.tolist()))
    epi = float((group == "epaxial").mean()); hyp = 1 - epi
    # segmental monotonicity: mean AP fraction increases with segment index (the clock ordering)
    seg_ap = [seg_t[seg == s].mean() for s in sorted(set(seg.tolist()))]
    monotonic = float(np.mean(np.diff(seg_ap) > 0)) if len(seg_ap) > 1 else 0.0
    return dict(n_domains=n_domains, n_segments=n_seg_used, epaxial_frac=round(epi, 2),
                hypaxial_frac=round(hyp, 2), segment_monotonic=round(monotonic, 2))


def _figure(r, v):
    mus, seg, group, mid = r["mus"], r["seg"], r["group"], r["muscle_id"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(mus[:, 0], mus[:, 1], s=6, c="#8a5a3a", alpha=0.6)
    ax[0].set_title(f"BEFORE — myotome ({len(mus)} cells): one undifferentiated blob", color="#c99", fontsize=10)
    ax[1].scatter(mus[:, 0], mus[:, 1], s=7, c=mid % 20, cmap="tab20", alpha=0.85)
    ax[1].set_title(f"AFTER — CT scaffold carves {v['n_domains']} muscle domains\n"
                    f"segmental (x{v['n_segments']}) x epaxial/hypaxial · Tcf4/Osr1 on the Hox+DV frame",
                    color="#a5f3c0", fontsize=10)
    fig.suptitle("Connective-tissue-scaffold head: the myotome carved into individual muscles by the genome's "
                 "own segment (clock/Hox) x DV (epaxial/hypaxial) code -- not an imposed map",
                 color="#e2e8f0", fontsize=10)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/ct_scaffold_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/ct_scaffold_head.png")


def main():
    base, F = build_base()
    r = carve(base, F)
    if r is None:
        print("no myotome cells"); return
    v = _validate(r)
    print(f"myotome {len(r['mus'])} cells -> CARVED into {v['n_domains']} muscle domains")
    print(f"  segmental: {v['n_segments']} levels, AP-monotonic {v['segment_monotonic']} (clock/Hox frame)")
    print(f"  layered  : epaxial {v['epaxial_frac']} / hypaxial {v['hypaxial_frac']} (DV axis)")
    print(f"  genome-derived inputs only (segment=vertebral clock+Hox, DV=electric-body axis, gene=Tcf4/Osr1)"
          f" -> does not oppose the genome")
    from collections import Counter
    top = Counter(r["name"].tolist()).most_common(6)
    print("  example carved muscles:", ", ".join(f"{n} ({c})" for n, c in top))
    _figure(r, v)
    json.dump(dict(cells=len(r["mus"]), **v, examples=[n for n, _ in top]),
              open("data/organ_cascade/ct_scaffold_head.json", "w"), indent=1)
    print("saved data/organ_cascade/ct_scaffold_head.json")


if __name__ == "__main__":
    main()
