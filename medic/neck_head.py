"""neck_head.py -- give the figure a NECK. (2026-08-09, Miles: "our figure does not really have a neck")

The head sits straight on the shoulders with no cervical narrowing. BodyParts3D (medic.bp3d_insitu_head)
measures the real neck: it spans ~11% of stature between the shoulders and the skull base and narrows to
~29% of the shoulder width. This head carves that pinch -- it compresses the medio-lateral and dorso-ventral
width of the cells in the cervical band toward the neck axis, ramped by a bell so it blends into the head above
and the shoulders below, leaving the cervical vertebrae and cord on the axis. Read-only outside the band; the
lateral limbs are excluded. Wired in build_base.

Run (self-test): cd cognimed && venv_win_new/Scripts/python.exe -m medic.neck_head
"""
from __future__ import annotations
import json
import numpy as np

from medic.unified_embryo import FIDX

HEAD_FATES = ("Forebrain", "Midbrain", "Hindbrain", "Cerebellum", "Eye", "Retina", "Otic", "OlfactoryBulb")


def _target():
    try:
        n = json.load(open("data/organ_cascade/bp3d_insitu_head.json")).get("neck", {})
        return float(n.get("length_frac", 0.114)), float(n.get("narrow_vs_shoulder", 0.30) or 0.30)
    except Exception:
        return 0.114, 0.30


def build_neck(base, F, length_frac=None, narrow=None):
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    L0, nrw = _target()
    length_frac = L0 if length_frac is None else length_frac
    narrow = nrw if narrow is None else narrow
    x = base[:, 0]; stature = np.ptp(x) + 1e-9
    head_ids = [FIDX[n] for n in HEAD_FATES if n in FIDX]
    hm = np.isin(F, head_ids)
    if hm.sum() < 20:
        return base
    limb = np.isin(F, [FIDX[n] for n in ("Limb Bud",) if n in FIDX]) if "Limb Bud" in FIDX else np.zeros(len(F), bool)
    skull_base = float(np.percentile(x[hm], 8))                       # bottom of the head
    nb_hi = skull_base
    nb_lo = skull_base - length_frac * stature                       # neck spans one neck-length below the skull
    band = (x >= nb_lo) & (x < nb_hi) & (~hm) & (~limb)
    if band.sum() < 20:
        return base
    # shoulder reference: the ML half-width just below the neck band (the biacromial breadth)
    sh = (x >= nb_lo - 0.06 * stature) & (x < nb_lo) & (~limb)
    sh_hw = float(np.percentile(np.abs(base[sh, 2]), 90)) if sh.sum() > 20 else float(np.percentile(np.abs(base[band, 2]), 90))
    axis_z = 0.0                                                     # neck axis on the midline
    axis_y = float(np.median(base[band, 1]))                        # neck DV axis
    u = (x[band] - nb_lo) / (nb_hi - nb_lo + 1e-9)                  # 0 at shoulders .. 1 at skull base
    bell = np.exp(-((u - 0.5) / 0.42) ** 2)                        # deepest pinch mid-neck, blends at both ends
    cur_hw = float(np.percentile(np.abs(base[band, 2] - axis_z), 90)) + 1e-9
    tgt_hw = sh_hw * (narrow + (1.0 - narrow) * (1.0 - bell))       # -> narrow*shoulder at mid-neck
    sz = np.clip(tgt_hw / cur_hw, 0.2, 1.0)
    idx = np.where(band)[0]
    base[idx, 2] = axis_z + (base[idx, 2] - axis_z) * sz            # pinch ML
    base[idx, 1] = axis_y + (base[idx, 1] - axis_y) * (0.5 + 0.5 * sz)  # gentler DV pinch (keep some depth)
    return base


def main():
    from medic.adult_persistence_audit import build_base
    from medic import gray_integrity_audit as GIA
    base, F = build_base(30000)
    x = base[:, 0]; stat = np.ptp(x)
    hm = np.isin(F, [FIDX[n] for n in HEAD_FATES if n in FIDX])
    sb = np.percentile(x[hm], 8); nb_lo = sb - 0.114 * stat
    def hw(cloud, lo, hi):
        m = (cloud[:, 0] >= lo) & (cloud[:, 0] < hi)
        return float(np.percentile(np.abs(cloud[m, 2]), 90)) if m.sum() > 10 else float("nan")
    L0, nrw = _target(); print(f"[target] neck length {L0:.3f} stature, narrow {nrw:.2f} of shoulder")
    out = build_neck(base, F)
    sh = hw(base, nb_lo - 0.06 * stat, nb_lo)
    print(f"[shoulder ML half-width] {sh:.3f}")
    print(f"[neck ML half-width]  before {hw(base, nb_lo+0.03*stat, sb-0.01*stat):.3f} -> after {hw(out, nb_lo+0.03*stat, sb-0.01*stat):.3f}  (target {nrw*sh:.3f})")
    for tag, c in (("before", base), ("after", out)):
        rows = GIA.audit(c, F); npass = sum(1 for r in rows if r.get("ok"))
        fails = [r["relation"] for r in rows if not r.get("ok")]
        print(f"[integrity {tag}] {npass}/{len(rows)}" + (f"  FAIL {fails}" if fails else ""))
    # render upper body before/after
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lo = x.min() + 0.70 * stat
    fig, ax = plt.subplots(1, 2, figsize=(7, 6), facecolor="#0d1017")
    for a, (t, c) in zip(ax, (("before", base), ("after", out))):
        U = c[c[:, 0] >= lo]
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(U[:, 2], U[:, 0], s=3, c=("#f0a" if t == "before" else "#9fe6b0"), alpha=.5)
        a.set_title(f"{t} (front)", color="#cbd5e1")
    fig.suptitle("neck_head: cervical pinch (upper body, front)", color="#e2e8f0")
    fig.savefig("data/organ_cascade/_neck.png", dpi=115, facecolor="#0d1017"); print("saved _neck.png")


if __name__ == "__main__":
    main()
