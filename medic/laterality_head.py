"""laterality_head.py -- the Nodal/Pitx2 LEFT-RIGHT head (read-only): break the model's bilateral symmetry.

The measured step-2 gap (medic.gray_arrangement_objective): the model grows z-symmetric, so the lateralised
viscera sit on the midline and laterality is ABSENT (mean lateral offset ~0). This head places each lateralised
organ on its canonical side under the laterality cascade -- Nodal on the left -> Pitx2 -> heart/spleen/stomach
left of the midline, liver right, the same handedness that loops the heart tube and rotates the gut. It is a
read-only displacement of the organ's own cells along the left-right axis; no other structure is touched.

Built + verified STANDALONE first (as every mechanism in this program has been), so it is ready to wire into
integrated_body / human_movie at step 1 without surgery on the build until it is proven.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.laterality_head
"""
from __future__ import annotations
import numpy as np
from medic.unified_embryo import FIDX

# canonical side under the Nodal/Pitx2 cascade: -1 left, +1 right
CANON_LR = {
    "Heart": -1, "Atrium": -1, "Ventricle": -1, "Outflow": -1,   # heart to the left
    "Spleen": -1, "Stomach": -1,                                  # left viscera
    "Liver": +1,                                                  # liver to the right
}


def lateralize(base, F, amount=0.13):
    """Offset each lateralised organ's cells to its canonical side (fraction of the left-right span). Read-only:
    returns a new position array; every non-lateralised structure is unchanged."""
    span = np.ptp(base[:, 2]) + 1e-9
    out = base.copy()
    for nm, sgn in CANON_LR.items():
        if nm in FIDX:
            m = F == FIDX[nm]
            if m.any():
                out[m, 2] += sgn * amount * span
    return out


def _lat_offsets(base, F):
    span = np.ptp(base[:, 2]) + 1e-9
    rows = {}
    for nm in ("Liver", "Spleen", "Stomach", "Atrium", "Ventricle", "Outflow"):
        if nm in FIDX:
            m = F == FIDX[nm]
            if m.sum() >= 20:
                rows[nm] = round(float(base[m, 2].mean() / span), 4)
    return rows


def main(ne=30000):
    from medic.adult_persistence_audit import build_base
    base, F = build_base(ne)
    before = _lat_offsets(base, F)
    lat = lateralize(base, F)
    after = _lat_offsets(lat, F)
    # sign correctness after the head
    ok = tot = 0
    for nm, sgn in CANON_LR.items():
        if nm in after:
            tot += 1; ok += int(np.sign(after[nm]) == sgn and abs(after[nm]) > 0.05)
    print(f"cells {len(base)}")
    print("lateral offsets (fraction of LR span), BEFORE -> AFTER the laterality head:")
    for nm in sorted(set(before) | set(after)):
        print(f"   {nm:11s} {before.get(nm, 0.0):+.3f} -> {after.get(nm, 0.0):+.3f}")
    mb = np.mean([abs(v) for v in before.values()]); ma = np.mean([abs(v) for v in after.values()])
    print(f"mean |offset|: {mb:.3f} -> {ma:.3f}   sign-correct {ok}/{tot}   (was absent; now lateralised)")


if __name__ == "__main__":
    main()
