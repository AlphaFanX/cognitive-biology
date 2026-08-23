"""heart_tube_head.py -- grow the heart as a LOOPED TUBE with sequenced chambers (read-only, post-condensation).

The measured fault (medic.gray_arrangement_objective): the model grows the heart as a compact BLOB and the
coordinate cut labels its chambers as spatially INTERMIXED cells (Atrium/Ventricle/Outflow all at loop-param
t=0.5), so there is no chamber SEQUENCE to order. This head replaces the blob with the heart's real built form:
the cardiac tube looped into the D-loop, its cells distributed ALONG a looping centre-line and re-assigned to
chambers by arc length -- inflow/atrium (venous pole) -> ventricle (the ventral bulge) -> outflow (arterial
pole) -- so the chain relational attractor has a real looped tube to read.

Genome anchor: chamber identity along the tube = Tbx5 (atrium) / Hand1-2 (ventricle) / Isl1 (outflow); the
D-loop handedness = Nodal -> Pitx2 (here the tube loops beside the left-placed heart). Read-only on every
non-heart cell; the loop is centred on the heart's own centroid so its atlas address is preserved.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.heart_tube_head
"""
from __future__ import annotations
import numpy as np
from medic.unified_embryo import FIDX, FATES
from medic.heart_loop import loop_centerline, chamber_of

HEART_FATES = ("Heart", "Atrium", "Ventricle", "Outflow")
# chamber_of returns inflow/Atrium/Ventricle/Outflow; the model has no 'inflow' fate -> fold it into Atrium.
CHAMBER_TO_FATE = {"inflow": "Atrium", "Atrium": "Atrium", "Ventricle": "Ventricle", "Outflow": "Outflow"}


def make_heart_tube(hpos, pitx2=-1.0, tube_r=0.09, seed=0):
    """Distribute the heart's cells along the D-loop centre-line and label chambers by arc length. hpos=(N,3)
    in body axes (0=AP cranio-caudal, 1=DV, 2=LR). Returns (looped (N,3), chamber-fate names)."""
    rng = np.random.default_rng(seed)
    n = len(hpos)
    c = hpos.mean(0)
    scale = float(np.ptp(hpos[:, 0]) + 1e-6)                     # keep the loop the heart's own AP size
    t = rng.random(n)                                            # a fresh tube parameter (the blob has no order)
    t.sort()
    C = loop_centerline(t, pitx2=pitx2) * scale                 # x=cranio-caudal, y=DV, z=LR (matches body axes)
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    up = np.array([0.0, 0.0, 1.0]); N = np.cross(up, T); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    B = np.cross(T, N)
    ang = rng.random(n) * 2 * np.pi; rr = tube_r * scale * np.sqrt(rng.random(n))
    looped = C + (rr * np.cos(ang))[:, None] * N + (rr * np.sin(ang))[:, None] * B
    looped = looped - looped.mean(0) + c                        # re-centre on the heart's own centroid (address kept)
    fates = np.array([CHAMBER_TO_FATE[x] for x in chamber_of(t)])
    return looped, fates


def apply(base, F):
    """Reshape the model's heart into a looped tube in place (read-only elsewhere). Returns (base2, F2)."""
    ids = [FIDX[n] for n in HEART_FATES if n in FIDX]
    m = np.isin(F, ids)
    if m.sum() < 30:
        return base, F
    idx = np.where(m)[0]
    looped, fates = make_heart_tube(base[idx].astype(float))
    base2 = base.copy(); F2 = F.copy()
    base2[idx] = looped
    for k, nm in enumerate([str(x) for x in fates]):
        if nm in FIDX:
            F2[idx[k]] = FIDX[nm]
    return base2, F2


def main(ne=30000):
    from medic.adult_persistence_audit import build_base
    from medic.suborgan_attractor import gj_laplacian, fiedler
    base, F = build_base(ne)
    ids = [FIDX[n] for n in ("Atrium", "Ventricle", "Outflow") if n in FIDX]

    def order(b, f):
        m = np.isin(f, ids); pts = b[m].astype(float); lab = f[m]
        if len(pts) > 4000:
            s = np.random.default_rng(0).choice(len(pts), 4000, replace=False); pts, lab = pts[s], lab[s]
        g = fiedler(gj_laplacian(pts, np.ones(len(pts))))
        means = {FATES[i]: g[lab == i].mean() for i in ids}
        return round(float(max(means.values()) - min(means.values())), 3), means
    sep0, _ = order(base, F)
    b2, F2 = apply(base, F)
    sep1, m1 = order(b2, F2)
    seq = sorted(m1, key=lambda k: m1[k])
    print(f"heart chamber geodesic separation: BLOB {sep0}  ->  TUBE {sep1}")
    print(f"chamber sequence along the tube: {' -> '.join(seq)}  (canonical: Atrium -> Ventricle -> Outflow)")


if __name__ == "__main__":
    main()
