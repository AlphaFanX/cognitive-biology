"""Thin-walled LUMINAL heart model -- the refinement flagged in Paper #9 Section 5.

A solid filled coil scores low tortuosity, because a kNN geodesic short-circuits straight across the touching
windings of a solid organ. The real E16.5 heart has BOTH a high tortuosity (4.47, the looped/convoluted tube)
AND a low hollowness (0.07, the centre of mass sits on tissue). A thin myocardial WALL around a lumen recovers
the tortuosity (the geodesic must follow the winding wall, windings separated by the loop pitch), and the one
trick that also keeps hollowness low is to pinch the loop radius toward zero at mid-axis, so the tube passes
through its own centre once -- a cell lands next to the centroid without filling the organ solid.

So the cardiac tube is modelled as it is built: a thin wall swept along a looping centre-line whose radius
pinches at the middle. build_heart_luminal() returns the wall point cloud; run as __main__ to search the
centre-line parameters against the real E16.5 heart fingerprint.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.heart_luminal
"""
import numpy as np
from medic.organ_3d_forward import sweep_tube


def heart_centerline(nturns=3.0, r_helix=0.55, span=2.4, axis_bend=0.45, pinch=0.12, twist=1.0, N=360):
    """A looping tube centre-line: a gentle C-arc major axis carrying a helix whose radius is PINCHED to
    ~pinch*r_helix at mid-axis, so the wall passes through the organ's own centre once (low hollowness) while
    the helix keeps the path winding (high tortuosity)."""
    t = np.linspace(0.0, 1.0, N)
    x = span * (t - 0.5)                                  # antero-posterior major axis
    ybend = axis_bend * (0.25 - (t - 0.5) ** 2)           # the D-loop's overall C-arc (sets bend)
    rh = r_helix * (pinch + (1.0 - pinch) * np.abs(2.0 * t - 1.0))   # radius -> pinch at t=0.5
    th = 2.0 * np.pi * nturns * t
    y = ybend + rh * np.cos(th)
    z = twist * rh * np.sin(th)
    return np.c_[x, y, z]


def build_heart_luminal(wall=0.05, nseg=9, **kw):
    """The thin-walled cardiac tube: a lumen wall of radius `wall` swept along the pinched looping centre-line."""
    C = heart_centerline(**kw)
    return sweep_tube(C, radius=wall, nseg=nseg)[0]


def e165_heart_luminal():
    """The tuned E16.5 luminal heart (parameters fixed by the search below)."""
    return build_heart_luminal(**BEST)


# Fixed by the __main__ search against the real E16.5 heart fingerprint (match 0.88; tortuosity 4.77 vs real
# 4.47, hollowness 0.07 vs 0.07, bend 0.17 vs 0.17): the loop is now scored, not short-circuited.
BEST = dict(nturns=4.0, r_helix=0.75, span=2.2, axis_bend=0.3, pinch=0.08, twist=1.0, wall=0.045, nseg=9)


if __name__ == "__main__":
    import itertools, json
    from medic.embryo_match_score import full_desc, fingerprint, shape_match
    d = np.load("data/mosta/mouse_e165_3d.npz", allow_pickle=True)
    real = fingerprint(d["xyz"][d["tissue"] == "Heart"])
    print("real heart:", {k: round(real[k], 2) for k in ["elongation", "flatness", "bend", "tortuosity", "hollowness"]})

    grid = dict(nturns=[3.0, 3.5, 4.0, 4.5], r_helix=[0.55, 0.65, 0.75], span=[2.2, 2.6, 3.0],
                axis_bend=[0.3, 0.45], pinch=[0.08, 0.12, 0.18], wall=[0.035, 0.045], nseg=[9])
    keys = list(grid)
    best, best_s = None, -1
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(keys, combo))
        try:
            m = full_desc(build_heart_luminal(**kw))
        except Exception:
            continue
        s = np.mean([shape_match(full_desc(build_heart_luminal(**kw)), real) for _ in range(2)])
        if s > best_s:
            best_s, best = s, kw
    m = full_desc(build_heart_luminal(**best))
    print(f"\nBEST match {best_s:.3f}  params {best}")
    print("  model:", {k: round(m[k], 2) for k in ["elongation", "flatness", "bend", "tortuosity", "hollowness"]})
    scal = lambda fp: {k: round(float(v), 3) for k, v in fp.items() if np.isscalar(v)}
    json.dump({"best": best, "match": round(float(best_s), 3), "model": scal(m), "real": scal(real)},
              open("data/organ_cascade/heart_luminal.json", "w"), indent=1)
    print("saved data/organ_cascade/heart_luminal.json  (paste BEST into heart_luminal.BEST)")
