"""Thin-walled LUMINAL forward models for the kidney and spinal cord -- the build the analysis-by-synthesis
loop PREDICTED (medic.organ_absynth). The solid-filled kidney bean and spinal rod leave a residual floor made
of TORTUOSITY and HOLLOWNESS: a kNN geodesic short-circuits straight across a filled volume, so a solid organ
cannot score the winding of a convoluted tubule or the cavity of a lumen, however correct its outer moments.

The heart already showed the fix (medic.heart_luminal, 0.71 solid -> 0.88 luminal): model the organ AS BUILT
-- a thin wall around a cavity, so the geodesic must follow the wall. Here the same treatment is given to:

  kidney  -- a reniform (C-bent) major axis carrying a WINDING tubule (the nephron/collecting convolution);
             the thin wall makes the winding score as tortuosity, a partial mid-pinch sets the moderate
             hollowness of the renal pelvis.
  spinal  -- a long curved cord as a thin-walled tube (the central canal), so the geodesic runs the length
             instead of short-circuiting across the rod, recovering the tortuosity of the body-curl arc.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.organ_luminal   (self-test vs real E16.5)
"""
import numpy as np
from medic.organ_3d_forward import rod_3d, sweep_tube


def kidney_centerline(nturns=3.5, r_helix=0.42, span=1.0, bend=2.0, pinch=0.30, N=300):
    """A reniform bean: a C-bent major axis (bend) carrying a helix (nturns = nephron convolution) whose radius
    pinches partway toward the axis at mid-organ (pinch = the renal-pelvis cavity -> moderate hollowness)."""
    t = np.linspace(0.0, 1.0, N)
    x = span * (t - 0.5)
    ybend = bend * (0.25 - (t - 0.5) ** 2)                       # the reniform C-arc (sets bend)
    rh = r_helix * (pinch + (1.0 - pinch) * np.abs(2.0 * t - 1.0))  # pinch toward the axis at t=0.5
    th = 2.0 * np.pi * nturns * t
    y = ybend + rh * np.cos(th)
    z = rh * np.sin(th)
    return np.c_[x, y, z]


def build_kidney_luminal(wall=0.05, nseg=9, **kw):
    C = kidney_centerline(**kw)
    return sweep_tube(C, radius=wall, nseg=nseg)[0]


def build_spinal_luminal(span=3.0, bend=0.55, turns=0.8, wall=0.06, nseg=8, N=110):
    """The spinal cord as a thin-walled tube (central canal) following the curved-body arc: the wall stops the
    geodesic short-circuiting across the cross-section, so the arc's length scores as tortuosity."""
    C = rod_3d(N=N, excess=0.35, span=span, handed=1.0, turns=turns, iters=900, bend=bend, amp=0.16)
    return sweep_tube(C, radius=wall, nseg=nseg)[0]


# tuned defaults (fixed by the self-test search below against the real E16.5 fingerprints)
KIDNEY_BEST = dict(nturns=3.5, r_helix=0.42, span=1.0, bend=2.0, pinch=0.30, wall=0.05, nseg=9)
SPINAL_BEST = dict(span=3.0, bend=0.55, turns=0.8, wall=0.06, nseg=8)


def e165_kidney_luminal():
    return build_kidney_luminal(**KIDNEY_BEST)


def e165_spinal_luminal():
    return build_spinal_luminal(**SPINAL_BEST)


if __name__ == "__main__":
    import itertools
    from medic.embryo_match_score import full_desc, fingerprint, shape_match
    K = ["elongation", "flatness", "bend", "tortuosity", "hollowness"]
    d = np.load("data/mosta/mouse_e165_3d.npz", allow_pickle=True)
    for name, real_key, grid, builder in [
        ("kidney", "Kidney",
         dict(nturns=[3.0, 3.5, 4.0], r_helix=[0.35, 0.42, 0.5], span=[0.9, 1.1],
              bend=[1.7, 2.0, 2.3], pinch=[0.25, 0.35, 0.5], wall=[0.045, 0.06]),
         build_kidney_luminal),
        ("spinal", "Spinal cord",
         dict(span=[2.8, 3.2], bend=[0.45, 0.6, 0.75], turns=[0.7, 1.0, 1.3], wall=[0.05, 0.07]),
         build_spinal_luminal),
    ]:
        real = fingerprint(d["xyz"][d["tissue"] == real_key])
        print(f"\n== {name} vs real {real_key}:", {k: round(real[k], 2) for k in K})
        keys = list(grid); best, best_s = None, -1
        for combo in itertools.product(*grid.values()):
            kw = dict(zip(keys, combo))
            try:
                s = np.mean([shape_match(full_desc(builder(**kw)), real) for _ in range(2)])
            except Exception:
                continue
            if s > best_s:
                best_s, best = s, kw
        m = full_desc(builder(**best))
        print(f"   BEST match {best_s:.3f}  {best}")
        print("   model:", {k: round(m[k], 2) for k in K})
