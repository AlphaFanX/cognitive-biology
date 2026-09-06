"""_bisect_build_base.py -- bisect the nondeterminism INSIDE adult_persistence_audit.build_base (cycle 82).

_phase_bisect found phase A (simulate 1->120k, 50 frames) IDENTICAL across three parallel processes and the
FIRST divergence at C0.build_base.B0 (raw AND sorted differ = value drift, not reordering). build_base is
simulate(30k) followed by ~20 heads; this harness monkeypatches every head the function calls (module-level
imports on adult_persistence_audit, lazy `from medic.X import f` inside the body -> patch medic.X.f) with a
wrapper that checksums its INPUT cloud and its OUTPUT, so the first stage whose OUTPUT differs while its INPUT
agrees is the leaking head. Inline code between two heads shows as out(k) identical / in(k+1) different.

    venv_win_new\\Scripts\\python.exe -m medic._bisect_build_base --tag bb1 [--repeat 2]
    venv_win_new\\Scripts\\python.exe -m medic._phase_bisect compare bb1 bb2 bb3
--repeat 2 runs build_base twice in ONE process (stages prefixed "p2:") -> in-process vs cross-process class.
"""
from __future__ import annotations
import argparse
import importlib
import os

import numpy as np

from medic._phase_bisect import Ledger, _h, _hs

L = None
PFX = ""

# (module, attribute, label) in build_base's call order
HEADS = [
    ("medic.adult_persistence_audit", "simulate", "simulate"),
    ("medic.adult_persistence_audit", "_symmetrize", "symmetrize"),
    ("medic.adult_persistence_audit", "flex", "flex"),
    ("medic.shoulder_girdle_head", "augment", "sg_augment"),
    ("medic.adult_persistence_audit", "_condense_organs", "condense"),
    ("medic.laterality_head", "lateralize", "lateralize"),
    ("medic.heart_tube_head", "apply", "heart_tube"),
    ("medic.subhead_program", "apply", "subheads"),
    ("medic.organ_aspect_head", "apply", "organ_aspect"),
    ("medic.ear_head", "shape_ears", "ears"),
    ("medic.body_cleanup_head", "apply", "cleanup"),
    ("medic.neck_head", "build_neck", "neck"),
    ("medic.skull_articulation_head", "build", "skull"),
    ("medic.grown_arm_head", "grow_arms", "grow_arms"),
    ("medic.grown_arm_head", "grow_feet", "grow_feet"),
    ("medic.dv_spread_head", "spread", "dv_spread"),
    ("medic.anoikis_head", "apply", "anoikis"),
    ("medic.density_floor_head", "apply", "density_floor"),
    ("medic.small_organ_form_head", "apply", "small_organ_form"),
]

_ORIG = {}


def _arr(x):
    return isinstance(x, np.ndarray) and x.dtype.kind in "fiub"


def _wrap(mod, name, label):
    m = importlib.import_module(mod)
    fn = getattr(m, name)
    _ORIG[(mod, name)] = fn

    def w(*a, **k):
        if label == "simulate":
            out = fn(*a, **k)
            fr = out[0]
            L.add(f"{PFX}{label}.out.P", np.asarray(fr[-1][3]),
                  dict(F_raw=_h(np.asarray(fr[-1][5])), F_sorted=_hs(np.asarray(fr[-1][5])), nfr=len(fr)))
            return out
        if len(a) and _arr(a[0]):
            ex = {}
            if len(a) > 1 and _arr(a[1]) and a[1].dtype.kind in "iu":
                ex = dict(F_raw=_h(a[1]))
            L.add(f"{PFX}{label}.in", a[0], ex)
        out = fn(*a, **k)
        first = out[0] if isinstance(out, tuple) else out
        ex = {}
        if isinstance(out, tuple):
            for j, o in enumerate(out[1:], 1):
                if _arr(o):
                    ex[f"out{j}_raw"] = _h(o)
        if _arr(first):
            L.add(f"{PFX}{label}.out", first, ex)
        return out

    setattr(m, name, w)


def main():
    global L, PFX
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ne", type=int, default=30000)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    L = Ledger(a.tag)
    print(f"build_base bisect tag={a.tag} pid={os.getpid()} hashseed={os.environ.get('PYTHONHASHSEED')} "
          f"omp={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    for mod, name, label in HEADS:
        _wrap(mod, name, label)
    from medic.adult_persistence_audit import build_base
    for r in range(a.repeat):
        PFX = "" if r == 0 else f"p{r + 1}:"
        base, F = build_base(a.ne)
        L.add(f"{PFX}build_base.RETURN", np.asarray(base, float), dict(F_raw=_h(F), n=int(len(base))))
    L.save()


if __name__ == "__main__":
    main()
