"""curve_train.py -- THE WIRED LOOP (Miles 2026-08-30): genome <-> heads <-> atlas, automated.

Guarded coordinate descent of the maturation's per-family ISOTROPY knobs (human_movie.ISO_LAMBDA)
against the staged-canon objective. The inner loop is FAST: one build_base, then each evaluation is a
mature_cloud call (seconds) scored by D2 against the ADULT canonical reference organs -- the same
signatures the frame-score curve uses at its adult end. Guards are non-tradeable: the whole-body D2
must not fall, and no other measured organ may drop more than GUARD points from the incumbent.
The winner is persisted to data/organ_cascade/iso_lambda.json (human_movie loads it on import), and
the caller then regenerates the movie + curve for the end-to-end verdict.

This is the harness pattern: any head's knob set can be descended the same way -- swap the knob dict
and the setter. Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.curve_train
"""
from __future__ import annotations
import json
import numpy as np

from zlib import crc32

from medic.canonical_atlas import d2_signature, _sub, surface_pts
from medic.unified_embryo import FIDX
from medic.subhead_program import expand_names


def _rng(*key):
    return np.random.default_rng(crc32("|".join(map(str, key)).encode()))


def _sig(P, n, *key):
    """Scorer v2 (matches canon_frame_score): surface-normalised cloud + per-organ seeded rng, deterministic."""
    r = _rng(*key)
    return d2_signature(_sub(surface_pts(P), n, r), r)

# reference organ label -> model fate parents (the frame-score groups, adult end)
GROUPS = {
    "heart":   ["Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow"],
    "kidney":  ["Kidney", "Nephron"],
    "liver":   ["Liver", "LiverHaem"],
    "lung":    ["Lung"],
    "spleen":  ["Spleen"],
    "stomach": ["Stomach", "Duodenum", "Foregut"],
    "brain":   ["Forebrain", "Telencephalon", "Midbrain", "Hindbrain", "Cerebellum", "OlfactoryBulb"],
    "gut":     ["Gut", "Hindgut"],
}
SEARCH = ["heart", "kidney", "liver", "lung", "spleen", "stomach"]   # families with an iso knob + a reference
LAMBDAS = [0.0, 0.5, 1.0]
# knob set 2: adult principal-frame aspect candidates (a1, a2) -- (1,1) = identity incumbent.
# gut/skin are deliberately NOT searched: gut is topology-limited (needs the coil mechanism, aspect
# would only game D2) and skin is the envelope's job, not a family capsule's.
A_GRID = [(1.0, 1.0), (0.8, 0.8), (0.65, 0.8), (0.8, 0.65), (0.65, 0.65),
          (0.5, 0.7), (0.7, 0.5), (1.25, 0.8), (0.8, 1.25), (1.25, 1.25), (0.5, 0.5)]
GUARD_ORGAN = 2.0                                          # no other organ may drop more than this
GUARD_WHOLE = 0.4                                          # whole-body may not drop more than this


def run():
    import medic.human_movie as HM
    from medic.adult_persistence_audit import build_base
    rng = np.random.default_rng(0)

    ad = json.load(open("data/movie/canon_reference.json"))
    ref = {}
    allp = [np.asarray(ad["skin"], float).reshape(-1, 3)]
    for o in ad["organs"]:
        P = np.asarray(o["xyz"], float).reshape(-1, 3)
        allp.append(P)
        if len(P) >= 30:
            ref[o["label"]] = _sig(P, 3000, "ADULT", o["label"])
    ref["_whole"] = _sig(np.vstack(allp), 4000, "ADULT", "_whole")

    print("building the body once ...")
    B0, BF = build_base(30000)
    gids = {g: np.isin(BF, [FIDX[n] for n in expand_names(names) if n in FIDX])
            for g, names in GROUPS.items()}

    def score(lam, asp):
        HM.ISO_LAMBDA.clear(); HM.ISO_LAMBDA.update(lam)
        HM.ADULT_ASPECT.clear(); HM.ADULT_ASPECT.update(asp)
        Q = HM.mature_cloud(B0.copy(), BF, 1.0, HM.MATURE_SEARCHED)
        out = {}
        for g, m in gids.items():
            if g in ref and m.sum() >= 30:
                out[g] = 100 * (1 - 0.5 * np.abs(ref[g] - _sig(Q[m], 3000, "ADULT", g)).sum())
        out["_whole"] = 100 * (1 - 0.5 * np.abs(ref["_whole"] - _sig(Q, 4000, "ADULT", "_whole")).sum())
        return out

    def guard_ok(fam, s, best):
        return (s.get(fam, 0) > best.get(fam, 0) + 0.2
                and s["_whole"] >= best["_whole"] - GUARD_WHOLE
                and all(s.get(g, 0) >= best.get(g, 0) - GUARD_ORGAN
                        for g in GROUPS if g != fam and g in s))

    best_lam = dict(HM.ISO_LAMBDA)
    best_asp = dict(HM.ADULT_ASPECT)
    best = score(best_lam, best_asp)
    print("incumbent:", {k: round(v, 1) for k, v in best.items()}, "lam:", best_lam, "asp:", best_asp)
    for it in range(2):                                    # phase 1: isotropy (knob set 1), two passes
        for fam in SEARCH:
            for lv in LAMBDAS:
                if abs(best_lam.get(fam, 0.0) - lv) < 1e-9:
                    continue
                trial = dict(best_lam); trial[fam] = lv
                s = score(trial, best_asp)
                ok = guard_ok(fam, s, best)
                print(f"  iso pass{it} {fam}={lv}: {fam} {s.get(fam,0):.1f} whole {s['_whole']:.1f}  "
                      f"[{'ACCEPT' if ok else 'reject'}]")
                if ok:
                    best_lam, best = trial, s
    for it in range(2):                                    # phase 2: adult aspect (knob set 2), two passes
        for fam in SEARCH:
            if fam == "kidney":
                continue    # PAIRED organ: family PCA axis-1 is the pair line, family aspect distorts pair
                            # geometry (cycle-3 A/B). Needs a per-side aspect knob before it can be searched.
            cur = best_asp.get(fam, (1.0, 1.0))
            for av in A_GRID:
                if abs(cur[0] - av[0]) < 1e-9 and abs(cur[1] - av[1]) < 1e-9:
                    continue
                trial = dict(best_asp); trial[fam] = av
                s = score(best_lam, trial)
                ok = guard_ok(fam, s, best)
                print(f"  asp pass{it} {fam}={av}: {fam} {s.get(fam,0):.1f} whole {s['_whole']:.1f}  "
                      f"[{'ACCEPT' if ok else 'reject'}]")
                if ok:
                    best_asp, best = trial, s
                    cur = av
    best_asp = {k: v for k, v in best_asp.items() if abs(v[0] - 1.0) > 1e-9 or abs(v[1] - 1.0) > 1e-9}
    print("\nWINNER lam:", best_lam)
    print("WINNER asp:", best_asp)
    print("scores:", {k: round(v, 1) for k, v in best.items()})
    json.dump(best_lam, open("data/organ_cascade/iso_lambda.json", "w"), indent=1)
    json.dump(best_asp, open("data/organ_cascade/adult_aspect.json", "w"), indent=1)
    print("saved iso_lambda.json + adult_aspect.json -- regenerate the movie + curve to verify end-to-end")


if __name__ == "__main__":
    run()
