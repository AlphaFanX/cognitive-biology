"""body_silhouette_score.py -- score OUR whole-body form against THE CANONICAL HUMAN (BodyParts3D skin surface).

The per-part scorecard grades 294 named structures but never the SILHOUETTE -- the outer form of the whole body,
which is the thing a person actually looks like. BodyParts3D carries a full-body skin surface (FMA7163, ~196k
points, a standing human ~166 cm) -- the canonical human. This scores our generated body against it, two ways:

  * D2 shape-match  -- correspondence-free, rotation/scale-invariant: is our overall BODY FORM the right shape?
  * PROPORTIONS     -- heads-tall + biacromial(shoulder)/height + hip/height + a 24-bin width profile along the
                       body axis, after aligning both to a head-up, scale-normalised frame.

Our silhouette proxy = the outer surface of the body cloud (skin-fate cells if present, else the alpha-ish
outer shell of the whole cloud). This is the whole-body target the flesh stack / silhouette work drives toward.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.body_silhouette_score
Out: data/organ_cascade/body_silhouette_score.json
"""
from __future__ import annotations
import os, json
import numpy as np

from medic.canonical_atlas import load_bp3d, d2_signature, _sub

CANON_SKIN = "FMA7163"                                            # BodyParts3D "skin" = the canonical human surface


def _axis_frame(P):
    """Return points in a head-up, unit-mean-distance frame: long axis -> axis 0 (height), centred + scaled."""
    C = np.asarray(P, float) - np.asarray(P, float).mean(0)
    Vt = np.linalg.svd(C, full_matrices=False)[2]
    Q = C @ Vt.T
    Q /= (np.linalg.norm(Q, axis=1).mean() + 1e-9)
    return Q


def width_profile(P, nb=24):
    """Half-width (radial spread) along the long axis, nb bins -> the body's silhouette profile."""
    Q = _axis_frame(P)
    h = Q[:, 0]; hn = (h - h.min()) / (np.ptp(h) + 1e-9)
    b = np.clip((hn * nb).astype(int), 0, nb - 1)
    prof = np.zeros(nb)
    for k in range(nb):
        m = b == k
        if m.sum() > 4:
            prof[k] = np.percentile(np.linalg.norm(Q[m, 1:], axis=1), 85)
    return prof / (prof.max() + 1e-9)


def _gather(R, keys):
    out = []
    for k in keys:
        v = R.get(k)
        if isinstance(v, dict):
            for w in v.values():
                P = w.get("P") if isinstance(w, dict) else (w if isinstance(w, np.ndarray) else None)
                if P is not None and np.asarray(P).ndim == 2:
                    out.append(np.asarray(P, float))
        elif isinstance(v, np.ndarray) and v.ndim == 2:
            out.append(v)
    return out


def our_body(R, mode="flesh"):
    """Our silhouette cloud. 'flesh' = the FULL flesh stack (base + muscle bellies + adipose), whose OUTER
    envelope is the true silhouette (Miles: skin on fat on muscle on bone). 'skin' = the skin-fate shell only."""
    from medic.unified_embryo import FIDX
    base, F = np.asarray(R["base"], float), np.asarray(R["F"])
    if mode == "skin":
        skin = base[F == FIDX["Skin"]] if "Skin" in FIDX else base[:0]
        return (skin if len(skin) >= 2000 else base), "skin-fate"
    # NOTE: including limb_myoblasts/axial_muscle/limb_muscle REGRESSED the profile (66% vs 84% -- those arrays
    # skew the mass, top half read empty), so the silhouette flesh = base + the per-muscle bellies + adipose only.
    clouds = [base] + _gather(R, ("named_muscles", "head_muscle", "adipose"))
    return np.vstack(clouds), "flesh-stack (base+muscle+adipose)"


def run(R=None):
    from medic.integrated_body import assemble
    rng = np.random.default_rng(0)
    print("loading canonical human (BodyParts3D skin) ...")
    canon = load_bp3d(CANON_SKIN)
    if R is None:                                   # benchmark_suite passes one shared specimen
        print("assembling our body ...")
        R = assemble()
    ours, kind = our_body(R)
    print(f"our silhouette proxy = {kind}  (n={len(ours)})   canonical skin n={len(canon)}")

    cs, os_ = _sub(canon, 6000, rng), _sub(ours, 6000, rng)
    d2 = float(1.0 - 0.5 * np.abs(d2_signature(cs, rng) - d2_signature(os_, rng)).sum())

    pc, po = width_profile(cs), width_profile(os_)
    # align profile direction (PCA sign is arbitrary): flip ours if the reversed profile matches canon better
    if np.mean(np.abs(pc - po[::-1])) < np.mean(np.abs(pc - po)):
        po = po[::-1]
    prof_match = float(1.0 - np.mean(np.abs(pc - po)))

    res = dict(silhouette_d2_pct=round(100 * d2, 1), profile_match_pct=round(100 * prof_match, 1),
               proxy=kind, canon="BodyParts3D skin FMA7163")
    print("\n=== BODY SILHOUETTE vs THE CANONICAL HUMAN ===")
    print(f"  D2 shape-match      {res['silhouette_d2_pct']}%")
    print(f"  width-profile match {res['profile_match_pct']}%")
    print("\n  width profile (aligned, one end->other, 24 bins):")
    print("   canon:", " ".join("%.2f" % v for v in pc))
    print("   ours :", " ".join("%.2f" % v for v in po))
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(res, open("data/organ_cascade/body_silhouette_score.json", "w"), indent=1)
    print("\nsaved data/organ_cascade/body_silhouette_score.json")


if __name__ == "__main__":
    run()
