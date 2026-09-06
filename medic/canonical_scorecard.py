"""canonical_scorecard.py -- WHOLE-BODY D2 shape scorecard: every roster part vs its wired isolated canonical mesh.

Joins three pieces already built:
  * grays_scorecard.collect_all_parts(assemble())  -> our cloud for each of the 294 roster parts
  * data/canonical_map.json (from canonical_coverage) -> part -> canonical mesh spec (oa VTK | bp3d composite)
  * canonical_atlas.d2_signature                   -> correspondence-free, rotation/scale-invariant shape match

For each wired part: load the canonical mesh, D2-match our cloud against it, record descriptors. Aggregate by
category + overall + worst offenders. This is the whole-body analogue of the per-part identity scorecard, but
graded against REAL isolated human geometry instead of hand checklists.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.canonical_scorecard
Out:  data/organ_cascade/canonical_scorecard.json  (+ printed table)
"""
from __future__ import annotations
import os, json, zipfile, collections, zlib
import numpy as np

from medic.canonical_atlas import d2_signature, _sub, _vtk_points, load_bp3d, descriptors, MDIR

MAP = "data/canonical_map.json"


def load_spec(spec):
    """spec = ["oa", zip, member] | ["bp3d", fma, name] -> canonical point cloud. A LIST of bp3d ids =
    the union of the declared subtrees (organ__Gut = small+large intestine -- no single FMA concept
    bounds exactly the intestines; cycle 43, the boundary declaration)."""
    if spec[0] == "oa":
        _, zf, member = spec
        return _vtk_points(zipfile.ZipFile(os.path.join(MDIR, zf)).read(member))
    if spec[0] == "bp3d":
        if isinstance(spec[1], (list, tuple)):
            return np.vstack([load_bp3d(r) for r in spec[1]])
        return load_bp3d(spec[1])
    raise ValueError(spec)


def run(R=None):
    from medic.integrated_body import assemble
    from medic.grays_scorecard import collect_all_parts
    rng = np.random.default_rng(0)
    wired = json.load(open(MAP))
    print(f"wired map: {len(wired)} parts")
    if R is None:                                   # benchmark_suite passes one shared specimen
        print("assembling body ...")
        R = assemble()
    parts = collect_all_parts(R)
    # file-safe key = the scorecard json naming: ":" -> "__" and spaces -> "_" (e.g. "sternocleidomastoid R")
    our = {name.replace(":", "__").replace(" ", "_"): (cat, np.asarray(P, float)) for name, cat, P, meta in parts}

    results, cat_scores = [], collections.defaultdict(list)
    n_scored = n_skip = 0
    for key, spec in wired.items():
        if key not in our:
            n_skip += 1; continue
        cat, P = our[key]
        if len(P) < 20:
            n_skip += 1; continue
        canon = load_spec(spec)
        if len(canon) < 20:
            n_skip += 1; continue
        # PER-PART SEEDED STREAM (cycle 82c, SUITE v1.4): one shared `rng` consumed in loop order made a
        # part's D2 depend on everything scored before it -- and some upstream iteration runs in str-hash
        # order, so identical bodies scored +-0.3 apart between processes (PYTHONHASHSEED=0 twice ->
        # identical; seed 1 -> autopod 56.4 vs 55.5). crc32(part key) = the canon_frame_score v2 idiom.
        prng = np.random.default_rng(zlib.crc32(key.encode("utf-8")))
        # ORDER-INVARIANT (cycle 82c, the second half): the per-part stream alone left autopod / organ /
        # shoulder wobbling across hash seeds while every one of the 335 part clouds hashed identical
        # (_part_hash82.py) -- the canonical MESH loaders hand back their points in a hash-dependent
        # order, and D2 draws index pairs. Lexsort both clouds first: the score becomes a pure function
        # of the two point SETS.
        canon = canon[np.lexsort(np.round(canon, 9).T[::-1])]
        P = P[np.lexsort(np.round(P, 9).T[::-1])]
        cs, os_ = _sub(canon, 4000, prng), _sub(P, 4000, prng)
        d2 = float(1.0 - 0.5 * np.abs(d2_signature(cs, prng) - d2_signature(os_, prng)).sum())
        results.append(dict(part=key, cat=cat, src=spec[0], ref=spec[-1], pct=round(100 * d2, 1),
                            canonical=descriptors(cs), ours=descriptors(os_)))
        cat_scores[cat].append(d2)
        n_scored += 1

    results.sort(key=lambda r: r["pct"])
    overall = float(np.mean([r["pct"] for r in results])) if results else 0.0
    print(f"\nWHOLE-BODY D2 SHAPE SCORECARD vs isolated canonical meshes -- {n_scored} parts scored "
          f"({n_skip} wired parts had no cloud/mesh)")
    print(f"OVERALL MEAN D2 SHAPE-MATCH = {overall:.1f}%\n")
    print("by category (mean D2 %, n):")
    for cat in sorted(cat_scores):
        v = cat_scores[cat]
        print(f"  {cat:16s} {100*np.mean(v):5.1f}%   (n={len(v)})")
    print("\nworst 20 parts:")
    for r in results[:20]:
        print(f"  {r['pct']:5.1f}%  {r['part']:26s} [{r['src']}]  vs {r['ref']}")
    print("\nbest 10 parts:")
    for r in results[-10:]:
        print(f"  {r['pct']:5.1f}%  {r['part']:26s} [{r['src']}]  vs {r['ref']}")

    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(dict(n_scored=n_scored, overall_mean_pct=round(overall, 1),
                   by_category={c: round(100 * float(np.mean(v)), 1) for c, v in cat_scores.items()},
                   parts=results),
              open("data/organ_cascade/canonical_scorecard.json", "w"), indent=1)
    print("\nsaved data/organ_cascade/canonical_scorecard.json")


if __name__ == "__main__":
    run()
