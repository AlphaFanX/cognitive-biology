"""FROZEN BENCHMARK HARNESS (suite v1.0).

This is the right-hand arrow of the standard loop
    ABC/SEdb/AlphaGenome  ->  heads  ->  MOSTA / Gray's / FMA-BodyParts3D anatomy
made rigorous and frozen: one specimen, every instrument, one scorecard JSON.

The harness is MODEL-AGNOSTIC by design: it scores only OUTPUTS (grown clouds,
forward organ shapes, physics equilibria) against measured references. It never
inspects the architecture. That is what lets it referee between architectures
(NCA+LGM vs ablations vs rivals) and between plants (kinematic vs PBD).

Tiers (the anatomy standards each tier answers to):
  A  grown-anatomy   -- ONE shared assemble() specimen scored by:
                          grays      : Gray's (1918) relational/identity checklist, 294 parts
                          canonical  : per-part D2 geometry vs FMA/BodyParts3D + OpenAnatomy
                          silhouette : whole-body D2 + width profile vs BP3D skin FMA7163
  B  embryo-forward  -- forward organs vs blob null on real atlases:
                          e165       : mouse MOSTA E16.5 dense 13-section reconstruction
                          zesta      : zebrafish ZESTA 3D
  C  equilibrium     -- PBD scrambled-start viscera packing: do measured DV depths
                        EMERGE from attachments+contacts+containment alone?
                        (the physics-plant gate: a kinematic model cannot game this)

Usage:
  python -m medic.benchmark_suite --tag kinematic-baseline
  python -m medic.benchmark_suite --tag quick --tiers A
Output:
  data/benchmark/scorecard_<tag>.json   (+ console summary table)

FREEZE DISCIPLINE: bump SUITE_VERSION whenever any metric module, reference
data file, or default config changes -- scorecards are comparable ONLY within
one suite version.
"""
import argparse, datetime, json, os, subprocess, time, traceback

import matplotlib
matplotlib.use("Agg")           # metric modules save PNGs; never open a GUI
import numpy as np

SUITE_VERSION = "1.4"    # v1.4 (cycle 82c): canonical tier per-part crc32-seeded rng (one shared stream ran in str-hash order: +-0.3 between processes). v1.3: tier A scores the MATURED STANDING specimen (mature_parts; the pre-standing
                         # assembly gut/heart/kidney were un-matured blobs) + Gut/Heart identity checks
                         # recalibrated on the reference (see grays_scorecard). v1.2 tier C median over seeds.
OUT_DIR = "data/benchmark"

# frozen defaults -- part of the suite definition, changing them = new version
FROZEN = dict(pbd_ne=30000, pbd_iters=400, pbd_seed=0, pbd_seeds=5)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       text=True).strip()
    except Exception:
        return "unknown"


def _timed(fn):
    t0 = time.time()
    out = fn()
    return out, round(time.time() - t0, 1)


def _guard(name, fn, results, durations):
    """Run one instrument; a failure is RECORDED, never fatal (the suite must always emit a scorecard)."""
    try:
        out, dt = _timed(fn)
        results[name] = out
        durations[name] = dt
        print(f"[suite] {name} done in {dt}s")
    except Exception as ex:
        traceback.print_exc()
        results[name] = {"error": f"{type(ex).__name__}: {ex}"}
        durations[name] = None
        print(f"[suite] {name} FAILED: {ex}")


# ---------------------------------------------------------------- tier A
def tier_a(results, durations):
    from medic.integrated_body import assemble, mature_parts
    print("[suite] tier A: assembling the ONE shared specimen ...")
    R, dt = _timed(assemble)
    durations["assemble"] = dt
    print(f"[suite] specimen assembled in {dt}s")
    # v1.3: score the MATURED STANDING body -- the same object the movie ships and the curve scores.
    # The raw assembly's gut/heart/kidney are un-matured (gut localE 1.30 = the blob control; the coil,
    # chamber split and kidney bean all live in the mature chain) -- scoring it failed real anatomy.
    R, dt = _timed(lambda: mature_parts(R))
    durations["mature_parts"] = dt
    print(f"[suite] specimen matured (standing body) in {dt}s")

    def grays():
        from medic import grays_scorecard
        grays_scorecard.run(R=R)
        s = json.load(open("data/grays_scorecard/_summary.json"))
        return {"n_parts": s["n_parts"], "mean_score": s["mean_score"],
                "fully_passing": s["fully_passing"], "by_category": s["by_category"],
                "worst": s["worst"][:6], "standard": "Gray's 1918 relational checklist"}

    def canonical():
        from medic import canonical_scorecard
        canonical_scorecard.run(R=R)
        s = json.load(open("data/organ_cascade/canonical_scorecard.json"))
        worst = [(p["part"], p["pct"]) for p in s["parts"][:6]]
        return {"n_scored": s["n_scored"], "overall_mean_pct": s["overall_mean_pct"],
                "by_category": s["by_category"], "worst": worst,
                "standard": "FMA / BodyParts3D + OpenAnatomy isolated meshes (D2)"}

    def silhouette():
        from medic import body_silhouette_score
        body_silhouette_score.run(R=R)
        s = json.load(open("data/organ_cascade/body_silhouette_score.json"))
        s["standard"] = "BodyParts3D skin FMA7163"
        return s

    _guard("grays", grays, results, durations)
    _guard("canonical", canonical, results, durations)
    _guard("silhouette", silhouette, results, durations)


# ---------------------------------------------------------------- tier B
def tier_b(results, durations):
    def e165():
        from medic import e165_match_score
        e165_match_score.main()
        s = json.load(open("data/organ_cascade/e165_match_score.json"))
        return {"forward_blob": s["forward_blob"], "forward_shape": s["forward_shape"],
                "gain_pct": round(100 * (s["forward_shape"] / s["forward_blob"] - 1), 1),
                "heart_tort": s["scores"].get("Heart", {}).get("tort"),
                "standard": "MOSTA E16.5 dense reconstruction (13 sections)"}

    def zesta():
        from medic import zesta_match_score
        zesta_match_score.main()
        s = json.load(open("data/organ_cascade/zesta_match_score.json"))
        out = {k: s[k] for k in ("forward_blob", "forward_shape") if k in s}
        if "forward_blob" in out and "forward_shape" in out:
            out["gain_pct"] = round(100 * (out["forward_shape"] / out["forward_blob"] - 1), 1)
        out["standard"] = "ZESTA zebrafish 3D atlas"
        return out

    _guard("e165", e165, results, durations)
    _guard("zesta", zesta, results, durations)


# ---------------------------------------------------------------- tier C
def tier_c(results, durations, ne, iters, seed):
    def pbd():
        from medic import pbd_viscera as PV
        from medic.adult_persistence_audit import build_base
        base, F = build_base(ne)
        tgt = PV._targets()
        masks = PV.organ_groups(F, tgt)
        built = PV.measure_dv(base, F, masks)
        organs = sorted(masks)
        # MEDIAN OVER SEEDS: the scrambled-start pack is jamming-multistable (the same body gives rank
        # 0.76-0.95 across seeds), so a single seed swings wildly under a small body change. The median
        # emergent depth over N_SEEDS samples the basins and gives a stable statistic that shifts smoothly
        # with the body instead of jumping -- suite v1.2, the fix that lets core-fate changes be judged.
        packed_last = None
        em_stack = []
        for s in range(FROZEN["pbd_seeds"]):
            packed = PV.pack(base, F, masks, iters=iters, seed=seed + s, verbose=False)
            em = PV.measure_dv(packed, F, masks)
            em_stack.append([em[o] for o in organs])
            packed_last = packed
        emergent = {o: float(np.median([row[i] for row in em_stack])) for i, o in enumerate(organs)}
        rows = [{"organ": o, "measured": round(tgt[o], 3), "built": round(built[o], 3),
                 "emergent": round(emergent[o], 3), "abs_err": round(abs(emergent[o] - tgt[o]), 3)}
                for o in organs]
        errs = [r["abs_err"] for r in rows]
        ms = [tgt[o] for o in organs]; es = [emergent[o] for o in organs]
        rho = float(np.corrcoef(np.argsort(np.argsort(ms)), np.argsort(np.argsort(es)))[0, 1])
        integ = None
        try:
            from medic import gray_integrity_audit as GIA
            rows_a = GIA.audit(packed_last, F)
            integ = f"{sum(1 for r in rows_a if r.get('ok'))}/{len(rows_a)}"
        except Exception as ex:
            integ = f"skipped: {ex}"
        n_emerge = sum(1 for r in rows if r["abs_err"] <= 0.10)
        return {"mean_abs_err": round(float(np.mean(errs)), 3), "rank_corr": round(rho, 2),
                "n_within_0p10": f"{n_emerge}/{len(rows)}", "integrity_after_packing": integ,
                "ne": ne, "iters": iters, "seeds": FROZEN["pbd_seeds"], "rows": rows,
                "standard": "BP3D in-situ DV depths, scrambled-start MEDIAN over seeds (jamming-robust)"}

    _guard("pbd_equilibrium", pbd, results, durations)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Cognimed frozen benchmark suite")
    ap.add_argument("--tag", default=None, help="scorecard tag, e.g. kinematic-baseline")
    ap.add_argument("--tiers", default="A,B,C", help="comma subset of A,B,C")
    ap.add_argument("--ne", type=int, default=FROZEN["pbd_ne"])
    ap.add_argument("--iters", type=int, default=FROZEN["pbd_iters"])
    ap.add_argument("--seed", type=int, default=FROZEN["pbd_seed"])
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag or stamp
    tiers = [t.strip().upper() for t in args.tiers.split(",") if t.strip()]
    results, durations = {}, {}
    t0 = time.time()
    print(f"[suite] v{SUITE_VERSION}  tag={tag}  tiers={tiers}  commit={_git_commit()}")

    if "A" in tiers:
        tier_a(results, durations)
    if "B" in tiers:
        tier_b(results, durations)
    if "C" in tiers:
        tier_c(results, durations, args.ne, args.iters, args.seed)

    card = {"suite_version": SUITE_VERSION, "tag": tag, "timestamp": stamp,
            "git_commit": _git_commit(), "tiers_run": tiers,
            "config": {"ne": args.ne, "iters": args.iters, "seed": args.seed},
            "durations_s": durations, "total_s": round(time.time() - t0, 1),
            "results": results}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"scorecard_{tag}.json")
    json.dump(card, open(path, "w"), indent=1)

    # ---- console summary
    print("\n" + "=" * 68)
    print(f"BENCHMARK SCORECARD  v{SUITE_VERSION}  [{tag}]  commit {card['git_commit']}")
    print("=" * 68)
    g = results.get("grays", {})
    if "mean_score" in g:
        print(f"  A grays       mean {g['mean_score']}   fully-pass {g['fully_passing']}/{g['n_parts']}")
    c = results.get("canonical", {})
    if "overall_mean_pct" in c:
        print(f"  A canonical   D2 {c['overall_mean_pct']}%   ({c['n_scored']} parts)")
    s = results.get("silhouette", {})
    if "silhouette_d2_pct" in s:
        print(f"  A silhouette  D2 {s['silhouette_d2_pct']}%   profile {s['profile_match_pct']}%")
    e = results.get("e165", {})
    if "forward_shape" in e:
        print(f"  B e165        blob {e['forward_blob']} -> forward {e['forward_shape']}  (+{e['gain_pct']}%)")
    z = results.get("zesta", {})
    if "forward_shape" in z:
        print(f"  B zesta       blob {z['forward_blob']} -> forward {z['forward_shape']}  (+{z.get('gain_pct')}%)")
    p = results.get("pbd_equilibrium", {})
    if "mean_abs_err" in p:
        print(f"  C equilibrium |err| {p['mean_abs_err']}  rank-corr {p['rank_corr']}  "
              f"within-0.10 {p['n_within_0p10']}  integrity {p['integrity_after_packing']}")
    for k, v in results.items():
        if isinstance(v, dict) and "error" in v:
            print(f"  !! {k} FAILED: {v['error']}")
    print(f"  total {card['total_s']}s -> {path}")


if __name__ == "__main__":
    main()
