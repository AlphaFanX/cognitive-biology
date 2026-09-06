"""
foot_completion.py -- THE DEVELOPMENTAL COMPLETION CHECK, foot pilot (cycle 63; Miles 2026-09-05:
"the foot itself will be some type of hierarchical manifold, roughly in the order of the descent of
the cells and the tissues. that development order is known, so it should be possible to follow the
development and to check that it completes. maybe fma can help here").

THE TARGET (from the FMA): the part-of subtree under the foot -- 26 named bones per side, every one
meshed in the bp3d roster (talus, calcaneus, navicular, 3 cuneiforms, cuboid, 5 metatarsals, 14
phalanges). THE LADDER (the known descent, each level TF-owned, each with a Carnegie window):
  plate      autopod territory condenses (HOXA13/HOXD13)          ~CS15
  rays       5 digital rays (SOX9 condensations)                   CS17-18
  segments   proximal->distal segmentation (GDF5 joint interzones) CS18-20
  separation interdigital apoptosis (BMP -- the death head's own)  CS19-21
  free toes                                                        CS21-23

THE INSTRUMENT: per staged movie frame (the shipped cloud frames carry fates), measure which levels
exist AS CELLS in the hind-distal territory: plate mass fraction, ray count (ML density peaks),
segments per ray (PD density gaps), digit separation (interdigital valleys). The completion curve =
level reached per stage vs the ladder. The cycle-61 null predicted: the middle levels (rays/segments
as condensations) never exist -- the foot jumps bud -> schematic shell, so the landed cells have no
descent to follow.

Run:  venv_win_new/Scripts/python.exe -m medic.foot_completion
Out:  data/organ_cascade/foot_completion.json
"""
from __future__ import annotations
import json, os
import numpy as np

OUT = "data/organ_cascade/foot_completion.json"


def _autopod_cells(xyz, fate, LIMB, kind):
    """An autopod territory: the limb-half's cells in the distal 12% of its own span.
    kind='foot' -> the leg half (apf < 0.5); kind='hand' -> the arm half (apf >= 0.5) --
    the sibling generalisation (cycle 67): the HAND ladder leads the foot by ~one Carnegie
    stage (hand plate CS15, digital rays CS16-17, free fingers CS20-22)."""
    x = xyz[:, 0]
    H = float(np.ptp(x)) + 1e-9
    apf = (x - x.min()) / H
    half = (fate == LIMB) & ((apf < 0.5) if kind == "foot" else (apf >= 0.5))
    if half.sum() < 30:
        return np.zeros(0, bool), half
    lo = np.percentile(xyz[half, 0], 12)
    return half & (x <= lo), half


def _peaks(vals, bins, rel_thr, floor=0.0):
    hist, _ = np.histogram(vals, bins=bins)
    thr = max(floor, rel_thr * hist.max())
    n, up = 0, False
    for v in hist:
        if v >= thr and not up:
            n += 1; up = True
        elif v < thr:
            up = False
    return int(n)


def _bud_side(xyz, fate, LIMB, kind, sgn, body_c=None):
    """One limb on one side: the kind-half's limb cells (foot = apf < 0.5, hand = apf >= 0.5) with
    z on side `sgn` of the half's own ML median; the AUTOPOD = the distal 12% along the bud's OWN
    principal axis, oriented away from the body centroid. FRAME-AGNOSTIC (cycle 82f): the old rule
    took the lowest-x 12% of the half, which is the standing limb's distal end but, on the laterally
    projecting embryonic paddle, only the CAUDAL edge of the limb half -- the ladder was being read
    off the wrong cells at every cloud stage. Returns (autopod indices, bud axis)."""
    x = xyz[:, 0]
    apf = (x - x.min()) / (float(np.ptp(x)) + 1e-9)
    half = (fate == LIMB) & ((apf < 0.5) if kind == "foot" else (apf >= 0.5))
    if half.sum() < 30:
        return np.zeros(0, int), None
    zmed = float(np.median(xyz[half, 2]))
    idx = np.where(half & (np.sign(xyz[:, 2] - zmed) == sgn))[0]
    if len(idx) < 30:
        return np.zeros(0, int), None
    B = xyz[idx]
    C = B - B.mean(0)
    ax = np.linalg.svd(C, full_matrices=False)[2][0]
    bc = xyz.mean(0) if body_c is None else np.asarray(body_c, float)
    if float((B.mean(0) - bc) @ ax) < 0:
        ax = -ax
    t = C @ ax
    return idx[t >= np.quantile(t, 0.88)], ax


def _rays_segs(A, ax):
    """(rays, segments) of one autopod: rays = density peaks across the FAN (the in-plane principal
    axis of the autopod cells, perpendicular to the bud axis); segments = density runs along the bud
    axis (joint interzones + 1). The old instrument histogrammed rays along ML and segments along the
    cells' own PCA-1 -- right for a standing foot, wrong for the embryonic paddle and the hanging hand
    (whose fan runs across DV)."""
    if len(A) < 12:
        return 0, 0
    C = A - A.mean(0)
    Cp = C - np.outer(C @ ax, ax)
    fan = np.linalg.svd(Cp, full_matrices=False)[2][0]
    rays = _peaks(Cp @ fan, 12, 0.25, floor=1.0)
    segs = _peaks(C @ ax, 10, 0.30)
    return rays, segs


def run():
    from medic.unified_embryo import FIDX
    LIMB = FIDX.get("Limb Bud")
    fr = json.load(open("data/movie/human_movie_frames.json"))
    frames = fr["frames"] if isinstance(fr, dict) and "frames" in fr else fr
    rows = []
    for f in frames:
        if not isinstance(f, dict) or "fate" not in f or "xyz" not in f:
            continue
        stage = str(f.get("stage", ""))
        if stage.startswith("cloud"):
            # cloud frames are labelled by PRC2; map to CS via the clock's anchors
            # (CS09=0.59, CS12=0.445, CS23=0.233 -- the cycle-19 deterministic ladder)
            try:
                prc2 = float(stage.split("PRC2")[-1].strip().split()[0])
            except (ValueError, IndexError):
                continue
            cs = 9 + (0.59 - prc2) / (0.59 - 0.445) * 3 if prc2 >= 0.445 else \
                 12 + (0.445 - prc2) / (0.445 - 0.233) * 11
            if cs < 13:
                continue                                     # pre-autopod window
            stage = f"CS{cs:.0f} (prc2 {prc2:.2f})"
        elif not any(k in stage for k in ("fetus", "newborn")):
            continue
        xyz = np.asarray(f["xyz"], float).reshape(-1, 3)
        fate = np.asarray(f["fate"])
        if len(fate) != len(xyz):
            continue
        for kind in ("foot", "hand"):
            n = 0
            rays = seg = 0
            for sgn in (-1.0, 1.0):
                A, ax = _bud_side(xyz, fate, LIMB, kind, sgn)
                n += len(A)
                if len(A) >= 12:
                    r_, s_ = _rays_segs(xyz[A], ax)
                    rays, seg = max(rays, r_), max(seg, s_)
            plate = n >= 30
            sep = plate and rays >= 4
            level = ("free-digits" if sep and seg >= 3 else "separation" if sep
                     else "segments" if seg >= 2 else "rays" if rays >= 3
                     else "plate" if plate else "none")
            # THE NOISE FLOOR, declared: below ~150 cells per autopod the ray/segment histograms
            # count NOISE, not condensations (a real 5-ray structure reads a stable 5). Low-N rows
            # are marked; the instrument's first conviction is ALLOCATION BEFORE CONDENSATION.
            if n < 150:
                level += " (low-N)"
            rows.append(dict(stage=stage[:28], kind=kind, n_cells=n, rays=rays, segments=seg,
                             separated=bool(sep), level=level))
    print("\nTHE AUTOPOD COMPLETION CURVES (foot ladder: plate CS15 | rays CS17-18 | segments "
          "CS18-20 | separation CS19-21 | free toes CS21-23; the HAND leads by ~one stage):")
    print(f"{'stage':30s} {'kind':5s} {'cells':>6s} {'rays':>5s} {'segs':>5s} {'sep':>4s}  level")
    for r in rows:
        print(f"{r['stage']:30s} {r['kind']:5s} {r['n_cells']:6d} {r['rays']:5d} {r['segments']:5d} "
              f"{'yes' if r['separated'] else 'no':>4s}  {r['level']}")
    for kind in ("foot", "hand"):
        done = [r for r in rows if r["kind"] == kind and r["level"] == "free-digits"]
        print(f"COMPLETION ({kind}): "
              f"{'reached free-digits at ' + done[0]['stage'] if done else 'NEVER COMPLETES'} "
              f"(target: {'26' if kind == 'foot' else '27'} named bones per side in the FMA roster)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w"), indent=1)
    return rows


def run_full(cache="data/organ_cascade/autopod_full_frames.npz"):
    """THE FULL-CLOUD COMPLETION CURVES (cycle 68). The shipped movie JSON carries the DISPLAY
    subsample (the rep_idx dots, ~30k/frame vs the 240k cloud -- the cycle-8/18 lying-instrument
    class), so run()'s ladder N was ~6-10x under-read and the 150 noise floor was partly a display
    artifact. This path re-runs the deterministic cloud phase (seed=0, the movie's own call) and
    scores the TRUE staged clouds; the limb cells + frame x-ranges cache to `cache` (one-time cost)."""
    from medic.unified_embryo import FIDX
    LIMB = FIDX.get("Limb Bud")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        packs = list(z["packs"])
    else:
        from medic.unified_embryo import simulate, _symmetrize
        from medic.human_movie import LIMB_SEARCHED, fate_map_for, CLOUD_N
        frames, _ = simulate(use_ecm=True, seed=0, n_start=1, n_end=CLOUD_N, limb_buds=True,
                             convergent_ext=1.0, limb_params=LIMB_SEARCHED,
                             fate_params=fate_map_for(CLOUD_N))
        packs = []
        from medic.digital_ray_head import apply_frame as _digital_rays_frame
        for born, _t, prc2, P, V, F in frames:
            if prc2 > 0.42:
                continue                                    # pre-autopod window
            Ps, _Vs, Fs = _symmetrize(P, V, F)
            Ps = np.asarray(Ps, np.float64)
            _digital_rays_frame(Ps, np.asarray(Fs), float(prc2), LIMB)   # the movie's own ladder head, same frame
            lm = np.asarray(Fs) == LIMB
            packs.append(dict(prc2=float(prc2), n_cloud=len(Ps),
                              xr=(float(Ps[:, 0].min()), float(Ps[:, 0].max())),
                              limb_xyz=np.asarray(Ps)[lm].astype(np.float32)))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, packs=np.array(packs, dtype=object))
    rows = []
    for pk in packs:
        prc2 = pk["prc2"]
        cs = 9 + (0.59 - prc2) / (0.59 - 0.445) * 3 if prc2 >= 0.445 else \
             12 + (0.445 - prc2) / (0.445 - 0.233) * 11
        if cs < 13:
            continue
        x0, x1 = pk["xr"]
        Lxyz = pk["limb_xyz"]
        # the cache holds limb cells only: the body's AP range is xr, its centroid ~ (mid-x, limb mean y,
        # the symmetrised midline z=0); _bud_side reads apf off the limb cells' own x, so pin the range
        # by appending the two body-extent anchors as non-limb fate rows
        anchors = np.array([[x0, Lxyz[:, 1].mean(), 0.0], [x1, Lxyz[:, 1].mean(), 0.0]], np.float32)
        xyz_ = np.vstack([Lxyz, anchors])
        fate_ = np.concatenate([np.full(len(Lxyz), LIMB), np.full(2, -1)])
        body_c = np.array([0.5 * (x0 + x1), float(Lxyz[:, 1].mean()), 0.0])
        for kind in ("foot", "hand"):
            n = 0
            rays = seg = 0
            for sgn in (-1.0, 1.0):
                A, ax = _bud_side(xyz_, fate_, LIMB, kind, sgn, body_c=body_c)
                n += len(A)
                if len(A) >= 12:
                    r_, s_ = _rays_segs(xyz_[A], ax)
                    rays, seg = max(rays, r_), max(seg, s_)
            plate = n >= 30
            sep = plate and rays >= 4
            level = ("free-digits" if sep and seg >= 3 else "separation" if sep
                     else "segments" if seg >= 2 else "rays" if rays >= 3
                     else "plate" if plate else "none")
            if n < 150:
                level += " (low-N)"
            rows.append(dict(stage=f"CS{cs:.0f} (prc2 {prc2:.2f})", kind=kind, n_cells=n,
                             rays=rays, segments=seg, separated=bool(sep), level=level))
    print("\nTHE FULL-CLOUD AUTOPOD COMPLETION CURVES (true staged clouds, not the display dots):")
    print(f"{'stage':30s} {'kind':5s} {'cells':>6s} {'rays':>5s} {'segs':>5s} {'sep':>4s}  level")
    for r in rows:
        print(f"{r['stage']:30s} {r['kind']:5s} {r['n_cells']:6d} {r['rays']:5d} {r['segments']:5d} "
              f"{'yes' if r['separated'] else 'no':>4s}  {r['level']}")
    for kind in ("foot", "hand"):
        done = [r for r in rows if r["kind"] == kind and r["level"] == "free-digits"]
        print(f"COMPLETION ({kind}, full cloud): "
              f"{'reached free-digits at ' + done[0]['stage'] if done else 'NEVER COMPLETES'}")
    json.dump(rows, open(OUT.replace(".json", "_full.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    import sys
    run_full() if "--full" in sys.argv else run()
