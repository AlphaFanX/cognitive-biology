"""Analysis-by-synthesis loop across ALL the forward organs (generalises medic.heart_absynth).

For each organ: a parametric forward generator whose parameters are named GENES (each a frozen direction,
free magnitude), a real E16.5 target from the dense reconstruction, and the same coarse-search + LM relaxation
that tunes the gene magnitudes within biological ranges to match the real organ. Reports, per organ, whether
the loop improves the match, how much of the shape gap the genomic knobs close (genomic-closable) versus the
residual floor no gene can close (the model-class / missing-physics limit), and whether the fitted knobs move
in their gene's expected direction. Then the whole-body trunk is run the same way on the generated silhouette.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.organ_absynth
"""
import os, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from medic.embryo_match_score import full_desc, fingerprint, KEYS, shape_match
from medic.heart_luminal import build_heart_luminal
from medic.organ_3d_forward import rod_3d, sweep_tube
from medic.organ_luminal import build_kidney_luminal, build_spinal_luminal

HERE = os.path.dirname(__file__)
NPZ = os.path.join(HERE, "..", "data", "mosta", "mouse_e165_3d.npz")
OUT = os.path.join(HERE, "..", "data", "organ_cascade", "organ_absynth.json")
FIG = os.path.join(HERE, "..", "data", "organ_cascade", "organ_absynth.png")


def _ellipsoid(ax, ay, az, n=1400, seed=7):
    rng = np.random.RandomState(seed)
    v = rng.standard_normal((n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    r = rng.uniform(0, 1, n) ** (1 / 3.)
    return (r[:, None] * v) * np.array([ax, ay, az])


# module-level builders (picklable, so the organs can run across CPU processes -- the GPU does not help this
# workload: tiny point clouds + a scipy graph-geodesic + a sequential buckling loop; the win is CPU parallelism)
def heart_build(p):   return build_heart_luminal(twist=1.0, nseg=9, **p)
def gut_build(p):     return sweep_tube(rod_3d(100, 0.5, p["span"], 1.0, p["turns"], 700, p["bend"], 0.22),
                                        radius=p["radius"], nseg=12)[0]
def kidney_build(p):  return build_kidney_luminal(nturns=p["nturns"], r_helix=p["r_helix"], span=1.0,
                                                  bend=p["bend"], pinch=p["pinch"], wall=0.045, nseg=9)
def spinal_build(p):  return build_spinal_luminal(span=p["span"], bend=p["bend"], turns=p["turns"],
                                                  wall=p["wall"], nseg=8)
def liver_build(p):   return _ellipsoid(p["ax"], p["ay"], p["az"])
def brain_build(p):   return _ellipsoid(p["ax"], p["ay"], p["az"], n=1600)


# organ -> forward builder + gene knobs (gene, param, start, lo, hi, sign toward the real organ). Kidney and
# spinal cord now use the LUMINAL builders (medic.organ_luminal): the analysis-by-synthesis loop predicted their
# residual floors were unclosed TORTUOSITY / HOLLOWNESS a solid organ cannot score -- this tests whether the
# thin-walled treatment (proven on the heart, 0.71->0.88) closes them.
ORGANS = {
    "Heart": dict(build=heart_build, knobs=[
        ("PITX2", "axis_bend", 0.12, 0.05, 0.50, +1),   # D-loop laterality
        ("NKX2-5", "nturns",   2.00, 1.50, 5.00, +1),   # looping / tortuosity
        ("HAND",  "r_helix",   0.45, 0.30, 0.90, +1),   # chamber ballooning
        ("TTN",   "span",      2.90, 1.80, 3.20, -1),   # tube length
        ("NPPA",  "pinch",     0.30, 0.04, 0.40, -1),   # lumen pinch
        ("TBX5",  "wall",      0.060, 0.030, 0.080, -1)]),  # wall
    "GI tract": dict(build=gut_build, knobs=[
        ("CDX2", "turns",  0.50, 0.30, 2.00, +1),       # hindgut coiling
        ("PITX2", "bend",  0.30, 0.20, 1.20, +1),       # midgut flexure / rotation
        ("FGF8", "span",   1.80, 0.80, 2.20, -1),       # axis elongation
        ("VIL1", "radius", 0.55, 0.30, 0.60, -1)]),     # lumen / wall
    "Kidney": dict(build=kidney_build, knobs=[          # LUMINAL: reniform bean + winding nephron tubule
        ("GDNF",  "bend",    1.00, 0.50, 2.50, +1),     # ureteric branching -> reniform bend
        ("SIX2",  "nturns",  1.50, 1.00, 4.00, +1),     # nephron convolution -> tortuosity
        ("PAX2",  "r_helix", 0.35, 0.25, 0.50, +1),     # nephrogenic-zone radius
        ("WT1",   "pinch",   0.40, 0.15, 0.60, -1)]),   # renal pelvis cavity -> hollowness
    "Spinal cord": dict(build=spinal_build, knobs=[     # LUMINAL: central-canal tube along the curved cord
        ("HOXC",  "span",  2.60, 2.00, 3.20, +1),       # AP length (Hox)
        ("GDF11", "turns", 1.50, 0.70, 3.50, +1),       # segmental meander -> tortuosity
        ("FGF8",  "bend",  0.50, 0.20, 0.90, +1),       # flexure
        ("PAX6",  "wall",  0.10, 0.06, 0.20, +1)]),     # calibre (thicker -> less elongated)
    "Liver": dict(build=liver_build, knobs=[
        ("HNF4A", "ax", 1.00, 0.80, 2.00, +1),          # hepatoblast expansion (elongation)
        ("FOXA2", "ay", 1.00, 0.40, 1.20, -1),          # lobe flattening
        ("TBX3",  "az", 1.00, 0.40, 1.00, -1)]),
    "Brain": dict(build=brain_build, knobs=[
        ("OTX2",  "ax", 1.00, 0.80, 2.00, +1),          # AP elongation of the vesicles
        ("FOXG1", "ay", 1.20, 0.60, 1.40, -1),
        ("EMX2",  "az", 1.20, 0.60, 1.30, -1)]),
}


def optimize(build, knobs, real, n_search=140, n_lm=12, seed=0):
    params = [k[1] for k in knobs]
    base = np.array([k[2] for k in knobs]); lo = np.array([k[3] for k in knobs]); hi = np.array([k[4] for k in knobs])
    lob, hib = lo - base, hi - base
    ref = np.array([real[k] for k in KEYS], float); scale = np.abs(ref) + 1e-6

    def pof(b):
        p = np.clip(base + b, lo, hi); return {params[i]: float(p[i]) for i in range(len(knobs))}

    def dv(b, navg=3):
        ds = [full_desc(build(pof(b))) for _ in range(navg)]
        d = {k: float(np.mean([di[k] for di in ds])) for k in KEYS}
        return np.array([d[k] for k in KEYS], float), d

    def resid(b):
        d, _ = dv(b); return (d - ref) / scale

    b = np.zeros(len(knobs)); d0, _ = dv(b); r0 = resid(b); gap0 = float(np.linalg.norm(r0)) + 1e-9
    hist = [shape_match(dict(zip(KEYS, d0)), real)]
    rng = np.random.RandomState(seed); best_b, best_m = b.copy(), hist[0]
    for _ in range(n_search):
        c = lob + rng.rand(len(knobs)) * (hib - lob)
        dS, _ = dv(c, 2); m = shape_match(dict(zip(KEYS, dS)), real)
        if m > best_m: best_m, best_b = m, c.copy()
        hist.append(best_m)
    b = best_b
    lam = 1e-2
    for _ in range(n_lm):
        r = resid(b); c = float(r @ r)
        J = np.zeros((len(KEYS), len(knobs)))
        for j in range(len(knobs)):
            h = 0.05 * (hi[j] - lo[j]); bj = b.copy(); bj[j] = np.clip(bj[j] + h, lob[j], hib[j])
            J[:, j] = (resid(bj) - r) / (bj[j] - b[j] + 1e-12)
        A = J.T @ J; g = J.T @ r
        for _t in range(8):
            step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
            bn = np.clip(b + step, lob, hib); rn = resid(bn)
            if float(rn @ rn) < c: b = bn; lam = max(lam * 0.5, 1e-4); break
            lam = min(lam * 3.0, 1e4)
        dS, _ = dv(b); hist.append(shape_match(dict(zip(KEYS, dS)), real))
    dF, _ = dv(b); rF = resid(b); floor = float(np.linalg.norm(rF)) / gap0
    # residual attribution: which descriptor the genes could NOT close = what the floor is made of
    ar = np.abs(rF); attr = {KEYS[i]: round(float(ar[i] / (ar.sum() + 1e-9)), 3) for i in range(len(KEYS))}
    dominant = KEYS[int(np.argmax(ar))]
    # direction is only TESTED for a knob that moved meaningfully (>8% of its range); a knob left in place
    # means its descriptor was already satisfied, so it carries no directional evidence either way.
    dirs, meaningful, correct = [], 0, 0
    for i, (g, p, st, l, h, sgn) in enumerate(knobs):
        moved = abs(b[i]) > 0.08 * (h - l)
        ok = (np.sign(b[i]) == sgn) if moved else None
        if moved:
            meaningful += 1; correct += int(ok)
        dirs.append(dict(gene=g, param=p, delta=round(float(b[i]), 3), expect=sgn,
                         moved=bool(moved), ok=(bool(ok) if moved else None)))
    return dict(match_start=round(hist[0], 3), match_final=round(hist[-1], 3),
                genomic_closable=round(1 - floor, 3), residual_floor=round(floor, 3),
                dir_correct=correct, dir_meaningful=meaningful, n_knobs=len(knobs), knobs=dirs,
                floor_attr=attr, floor_dominant=dominant,
                real={k: round(real[k], 2) for k in KEYS},
                final={k: round(float(dF[i]), 2) for i, k in enumerate(KEYS)})


def _worker(name):
    z = np.load(NPZ, allow_pickle=True); xyz, tissue = z["xyz"], z["tissue"]
    real = fingerprint(xyz[tissue == name])
    return name, optimize(ORGANS[name]["build"], ORGANS[name]["knobs"], real)


def run(parallel=True):
    names = list(ORGANS)
    res = {}
    if parallel:                                        # the 6 organs are independent -> run across CPU cores
        with ProcessPoolExecutor(max_workers=min(len(names), os.cpu_count() or 4)) as ex:
            for name, r in ex.map(_worker, names):
                res[name] = r
    else:
        for name in names:
            _, res[name] = _worker(name)
    print(f"{'organ':13s} {'match':>13s}  {'genomic':>8s} {'floor':>6s}  {'dir':>5s}  floor made of")
    for name in names:
        r = res[name]
        print(f"{name:13s} {r['match_start']:.2f} -> {r['match_final']:.2f}   "
              f"{100*r['genomic_closable']:5.1f}%  {100*r['residual_floor']:4.1f}%  "
              f"{r['dir_correct']}/{r['dir_meaningful']}   {r['floor_dominant']}")
    gc = np.mean([r["genomic_closable"] for r in res.values()])
    m0 = np.mean([r["match_start"] for r in res.values()]); mF = np.mean([r["match_final"] for r in res.values()])
    dok = sum(r["dir_correct"] for r in res.values()); dn = sum(r["dir_meaningful"] for r in res.values())
    summ = dict(mean_match_start=round(float(m0), 3), mean_match_final=round(float(mF), 3),
                mean_genomic_closable=round(float(gc), 3), dir_consistency=f"{dok}/{dn}")
    print(f"\nMEAN  match {m0:.2f} -> {mF:.2f}   genomic-closable {100*gc:.1f}%   "
          f"gene-direction {dok}/{dn} (of meaningfully-moved knobs)")
    json.dump(dict(summary=summ, organs=res), open(OUT, "w"), indent=1)
    _figure(res)
    print(f"wrote {OUT} and {FIG}")
    return res


def _figure(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(res); x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor="white")
    ax[0].bar(x - 0.2, [res[n]["match_start"] for n in names], 0.4, label="start (bland organ)", color="#bbb")
    ax[0].bar(x + 0.2, [res[n]["match_final"] for n in names], 0.4, label="after the loop", color="#d64545")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=25, ha="right"); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("shape match to real E16.5 organ"); ax[0].legend(); ax[0].set_title("Analysis-by-synthesis, every organ")
    ax[1].bar(x, [res[n]["genomic_closable"] for n in names], color="#3aa869")
    ax[1].bar(x, [res[n]["residual_floor"] for n in names], bottom=[res[n]["genomic_closable"] for n in names], color="#ccc")
    for i, n in enumerate(names):                     # name what the floor is made of
        if res[n]["residual_floor"] > 0.05:
            ax[1].text(i, min(0.98, res[n]["genomic_closable"] + res[n]["residual_floor"] / 2),
                       res[n]["floor_dominant"][:4], ha="center", va="center", fontsize=7, color="#555", rotation=90)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=25, ha="right"); ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("fraction of the shape gap")
    ax[1].set_title("genomic-closable (green) vs residual floor (grey, labelled by dominant descriptor)")
    plt.tight_layout(); plt.savefig(FIG, dpi=130, facecolor="white")


if __name__ == "__main__":
    run()
