"""Richer body-plan generator v2 -- the two fixes the stage-matched builder named (Miles, 2026-07-26).

The v1 generator (medic.body_plan_absynth.build_body_plan) is a single circular-arc tube with a stub limb, so
(a) length and curvature are COUPLED -- a strong curl collapses the elongation descriptor -- which capped the
fit to the long-and-curled human embryo, and (b) it has no articulated limb, which capped the fit to the flat,
limbed adult. v2 fixes both:

  SEGMENTED AXIS -- the centre-line is built from a per-segment curvature (an anterior/cervical flexure and a
  trunk flexure) integrated at UNIT rate, so total turn is set by the flexure knobs and the axis LENGTH is a
  separate scale knob: a long body can carry a gentle C (high elongation AND real curvature), decoupling the
  two descriptors that fought in v1.

  ARTICULATED LIMBS -- four two-segment limbs (paired fore and hind) project laterally with an elbow/knee
  bend, with a length and a spread knob, so the mediolateral extent (flatness) and the four-branch topology
  (tortuosity) of a limbed body are reachable.

run() refits v2 STAGE-MATCHED (mouse embryo -> human embryo HESTA = species adapter; -> adult MakeHuman =
maturation) and reports the distance to each, to see how far the richer generator closes the ~16% v1 gap
toward the 10% target.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.body_builder_v2
Out:  data/organ_cascade/body_builder_v2.{json,png}
"""
import os, json
import numpy as np
from medic.embryo_match_score import fingerprint, KEYS
from medic.organ_absynth import optimize
from medic.body_plan_absynth import _pca2

HERE = os.path.dirname(__file__)
HESTA = os.path.join(HERE, "..", "data", "hesta", "hesta_dense_3d.npz")
MH = os.path.join(HERE, "..", "data", "bodybase", "makehuman_decimated.npz")
OUT = os.path.join(HERE, "..", "data", "organ_cascade", "body_builder_v2.json")
FIG = os.path.join(HERE, "..", "data", "organ_cascade", "body_builder_v2.png")

# gene, param, start, lo, hi, sign
KNOBS_V2 = [
    ("CDX2/HOX",   "L",          2.4, 1.2, 5.0, +1),   # axis length -> elongation (decoupled from curl)
    ("WNT-PCP",    "w_ml",       0.45, 0.20, 1.00, -1),
    ("BMP/FOXA2",  "w_dv",       0.40, 0.18, 0.90, -1),
    ("HOXB/cerv",  "flexA",      0.30, 0.00, 2.20, +1), # anterior/cervical flexure (segmented)
    ("SHH/noto",   "flexB",      0.30, 0.00, 2.20, +1), # trunk flexure (segmented)
    ("OTX2/SIX3",  "head",       1.30, 1.00, 2.40, +1),
    ("CDX/tail",   "taper",      0.40, 0.10, 0.90, +1),
    ("TBX5/FGF10", "limb_len",   0.30, 0.00, 1.60, +1), # articulated limb length (x body scale)
    ("SHH-ZRS",    "limb_spread",0.50, 0.00, 1.00, +1), # lateral projection of the limbs
]


def _centerline(t, L, flexA, flexB):
    """segmented curvature integrated at unit rate: total turn = flexA (anterior) + flexB (posterior),
    independent of L; position scales with L (arc length)."""
    segA = np.clip(t / 0.4, 0, 1)                     # 0..1 over the anterior 40%
    segB = np.clip((t - 0.4) / 0.6, 0, 1)             # 0..1 over the posterior 60%
    phi = flexA * (segA - 0.5) + flexB * segB         # tangent angle (radians)
    # integrate (cos phi, sin phi) along t, scaled by L
    order = np.argsort(t)
    dt = np.gradient(np.sort(t))
    cx = np.zeros_like(t); cy = np.zeros_like(t)
    xs = np.cumsum(np.cos(phi[order]) * dt) * L
    ys = np.cumsum(np.sin(phi[order]) * dt) * L
    cx[order] = xs - xs.mean(); cy[order] = ys - ys.mean()
    return cx, cy, phi


def build_body_plan_v2(p, n=2000, seed=5):
    rng = np.random.RandomState(seed)
    L, w_ml, w_dv = p["L"], p["w_ml"], p["w_dv"]
    flexA, flexB, head, taper = p["flexA"], p["flexB"], p["head"], p["taper"]
    limb_len, limb_spread = p["limb_len"], p["limb_spread"]
    nt = int(n * 0.6)
    t = rng.rand(nt)
    head_bulge = 1 + (head - 1) * np.exp(-((t - 0.12) / 0.12) ** 2)
    taper_prof = 1 - taper * np.clip((t - 0.5) / 0.5, 0, 1)
    rad = head_bulge * taper_prof
    ang = rng.rand(nt) * 2 * np.pi; rr = np.sqrt(rng.rand(nt))
    c_ml = rr * np.cos(ang) * w_ml * rad
    c_dv = rr * np.sin(ang) * w_dv * rad
    cx, cy, phi = _centerline(t, L, flexA, flexB)
    x = cx + c_dv * (-np.sin(phi)); y = cy + c_dv * np.cos(phi); z = c_ml
    pts = [np.stack([x, y, z], 1)]

    # articulated limbs: fore (t~0.28) and hind (t~0.72), left/right, two segments with an elbow/knee bend
    if limb_len > 0.02:
        cxx, cyy, pp = _centerline(np.array([0.28, 0.72]), L, flexA, flexB)
        nlp = max(20, int(n * 0.1))
        for j, tb in enumerate((0, 1)):                 # fore, hind
            base = np.array([cxx[tb], cyy[tb], 0.0])
            for side in (+1, -1):
                d = np.array([0.15, -0.35, side * (0.4 + 0.6 * limb_spread)])
                d = d / np.linalg.norm(d)
                p0 = base + np.array([0, 0, side * w_ml * 0.8])
                seglen = limb_len * L / 2
                for s in range(2):                       # upper + lower limb segment
                    k = nlp // 2
                    u = rng.rand(k)
                    seg = p0[None, :] + d[None, :] * (seglen * u[:, None]) + rng.randn(k, 3) * 0.03
                    pts.append(seg)
                    p0 = p0 + d * seglen
                    d = d * 0.7 + np.array([0.0, -0.6, 0.0]); d /= np.linalg.norm(d)  # joint flex ventrally
    return np.vstack(pts)


def _fit(fp):
    res = optimize(build_body_plan_v2, KNOBS_V2, fp, n_search=200, n_lm=14, seed=0)
    perr = {k: round(100 * abs(res["final"][k] - fp[k]) / (abs(fp[k]) + 1e-6), 1) for k in KEYS}
    return res, perr, float(np.mean(list(perr.values())))


def run():
    rng = np.random.default_rng(0)
    ze = np.load(HESTA, allow_pickle=True); Xe = ze["xyz"]; Ve = Xe[rng.choice(len(Xe), 3000, replace=False)]
    emb = fingerprint(Ve)
    za = np.load(MH, allow_pickle=True); Va = np.asarray(za["V"], float); Va = Va[rng.choice(len(Va), 3000, replace=False)]
    adult = fingerprint(Va)
    print("human EMBRYO (HESTA):", {k: emb[k] for k in KEYS})
    print("human ADULT (MakeHuman):", {k: adult[k] for k in KEYS})

    res_e, perr_e, mean_e = _fit(emb)
    res_a, perr_a, mean_a = _fit(adult)

    print(f"\n=== v2 richer builder ===")
    print(f"SPECIES (mouse->human EMBRYO): match {res_e['match_start']:.2f} -> {res_e['match_final']:.2f}, "
          f"MEAN {mean_e:.1f}% off  (v1 was ~16.3%)")
    for k in KEYS:
        print(f"   {k:12s} builder {res_e['final'][k]:6.2f}  target {emb[k]:6.2f}  {perr_e[k]:5.1f}%")
    print(f"MATURATION endpoint (ADULT MakeHuman): match {res_a['match_start']:.2f} -> {res_a['match_final']:.2f}, "
          f"MEAN {mean_a:.1f}% off  (v1 was ~16.8%)")
    for k in KEYS:
        print(f"   {k:12s} builder {res_a['final'][k]:6.2f}  target {adult[k]:6.2f}  {perr_a[k]:5.1f}%")
    print(f"\nknobs (human embryo fit): " + ", ".join(f"{kb[0].split('/')[0]} {kb[1]}={KNOBS_V2[i][2]+res_e['knobs'][i]['delta']:.2f}"
                                                       for i, kb in enumerate(KNOBS_V2)))

    out = dict(human_embryo={k: round(emb[k], 3) for k in KEYS}, human_adult={k: round(adult[k], 3) for k in KEYS},
               species=dict(match=res_e["match_final"], final={k: round(res_e["final"][k], 3) for k in KEYS},
                            percent_off=perr_e, mean_percent_off=round(mean_e, 1)),
               maturation=dict(match=res_a["match_final"], final={k: round(res_a["final"][k], 3) for k in KEYS},
                               percent_off=perr_a, mean_percent_off=round(mean_a, 1)),
               v1_mean_species=16.3, v1_mean_adult=16.8,
               knobs_embryo={KNOBS_V2[i][1]: round(KNOBS_V2[i][2] + res_e["knobs"][i]["delta"], 3) for i in range(len(KNOBS_V2))})
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(Ve, Va, res_e, res_a, perr_e, perr_a, mean_e, mean_a)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _figure(Ve, Va, res_e, res_a, perr_e, perr_a, mean_e, mean_a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    emb_fit = build_body_plan_v2({KNOBS_V2[i][1]: KNOBS_V2[i][2] + res_e["knobs"][i]["delta"] for i in range(len(KNOBS_V2))})
    adult_fit = build_body_plan_v2({KNOBS_V2[i][1]: KNOBS_V2[i][2] + res_a["knobs"][i]["delta"] for i in range(len(KNOBS_V2))})
    Yhe, Yfe, Yad, Yfa = _pca2(Ve), _pca2(emb_fit), _pca2(Va), _pca2(adult_fit)
    fig, ax = plt.subplots(1, 4, figsize=(16.5, 4.6), facecolor="white")
    ax[0].scatter(Yhe[:, 0], Yhe[:, 1], s=3, color="#2a6fb0", alpha=0.45)
    ax[0].set_title(f"human EMBRYO (HESTA)\nv2 species fit {res_e['match_final']:.2f}"); ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[1].scatter(Yfe[:, 0], Yfe[:, 1], s=3, color="#3aa869", alpha=0.45)
    ax[1].set_title("v2 builder fit\n(segmented axis)"); ax[1].set_aspect("equal"); ax[1].axis("off")
    ax[2].scatter(Yfa[:, 0], Yfa[:, 1], s=3, color="#8a6fb0", alpha=0.45)
    ax[2].set_title(f"v2 fit to ADULT\n(articulated limbs, {res_a['match_final']:.2f})"); ax[2].set_aspect("equal"); ax[2].axis("off")
    xk = np.arange(len(KEYS)); w = 0.38
    ax[3].bar(xk - w/2, [perr_e[k] for k in KEYS], w, label=f"embryo (mean {mean_e:.0f}%)", color="#3aa869")
    ax[3].bar(xk + w/2, [perr_a[k] for k in KEYS], w, label=f"adult (mean {mean_a:.0f}%)", color="#8a6fb0")
    ax[3].axhline(10, ls="--", color="#444", lw=1); ax[3].text(0, 11, "10%", fontsize=8)
    ax[3].set_xticks(xk); ax[3].set_xticklabels(KEYS, rotation=55, ha="right", fontsize=7)
    ax[3].set_ylabel("% off"); ax[3].legend(fontsize=7.5); ax[3].set_title("v2 distance to human\n(vs v1 ~16-17%)")
    fig.suptitle("Richer body-plan generator v2 (segmented axis + articulated limbs): refit stage-matched to the "
                 "human embryo (species adapter) and the adult (maturation), to close the v1 ~16%% gap toward 10%%.",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG, dpi=135, facecolor="white")


if __name__ == "__main__":
    run()
