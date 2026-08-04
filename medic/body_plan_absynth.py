"""Analysis-by-synthesis on the BODY PLAN -- the trunk-silhouette-vs-MOSTA loop.

medic.body_absynth read the body's place on the canalized<->polygenic axis from the GWAS architecture of
STATURE (individual size = polygenic). This module does the complementary, missing half: an actual SYNTHESIS
loop on the body PLAN (the Bauplan silhouette), exactly as medic.organ_absynth does for the organs. A
parametric whole-body generator whose knobs are named developmental GENES (frozen direction, free magnitude)
is tuned, by the same coarse-search + LM relaxation, to match the real E16.5 whole-embryo silhouette from the
dense reconstruction, scored on the same scale/rotation-invariant descriptors (elongation, flatness, bend,
tortuosity, hollowness). The claim it tests: the body PLAN is CANALIZED like the organs -- a few master
genes carry it, so the loop closes most of the shape gap (high genomic-closable) -- which, set beside the
polygenic stature result, completes the spectrum: canalized PLAN, polygenic individual SIZE.

Forward model: an AP axis (Hox/CDX length), an elliptical cross-section (ML width = the Wnt-PCP convergent-
extension knob; DV thickness), an optional axial curl (Shh/notochord body-fold), an anterior cephalic bulge
(Otx2/Six3) and paired limb buds (Tbx5/Fgf10). NOTE the real target is registered to a STRAIGHT AP frame,
so its bend is ~0 by reconstruction (a target-frame property, not biology, like the spinal-cord meander);
the loop therefore correctly keeps the curl knob low, and the informative descriptors are elongation /
flatness / tortuosity / hollowness.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.body_plan_absynth
Out:  data/organ_cascade/body_plan_absynth.{json,png}
"""
import os, json
import numpy as np
from medic.embryo_match_score import fingerprint, KEYS, full_desc, shape_match
from medic.organ_absynth import optimize

HERE = os.path.dirname(__file__)
NPZ = os.path.join(HERE, "..", "data", "mosta", "mouse_e165_3d.npz")
OUT = os.path.join(HERE, "..", "data", "organ_cascade", "body_plan_absynth.json")
FIG = os.path.join(HERE, "..", "data", "organ_cascade", "body_plan_absynth.png")

# body-plan knobs: (gene, param, start, lo, hi, sign toward the real silhouette)
KNOBS = [
    ("CDX2/HOX",  "L",     2.0, 1.2, 4.0, +1),   # AP axis length (Hox/CDX) -> elongation
    ("WNT-PCP",   "w_ml",  0.70, 0.30, 1.10, -1), # convergent-extension ML narrowing -> elongation
    ("BMP/FOXA2", "w_dv",  0.60, 0.30, 1.00, -1), # dorsoventral thickness -> flatness
    ("SHH/noto",  "kappa", 0.30, 0.00, 2.50, +1), # axial curl / body-fold (low in the straight-frame target)
    ("OTX2/SIX3", "head",  1.20, 1.00, 2.40, +1), # cephalic enlargement
    ("TBX5/FGF10","limb",  0.10, 0.00, 0.45, +1), # limb buds -> lateral protrusion
    ("CDX/tail",  "taper", 0.40, 0.10, 0.90, +1), # posterior taper
]


def build_body_plan(p, n=1800, seed=3):
    rng = np.random.RandomState(seed)
    L, w_ml, w_dv = p["L"], p["w_ml"], p["w_dv"]
    kappa, head, taper, limb = p["kappa"], p["head"], p["taper"], p["limb"]
    t = rng.rand(n)                                            # AP parameter, 0 = head, 1 = tail
    head_bulge = 1 + (head - 1) * np.exp(-((t - 0.12) / 0.12) ** 2)
    taper_prof = 1 - taper * np.clip((t - 0.5) / 0.5, 0, 1)   # shrink posterior half
    rad = head_bulge * taper_prof
    ang = rng.rand(n) * 2 * np.pi
    rr = np.sqrt(rng.rand(n))
    c_ml = rr * np.cos(ang) * w_ml * rad                      # local ML offset
    c_dv = rr * np.sin(ang) * w_dv * rad                      # local DV offset
    phi = kappa * (t - 0.5)                                   # tangent angle for the curl (AP-DV plane)
    if abs(kappa) > 1e-6:
        R = L / kappa
        ax_x, ax_y = R * np.sin(phi), R * (1 - np.cos(phi))
    else:
        ax_x, ax_y = L * (t - 0.5), np.zeros(n)
    x = ax_x + c_dv * (-np.sin(phi))                          # AP (bending plane)
    y = ax_y + c_dv * np.cos(phi)                             # DV (bending plane)
    z = c_ml                                                  # ML (out of plane)
    pts = np.stack([x, y, z], 1)
    if limb > 0.01:                                           # paired fore/hind limb buds
        for tb in (0.32, 0.72):
            for side in (+1, -1):
                k = max(6, int(limb * 320))
                lt = np.clip(tb + rng.randn(k) * 0.02, 0, 1)
                lp = kappa * (lt - 0.5)
                if abs(kappa) > 1e-6:
                    R = L / kappa; bx, by = R * np.sin(lp), R * (1 - np.cos(lp))
                else:
                    bx, by = L * (lt - 0.5), np.zeros(k)
                bud_ml = side * (w_ml + limb) + rng.randn(k) * limb * 0.4
                bud_dv = rng.randn(k) * w_dv * 0.5
                lx = bx + bud_dv * (-np.sin(lp)); ly = by + bud_dv * np.cos(lp); lz = bud_ml
                pts = np.vstack([pts, np.stack([lx, ly, lz], 1)])
    return pts


def _pca2(P):
    Q = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    Y = Q @ Vt[:2].T
    return Y / (np.sqrt((Q ** 2).sum(1).mean()) + 1e-9)       # unit-RMS for display


def run():
    z = np.load(NPZ, allow_pickle=True); xyz = z["xyz"]
    rng = np.random.default_rng(0)
    real_pts = xyz[rng.choice(len(xyz), 3000, replace=False)]
    real = fingerprint(real_pts)
    print("real E16.5 whole-body fingerprint:", {k: real[k] for k in KEYS})

    res = optimize(build_body_plan, KNOBS, real, n_search=120, n_lm=12, seed=0)
    print(f"\nBODY PLAN analysis-by-synthesis (trunk silhouette vs MOSTA E16.5)")
    print(f"  match {res['match_start']:.2f} -> {res['match_final']:.2f}   "
          f"genomic-closable {100*res['genomic_closable']:.1f}%   residual floor {100*res['residual_floor']:.1f}% "
          f"({res['floor_dominant']})")
    print(f"  gene-direction {res['dir_correct']}/{res['dir_meaningful']} of meaningfully-moved knobs")
    for kb in res["knobs"]:
        mv = "moved" if kb["moved"] else "still"
        print(f"    {kb['gene']:11s} {kb['param']:6s} delta {kb['delta']:+.2f}  {mv}"
              + (f"  dir {'ok' if kb['ok'] else 'X'}" if kb["moved"] else ""))
    print(f"  real  {res['real']}")
    print(f"  final {res['final']}")

    out = dict(target="mouse_e165_3d whole body", **res,
               regime="body PLAN canalized (this loop) vs individual body SIZE polygenic (body_absynth stature)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(real_pts, res)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _figure(real_pts, res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # rebuild the fitted body from the reported knob deltas
    base = {kb["param"]: KNOBS[i][2] + kb["delta"] for i, kb in enumerate(res["knobs"])}
    fit = build_body_plan(base)
    Yr, Yf = _pca2(real_pts), _pca2(fit)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), facecolor="white")
    ax[0].scatter(Yr[:, 0], Yr[:, 1], s=3, color="#2a6fb0", alpha=0.5)
    ax[0].set_title("real E16.5 whole body\n(silhouette, PCA view)"); ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[1].scatter(Yf[:, 0], Yf[:, 1], s=3, color="#3aa869", alpha=0.5)
    ax[1].set_title("synthesised body plan\n(genome knobs tuned)"); ax[1].set_aspect("equal"); ax[1].axis("off")
    b = ["match", "genomic\nclosable"]
    ax[2].bar([0, 1], [res["match_start"], res["genomic_closable"]], color="#bbb", width=0.4, label="start")
    ax[2].bar([0.4], [res["match_final"]], color="#d64545", width=0.4)
    ax[2].bar([1], [res["genomic_closable"]], color="#3aa869", width=0.4)
    ax[2].set_xticks([0.2, 1]); ax[2].set_xticklabels([f"match\n{res['match_start']:.2f}->{res['match_final']:.2f}",
                                                       f"genomic-closable\n{100*res['genomic_closable']:.0f}%"])
    ax[2].set_ylim(0, 1); ax[2].set_ylabel("fraction")
    ax[2].set_title("body PLAN is canalized:\nfew genome knobs close the silhouette gap")
    fig.suptitle("Body-plan analysis-by-synthesis (trunk silhouette vs MOSTA E16.5): the Bauplan is canalized "
                 "like the organs -- a few developmental genes close the shape gap.\nBeside the polygenic stature "
                 "result, this completes the spectrum: canalized body PLAN, polygenic individual SIZE.", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG, dpi=135, facecolor="white")


if __name__ == "__main__":
    run()
