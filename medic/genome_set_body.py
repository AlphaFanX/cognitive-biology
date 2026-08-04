"""The GENOME-SET body -- knobs set from GWAS/anthropometric proportions, NOT free-fit (Miles 2026-07-26).

Every fit so far FREE-fit the knobs to the target. The real test the surface metric named is genome -> knobs
-> surface: set the proportion knobs from the population values the GWAS-covered anthropometric traits give,
build, and MEASURE the surface distance to the real human -- no surface optimisation. That distance is the
honest genome->form validation.

THE VITRUVIAN QUESTION (Miles): does GWAS resolve the canon of proportions? The catalog scan says the GROSS
canon is covered and the FINE canon is not:
   total height 7466 loci | waist-hip/girth 2338 | leg/limb length 674 | head circ/ICV 112 | face 327
   sitting-height ratio (body-proportion) ~1231  --  ALL usable to set the major ratios;
   BUT arm span 0 loci, foot length 0 loci, finger/digit ratio 5, shoulder 37  --  the fine Vitruvian ratios
   (arm-span=height, foot=1/6 height, hand/finger segments) are essentially ABSENT.
Crucially the absent ones are heritable (arm span h^2 ~0.8) but not PHENOTYPED at biobank scale (arm span is
redundant with height, foot is not imaged) -- so the Vitruvian gap is a MEASUREMENT gap, not a biology gap,
and it is exactly the boundary of the computable-human spec: GWAS specifies what has been measured x varied x
localised. Here the GWAS-covered proportions set the knobs; the arm-span/foot canon is borrowed from
anthropometry and FLAGGED as the phenotyping gap.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.genome_set_body
Out:  data/organ_cascade/genome_set_body.{json,png}
"""
import os, json
import numpy as np
from medic.body_builder_v3 import _centerline, _seg_limb
from medic.surface_metric import pca_align, chamfer

HERE = os.path.dirname(__file__)
MH = os.path.join(HERE, "..", "data", "bodybase", "makehuman_decimated.npz")
OUT = os.path.join(HERE, "..", "data", "organ_cascade", "genome_set_body.json")
FIG = os.path.join(HERE, "..", "data", "organ_cascade", "genome_set_body.png")

# population / canon proportions and whether GWAS resolves them (loci) or it is a phenotyping gap (canon)
CANON = {
    "leg_fraction":   (0.48, "leg/limb length + body proportion", 674 + 1231),   # leg length / height
    "armspan_ratio":  (1.00, "arm span = height (CANON; not phenotyped)", 0),      # arm span / height -- GAP
    "girth_ratio":    (0.13, "waist-hip / body-mass proportion", 2338),           # torso radius / height
    "head_fraction":  (0.13, "head circumference / intracranial volume", 112),    # head / height
}


def build_labeled(p, n=2400, seed=5):
    """v3 geometry, straight (adult) axis, returning points + part labels (0 trunk,1 arm,2 leg,3 head)."""
    rng = np.random.RandomState(seed)
    L, w_ml, w_dv, head, taper = p["L"], p["w_ml"], p["w_dv"], p["head"], p["taper"]
    arm_len, leg_len = p["arm_len"], p["leg_len"]
    nt = int(n * 0.55); t = rng.rand(nt)
    head_bulge = 1 + (head - 1) * np.exp(-((t - 0.10) / 0.10) ** 2)
    taper_prof = 1 - taper * np.clip((t - 0.5) / 0.5, 0, 1)
    rad = head_bulge * taper_prof
    a = rng.rand(nt) * 2 * np.pi; rr = np.sqrt(rng.rand(nt))
    c_ml = rr * np.cos(a) * w_ml * rad; c_dv = rr * np.sin(a) * w_dv * rad
    cx, cy, phi = _centerline(t, L, 0.0, 0.0)
    trunk = np.stack([cx + c_dv * (-np.sin(phi)), cy + c_dv * np.cos(phi), c_ml], 1)
    lab = [np.where(t < 0.16, 3, 0)]                      # anterior 16% = head
    pts = [trunk]
    cxx, cyy, pp = _centerline(np.array([0.20, 0.82]), L, 0.0, 0.0)
    nlimb = max(24, int(n * 0.11))
    if arm_len > 0.02:
        base = np.array([cxx[0], cyy[0], 0.0])
        for side in (+1, -1):
            arm = _seg_limb(base + np.array([0, 0, side * w_ml]), np.array([0.12, 0.0, side * 1.0]),
                            arm_len, rng, nlimb, flex=np.array([0.15, 0.25, 0.0]))
            pts.append(arm); lab.append(np.full(len(arm), 1))
    if leg_len > 0.02:
        base = np.array([cxx[1], cyy[1], 0.0]); tan = np.array([np.cos(pp[1]), np.sin(pp[1]), 0.0])
        for side in (+1, -1):
            leg = _seg_limb(base + np.array([0, 0, side * w_ml * 0.5]), tan + np.array([0, 0, side * 0.18]),
                            leg_len, rng, nlimb, flex=np.array([0.0, -0.2, 0.0]))
            pts.append(leg); lab.append(np.full(len(leg), 2))
    return np.vstack(pts), np.concatenate(lab)


def measure(pts, lab):
    """canonical proportions from the labelled cloud (PCA long axis = height, second axis = span)."""
    Y = pca_align(pts)
    ax_h = 0                                              # long axis after alignment
    height = np.ptp(Y[:, ax_h])
    span = np.ptp(Y[:, 1])                                # arm span (widest transverse)
    legs = Y[lab == 2]; leg_frac = (np.ptp(legs[:, ax_h]) / height) if len(legs) else 0.0
    head = Y[lab == 3]; head_frac = (np.ptp(head[:, ax_h]) / height) if len(head) else 0.0
    trunk = Y[lab == 0]
    girth = (np.abs(trunk[:, 2]).mean() * 2 / height) if len(trunk) else 0.0
    return dict(leg_fraction=leg_frac, armspan_ratio=span / height, head_fraction=head_frac, girth_ratio=girth)


def fit_proportions(targets, n_iter=60):
    """set the knobs to REPRODUCE the genome/canon proportions (a proportion-solve, NOT a surface fit)."""
    knob = dict(L=3.0, w_ml=0.35, w_dv=0.28, head=1.4, taper=0.5, arm_len=0.9, leg_len=0.9)
    lo = dict(L=1.5, w_ml=0.15, w_dv=0.12, head=1.0, taper=0.1, arm_len=0.0, leg_len=0.0)
    hi = dict(L=5.0, w_ml=0.9, w_dv=0.7, head=2.4, taper=0.9, arm_len=2.2, leg_len=2.2)
    keys = list(knob)
    def err(k):
        m = measure(*build_labeled(k))
        return sum((m[t] - targets[t][0]) ** 2 / (targets[t][0] ** 2 + 1e-9) for t in targets)
    step = {k: 0.3 * (hi[k] - lo[k]) for k in keys}
    e = err(knob)
    for _ in range(n_iter):
        improved = False
        for k in keys:
            for s in (+1, -1):
                cand = dict(knob); cand[k] = float(np.clip(knob[k] + s * step[k], lo[k], hi[k]))
                ec = err(cand)
                if ec < e:
                    e, knob = ec, cand; improved = True
        if not improved:
            for k in keys:
                step[k] *= 0.5
    return knob, measure(*build_labeled(knob))


def run():
    rng = np.random.default_rng(0)
    za = np.load(MH, allow_pickle=True); Va = np.asarray(za["V"], float); Va = Va[rng.choice(len(Va), 3000, replace=False)]
    Aa = pca_align(Va)

    knob, got = fit_proportions(CANON)
    body, lab = build_labeled(knob)
    ch = chamfer(pca_align(body), Aa, pre_aligned=True)
    # reference: the free surface-fit distance from surface_metric
    try:
        free = json.load(open(os.path.join(HERE, "..", "data", "organ_cascade", "surface_metric.json")))["adult_surface_fit"]["chamfer_pct"]
    except Exception:
        free = None

    print("GENOME-SET body -- knobs set from GWAS/canon proportions, NOT free-fit\n")
    print(f"{'proportion':16s} {'target':>8s} {'achieved':>9s}   source (GWAS loci)")
    for t, (val, src, n) in CANON.items():
        print(f"  {t:16s} {val:8.2f} {got[t]:9.2f}   {src}" + (f"  [{n} loci]" if n else "  [GAP: 0 loci]"))
    print(f"\nGENOME-SET surface distance to MakeHuman = {ch:.1f}%  "
          f"(free surface-fit was {free}%)" if free else f"\nGENOME-SET surface distance = {ch:.1f}%")
    print("So the body proportioned by the genome-covered ratios (arm-span borrowed from canon, the phenotyping "
          "gap) sits within this distance of the real human WITHOUT fitting the surface = the genome->form readout.")

    out = dict(genome_set_chamfer_pct=round(ch, 2), free_fit_chamfer_pct=free,
               proportions_target={t: CANON[t][0] for t in CANON}, proportions_achieved={t: round(got[t], 3) for t in CANON},
               gwas_coverage={t: dict(source=CANON[t][1], loci=CANON[t][2]) for t in CANON},
               knobs=knob,
               vitruvian=dict(covered=["total height 7466", "waist-hip 2338", "leg/limb 674", "head/ICV 112", "face 327", "body-proportion ~1231"],
                              gaps=["arm span 0", "foot length 0", "finger/digit ratio 5", "shoulder 37"],
                              note="gaps are heritable but not phenotyped at scale = measurement gap, the boundary of the computable-human spec"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(Aa, body, lab, ch, free, got)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _figure(Aa, body, lab, ch, free, got):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Yb = pca_align(body)
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor="white")
    ax[0].scatter(Aa[:, 0], Aa[:, 1], s=3, color="#8a6fb0", alpha=0.45)
    ax[0].set_title("MakeHuman (target)"); ax[0].set_aspect("equal"); ax[0].axis("off")
    cols = {0: "#3aa869", 1: "#d64545", 2: "#2a6fb0", 3: "#c98a2b"}
    for L in (0, 3, 2, 1):
        m = lab == L
        ax[1].scatter(Yb[m, 0], Yb[m, 1], s=3, color=cols[L], alpha=0.5)
    ax[1].set_title(f"GENOME-SET body (not fitted)\nsurface {ch:.0f}% off" + (f" (free-fit {free:.0f}%)" if free else ""))
    ax[1].set_aspect("equal"); ax[1].axis("off")
    names = list(CANON); x = np.arange(len(names))
    ax[2].bar(x - 0.2, [CANON[n][0] for n in names], 0.4, label="genome/canon target", color="#888")
    ax[2].bar(x + 0.2, [got[n] for n in names], 0.4, label="achieved", color="#3aa869")
    for i, n in enumerate(names):
        if CANON[n][2] == 0:
            ax[2].text(i, max(CANON[n][0], got[n]) + 0.03, "GAP", ha="center", fontsize=7, color="#d64545")
    ax[2].set_xticks(x); ax[2].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7.5)
    ax[2].legend(fontsize=8); ax[2].set_title("Proportions set from GWAS\n(arm span = the Vitruvian phenotyping GAP)")
    fig.suptitle("The genome-set body: knobs set from the GWAS-covered anthropometric proportions (height, leg, "
                 "girth, head), arm-span borrowed from the Vitruvian canon (0 GWAS loci = a phenotyping gap), then "
                 "the surface distance to the real human MEASURED, not fitted -- the genome->form readout.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG, dpi=135, facecolor="white")


if __name__ == "__main__":
    run()
