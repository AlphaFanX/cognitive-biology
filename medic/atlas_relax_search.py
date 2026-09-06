"""
atlas_relax_search.py -- make the matured cloud RELAX INTO the Phase-D Hill-organ atlas, by KNOB SEARCH.

The Phase-D atlas (`menagerie.reference_genome("human_male")` -> bones + Hill-muscles + viscera) is the
TARGET MORPHOLOGY: `skeleton.py` frames each part's boundary capsule as "the attractor target the
cell-NCA relaxes into". Until now the movie just SWAPPED the model cloud for that separate decode at the
reveal, so the mechanism-placed organs never reached the visible adult.

This wires the two together the way the whole project does everything else -- a von Dassow-Odell
DERIVATIVE-FREE KNOB SEARCH (cf. medic.develop_search, medic.differentiation_search), NOT a hand-coded
warp. The searchable knobs are the maturation allometry parameters of `human_movie.mature_cloud`
(trunk elongation, DV/ML girth, head proportion, limb extension). The OBJECTIVE drives the matured
cloud's own form onto the atlas target:

  * SILHOUETTE   -- the cloud's per-AP-band width profile matches the atlas's (fill the body capsule);
  * PROPORTION   -- ~7 heads tall (the canonical adult the atlas encodes);
  * ORGAN ADDRESS-- each shared viscus (eye/heart/lung/liver/pancreas/kidney) lands on the atlas's AP/DV
                    address (its residual is the embryo<->atlas CONSISTENCY the search can't fix with
                    maturation knobs alone -- reported at the end).

...all under a MECHANISM PENALTY (the 5 placement mechanisms + integrin continuity + CE from
medic.adult_persistence_audit._metrics) so the search can NEVER buy a better atlas fit by breaking a
mechanism. Best knobs cache to data/organ_cascade/atlas_relax_search.json and are picked up by
human_movie.MATURE_SEARCHED (like LIMB_SEARCHED / FATE_SEARCHED).

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.atlas_relax_search
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from medic.unified_embryo import FIDX
from medic.human_movie import (mature_cloud, grow_limbs, _limb_grow_model, _long_axis_len, LIMB,
                               MATURE_DEFAULTS, build_human_atlas, ATLAS_BASE, ATLAS_IDX)
from medic.adult_persistence_audit import build_base, _metrics, LIMB_IDS

OUT = Path("data/organ_cascade/atlas_relax_search")
HEAD_FATES = ("Forebrain", "Eye", "Midbrain", "Hindbrain")
HT_TARGET = 7.5                                          # canonical adult heads-tall (Vitruvian/anthropometric)
# The Vitruvian CANON, as fractions of standing height H -- the explicit ideal-man target (Da Vinci /
# anthropometric norms). leg = hip-to-heel ~1/2 H; arm = shoulder-to-fingertip ~0.44 H (hangs to mid-thigh);
# shoulder (biacromial) width ~1/4 H. These are ABSOLUTE ratios (fraction of height), so unlike the old
# self-referential limb-span metric (normalised by body length, which GROWS with the legs -> unoptimisable)
# the search can actually drive the limbs to the canonical length.
CANON = dict(leg=0.50, arm=0.44, shoulder=0.25)
CHAMBERS = {"Heart": ["Heart", "Atrium", "Ventricle", "Outflow"]}
SHARED = {"Eye": "eye", "Heart": "heart", "Lung": "lung",   # cloud fate -> atlas organ class
          "Liver": "liver", "Pancreas": "pancreas", "Kidney": "kidney"}
PAIRED = ["Eye", "Otic", "Lung", "Kidney"]              # organs the lateral-inhibition penalty guards

# GUARDRAILS (2026-08-09): the free re-run collapsed to a squat FLAT slab (heads-tall 5.8, trunk_e 0.12, dv_girth
# -0.55). Enforce the memory's known-good floors: dv_girth>=0.40 (else the body goes flat frontal), trunk_e>=0.40
# (else the trunk is too short -> low heads-tall). Keeps the search off the degenerate optima.
RANGES = dict(trunk_e=(0.40, 0.90), dv_girth=(0.40, 0.85), ml_girth=(0.00, 0.90),
              head_ht0=(3.0, 6.0), head_ht1=(1.0, 5.0), limb_ext=(2.0, 2.8), leg_ext=(1.0, 3.2),
              shoulder_w=(1.0, 3.5), waist_w=(0.5, 1.0), hip_w=(1.0, 1.6))  # regional taper (Vitruvian) + split legs
# NOTE: widening shoulder_w (3.5->5.0) did NOT raise the shoulder canon (stayed 0.23 -- it is MECHANISM-limited
# by the girdle acromial mass, not range-limited); reverted. The last ~0.02 to canon 0.25 is closed by the
# girdle `span` (a wider acromion), not a knob.
# NOTE: dv_girth FLOORED at 0.40 (its default) -- the ratio-tie over-corrected and the search zeroed DV depth,
# making the body a flat frontal SLAB (the flat side profile seen in the frames). shoulder_w (2.4->3.5) +
# ml_girth (0.60->0.90) upper bounds widened -- at 240k both pinned at their old
# ceilings (the search wanted more upper-body width than allowed, so the canon shoulder ratio undershot 0.25).
# W_SHAPE = trunk ML+DV silhouette (ratio-preserving), W_HT = heads-tall, W_LIMB = limb length vs the
# atlas's own limbs, W_ORG = shared-organ address. W_LIMB gives the search a REASON to extend the limbs
# (the old ML-only profile only PENALISED extension via splay -> stubby); W_SHAPE's shared ML scale ties
# DV depth to ML width so dv_girth is constrained (it used to be free, so the search maxed it).
W_SHAPE, W_HT, W_LIMB, W_ORG, W_CANON, W_DV = 1.0, 1.2, 0.2, 0.2, 2.0, 4.0   # W_CANON dominant = drive the Vitruvian
W_BODY = 3.0                                             # NEW: the actual body-proportion silhouette metric (posture_silhouette, ~85%) as a dominant objective
ANTHRO_DV = 0.15                                                   # a man's dorsoventral body depth = 0.15 of stature
# canon (leg/arm/shoulder as fractions of height); W_LIMB (self-referential atlas span) demoted to a hint.


# ------------------------------------------------------------------ shared descriptors
def _shape_profiles(P, trunk):
    """Per-AP-band ML and DV half-extents of the TRUNK (limbs excluded), 24 bins, BOTH normalised by the
    SAME ML scale so the ML:DV ratio is preserved -- a deep trunk (high dv_girth) shows as a tall DV
    profile relative to ML, so matching the atlas constrains girth in both axes, not just width."""
    Pm = P[trunk]
    apf = (Pm[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)   # bin by FULL-body AP so bins align
    apbin = np.clip((apf * 24).astype(int), 0, 23)
    wml = np.zeros(24); wdv = np.zeros(24)
    for k in range(24):
        m = apbin == k
        if m.sum() > 4:
            wml[k] = np.percentile(np.abs(Pm[m, 2]), 85)
            wdv[k] = np.percentile(np.abs(Pm[m, 1]), 85)
    s = wml.max() + 1e-9
    return wml / s, wdv / s


def _limb_span(P, limb):
    """Arm and leg reach (max radial spread from the limb's own centroid, normalised by body AP length),
    split by AP. Direction-agnostic, so a splayed (T-pose) cloud arm and an arms-down atlas arm are
    compared on LENGTH -- the search extends the cloud limb to the atlas limb length regardless of pose."""
    L = np.ptp(P[:, 0]) + 1e-9
    apf = (P[:, 0] - P[:, 0].min()) / L
    def span(m):
        if m.sum() < 4:
            return 0.0
        return float(np.linalg.norm(P[m] - P[m].mean(0), axis=1).max() / L)
    return span(limb & (apf >= 0.5)), span(limb & (apf < 0.5))


def _atlas_limb_mask(P):
    """Spatially flag the atlas's limb points (arms lateral at shoulder-hand AP; legs low AP) -- the atlas
    fate is by class, not 'arm bone', so limbs are identified geometrically (as in _fit_atlas_to_model)."""
    apf = (P[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)
    hw = np.percentile(np.abs(P[:, 2]), 75) + 1e-9
    arm = (np.abs(P[:, 2]) > hw) & (apf > 0.42) & (apf < 0.90)
    leg = apf < 0.34
    return arm | leg


def _heads_tall(P, F):
    hm = np.isin(F, [FIDX[n] for n in HEAD_FATES if n in FIDX])
    if hm.sum() < 8:
        return HT_TARGET
    return float(np.ptp(P[:, 0]) / (np.ptp(P[hm, 0]) + 1e-9))


def _canon_ratios(P, F):
    """The Vitruvian ratios of the matured figure, as fractions of standing height H (AP extent):
    leg = hip->heel, arm = shoulder->fingertip, shoulder = biacromial ML width. ABSOLUTE (over H), so the
    search has a real gradient on limb LENGTH (the old span/body metric normalised away exactly that)."""
    x = P[:, 0]; H = np.ptp(x) + 1e-9
    apf = (x - x.min()) / H
    limb = np.isin(F, LIMB_IDS)
    arm = limb & (apf >= 0.5); leg = limb & (apf < 0.5); body = ~limb
    def seg(m):                                    # AP length of a limb segment as a fraction of height
        return float((np.percentile(x[m], 92) - x[m].min()) / H) if m.sum() >= 8 else 0.0
    sb = body & (apf > 0.74) & (apf < 0.90)        # upper-trunk shoulder band
    shoulder = float(2.0 * np.percentile(np.abs(P[sb, 2]), 90) / H) if sb.sum() >= 8 else 0.0
    return dict(leg=seg(leg), arm=seg(arm), shoulder=shoulder)


def _canon_loss(P, F):
    r = _canon_ratios(P, F)
    return abs(r["leg"] - CANON["leg"]) + abs(r["arm"] - CANON["arm"]) + abs(r["shoulder"] - CANON["shoulder"])


def _organ_addr(P, idx):
    """(ap-fraction, dv-fraction) of an organ's cells -- head at +x (apf~1)."""
    ax, dv = P[:, 0], P[:, 1]
    apf = (ax - ax.min()) / (np.ptp(ax) + 1e-9)
    dvf = (dv - dv.min()) / (np.ptp(dv) + 1e-9)
    return float(apf[idx].mean()), float(dvf[idx].mean())


# ANTHROPOMETRIC SILHOUETTE: the trunk HALF-width (ML breadth) and HALF-depth (DV) of a real adult male as
# fractions of stature, along the body (apf 0 = feet, 1 = crown) -- WITH a neck, broad shoulders, a pinched waist,
# and true hips, which the coarse Hill-organ atlas lacks. Control points (apf, ml_half/H, dv_half/H); the trunk
# begins at the pelvis (~0.45), below which the body is legs (limb, excluded from the trunk profile).
ANTHRO_SIL = [
    (0.00, 0.000, 0.000), (0.44, 0.070, 0.055), (0.50, 0.095, 0.068),   # legs -> pelvis (hips widest)
    (0.60, 0.070, 0.055), (0.68, 0.087, 0.075), (0.78, 0.122, 0.065),   # waist pinch -> chest -> shoulders (widest ML)
    (0.84, 0.035, 0.045), (0.90, 0.050, 0.055), (0.96, 0.045, 0.055), (1.00, 0.025, 0.035),  # NECK -> face -> crown
]


# THE CANONICAL HUMAN silhouette, MEASURED from the BodyParts3D full-body skin (FMA7163) -- 24-bin ML/DV
# half-width profiles (feet->head), normalised by max ML. Replaces the hand-guessed ANTHRO_SIL, which was
# wrong-shaped (its ML peaked near the head, feet ~0; a real human peaks at the SHOULDERS, bins 12-14, with a
# leg/hip taper). This makes the knob-search objective the REAL human form, not a synthetic proxy.
CANON_HUMAN_ML = [0.21, 0.20, 0.17, 0.37, 0.69, 0.72, 0.62, 0.70, 0.82, 0.82, 0.80, 0.85,
                  1.00, 0.98, 0.94, 0.51, 0.45, 0.42, 0.43, 0.44, 0.42, 0.39, 0.35, 0.43]
CANON_HUMAN_DV = [0.26, 0.29, 0.31, 0.29, 0.36, 0.38, 0.39, 0.39, 0.33, 0.32, 0.35, 0.47,
                  0.39, 0.24, 0.27, 0.32, 0.29, 0.29, 0.38, 0.43, 0.39, 0.36, 0.34, 0.37]


def _anthro_silhouette(nbin=24):
    """The CANONICAL HUMAN silhouette (measured from FMA7163) as (wml, wdv) half-width profiles, normalised by
    max ML (like _shape_profiles) -> the objective drives the cloud onto a REAL human's proportions."""
    ml = np.array(CANON_HUMAN_ML, float); dv = np.array(CANON_HUMAN_DV, float)
    if nbin != len(ml):
        src = (np.arange(len(ml)) + 0.5) / len(ml); centers = (np.arange(nbin) + 0.5) / nbin
        ml = np.interp(centers, src, ml); dv = np.interp(centers, src, dv)
    s = ml.max() + 1e-9
    return ml / s, dv / s


def atlas_target():
    """The objective's target descriptors (laid frame, head +x): trunk ML+DV shape from the ANTHROPOMETRIC
    silhouette (a real man, with neck/waist/hips), plus arm/leg length and shared-organ addresses from the atlas."""
    rng = np.random.default_rng(0)
    A_pos, A_fate, _counts = build_human_atlas(rng)          # laid, scaled to the adult long axis
    limb = _atlas_limb_mask(A_pos)
    wml_a, wdv_a = _anthro_silhouette()                      # trunk shape = the real anthropometric silhouette
    arm_a, leg_a = _limb_span(A_pos, limb)
    addr = {}
    for cloud_name, atlas_name in SHARED.items():
        idx = np.where(A_fate == ATLAS_BASE + ATLAS_IDX[atlas_name])[0]
        if len(idx) >= 3:
            addr[cloud_name] = _organ_addr(A_pos, idx)
    # the AP-address head now places each organ at its true adult (Gray's) axial level, not the coarse embryo-
    # atlas level; score the objective against that canonical AP (keep the atlas DV), so the organ term rewards
    # correct anatomy instead of penalising the registration for disagreeing with the blurry atlas.
    from medic.human_movie import ORGAN_AP_ADDRESS
    for cloud_name in list(addr):
        if cloud_name in ORGAN_AP_ADDRESS:
            addr[cloud_name] = (ORGAN_AP_ADDRESS[cloud_name], addr[cloud_name][1])
    # organ DV target: the MEASURED in-situ depth from BodyParts3D (medic.bp3d_insitu_dv), replacing the hand-
    # authored ORGAN_PLAN schematic that build_human_atlas compressed to ~0.5 for every viscus. Same whole-body
    # convention as _organ_addr on our model (spine dorsal = high fraction). The dv_spread head places the organs
    # there in build_base; scoring against the same measured target keeps the objective from fighting the mechanism.
    try:
        _meas = json.load(open("data/organ_cascade/bp3d_insitu_dv.json"))["organs"]
        for cloud_name in list(addr):
            if cloud_name in _meas:
                addr[cloud_name] = (addr[cloud_name][0], float(_meas[cloud_name]["dv_wholebody"]))
    except Exception:
        pass
    return dict(wml=wml_a, wdv=wdv_a, arm=arm_a, leg=leg_a, addr=addr)


# ------------------------------------------------------------------ maturation + objective
def mature_with(base, F, knobs, f=1.0):
    """The movie's model-native maturation at fraction f with a given knob set (mature_cloud + limb
    outgrowth + the length ramp) -- the adult cloud whose form the search is fitting to the atlas."""
    Q = mature_cloud(base, F, f, knobs)
    t = 0.55 + 0.45 * f
    Q = grow_limbs(Q, F == LIMB, _limb_grow_model(t, knobs["limb_ext"]),
                   _limb_grow_model(t, knobs.get("leg_ext", knobs["limb_ext"])), pose=f)  # match the movie adult (legs stand, arms Vitruvian)
    return Q * ((0.9 + (3.2 - 0.9) * f ** 1.2) / _long_axis_len(Q))


def _penalty(m, base_aspect, heads_tall=None):
    """Mechanism guard: the search cannot trade a mechanism for a better atlas fit."""
    pen = 0.0
    pen += 8.0 * max(0, m["n_components"] - 1)                     # integrin/ECM continuity (one continuum)
    for n in PAIRED:                                              # lateral inhibition (bilateral buds)
        if not m["organs"].get(n, {}).get("bilateral"):
            pen += 1.5
    pen += 6.0 * max(0.0, 0.80 * base_aspect - m["aspect"])       # convergent extension (axial elongation)
    # HEADS-TALL GUARDRAIL (2026-08-09): the soft W_HT term (1.2) is dwarfed by W_DV/W_BODY/W_CANON, so the search
    # let heads-tall leak UP to 8.5 (the RANGES floors only cap the squat-slab collapse, not a too-tall figure).
    # Enforce the canonical band as a NON-TRADEABLE guard here: free within 7.0-8.0, stiff outside (like the
    # continuity/bilaterality guards) so the silhouette fit cannot buy a giraffe-tall or dwarf body.
    if heads_tall is not None:
        pen += 6.0 * max(0.0, abs(heads_tall - HT_TARGET) - 0.5)
    return pen


def objective(base, F, knobs, tgt, base_aspect):
    Q = mature_with(base, F, knobs)
    m = _metrics(Q, F)
    trunk = ~np.isin(F, LIMB_IDS)
    wml_c, wdv_c = _shape_profiles(Q, trunk)
    arm_c, leg_c = _limb_span(Q, ~trunk)
    ht = _heads_tall(Q, F)
    s = W_SHAPE * float(np.mean(np.abs(wml_c - tgt["wml"])) + np.mean(np.abs(wdv_c - tgt["wdv"])))
    s += W_HT * abs(ht - HT_TARGET) / HT_TARGET
    s += W_LIMB * (abs(arm_c - tgt["arm"]) + abs(leg_c - tgt["leg"]))
    s += W_CANON * _canon_loss(Q, F)                              # the Vitruvian canon (leg/arm/shoulder over H)
    from medic.posture_silhouette import score as _body_score    # the reported body-proportion metric (arms excluded BY FATE)
    s += W_BODY * (1.0 - _body_score(Q, F)["iou"])               # directly drive the body proportion up
    # ANTHROPOMETRIC DV DEPTH: drive dv_girth so the CHEST band's dorsoventral depth = 0.145*stature (a real man).
    # Target the chest BAND, not the global max: the regional anthropometric envelope in mature_cloud caps the
    # head/neck hump separately, so targeting the max here would make dv_girth over-thin the chest.
    stature = np.ptp(Q[:, 0]) + 1e-9
    xf = (Q[:, 0] - Q[:, 0].min()) / stature
    chest = (xf >= 0.58) & (xf <= 0.80)
    if chest.sum() > 20:
        dv_depth = (np.percentile(Q[chest, 1], 95) - np.percentile(Q[chest, 1], 5)) / stature
        s += W_DV * abs(dv_depth - ANTHRO_DV)
    for name, (apa, dva) in tgt["addr"].items():
        o = m["organs"].get(name, {})
        if o.get("sprouted"):
            s += W_ORG * (abs(o["ap"] - apa) + abs(o["dv"] - dva))
    s += _penalty(m, base_aspect, heads_tall=ht)
    return float(s), m


# ------------------------------------------------------------------ search
def search(ne=None, n_coarse=40, cd_passes=3, grid=7, seed=0):
    base, F = build_base() if ne is None else build_base(ne)
    tgt = atlas_target()
    base_aspect = _metrics(mature_with(base, F, dict(MATURE_DEFAULTS), f=0.0), F)["aspect"]   # embryo axial aspect
    rng = np.random.default_rng(seed)

    def score(k):
        return objective(base, F, k, tgt, base_aspect)[0]

    best = dict(MATURE_DEFAULTS)
    best_s = score(best)
    default_s = best_s
    # WARM START from the cached best: the coarse random sweep is stochastic, so a re-run can otherwise land in
    # a WORSE local optimum and overwrite a good cache (this happened once). Seed from the cache so a re-run is
    # monotone -- it can only improve on, never regress below, the previous best.
    cache = Path(str(OUT) + ".json")
    if cache.exists():
        try:
            ck = json.load(open(cache)).get("best_knobs")
            if ck:
                warm = {**MATURE_DEFAULTS, **{k: ck[k] for k in RANGES if k in ck}}
                ws = score(warm)
                if ws < best_s:
                    best, best_s = warm, ws
        except Exception:
            pass
    # coarse random sweep
    for _ in range(n_coarse):
        cand = {k: float(rng.uniform(*RANGES[k])) for k in RANGES}
        s = score(cand)
        if s < best_s:
            best, best_s = cand, s
    # coordinate descent
    for _ in range(cd_passes):
        for k in RANGES:
            lo, hi = RANGES[k]
            for v in np.linspace(lo, hi, grid):
                cand = dict(best); cand[k] = float(v)
                s = score(cand)
                if s < best_s:
                    best, best_s = cand, s
    return dict(base=base, F=F, tgt=tgt, base_aspect=base_aspect,
                best=best, best_score=round(best_s, 4), default_score=round(default_s, 4))


def _consistency(res):
    """Embryo<->atlas organ-address agreement AFTER the searched maturation: what the maturation knobs
    could bring onto the atlas, and the residual (an organ far off is an EMBRYO placement disagreement,
    not something maturation can fix -- e.g. the pancreas mismatch)."""
    base, F = res["base"], res["F"]
    m = _metrics(mature_with(base, F, res["best"]), F)
    rows = []
    for name, (apa, dva) in res["tgt"]["addr"].items():
        o = m["organs"].get(name, {})
        if not o.get("sprouted"):
            rows.append(dict(organ=name, ok=False, note="not sprouted")); continue
        dap, ddv = abs(o["ap"] - apa), abs(o["dv"] - dva)
        rows.append(dict(organ=name, ap_model=round(o["ap"], 2), ap_atlas=round(apa, 2),
                         dv_model=round(o["dv"], 2), dv_atlas=round(dva, 2),
                         dap=round(dap, 2), ddv=round(ddv, 2), ok=bool(dap < 0.15 and ddv < 0.15)))
    return rows


def _print(res, cons):
    print("\n== atlas-relax knob search ==")
    print(f"default objective {res['default_score']}  ->  searched {res['best_score']}")
    print("best knobs:")
    for k in RANGES:
        print(f"  {k:10s} {MATURE_DEFAULTS[k]:6.2f} -> {res['best'][k]:6.2f}")
    base, F, tgt = res["base"], res["F"], res["tgt"]
    a0, l0 = _limb_span(mature_with(base, F, dict(MATURE_DEFAULTS)), ~np.isin(F, LIMB_IDS))
    a1, l1 = _limb_span(mature_with(base, F, res["best"]), ~np.isin(F, LIMB_IDS))
    print(f"limb length (span/body): arm default {a0:.2f} -> searched {a1:.2f} (atlas {tgt['arm']:.2f}) | "
          f"leg default {l0:.2f} -> searched {l1:.2f} (atlas {tgt['leg']:.2f})")
    rc = _canon_ratios(mature_with(base, F, res["best"]), F)
    print(f"VITRUVIAN CANON (fraction of height): leg {rc['leg']:.2f} (canon {CANON['leg']}) | "
          f"arm {rc['arm']:.2f} (canon {CANON['arm']}) | shoulder {rc['shoulder']:.2f} (canon {CANON['shoulder']}) | "
          f"heads-tall {_heads_tall(mature_with(base, F, res['best']), F):.1f} (canon {HT_TARGET})")
    print("\n== embryo <-> atlas organ-address consistency (after searched maturation) ==")
    print(f"{'organ':10s} {'ap_mdl':>7s} {'ap_atl':>7s} {'dv_mdl':>7s} {'dv_atl':>7s}  match")
    nok = 0
    for r in cons:
        if not r.get("ok") and "note" in r:
            print(f"{r['organ']:10s}  {r['note']}"); continue
        nok += int(r["ok"])
        print(f"{r['organ']:10s} {r['ap_model']:7.2f} {r['ap_atlas']:7.2f} {r['dv_model']:7.2f} "
              f"{r['dv_atlas']:7.2f}  {'OK' if r['ok'] else 'off (dap %.2f ddv %.2f)' % (r['dap'], r['ddv'])}")
    print(f"{nok}/{len(cons)} shared organs on the atlas address (residual = embryo placement, not maturation)")


if __name__ == "__main__":
    res = search()
    cons = _consistency(res)
    _print(res, cons)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(best_knobs=res["best"], best_score=res["best_score"],
                   default_score=res["default_score"], consistency=cons), open(str(OUT) + ".json", "w"), indent=1)
    print(f"\nsaved {OUT}.json  (human_movie.MATURE_SEARCHED now picks these up)")
