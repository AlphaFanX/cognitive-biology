"""grays_scorecard.py -- the per-part IDENTITY-LEVEL Gray's Anatomy scorecard (Miles, 2026-08-06).

The aggregate scores (Stage-1 bone shape 0.878, integrity 13/13, roster 15/15) are the "good index" that
Paper #10's bar explicitly rejects. This module discharges the real bar -- MIRROR GRAY'S STRUCTURE BY
STRUCTURE -- by scoring EACH named part in isolation against an identity-level checklist (not a type: not
"is this a long bone" but "is this THE femur -- head, neck, shaft, condyles, longest").

Two levels of check:
  * SPECIFIC checklists for structures with crisp Gray's features (skull / hand / femur here; extend the
    CHECKLISTS registry across the roster).
  * GENERIC form/topology/non-degeneracy/symmetry for every other named part, so the sweep is COMPLETE.

It runs AUTOMATICALLY over every part the assembled body exposes and writes one JSON per part plus a master
summary into a NEW directory (data/grays_scorecard/). Isolation validates FORM; RELATIONS/articulations/O-I
are covered by the assembled audits (integrity, musculoskeletal) -- this is additive, the missing per-part leg.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.grays_scorecard
"""
from __future__ import annotations
import os, json
import numpy as np

from medic.flesh_curriculum import _desc, _tortuosity

OUTDIR = "data/grays_scorecard"


def _np(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o)} not serializable")


# ----------------------------------------------------------------- form primitives (correspondence-free)
def _pca(P):
    C = np.asarray(P, float) - np.asarray(P, float).mean(0)
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    return C, S / (np.sqrt(len(P)) + 1e-9), Vt


def sphericity(P):
    _, l, _ = _pca(P)
    return float(l[2] / (l[0] + 1e-9))                       # 1 = round (braincase), 0 = flat/needle


def hollowness(P):
    """Fraction by which the interior is emptier than the shell -- a cranial/organ cavity reads high."""
    C, l, _ = _pca(P)
    r = np.linalg.norm(C, axis=1); rmax = np.percentile(r, 95) + 1e-9
    inner = (r < 0.4 * rmax).mean(); shell = ((r >= 0.6 * rmax) & (r <= rmax)).mean()
    # normalise by the volume ratio a SOLID body would give (inner sphere vs shell)
    solid_inner = 0.4 ** 3; solid_shell = 1 - 0.6 ** 3
    return float(max(0.0, 1.0 - (inner / (solid_inner + 1e-9)) / ((shell / (solid_shell + 1e-9)) + 1e-9)))


def radius_profile(P, nb=10):
    """Perpendicular radius along the principal axis -> detects head/condyle bulges vs a plain shaft."""
    C, l, Vt = _pca(P); t = C @ Vt[0]
    o = np.argsort(t); prof = []
    for idx in np.array_split(o, nb):
        if len(idx) < 3:
            prof.append(0.0); continue
        seg = C[idx]; ax = seg @ Vt[0]
        perp = seg - np.outer(ax, Vt[0])
        prof.append(float(np.percentile(np.linalg.norm(perp, axis=1), 80)))
    return np.array(prof)


def bilateral_ok(P):
    """A paired part should be one-sided (not straddling the midline z=0)."""
    z = np.asarray(P, float)[:, 2]
    return float(abs(z.mean()) / (np.std(z) + 1e-9))        # >~1 = cleanly lateralised


def _span(P):
    """Longest extent of a cloud = ptp along its OWN principal axis (orientation-free length)."""
    if len(P) < 3:
        return 0.0
    C = np.asarray(P, float) - np.asarray(P, float).mean(0)
    return float(np.ptp(C @ np.linalg.svd(C, full_matrices=False)[2][0]))


def _stature(R):
    """Standing height of the scored specimen = the base cloud's longest principal extent (crown->sole on
    the matured standing body). Cached on R -- the organ checks that need a size fraction share one measure."""
    if "_stature" not in R:
        R["_stature"] = _span(np.asarray(R["base"], float))
    return R["_stature"]


# ----------------------------------------------------------------- SPECIFIC Gray's checklists
def check_skull(parts):
    """Gray's skull: a rounded braincase of several named bones, bilateral pairs, a separate mandible, orbits."""
    feats = []
    named = {k: v for k, v in parts.items() if isinstance(v, dict) and v.get("P") is not None}
    counts = {k: len(v["P"]) for k, v in named.items()}
    # (1) >=6 named cranial bones present and NON-DEGENERATE (>=20 cells)
    solid = [k for k, n in counts.items() if n >= 20]
    feats.append(dict(feature=">=6 non-degenerate named bones", passed=len(solid) >= 6,
                      detail=f"{len(solid)} solid of {len(counts)}; degenerate: "
                             f"{[k for k, n in counts.items() if n < 20]}"))
    # (2) braincase roundedness (vault = the full NEUROCRANIUM: frontal+parietal+temporal+occipital -- the
    # temporal squama IS a vault bone; excluding it made this a top-heavy bowl that rewarded a small
    # non-enclosing cap over a vault that actually contains the brain, 2026-08-29 fix)
    vault = np.vstack([named[k]["P"] for k in named
                       if any(s in k for s in ("frontal", "occipital", "parietal", "temporal"))]
                      or [np.zeros((1, 3))])
    sph = sphericity(vault) if len(vault) > 6 else 0.0
    feats.append(dict(feature="braincase rounded (sphericity>=0.45)", passed=sph >= 0.45, detail=f"sphericity {sph:.2f}"))
    # (3) bilateral pairs present & lateralised (temporal, zygomatic)
    pairs = [b for b in ("temporal", "zygomatic", "parietal") if f"{b}-R" in named and f"{b}-L" in named]
    feats.append(dict(feature="bilateral pairs present", passed=len(pairs) >= 1,
                      detail=f"paired: {pairs}"))
    # (4) mandible present & separate (a distinct sizeable bone)
    feats.append(dict(feature="mandible present", passed=counts.get("mandible", 0) >= 50,
                      detail=f"mandible n={counts.get('mandible', 0)}"))
    # (5) cranial cavity (hollow vault)
    hol = hollowness(vault) if len(vault) > 30 else 0.0
    feats.append(dict(feature="cranial cavity (hollow)", passed=hol >= 0.2, detail=f"hollowness {hol:.2f}"))
    return feats


def check_hand(rays_names, rays_P):
    """Gray's hand: 5 digital rays, 5 metacarpals + 14 phalanges = 27 bones with carpals, middle digit longest."""
    feats = []
    names = np.asarray(rays_names)
    digs = sorted(set(int(n.split()[-1]) for n in names if n.split()[-1].isdigit()))
    feats.append(dict(feature="5 digital rays", passed=len(digs) == 5, detail=f"rays {digs}"))
    has_mc = any(("metacarpal" in n or "metatarsal" in n) for n in names)   # hand OR foot
    has_ph = any("phalanx" in n for n in names)
    feats.append(dict(feature="meta(carpals/tarsals) + phalanges", passed=has_mc and has_ph,
                      detail=f"metapodial={has_mc} phalanx={has_ph}"))
    # digit length profile: middle ray (3) longest (span along each ray's OWN axis, not a fixed axis)
    lens = {}
    for d in digs:
        m = np.array([n.endswith(f" {d}") for n in names])
        if m.sum() >= 3:
            lens[d] = _span(rays_P[m])
    longest = max(lens, key=lens.get) if lens else 0
    feats.append(dict(feature="middle digit (III) longest", passed=longest == 3,
                      detail=f"longest=D{longest}; lengths={ {k: round(v,1) for k,v in lens.items()} }"))
    return feats


def check_femur(P, sibling_lengths):
    """Gray's femur: the longest bone; a long bone with a proximal HEAD, a NECK, a shaft, distal CONDYLES."""
    feats = []
    d = _desc(P)
    feats.append(dict(feature="long bone (elong>=2)", passed=d["elong"] >= 2.0, detail=f"elong {d['elong']:.2f}"))
    prof = radius_profile(P)
    ends_vs_mid = (prof[:2].mean() + prof[-2:].mean()) / 2 / (prof[3:7].mean() + 1e-9)
    feats.append(dict(feature="epiphyses: head+condyles wider than shaft (>=1.3)", passed=ends_vs_mid >= 1.3,
                      detail=f"end/shaft radius {ends_vs_mid:.2f} (model femur is a plain rod if ~1.0)"))
    fspan = _span(P)
    longest = all(fspan >= L for L in sibling_lengths)       # longest of the limb bones (own-axis span)
    feats.append(dict(feature="longest limb bone", passed=bool(longest),
                      detail=f"femur span {fspan:.2f} vs siblings {[round(L,2) for L in sibling_lengths]}"))
    return feats


# ----------------------------------------------------------------- GENERIC checklist (every other part)
def check_generic(P, paired=False):
    feats = []
    d = _desc(P)
    feats.append(dict(feature="non-degenerate (>=20 cells)", passed=len(P) >= 20, detail=f"n={len(P)}"))
    if d is not None:
        feats.append(dict(feature="coherent form (has extent on >=2 axes)", passed=d["flat"] >= 0.03,
                          detail=f"elong {d['elong']:.2f} flat {d['flat']:.2f}"))
    if paired:
        b = bilateral_ok(P)
        feats.append(dict(feature="lateralised (paired)", passed=b >= 0.8, detail=f"|zbar|/sz {b:.2f}"))
    return feats


# ----------------------------------------------------------------- SPECIFIC Gray's organ checklists
def local_anisotropy(P, k=12, nsamp=800, seed=7):
    """Median local PCA elongation (l1/l2) over kNN neighbourhoods -- tube-ness that survives coiling:
    a coiled tube is locally 1D everywhere while globally compact; a blob is locally isotropic.
    Calibration (2026-09-05, matured specimen): reference gut 1.55, model gut 1.42, gaussian blob 1.29-1.30."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    if len(P) < 24:
        return 1.0
    r = np.random.default_rng(seed)
    idx = r.choice(len(P), size=min(nsamp, len(P)), replace=False)
    T = cKDTree(P)
    es = []
    for i in idx:
        _, nb = T.query(P[i], k=min(k, len(P)))
        C = P[nb] - P[nb].mean(0)
        s = np.linalg.svd(C, full_matrices=False)[1]
        es.append(s[0] / (s[1] + 1e-9))
    return float(np.median(es))


def _paired(P):
    """A PAIRED organ = two lateral masses with a midline GAP, not one midline blob. Returns (is_paired, detail)."""
    z = np.asarray(P, float)[:, 2] - np.median(np.asarray(P, float)[:, 2])
    zmax = np.percentile(np.abs(z), 95) + 1e-9
    mid = (np.abs(z) < 0.25 * zmax).mean(); flank = (np.abs(z) > 0.55 * zmax).mean()
    return flank > 1.15 * mid, f"midline mass {mid:.2f} vs flank {flank:.2f}"


def _organ_feats(name, P, R):
    """Identity-level Gray's checklist per organ (shape features measurable from the cloud)."""
    d = _desc(P); e, f, sph, hol, lat = d["elong"], d["flat"], sphericity(P), hollowness(P), bilateral_ok(P)
    tort = _tortuosity(P); F = []
    def add(feat, ok, detail): F.append(dict(feature=feat, passed=bool(ok), detail=detail))
    if name == "Heart":
        # v1.3 recalibration (2026-09-05): the chamber split assigns "Left/Right Ventricle", never bare
        # "Ventricle" -- the old check failed on naming. And "hollow (has cavities)" was unpassable even by
        # the bp3d reference heart (hollow 0.00: the whole-mass cavity statistic cannot see 4 chamber lumens
        # around a septum) -- replaced by REAL SIZE, which catches the oversized-chamber fault class
        # (reference heart span = 0.072 of stature; bp3d 115/1655mm = 0.070).
        pres = R.get("_organ_present", set())
        vent = [c for c in ("Ventricle", "Left Ventricle", "Right Ventricle") if c in pres]
        chambers = [c for c in ("Atrium", "Outflow") if c in pres] + vent
        add("compact (not elongated, elong<2.5)", e < 2.5, f"elong {e:.2f}")
        add("chambered (atrium + ventricle + outflow)",
            "Atrium" in pres and vent and "Outflow" in pres, f"chambers {chambers}")
        frac = _span(P) / (_stature(R) + 1e-9)
        add("real-sized (span 4-11% of stature)", 0.04 <= frac <= 0.11, f"span/stature {frac:.3f}")
    elif name == "Lung":
        # v1.3: "low-density/air (hollow>=0.4)" was unpassable by the reference lung too (hollow 0.00 --
        # cells cannot encode air, and the reference surface reads solid under the whole-mass statistic).
        # The measurable identity = LOBATION: >=4 of the 5 named lobe fates present.
        lobes = [c for c in ("Right Superior Lobe", "Right Middle Lobe", "Right Inferior Lobe",
                             "Left Superior Lobe", "Left Inferior Lobe") if c in R.get("_organ_present", set())]
        add("lobed (>=4 of 5 named lobes)", len(lobes) >= 4, f"lobes {len(lobes)}/5")
        p, pd = _paired(P); add("paired (L+R lungs)", p, pd)
    elif name == "Kidney":
        add("reniform (1.5<=elong<=3.2)", 1.5 <= e <= 3.2, f"elong {e:.2f}")
        p, pd = _paired(P); add("paired (L+R kidneys)", p, pd)
    elif name == "Liver":
        add("lateralised to one side (|lat|>=1)", lat >= 1.0, f"lat {lat:.2f}")
        add("bulky solid (not hollow)", hol < 0.2, f"hollow {hol:.2f}")
    elif name == "Gut":
        # v1.3 recalibration (2026-09-05): the old "tube (elong>=2)" was the EMBRYONIC spanning-thread
        # signature -- the reference adult gut, coiled into the abdomen, reads elong 1.24 and could never
        # pass. And "has a lumen (hollow>=0.3)" was representation-biased (bp3d surface passes at 0.43; a
        # solid cell cloud reads 0.00; the lumen is sub-resolution at ~88k body cells). The measurable adult
        # identity: locally-1D tube (survives coiling), tortuous path, packed into a compact mass.
        le = local_anisotropy(P)
        add("tube (locally 1D, local anisotropy>=1.38)", le >= 1.38, f"localE {le:.2f} (ref 1.55, blob 1.30)")
        add("coiled (tortuous>=1.2)", tort >= 1.2, f"tort {tort:.2f}")
        add("packed abdominal mass (elong<2.2)", e < 2.2, f"elong {e:.2f} (ref 1.24)")
    elif name in ("Forebrain", "Cerebellum"):               # cerebrum / cerebellum = ROUNDED masses
        add("rounded (elong<2.6)", e < 2.6, f"elong {e:.2f}")
        add("bulky (>=200 cells)", len(P) >= 200, f"n={len(P)}")
    elif name in ("Midbrain", "Hindbrain"):                 # BRAINSTEM -> elongated, NOT round (Gray's)
        add("brainstem elongated (elong>=1.5)", e >= 1.5, f"elong {e:.2f}")
        add("bulky (>=200 cells)", len(P) >= 200, f"n={len(P)}")
    elif name == "Spinal Cord":
        add("thin cord (elong>=4)", e >= 4.0, f"elong {e:.2f}")
        add("midline (not lateralised)", lat < 0.5, f"lat {lat:.2f}")
    elif name == "Eye":
        add("hollow globe (hollow>=0.5)", hol >= 0.5, f"hollow {hol:.2f}")
        p, pd = _paired(P); add("paired (L+R eyes)", p, pd)
    elif name == "Spleen":
        add("lateralised (|lat|>=1)", lat >= 1.0, f"lat {lat:.2f}")
    elif name == "Bladder":
        add("hollow sac (hollow>=0.2)", hol >= 0.2, f"hollow {hol:.2f}")
    elif name == "Pancreas":
        add("elongated gland (elong>=1.5)", e >= 1.5, f"elong {e:.2f}")
    else:
        return None
    return F


ORGAN_IDENTITY = {"Heart", "Lung", "Kidney", "Liver", "Gut", "Forebrain", "Cerebellum", "Midbrain",
                  "Hindbrain", "Spinal Cord", "Eye", "Spleen", "Bladder", "Pancreas"}


# ----------------------------------------------------------------- vertebrae / ribs / girdle / muscle identity
def _skew(P):
    """Max directional skewness (in the PCA frame). A vertebra with a spinous/transverse PROCESS is skewed;
    a plain symmetric blob is not."""
    C = np.asarray(P, float) - np.asarray(P, float).mean(0)
    if len(C) < 12:
        return 0.0
    proj = C @ np.linalg.svd(C, full_matrices=False)[2].T
    return max(abs((((proj[:, i] - proj[:, i].mean()) / (proj[:, i].std() + 1e-9)) ** 3).mean()) for i in range(3))


def check_vertebra(name, P):
    d = _desc(P); F = []
    F.append(dict(feature="non-degenerate (>=20 cells)", passed=len(P) >= 20, detail=f"n={len(P)}"))
    if "sacrum" in name:                                     # the fused S1-S5 composite (cycle 47)
        F.append(dict(feature="midline (not lateralised)", passed=bilateral_ok(P) < 0.5,
                      detail=f"lat {bilateral_ok(P):.2f}"))
        F.append(dict(feature="fused wedge (elong<=3)", passed=d["elong"] <= 3.0, detail=f"elong {d['elong']:.2f}"))
        return F
    F.append(dict(feature="blocky/cuboidal body (elong<=2.3)", passed=d["elong"] <= 2.3, detail=f"elong {d['elong']:.2f}"))
    sk = _skew(P)
    F.append(dict(feature="posterior process (skew>=0.3)", passed=sk >= 0.3, detail=f"skew {sk:.2f}"))
    return F


def check_rib(name, P):
    d = _desc(P); F = []
    import re
    mnum = re.search(r"rib(\d+)", name); num = int(mnum.group(1)) if mnum else 6
    floating = num >= 11                                     # ribs 11-12 are FLOATING: short + straighter (Gray's)
    F.append(dict(feature="long (elong>=1.4)", passed=d["elong"] >= 1.4, detail=f"elong {d['elong']:.2f}"))
    ct = 1.02 if floating else 1.1                           # floating ribs are genuinely less curved
    F.append(dict(feature=f"curved (tort>={ct})", passed=_tortuosity(P) >= ct, detail=f"tort {_tortuosity(P):.2f} (floating={floating})"))
    F.append(dict(feature="lateralised (paired)", passed=bilateral_ok(P) >= 1.0, detail=f"lat {bilateral_ok(P):.2f}"))
    return F


def check_girdle(name, P):
    F = []
    if "glenoid" in name or "acetabulum" in name:            # a SOCKET (may be degenerate = not modelled)
        F.append(dict(feature="socket present (>=20 cells)", passed=len(P) >= 20,
                      detail=f"n={len(P)} (socket not modelled if ~1)"))
        if len(P) >= 6:
            F.append(dict(feature="lateralised (paired)", passed=bilateral_ok(P) >= 1.0, detail=f"lat {bilateral_ok(P):.2f}"))
        return F
    d = _desc(P)
    if d is None:
        return [dict(feature="non-degenerate (>=20 cells)", passed=False, detail=f"n={len(P)}")]
    e, f = d["elong"], d["flat"]
    if "hip_bone" in name:                                   # the fused os coxae composite (cycle 47)
        F.append(dict(feature="fused of 3 parts (>=60 cells)", passed=len(P) >= 60, detail=f"n={len(P)}"))
        F.append(dict(feature="lateralised (paired)", passed=bilateral_ok(P) >= 1.0,
                      detail=f"lat {bilateral_ok(P):.2f}"))
        return F
    if "clavicle" in name:
        F.append(dict(feature="long bone (elong>=3)", passed=e >= 3.0, detail=f"elong {e:.2f}"))
    elif "scapula" in name:
        F.append(dict(feature="flat blade (flat<=0.25)", passed=f <= 0.25, detail=f"flat {f:.2f}"))
    elif "ilium" in name:
        F.append(dict(feature="iliac blade (flat<=0.58)", passed=f <= 0.58, detail=f"flat {f:.2f}"))
    elif "ischium" in name or "pubis" in name:
        F.append(dict(feature="irregular ramus (1.3<=elong<=4.5 & flat>=0.25)",
                      passed=(1.3 <= e <= 4.5 and f >= 0.25), detail=f"elong {e:.2f} flat {f:.2f}"))
    elif "glenoid" in name or "acetabulum" in name:
        F.append(dict(feature="socket present (>=20 cells)", passed=len(P) >= 20, detail=f"n={len(P)} (socket not modelled if ~1)"))
    else:
        F.append(dict(feature="coherent form", passed=f >= 0.03, detail=f"elong {e:.2f} flat {f:.2f}"))
    F.append(dict(feature="lateralised (paired)", passed=bilateral_ok(P) >= 1.0, detail=f"lat {bilateral_ok(P):.2f}"))
    return F


def check_muscle(P, O, I):
    from medic.flesh_curriculum import _muscle_shape
    F = []
    F.append(dict(feature="non-degenerate (>=20 cells)", passed=len(P) >= 20, detail=f"n={len(P)}"))
    sh = _muscle_shape(P, O, I)
    if sh is not None:
        spans = sh["align"] >= 0.55 and sh["spanfrac"] >= 0.45
        F.append(dict(feature="belly spans its O->I action line", passed=spans,
                      detail=f"align {sh['align']} span {sh['spanfrac']}"))
    return F


# a girdle entry that is actually a MUSCLE (deltoid etc.) -> route to the muscle check, not a bone check
_GIRDLE_MUSCLES = ("deltoid", "trapezius", "pectoralis", "latissimus")


# ----------------------------------------------------------------- enumerate EVERY named part
def collect_all_parts(R):
    import numpy as np
    from medic.unified_embryo import FIDX
    parts = []                                               # (name, category, cloud, meta)
    # skull (composite -> also scored as a whole below)
    for grp in ("shoulder_girdle", "pelvic_girdle", "rib_cage", "skull"):
        for nm, part in R.get(grp, {}).items():
            P = part.get("P") if isinstance(part, dict) else None
            if P is not None:
                parts.append((f"{grp}:{nm}", grp, np.asarray(P), dict(paired=nm.endswith(("-R", "-L")))))
    # THE HIP BONE (cycle 47, the declared-union join): bp3d ships the FUSED os coxae only, so the
    # model's ilium+ischium+pubis score as one composite against it (the wall-of-heart pattern);
    # the constituent parts keep their own rows.
    for side in ("L", "R"):
        comp = [np.asarray(R["pelvic_girdle"][f"{nm}-{side}"]["P"]) for nm in ("ilium", "ischium", "pubis")
                if f"{nm}-{side}" in R.get("pelvic_girdle", {})
                and R["pelvic_girdle"][f"{nm}-{side}"].get("P") is not None]
        if len(comp) == 3:
            parts.append((f"pelvic_girdle:hip_bone-{side}", "pelvic_girdle", np.vstack(comp), dict(paired=True)))
    V = R.get("vertebrae", {})
    if "P" in V:
        Vn = np.asarray(V["name"])
        for nm in np.unique(Vn):
            parts.append((f"vertebra:{nm}", "vertebrae", V["P"][Vn == nm], {}))
        # THE SACRUM (cycle 47): the fused bone = the S1-S5 union, the composite that matches the
        # bp3d sacrum mesh (single sacral vertebrae go honestly unmatched -- no source ships them)
        sm = np.isin(Vn, [f"S{i}" for i in range(1, 6)])
        if sm.sum() >= 20:
            parts.append(("vertebra:sacrum", "vertebrae", V["P"][sm], {}))
    for limb, dct in R.get("limb_bones", {}).items():
        P = dct.get("P"); bl = np.asarray(dct.get("bone", []))
        if P is not None and len(bl) == len(P):
            for bn in np.unique(bl):
                parts.append((f"{limb}:{bn}", "limb_bone", P[bl == bn], dict(paired=True)))
    for limb, dct in R.get("digits", {}).items():
        P = dct.get("P"); nm = np.asarray(dct.get("name", []))
        if P is not None and len(nm) == len(P):
            parts.append((f"{limb}:hand-or-foot", "autopod", P, dict(names=nm, paired=True)))
    for grp in ("patella", "hyoid", "tongue", "sinus"):
        for nm, part in R.get(grp, {}).items():
            P = part.get("P") if isinstance(part, dict) else None
            if P is not None:
                parts.append((f"{grp}:{nm}", grp, np.asarray(P), dict(paired=nm.endswith(("-R", "-L")))))
    # organs (from the labelled base). Composite Heart (chambers) + Gut (foregut+hindgut) recombined so the
    # identity checklist sees the whole organ, not the split subheads.
    base, F = R.get("base"), R.get("F")
    if base is not None and F is not None:
        present = {k for k in FIDX if (F == FIDX[k]).sum() >= 8}
        R["_organ_present"] = present
        composites = {"Heart": ("Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow"),
                      "Gut": ("Gut", "Foregut", "Hindgut")}
        from medic.subhead_program import composites as _sub_comp    # sub-head PROGRAM: children fold into their organ
        composites.update(_sub_comp())
        _sub_children = {c for parts in composites.values() for c in parts[1:]}
        for organ, idx in FIDX.items():
            if organ in _sub_children:
                continue                                     # folded into the Heart / Gut composite
            if organ in composites:
                Po = base[np.isin(F, [FIDX[c] for c in composites[organ] if c in FIDX])]
            else:
                Po = base[F == idx]
            if len(Po) >= 8:
                parts.append((f"organ:{organ}", "organ", Po, {}))
    # muscles -- each is now an explicit FUSIFORM BELLY between its O/I (head_muscle = jaw, named_muscles =
    # the rest), replacing the shared-scaffold carve that starved most muscles.
    for grp in ("head_muscle", "named_muscles"):
        for nm, hm in R.get(grp, {}).items():
            if hm.get("P") is not None and len(hm["P"]) >= 6:
                parts.append((f"muscle:{nm}", "muscle", np.asarray(hm["P"]), dict(O=hm["O"], I=hm["I"])))
    # TEETH (cycle 47, the per-tooth join): the model HAS 32 lateral-inhibition-spaced teeth; each
    # gets a row NAMED ON THE BP3D CONVENTION (position from midline -> central/lateral incisor,
    # canine, first/second premolar, first..third molar) so the canonical join is mechanical. bp3d
    # ships no third molars -- those go honestly unmatched.
    _TOOTH_POS = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar",
                  5: "second premolar", 6: "first molar", 7: "second molar", 8: "third molar"}
    for arch, teeth in R.get("teeth", {}).items():
        up = "upper" if arch == "maxillary" else "lower"
        for side in ("R", "L"):
            row = sorted((t for t in teeth if t.get("P") is not None and len(t["P"]) >= 6
                          and (float(np.asarray(t["pos"])[2]) > 0) == (side == "R")),
                         key=lambda t: abs(t.get("t", 0.0)))
            for i, t in enumerate(row[:8], 1):
                parts.append((f"tooth:{up} {_TOOTH_POS[i]}-{side}", "tooth",
                              np.asarray(t["P"]), dict(paired=True)))
    return parts


# ----------------------------------------------------------------- run the whole sweep
def score_part(name, category, P, meta, R):
    if category == "skull" and name == "skull:__whole__":
        feats = check_skull(R.get("skull", {}))
    elif category == "autopod":
        feats = check_hand(meta["names"], P)
    elif category == "limb_bone" and name.endswith(":femur"):
        sib = [_span(v["P"]) for k, v in _limb_siblings(R, name)]
        feats = check_femur(P, sib)
    elif category == "organ" and name.split(":", 1)[1] in ORGAN_IDENTITY:
        feats = _organ_feats(name.split(":", 1)[1], P, R) or check_generic(P)
    elif category == "vertebrae":
        feats = check_vertebra(name, P)
    elif category == "rib_cage" and "rib" in name and "sternum" not in name:
        feats = check_rib(name, P)
    elif category in ("shoulder_girdle", "pelvic_girdle"):
        if any(mm in name for mm in _GIRDLE_MUSCLES):
            feats = check_generic(P)                             # deltoid mass (the belly is in named_muscles)
        else:
            feats = check_girdle(name, P)
    elif category == "muscle":
        feats = check_muscle(P, meta["O"], meta["I"])
    else:
        feats = check_generic(P, paired=meta.get("paired", False))
    passed = sum(f["passed"] for f in feats); n = len(feats)
    return dict(part=name, category=category, n_cells=int(len(P)), features=feats,
                passed=passed, n_features=n, score=round(passed / (n + 1e-9), 3))


def _limb_siblings(R, femur_name):
    limb = femur_name.split(":")[0]
    dct = R["limb_bones"][limb]; bl = np.asarray(dct["bone"]); P = dct["P"]
    return [(bn, dict(P=P[bl == bn])) for bn in np.unique(bl) if bn != "femur"]


def run(R=None):
    from medic.integrated_body import assemble
    os.makedirs(OUTDIR, exist_ok=True)
    for _f in os.listdir(OUTDIR):                               # clear stale per-part files from prior runs
        if _f.endswith(".json"):
            os.remove(os.path.join(OUTDIR, _f))
    if R is None:                                               # benchmark_suite passes one shared specimen
        R = assemble()
    parts = collect_all_parts(R)
    # add the skull as a WHOLE composite (identity-level braincase checklist)
    scored = [score_part("skull:__whole__", "skull", np.vstack([v["P"] for v in R.get("skull", {}).values()
                                                                if isinstance(v, dict) and v.get("P") is not None]),
                         {}, R)]
    for name, cat, P, meta in parts:
        scored.append(score_part(name, cat, P, meta, R))
    for s in scored:
        safe = s["part"].replace(":", "__").replace(" ", "_")
        json.dump(s, open(os.path.join(OUTDIR, f"{safe}.json"), "w"), indent=1, default=_np)
    by_cat = {}
    for s in scored:
        by_cat.setdefault(s["category"], []).append(s["score"])
    summary = dict(n_parts=len(scored),
                   mean_score=round(float(np.mean([s["score"] for s in scored])), 3),
                   fully_passing=sum(s["score"] == 1.0 for s in scored),
                   by_category={c: round(float(np.mean(v)), 3) for c, v in by_cat.items()},
                   worst=sorted([(s["part"], s["score"]) for s in scored], key=lambda x: x[1])[:12])
    json.dump(summary, open(os.path.join(OUTDIR, "_summary.json"), "w"), indent=1, default=_np)
    print(f"GRAY'S PER-PART SCORECARD -> {OUTDIR}/  ({summary['n_parts']} parts, {summary['fully_passing']} fully pass)")
    print(f"  mean per-part score = {summary['mean_score']}")
    print(f"  by category: {summary['by_category']}")
    print("  --- the three exemplars (identity-level) ---")
    for s in scored:
        if s["part"] in ("skull:__whole__", "hind-R:femur") or s["part"].startswith("fore-R:hand"):
            print(f"  {s['part']}  score {s['score']} ({s['passed']}/{s['n_features']}):")
            for f in s["features"]:
                print(f"      [{'PASS' if f['passed'] else 'FAIL'}] {f['feature']} -- {f['detail']}")
    print(f"  worst 12: {summary['worst']}")


if __name__ == "__main__":
    run()
