"""
integrated_body.py -- THE INTEGRATED BODY: run EVERY head on ONE high-fidelity cloud and composite them
into a single anatomical model. The "one man" assembly -- until now each head ran as its own demo; here
they all share one 60k cloud and are drawn together, every system genome-derived, in one figure.

This is also the re-verification of all heads at high fidelity (they each run on the shared cloud), and the
payoff of the composition motif: because the heads are read-only, compositing them is just running them all
on the same base -- nothing conflicts.

Systems composited: skeleton (30 named vertebrae + limb bones), muscle (axial CT-scaffold + limb),
tendon (Scx), vasculature (aorta tree), branching organs (lung/kidney), neural wiring (motor + commissures),
skin placodes.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.integrated_body
Out: data/organ_cascade/integrated_body.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.tuned_knobs import tuned
from medic.human_movie import (mature_cloud, grow_limbs, _limb_grow_model, _long_axis_len,
                               shape_limbs, flex, LIMB, MATURE_SEARCHED)
from medic import part_resolved_anatomy as PR
from medic import limb_chondrogenesis_head as LC
from medic import ct_scaffold_head as CT
from medic import limb_muscle_head as LM
from medic import tendon_head as TD
from medic import vasculature_head as VA
from medic import branching_morphogenesis_head as BR
from medic import neural_wiring_head as NW
from medic import skin_placode_head as SK
from medic import spine_curvature_head as SC
from medic import limb_myoblast_migration as MIG
from medic import soft_tissue_contour_head as CN
from medic import shoulder_girdle_head as SG
from medic import pelvic_girdle_head as PG
from medic import rib_cage_head as RC
from medic import skull_head as SKU
from medic import face_features_head as FAC
from medic import teeth_head as TE
from medic import adipose_head as ADI
from medic import fascia_head as FAS
from medic import skin_shell_head as SKN


def assemble():
    base, F = build_base()                         # ONE high-fidelity cloud, shared by every head
    rng = np.random.default_rng(0)
    R = {"base": base, "F": F}

    # SKELETON
    V, Vname, Vreal, vpop, vreal, vnames = PR.complete_column(base, F, rng)
    V = SC.curve_column(V, Vname)                          # SPINE-CURVATURE head: give the column its sagittal S
    R["vertebrae"] = dict(P=V, name=Vname, n=vpop, from_model=vreal)
    R["limb_bones"] = LC.carve(base, F)
    R["digits"] = LC.finger_toes(R["limb_bones"])          # fingers (fore) + toes (hind), 5 rays each
    # SHOULDER-GIRDLE head: clavicle + scapula + glenoid + deltoid at the biacromial span -- the cells that
    # give the standing man his shoulder WIDTH (the shoulder_w knob had none to widen).
    R["shoulder_girdle"] = SG.build(base, F)["parts"]
    # PELVIC-GIRDLE head: ilium + ischiopubis + acetabulum -- the hip bone joining the sacrum to each femur
    # (the "2 pelvic bones, no pelvis" gap; the counterpart of the shoulder girdle).
    R["pelvic_girdle"] = PG.build(base, F)["parts"]
    # RIB-CAGE head: 12 rib pairs + sternum off the thoracic vertebrae (Gray's true/false/floating).
    R["rib_cage"] = RC.build(base, F)["parts"]
    # SKULL head: carve the cranial shell into named bones (neurocranium + facial), Gray's roster.
    R["skull"] = SKU.build(base, F)["parts"]
    # FACE head: carve the viscerocranium into recognisable FEATURES (orbits/nose/cheeks/mouth/chin) -- prominences
    # on the antinodes of the face cells' own gap-junction eigenmodes, magnitudes = real C-GWAS betas.
    R["face"] = FAC.build(base, F, skull=R["skull"])["features"]
    # TEETH head: lateral-inhibition-spaced arcades on the maxilla (upper) + mandible (lower) = the DV split.
    # Reuses the already-carved skull so the skull head is not re-run. 32 teeth, incisor->molar graded.
    R["teeth"] = TE.build(base, F, skull=R["skull"]).get("arches", {})
    # FLESH: adipose (PPARG fat -- subcutaneous contour + visceral) + fascia (COL1A1/SCX -- deep fascia + ligaments)
    R["adipose"] = ADI.build(base, F)
    R["fascia"] = FAS.build(base, F)
    # FINER ROSTER (Gray's): PATELLA (sesamoid in the quadriceps tendon, anterior to each knee) + HYOID (the
    # U-bone of the anterior neck, below the mandible) -- computed read-only from the hind-limb knee + mandible.
    R["patella"] = {}
    dsn = -1.0 if ("Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8
                   and np.median(base[F == FIDX["Notochord"]][:, 1]) >= np.median(base[:, 1])) else 1.0
    Hb = float(np.ptp(base[:, 0]))
    for side in ("R", "L"):
        hb = R["limb_bones"].get(f"hind-{side}")
        if hb is None:
            continue
        Pc, bl = hb["P"], np.asarray(hb["bone"])
        fem, tib = Pc[bl == "femur"], Pc[bl == "tibia"]
        if len(fem) < 4 or len(tib) < 4:
            continue
        knee = 0.5 * (fem.mean(0) + tib.mean(0))
        knee = knee + np.array([0.0, dsn * 0.03 * Hb, 0.0])         # nudge to the ventral (extensor) face of the knee
        pat = knee + rng.normal(size=(14, 3)) * 0.012
        R["patella"][f"patella-{side}"] = dict(kind="bone", part="patella", bone="patella", side=side, P=pat)
    R["hyoid"] = {}
    mand = R["skull"].get("mandible")
    if mand is not None:
        mc = mand["P"].mean(0) + np.array([-0.05 * Hb, dsn * 0.02 * Hb, 0.0])   # inferior + ventral to the jaw
        t = np.linspace(-1, 1, 16)                                  # a small U (the hyoid body + greater cornua)
        u = np.c_[mc[0] + 0.015 * Hb * t ** 2, np.full(16, mc[1]), 0.03 * Hb * t]
        R["hyoid"]["hyoid"] = dict(kind="bone", part="hyoid", bone="hyoid", side="M", P=u + rng.normal(size=(16, 3)) * 0.004)

    # MUSCLE
    R["axial_muscle"] = CT.carve(base, F)
    R["limb_muscle"] = LM.carve(base, F)
    # LIMB-MYOBLAST MIGRATION head -> fill the limbs with muscle, then the SOFT-TISSUE CONTOUR head makes it
    # fusiform. These migrated cells (Muscle-fate, in the limbs) are the mass that gives the standing man real
    # arms-and-legs muscle; they ride the maturation warp like every other part.
    b2, F2, nadd = MIG.augment(base, F)
    if nadd:
        Pc = CN.contour(b2, F2, target=0.11)               # fusiform limb bellies at anthropometric girth
        R["limb_myoblasts"] = Pc[len(base):]               # just the migrated (contoured) limb muscle cells
    else:
        R["limb_myoblasts"] = np.zeros((0, 3))

    # TENDON (join each muscle end to its nearest bone)
    groups = TD._muscle_groups(base, F); bpts = TD._bone_points(base, F); bt = cKDTree(bpts)
    tendons = []
    for g in groups:
        P = g["P"]; C = P - P.mean(0); ax = C @ np.linalg.svd(C, full_matrices=False)[2][0]
        for e in (P[ax.argmin()], P[ax.argmax()]):
            tendons.append(np.array([e, bpts[bt.query(e)[1]]]))
    R["tendons"] = tendons; R["n_muscles"] = len(groups)

    # VASCULATURE (dorsal-aorta tree)
    ay = (np.median(base[F == FIDX["Notochord"]][:, 1]) + 0.02 * np.ptp(base[:, 1])
          if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8 else np.median(base[:, 1]))
    xs = np.linspace(np.percentile(base[:, 0], 4), np.percentile(base[:, 0], 96), 16)
    vroot = [np.array([x, ay, 0.0]) for x in xs]
    tk = tuned("vasculature", {"step": 0.028, "d_inf": 0.22, "d_kill": 0.05})
    vn, ve = VA.grow_tree(base, vroot, step=tk["step"], d_inf=tk["d_inf"], d_kill=tk["d_kill"])
    R["vessels"] = dict(nodes=vn, edges=ve,
                        branch=sum(1 for i in range(len(vn)) if sum(e[0] == i for e in ve) >= 2))

    # BRANCHING organs
    R["branching"] = {}
    for on in BR.ORGANS:
        if on in FIDX and (F == FIDX[on]).sum() >= 12:
            R["branching"][on] = BR.branch_organ(base[F == FIDX[on]], base.mean(0))

    # NEURAL wiring
    R["neural"] = NW.build(base=base, F=F)

    # SKIN placodes
    sids = [FIDX[n] for n in SK.SKIN if n in FIDX]; skin = base[np.isin(F, sids)]
    wl = tuned("skin_placode", {"wl_frac": 0.09})["wl_frac"] * (np.ptp(skin, 0).max() + 1e-9)
    R["placodes"] = skin[SK.place(skin, wl)]
    return R


def mature_parts(R):
    """Mature the whole integrated body into the standing adult. The cloud's own maturation (mature_cloud +
    limb outgrowth + the standing pose) defines a DEFORMATION FIELD embryo->adult; every embedded system
    (bones, muscles, tendons, vessels, nerves, placodes) rides that warp -- exactly how tissue carries its
    structures as the body grows. Each part point moves by the (kNN-interpolated) displacement of the cloud
    cells around it, so head parts scale up, limb parts swing down, etc., coherently."""
    base, F = R["base"], R["F"]
    # process the cloud exactly as the movie does before maturing (shape the limb paddles + cephalic flex),
    # so the maturation produces the movie's clean arms-down standing form; the full raw-base -> adult
    # transform is the displacement field the parts ride.
    c = base.mean(0); c[2] = 0.0
    proc = flex(shape_limbs((base - c) * (0.9 / _long_axis_len(base)), F, 1.0, LIMB), 1.0)
    # the deformation field the SKELETON + systems ride is the SMOOTH allometry (register=False): the visceral
    # AP-address migration moves organ cells only, so folding it into the warp field would drag a rib near the
    # heart's old spot up with it. The parts ride the smooth growth; the organ surfaces get the registered cloud.
    def _grow(Q):
        Q = grow_limbs(Q, F == LIMB, _limb_grow_model(1.0, MATURE_SEARCHED["limb_ext"]),
                       _limb_grow_model(1.0, MATURE_SEARCHED.get("leg_ext", MATURE_SEARCHED["limb_ext"])), pose=1.0)
        return Q * (3.2 / _long_axis_len(Q))
    A_field = _grow(mature_cloud(proc, F, 1.0, MATURE_SEARCHED, register=False))   # skeleton/systems warp
    A = _grow(mature_cloud(proc, F, 1.0, MATURE_SEARCHED, register=True))          # registered organ cloud
    disp = A_field - base
    tree = cKDTree(base)

    def warp(P):
        P = np.atleast_2d(np.asarray(P, float))
        d, idx = tree.query(P, k=6)
        w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
        return P + (w[..., None] * disp[idx]).sum(1)

    M = {"base": A, "F": F}
    M["vertebrae"] = {**R["vertebrae"], "P": warp(R["vertebrae"]["P"])}
    M["limb_bones"] = {k: {**r, "P": warp(r["P"])} for k, r in R["limb_bones"].items()}
    M["digits"] = {k: {**r, "P": warp(r["P"])} for k, r in R["digits"].items()}
    M["shoulder_girdle"] = {k: {**p, "P": warp(p["P"])} for k, p in R["shoulder_girdle"].items()}
    M["pelvic_girdle"] = {k: {**p, "P": warp(p["P"])} for k, p in R["pelvic_girdle"].items()}
    M["rib_cage"] = {k: {**p, "P": warp(p["P"])} for k, p in R["rib_cage"].items()}
    M["skull"] = {k: {**p, "P": warp(p["P"])} for k, p in R["skull"].items()}
    M["face"] = {k: {**p, "P": warp(p["P"]), "landmark": warp(np.atleast_2d(p["landmark"]))[0]}
                 for k, p in R["face"].items()}
    M["teeth"] = {arch: [{**t, "pos": warp(np.atleast_2d(t["pos"]))[0], "P": warp(t["P"])} for t in teeth]
                  for arch, teeth in R.get("teeth", {}).items()}
    M["patella"] = {k: {**p, "P": warp(p["P"])} for k, p in R.get("patella", {}).items()}
    _w = lambda P: warp(P) if len(P) else P                     # warp helper that tolerates empty arrays
    M["adipose"] = {k: _w(R["adipose"][k]) for k in ("subcutaneous", "visceral")}
    M["fascia"] = dict(fascia=_w(R["fascia"]["fascia"]),
                       ligaments=[{**l, "p0": _w(np.array(l["p0"])[None])[0].tolist(),
                                   "p1": _w(np.array(l["p1"])[None])[0].tolist()} for l in R["fascia"]["ligaments"]])
    M["hyoid"] = {k: {**p, "P": warp(p["P"])} for k, p in R.get("hyoid", {}).items()}
    M["axial_muscle"] = ({**R["axial_muscle"], "mus": warp(R["axial_muscle"]["mus"])}
                         if R["axial_muscle"] is not None else None)
    M["limb_muscle"] = {k: {**r, "P": warp(r["P"])} for k, r in R["limb_muscle"].items()}
    M["limb_myoblasts"] = warp(R["limb_myoblasts"]) if len(R.get("limb_myoblasts", [])) else R.get("limb_myoblasts", np.zeros((0, 3)))
    M["tendons"] = [warp(t) for t in R["tendons"]]
    vn = warp(np.array(R["vessels"]["nodes"]))
    M["vessels"] = {"nodes": vn, "edges": R["vessels"]["edges"], "branch": R["vessels"]["branch"]}
    M["branching"] = {n: {**b, "nodes": warp(np.array(b["nodes"])), "cells": warp(b["cells"])}
                      for n, b in R["branching"].items()}
    M["neural"] = {**R["neural"], "tracts": [warp(t) for t in R["neural"]["tracts"]],
                   "neural": warp(R["neural"]["neural"])}
    M["placodes"] = warp(R["placodes"])
    # ARM-OVERSHOOT FIX: warp() EXTRAPOLATES limb parts that extend past the base cloud (digits especially,
    # after finger_toes) -> the arm bones stretch down the midline to the floor. The envelope clip can't catch
    # this (the overshoot is axial + radially INSIDE the silhouette). Instead clamp each limb's parts to the
    # matured limb-SKIN axial extent: the arm skin in A hangs to ~mid-thigh, so no arm bone should reach past
    # it. Fore vs hind limb skin split at the limb cells' AP median (legs posterior, arms anterior).
    lc = A[F == LIMB] if LIMB in np.unique(F) else A[:0]
    if len(lc) > 8:
        apm = float(np.median(lc[:, 0]))
        fore_lo = float(lc[lc[:, 0] >= apm][:, 0].min()) if (lc[:, 0] >= apm).any() else A[:, 0].min()
        hind_lo = float(lc[lc[:, 0] < apm][:, 0].min()) if (lc[:, 0] < apm).any() else A[:, 0].min()
        # per (kind, side) matured limb-SKIN ML column centre -> SEAT each limb bone into its limb, so the
        # thigh/upper-arm descend as columns instead of splaying laterally as wedges (the femur-wing artifact).
        mlc = {}
        for kind, km in (("fore", lc[:, 0] >= apm), ("hind", lc[:, 0] < apm)):
            sub = lc[km]
            for side, sm in (("R", sub[:, 2] > 0), ("L", sub[:, 2] < 0)):
                if sm.sum() >= 4:
                    mlc[(kind, side)] = float(np.median(sub[sm, 2]))

        def _clamp_limb(P, kind):
            P = np.array(P, float).copy()
            P[:, 0] = np.clip(P[:, 0], fore_lo if kind == "fore" else hind_lo, None)   # axial: no overshoot
            side = "R" if np.median(P[:, 2]) > 0 else "L"
            if (kind, side) in mlc:
                m = mlc[(kind, side)]
                P[:, 2] = m + (P[:, 2] - m) * 0.30           # pull ML into the limb column (de-splay; 0.45->0.30 kills the residual femur wedge)
            return P
        for key in ("limb_bones", "digits", "limb_muscle"):
            M[key] = {k: {**r, "P": _clamp_limb(r["P"], r.get("kind", "fore"))} for k, r in M[key].items()}
    return M


# the model's own anatomy systems (for the movie reveal) -- internal systems only (skin is what dissolves)
SYS = ["bone", "muscle", "digit", "tendon", "vessel", "nerve", "face", "fat", "collagen", "tooth"]
SYS_COLS = [[0.85, 0.83, 0.78], [0.72, 0.26, 0.22], [0.92, 0.90, 0.82],
            [0.90, 0.85, 0.55], [0.78, 0.16, 0.16], [0.30, 0.62, 0.88], [0.36, 0.85, 0.90],
            [0.95, 0.82, 0.50], [0.80, 0.88, 0.80], [0.97, 0.97, 0.92]]   # fat=yellow, collagen=green, tooth=enamel-white
_QUOTA = dict(bone=0.23, muscle=0.22, digit=0.07, tendon=0.06, vessel=0.11, nerve=0.06, face=0.06,
              fat=0.11, collagen=0.05, tooth=0.03)


def _line_pts(segs, per=4):
    out = []
    for s in segs:
        s = np.atleast_2d(np.asarray(s, float))
        for k in range(len(s) - 1):
            for t in np.linspace(0, 1, per, endpoint=False):
                out.append(s[k] * (1 - t) + s[k + 1] * t)
    return np.array(out) if out else np.zeros((0, 3))


def _retroperitoneal(M, rng):
    """The structures BEHIND and BETWEEN the kidneys, which the cloud lacks as distinct parts: the PSOAS MAJOR
    (paired, lumbar bodies -> lesser trochanter), the QUADRATUS LUMBORUM (paired, 12th rib -> iliac crest, the
    posterior wall), and the INFERIOR VENA CAVA (pelvis -> right atrium, just right of and in front of the
    vertebral column). Placed by anatomical rule relative to the matured spine + kidneys. Reveal-only (added to
    the muscle/vessel systems), so the audits on the raw cloud are untouched. Laid frame x=AP head+x, y=DV, z=ML."""
    base, F = M["base"], M["F"]
    x = base[:, 0]; xmin = x.min(); H = np.ptp(x) + 1e-9
    def band(a, b):                                            # cells in an apf window
        apf = (x - xmin) / H
        return base[(apf >= a) & (apf < b)]
    # dorsal reference = the axial midline (notochord/spinal cord); ventral is the opposite DV side
    axf = FIDX.get("Notochord") if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8 else None
    spine = base[F == axf] if axf is not None else band(0.45, 0.65)
    if not len(spine):
        return dict(muscle=np.zeros((0, 3)), vessel=np.zeros((0, 3)))
    dvmid = float(np.median(spine[:, 1])); zmid = float(np.median(spine[:, 2]))
    dorsal = np.sign(dvmid - np.median(base[:, 1])) or 1.0     # +1 if spine is on the high-DV (dorsal) side
    ventral = -dorsal
    kid = base[np.isin(F, [FIDX[n] for n in ("Kidney", "Nephron") if n in FIDX])]
    kml = np.percentile(np.abs(kid[:, 2] - zmid), 70) if len(kid) else 0.06 * H   # kidney lateral offset
    def _ap(a):                                                # x-coordinate at apf a
        return xmin + a * H
    def _tube(p0, p1, n, r):
        t = rng.random(n)[:, None]
        line = p0[None] + t * (p1 - p0)[None]
        return line + rng.normal(size=(n, 3)) * r
    mus = []
    for s in (+1.0, -1.0):                                     # bilateral
        # psoas: lumbar body (near midline, ventral) -> hip (lateral, ventral, caudal)
        p0 = np.array([_ap(0.60), dvmid + ventral * 0.02 * H, zmid + s * 0.03 * H])
        p1 = np.array([_ap(0.40), dvmid + ventral * 0.10 * H, zmid + s * 0.09 * H])
        mus.append(_tube(p0, p1, 130, 0.018 * H))
        # quadratus lumborum: 12th-rib level (dorsal, lateral) -> iliac crest (dorsal), a posterior sheet
        q0 = np.array([_ap(0.62), dvmid + dorsal * 0.02 * H, zmid + s * kml])
        q1 = np.array([_ap(0.50), dvmid + dorsal * 0.02 * H, zmid + s * (kml + 0.03 * H)])
        mus.append(_tube(q0, q1, 90, 0.016 * H))
    muscle = np.vstack(mus)
    # IVC: common-iliac confluence (pelvis) -> right atrium, right of + ventral to the vertebral column
    v0 = np.array([_ap(0.45), dvmid + ventral * 0.04 * H, zmid + 0.035 * H])
    v1 = np.array([_ap(0.70), dvmid + ventral * 0.05 * H, zmid + 0.045 * H])
    vessel = _tube(v0, v1, 160, 0.012 * H)
    return dict(muscle=muscle, vessel=vessel)


def anatomy_points(rng=None, n_target=2800):
    """The model's OWN matured anatomy as n_target colored points (internal systems), for the movie reveal.
    Returns (pos[n_target,3], sysid[n_target], SYS_COLS, SYS, counts)."""
    if rng is None:
        rng = np.random.default_rng(0)
    M = mature_parts(assemble())
    P, S = {k: [] for k in SYS}, {}
    P["bone"].append(M["vertebrae"]["P"])
    for r in M["limb_bones"].values():
        P["bone"].append(r["P"])
    for r in M["digits"].values():
        P["digit"].append(r["P"])
    for p in M.get("shoulder_girdle", {}).values():        # clavicle/scapula/glenoid=bone, deltoid=muscle
        P["muscle" if p["kind"] == "muscle" else "bone"].append(p["P"])
    for p in M.get("pelvic_girdle", {}).values():          # ilium/ischium/pubis/acetabulum = bone
        P["bone"].append(p["P"])
    for p in M.get("rib_cage", {}).values():               # ribs + sternum = bone
        P["bone"].append(p["P"])
    for p in M.get("skull", {}).values():                  # cranial bones = bone
        P["bone"].append(p["P"])
    for p in M.get("face", {}).values():                   # facial features (orbits/nose/cheeks/mouth/chin)
        # orbits carry their optic-cup cells; single-point features are replicated so they survive subsample.
        P["face"].append(p["P"] if len(p["P"]) > 3 else np.repeat(p["landmark"][None], 12, axis=0))
    for teeth in M.get("teeth", {}).values():              # TEETH: the maxillary + mandibular arcades
        for t in teeth:
            P["tooth"].append(t["P"])
    for key in ("subcutaneous", "visceral"):               # FAT: subcutaneous contour + visceral depot
        fp = M.get("adipose", {}).get(key, np.zeros((0, 3)))
        if len(fp):
            P["fat"].append(fp)
    fasc = M.get("fascia", {})                              # COLLAGEN: deep fascia + ligament bands
    if len(fasc.get("fascia", [])):
        P["collagen"].append(fasc["fascia"])
    if fasc.get("ligaments"):
        P["collagen"].append(_line_pts([np.array([l["p0"], l["p1"]]) for l in fasc["ligaments"]]))
    for p in M.get("patella", {}).values():                # kneecaps = bone
        P["bone"].append(p["P"])
    for p in M.get("hyoid", {}).values():                  # hyoid = bone
        P["bone"].append(p["P"])
    if M["axial_muscle"] is not None:
        P["muscle"].append(M["axial_muscle"]["mus"])
    for r in M["limb_muscle"].values():
        P["muscle"].append(r["P"])
    retro = _retroperitoneal(M, rng)                       # psoas + quadratus lumborum + IVC (behind/between kidneys)
    if len(retro["muscle"]):
        P["muscle"].append(retro["muscle"])
    if len(retro["vessel"]):
        P["vessel"].append(retro["vessel"])
    if len(M.get("limb_myoblasts", [])):
        P["muscle"].append(M["limb_myoblasts"])            # migrated + contoured limb muscle mass
    P["tendon"].append(_line_pts(M["tendons"]))
    vn = M["vessels"]["nodes"]
    P["vessel"].append(_line_pts([np.array([vn[a], vn[b]]) for a, b in M["vessels"]["edges"]]))
    for b in M["branching"].values():
        nb = b["nodes"]
        P["vessel"].append(_line_pts([np.array([nb[a], nb[c]]) for a, c in b["edges"]]))
    P["nerve"].append(_line_pts(M["neural"]["tracts"]))
    pos, sid = [], []
    for i, k in enumerate(SYS):
        pk = np.vstack([p for p in P[k] if len(p)]) if any(len(p) for p in P[k]) else np.zeros((0, 3))
        if not len(pk):
            continue
        take = max(1, int(_QUOTA[k] * n_target))
        idx = rng.choice(len(pk), take, replace=len(pk) < take)
        pos.append(pk[idx]); sid.append(np.full(take, i))
    pos = np.vstack(pos); sid = np.concatenate(sid)
    if len(pos) > n_target:                                     # trim / pad to exactly n_target
        j = rng.choice(len(pos), n_target, replace=False); pos, sid = pos[j], sid[j]
    elif len(pos) < n_target:
        j = rng.choice(len(pos), n_target - len(pos), replace=True)
        pos = np.vstack([pos, pos[j]]); sid = np.concatenate([sid, sid[j]])
    # SPLAY FIX: bound the anatomy inside the body's own skin envelope -- pulls in any part (limb/vessel/nerve
    # spike) that over-warped BEYOND the body, without disturbing anatomy already inside (a posed arm stays).
    pos = SKN.envelope_clip(pos, M["base"]).astype(np.float32)
    counts = _stats(M)
    # ORGAN SURFACES: alpha-shape solids per organ on the MATURED condensed cloud (M["base"]), so the reveal can
    # draw the organs as Gray's-crisp SOLIDS, not a point scatter. Returned in the SAME (M) frame as `pos`, so the
    # caller applies its one pos->adult alignment to them too.
    from medic.organ_surface_head import build as _organ_surfaces
    organ_meshes = _organ_surfaces(M["base"], M["F"])
    return pos.astype(np.float32), sid.astype(int), SYS_COLS, SYS, counts, organ_meshes


def _stats(R):
    lb = R["limb_bones"]; lm = R["limb_muscle"]
    return dict(
        cloud_cells=len(R["base"]),
        vertebrae=f"{R['vertebrae']['n']}/{PR.N_VERT} ({int(R['vertebrae']['from_model'])} from model cells)",
        limb_bones=len({b for r in lb.values() for b in set(r['bone'].tolist())}),
        digits=sum(len({n for n in r['name']}) for r in R.get('digits', {}).values()),
        axial_muscle_domains=len(set(zip(R['axial_muscle']['seg'].tolist(),
                                         R['axial_muscle']['group'].tolist()))) if R['axial_muscle'] else 0,
        limb_muscles=len({m for r in lm.values() for m in set(r['muscle'].tolist())}),
        tendons=len(R["tendons"]),
        vessel_branch_points=R["vessels"]["branch"],
        lung_branches=R["branching"].get("Lung", {}).get("branch_points", 0),
        kidney_branches=R["branching"].get("Kidney", {}).get("branch_points", 0),
        motor_tracts=R["neural"]["n_motor"], commissures=R["neural"]["n_comm"],
        placodes=len(R["placodes"]),
        shoulder_girdle_parts=len(R.get("shoulder_girdle", {})),
        pelvic_girdle_parts=len(R.get("pelvic_girdle", {})),
        rib_pairs=sum(1 for n in range(1, 13) if f"rib{n}-R" in R.get("rib_cage", {})),
        skull_bones=len(R.get("skull", {})),
        teeth=sum(len(t) for t in R.get("teeth", {}).values()))


def _figure(R, out="data/organ_cascade/integrated_body.png", title="The integrated body"):
    base = R["base"]
    fig, ax = plt.subplots(1, 1, figsize=(9, 12), facecolor="#0d1017")
    ax.set_facecolor("#0d1017"); ax.set_aspect("equal"); ax.axis("off")
    P2 = lambda A: (A[:, 2], A[:, 0])                                   # front view: ML (x) x AP (y)
    S2 = lambda segs: [np.column_stack([s[:, 2], s[:, 0]]) for s in segs]
    from medic.skin_shell_head import silhouette_bounds                 # SKIN SHELL silhouette backdrop
    sx, slo, shi = silhouette_bounds(base)
    ax.fill_betweenx(sx, slo, shi, color="#c9a888", alpha=0.16, zorder=0)   # the body reads as a body
    ax.scatter(*P2(base), s=1, c="#1c2330", alpha=0.30)                 # body cloud (faint)
    # muscle (red) -- axial + limb
    if R["axial_muscle"] is not None:
        ax.scatter(*P2(R["axial_muscle"]["mus"]), s=3, c="#b0403a", alpha=0.35)
    for r in R["limb_muscle"].values():
        ax.scatter(*P2(r["P"]), s=3, c="#b0403a", alpha=0.35)
    # skeleton (bone) -- vertebrae + limb bones
    ax.scatter(*P2(R["vertebrae"]["P"]), s=9, c="#d8d2c4")
    for r in R["limb_bones"].values():
        ax.scatter(*P2(r["P"]), s=7, c="#d8d2c4", alpha=0.8)
    for r in R.get("digits", {}).values():                             # fingers + toes
        ax.scatter(*P2(r["P"]), s=5, c="#e8e2d4", alpha=0.9)
    for p in R.get("shoulder_girdle", {}).values():                    # clavicle/scapula/glenoid + deltoid
        ax.scatter(*P2(p["P"]), s=(4 if p["kind"] == "muscle" else 7),
                   c=("#b0403a" if p["kind"] == "muscle" else "#d8d2c4"), alpha=0.85)
    for p in R.get("pelvic_girdle", {}).values():                      # ilium/ischium/pubis/acetabulum
        ax.scatter(*P2(p["P"]), s=7, c="#d8d2c4", alpha=0.85)
    for p in R.get("rib_cage", {}).values():                           # ribs + sternum
        ax.scatter(*P2(p["P"]), s=5, c="#c7cdd6", alpha=0.8)
    for p in R.get("skull", {}).values():                              # cranial bones
        ax.scatter(*P2(p["P"]), s=6, c="#e8e2d4", alpha=0.85)
    ad = R.get("adipose", {})                                          # FAT (drawn faint, under the surface)
    if len(ad.get("subcutaneous", [])):
        ax.scatter(*P2(ad["subcutaneous"]), s=1, c="#f4d58d", alpha=0.12)
    if len(ad.get("visceral", [])):
        ax.scatter(*P2(ad["visceral"]), s=2, c="#e8a04b", alpha=0.3)
    ligs = [np.array([l["p0"], l["p1"]]) for l in R.get("fascia", {}).get("ligaments", [])]
    if ligs:                                                           # COLLAGEN ligaments (bone-to-bone bands)
        ax.add_collection(LineCollection(S2(ligs), colors="#bfe3c0", linewidths=0.5, alpha=0.7))
    for nm, p in R.get("face", {}).items():                            # facial features (eigenmode-anchored)
        lm = np.atleast_2d(p["landmark"])
        ax.scatter(*P2(lm), s=22, c="#5bd8e0", edgecolors="#0d1017", linewidths=0.3, zorder=6)
    for teeth in R.get("teeth", {}).values():                          # teeth (maxillary + mandibular arcades)
        for t in teeth:
            ax.scatter(*P2(t["P"]), s=5 * t["size"], c="#f6f6ee", alpha=0.9, zorder=6)
    for p in list(R.get("patella", {}).values()) + list(R.get("hyoid", {}).values()):   # patella + hyoid
        ax.scatter(*P2(p["P"]), s=7, c="#e8e2d4", alpha=0.9)
    # tendons (yellow), vessels (crimson), organ trees (gold), nerves (cyan)
    ax.add_collection(LineCollection(S2(R["tendons"]), colors="#f0e28a", linewidths=0.5, alpha=0.8))
    ve = R["vessels"]; vn = ve["nodes"]
    ax.add_collection(LineCollection(S2([np.array([vn[a], vn[b]]) for a, b in ve["edges"]]),
                                     colors="#c02828", linewidths=0.5, alpha=0.8))
    for br in R["branching"].values():
        nb = br["nodes"]
        ax.add_collection(LineCollection(S2([np.array([nb[a], nb[b]]) for a, b in br["edges"]]),
                                         colors="#e0b040", linewidths=0.8, alpha=0.9))
    for t, k in zip(R["neural"]["tracts"], R["neural"]["kinds"]):
        c = {"motor": "#3aa0e0", "commissure": "#40d0d0", "longitudinal": "#7ad0ff"}[k]
        ax.add_collection(LineCollection(S2([t]), colors=c, linewidths=0.4, alpha=0.6))
    ax.scatter(*P2(R["placodes"]), s=14, c="#f0a840", edgecolors="#fff", linewidths=0.2)   # placodes
    st = _stats(R)
    ax.set_title(f"{title} — every head on one high-fidelity cloud\n"
                 f"bone · muscle · tendon(yellow) · vessel(crimson) · organ-tree(gold) · nerve(cyan) · placode(orange)\n"
                 f"{st['vertebrae']} vertebrae · {st['limb_bones']} limb bones · {st['axial_muscle_domains']} axial + "
                 f"{st['limb_muscles']} limb muscles · {st['tendons']} tendons · {st['vessel_branch_points']} vessel "
                 f"branches · {st['motor_tracts']} motor + {st['commissures']} commissural tracts · {st['placodes']} placodes",
                 color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig(out, dpi=130, facecolor="#0d1017")
    print(f"saved {out}")


def main():
    print("assembling the integrated body (all heads on one 60k cloud) ...")
    R = assemble()
    st = _stats(R)
    print("\n== THE INTEGRATED BODY (every system, genome-derived, one cloud) ==")
    for k, v in st.items():
        print(f"  {k:22s} {v}")
    _figure(R)
    json.dump(st, open("data/organ_cascade/integrated_body.json", "w"), indent=1)
    print("saved data/organ_cascade/integrated_body.json")
    print("\nmaturing the integrated body into the standing all-parts man ...")
    M = mature_parts(R)
    _figure(M, out="data/organ_cascade/integrated_body_adult.png",
            title="The integrated body — MATURED (standing adult)")


if __name__ == "__main__":
    main()
