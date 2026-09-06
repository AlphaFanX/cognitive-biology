"""
limb_muscle_head.py -- the LIMB MUSCLE head: carve the limb muscle mass into named limb muscles.

Closes muscle-for-muscle on the appendicular side, and composes on TWO prior heads' fields (the limb frame
+ the limb bones) -- read-only, so it cannot oppose either.

Biology (all genome-anchored, all fields already produced upstream):
  * PRECURSORS = Lbx1+ migratory hypaxial muscle cells that DELAMINATE from the somite and migrate into the
    limb bud (the CT-scaffold head's hypaxial group is their source). Naive -- they fill whatever the limb CT
    dictates.
  * LIMB CT = Tcf4/Osr1 lateral-plate connective tissue IN the limb, which reads the limb skeleton and lays
    the muscle cleavage planes (same CT logic as the axial CT-scaffold head).
  * The pattern it reads: DORSOVENTRAL = Lmx1b (dorsal -> EXTENSORS) vs En1 (ventral -> FLEXORS); PROXIMODISTAL
    = the limb PD Hox frame (stylopod / zeugopod), the SAME field limb_chondrogenesis used for the bones.
So each limb muscle = (limb) x (dorsal-extensor / ventral-flexor) x (PD segment), named off the limb bones it
spans -- e.g. fore ventral stylopod = biceps, fore dorsal stylopod = triceps/deltoid.

READ-ONLY: reads {limb frame, limb bones}, writes muscle identity; never moves the limb or the bones.
Validation = every limb carves into BOTH a dorsal and a ventral mass, PD-segmented, bilaterally symmetric
(it shares the limb frame), not told the answer.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.limb_muscle_head
Out: data/organ_cascade/limb_muscle_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.limb_chondrogenesis_head import carve as carve_bones, PD_SEGMENTS
from medic.tuned_knobs import tuned

# named limb muscles by (kind, dorsoventral, PD segment)
MUSCLE = {
    ("fore", "dorsal", "stylopod"): "deltoid/triceps", ("fore", "ventral", "stylopod"): "biceps/pectoral",
    ("fore", "dorsal", "zeugopod"): "wrist/digit extensors", ("fore", "ventral", "zeugopod"): "wrist/digit flexors",
    ("fore", "dorsal", "autopod"): "dorsal interossei", ("fore", "ventral", "autopod"): "palmar interossei",
    ("hind", "dorsal", "stylopod"): "gluteal/quadriceps", ("hind", "ventral", "stylopod"): "hamstring/adductor",
    ("hind", "dorsal", "zeugopod"): "tibialis/digit extensors", ("hind", "ventral", "zeugopod"): "gastrocnemius/flexors",
    ("hind", "dorsal", "autopod"): "dorsal pedal", ("hind", "ventral", "autopod"): "plantar",
}


def carve(base, F, dv_off=None, wrap=None):
    """The limb muscle head. For each limb: the migratory muscle mass (a dorsal + ventral shell around the
    PD-segmented skeleton) carved by DV x PD into named limb muscles. Returns per-limb dict.

    Knobs (exposed for tuning; None -> load the tuned value, else the explicit arg overrides): `dv_off` shifts
    the dorsal/ventral (Lmx1b/En1) cleavage plane along the DV axis, in units of the limb's DV std -- 0.0 =
    split at the geometric centre; a nonzero value balances the extensor/flexor masses when the bud is
    DV-asymmetric. `wrap` = how far the muscle shell offsets OFF the bone core along the DV axis (muscle belly
    thickness); larger = thicker wrap."""
    tk = tuned("limb_muscle", {"dv_off": 0.0, "wrap": 0.6})
    if dv_off is None:
        dv_off = tk["dv_off"]
    if wrap is None:
        wrap = tk["wrap"]
    bones = carve_bones(base, F)              # reuse the limb frame + PD from the chondrogenesis head
    rng = np.random.default_rng(0)
    res = {}
    for name, b in bones.items():
        kind, Q, pdn, seg = b["kind"], b["P"], b["pdn"], b["seg"]
        # DV axis of the limb = the bud's SECOND principal axis (after PD): dorsal (Lmx1b) vs ventral (En1)
        C = Q - Q.mean(0)
        V = np.linalg.svd(C, full_matrices=False)[2]
        dv = C @ V[1]                          # the transverse axis; sign = dorsal/ventral
        thr = dv_off * (dv.std() + 1e-9)       # DV cleavage plane, shiftable off centre
        # migratory muscle precursors: a dorsal shell + a ventral shell wrapping the skeleton (Lbx1 -> limb)
        Pm, seg_m, dvg = [], [], []
        for shell, sgn in (("dorsal", +1.0), ("ventral", -1.0)):
            sel = (dv >= thr) if sgn > 0 else (dv < thr)
            if sel.sum() < 3:
                continue
            # offset the shell OFF the bone core along the DV axis (muscle wraps the bone), keep PD position
            base_pts = Q[sel] + sgn * wrap * (np.abs(dv[sel])[:, None]) * V[1][None] \
                + rng.normal(size=(sel.sum(), 3)) * 0.005
            Pm.append(base_pts); seg_m += list(seg[sel]); dvg += [shell] * int(sel.sum())
        if not Pm:
            continue
        Pm = np.vstack(Pm); seg_m = np.array(seg_m); dvg = np.array(dvg)
        muscle = np.array([MUSCLE.get((kind, d, s), f"{kind} {d} {s}") for d, s in zip(dvg, seg_m)])
        res[name] = dict(kind=kind, P=Pm, seg=seg_m, dv=dvg, muscle=muscle)
    return res


# menagerie limb bone -> the MODEL PD segment (kind, segment) it corresponds to. Restricted to the pure
# limb long bones; the girdle bones (scapula/pelvis) and all axial bones keep the bbox map, which already
# places the shoulder/hip muscles that attach to them well. This is the SURGICAL set: the distal-limb
# muscles were the ones the whole-body bbox affine mislocated (the forearm hangs by the hip in this pose,
# so a global affine cannot put a forearm muscle on the forearm).
_BONE2SEG = {"humerus": ("fore", "stylopod"), "radius-ulna": ("fore", "zeugopod"), "manus": ("fore", "autopod"),
             "femur": ("hind", "stylopod"), "tibia-fibula": ("hind", "zeugopod"), "pes": ("hind", "autopod")}
_SEGR = {"stylopod": (0.0, 0.34), "zeugopod": (0.34, 0.67), "autopod": (0.67, 1.01)}


def _model_bone_point(limbs_model, bone, frac, side):
    """A point at fraction `frac` (proximal->distal) along the MODEL's own grown limb bone. Returns None
    for a non-limb bone or a missing limb, so the caller falls back to the bbox map for that endpoint."""
    seg = _BONE2SEG.get(bone)
    if seg is None or side not in ("R", "L"):
        return None
    lk = f"{seg[0]}-{side}"
    lm = limbs_model.get(lk)
    if lm is None:
        return None
    lo, hi = _SEGR[seg[1]]
    tgt = lo + float(np.clip(frac, 0.0, 1.0)) * (hi - lo)          # target pdn along the whole limb
    pdn, P = lm["pdn"], lm["P"]
    for bw in (0.06, 0.12, 0.20):                                  # widen the band until enough cells
        band = np.abs(pdn - tgt) < bw
        if band.sum() >= 5:
            return P[band].mean(0)
    return None


def carve_by_action_line(base, F, reg=None):
    """CT-carving by LINE OF ACTION -- the real cleavage rule. Each myotome / limb-myoblast cell cleaves to
    the muscle whose ORIGIN->INSERTION line it lies along (the Tcf4/Osr1 CT lays the cleavage planes exactly
    between adjacent action-lines). Unlike nearest-MIDPOINT (which clumps every muscle into a blob at its
    centre), nearest-SEGMENT makes each muscle SPAN from its origin bone to its insertion bone, and lets
    adjacent muscles (quadriceps/hamstrings/gluteal) partition a shared mass by which line each cell is on.

    Limb-bone endpoints are resolved on the MODEL's own grown bones (carve_bones) by name + PD fraction, so a
    forearm muscle attaches to the model's radius/ulna rather than to wherever a whole-body bbox affine sends
    the atlas point; axial and girdle endpoints keep the bbox map.
    `reg` = an atlas_raw->model registration (medic.model_atlas_registration.register); None -> bbox fallback.
    Returns (mus_cells, assign, muscles, O, I) with O/I = origin/insertion in the MODEL frame."""
    from menagerie.targets import reference_genome
    from menagerie.skeleton import build_skeleton
    from menagerie.muscles import build_muscles, MUSCLE_PLAN
    from medic.human_movie import _atlas_to_laid
    from medic.unified_embryo import FIDX
    g = reference_genome("human_male"); bones = build_skeleton(g); muscles = build_muscles(g, bones)
    if reg is not None:
        cg = reg                                                                   # fitted affine atlas->model
    else:
        A = np.vstack([_atlas_to_laid(np.vstack([b.a, b.b])) for b in bones])       # bbox fallback
        a_lo, a_sp = A.min(0), np.ptp(A, 0) + 1e-9
        m_lo, m_sp = base.min(0), np.ptp(base, 0) + 1e-9
        cg = lambda P: (_atlas_to_laid(np.atleast_2d(P)) - a_lo) / a_sp * m_sp + m_lo

    # the model's OWN limb skeleton (per-cell PD fraction + bone name), to attach limb muscles correctly
    limbs_model = carve_bones(base, F)
    plan = {nm: (ob, of, ib, ifr) for (nm, ob, of, ib, ifr, *_rest) in MUSCLE_PLAN}

    O, I = [], []
    for m in muscles:
        base_nm = m.name[:-2] if m.name[-2:] in (" R", " L") else m.name
        p = plan.get(base_nm)
        o_pt = i_pt = None
        if p is not None:
            o_pt = _model_bone_point(limbs_model, p[0], p[1], m.side)               # origin on the model bone
            i_pt = _model_bone_point(limbs_model, p[2], p[3], m.side)               # insertion on the model bone
        O.append(o_pt if o_pt is not None else cg(m.origin)[0])
        I.append(i_pt if i_pt is not None else cg(m.insertion)[0])
    O = np.array(O); I = np.array(I)
    mus = base[F == FIDX["Muscle"]] if "Muscle" in FIDX else base[:0]
    if not len(mus):
        return mus, np.array([], int), muscles, O, I
    seg = I - O; L2 = (seg ** 2).sum(1) + 1e-9
    diff = mus[:, None, :] - O[None]                                                # (Ncell, Nmus, 3)
    t = np.clip((diff * seg[None]).sum(-1) / L2[None], 0.0, 1.0)                    # projection along each action-line
    closest = O[None] + t[..., None] * seg[None]
    assign = ((mus[:, None, :] - closest) ** 2).sum(-1).argmin(1)                   # nearest action-LINE, not midpoint
    return mus, assign, muscles, O, I


def _validate(res):
    both = 0
    named = set()
    for r in res.values():
        d = set(r["dv"].tolist())
        both += int({"dorsal", "ventral"} <= d)
        named |= set(r["muscle"].tolist())
    fore = sorted({m for r in res.values() if r["kind"] == "fore" for m in set(r["muscle"].tolist())})
    hind = sorted({m for r in res.values() if r["kind"] == "hind" for m in set(r["muscle"].tolist())})
    return dict(limbs=len(res), dorsal_and_ventral=both, n_named_muscles=len(named),
                fore_muscles=fore, hind_muscles=hind)


def _figure(res):
    mus_all = sorted({m for r in res.values() for m in set(r["muscle"].tolist())})
    cmap = {m: i for i, m in enumerate(mus_all)}
    fig, ax = plt.subplots(1, 1, figsize=(8, 7), facecolor="#0d1017")
    ax.set_facecolor("#0d1017"); ax.set_aspect("equal"); ax.axis("off")
    for r in res.values():
        ax.scatter(r["P"][:, 0], r["P"][:, 2], s=9, c=[cmap[m] for m in r["muscle"]], cmap="tab20", alpha=0.85)
    v = _validate(res)
    ax.set_title(f"Limb muscle head: {v['n_named_muscles']} named limb muscles across {v['limbs']} limbs\n"
                 f"Lbx1 migratory mass carved by the limb CT (Tcf4) on the DV (Lmx1b/En1) x PD (Hox) frame\n"
                 f"dorsal = extensors · ventral = flexors · reads the limb bones, read-only",
                 color="#7dd3fc", fontsize=10)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/limb_muscle_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/limb_muscle_head.png")


def main():
    base, F = build_base()
    res = carve(base, F)
    v = _validate(res)
    print(f"{v['limbs']} limbs -> {v['n_named_muscles']} named limb muscles")
    print(f"  dorsal+ventral present: {v['dorsal_and_ventral']}/{v['limbs']} limbs (Lmx1b/En1 DV axis)")
    print(f"  fore muscles: {v['fore_muscles']}")
    print(f"  hind muscles: {v['hind_muscles']}")
    print("  genome-derived: precursors=Lbx1(migratory hypaxial), CT=Tcf4/Osr1, DV=Lmx1b/En1, PD=Hox")
    _figure(res)
    json.dump(v, open("data/organ_cascade/limb_muscle_head.json", "w"), indent=1)
    print("saved data/organ_cascade/limb_muscle_head.json")


if __name__ == "__main__":
    main()
