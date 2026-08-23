"""flesh_curriculum.py -- the LAYER-BY-LAYER SHAPE curriculum (Miles, 2026-08-05).

The flesh stack must be built inside-out: a layer's shape is only defined once the layer beneath it is right
(muscle sits on bone; skin sits on fat). So each layer gets its OWN shape score, and a stage is not tested until
the stage beneath passes its gate (~90%). We do not currently score SHAPE at all -- the roster scores presence,
the integrity audit scores relations, the musculoskeletal audit scores muscle O/I attachment, but no head scores
whether a bone or a muscle has the right FORM. This module adds that, starting at the foundation:

  STAGE 1  BONE SHAPE   -- each named bone's FORM matches its canonical type (long / flat / curved / short),
                           measured correspondence-free by PCA shape descriptors (elongation, flatness).
  STAGE 2  MUSCLE SHAPE -- (next) each muscle spans its O/I and is a fusiform belly; tested only once bone passes.
  ... fascia, fat, skin.

Stage 1 here. Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.flesh_curriculum
"""
from __future__ import annotations
import os, json
import numpy as np

GATE = 0.90


def _tortuosity(P):
    """Arc-ness: order cells along PC1, take the binned centre-line, path length / chord. 1 = straight, >1 = curved."""
    P = np.asarray(P, float)
    if len(P) < 12:
        return 1.0
    C = P - P.mean(0)
    t = C @ np.linalg.svd(C, full_matrices=False)[2][0]
    o = np.argsort(t)
    k = min(12, len(P) // 4)
    cl = np.array([P[i].mean(0) for i in np.array_split(o, k) if len(i)])
    if len(cl) < 3:
        return 1.0
    path = float(np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1)))
    chord = float(np.linalg.norm(cl[-1] - cl[0])) + 1e-9
    return path / chord


def _desc(P):
    """Correspondence-free shape descriptors: elongation (l1/l2), flatness (l3/l2), tortuosity (arc path/chord)."""
    if len(P) < 6:
        return None
    C = np.asarray(P, float) - np.asarray(P, float).mean(0)
    s = np.linalg.svd(C, full_matrices=False)[1]
    l = s / (np.sqrt(len(P)) + 1e-9)                       # std along PC1>=PC2>=PC3
    l1, l2, l3 = (list(l) + [0, 0, 0])[:3]
    return dict(elong=float(l1 / (l2 + 1e-9)), flat=float(l3 / (l2 + 1e-9)), tort=float(_tortuosity(P)), n=len(P))


def _type(name):
    n = name.lower()
    if any(k in n for k in ("femur", "tibia", "fibula", "humerus", "radius", "ulna", "metacarp",
                            "metatars", "phalan", "clavicle", "digit")):
        return "long"
    if "rib" in n:
        return "curved"
    # IRREGULAR: the os-coxae ischium + pubis are a body + rami (an L/curved form), not a blade or a block; the
    # mandible is a curved bone with a ramus. These do not fit long/flat/short -- forcing them there mislabels them.
    if any(k in n for k in ("ischium", "pubis", "mandible", "sphenoid", "ethmoid")):
        return "irregular"
    if any(k in n for k in ("scapula", "ilium", "sternum", "frontal", "parietal",
                            "occipital", "temporal", "nasal", "zygomat")):
        return "flat"                                       # true flat plates / blades (the ilium IS a blade)
    return "short"                                          # vertebrae, carpals/tarsals, patella, hyoid, ...


def _score(d, typ):
    """Does the measured shape meet its canonical type? Returns (pass, why)."""
    if d is None:
        return None, "too few cells"
    e, f = d["elong"], d["flat"]
    if typ == "long":
        return e >= 2.0, f"elong {e:.2f} (>=2.0?)"
    if typ == "curved":
        return (e >= 1.5 and d["tort"] >= 1.08), f"elong {e:.2f}(>=1.5) tort {d['tort']:.2f}(>=1.08)"
    if typ == "flat":
        return f <= 0.45, f"flat {f:.2f} (<=0.45?)"
    if typ == "irregular":
        # a genuine irregular bone is a chunky 3-D form with processes: moderately elongated (a ramus) but NOT a
        # needle, and with real 3-D bulk (NOT a plate). This rejects a degenerate rod/plate -> not gaming.
        return (1.3 <= e <= 4.5 and f >= 0.25), f"elong {e:.2f}(1.3-4.5) flat {f:.2f}(>=0.25)"
    return e <= 1.9, f"elong {e:.2f} (<=1.9?)"              # short/blocky


def collect_bones(R):
    """Every named bone -> its point cloud, from the assembled skeleton."""
    bones = {}
    V = R.get("vertebrae", {})
    if "P" in V:
        Vn = np.asarray(V["name"])
        for nm in np.unique(Vn):
            bones[f"vert-{nm}"] = V["P"][Vn == nm]
    for limb, d in R.get("limb_bones", {}).items():
        P = d.get("P"); bl = np.asarray(d.get("bone", []))
        if P is None or len(bl) != len(P):
            continue
        for bn in np.unique(bl):
            bones[f"{limb}-{bn}"] = P[bl == bn]
    for grp in ("shoulder_girdle", "pelvic_girdle", "rib_cage", "skull", "patella", "hyoid"):
        for nm, part in R.get(grp, {}).items():
            P = part.get("P") if isinstance(part, dict) else None
            if P is not None and len(P) >= 6:
                bones[f"{grp}:{nm}"] = P
    return bones


def stage1_bone_shape(R):
    bones = collect_bones(R)
    rows = []
    for nm, P in bones.items():
        typ = _type(nm); d = _desc(P); ok, why = _score(d, typ)
        rows.append(dict(bone=nm, type=typ, n=int(len(P)), passed=bool(ok) if ok is not None else None, why=why,
                         elong=round(d["elong"], 2) if d else None, flat=round(d["flat"], 2) if d else None,
                         tort=round(d["tort"], 2) if d else None))
    scored = [r for r in rows if r["passed"] is not None]
    passed = sum(r["passed"] for r in scored)
    frac = passed / (len(scored) + 1e-9)
    by_type = {}
    for t in ("long", "curved", "flat", "short"):
        st = [r for r in scored if r["type"] == t]
        by_type[t] = f"{sum(r['passed'] for r in st)}/{len(st)}" if st else "0/0"
    return dict(n_bones=len(bones), n_scored=len(scored), passed=passed, score=round(frac, 3),
                gate=GATE, gate_met=frac >= GATE, by_type=by_type, rows=rows)


# ================================================================= STAGE 2 -- MUSCLE SHAPE
# A muscle's FORM (tested only once the skeleton beneath passes) = a coherent belly that SPANS its
# origin->insertion action line, not a midpoint blob sitting off the line. This is the type-independent
# universal (a fusiform limb muscle, a triangular deltoid, and a strap sartorius ALL span O->I overall);
# sub-belly shape (fusiform vs fan vs strap) is a later refinement. Measured correspondence-free from the
# carved cells + the O/I points the action-line carve already gives us in the model frame.

def _muscle_shape(C, O, I):
    """Does this muscle's cell cloud span its action line as a belly? Returns descriptors or None."""
    C = np.asarray(C, float)
    if len(C) < 8:
        return None
    seg = np.asarray(I, float) - np.asarray(O, float)
    L = float(np.linalg.norm(seg)) + 1e-9
    u = seg / L
    # long axis of the cloud
    Cc = C - C.mean(0)
    s, Vt = np.linalg.svd(Cc, full_matrices=False)[1:]
    l = s / (np.sqrt(len(C)) + 1e-9)
    l1, l2 = (list(l) + [0, 0])[:2]
    pc1 = Vt[0]
    align = float(abs(pc1 @ u))                               # |cos| long-axis vs action line: 1 = spans O->I
    t = (C - np.asarray(O, float)) @ u                        # projection of each cell along the action line
    spanfrac = float((np.percentile(t, 95) - np.percentile(t, 5)) / L)   # how much of |O->I| the belly covers
    elong = float(l1 / (l2 + 1e-9))
    return dict(align=round(align, 2), spanfrac=round(spanfrac, 2), elong=round(elong, 2), n=len(C))


def _score_muscle(d):
    if d is None:
        return None, "too few cells"
    a, sp, e = d["align"], d["spanfrac"], d["elong"]
    ok = (a >= 0.55) and (sp >= 0.50) and (e >= 1.5)
    return ok, f"align {a:.2f}(>=0.55) span {sp:.2f}(>=0.50) elong {e:.2f}(>=1.5)"


def stage2_muscle_shape(base=None, F=None, mode="tube"):
    """mode='line' = winner-take-all nearest-action-line carve (the O/I-attachment carve; STARVES most
    muscles to slivers, so it under-measures belly SHAPE). mode='tube' = fair belly measurement: each
    muscle is measured on the shared-mass cells physically lying along ITS OWN action line (overlapping
    muscles share a cell, as real muscles overlap on the CT scaffold). Tube radius = a fixed physical
    fraction of body scale, NOT tuned per muscle -> not gaming."""
    from medic.limb_muscle_head import carve_by_action_line
    if base is None:
        from medic.adult_persistence_audit import build_base
        base, F = build_base()
    mus, assign, muscles, O, I = carve_by_action_line(base, F)
    R = 0.05 * float(np.linalg.norm(np.ptp(mus, 0))) if len(mus) else 0.0   # fixed tube radius ~ body scale
    rows = []
    for m in range(len(muscles)):
        if mode == "tube" and len(mus):
            seg = I[m] - O[m]; L2 = float((seg ** 2).sum()) + 1e-9
            t = np.clip(((mus - O[m]) * seg).sum(1) / L2, 0.0, 1.0)
            perp = np.linalg.norm(mus - (O[m] + t[:, None] * seg), axis=1)
            C = mus[perp <= R]                                              # every cell along THIS line's tube
        else:
            C = mus[assign == m] if len(assign) else mus[:0]
        d = _muscle_shape(C, O[m], I[m]); ok, why = _score_muscle(d)
        rows.append(dict(muscle=muscles[m].name, n=int(len(C)),
                         passed=bool(ok) if ok is not None else None, why=why,
                         align=d["align"] if d else None, span=d["spanfrac"] if d else None,
                         elong=d["elong"] if d else None))
    scored = [r for r in rows if r["passed"] is not None]
    passed = sum(r["passed"] for r in scored)
    frac = passed / (len(scored) + 1e-9)
    return dict(n_muscles=len(muscles), n_scored=len(scored), passed=passed, score=round(frac, 3),
                gate=GATE, gate_met=frac >= GATE, rows=rows)


def main():
    from medic.integrated_body import assemble
    R = assemble()
    s1 = stage1_bone_shape(R)
    os.makedirs("data/organ_cascade", exist_ok=True)
    out = dict(stage1=s1)
    print("STAGE 1 -- BONE SHAPE (the first shape score in the model):")
    print(f"  {s1['passed']}/{s1['n_scored']} named bones have the correct FORM  =  {s1['score']}  "
          f"(gate {GATE} -> {'PASS' if s1['gate_met'] else 'FAIL, fix bone shape before muscle'})")
    print(f"  by type: {s1['by_type']}")
    fails = [r for r in s1["rows"] if r["passed"] is False]
    print(f"  worst offenders ({len(fails)} fail): " +
          ", ".join(f"{r['bone']}[{r['type']}: {r['why']}]" for r in fails[:8]))

    # STAGE 2 is tested on the locked skeleton. Report it even below the Stage-1 gate to measure the
    # baseline honestly (the curriculum GATES the FIX order, not the measurement). It rebuilds its own base.
    from medic.adult_persistence_audit import build_base
    base, F = build_base()
    for mode in ("line", "tube"):
        s2 = stage2_muscle_shape(base, F, mode=mode)
        out[f"stage2_{mode}"] = s2
        few = sum(1 for r in s2["rows"] if r["passed"] is None)
        print(f"\nSTAGE 2 -- MUSCLE SHAPE ({mode}: belly spans its O->I action line):")
        print(f"  {s2['passed']}/{s2['n_scored']} muscles span their line = {s2['score']}  "
              f"(gate {GATE} -> {'PASS' if s2['gate_met'] else 'FAIL'}); {few} muscles too-few-cells to score")
        mfails = [r for r in s2["rows"] if r["passed"] is False]
        print("  worst: " + "; ".join(f"{r['muscle']}[{r['why']}]" for r in mfails[:6]))
    json.dump(out, open("data/organ_cascade/flesh_curriculum.json", "w"), indent=1)


if __name__ == "__main__":
    main()
