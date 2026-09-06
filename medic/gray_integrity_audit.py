"""
gray_integrity_audit.py -- the NESTED-BODY integrity check: do the parts FIT TOGETHER as Gray's says, not just
exist? The roster audit (medic.gray_roster_audit) checks presence and counts; this checks RELATIONS -- the
containment nesting and the articulations -- because a body with the right parts in the wrong arrangement is not a
body. Gray's Anatomy is fundamentally a book of relations, so the ground truth here is fit, not roster.

Two families of relation are checked against Gray's, each as a fraction that should be high:

  CONTAINMENT (the nested body): the spinal cord runs inside the vertebral canal; the heart and lungs sit inside
  the thoracic cage; the abdominal viscera lie inferior to the thorax, below the diaphragm; the whole skeleton is
  one connected piece.

  ARTICULATION (the parts fit): each rib reaches back to a thoracic vertebra and the true ribs reach forward to
  the sternum; the shoulder girdle bridges the axial skeleton to the arm (scapula to the humerus head); the pelvic
  girdle bridges the sacrum to the leg (acetabulum to the femur head); the skull sits atop the cervical column.

READ-ONLY. Distances are judged near at a tissue-conglomerate tolerance (a fraction of body height), the
resolution the model works at. Run at a modest ne for speed.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.gray_integrity_audit [--ne 60000]
Out: data/organ_cascade/gray_integrity_audit.json  (+ printed scorecard)
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic import rib_cage_head as RC
from medic import skull_head as SKU
from medic import shoulder_girdle_head as SG
from medic import pelvic_girdle_head as PG
from medic import limb_chondrogenesis_head as LC
from medic import part_resolved_anatomy as PR
from medic import adipose_head as ADI
from medic import fascia_head as FAS

TOL = 0.09           # "near" tolerance, as a fraction of body height H


def _cells(base, F, fates):
    from medic.subhead_program import expand_names
    ids = [FIDX[n] for n in expand_names(fates) if n in FIDX]  # a parent resolves to its sub-head children too
    return base[np.isin(F, ids)] if ids else base[:0]


def _near_frac(A, B, H):
    """fraction of points in A whose nearest point in B is within TOL*H (A is contained by / attached to B)."""
    if len(A) == 0 or len(B) == 0:
        return 0.0
    d, _ = cKDTree(B).query(A)
    return float(np.mean(d < TOL * H))


def _min_dist(A, B, H):
    if len(A) == 0 or len(B) == 0:
        return 9.9
    d, _ = cKDTree(B).query(A)
    return float(d.min() / H)


def audit(base, F):
    H = float(np.ptp(base[:, 0])) + 1e-9
    rows = []

    def row(name, val, ok, detail=""):
        rows.append(dict(relation=name, value=round(float(val), 3), ok=bool(ok), detail=detail))

    # parts
    rc = RC.build(base, F)["parts"]
    ribs = {k: p for k, p in rc.items() if p.get("part") == "rib"}
    stern = np.vstack([p["P"] for k, p in rc.items() if p.get("part") == "sternum"]) if any(
        p.get("part") == "sternum" for p in rc.values()) else base[:0]
    sk = SKU.build(base, F)["parts"]
    skull_pts = np.vstack([p["P"] for p in sk.values()]) if sk else base[:0]
    sg = SG.build(base, F)["parts"]; pg = PG.build(base, F)["parts"]
    lb = LC.carve(base, F)
    try:
        V, Vname, *_ = PR.complete_column(base, F, np.random.default_rng(0))
        Vname = [str(n) for n in Vname]
        thoracic = V[np.array([n.startswith("T") for n in Vname])] if len(V) else base[:0]
        cervical = V[np.array([n.startswith("C") for n in Vname])] if len(V) else base[:0]
    except Exception:
        V, thoracic, cervical = base[:0], base[:0], base[:0]

    # organ / cord cell sets
    cord = _cells(base, F, ["Nervous System", "Spinal Cord", "Neural tube"])
    heart = _cells(base, F, ["Heart", "Atrium", "Ventricle", "Outflow"])
    lung = _cells(base, F, ["Lung"])
    abd = _cells(base, F, ["Liver", "Kidney", "Gut", "Foregut", "Hindgut"])

    # ---- CONTAINMENT ---------------------------------------------------------------------------------
    if len(cord) and len(V):
        row("spinal cord inside the vertebral canal", _near_frac(cord, V, H), _near_frac(cord, V, H) > 0.6,
            "cord threads the column")
    if len(heart) and len(thoracic):
        lo, hi = thoracic[:, 0].min(), thoracic[:, 0].max()
        fr = float(np.mean((heart[:, 0] >= lo - 0.1 * H) & (heart[:, 0] <= hi + 0.1 * H)))
        row("heart inside the thoracic cage", fr, fr > 0.6, "AP within the thoracic vertebral band")
    if len(lung) and len(thoracic):
        lo, hi = thoracic[:, 0].min(), thoracic[:, 0].max()
        fr = float(np.mean((lung[:, 0] >= lo - 0.1 * H) & (lung[:, 0] <= hi + 0.1 * H)))
        row("lungs inside the thoracic cage", fr, fr > 0.5, "AP within the thoracic band")
    if len(abd) and len(heart):
        # abdominal viscera should sit INFERIOR to the heart (lower AP, head-forward => cranial is high AP)
        fr = float(np.mean(abd[:, 0] < np.median(heart[:, 0])))
        row("abdominal viscera inferior to the thorax", fr, fr > 0.6, "caudal to the heart (below diaphragm)")

    # ---- ARTICULATION --------------------------------------------------------------------------------
    if ribs and len(thoracic):
        # each rib's DORSAL end should touch a thoracic vertebra. Pick it as the most-dorsal point: for a TRUE rib
        # the arc returns to the midline at the sternal end too, so z~0 is ambiguous -- disambiguate by DV side.
        # The ventral reference is the STERNUM (a bone), not the whole-body DV mean: when the viscera are placed
        # with a proper dorso-ventral spread the body mean rises toward the spine and would flip this sign, a false
        # negative -- the sternum is unaffected by viscera placement. Fall back to the body mean only if no sternum.
        ref = float(stern[:, 1].mean()) if len(stern) else float(base[:, 1].mean())
        dsn = float(np.sign(thoracic[:, 1].mean() - ref)) or 1.0
        near = []
        for p in ribs.values():
            P = p["P"]; dorsal = P[np.argmax(dsn * P[:, 1])]         # the end on the vertebral (dorsal) side
            near.append(_min_dist(dorsal[None], thoracic, H) < TOL)
        fr = float(np.mean(near))
        row("ribs articulate the thoracic vertebrae", fr, fr > 0.6, f"{sum(near)}/{len(near)} ribs reach a T-vertebra")
    if ribs and len(stern):
        true_ribs = [p for k, p in ribs.items() if p.get("rib", 99) <= 7]
        near = [_min_dist(p["P"], stern, H) < 1.5 * TOL for p in true_ribs]
        fr = float(np.mean(near)) if near else 0.0
        row("true ribs (1-7) reach the sternum", fr, fr > 0.5, f"{sum(near)}/{len(near)} true ribs reach the sternum")
    # shoulder girdle -> humerus (glenohumeral)
    scap = np.vstack([p["P"] for k, p in sg.items() if "scapula" in k]) if any("scapula" in k for k in sg) else base[:0]
    hum = np.vstack([v["P"] for k, v in lb.items() if v.get("kind") == "fore"]) if lb else base[:0]
    if len(scap) and len(hum):
        d = _min_dist(scap, hum, H)
        row("shoulder girdle meets the arm (scapula-humerus)", d, d < 2 * TOL, "glenohumeral gap")
    # pelvic girdle -> femur (acetabulum) and -> sacrum
    acet = np.vstack([p["P"] for k, p in pg.items() if "acetabulum" in k or "ilium" in k]) if any(
        ("acetabulum" in k or "ilium" in k) for k in pg) else base[:0]
    fem = np.vstack([v["P"] for k, v in lb.items() if v.get("kind") == "hind"]) if lb else base[:0]
    if len(acet) and len(fem):
        d = _min_dist(acet, fem, H)
        row("pelvic girdle meets the leg (acetabulum-femur)", d, d < 2 * TOL, "hip-joint gap")
    # skull atop the cervical column
    if len(skull_pts) and len(cervical):
        d = _min_dist(skull_pts, cervical, H)
        row("skull sits atop the cervical column", d, d < 2.5 * TOL, "cranio-cervical gap")

    # ---- FLESH (muscle / fat / collagen fit) ---------------------------------------------------------
    mus = _cells(base, F, ["Muscle"])
    bone_c = _cells(base, F, ["Cartilage"])
    # per-AP mean radius (fraction of the local surface radius) of a tissue about the body centre-line
    ap = base[:, 0]; x0 = ap.min()

    def _mean_r(P, nb=18):
        if len(P) < 6:
            return None
        rs = []
        for i in range(nb):
            lo = x0 + i / nb * H
            sl = P[(P[:, 0] >= lo) & (P[:, 0] < lo + H / nb)]
            bl = base[(ap >= lo) & (ap < lo + H / nb)]
            if len(sl) < 2 or len(bl) < 4:
                continue
            c = bl[:, 1:].mean(0)
            surf = np.percentile(np.linalg.norm(bl[:, 1:] - c, axis=1), 92) + 1e-9
            rs.append(np.median(np.linalg.norm(sl[:, 1:] - c, axis=1)) / surf)
        return float(np.mean(rs)) if rs else None

    adi = ADI.build(base, F)
    r_fat, r_mus, r_bone = _mean_r(adi["subcutaneous"]), _mean_r(mus), _mean_r(bone_c)
    if r_fat is not None and r_mus is not None:
        row("fat is the outer flesh layer (bone/muscle inside, fat outside)", r_fat - r_mus,
            r_fat > r_mus and (r_bone is None or r_fat > r_bone), f"r_fat={r_fat:.2f} r_mus={r_mus:.2f} r_bone={r_bone if r_bone is None else round(r_bone,2)}")
    # muscles attach to the skeleton (not floating flesh): fraction of muscle cells near a bone
    allbone = np.vstack([x for x in (bone_c, V, np.vstack([p["P"] for p in ribs.values()]) if ribs else base[:0],
                                     np.vstack([v["P"] for v in lb.values()]) if lb else base[:0]) if len(x)]) if len(bone_c) or len(V) else base[:0]
    if len(mus) and len(allbone):
        fr = _near_frac(mus, allbone, H)
        row("muscles attach to the skeleton (origin/insertion)", fr, fr > 0.6, "fraction of muscle near a bone")
    # ligaments tie bone to bone (from the fascia head)
    fas = FAS.build(base, F)
    lig = fas["ligaments"]
    spans2 = sum(1 for l in lig if l["a"] != l["b"])
    row("ligaments tie bone to bone", spans2, len(lig) > 0 and spans2 == len(lig), f"{spans2}/{len(lig)} span two distinct bones")

    # ---- ONE BODY ------------------------------------------------------------------------------------
    try:
        girdle_pts = [p["P"] for p in sg.values()] + [p["P"] for p in pg.values()]   # the axial<->limb bridges
        S = np.vstack([x for x in ([skull_pts, V, stern] + [p["P"] for p in ribs.values()] +
                                   girdle_pts + [v["P"] for v in lb.values()]) if len(x)])
        if len(S) > 20:
            S = S[np.random.default_rng(0).choice(len(S), min(len(S), 2000), replace=False)]
            tree = cKDTree(S)
            # union-find over kNN edges within TOL*H
            parent = list(range(len(S)))
            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]; a = parent[a]
                return a
            for i, nbrs in enumerate(tree.query_ball_point(S, TOL * H)):
                for j in nbrs:
                    parent[find(i)] = find(j)
            comps = len({find(i) for i in range(len(S))})
            row("skeleton is one connected body", comps, comps <= 3, f"{comps} components (<=3 ok at this res)")
    except Exception as e:
        row("skeleton is one connected body", 99, False, str(e)[:30])

    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ne", type=int, default=60000); args = ap.parse_args()
    base, F = build_base(args.ne)
    rows = audit(base, F)
    npass = sum(r["ok"] for r in rows)
    print(f"GRAY'S NESTED-BODY INTEGRITY -- {npass}/{len(rows)} relations hold ({npass/len(rows):.0%})")
    for r in rows:
        print(f"  [{'OK ' if r['ok'] else 'XX '}] {r['relation']:46s} {r['value']:<7} {r['detail']}")
    json.dump(dict(pass_fraction=round(npass / len(rows), 3), n_pass=npass, n=len(rows), rows=rows),
              open("data/organ_cascade/gray_integrity_audit.json", "w"), indent=1, default=str)
    print("saved data/organ_cascade/gray_integrity_audit.json")


if __name__ == "__main__":
    main()
