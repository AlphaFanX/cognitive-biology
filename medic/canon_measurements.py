"""
canon_measurements.py -- THE HIERARCHICAL CANON OF MEASUREMENTS (cycle 41, Miles's ask 2026-09-05:
"do we also have a detailed and hierarchical canon of measurements?" -- we did not; measured constants
were extracted ONE AT A TIME, cycle by cycle, and D2 is scale-blind + position-blind by construction,
which is how the 3x kidney and the pelvis-level liver hid from it).

ONE extractor over the canon sources already on disk -> data/canon/measurement_table.json:

  parts   : every canonical_map part (250: 163 BodyParts3D + 87 OpenAnatomy VTK). Shape stats for all
            (aspect l1:l2:l3, elong, flat); for bp3d parts -- which share ONE anatomical frame in real
            mm -- also ABSOLUTE size (span_mm, span/stature) and the IN-SITU ADDRESS (centroid as a
            fraction of the skin body frame: ap 0=caudal..1=cranial, ml signed, dv signed). OA meshes
            are per-atlas scan frames: shape-only, scale_comparable=False.
  hierarchy: the BodyParts3D PART-OF graph (shipped in obj_99.zip) -- each bp3d part records its FMA id
            + which OTHER wired parts are its descendants. FMA = the Foundational Model of Anatomy
            (UW), the formal is-a/part-of ontology bp3d meshes are keyed by.
  groups  : the adult in-situ organ groups from canon_reference.json (the curve's reference): span
            fraction, aspect, centroid address in the skin frame.
  stages  : the Carnegie/parametric ladder from canon_stages.json: per stage per labelled organ --
            cell fraction, span/stage-height, centroid address in the stage skin frame.
  ratios  : canonical PROPORTION ratios computed from the parts table (femur/tibia, humerus/ulna,
            organ/stature...). The Marquardt/phi FACE layer is the designed extension -- it needs
            canon face landmarks, queued.

THE GWAS LINK (the design intent): every table row is a MEASURABLE PHENOTYPE -- exactly the coordinate
GWAS effect sizes are defined on (the Xiong C-GWAS face landmarks in Paper #7 are landmark
distances/ratios). The canon value = the population MEAN of the measurement; GWAS betas = per-genome
offsets around it (mean + sum beta*dosage). The table is the join key that reactivates the variation
axis (adapter_table.json / C-GWAS) the fidelity loop has not been using.

Run:    venv_win_new/Scripts/python.exe -m medic.canon_measurements            (extract the table)
        venv_win_new/Scripts/python.exe -m medic.canon_measurements --check    (measure the standing
            specimen against the table -> the model-vs-canon instrument, worst deltas first)
Out:    data/canon/measurement_table.json
"""
from __future__ import annotations
import argparse, json, os, zipfile
import numpy as np

from medic.canonical_atlas import _vtk_points, load_bp3d, _bp3d, MDIR

OUT = "data/canon/measurement_table.json"
MAP = "data/canonical_map.json"
SKIN_FMA = "FMA7163"                     # BodyParts3D whole-body skin = the shared body frame + stature


def _shape(P):
    """Correspondence-free shape stats: principal stds l1>=l2>=l3, elong, flat, span along PC1."""
    P = np.asarray(P, float)
    if len(P) < 6:
        return None
    C = P - P.mean(0)
    s = np.linalg.svd(C, full_matrices=False)[1] / (np.sqrt(len(P)) + 1e-9)
    l1, l2, l3 = (list(s) + [0, 0, 0])[:3]
    span = float(np.ptp(C @ np.linalg.svd(C, full_matrices=False)[2][0]))
    return dict(l=[round(float(x), 5) for x in (l1, l2, l3)],
                elong=round(float(l1 / (l2 + 1e-9)), 3), flat=round(float(l3 / (l2 + 1e-9)), 3),
                span=span)


class BodyFrame:
    """The bp3d skin's own principal frame: axis 0 = cranio-caudal (oriented so the BRAIN end reads 1),
    axis 1 = the wider of the remaining extents (ML on a standing body), axis 2 = DV. Addresses are
    centroid fractions of the skin extent along each axis -- the same convention the model's standing
    registers use (fractions of stature)."""

    def __init__(self, skin, brain=None):
        C = skin - skin.mean(0)
        Vt = np.linalg.svd(C[np.random.default_rng(0).choice(len(C), min(len(C), 20000), replace=False)],
                           full_matrices=False)[2]
        self.origin = skin.mean(0)
        proj = C @ Vt.T
        ext = np.ptp(proj, 0)
        order = [0, 1, 2] if ext[1] >= ext[2] else [0, 2, 1]
        self.axes = Vt[order]
        self.lo = proj[:, order].min(0)
        self.ext = ext[order]
        if brain is not None and len(brain):
            bf = ((brain.mean(0) - self.origin) @ self.axes[0] - self.lo[0]) / (self.ext[0] + 1e-9)
            if bf < 0.5:                                     # orient cranio-caudal axis so brain -> 1
                self.axes[0] = -self.axes[0]
                p = C @ self.axes[0]
                self.lo[0] = p.min()
        self.stature = float(self.ext[0])

    def address(self, P):
        c = np.asarray(P, float).mean(0) - self.origin
        p = c @ self.axes.T
        ap = float((p[0] - self.lo[0]) / (self.ext[0] + 1e-9))
        ml = float(p[1] / (self.ext[1] / 2 + 1e-9))          # signed, ~[-1,1] of the half-width
        dv = float(p[2] / (self.ext[2] / 2 + 1e-9))
        return dict(ap=round(ap, 4), ml=round(ml, 4), dv=round(dv, 4))


def _load_spec(spec):
    if spec[0] == "oa":
        zf, member = spec[1], spec[2]
        return _vtk_points(zipfile.ZipFile(os.path.join(MDIR, zf)).read(member))
    if isinstance(spec[1], (list, tuple)):               # union of declared subtrees (organ__Gut)
        return np.vstack([load_bp3d(r) for r in spec[1]])
    return load_bp3d(spec[1])


# THE BOUNDARY DECLARATIONS (cycle 43, Miles's rule: every reference must declare its object; a
# reference without an ontology key is bound by a name string with an undeclared boundary and is
# unauditable). Every surviving OA row gets its declaration; MISMATCH entries are known boundary
# faults kept only until a properly-bounded reference exists.
OA_BOUNDARY = {
    "rib_cage__rib": "single mirrored mesh serves both sides (D2 is reflection-invariant)",
    "vertebra__S": "MISMATCH: whole-sacrum mesh referenced by a single sacral vertebra "
                   "(the atlas ships the fused bone only)",
    "skull____whole__": "composite skull by design (scored as the whole braincase)",
    "muscle__erector_spinae": "right-side mesh for the unsided composite part",
    "vertebra__": "single vertebra mesh, clean boundary",
}


def _boundary(key):
    for prefix, note in OA_BOUNDARY.items():
        if key.startswith(prefix):
            return note
    return "UNDECLARED (name-string match only)"


def _descendants(root):
    """All FMA ids under a bp3d root via the shipped part-of graph."""
    S = _bp3d()
    seen, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(S["edges"].get(n, ()))
    seen.discard(root)
    return seen


def extract():
    wired = json.load(open(MAP))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    skin = load_bp3d(SKIN_FMA)
    brain = load_bp3d("FMA50801")                            # brain composite orients the cranial end
    frame = BodyFrame(skin, brain)
    print(f"body frame: stature {frame.stature:.0f} mm (bp3d skin {SKIN_FMA}), "
          f"ext ML {frame.ext[1]:.0f} DV {frame.ext[2]:.0f}")

    parts, fma_of = {}, {}
    for key, spec in sorted(wired.items()):
        try:
            P = _load_spec(spec)
        except Exception as e:                               # a missing member is recorded, never fatal
            parts[key] = dict(error=str(e)[:80], source=spec[0]); continue
        sh = _shape(P)
        if sh is None:
            parts[key] = dict(error="degenerate", source=spec[0]); continue
        row = dict(source=spec[0], ref=spec[1] if spec[0] == "bp3d" else spec[2],
                   n_pts=int(len(P)), l=sh["l"], elong=sh["elong"], flat=sh["flat"],
                   scale_comparable=spec[0] == "bp3d")
        if spec[0] == "bp3d":
            row["fma"] = spec[1]
            row["span_mm"] = round(sh["span"], 1)
            row["span_frac"] = round(sh["span"] / frame.stature, 4)
            row["address"] = frame.address(P)
            for f in (spec[1] if isinstance(spec[1], (list, tuple)) else [spec[1]]):
                fma_of[f] = key
        else:
            row["boundary"] = _boundary(key)
        parts[key] = row
    # hierarchy: which wired parts sit UNDER which (the FMA part-of graph, resolved to table rows)
    for key, row in parts.items():
        fma = row.get("fma")
        for f in (fma if isinstance(fma, (list, tuple)) else [fma]) if fma else []:
            kids = sorted(fma_of[d] for d in _descendants(f) if d in fma_of and fma_of[d] != key)
            if kids:
                row.setdefault("wired_descendants", []).extend(k for k in kids
                                                              if k not in row.get("wired_descendants", []))

    # adult in-situ organ groups (the curve's reference)
    groups = {}
    ad = json.load(open("data/movie/canon_reference.json"))
    gskin = np.asarray(ad["skin"], float).reshape(-1, 3)
    gframe = BodyFrame(gskin, next((np.asarray(o["xyz"], float).reshape(-1, 3)
                                    for o in ad["organs"] if o["label"] == "brain"), None))
    for o in ad["organs"]:
        P = np.asarray(o["xyz"], float).reshape(-1, 3)
        sh = _shape(P)
        if sh:
            groups[o["label"]] = dict(n_pts=len(P), l=sh["l"], elong=sh["elong"], flat=sh["flat"],
                                      span_frac=round(sh["span"] / gframe.stature, 4),
                                      address=gframe.address(P))

    # the Carnegie / parametric ladder
    stages = {}
    cs = json.load(open("data/movie/canon_stages.json"))
    for st in cs["stages"]:
        sk = np.asarray(st["skin"], float).reshape(-1, 3)
        if len(sk) < 100:
            continue
        sframe = BodyFrame(sk, next((np.asarray(o["xyz"], float).reshape(-1, 3) for o in st["organs"]
                                     if "brain" in str(o.get("label", "")).lower()), None))
        total = sum(len(o["xyz"]) // 3 for o in st["organs"]) or 1
        srow = {}
        for o in st["organs"]:
            P = np.asarray(o["xyz"], float).reshape(-1, 3)
            sh = _shape(P)
            if sh:
                srow[o["label"]] = dict(frac=round(len(P) / total, 4),
                                        span_frac=round(sh["span"] / (sframe.stature + 1e-9), 4),
                                        elong=sh["elong"], flat=sh["flat"], address=sframe.address(P))
        stages[st["label"]] = dict(cs=st.get("cs"), organs=srow)

    # canonical proportion RATIOS (bp3d scale-comparable rows only). The Marquardt/phi face layer is
    # the designed extension (needs canon face landmarks).
    def _sp(k):
        return parts.get(k, {}).get("span_mm")
    ratios = {}
    for name, a, b in (("femur/tibia", "hind-R__femur", "hind-R__tibia"),
                       ("femur/stature", "hind-R__femur", None),
                       ("humerus/ulna", "fore-R__humerus", "fore-R__ulna"),
                       ("humerus/stature", "fore-R__humerus", None)):
        sa = _sp(a); sb = _sp(b) if b else frame.stature
        if sa and sb:
            ratios[name] = round(sa / sb, 4)
    for org in ("organ__Heart", "organ__Kidney", "organ__Liver", "organ__Spleen"):
        s = _sp(org)
        if s:
            ratios[org.split("__")[1].lower() + "/stature"] = round(s / frame.stature, 4)

    # THE FMA ROSTER (cycle 44, Miles's climb): EVERY meshed part in the archive becomes a measured row
    # -- the full 934, each with its official name, absolute size, in-situ address and its PARENTS in
    # the part-of graph (rows roll up the tree). The coverage join then says honestly which parts of
    # the ontology the model does not yet grow or score -- the long-term worklist, derived not curated.
    from medic.canonical_atlas import _bp3d, _obj_verts_raw
    from collections import defaultdict
    S = _bp3d()
    fparents = defaultdict(set)
    for pa, kids in S["edges"].items():
        for k in kids:
            fparents[k].add(pa)
    fnames = _fma_names()
    roster = {}
    for fid, member in sorted(S["members"].items()):
        Pf = _obj_verts_raw(S["zf"].read(member))
        shf = _shape(Pf)
        if shf is None:
            continue
        roster[fid] = dict(name=fnames.get(fid, "?"), n_pts=int(len(Pf)),
                           span_mm=round(shf["span"], 1),
                           span_frac=round(shf["span"] / frame.stature, 4),
                           elong=shf["elong"], flat=shf["flat"], address=frame.address(Pf),
                           parents=sorted(fparents.get(fid, ())))
    # coverage: an FMA id is COVERED if a wired model part references it directly or reaches it as a
    # leaf of its declared subtree(s)
    covered = set()
    for key, spec in wired.items():
        if spec[0] != "bp3d":
            continue
        for root in (spec[1] if isinstance(spec[1], (list, tuple)) else [spec[1]]):
            covered.add(root)
            covered |= {d for d in _descendants(root) if d in roster}
    for fid in roster:
        roster[fid]["covered"] = fid in covered
    ncov = sum(1 for r in roster.values() if r["covered"])
    print(f"FMA roster: {len(roster)} meshed parts measured; covered by the wired model roster: "
          f"{ncov} ({100 * ncov // max(1, len(roster))}%), NOT yet scored: {len(roster) - ncov}")

    table = dict(meta=dict(
        extracted="2026-09-05", stature_mm=round(frame.stature, 1),
        sources=["BodyParts3D (CC-BY-SA, real mm, one body frame + FMA part-of graph)",
                 "OpenAnatomy/SPL VTK (shape-only, per-atlas frames)",
                 "canon_reference.json (adult in-situ groups)",
                 "canon_stages.json (Amsterdam Carnegie + parametric ladder)"],
        frame="ap: 0=caudal 1=cranial (brain-oriented); ml,dv signed fractions of the half-extent",
        gwas_link="each row = a measurable phenotype; canon value = population mean; GWAS betas "
                  "(adapter_table.json / Xiong C-GWAS) = per-genome offsets -- the designed join",
        face_layer="Marquardt/phi landmark ratios = queued extension (needs canon face landmarks)"),
        parts=parts, groups=groups, stages=stages, ratios=ratios, fma_roster=roster)
    json.dump(table, open(OUT, "w"), indent=1)
    ok = sum(1 for r in parts.values() if "error" not in r)
    print(f"MEASUREMENT TABLE -> {OUT}: {ok}/{len(parts)} parts "
          f"({sum(1 for r in parts.values() if r.get('scale_comparable'))} scale-comparable bp3d), "
          f"{len(groups)} adult groups, {len(stages)} stages, {len(ratios)} ratios, "
          f"fma_roster {len(roster)}")
    return table


# ------------------------------------------------------------------ the model-vs-canon instrument
def check(table=None):
    """Measure the STANDING specimen against the table: per-part span_frac + elong deltas (bp3d rows),
    per-group address deltas. The instrument D2 cannot be: scale- and place-sensitive."""
    from medic.integrated_body import assemble, mature_parts
    from medic.grays_scorecard import collect_all_parts
    if table is None:
        table = json.load(open(OUT))
    M = mature_parts(assemble())
    base = np.asarray(M["base"], float)
    stat = float(np.ptp(base[:, 0]))                        # standing body: x = crown->sole
    rows = []
    for name, cat, P, meta in collect_all_parts(M):
        key = name.replace(":", "__").replace(" ", "_")
        t = table["parts"].get(key)
        if not t or "error" in t or not t.get("scale_comparable"):
            continue
        sh = _shape(P)
        if sh is None:
            continue
        sf = sh["span"] / stat
        rows.append((key, round(sf, 4), t["span_frac"], round(sf / (t["span_frac"] + 1e-9), 2),
                     sh["elong"], t["elong"]))
    rows.sort(key=lambda r: abs(np.log(max(r[3], 1e-3))), reverse=True)
    print(f"\nMODEL vs CANON MEASUREMENTS ({len(rows)} scale-comparable parts; worst size ratios first)")
    print(f"{'part':38s} {'model':>7s} {'canon':>7s} {'ratio':>6s}   {'elong m/c'}")
    for k, sf, tf, ratio, em, ec in rows[:20]:
        print(f"{k:38s} {sf:7.4f} {tf:7.4f} {ratio:6.2f}   {em:.2f}/{ec:.2f}")
    within = sum(1 for r in rows if 0.67 <= r[3] <= 1.5)
    print(f"\nwithin 1.5x of canon size: {within}/{len(rows)}")
    return rows


# ------------------------------------------------------------------ the canonical-map audit
def _fma_names():
    """id -> official English name, from the bp3d release name index + the graph files' name columns."""
    names = {}
    root = "data/bodyparts3d"
    for ln in open(os.path.join(root, "parts_list_e.txt"), encoding="utf-8", errors="ignore").read().splitlines()[1:]:
        t = ln.split("\t")
        if len(t) >= 2:
            names[t[0]] = t[1]
    for f in ("conventional_part_of.txt", "composite_parts.txt"):
        for ln in open(os.path.join(root, f), encoding="utf-8", errors="ignore").read().splitlines()[1:]:
            t = ln.split("\t")
            if len(t) >= 4:
                names.setdefault(t[0], t[1]); names.setdefault(t[2], t[3])
    return names


_STOP = {"of", "the", "and", "left", "right", "l", "r", "bone", "muscle", "first", "second", "third",
         "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"}


def _words(s):
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if w not in _STOP}


def audit():
    """The wrong-mesh detector (cycle 42): for every bp3d canonical_map entry, compare the map's own label
    against the OFFICIAL FMA name of the mapped id, and flag implausible canon sizes. A wired reference that
    is the wrong object poisons its part's canonical-tier score from the reference side -- the anconeus /
    supinator / spinal-cord floor suspects."""
    wired = json.load(open(MAP))
    names = _fma_names()
    table = json.load(open(OUT)) if os.path.exists(OUT) else {"parts": {}}
    mism, unnamed = [], []
    for key, spec in sorted(wired.items()):
        if spec[0] != "bp3d":
            continue
        fma, label = spec[1], spec[2]
        if isinstance(fma, (list, tuple)):
            continue                       # declared union override -- boundary stated in REF_OVERRIDE
        official = names.get(fma)
        sf = table["parts"].get(key, {}).get("span_frac")
        if official is None:
            unnamed.append((key, fma, label, sf)); continue
        if not (_words(label) & _words(official)):
            mism.append((key, fma, label, official, sf))
    print(f"CANONICAL-MAP AUDIT: {sum(1 for s in wired.values() if s[0]=='bp3d')} bp3d entries, "
          f"{len(mism)} label-vs-FMA-name mismatches, {len(unnamed)} ids with no name record")
    if mism:
        print(f"\n{'part':38s} {'FMA id':10s} {'map label':22s} {'OFFICIAL FMA name':34s} {'span/stat'}")
        for key, fma, label, official, sf in mism:
            print(f"{key:38s} {fma:10s} {label:22s} {official[:34]:34s} {sf if sf is not None else '-'}")
    for key, fma, label, sf in unnamed:
        print(f"  NO NAME: {key} {fma} ({label}) span_frac {sf}")
    return mism


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    if a.audit:
        audit()
    else:
        t = extract() if not os.path.exists(OUT) or not a.check else None
        if a.check:
            check(t)
