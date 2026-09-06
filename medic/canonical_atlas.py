"""canonical_atlas.py -- score our generated organ/bone shapes against ISOLATED canonical human meshes.

Gray's is 2D and shows structures in situ; a better target is an ISOLATED 3D mesh per structure. This pulls the
OpenAnatomy / SPL (Harvard) CT/MRI-derived models (skull, mandible from the head-neck atlas; colon+ileum "gut"
from the abdomen atlas -- CC / Slicer licence) and scores our model's part against the real geometry, correspondence
-free (no registration, no point matching):
  * D2 SHAPE DISTRIBUTION (Osada 2002): histogram of normalised pairwise point distances -- rotation / translation
    / scale invariant. similarity = histogram intersection in [0,1] -> the headline shape-match %.
  * DESCRIPTOR TABLE: PCA elongation / flatness / sphericity / hollowness / tortuosity, canonical vs ours, so we see
    WHERE the shape is off (e.g. a solid vs hollow braincase, a straight vs coiled gut).

Meshes cache under data/canonical_meshes/ (zips already downloaded). Run:
  cd cognimed && venv_win_new/Scripts/python.exe -m medic.canonical_atlas
Out: data/organ_cascade/canonical_atlas.{png,json}
"""
from __future__ import annotations
import os, io, json, zipfile
import numpy as np

from medic.grays_scorecard import sphericity, hollowness, _tortuosity
from medic.flesh_curriculum import _desc

MDIR = "data/canonical_meshes"
BP3D_DIR = "data/bodyparts3d"                          # DBCLS BodyParts3D (CC-BY-SA): obj_99.zip + part-of graph
# structure -> spec. TWO sources:
#   ("oa", zip, member)  = isolated OpenAnatomy/SPL VTK mesh (one scan frame, x=L-R y=S-I z=A-P)
#   ("bp3d", root_id)    = BodyParts3D composite -> union of every obj primitive under it (heart/lung/brain are
#                          composites with no single file). D2/descriptors are frame-invariant; orientation for
#                          bp3d parts is left "n/a" (its frame differs from OpenAnatomy -- calibrate at stage-2).
CANON = {
    # --- skeleton (head-neck atlas) ---
    "skull":    ("oa", "_headneck.zip", "head-neck-2016-09/models/Model_10_skull.vtk"),
    "mandible": ("oa", "_headneck.zip", "head-neck-2016-09/models/Model_25_mandible.vtk"),
    # --- viscera (abdomen atlas) ---
    "gut":      ("oa", "_abdomen.zip",  "abdomen-2016-09/Data/Model_9_colon_and_ileum.vtk"),   # colon+ileum = coiled distal gut
    "stomach":  ("oa", "_abdomen.zip",  "abdomen-2016-09/Data/Model_13_stomach_and_duodenum.vtk"),
    "liver":    ("oa", "_abdomen.zip",  "abdomen-2016-09/Data/Model_3_liver.vtk"),
    "spleen":   ("oa", "_abdomen.zip",  "abdomen-2016-09/Data/Model_4_spleen.vtk"),
    "kidney":   ("oa", "_abdomen.zip",  "abdomen-2016-09/Data/Model_51_right_kidney.vtk"),
    "pancreas": ("oa", "_abdomen.zip",  "abdomen-2016-09/Data/Model_6_pancreas.vtk"),
    "adrenal":  ("oa", "_abdomen.zip",  "abdomen-2016-09/Data/Model_111_right_adrenal_gland.vtk"),
    # --- thorax + neural (BodyParts3D composites; OpenAnatomy has no downloadable thorax bundle) ---
    "heart":    ("bp3d", "FMA7088"),     # 23 sub-part primitives (chambers/walls/great-vessel roots)
    "lung":     ("bp3d", "FMA7195"),     # 5 lobes (R upper/mid/lower + L upper/lower) = both lungs
    "brain":    ("bp3d", "FMA50801"),    # 74 region primitives = whole brain
    "eye":      ("bp3d", "FMA12513"),    # eyeball
}


def _vtk_points(raw):
    """Vertices of a legacy VTK PolyData (ASCII or BINARY) -- we only need the POINTS for a shape cloud."""
    ascii_mode = b"\nASCII" in raw[:256]
    nl = raw.find(b"POINTS "); le = raw.find(b"\n", nl)
    n = int(raw[nl:le].split()[1]); body = raw[le + 1:]
    if ascii_mode:
        v = np.array(body.split()[:n * 3], dtype=float)
    else:
        v = np.frombuffer(body[:n * 3 * 4], dtype=">f4").astype(float)
    return v.reshape(-1, 3)[:n]


_BP3D = {}


def _bp3d():
    """Lazy-load the BodyParts3D obj zip + the parent->child part-of graph (both mapping files)."""
    if _BP3D:
        return _BP3D
    import collections
    zf = zipfile.ZipFile(os.path.join(BP3D_DIR, "obj_99.zip"))
    members = {m.split("/")[-1][:-4]: m for m in zf.namelist() if m.endswith(".obj")}
    edges = collections.defaultdict(set)
    for f in ("conventional_part_of.txt", "composite_parts.txt"):    # col0 = parent id, col2 = child id
        for ln in open(os.path.join(BP3D_DIR, f), encoding="utf-8", errors="ignore").read().splitlines()[1:]:
            t = ln.split("\t")
            if len(t) >= 3:
                edges[t[0]].add(t[2])
    _BP3D.update(zf=zf, members=members, edges=edges)
    return _BP3D


def _obj_verts_raw(raw):
    V = [(float(p[1]), float(p[2]), float(p[3])) for p in
         (ln.split() for ln in raw.decode("utf-8", "ignore").splitlines()) if p and p[0] == "v"]
    return np.array(V, dtype=float) if V else np.zeros((0, 3))


def load_bp3d(root):
    """Union the vertices of every obj primitive under a composite BodyParts3D structure (heart, lung, brain...)."""
    S = _bp3d(); seen, leaves, stack = set(), [], [root]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in S["members"]:
            leaves.append(n)
        stack.extend(S["edges"].get(n, ()))
    pts = [v for l in leaves for v in (_obj_verts_raw(S["zf"].read(S["members"][l])),) if len(v)]
    return np.vstack(pts) if pts else np.zeros((0, 3))


def load_canonical(name):
    spec = CANON[name]
    if spec[0] == "oa":
        _, zf, member = spec
        return _vtk_points(zipfile.ZipFile(os.path.join(MDIR, zf)).read(member))
    if spec[0] == "bp3d":
        return load_bp3d(spec[1])
    raise ValueError(f"unknown source {spec[0]!r} for {name}")


def _sub(P, n, rng):
    P = np.asarray(P, float)
    return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]


def d2_signature(P, rng, npair=8000, nbin=48):
    """Osada D2: histogram of normalised pairwise distances -- a rotation/scale-invariant shape fingerprint."""
    i = rng.integers(0, len(P), npair); j = rng.integers(0, len(P), npair)
    d = np.linalg.norm(P[i] - P[j], axis=1); d /= (d.mean() + 1e-9)
    h, _ = np.histogram(d, bins=nbin, range=(0, 4.0), density=False)
    return h / (h.sum() + 1e-9)


def surface_pts(P, k=12, thresh=0.45, cap=6000):
    """SCORER v2 helper (2026-08-30): make solid model clouds comparable to surface-vertex references.

    A point is on the BOUNDARY of a solid cloud if its kNN neighbourhood is one-sided (the resultant of unit
    vectors to its k neighbours is long); an interior point sees neighbours all around (resultant ~1/sqrt(k)).
    SELF-CLASSIFYING: a cloud that is already a shell (mesh vertices -- the adult bp3d references) passes
    through unchanged, a solid (our model clouds, the Carnegie voxel volumes) gets peeled to its boundary.
    D2 on solid-vs-surface otherwise mismatches interior mass -- it preferred the anatomically-wrong fused
    kidney over the measured pair (the cycle-3 finding). Deterministic (fixed seed for the cap subsample).
    Used by the CURVE scorer (canon_frame_score/curve_train), NOT by the frozen benchmark suite (v1.2)."""
    P = np.asarray(P, float)
    if len(P) > cap:
        P = P[np.random.default_rng(0).choice(len(P), cap, replace=False)]
    if len(P) <= k + 2:
        return P
    from scipy.spatial import cKDTree
    _, j = cKDTree(P).query(P, k=k + 1)
    v = P[j[:, 1:]] - P[:, None, :]
    v /= (np.linalg.norm(v, axis=2, keepdims=True) + 1e-12)
    res = np.linalg.norm(v.mean(1), axis=1)
    m = res >= thresh
    if m.mean() > 0.80:                     # already a shell -- pass through
        return P
    if m.sum() < 60:                        # degenerate/thin cloud -- keep the outer half
        m = res >= np.quantile(res, 0.5)
    return P[m]


def descriptors(P):
    d = _desc(P) or {"elong": 0, "flat": 0}
    return dict(elong=round(float(d["elong"]), 2), flat=round(float(d["flat"]), 2),
                sphericity=round(float(sphericity(P)), 2), hollowness=round(float(hollowness(P)), 2),
                tortuosity=round(float(_tortuosity(P)), 2), n=int(len(P)))


def _long_axis(P):
    C = np.asarray(P, float) - np.asarray(P, float).mean(0)
    v = np.linalg.svd(C, full_matrices=False)[2][0]
    return v / (np.linalg.norm(v) + 1e-9)


# OpenAnatomy/SPL scan frame (x=L-R, y=S-I, z=A-P) -> our body frame (AP, DV, ML): AP<-S-I(y), DV<-A-P(z), ML<-L-R(x)
_CANON_TO_OURS = [1, 2, 0]
_AX = ["AP", "DV", "ML"]


def orientation(cs, os_):
    """D2 is rotation-INVARIANT, so orientation is scored separately: put the canonical mesh in OUR body frame and
    compare the principal (long) axis direction -- catches a part that is tilted / pointing along the wrong axis."""
    la_c = _long_axis(cs[:, _CANON_TO_OURS]); la_o = _long_axis(os_)
    cos = float(abs(np.dot(la_c, la_o)))
    return dict(tilt_deg=round(float(np.degrees(np.arccos(min(1.0, cos)))), 1),
                canon_axis=_AX[int(np.argmax(np.abs(la_c)))], ours_axis=_AX[int(np.argmax(np.abs(la_o)))],
                match=round(cos, 3))


def score_one(name, ours, rng):
    src = CANON[name][0]
    canon = load_canonical(name)
    cs, os_ = _sub(canon, 4000, rng), _sub(np.asarray(ours, float), 4000, rng)
    h_c, h_o = d2_signature(cs, rng), d2_signature(os_, rng)
    d2_sim = float(1.0 - 0.5 * np.abs(h_c - h_o).sum())           # histogram intersection in [0,1]
    # orientation only for OpenAnatomy (known scan frame); BodyParts3D frame is not calibrated to ours yet.
    orient = orientation(cs, os_) if src == "oa" else dict(
        tilt_deg=None, canon_axis="n/a", ours_axis=_AX[int(np.argmax(np.abs(_long_axis(os_))))], match=None)
    return dict(part=name, src=src, d2_match=round(d2_sim, 3), pct=round(100 * d2_sim, 1),
                orient=orient, canonical=descriptors(cs), ours=descriptors(os_),
                _hc=h_c.tolist(), _ho=h_o.tolist(), _cs=cs, _os=os_)


def _our_parts(R):
    from medic.unified_embryo import FIDX
    import numpy as _np
    base, F = _np.asarray(R["base"], float), _np.asarray(R["F"])
    skull = R.get("skull", {})
    skull_P = _np.vstack([v["P"] for v in skull.values() if isinstance(v, dict) and v.get("P") is not None])
    mand = skull.get("mandible", {}).get("P")

    def cells(*names):
        from medic.subhead_program import expand_names
        ids = [FIDX[n] for n in expand_names(names) if n in FIDX]   # + any sub-head children of each parent
        return base[_np.isin(F, ids)] if ids else base[:0]

    # our FATE labels -> the canonical isolated structure. gut = midgut+hindgut (colon+ileum, NOT stomach);
    # kidney folds in Nephron; liver folds in the fetal-haem pool. cells() expands each parent into its
    # sub-head children (Lung -> lobes etc.), so the map survives future splits.
    return {
        "skull":    skull_P,
        "mandible": _np.asarray(mand, float) if mand is not None else base[:0],
        "gut":      cells("Gut", "Hindgut"),
        "stomach":  cells("Stomach", "Duodenum"),   # ref mesh = stomach AND duodenum (Foregut minus oesophagus)
        "liver":    cells("Liver", "LiverHaem"),
        "spleen":   cells("Spleen"),
        "kidney":   cells("Kidney", "Nephron"),
        "pancreas": cells("Pancreas"),
        "adrenal":  cells("Adrenal"),
        # thorax + neural (vs BodyParts3D composites)
        "heart":    cells("Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow"),
        "lung":     cells("Lung"),
        "brain":    cells("Forebrain", "Midbrain", "Hindbrain", "Cerebellum", "OlfactoryBulb"),
        "eye":      cells("Eye", "Retina"),
    }


def _figure(results):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(results); fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), facecolor="#0d1017")
    for k, r in enumerate(results):
        for row, (P, lab, col) in enumerate([(r["_cs"], "canonical", "#7dd3fc"), (r["_os"], "ours", "#ffd23a")]):
            ax = axes[row][k]; ax.set_facecolor("#0d1017")
            Pn = P - P.mean(0); Pn /= (np.linalg.norm(Pn, axis=1).mean() + 1e-9)
            # front-ish view: 2 largest-variance axes
            V = np.linalg.svd(Pn - Pn.mean(0), full_matrices=False)[2]
            xy = Pn @ V[:2].T
            ax.scatter(xy[:, 0], xy[:, 1], s=0.5, c=col, alpha=0.4, linewidths=0)
            ax.set_aspect("equal"); ax.axis("off")
            ax.set_title(f"{r['part']} · {lab}" + (f"\nmatch {r['pct']}%" if row == 0 else ""),
                         color="#e2e8f0", fontsize=9)
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.tight_layout(); fig.savefig("data/organ_cascade/canonical_atlas.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/canonical_atlas.png")


def run():
    from medic.integrated_body import assemble
    rng = np.random.default_rng(0)
    print("assembling body ...")
    R = assemble()
    ours = _our_parts(R)
    results = [score_one(name, ours[name], rng) for name in CANON if len(ours.get(name, [])) >= 20]
    print("\nSHAPE vs ISOLATED CANONICAL MESH (OpenAnatomy/SPL + BodyParts3D) -- correspondence-free D2 match")
    for r in results:
        o = r["orient"]
        if o["tilt_deg"] is None:
            ori = f"orientation n/a ({r['src']})"
        else:
            flag = "  <-- ANGLED" if (o["canon_axis"] != o["ours_axis"] or o["tilt_deg"] > 20) else ""
            ori = f"orientation tilt {o['tilt_deg']}° (canonical long axis {o['canon_axis']} vs ours {o['ours_axis']}){flag}"
        print(f"\n  {r['part'].upper()} [{r['src']}]  D2 shape-match {r['pct']}%  |  {ori}")
        print(f"    {'':14s}{'elong':>7}{'flat':>7}{'spher':>7}{'hollow':>7}{'tort':>7}")
        for who in ("canonical", "ours"):
            d = r[who]
            print(f"    {who:14s}{d['elong']:>7}{d['flat']:>7}{d['sphericity']:>7}{d['hollowness']:>7}{d['tortuosity']:>7}")
    _figure(results)
    out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    json.dump(dict(source="OpenAnatomy/SPL VTK + BodyParts3D composites (isolated)", parts=out),
              open("data/organ_cascade/canonical_atlas.json", "w"), indent=1)
    print("\nsaved data/organ_cascade/canonical_atlas.json")


if __name__ == "__main__":
    run()
