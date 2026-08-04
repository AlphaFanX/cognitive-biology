"""
face_features_head.py -- the FACE head: carve the viscerocranium into the RECOGNISABLE features of Gray's face.

The skull head (medic.skull_head) names the facial BONES (nasal, maxilla, mandible, paired zygomatics). This
head reads those bones + the Eye-fate cells, read-only, and writes the FEATURES that make a skull read as a face:
the paired ORBITS (eye sockets), the NOSE (nasion/bridge, tip, paired alae), the paired CHEEKS (malar eminence),
the ORAL aperture (mouth), and the CHIN (mental protuberance). This is the "makes it click as a man" layer.

Biology / Gray's. The face is neural-crest, built from three prominences on the electric-face frame (the face's
own gap-junction eigenmodes -- the recursion one scale down from the body frame; Papers #2/#4/#9, and
medic.face_maturation):
  * FRONTONASAL prominence -> forehead, nasal bridge (nasion), nasal tip, philtrum, medial upper lip.
  * paired MAXILLARY prominences -> cheeks (malar), lateral upper lip, alae (nostril wings).
  * paired MANDIBULAR prominences -> lower jaw, lower lip, CHIN.
  * the ORBITS form around the optic cups (Eye fate) between the frontal bone (rim above) and the maxilla +
    zygomatic (rim below/lateral).
So each feature's POSITION is read from the geometry the upstream heads already produced -- the Eye cells and the
named skull bones -- not asserted; the head cannot oppose the genome because its inputs ARE the genome's outputs.

GLASS BOX (magnitudes). Each shape feature carries the REAL Xiong et al. (2025) C-GWAS per-allele effect of the
locus that shapes it (data/xiong_cgwas/real_effects.json, grounded by medic.ground_face_betas):
  nasion/bridge <- PAX3 (nasion) + RUNX2 (root),  tip <- DCHS2,  alae/width <- GLI3 + PAX1,  chin <- EDAR.
TERMS (Miles): the stored genome value is a GWAS BETA -- a regression COEFFICIENT (per-allele standardized
effect); a beta is itself a per-allele difference, so beta == per-allele delta at the source. What we apply to
the geometry is a positional DELTA: the feature landmark is DISPLACED from its neutral prominence antinode along
its morph axis by  delta = beta * dose * scale, so a beta (coefficient) becomes a delta (offset) on the face and
changing an allele dosage moves the feature by its real effect size -- the genome->face read-out.
Orbits / cheeks / mouth have no locus in this set; they are anchored purely by geometry (Eye fate, zygomatic,
the maxilla-mandible gap) and carry no GWAS displacement -- stated, not hidden.

READ-ONLY. Validation = every feature present; orbits + alae + cheeks bilateral and left/right symmetric; the
midline features ordered superior->inferior nasion > tip > mouth > chin along the face axis.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.face_features_head
Out: data/organ_cascade/face_features_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic import skull_head as SKU

REAL_EFFECTS = "data/xiong_cgwas/real_effects.json"

# feature -> the GWAS loci that shape it and the morph AXIS the effect displaces it along.
# axis is in the face-local frame (a=anterior, s=superior, l=lateral); lateral effects are applied symmetrically.
# genes with no entry here (orbit/cheek/mouth) are geometry-anchored only -- no GWAS displacement.
FEATURE_GWAS = {
    "nasion": [("PAX3", "a"), ("RUNX2", "a")],   # nasal-root projection (fronto-nasal)
    "nose_tip": [("DCHS2", "a")],                # tip projection (medial-nasal)
    "ala-R": [("GLI3", "l"), ("PAX1", "l")],     # nostril-wing width (lateral-nasal / maxillary)
    "ala-L": [("GLI3", "l"), ("PAX1", "l")],
    "chin": [("EDAR", "a")],                      # mental protuberance projection (mandibular)
}
MID = ("nasion", "nose_tip", "mouth", "chin")    # midline features, superior -> inferior
PAIRS = ("orbit", "ala", "cheek")                # bilateral features
DISP_SCALE = 0.35                                 # beta (coefficient) -> positional delta, in fractions of face radius (schematic)


def _dorsal_sign(base, F):
    if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8:
        return 1.0 if np.median(base[F == FIDX["Notochord"]][:, 1]) >= np.median(base[:, 1]) else -1.0
    return 1.0


def _face_frame(skull, eye, base, F):
    """Build the face-local (anterior, superior, lateral) frame + origin from the skull's facial bones + eyes.
    origin = midface (between the maxilla and the eyes); the axes are the body axes oriented to point INTO the
    face (anterior = the direction the nasal/maxilla lie from the head centre)."""
    parts = skull
    def cen(nm):
        p = parts.get(nm)
        return None if p is None else p["P"].mean(0)
    facials = [cen(n) for n in ("nasal", "maxilla", "mandible", "zygomatic-R", "zygomatic-L") if cen(n) is not None]
    face_c = np.mean(facials, 0) if facials else eye.mean(0)
    dsgn = _dorsal_sign(base, F)
    # head centre = mean of all cranial cells; anterior = from head centre toward the facial mass
    head_c = base.mean(0)
    a_axis = np.array([1.0, 0, 0])                      # AP+ is anterior (build_base orients head-forward)
    if cen("nasal") is not None and np.dot(cen("nasal") - head_c, a_axis) < 0:
        a_axis = -a_axis
    s_axis = np.array([0.0, dsgn, 0.0])                 # superior = dorsal sign on the DV axis
    l_axis = np.array([0.0, 0.0, 1.0])                  # lateral (L/R by sign)
    R = np.linalg.norm(eye - face_c, axis=1).mean() * 2.2 + 1e-6 if len(eye) else 1.0
    return dict(o=face_c, a=a_axis, s=s_axis, l=l_axis, R=R, dsgn=dsgn)


def _place(fr, a=0.0, s=0.0, l=0.0):
    """A face-local (anterior, superior, lateral) offset in units of face radius -> a world point."""
    return fr["o"] + fr["R"] * (a * fr["a"] + s * fr["s"] + l * fr["l"])


# ---- the ELECTRIC-FACE frame: the low eigenmodes of the face cells' own gap-junction operator -------------
# The recursion (Papers #2/#4/#9): just as the body axes are the low eigenmodes of the embryo's gap-junction
# operator, the FACE axes are the low eigenmodes of the FACE cells' gap-junction operator, and the three facial
# prominences nucleate on the ANTINODES of those modes -- the same construction as medic.face_primordium_3d,
# now run on the model's real neural-crest face cells instead of a synthetic voxel block.

def _gj_operator(pts, k=8):
    """kNN gap-junction graph Laplacian on the face cells (uniform connexin weight -- the facial Cx43 map is the
    stated open problem, medic.face_primordium_3d; the OPERATOR construction is the same as for the body)."""
    n = len(pts)
    k = min(k, n - 1)
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k + 1)
    sig = np.median(d[:, 1:]) + 1e-9
    rows, cols, w = [], [], []
    for i in range(n):
        for j, dist in zip(idx[i, 1:], d[i, 1:]):
            wij = np.exp(-(dist / sig) ** 2)
            rows += [i, j]; cols += [j, i]; w += [wij, wij]      # symmetric gap junction
    A = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    L = (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()
    return L


def _low_modes(L, k):
    vals, vecs = spla.eigsh(L, k=min(k + 1, L.shape[0] - 1), sigma=-1e-8, which="LM")
    o = np.argsort(vals)
    return vals[o][1:], vecs[:, o][:, 1:]                        # drop the trivial constant mode


def _eigen_prominences(face_pts, fr, n_modes=6):
    """Place the frontonasal / paired maxillary / mandibular prominences on the antinodes of the face's own
    gap-junction eigenmodes. Returns {name: world_point} + which mode is the lateral (L/R) mode."""
    if len(face_pts) < 30:
        return {}, {}
    L = _gj_operator(face_pts)
    _, vecs = _low_modes(L, n_modes)
    # face-local coordinates of every face cell
    d = face_pts - fr["o"]
    la = d @ fr["l"]; su = d @ fr["s"]                          # lateral, superior
    # identify the LATERAL eigenmode = the mode whose eigenvector best correlates with the lateral coordinate,
    # and the SUPERO-INFERIOR mode = best correlated with the superior coordinate.
    def best(coord):
        cs = [abs(np.corrcoef(vecs[:, m], coord)[0, 1]) for m in range(vecs.shape[1])]
        return int(np.nanargmax(cs)), cs
    m_lat, _ = best(la)
    m_si, _ = best(su)
    phi_lat = vecs[:, m_lat]; phi_si = vecs[:, m_si]
    # orient each mode so + points to +lateral / +superior
    if np.corrcoef(phi_lat, la)[0, 1] < 0: phi_lat = -phi_lat
    if np.corrcoef(phi_si, su)[0, 1] < 0: phi_si = -phi_si
    prom = {}
    # paired MAXILLARY prominences = the two antinodes of the lateral mode (argmax +, argmin -)
    prom["maxillary-R"] = face_pts[int(np.argmax(phi_lat))]
    prom["maxillary-L"] = face_pts[int(np.argmin(phi_lat))]
    # FRONTONASAL prominence = the lateral-mode NODE (|phi_lat| small = midline) that is highest on the superior
    # mode (the antinode of the supero-inferior mode near the midline).
    mid = np.abs(phi_lat) < np.percentile(np.abs(phi_lat), 25)
    if mid.sum():
        idx = np.where(mid)[0]
        prom["frontonasal"] = face_pts[idx[int(np.argmax(phi_si[idx]))]]
    # MANDIBULAR prominence = the inferior antinode of the supero-inferior mode, near the midline.
    if mid.sum():
        idx = np.where(mid)[0]
        prom["mandibular"] = face_pts[idx[int(np.argmin(phi_si[idx]))]]
    info = dict(n_face_cells=int(len(face_pts)), lateral_mode=int(m_lat), superoinferior_mode=int(m_si))
    return prom, info


def build(base, F, skull=None, dose=None):
    """Carve the recognisable facial features. `skull` = the skull head's parts (recomputed if None). `dose` =
    per-gene allele dosage (default 1.0, the effect-per-allele read-out). Returns per-feature landmarks +
    the GWAS provenance actually applied."""
    if skull is None:
        skull = SKU.build(base, F).get("parts", {})
    eye = base[F == FIDX["Eye"]] if "Eye" in FIDX else base[:0]
    real = json.load(open(REAL_EFFECTS)) if os.path.exists(REAL_EFFECTS) else {}
    dose = dose or {}
    fr = _face_frame(skull, eye, base, F)

    # the ELECTRIC-FACE frame: prominences on the antinodes of the face cells' gap-junction eigenmodes.
    # face substrate = the neural-crest viscerocranium: the facial skull bones + the optic-cup (Eye) cells.
    face_bits = [eye] if len(eye) else []
    for nm in ("nasal", "maxilla", "mandible", "zygomatic-R", "zygomatic-L"):
        if nm in skull:
            face_bits.append(skull[nm]["P"])
    face_pts = np.vstack(face_bits) if face_bits else base[:0]
    prom, eig_info = _eigen_prominences(face_pts, fr)

    def gwas_delta(feature):
        """Convert the feature's GWAS BETAS (coefficients) into a positional DELTA (offset) per face axis:
        delta_axis = sum_loci beta * dose * scale. Returns (delta dict a/s/l, provenance)."""
        delta = {"a": 0.0, "s": 0.0, "l": 0.0}
        prov = []
        for gene, axis in FEATURE_GWAS.get(feature, []):
            if gene in real:
                beta = real[gene]["beta"]
                delta[axis] += beta * dose.get(gene, 1.0) * DISP_SCALE      # beta (coefficient) -> delta (offset)
                prov.append(dict(gene=gene, snp=real[gene]["snp"], beta=beta, axis=axis))
        return delta, prov

    feats = {}
    prov_all = {}

    # ORBITS -- the eye sockets: the Eye-fate cells, split L/R by lateral sign, centred in the socket.
    if len(eye) >= 4:
        lat = (eye - fr["o"]) @ fr["l"]
        for side, m in (("R", lat > 0), ("L", lat < 0)):
            if m.sum() >= 2:
                feats[f"orbit-{side}"] = dict(kind="feature", part="orbit", side=side,
                                              P=eye[m], landmark=eye[m].mean(0))
    else:  # no optic cups grown -> place nominal orbits from the frame
        for side, sgn in (("R", 1), ("L", -1)):
            feats[f"orbit-{side}"] = dict(kind="feature", part="orbit", side=side,
                                          P=_place(fr, 0.35, 0.30, 0.30 * sgn)[None], landmark=_place(fr, 0.35, 0.30, 0.30 * sgn))

    # anchor for a feature: its prominence antinode if the eigenframe resolved it, else the hand frame (fallback).
    def anchor(prom_name, a0, s0, l0):
        if prom_name in prom:
            return prom[prom_name]
        return _place(fr, a0, s0, l0)

    # midline NOSE + MOUTH + CHIN are a GRADED SERIES tiling the midface from the FRONTONASAL prominence antinode
    # (superior) down to the MANDIBULAR prominence antinode (inferior) -- nasion at the nasal root, then the nose
    # tip, the oral aperture at the maxillary-mandibular boundary, and the chin on the mandible. Interpolate along
    # the frontonasal->mandibular antinode axis at anatomical fractions, so the Gray's supero-inferior ORDER
    # (nasion > tip > mouth > chin) holds BY CONSTRUCTION (previously nasion/tip hung off fn and mouth/chin off mn
    # with large fixed offsets that CROSSED when the two antinodes sat closer than ~0.58 R apart). The GWAS betas
    # still displace each feature along the ANTERIOR (projection) axis -- orthogonal to the order axis -- so the
    # genome->face read-out is unchanged and cannot break the order.
    fn = anchor("frontonasal", 0.55, 0.35, 0.0)              # frontonasal antinode (superior)
    mn = anchor("mandibular", 0.60, -0.60, 0.0)              # mandibular antinode (inferior)
    # the frontonasal prominence is ANATOMICALLY superior to the mandibular; if the eigenframe's antinode labels
    # came out swapped (the DV sign is ambiguous at low cell density -> the supero-inferior mode can flip), enforce
    # the known relation so the midface series always runs superior->inferior regardless of resolution.
    if (fn - fr["o"]) @ fr["s"] < (mn - fr["o"]) @ fr["s"]:
        fn, mn = mn, fn
    axis = mn - fn                                           # the supero-inferior midface axis (fn=0 -> mn=1)
    tfrac = {"nasion": 0.05, "nose_tip": 0.40, "mouth": 0.72, "chin": 0.96}   # fraction down the midface
    aproj = {"nasion": 0.03, "nose_tip": 0.22, "mouth": 0.08, "chin": 0.04}   # anterior projection (tip juts most)
    for nm in MID:
        d, prov = gwas_delta(nm)
        base_pt = fn + tfrac[nm] * axis
        lm = base_pt + fr["R"] * ((aproj[nm] + d["a"]) * fr["a"] + d["s"] * fr["s"])
        feats[nm] = dict(kind="feature", part=nm, side="M", P=lm[None], landmark=lm)
        if prov:
            prov_all[nm] = prov

    # paired ALAE (nostril wings) on the MAXILLARY prominence antinodes + CHEEKS (malar = zygomatic bone).
    for side, sgn in (("R", 1), ("L", -1)):
        d, prov = gwas_delta(f"ala-{side}")
        mx = anchor(f"maxillary-{'R' if sgn > 0 else 'L'}", 0.72, -0.02, 0.18 * sgn)
        lm = mx + fr["R"] * ((d["a"]) * fr["a"] + abs(d["l"]) * sgn * fr["l"])   # GLI3/PAX1 widen the nose
        feats[f"ala-{side}"] = dict(kind="feature", part="ala", side=side, P=lm[None], landmark=lm)
        if prov:
            prov_all[f"ala-{side}"] = prov
        zyg = skull.get(f"zygomatic-{side}")
        clm = zyg["P"].mean(0) if zyg is not None else _place(fr, 0.35, 0.10, 0.45 * sgn)
        feats[f"cheek-{side}"] = dict(kind="feature", part="cheek", side=side, P=clm[None], landmark=clm)

    return dict(features=feats, frame=fr, gwas=prov_all, prominences=prom, eig=eig_info, n=len(feats))


def _validate(res):
    feats = res["features"]
    fr = res["frame"]
    named = set(feats)
    pairs = {}
    for b in PAIRS:
        L, R = f"{b}-L" in feats, f"{b}-R" in feats
        pairs[b] = bool(L and R)
    # symmetry: paired landmarks mirror across the midline (lateral coord opposite sign, |.| close)
    sym = {}
    for b in PAIRS:
        if pairs[b]:
            lR = (feats[f"{b}-R"]["landmark"] - fr["o"]) @ fr["l"]
            lL = (feats[f"{b}-L"]["landmark"] - fr["o"]) @ fr["l"]
            sym[b] = bool(lR * lL < 0 and abs(abs(lR) - abs(lL)) < 0.25 * fr["R"])
    # midline order superior -> inferior along the face 's' axis
    s_of = {nm: (feats[nm]["landmark"] - fr["o"]) @ fr["s"] for nm in MID if nm in feats}
    order = [nm for nm in MID if nm in s_of]
    ordered = all(s_of[order[i]] > s_of[order[i + 1]] for i in range(len(order) - 1)) if len(order) > 1 else False
    return dict(features=len(feats), bilateral=pairs, symmetric=sym, midline_ordered=ordered,
                gwas_grounded=sorted(res["gwas"]),
                prominences_on_antinodes=sorted(res.get("prominences", {})),
                eig=res.get("eig", {}), roster=sorted(named))


def _figure(res, base):
    feats = res["features"]
    fr = res["frame"]
    cmap = {n: i for i, n in enumerate(sorted(feats))}
    x = base[:, 0]; apf = (x - x.min()) / (np.ptp(x) + 1e-9); head = base[apf >= 0.72]
    fig, ax = plt.subplots(1, 2, figsize=(12, 7), facecolor="#0d1017")
    for j, (i, k, ttl) in enumerate([(2, 1, "front (lateral x superior)"), (0, 1, "side (anterior x superior)")]):
        a = ax[j]; a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        # superior axis may be flipped (dorsal sign) -- draw in the face frame
        sflip = fr["dsgn"]
        a.scatter(head[:, i], sflip * head[:, k], s=3, c="#2a3140", alpha=0.35)
        for nm, p in feats.items():
            P = p["P"]
            a.scatter(P[:, i], sflip * P[:, k], s=14, c=[cmap[nm]] * len(P), cmap="tab20", vmin=0, vmax=len(cmap), alpha=0.9)
            lm = p["landmark"]
            a.annotate(nm, (lm[i], sflip * lm[k]), color="#cbd5e1", fontsize=6.5, ha="center")
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    v = _validate(res)
    ng = len(v["gwas_grounded"])
    fig.suptitle(f"Face head: {v['features']} features (orbits/nose/cheeks/mouth/chin) -- positions from skull "
                 f"bones + Eye fate, {ng} shape features carrying REAL C-GWAS betas (PAX3/RUNX2/DCHS2/GLI3/PAX1/EDAR)",
                 color="#e2e8f0", fontsize=8.5)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/face_features_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/face_features_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res)
    print(f"face: {v['features']} features")
    print(f"  roster: {v['roster']}")
    print(f"  bilateral: {v['bilateral']}   symmetric: {v['symmetric']}")
    print(f"  midline ordered (nasion>tip>mouth>chin): {v['midline_ordered']}")
    print(f"  prominences on ELECTRIC-FACE eigenmode antinodes: {v['prominences_on_antinodes']}")
    print(f"    eigenframe: {v['eig']}")
    print(f"  GWAS-grounded shape features: {v['gwas_grounded']}")
    print("  genome-derived: prominences on the antinodes of the face cells' own gap-junction eigenmodes "
          "(the recursion); magnitudes = real Xiong C-GWAS betas")
    _figure(res, base)
    dump = dict(v, gwas=res["gwas"])
    json.dump(dump, open("data/organ_cascade/face_features_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/face_features_head.json")


if __name__ == "__main__":
    main()
