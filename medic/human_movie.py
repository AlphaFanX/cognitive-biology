"""
human_movie.py  ---  the end-to-end "tetrapod -> human adult" developmental movie.

ONE genome, ONE clock coordinate t in [0,1], THREE phases stitched into a single timeline:

  A. CLOUD  (morphogenesis, t ~ 0.00 -> 0.55)   the generator.
     unified_embryo.simulate() grows 1 -> N cells into the tetrapod (limb buds on the
     electric-body antinodes, organs, gentle flexure). Topology is CREATED here -- neural
     tube, heart, limb buds -- which is exactly why this phase must be a cell cloud and
     cannot be a MakeHuman slider.

  B. HANDOFF (coarse-graining, t ~ 0.55 -> 0.62)  the missing primitive.
     The final tetrapod cloud is put in correspondence with points sampled on the human
     EMBRYO surface (MakeHuman de-matured to embryo proportions) and morphed onto it. The
     "same clump of cells" flows onto the human surface -- the resample/handoff step.

  C. MESH  (allometric maturation, t ~ 0.62 -> 1.00)  the readout.
     Topology is now FIXED; only proportions change. mature_body() runs the MakeHuman mesh
     from embryo -> fetus -> infant -> adult by whole-body allometry (baby-schema: big
     cranium, short limbs early). This is the cheap parametric slice, correctly used only
     where topology no longer changes.

A right-hand PANEL shows which genes/heads are dialing at each frame. Genes that act in
BOTH the cloud phase and the mesh phase (WNT-PCP width, CDX2/HOX axis, SHH midline/fold,
TBX5/FGF10 limb, OTX2/SIX3 head, PITX2 heart, RUNX2 skeleton, PAX3, BMP/FOXA2, CDX/tail)
are flagged gold -- the spanning set Miles asked to see.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.human_movie
Then: cd cognimed && python -m http.server 8903 --directory data
      open http://localhost:8903/human_movie_viewer.html

HONEST SCOPE (first end-to-end pass):
  * whole-body allometry is coarse (vertical-band region scaling), not measured longitudinal data;
  * the cloud->mesh correspondence is AP-ring rank matching (visual continuity, not a true
    cell lineage map);
  * the movie is shown in one laid-down (head +x) frame for continuity -- the adult lies
    horizontal, orbit to inspect.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from medic.unified_embryo import simulate, _symmetrize, VMIN, VMAX, FATES, FIDX

# the menagerie "Hill-organ" parts atlas: the homologous roster (bones + Hill-muscles + viscera),
# decoded from a human genome and placed in situ. Used for the adult-stage anatomy reveal.
from menagerie.targets import reference_genome
from menagerie.skeleton import build_skeleton, render_points as _bone_pts
from menagerie.muscles import build_muscles, render_points as _musc_pts
from menagerie.organs import build_organs, render_points as _organ_pts, ORGAN_PLAN

# anatomy palette (per fate); buds pop: limb=green eye=cyan heart=red otic=gold. Any FATE not
# listed falls back to a neutral grey (unified_embryo's FATES list grows over time).
_ANAT = {"Forebrain": (0.30, 0.46, 0.95), "Eye": (0.20, 0.85, 1.00), "Nervous System": (0.36, 0.55, 0.95),
         "Spinal Cord": (0.46, 0.62, 0.92), "Neural Crest": (0.66, 0.42, 0.86), "Mesoderm": (0.92, 0.56, 0.46),
         "Somite": (0.96, 0.66, 0.42), "Epidermal": (0.82, 0.86, 0.90), "Hypoblast": (0.86, 0.76, 0.46),
         "Yolk Syncytial Layer": (0.90, 0.80, 0.40), "Blastodisc": (0.72, 0.74, 0.78),
         "Proliferative Like Cell": (0.66, 0.68, 0.72), "Limb Bud": (0.28, 0.86, 0.46),
         "Heart": (0.93, 0.16, 0.22), "Otic": (1.00, 0.82, 0.20), "Liver": (0.72, 0.34, 0.62),
         "Lung": (0.55, 0.78, 0.90), "Pancreas": (0.80, 0.82, 0.30), "Gut": (0.82, 0.60, 0.40),
         "Rib": (0.94, 0.94, 0.86), "Kidney": (0.66, 0.28, 0.46), "Muscle": (0.86, 0.42, 0.42),
         "Notochord": (0.60, 0.82, 0.72), "Skin": (0.96, 0.82, 0.74), "Cartilage": (0.80, 0.86, 0.92),
         "DRG": (0.72, 0.30, 0.90), "Sympathetic": (0.90, 0.45, 0.85), "Vessel": (0.80, 0.10, 0.20),
         "Meninges": (0.58, 0.50, 0.78), "Connective": (0.78, 0.70, 0.58), "Jaw": (0.86, 0.62, 0.70),
         "Choroid": (0.40, 0.78, 0.86), "Gonad": (0.90, 0.52, 0.60), "Midbrain": (0.34, 0.52, 0.94),
         "Hindbrain": (0.40, 0.62, 0.90), "Cerebellum": (0.52, 0.72, 0.94), "Mesothelium": (0.66, 0.74, 0.68),
         "Mesentery": (0.72, 0.66, 0.52), "Mucosa": (0.88, 0.72, 0.52), "HeadMes": (0.70, 0.62, 0.52),
         "Branchial": (0.80, 0.56, 0.62), "Blood": (0.74, 0.10, 0.14), "Atrium": (0.95, 0.34, 0.40),
         "Ventricle": (0.85, 0.12, 0.18), "Outflow": (0.99, 0.52, 0.44), "LiverHaem": (0.80, 0.20, 0.34),
         "Foregut": (0.88, 0.68, 0.44), "Hindgut": (0.68, 0.48, 0.32), "Nephron": (0.58, 0.30, 0.54),
         "Retina": (0.24, 0.76, 0.86), "Adrenal": (0.94, 0.72, 0.30), "Thymus": (0.76, 0.82, 0.58),
         "Spleen": (0.58, 0.16, 0.30), "Bladder": (0.86, 0.76, 0.56), "Adipose": (0.96, 0.86, 0.54),
         "OlfactoryBulb": (0.42, 0.56, 0.92)}
ANAT_LIST = [list(_ANAT.get(f, (0.62, 0.65, 0.70))) for f in FATES]

# ---- Hill-organ atlas palette (appended AFTER the cloud fates so indices stay stable) ----
# class colours for the adult anatomy reveal: bone, muscle, one per distinct viscus, teeth.
_ORGAN_NAMES, _seen = [], set()
for _row in ORGAN_PLAN:                                     # (name, ap, dv, lat, size, fate, col, paired)
    if _row[0] not in _seen:
        _seen.add(_row[0]); _ORGAN_NAMES.append((_row[0], _row[6]))
ATLAS_PALETTE = ([("bone", (0.85, 0.83, 0.78)), ("muscle", (0.74, 0.20, 0.20))]
                 + _ORGAN_NAMES + [("teeth", (0.97, 0.97, 0.93))])
ATLAS_BASE = len(ANAT_LIST)                                 # first atlas fate index
ATLAS_IDX = {name: i for i, (name, _c) in enumerate(ATLAS_PALETTE)}   # class -> local offset
ATLAS_COLS = [list(c) for _n, c in ATLAS_PALETTE]
# organs + teeth pop as larger spheres in the reveal
ATLAS_HI = [ATLAS_BASE + ATLAS_IDX[n] for n, _c in _ORGAN_NAMES] + [ATLAS_BASE + ATLAS_IDX["teeth"]]
ATLAS_LAYERS = ["Skeleton", "Muscle", "Viscera", "Teeth"]


def shape_limbs(Q, fate, f, limb_id):
    """Outgrow the four limb buds into flattened paddles with digit rays (late). Copied from
    basic_vertebrate_browser (which is import-broken by a stale palette)."""
    isb = fate == limb_id
    if f < 0.55 or not isb.any():
        return Q
    Q = Q.copy()
    prog = (f - 0.55) / 0.45
    idx = np.where(isb)[0]
    P = Q[idx]
    xmid = np.median(P[:, 0])
    for fore in (True, False):
        for side in (+1, -1):
            sel = ((P[:, 0] < xmid) if fore else (P[:, 0] >= xmid)) & (np.sign(P[:, 2]) == side)
            if sel.sum() < 12:
                continue
            L = P[sel]
            cen = L.mean(0)
            L[:, 0] = cen[0] + (L[:, 0] - cen[0]) * (1 - 0.25 * prog)
            dz = np.clip(np.abs(L[:, 2]) - np.abs(L[:, 2]).min(), 0, None)
            u = dz / (dz.max() + 1e-9)
            ymid = np.median(L[:, 1])
            L[:, 1] = ymid + (L[:, 1] - ymid) * (1 - 0.55 * prog)
            L[:, 2] += side * 0.13 * prog * (0.35 + 0.65 * u)
            xc = L[:, 0].mean(); xspan = np.ptp(L[:, 0]) + 1e-9
            tx = (L[:, 0] - xc) / xspan
            centers = np.array([-0.34, 0.0, 0.34])
            snap = centers[np.argmin(np.abs(tx[:, None] - centers[None, :]), 1)] * xspan + xc
            w = np.clip((u - 0.55) / 0.45, 0, 1) * prog * 0.5
            L[:, 0] = L[:, 0] * (1 - w) + snap * w
            P[sel] = L
    Q[idx] = P
    return Q


def flex(Q, f):
    """Late cephalo-caudal flexure: a gentle cephalic curve. Copied from basic_vertebrate_browser.

    The spine is re-parametrised by arc length s = x - x.min() with the body flank h = y - median(y).
    IMPORTANT: the zero-bend case must use this SAME convention (not the raw input), otherwise the
    instant the bend ramps on (f=0.42 -> clock t~0.23) the whole cloud translates by (x.min, median y)
    -- a visible jump. So bend~0 returns the straight limit [s, h, z], which is exactly what the bent
    branch converges to as b->0."""
    bend = np.radians(24.0) * float(np.clip((f - 0.42) / 0.58, 0, 1))
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    s = x - x.min()
    h = y - np.median(y)
    if bend < 1e-6:
        return np.stack([s, h, z], 1).astype(np.float32)     # straight limit, same convention
    L = np.ptp(x) + 1e-9
    b = bend / L
    th = b * s
    cx = np.sin(th) / b
    cy = (np.cos(th) - 1.0) / b
    nx, ny = -np.sin(th), np.cos(th)
    return np.stack([cx + h * nx, cy + h * ny, z], 1).astype(np.float32)

# ---------------------------------------------------------------- config
CLOUD_N   = 120000    # cells grown in the cloud phase (ultra fidelity; -> ~240k after symmetrization)
N_R       = 20000     # points rendered per frame. Now that the viewer draws a GPU POINT CLOUD (THREE.Points,
                      # not instanced spheres), the iGPU handles this easily -> a genuinely dense body (a real
                      # fraction of the 240k cloud) instead of a sparse scatter. JSON grows (~55 MB, loading bar)
N_HANDOFF = 10        # handoff morph frames
N_MESH    = 36        # maturation frames (embryo -> adult); more frames = smoother allometry morph
N_REVEAL  = 10        # adult-skin -> Hill-organ-atlas reveal frames
N_HOLD    = 6         # frames held on the finished anatomy atlas
NEUT      = -50.0     # neutral Vm for the mesh phase (voltage is a morphogenetic-phase read)
MH        = "data/bodybase/makehuman_decimated.npz"
JSON      = Path("data/movie/human_movie_frames.json")
HTML      = Path("data/human_movie_viewer.html")

LIMB = FIDX["Limb Bud"]

# limb-program knobs found by the von Dassow-Odell search (medic.develop_search): the AER-proliferation
# mechanism that fills the four buds into real limbs while the fish stays limbless. None -> model defaults.
try:
    LIMB_SEARCHED = json.load(open("data/organ_cascade/limb_search.json"))["best_knobs"]
except Exception:
    LIMB_SEARCHED = None

# fate-map knobs from the joint HESTA differentiation search (medic.differentiation_search): the coupled
# tissue thresholds tuned so the whole embryo composition matches HESTA. SCALE-AWARE -- the map is
# calibrated FOR a cell count (the normalised fate bands capture a different share as density shifts), so
# high-resolution runs use the map re-searched at 60k (medic.fate_rescale) instead of the 18k one; else the
# neural fates balloon and Muscle/Cartilage/Kidney collapse. fate_map_for(ne) picks the right one.
def _load_fate(path):
    try:
        return json.load(open(path))["best_fate_params"]
    except Exception:
        return None

FATE_18K = _load_fate("data/organ_cascade/differentiation_search.json")
FATE_60K = _load_fate("data/organ_cascade/differentiation_search_60k.json")
FATE_240K = _load_fate("data/organ_cascade/differentiation_search_240k.json")
FATE_SEARCHED = FATE_18K                                  # back-compat default (18k map)

def fate_map_for(ne):
    """The fate map calibrated for this cell count (each re-searched vs HESTA at that density, because the
    normalised fate bands capture a different share as the cell distribution changes): 240k map at ultra-high
    resolution, 60k map at high, else the 18k map."""
    if ne is not None and ne >= 90000 and FATE_240K is not None:
        return FATE_240K
    if ne is not None and ne >= 25000 and FATE_60K is not None:
        return FATE_60K
    return FATE_18K

# maturation-allometry knobs -- the tweakable parameters of mature_cloud (embryo cloud -> adult) that the
# von Dassow-Odell search (medic.atlas_relax_search) tunes so the matured cloud RELAXES INTO the Phase-D
# Hill-organ atlas (the target morphology / attractor capsules), WITHOUT breaking the 5 placement
# mechanisms + 3 heads (folded into that search's objective as a penalty). None -> the hand-set defaults.
MATURE_DEFAULTS = dict(trunk_e=0.28, dv_girth=0.40, ml_girth=0.18, head_ht0=3.5, head_ht1=3.5,
                       limb_ext=1.5, leg_ext=1.5,                # arms (limb_ext) and legs (leg_ext) extend independently
                       shoulder_w=1.0, waist_w=1.0, hip_w=1.0,    # regional taper (Vitruvian canon), identity=1.0
                       head_dv=0.48)                              # head dorsoventral depth (flat disc -> rounded head, ~0.09->0.13 H)
try:
    MATURE_SEARCHED = {**MATURE_DEFAULTS,
                       **json.load(open("data/organ_cascade/atlas_relax_search.json"))["best_knobs"]}
except Exception:
    MATURE_SEARCHED = dict(MATURE_DEFAULTS)

# highlight fates (pop as larger spheres) -- reuse the browser's organ set
HI_FATES = [n for n in ("Limb Bud", "Eye", "Heart", "Otic", "Liver", "Lung", "Pancreas",
                        "Gut", "Rib", "Kidney", "Muscle", "Notochord", "Cartilage", "DRG",
                        "Sympathetic", "Vessel", "Jaw", "Choroid", "Gonad", "Meninges",
                        "Atrium", "Ventricle", "Outflow", "LiverHaem", "Foregut", "Hindgut",
                        "Nephron", "Retina", "Adrenal", "Thymus", "Spleen", "Bladder",
                        "Adipose", "OlfactoryBulb", "Forebrain",     # small/structural organs get bigger points;
                        "Midbrain", "Hindbrain", "Cerebellum")       # the whole HEAD highlighted so it reads
                        if n in FIDX]                                # (Spinal Cord REMOVED -- it is huge, it was
HI_IDX = [FIDX[n] for n in HI_FATES]                                 # dominating the draw and thickening the spine)


def _rep_idx(F, n_target, rng, alpha=0.32):
    """REPRESENTATIVE subsample: draw n_target cells weighted by 1/count^alpha of each cell's fate, so the
    drawn set roughly follows the true tissue composition BUT rare organs (eye/heart/pancreas) stay visible
    and no single huge fate (Spinal Cord ~20% of cells) dominates the picture. alpha=0 = strictly
    proportional, alpha=1 = equal per fate; 0.32 keeps the body's real shape while lifting the small parts."""
    n = len(F)
    if n <= n_target:
        return np.arange(n)
    _, inv, cnt = np.unique(F, return_inverse=True, return_counts=True)
    w = 1.0 / (cnt[inv].astype(float) ** alpha)
    w /= w.sum()
    return rng.choice(n, n_target, replace=False, p=w)


# ---------------------------------------------------------------- gene registry
# each gene: cloud = (role text, [fates it is read from]) or None ; mesh = (role text, knob) or None.
# a gene with BOTH cloud and mesh roles is a SPANNING gene (gold in the panel).
GENE_ROLES = {
    "WNT-PCP (Wnt5a/Vangl2)": dict(
        cloud=("convergent extension -> body width (fish<->tetrapod)", ["Somite", "Mesoderm", "Notochord"]),
        mesh=("w_ml -- mediolateral width", "w_ml")),
    "CDX2 / HOX": dict(
        cloud=("posterior AP address (Hox colinearity)", ["Somite", "Spinal Cord", "Hindgut", "Rib"]),
        mesh=("L -- AP axis length", "L")),
    "SHH / notochord": dict(
        cloud=("midline notochord & floor plate (DV patterning)", ["Notochord", "Spinal Cord", "Nervous System"]),
        mesh=("kappa -- axial fold / straightening", "kappa")),
    "TBX5 / FGF10": dict(
        cloud=("limb-bud initiation (+ heart field)", ["Limb Bud", "Heart"]),
        mesh=("limb outgrowth / articulation", "limb")),
    "OTX2 / SIX3": dict(
        cloud=("anterior neural -- forebrain & eye", ["Forebrain", "Eye", "Midbrain", "Retina", "OlfactoryBulb"]),
        mesh=("head -- cephalic bulge / cranial allometry", "head")),
    "PITX2": dict(
        cloud=("cardiac looping & L-R asymmetry", ["Heart", "Atrium", "Ventricle", "Outflow"]),
        mesh=("cardiac chamber size (adapter)", "heart")),
    "RUNX2": dict(
        cloud=("skeletogenesis -- ribs, cartilage, jaw", ["Rib", "Cartilage", "Jaw", "Connective"]),
        mesh=("bone / face bridge -- skeletal maturation", "face_bridge")),
    "PAX3": dict(
        cloud=("neural crest & somite (dermomyotome)", ["Somite", "Muscle", "DRG", "Sympathetic"]),
        mesh=("nasion depth (face)", "face_nasion")),
    "BMP / FOXA2": dict(
        cloud=("ventral endoderm / DV thickness", ["Foregut", "Hindgut", "Gut", "Liver", "Lung", "Pancreas"]),
        mesh=("w_dv -- DV thickness", "w_dv")),
    "CDX / tailbud": dict(
        cloud=("posterior taper / tailbud", ["Hindgut", "Spinal Cord"]),
        mesh=("taper", "taper")),
    # cloud-only morphogenetic heads (clock-gated)
    "Division (cell cycle)": dict(cloud=("proliferation, declines as the clock runs down", None), mesh=None),
    "Differentiation (PRC2/telomere clock)": dict(cloud=("fate commitment ramps as the clock runs down", None), mesh=None),
    "Electric-body frame (gap junctions)": dict(cloud=("gap-junction eigenmodes = the body axes", None), mesh=None),
    # mesh-only maturation / individual-adapter genes
    "GH / IGF1": dict(cloud=None, mesh=("overall body growth (maturation)", "grow")),
    "SHOX": dict(cloud=None, mesh=("long-bone / limb elongation", "limbgrow")),
    "EDAR": dict(cloud=None, mesh=("chin protrusion (individual adapter)", "face_chin")),
    "Height PGS (~12k loci)": dict(cloud=None, mesh=("polygenic stature (individual adapter)", "height")),
}
SPANNING = [g for g, r in GENE_ROLES.items() if r["cloud"] and r["mesh"]]

# the five-primitive ontology (Miles): each gene/head acts in ONE primary LAYER, ordered by
# developmental DEPTH -- Clock (when) and Axes (where) run the whole trajectory; Heads (what a
# cell becomes) are the morphogenesis event that goes quiet at the handoff (the kernel dying);
# Proportions (the shape) are the surface readout that the adapter tunes in the mesh phase.
# "Genes" are not a layer -- they are the knobs threaded through every layer (every row IS a gene).
LAYER_ORDER = ["Clock", "Axes", "Heads", "Proportions"]
LAYER = {
    "Division (cell cycle)": "Clock",
    "Differentiation (PRC2/telomere clock)": "Clock",
    "Electric-body frame (gap junctions)": "Axes",
    "WNT-PCP (Wnt5a/Vangl2)": "Axes",
    "CDX2 / HOX": "Axes",
    "SHH / notochord": "Axes",
    "CDX / tailbud": "Axes",
    "OTX2 / SIX3": "Heads",
    "TBX5 / FGF10": "Heads",
    "PITX2": "Heads",
    "RUNX2": "Heads",
    "PAX3": "Heads",
    "BMP / FOXA2": "Heads",
    "GH / IGF1": "Proportions",
    "SHOX": "Proportions",
    "EDAR": "Proportions",
    "Height PGS (~12k loci)": "Proportions",
}

# mesh-knob move magnitudes for the panel bars, GROUNDED IN REAL FITTED DATA (no illustrative
# numbers). The seven body-plan knobs (L, w_ml, w_dv, kappa, head, limb, taper) carry the actual
# |knob move| the mouse->human builder fitted: a SPECIES move (mouse embryo -> human embryo) that
# fires during the handoff/reshape, and a separate MATURATION move (human embryo -> adult) that
# fires as the body grows. The face/heart individual-adapter knobs carry real GWAS |beta| (Xiong
# C-GWAS / literature). Only the pure growth ramp (GH/IGF1, SHOX) + the height PGS stay parametric
# (no single-allele morphology beta exists for them).  Source files:
#   data/organ_cascade/mouse_to_human_builder.json   (species + maturation moves)
#   data/adapter_table.json                          (Xiong C-GWAS / literature betas)
_BUILDER = "data/organ_cascade/mouse_to_human_builder.json"
_ADAPTER = "data/adapter_table.json"
BODYPLAN_KNOBS = ["L", "w_ml", "w_dv", "kappa", "head", "limb", "taper"]
# adapter panel-knob -> adapter_table.json key
_ADAPTER_KEY = {"face_chin": "face.chin", "face_nasion": "face.nasion",
                "face_bridge": "face.nasal_bridge", "heart": "heart.la_volume"}


def _load_real_moves():
    """SPECIES_MAG (reshape/handoff) + MATURE_MAG (embryo->adult) per mesh knob, from real fits.
    Body-plan knobs get the builder's |move| (each family normalised to its own max so bars are
    comparable); adapter knobs get GWAS |beta| scaled to a modest individual-variation range;
    growth genes keep the maturation ramp. Falls back to a safe default if a file is missing."""
    try:
        b = json.load(open(_BUILDER))
        sp = {m["param"]: abs(m["move"]) for m in b["species_adapter"]["move"]}
        ma = {m["param"]: abs(m["move"]) for m in b["maturation"]["move"]}
        smax = max(sp[k] for k in BODYPLAN_KNOBS) or 1.0
        mmax = max(ma[k] for k in BODYPLAN_KNOBS) or 1.0
        species = {k: sp[k] / smax for k in BODYPLAN_KNOBS}
        mature = {k: ma[k] / mmax for k in BODYPLAN_KNOBS}          # body-plan maturation moves
    except Exception as e:                                          # pragma: no cover
        print(f"    [warn] builder moves unavailable ({e}); using flat defaults")
        species = {k: 0.5 for k in BODYPLAN_KNOBS}
        mature = {k: 0.5 for k in BODYPLAN_KNOBS}
    try:
        a = json.load(open(_ADAPTER))
        betas = {k: abs(a[key]["beta"]) for k, key in _ADAPTER_KEY.items()
                 if a.get(key, {}).get("beta") is not None}
        bmax = max(betas.values()) or 1.0
        for k, v in betas.items():
            mature[k] = 0.6 * v / bmax                              # individual adapters read modest
    except Exception as e:                                          # pragma: no cover
        print(f"    [warn] adapter betas unavailable ({e})")
    mature["grow"] = 1.00      # GH/IGF1 overall growth ramp (no single-allele morphology beta)
    mature["limbgrow"] = 0.55  # SHOX long-bone elongation
    mature["height"] = 0.50    # polygenic stature (PGS, not one allele)
    return species, mature


SPECIES_MAG, MATURE_MAG = _load_real_moves()


# ---------------------------------------------------------------- helpers
def _long_axis_len(P):
    return float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), np.ptp(P[:, 2])) + 1e-9)


def _ap_ring_order(P):
    """Deterministic order: bin by AP fraction (x), then by angle around the AP axis.
    Pairing two point sets by this rank gives a coherent tube-like morph (anterior ring
    to anterior ring), which is what the cloud->mesh handoff needs."""
    x = P[:, 0]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    apbin = np.clip((apf * 30).astype(int), 0, 29)
    theta = np.arctan2(P[:, 1] - np.median(P[:, 1]), P[:, 2] - np.median(P[:, 2]))
    return np.lexsort((theta, apbin))


# ---- region-aware handoff correspondence (limb -> limb, head -> head, no axial collapse) ----
# The old handoff paired the cloud and the human embryo by a single AP-ring rank, which ignores
# limbs: a green leg-bud point mapped to whatever body point shared its rank, so limbs did not
# correspond and the morph midpoint collapsed onto the axis. Instead we label BOTH point sets by
# gross body region and match within region, so the tetrapod's four limbs flow onto the human's
# four limbs and the silhouette is preserved through the morph.
_REG = {"HEAD": 0, "TRUNK": 1, "ARM_L": 2, "ARM_R": 3, "LEG_L": 4, "LEG_R": 5}


def _body_regions(P, limb):
    """Label points (laid frame: x=AP head+x, y=DV, z=ML) as head / trunk / arm·leg (L,R).
    `limb` is a boolean mask of limb points (limb-bud fate on the cloud, limb identity on the mesh).
    Arms are the anterior limbs, legs the posterior limbs; L/R by the sign of the ML axis."""
    x, z = P[:, 0], P[:, 2]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    reg = np.full(len(P), _REG["TRUNK"], int)
    reg[(apf > 0.80) & ~limb] = _REG["HEAD"]
    arm = limb & (apf >= 0.5)
    leg = limb & (apf < 0.5)
    reg[arm & (z >= 0)] = _REG["ARM_R"]; reg[arm & (z < 0)] = _REG["ARM_L"]
    reg[leg & (z >= 0)] = _REG["LEG_R"]; reg[leg & (z < 0)] = _REG["LEG_L"]
    return reg


def _src_pool(srcreg, r):
    """Cloud indices to draw from for target region r, degrading gracefully: exact region first,
    then same-limb-type both sides, then any limb, then the whole cloud -- so a sparse side never
    leaves a target region unmatched."""
    order = [[r]]
    if r >= _REG["ARM_L"]:                              # a limb region
        same = [2, 3] if r in (2, 3) else [4, 5]
        order += [same, [2, 3, 4, 5]]
    order += [list(range(6))]
    for cand in order:
        si = np.where(np.isin(srcreg, cand))[0]
        if len(si) >= 4:
            return si
    return np.arange(len(srcreg))


def _align_cloud_to_human(S0, srcreg, E0, tgtreg):
    """Return perm so that S0[perm] aligns index-for-index with E0 by BODY REGION: for each target
    region, order both region point-sets by AP-ring and resample the cloud to the human count. The
    target array E0 is left untouched (so the last handoff frame is exactly the mesh f=0 frame --
    no boundary jump); only the cloud source is permuted."""
    perm = np.zeros(len(E0), int)
    for r in range(6):
        ti = np.where(tgtreg == r)[0]
        if not len(ti):
            continue
        si = _src_pool(srcreg, r)
        ti = ti[_ap_ring_order(E0[ti])]
        si = si[_ap_ring_order(S0[si])]
        pick = np.minimum((np.arange(len(ti)) * len(si) // len(ti)), len(si) - 1)
        perm[ti] = si[pick]
    return perm


def region_fate(V):
    """Assign each upright MakeHuman vertex a FATES index by body region, so the anatomy
    colouring keeps working through the mesh phase (heart pops red, head blue, limbs muscle)."""
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    h = (y - y.min()) / (np.ptp(y) + 1e-9)
    span = np.percentile(np.abs(x), 60) + 1e-9
    out = np.full(len(V), FIDX["Skin"], int)
    # legs -- painted with the LIMB-BUD identity (green) so the tetrapod's green legs flow
    # continuously into the infant's legs across the handoff (feet keep Cartilage).
    out[h < 0.48] = FIDX["Limb Bud"]
    out[h < 0.04] = FIDX["Cartilage"]                       # feet
    # arms (lateral, mid height) -- likewise the limb-bud identity so green arms -> green arms
    arm = (np.abs(x) > 1.6 * span) & (h > 0.45) & (h < 0.82)
    out[arm] = FIDX["Limb Bud"]
    out[arm & (np.abs(x) > 2.6 * span)] = FIDX["Cartilage"]  # hands
    # torso back = spinal cord
    out[(h >= 0.48) & (h < 0.82) & (z < np.percentile(z, 20))] = FIDX["Spinal Cord"]
    # heart pocket (upper-left central torso)
    out[(h > 0.62) & (h < 0.74) & (np.abs(x) < 0.8 * span) & (z > np.median(z))] = FIDX["Heart"]
    # head / cranium and face
    out[h >= 0.82] = FIDX["Forebrain"]
    out[(h >= 0.80) & (h < 0.90) & (z > np.percentile(z, 70))] = FIDX["Jaw"]   # face front
    return out


def mature_body(V, f):
    """Differential whole-body allometry, early fetus (f=0) -> adult (f=1). Changes PROPORTIONS
    only; absolute display size is applied separately by the caller, so a genuine baby-schema
    form MATURES into the adult -- it is not an adult being scaled up.

    Anchored to the classic 'heads-tall' allometry: the cranium is huge and the limbs tiny in the
    early fetus (~2.5 heads tall), the head is ~1/4 of body length at birth (~4 heads), and the
    adult MakeHuman base sits near ~5.5 heads. So when young the head enlarges strongly (~1.8x
    relative), the LEGS are far shorter (they elongate the most with age), the arms are shorter,
    the trunk stays relatively long, the body is chubbier/rounder, and the upper cranium bulges
    over a small face. Coarse vertical-band scaling about the neck/hip pivots -- labelled honest,
    not longitudinal data. V = upright MakeHuman (x span, y height, z depth)."""
    V = V.astype(float).copy()
    y = V[:, 1]
    ymin, H = y.min(), np.ptp(y) + 1e-9
    neck = ymin + 0.82 * H
    hf = float(np.clip(1.0 - f, 0.0, 1.0))     # "youth": 1 at the early fetus, 0 at the adult
    head_g = 1.0 + 0.80 * hf ** 0.85           # cranium ~1.8x relative when young -> 1.0 adult
    body_e = 0.72 + 0.28 * f                   # below-neck trunk elongation (trunk stays long young)
    leg_f  = 0.62 + 0.38 * f ** 1.15           # mild residual leg allometry; bud->limb outgrowth = grow_limbs
    arm_e  = 1.0                                # arm LENGTH handled by grow_limbs (bud -> full), not here
    wid    = 1.0 + 0.20 * hf                   # chubbier / rounder when young (x span and z depth)
    # elongate the whole body below the neck about the neck line
    below = V[:, 1] < neck
    V[below, 1] = neck + (V[below, 1] - neck) * body_e
    hip2 = neck + (ymin + 0.50 * H - neck) * body_e     # hip after body scaling
    # extra leg shortening/lengthening about the (new) hip line
    legs = V[:, 1] < hip2
    V[legs, 1] = hip2 + (V[legs, 1] - hip2) * leg_f
    # head enlarged about the neck point (all dims) + a young cranial (upper-head) bulge
    hm = V[:, 1] > neck
    if hm.any():
        piv = np.array([V[hm, 0].mean(), neck, V[hm, 2].mean()])
        V[hm] = piv + (V[hm] - piv) * head_g
        crown = V[:, 1] > neck + 0.5 * (V[hm, 1].max() - neck)   # upper cranium
        V[crown, 1] = neck + (V[crown, 1] - neck) * (1.0 + 0.18 * hf)
    # arm span about the midline (arm band only) -- shorter reach when young
    span = np.percentile(np.abs(V[:, 0]), 60) + 1e-9
    am = (np.abs(V[:, 0]) > 1.4 * span) & (V[:, 1] > hip2) & (V[:, 1] < neck)
    V[am, 0] = np.sign(V[am, 0]) * (1.4 * span + (np.abs(V[am, 0]) - 1.4 * span) * arm_e)
    # chubbier young body: widen span (x) globally and depth (z) most in the trunk
    trunk = (V[:, 1] >= hip2) & (V[:, 1] < neck)
    V[:, 0] *= wid
    V[trunk, 2] *= (1.0 + 0.12 * hf)
    return V


# ---- model-native maturation (NO MakeHuman): grow the cloud's own cells up ----------------------
# CRANIAL fates only -- deliberately EXCLUDES "Nervous System"/"Spinal Cord", which run the whole body
# length: including them made mature_cloud read the "head" as spanning the entire body, so it barely
# scaled the real cranium and left an 11-heads-tall pinhead. The head = the cephalic vesicles.
_HEAD_FATES = ("Forebrain", "Eye", "Midbrain", "Hindbrain",
               "Retina", "OlfactoryBulb", "Cerebellum")


def mature_cloud(Q, fate, f, params=None, register=True):
    """Heads-tall allometry applied to the model's OWN cell cloud -- the NCA+LGM path with NO
    MakeHuman mesh anywhere. The final tetrapod cloud IS the embryo (real cells, real fates, real
    topology); here we mature it in place, embryo -> adult, by growing the sub-cranial body while the
    head holds its size, so the classic heads-tall ratio increases (big-headed embryo -> long adult)
    purely by growing the model's own cells. Laid frame (x = AP head at +x, y = DV, z = ML). This
    sets the AXIAL/trunk allometry + the young->old slimming; limb OUTGROWTH is done by grow_limbs.

    The allometry knobs (trunk_e, dv_girth, ml_girth, head_ht0/1) are SEARCHABLE via
    medic.atlas_relax_search so the adult relaxes into the Phase-D atlas; `params=None` uses
    MATURE_DEFAULTS (identity behaviour). f = 0 is the identity so the cloud->maturation boundary is
    seamless; f -> 1 grows the sub-cranial body + head to the searched adult proportions."""
    p = params or MATURE_DEFAULTS
    Q = Q.astype(float).copy()
    x = Q[:, 0]
    xmin, L = x.min(), np.ptp(x) + 1e-9
    headf = [FIDX[n] for n in _HEAD_FATES if n in FIDX]
    hm = np.isin(fate, headf) if fate is not None else np.zeros(len(Q), bool)
    neck = float(Q[hm, 0].min()) if hm.sum() >= 8 else xmin + 0.78 * L
    trunk_e = 1.0 + p["trunk_e"] * f              # sub-cranial axial elongation
    below = x < neck
    Q[below, 0] = neck + (Q[below, 0] - neck) * trunk_e
    # GIRTH: the model trunk is a flat ribbon (no DV depth); build it out with age instead of slimming it,
    # so the adult is a rounded body not a sheet. DV (y) most, ML (z) some, strongest in the trunk band.
    trunk = below & (x > neck - 0.55 * L)
    Q[trunk, 1] *= 1.0 + p["dv_girth"] * f        # dorso-ventral depth grows -> a 3D trunk (not a flat sheet)
    Q[trunk, 2] *= 1.0 + p["ml_girth"] * f
    # REGIONAL TAPER (the Vitruvian canon): trunk width is Hox-addressed along the AP axis -- broad
    # shoulders (thoracic girdle), a pinched waist (lower lumbar), and the pelvic girdle -- NOT a uniform
    # tube. This is what turns the model's ribbon-trunk into a torso silhouette. Modulate ML width by the
    # within-trunk AP level (3 smooth bands); identity at 1.0 / f=0 so the maturation boundary is untouched.
    if trunk.any():
        top, bot = neck, neck - 0.55 * L
        u = np.clip((x[trunk] - bot) / (top - bot + 1e-9), 0.0, 1.0)   # 0 = pelvis, 1 = shoulders
        g = (1.0
             + (p.get("shoulder_w", 1.0) - 1.0) * np.exp(-((u - 0.82) / 0.16) ** 2)
             + (p.get("waist_w", 1.0) - 1.0) * np.exp(-((u - 0.45) / 0.14) ** 2)
             + (p.get("hip_w", 1.0) - 1.0) * np.exp(-((u - 0.12) / 0.16) ** 2))
        Q[trunk, 2] *= 1.0 + (g - 1.0) * f        # regional ML taper on top of the uniform ml_girth
    # HEAD proportion: scale the head up so the figure is ~human heads-tall (fetus -> adult), not the
    # 20+ heads-tall stick the tiny head + long body gives.
    if hm.sum() >= 8:
        target_ht = p["head_ht0"] + p["head_ht1"] * f
        total = np.ptp(Q[:, 0]) + 1e-9; hlen = np.ptp(Q[hm, 0]) + 1e-9
        # upper clip raised 4.5 -> 9.0: the model's embryo head is under-grown (~4.6% of cells), so with a
        # long adult body the head must scale a lot to reach ~7 heads-tall -- the old cap left a pinhead.
        hs = float(np.clip((total / target_ht) / hlen, 0.8, 9.0))
        hc = Q[hm].mean(0)
        Q[hm] = hc + (Q[hm] - hc) * hs            # enlarge the head about its centre to the target proportion
    # HEAD ON THE NOTOCHORD (genomic articulation): the cranium forms around the axial midline (notochord /
    # prechordal plate), so its dorsoventral position is PINNED to that axis -- the same organizer the vertebral
    # canal registered to. Register the head's DV onto the notochord's DV so the skull sits on the body axis
    # rather than floating fore/aft, instead of a hand-set geometric nudge.
    if hm.sum() >= 8:
        noto = FIDX.get("Notochord")
        ref = None
        if noto is not None and fate is not None and (fate == noto).sum() >= 8:
            ref = float(np.median(Q[fate == noto, 1]))            # the axial organizer (genomic reference)
        elif below.any():
            ref = float(np.median(Q[below, 1]))                   # fallback: the trunk axis
        if ref is not None:
            Q[hm, 1] += f * (ref - np.median(Q[hm, 1]))
    # HEAD DEPTH (anisotropic): the model's cranium is a flattened DISC -- it gets the head-scale (hs, isotropic)
    # but NO dv_girth (that is trunk-only), so it stays shallow front-to-back (~0.09 H vs the real ~0.13). The
    # cephalic cells are under-produced in DV, so give the head its OWN dorsoventral expansion about its axis
    # (the same build-out-the-depth move dv_girth makes for the trunk), so the skull reads as a rounded head not
    # a coin. Blended by f (f=0 identity). Capped by the cranial term of the DV envelope just below.
    if hm.sum() >= 8 and p.get("head_dv", 0.0) and f > 0:
        hcy = float(np.median(Q[hm, 1]))
        Q[hm, 1] = hcy + (Q[hm, 1] - hcy) * (1.0 + p["head_dv"] * f)
    # ANTHROPOMETRIC DV ENVELOPE (regional): a man's dorsoventral depth VARIES along the body -- chest ~0.15 H,
    # but head / neck / limbs ~0.10 -- so the single trunk dv_girth knob leaves the deep cranial region humping
    # at the head-neck junction (the dorsal hump). Clamp each AP slice's DV depth to the anthropometric envelope,
    # compressing about the axial (notochord) midline so the body stays on its axis. Blended by f (f=0 identity).
    stature = np.ptp(Q[:, 0]) + 1e-9
    xx = Q[:, 0]
    noto2 = FIDX.get("Notochord")
    axis_dv = (float(np.median(Q[fate == noto2, 1])) if (noto2 is not None and fate is not None
               and (fate == noto2).sum() >= 8) else float(np.median(Q[:, 1])))
    nb = 24
    for i in range(nb):
        lo = xx.min() + i / nb * stature
        m = (xx >= lo) & (xx < lo + stature / nb)
        if m.sum() < 8:
            continue
        sl = Q[m, 1]
        depth = (np.percentile(sl, 95) - np.percentile(sl, 5)) / stature
        u = (i + 0.5) / nb                                                    # AP fraction (0 = feet, 1 = crown)
        cap = (0.095                                                          # baseline (thin: limbs / neck)
               + 0.055 * np.exp(-((u - 0.68) / 0.13) ** 2)                    # chest ~0.15 (narrower tail: keeps the neck thin)
               + 0.040 * np.exp(-((u - 0.95) / 0.05) ** 2))                   # cranium ~0.135 (let the head fill FB)
        if depth > cap:
            s = 1.0 - f * (1.0 - cap / depth)
            idx = np.where(m)[0]
            Q[idx, 1] = axis_dv + (Q[idx, 1] - axis_dv) * s
    if register:
        Q = _visceral_ap_register(Q, fate, f)      # migrate each organ to its Hox-addressed axial level
        Q = _ventral_viscera_spread(Q, fate, f)     # spread the viscera off the dorsal wall to fill the coelom
    return Q


# organs that fill the COELOM (the ventral body cavity) by spreading VENTRALLY off the dorsal wall, giving the
# chest + belly their front-to-back depth. Retroperitoneal organs (kidney/nephron/adrenal) stay dorsal, so they
# are excluded, as are the axial structures (spinal cord/notochord/vertebrae).
_VENTRAL_VISCERA = ("Heart", "Atrium", "Ventricle", "Outflow", "Lung", "Liver", "Spleen", "Pancreas",
                    "Stomach", "Gut", "Bladder", "Thymus")


def _ventral_viscera_spread(Q, fate, f, fill=0.20, wall_frac=0.72):
    """THE VENTRAL-MIGRATION HEAD: the viscera are specified dorsally, against the foregut/notochord, then fill
    the coelom as the lateral-plate mesoderm splits and the ventral body wall closes. Measured, the ventral
    viscera already sit part-way into the cavity (~43--63% of the way from the spine to the ventral wall), but
    the heart/ventricle lags most dorsal; this brings each ventral viscus a fraction `fill` of its REMAINING way
    toward the ventral wall (an additive fill, anchored on the dorsal wall as reference), so the heart comes
    forward and the belly evens out, capped short of the wall (`wall_frac`) so no viscus touches the skin. It is
    a mild refinement, not a large shift, and it does not set the chest DEPTH -- that is the ribcage/body wall,
    a separate mechanism. Retroperitoneal organs (kidney/adrenal) stay dorsal by exclusion. Blended by f."""
    if fate is None or f <= 0:
        return Q
    y = Q[:, 1]; ymid = float(np.median(y))
    spine = ymid; dsn = 1.0                           # dorsal reference = the spine; ventral = opposite it
    for nm in ("Notochord", "Spinal Cord"):
        fid = FIDX.get(nm)
        if fid is not None and (fate == fid).sum() > 20:
            spine = float(np.median(y[fate == fid])); dsn = -1.0 if spine >= ymid else 1.0
            break
    wall = float(np.percentile(dsn * (y - spine), 98))            # the ventral body wall, as an offset from spine
    for nm in _VENTRAL_VISCERA:
        fid = FIDX.get(nm)
        if fid is None:
            continue
        m = fate == fid
        if m.sum() < 8:
            continue
        off = dsn * (Q[m, 1] - spine)                 # ventral offset from the spine (>0 = ventral)
        room = np.clip(wall_frac * wall - off, 0.0, None)         # remaining room to the (fractional) wall
        Q[m, 1] += dsn * f * fill * room              # additive fill: move a fraction of the way forward
    return Q


# Canonical antero-posterior level of each discrete organ (fraction of standing height, crown = 1, sole = 0),
# from standard adult anatomy -- the Hox-addressed segmental level the organ's condensed mass belongs at. Only
# discrete condensed organs are addressed; spanning structures (Gut/Notochord/Spinal Cord/Vessel) and the axial
# segmental series (Rib/Cartilage/Muscle/Somite) are NOT -- they occupy a range of levels by design.
ORGAN_AP_ADDRESS = {
    "Forebrain": 0.94, "Eye": 0.93, "Retina": 0.93, "Midbrain": 0.92, "Otic": 0.91, "Hindbrain": 0.90,
    "Cerebellum": 0.89, "Thymus": 0.78, "Lung": 0.74, "Outflow": 0.73, "Atrium": 0.72, "Heart": 0.71,
    "Ventricle": 0.70, "Spleen": 0.66, "Liver": 0.65, "Pancreas": 0.63, "Adrenal": 0.62, "Kidney": 0.60,
    "Nephron": 0.60, "Bladder": 0.48,
}


def _visceral_ap_register(Q, fate, f):
    """THE AP-ADDRESS HEAD: slide each discrete organ's condensed mass along the body axis to its canonical
    Hox-addressed level, read-only on every other cell. The head/notochord anchor apf=1..0.9; below them the
    build_base cloud snaps organs onto coarse body-electric antinodes near the middle, so the viscera sag and
    scramble (the heart drops onto the kidneys). This is the generalisation of the single hand-tuned cardiac
    descent: each organ is a coherent condensed body that migrates to its axial address. A pure AP translation
    (the organ keeps its shape + girth), blended by f so f=0 is the identity and the maturation boundary is
    seamless. Fixes both the sag (offsets) and the order (inversions) in one pass."""
    if fate is None or f <= 0:
        return Q
    x = Q[:, 0]; xmin = x.min(); stat = np.ptp(x) + 1e-9
    for nm, target in ORGAN_AP_ADDRESS.items():
        fid = FIDX.get(nm)
        if fid is None:
            continue
        m = fate == fid
        if m.sum() < 8:
            continue
        cur = float(((x[m] - xmin) / stat).mean())
        Q[m, 0] += f * (target - cur) * stat        # translate the organ to its address (read-only elsewhere)
    return Q


def _limb_grow_model(t, ext=1.5):
    """Limb-outgrowth schedule for the model-native maturation. Continues smoothly from the cloud
    phase (which uses _limb_grow, reaching 0.14 at the end of the cloud at t=0.55) and then elongates
    the buds PAST their cloud extent (grow > 1) into long adult limbs -- limbs are the fastest-growing
    part of the body (allometric exponent ~1.3), so they must extend well beyond the embryonic bud."""
    if t < 0.55:
        return _limb_grow(t)                     # cloud phase: buds emerge (0 -> 0.14)
    if t < 0.95:
        return 0.14 + (ext - 0.14) * (t - 0.55) / 0.40   # bud -> a natural limb (ext = adult extension knob)
    return ext


def _chest(Q, reg_fate):
    """The chest anchor in the laid frame: centroid of the (identity-constant) heart-region
    vertices. Because reg_fate labels the SAME cells every frame, anchoring this point keeps the
    chest fixed in space as the body matures -- the figure grows AROUND the chest instead of
    sliding when its proportions (big head -> small head, short legs -> long legs) shift the
    naive centroid. Falls back to an upper-trunk AP band if the heart pocket is too sparse."""
    hm = reg_fate == FIDX["Heart"]
    if hm.sum() >= 8:
        return Q[hm].mean(0)
    ap = Q[:, 0]
    frac = (ap - ap.min()) / (np.ptp(ap) + 1e-9)     # 0 = tail, 1 = head (head at +x)
    band = (frac > 0.55) & (frac < 0.72)             # upper trunk, just below the head
    return Q[band].mean(0) if band.any() else Q.mean(0)


# ---- fetal curl (the sagittal C-posture) --------------------------------------------------------
# Real embryos/fetuses are curled into a ventral C -- chin toward chest, spine flexed, hips drawn up
# -- and only extend toward the standing adult. We reuse the body_fold.py principle: the curl is the
# INTEGRAL of a ventral curvature profile along the AP axis (a body-wide spinal curl + a cephalic
# head-tuck + a hip/pelvic flexure), and every point rides the bent axis on its normal at its true
# DV offset. Bending is in the sagittal (AP-DV = x-y) plane; ML width (z) is untouched. `curl` in
# [0,1] scales the whole thing: 1 = full fetal C, 0 = straight. Ventral (+y here, the belly/front
# side) is concave.
_CURL_SIGN = 1.0         # sign that makes the body concave toward the ventral (+y) side (chin-to-chest)


def fetal_curl(Q, curl, fate=None):
    """Curl the body into the sagittal fetal C. ORIENTATION (fate given): a real fetal C has the DORSAL
    spine on the CONVEX (outer) back and the belly on the concave inside; the model's dorsal is at +y but
    the raw sign put the spine on the INSIDE (Miles: "folding with the spine inside"). So we compute the
    bend for both signs and keep the one that puts the spine/neural-tube cells further from the curl centre
    (convex). Without a fate array we fall back to the fixed _CURL_SIGN."""
    if curl <= 1e-3:
        return Q
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    nb = 60
    edges = np.linspace(x.min(), x.max(), nb + 1); ctr = 0.5 * (edges[:-1] + edges[1:])
    axis = np.full(nb, np.nan)
    for i in range(nb):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() > 3:
            axis[i] = np.median(y[m])                       # DV midline at this AP level
    ok = ~np.isnan(axis)
    if ok.sum() < 3:
        return Q
    axis = np.interp(ctr, ctr[ok], axis[ok])
    for _ in range(6):
        axis[1:-1] = 0.25 * axis[:-2] + 0.5 * axis[1:-1] + 0.25 * axis[2:]
    L = float(ctr[-1] - ctr[0]) + 1e-9; ds = L / (nb - 1)
    s_ap = (ctr - ctr[0]) / L                               # 0 = tail/feet .. 1 = head (head at +x)
    body = np.ones(nb) / nb                                 # body-wide spinal curl
    g_ceph = np.exp(-((s_ap - 0.85) / 0.09) ** 2); g_ceph /= g_ceph.sum() + 1e-9   # cephalic head-tuck
    g_hip = np.exp(-((s_ap - 0.22) / 0.11) ** 2); g_hip /= g_hip.sum() + 1e-9      # hip/pelvic flexion
    kappa = curl * (np.radians(135.0) * body + np.radians(75.0) * g_ceph + np.radians(85.0) * g_hip)
    sx = np.clip((x - ctr[0]) / L, 0, 1) * (nb - 1)
    i0 = np.floor(sx).astype(int); i1 = np.minimum(i0 + 1, nb - 1); tt = (sx - i0)[:, None]
    dv = (y - np.interp(x, ctr, axis))[:, None]             # signed DV offset from the axis

    def _bend(sign):
        theta = sign * np.cumsum(kappa); theta -= theta.mean()        # centred orientation
        C = np.c_[np.cumsum(np.cos(theta)) * ds, np.cumsum(np.sin(theta)) * ds]; C -= C.mean(0)
        T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
        Nn = np.c_[-T[:, 1], T[:, 0]]                       # unit normal of the bent axis
        Cc = C[i0] * (1 - tt) + C[i1] * tt
        Nc = Nn[i0] * (1 - tt) + Nn[i1] * tt; Nc /= np.linalg.norm(Nc, axis=1, keepdims=True) + 1e-9
        out = np.empty_like(Q)
        out[:, :2] = Cc + dv * Nc
        out[:, 2] = z                                       # ML width carried unchanged
        return out

    if fate is not None:
        spine = np.isin(fate, [FIDX[n] for n in ("Spinal Cord", "Nervous System", "Forebrain",
                               "Midbrain", "Hindbrain", "Cerebellum") if n in FIDX])
        if spine.sum() >= 8:
            op, om = _bend(1.0), _bend(-1.0)
            rp = np.linalg.norm(op[spine, :2] - op.mean(0)[:2], axis=1).mean()   # spine reach, sign +1
            rm = np.linalg.norm(om[spine, :2] - om.mean(0)[:2], axis=1).mean()   # spine reach, sign -1
            return op if rp >= rm else om                   # keep the sign with the spine further OUT (convex)
    return _bend(_CURL_SIGN)


def tuck_limbs(Q, reg_fate, amt):
    """Flex the limbs into the fetal tuck (arms folded toward the chest, knees drawn up to the belly)
    by `amt` in [0,1], before the axial curl carries them. On the laid frame (x=AP head+x, y=DV with
    +y ventral, z=ML): arms (anterior limb points) are pulled in laterally toward the shoulder and
    drawn ventrally (distal hand points move most = a fold); legs (posterior limb points) are shortened
    up toward the hip and drawn ventrally (distal foot points most = knees-to-belly). amt=0 leaves the
    full T-pose/standing extension untouched, so as the body unfurls the limbs open out."""
    if amt <= 1e-3:
        return Q
    Q = Q.copy()
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    limb = np.isin(reg_fate, [FIDX["Limb Bud"], FIDX["Cartilage"]])
    arm = limb & (apf >= 0.5)
    leg = limb & (apf < 0.5)
    if arm.any():
        za = z[arm]
        Q[arm, 2] = za * (1 - 0.60 * amt)                   # draw hands in toward the shoulder (ML)
        Q[arm, 1] = y[arm] + 0.9 * amt * np.abs(za)         # fold ventrally, distal (hands) most
    if leg.any():
        hipx = np.percentile(x[leg], 85)                    # near the top of the legs (hip)
        below = np.clip(hipx - x[leg], 0, None)             # distance down the leg (0 at hip)
        Q[leg, 0] = hipx - (hipx - x[leg]) * (1 - 0.55 * amt)   # draw knees up toward the hip (AP)
        Q[leg, 1] = y[leg] + 1.0 * amt * below              # tuck knees/feet to the belly (ventral)
    return Q


def grow_limbs(Q, limb, grow, leg_grow=None, pose=None):
    """Grow the limbs OUT from buds AND swing them down to a standing pose. `grow`/`leg_grow` set limb LENGTH
    (extension past the bud); `pose` in [0,1] sets the arms-down/legs-down ROTATION (0 = T-pose/splayed bud,
    1 = hanging at the side). Laid frame (x=AP head+x, z=ML).

    ★ pose is DECOUPLED from grow. Previously pose = clip(grow-1,0,1), so the arms only hung down if the arm
    was ALSO extended (grow>1); when the maturation search set limb_ext=1.0 (short arms) the arms silently
    stayed T-posed. Now `pose` is driven by the maturation STAGE (passed by the caller), so the arms hang down
    at the adult even if they are short. pose=None keeps the old grow-derived behaviour for back-compat."""
    lg = grow if leg_grow is None else leg_grow               # legs extend on their OWN knob (leg_ext)
    if not limb.any():
        return Q
    pose_a = 0.0                                               # arms OUT (the Vitruvian pose) -- extend laterally, don't hang
    pose_l = float(np.clip(lg - 1.0, 0.0, 1.0)) if pose is None else float(np.clip(pose, 0.0, 1.0))
    if abs(grow - 1.0) < 1e-3 and abs(lg - 1.0) < 1e-3 and pose_a < 1e-3 and pose_l < 1e-3:
        return Q
    Q = Q.copy()
    x, z = Q[:, 0], Q[:, 2]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    arm = limb & (apf >= 0.5)
    leg = limb & (apf < 0.5)
    body = ~limb
    w = np.percentile(np.abs(z[body]), 70) if body.any() else 0.12 * (np.ptp(z) + 1e-9)
    # NECK: seat the shoulders BELOW the head. The arm bud sits too high (its top reaches up to the jaw), so there
    # is no cervical gap and the arms fan out at the mouth level (reads as a "wide mouth"). Drop the arm cells so
    # the shoulder is ~0.22 H below the crown, opening a neck between the head and the shoulders.
    if arm.any():
        Hh = np.ptp(x) + 1e-9
        target = x.max() - 0.22 * Hh                          # shoulder AP target (below head + neck)
        top = np.percentile(x[arm], 95)                       # current shoulder (arm top)
        if top > target:
            Q[arm, 0] -= (top - target)
            x = Q[:, 0]                                       # refresh the AP view (arm mask kept by fate, unchanged)
    if arm.any() and pose_a < 1e-3 and grow <= 1.0:
        # bud stage (cloud/fetal): the arm buds splay outboard in ML as they emerge (unchanged behaviour).
        za = z[arm]; sgn = np.sign(za); a = np.abs(za)
        Q[arm, 2] = sgn * (np.minimum(a, w) + np.clip(a - w, 0, None) * grow)
    elif arm.any():
        # VITRUVIAN STANDING: extend each arm LATERALLY (out to the side, along ML) from the shoulder by `grow`,
        # so the arms reach the canonical span (~0.44 H each), then swing DOWN by pose_a (0 = arms out = Vitruvian,
        # 1 = hanging at the side). Extending laterally (not only by rotation) is what gives the arms real length.
        g = max(grow, 1.0)
        for sgn in (1.0, -1.0):
            m = arm & (np.sign(z) == sgn)
            if m.sum() < 4:
                continue
            sh_x = np.percentile(x[m], 85)                       # shoulder = top (AP) of this arm
            sh_z = sgn * w                                       # shoulder sits at the trunk's side
            out_z = sh_z + (z[m] - sh_z) * g                     # arms-OUT: extend laterally from the shoulder
            reach = np.abs(out_z - sh_z)                         # along-arm reach (now the extended length)
            Q[m, 0] = (1 - pose_a) * x[m] + pose_a * (sh_x - reach)          # swing down by pose_a
            Q[m, 2] = (1 - pose_a) * out_z + pose_a * (sh_z * (1 - 0.15 * reach / (reach.max() + 1e-9)))
    if arm.any():                                            # TRIM the faint outermost arm tail (runs after EITHER
        for sgn in (1.0, -1.0):                              # branch above), clamping the lateral reach to the solid arm
            m = arm & (np.sign(Q[:, 2]) == sgn)
            if m.sum() < 4:
                continue
            sh_z = sgn * w
            off = np.abs(Q[m, 2] - sh_z)
            # define the arm EDGE by the BULK of the cells (88th pct), not a high percentile that the tail
            # itself drags outward -- then hard-clamp everything to a hair past that edge. The old 94th-pct
            # cap left a faint ~0.5% tail out to 0.77 H (solid arm reaches ~0.60 H); this kills it.
            cap = 1.06 * np.percentile(off, 88)
            Q[m, 2] = sh_z + np.sign(Q[m, 2] - sh_z) * np.minimum(off, cap)
    if leg.any() and pose_l < 1e-3 and lg <= 1.0:
        hipx = np.percentile(x[leg], 88)
        Q[leg, 0] = hipx - (hipx - x[leg]) * lg                                 # bud stage: just grow the length
    elif leg.any():
        # STANDING: legs hang DOWN from the HIPS, seated at the pelvis width -- NOT 0.85x the (wider) trunk width,
        # which splayed the legs out beyond the hips (the wide-hip look). Use the body's ML extent AT the hip
        # level as the leg line, and tighten each leg into a column so it doesn't fan out.
        g = max(lg, 1.0)
        hipx0 = np.percentile(x[leg], 88)
        Hh = np.ptp(x) + 1e-9
        hipband = body & (np.abs(x - hipx0) < 0.06 * Hh)
        hipz = np.percentile(np.abs(z[hipband]), 70) if hipband.sum() > 8 else 0.85 * w   # the PELVIS half-width
        hipz = min(hipz, 0.085 * Hh)                             # cap to a real hip attachment (~0.17 H bi-femoral)
        for sgn in (1.0, -1.0):
            m = leg & (np.sign(z) == sgn)
            if m.sum() < 4:
                continue
            hipx = np.percentile(x[m], 88)                       # hip = top (AP) of this leg
            hz = sgn * hipz                                      # seat the leg at the hip width (not the trunk width)
            Q[m, 0] = hipx - (hipx - x[m]) * g                   # lengthen the leg downward on leg_ext
            Q[m, 2] = hz + (z[m] - hz) * (1.0 - 0.85 * pose_l)   # tighten hard into a column (0.55 -> 0.85)
    return Q


def _limb_grow(t):
    """Limb-outgrowth schedule vs the global clock: no limbs while the body is a ball, buds emerge
    in the late cloud, stay small nubs through the handoff + early fetus, then grow out over the
    fetal->child window to full length. (Individual/allometric limb PROPORTION at each mature stage
    still rides on top via the mesh, but the bud->limb emergence lives here.)"""
    if t < 0.42:
        return 0.0
    if t < 0.55:
        return 0.14 * (t - 0.42) / 0.13         # buds emerge in the late cloud
    if t < 0.68:
        return 0.14 + 0.16 * (t - 0.55) / 0.13  # small nubs through handoff + early fetus (0.14 -> 0.30)
    if t < 0.95:
        return 0.30 + 0.70 * (t - 0.68) / 0.27  # grow out, fetus -> child (0.30 -> 1.0)
    return 1.0


def _curl_amt(t):
    """Developmental curl schedule vs the global clock t: straight while the body is still a ball,
    ramp to a full fetal C as the embryo forms and through the handoff + early fetus, then unfurl
    across the fetus->child window to a standing (straight) adolescent/adult."""
    if t < 0.30:
        return 0.0
    if t < 0.55:
        return (t - 0.30) / 0.25            # ramp up as the embryo takes shape
    if t < 0.68:
        return 1.0                          # full fetal C: end of cloud, handoff, early fetus
    if t < 0.90:
        return max(0.0, 1.0 - (t - 0.68) / 0.22)   # unfurl fetus -> child
    return 0.0                              # standing adolescent / adult


def to_laid(V):
    """Upright human (x span, y height, z depth) -> laid cloud frame (AP=x head at +x, DV=y, ML=z),
    centred at origin. Matches the cloud's head-at-+x, long-axis-horizontal convention."""
    ap = V[:, 1] - V[:, 1].mean()     # height -> AP (head high -> +x)
    dv = V[:, 2] - V[:, 2].mean()     # depth  -> DV
    ml = V[:, 0] - V[:, 0].mean()     # span   -> ML
    return np.stack([ap, dv, ml], 1)


# ---------------------------------------------------------------- Hill-organ atlas (adult reveal)
def _atlas_to_laid(P):
    """menagerie atlas coords (x = AP depth, y = mediolateral, z = up, head high) -> the movie's
    laid frame (AP = x with head at +x, DV = y, ML = z). So head(z max) -> +x."""
    return np.stack([P[:, 2], P[:, 0], P[:, 1]], 1)


def _teeth_points(bones, rng):
    """Teeth are NOT in the homologous roster (no tooth bone) -- they are ADDED here: two dental
    arches (maxilla + mandible) of ~16 teeth each, seated at the front of the skull/jaw. Returned
    in atlas coords so the caller maps them with everything else."""
    bmap = {b.name: b for b in bones}
    cran = bmap.get("cranium"); mand = bmap.get("mandible")
    if cran is None or mand is None:
        return np.zeros((0, 3), np.float32)
    face_x = max(cran.b[0], mand.b[0]) + 0.02          # front of the face (+x forward)
    z_up = mand.b[2] + 0.02                             # lower arch height
    z_lo = mand.b[2] - 0.02
    pts = []
    for arch_z in (z_up + 0.06, z_up):                 # maxilla (upper) then mandible (lower)
        for k in range(16):
            u = (k / 15.0) * 2 - 1                      # -1..1 across the arch
            y = 0.09 * u                                # mediolateral spread
            x = face_x - 0.10 * u * u                   # parabolic arch, recedes at the back
            c = np.array([x, y, arch_z])
            pts.append(c + rng.normal(size=(6, 3)) * 0.006)   # a small blob per tooth
    return np.vstack(pts).astype(np.float32)


def build_human_atlas(rng):
    """Decode the human genome into the Hill-organ atlas (bones + Hill-muscles + viscera), ADD the
    teeth, map it into the adult movie frame, and subsample to N_R points (viscera + teeth kept
    preferentially). Returns (positions[N_R,3], fate[N_R] as ATLAS_BASE+class, counts dict)."""
    g = reference_genome("human_male")
    bones = build_skeleton(g)
    muscles = build_muscles(g, bones)
    organs = build_organs(g, bones)

    P, F = [], []
    bp, _ = _bone_pts(bones, bone_rgb=(0.85, 0.83, 0.78))
    P.append(bp); F.append(np.full(len(bp), ATLAS_BASE + ATLAS_IDX["bone"]))
    mp, _ = _musc_pts(muscles)
    P.append(mp); F.append(np.full(len(mp), ATLAS_BASE + ATLAS_IDX["muscle"]))
    for o in organs:                                    # tag each viscus by its class
        base = o.name[:-2] if o.name.endswith((" R", " L")) else o.name
        cls = base if base in ATLAS_IDX else "bone"
        op, _ = _organ_pts([o])
        P.append(op); F.append(np.full(len(op), ATLAS_BASE + ATLAS_IDX[cls]))
    tp = _teeth_points(bones, rng)
    if len(tp):
        P.append(tp); F.append(np.full(len(tp), ATLAS_BASE + ATLAS_IDX["teeth"]))

    P = np.vstack(P); F = np.concatenate(F)
    Q = _atlas_to_laid(P)
    Q = Q - Q.mean(0)
    Q *= 3.2 / _long_axis_len(Q)                        # match the adult body length
    Q = Q - Q.mean(0)

    # subsample to N_R with per-class QUOTAS so all three tissue systems stay legible: the skeleton
    # and muscle form the scaffold (many small points), the viscera + teeth pop (fewer, hi = bigger).
    bone_f = ATLAS_BASE + ATLAS_IDX["bone"]
    musc_f = ATLAS_BASE + ATLAS_IDX["muscle"]
    teeth_f = ATLAS_BASE + ATLAS_IDX["teeth"]
    groups = {"bone": (F == bone_f, 0.40), "muscle": (F == musc_f, 0.20),
              "teeth": (F == teeth_f, 0.06),
              "organ": (~np.isin(F, [bone_f, musc_f, teeth_f]), 0.34)}
    picks = []
    for _name, (mask, frac) in groups.items():
        pool = np.where(mask)[0]
        if not len(pool):
            continue
        take = min(len(pool), int(round(frac * N_R)))
        picks.append(rng.choice(pool, take, replace=False))
    idx = np.concatenate(picks)
    if len(idx) < N_R:                                      # top up from the biggest remaining pool
        rest = np.setdiff1d(np.arange(len(F)), idx)
        if len(rest):
            idx = np.concatenate([idx, rng.choice(rest, min(len(rest), N_R - len(idx)), replace=False)])
    idx = rng.permutation(idx)[:N_R]
    counts = dict(bones=len(bones), muscles=len(muscles), organs=len(organs),
                  organ_names=[n for n, _ in _ORGAN_NAMES])
    return Q[idx], F[idx], counts


def panel_anatomy(counts):
    """The Hill-organ roster for the adult reveal: skeleton / muscle / viscera (in situ) + teeth
    (ADDED). Rendered by the viewer's atlas branch, grouped by ATLAS_LAYERS."""
    rows = [dict(name=f"Skeleton — {counts['bones']} bones", layer="Skeleton",
                 role="axial + appendicular, endochondral (in situ)", mag=1.0, added=False, firing="atlas"),
            dict(name=f"Hill muscles — {counts['muscles']} groups", layer="Muscle",
                 role="origin → insertion on named bones (in situ)", mag=0.85, added=False, firing="atlas")]
    sizes = {r[0]: r[4] for r in ORGAN_PLAN}
    for n in counts["organ_names"]:
        rows.append(dict(name=n, layer="Viscera", role="homologous viscus, placed in situ",
                         mag=round(min(1.0, 0.35 + sizes.get(n, 0.4)), 2), added=False, firing="atlas"))
    rows.append(dict(name="Teeth — 32 (2 arches)", layer="Teeth",
                     role="ADDED — no tooth in the homologous roster; seated on maxilla + mandible",
                     mag=0.6, added=True, firing="atlas"))
    return rows


# ---------------------------------------------------------------- panel
def _frac_of(fid, fates):
    if not len(fid):
        return 0.0
    want = np.array([FIDX[f] for f in fates if f in FIDX])
    return float(np.isin(fid, want).mean())


def panel_cloud(fid, prc2, t_local):
    """t_local in [0,1] within the cloud phase. prc2 runs ~0.98 (early) -> ~0.2 (late)."""
    rows = []
    for g, r in GENE_ROLES.items():
        c = r["cloud"]
        if not c:
            continue
        role, fates = c
        if fates is None:                                  # clock-gated morphogenetic head
            if g.startswith("Division"):
                mag = float(np.clip(prc2, 0, 1))
            elif g.startswith("Differentiation"):
                mag = float(np.clip(1 - prc2, 0, 1))
            else:                                          # electric-body frame: fires late
                mag = float(np.clip((0.46 - prc2) / 0.46 + 0.1, 0, 1)) if prc2 <= 0.5 else 0.05
        else:
            mag = min(1.0, _frac_of(fid, fates) * 3.5)     # gain so sparse organs are visible
        if mag < 0.03:
            continue
        rows.append(dict(name=g, role=role, mag=round(mag, 3), layer=LAYER[g],
                         span=(g in SPANNING), firing="cloud"))
    rows.sort(key=lambda d: (-d["span"], -d["mag"]))
    return rows


def panel_mesh(t_local, morphing):
    """t_local in [0,1] across handoff+maturation. A knob's bar = its real fitted SPECIES move
    (weighted by the reshape window, peaking at the handoff) PLUS its real MATURATION move
    (weighted by the maturity ramp toward the adult). Body-plan knobs therefore keep firing into
    the adult (e.g. CDX2/HOX-L elongation, SHH-kappa straightening -- both large real maturation
    moves); adapter/growth knobs ramp in only with maturity."""
    rows = []
    reshape = float(np.clip(1 - t_local * 1.6, 0, 1))       # species-adapter move fires at handoff
    mature = float(np.clip((t_local - 0.15) / 0.85, 0, 1))  # maturation move ramps to adult
    for g, r in GENE_ROLES.items():
        m = r["mesh"]
        if not m:
            continue
        role, knob = m
        mag = SPECIES_MAG.get(knob, 0.0) * reshape + MATURE_MAG.get(knob, 0.0) * mature
        if mag < 0.03:
            continue
        rows.append(dict(name=g, role=role, mag=round(min(1.0, mag), 3), layer=LAYER[g],
                         span=(g in SPANNING), firing="mesh"))
    rows.sort(key=lambda d: (-d["span"], -d["mag"]))
    return rows


# ---------------------------------------------------------------- build
def _emit(Q, fate, vm, phase, stage, t, panel, ptype="genes", skin=None, skin_op=0.0):
    d = dict(phase=phase, stage=stage, t=round(float(t), 3),
             xyz=[round(float(x), 2) for x in Q.ravel()],
             vm=[round(float(x)) for x in vm],
             fate=[int(x) for x in fate],
             panel=panel, ptype=ptype)
    if skin is not None:                                    # per-frame skin-surface mesh verts (faces in doc)
        d["skin"] = [round(float(x), 2) for x in np.asarray(skin).ravel()]
        d["skin_op"] = round(float(skin_op), 2)
    return d


def _fit_atlas_to_model(A_pos, Qm, reg_fate):
    """Warp the Hill-organ atlas per BODY REGION onto the model's adult limb positions, so the internal
    parts appear WHERE THE MODEL'S LIMBS ARE (Miles: "the arms are not where the limbs were" -- the atlas
    is arms-down while the model matures in a T-pose, so at the dissolve the arms jumped). Per-region affine:
    for head / trunk / each arm / each leg, translate + scale the atlas region to the model region's
    centroid and spread on each axis, so e.g. the atlas's arm bones + muscles land inside the model's arm."""
    lim_m = np.isin(reg_fate, [FIDX[n] for n in ("Limb Bud", "Cartilage") if n in FIDX])
    reg_m = _body_regions(Qm, lim_m)
    apf = (A_pos[:, 0] - A_pos[:, 0].min()) / (np.ptp(A_pos[:, 0]) + 1e-9)
    hw = np.percentile(np.abs(A_pos[:, 2]), 75) + 1e-9
    arm = (np.abs(A_pos[:, 2]) > hw) & (apf > 0.42) & (apf < 0.9)   # atlas arms: lateral AND at shoulder-hand AP
    leg = apf < 0.34                                               # atlas legs: low AP
    lim_a = arm | leg
    reg_a = _body_regions(A_pos, lim_a)
    out = A_pos.astype(float).copy()
    for r in range(6):
        ma = reg_a == r; mm = reg_m == r
        if ma.sum() < 4 or mm.sum() < 6:
            continue
        for ax in range(3):
            ca = A_pos[ma, ax].mean(); sa = A_pos[ma, ax].std() + 1e-6
            cm = float(Qm[mm, ax].mean()); sm = float(Qm[mm, ax].std()) + 1e-6
            out[ma, ax] = cm + (A_pos[ma, ax] - ca) * float(np.clip(sm / sa, 0.5, 1.8))
    return out.astype(np.float32)


def build(source="model", json_path=None):
    """Build the movie. source='model' matures the NCA+LGM cloud's own cells (no MakeHuman);
    source='makehuman' is the old mannequin-readout path, kept for side-by-side comparison."""
    jp = Path(json_path) if json_path else JSON
    rng = np.random.default_rng(0)
    skin_faces = None                                       # skin-surface mesh topology (set in the model path)
    skin_adult_v = None                                    # the adult skin verts (dissolves in the reveal)

    # ---- Phase A: cloud morphogenesis (tetrapod) ----
    print(f"[A] growing the cloud 1 -> {CLOUD_N} cells (tetrapod, convergent_ext=1.0) ...")
    frames, _ = simulate(use_ecm=True, seed=0, n_start=1, n_end=CLOUD_N,
                         limb_buds=True, convergent_ext=1.0, limb_params=LIMB_SEARCHED,
                         fate_params=fate_map_for(CLOUD_N))
    sym = [_symmetrize(P, V, F) for (_, _, _, P, V, F) in frames]
    Pf = sym[-1][0]
    c = Pf.mean(0); c[2] = 0.0
    cloud_scale = 0.9 / _long_axis_len(Pf)                  # final cloud long axis -> 0.9

    nfr = len(frames)
    proc = [flex(shape_limbs((Ps - c) * cloud_scale, Fs, fi / (nfr - 1), LIMB), fi / (nfr - 1))
            for fi, (Ps, _, Fs) in enumerate(sym)]
    cc = proc[-1].mean(0)
    proc = [Q - cc for Q in proc]

    # orient head to +x (head = anterior neural fates)
    headf = [FIDX[n] for n in ("Forebrain", "Eye", "Midbrain", "Hindbrain") if n in FIDX]
    hmask = np.isin(sym[-1][2], headf)
    if hmask.any() and proc[-1][hmask, 0].mean() < 0:
        proc = [np.stack([-Q[:, 0], Q[:, 1], Q[:, 2]], 1) for Q in proc]

    def subsample(Q, V, F, fixed=None):
        idx = fixed if (fixed is not None and len(Q) > N_R) else _rep_idx(F, N_R, rng)   # representative draw
        return Q[idx], V[idx], F[idx], idx

    # lock a fixed sample for the FINAL cloud frame = the handoff source S0, with the SAME representative
    # draw as subsample() (so the last cloud frame and the handoff match, and the tetrapod's full width is
    # kept -- a deterministic prefix used to drop the outer limb tips and collapse the width).
    Qf, Vf_, Ff = proc[-1], sym[-1][1], sym[-1][2]
    S0_idx = _rep_idx(Ff, N_R, rng)

    out_frames = []
    T_A = 0.55
    for fi in range(nfr):
        Q, V, F, _ = subsample(proc[fi], sym[fi][1], sym[fi][2],
                               fixed=(S0_idx if fi == nfr - 1 else None))
        born, t_hpf, prc2 = frames[fi][0], frames[fi][1], frames[fi][2]
        t = T_A * fi / (nfr - 1)
        Q = grow_limbs(Q, F == LIMB, _limb_grow(t))          # limb buds emerge small, not big paddles
        Q = fetal_curl(Q, _curl_amt(t), F); Q = Q - Q.mean(0)   # curl (spine to the OUTSIDE) into the fetal C
        stage = f"cloud · N={born:,} · {t_hpf:.0f} hpf · PRC2 {prc2:.2f}"
        out_frames.append(_emit(Q, F, V, "cloud", stage, t, panel_cloud(F, prc2, fi / (nfr - 1))))
    print(f"    {nfr} cloud frames")

    # handoff source (locked; unordered -- the region-aware pairing below sets the correspondence)
    S0 = proc[-1][S0_idx]
    S0col_fate = sym[-1][2][S0_idx]

    if source == "makehuman":
        # ================= MakeHuman readout path (the mannequin, kept for comparison) =============
        # ---- MakeHuman mesh: sample N_R vertices, region-colour ----
        z = np.load(MH, allow_pickle=True)
        Vmh = np.asarray(z["V"], float)
        vidx = rng.choice(len(Vmh), N_R, replace=False) if len(Vmh) >= N_R \
            else rng.choice(len(Vmh), N_R, replace=True)
        Vsamp = Vmh[vidx]
        reg_fate = region_fate(Vsamp)

        # embryo target E0 (laid, scaled to 0.9) and its AP-ring order -> reorder to match S0
        E0 = to_laid(mature_body(Vsamp, 0.0)); E0 *= 0.9 / _long_axis_len(E0)
        ord_T = _ap_ring_order(E0)
        Vsamp = Vsamp[ord_T]; reg_fate = reg_fate[ord_T]    # lock mesh identity order = handoff order
        E0 = to_laid(mature_body(Vsamp, 0.0)); E0 *= 0.9 / _long_axis_len(E0)
        E0 = E0 - _chest(E0, reg_fate)          # anchor the chest (matches phase C, no jump at C start)

        # region-aware pairing: match the cloud's four limbs / head / trunk to the human's, so the
        # green limbs flow limb-to-limb and the morph keeps its silhouette (no axial collapse). Only
        # the cloud source is permuted; E0 is untouched, so the last handoff frame is exactly mesh f=0.
        limb_S0 = (S0col_fate == LIMB)
        limb_E0 = np.isin(reg_fate, [FIDX["Limb Bud"], FIDX["Cartilage"]])
        perm = _align_cloud_to_human(S0, _body_regions(S0, limb_S0), E0, _body_regions(E0, limb_E0))
        S0 = S0[perm]; S0col_fate = S0col_fate[perm]

        # ---- Phase B: handoff morph S0 -> E0 ----
        print(f"[B] handoff: morphing the cloud onto the human-embryo surface ({N_HANDOFF} frames) ...")
        for i in range(1, N_HANDOFF + 1):
            a = i / N_HANDOFF
            s = 0.5 - 0.5 * np.cos(np.pi * a)               # smoothstep
            Q = S0 * (1 - s) + E0 * s
            fate = S0col_fate if s < 0.5 else reg_fate      # colours flip to human mid-morph
            vm = np.full(N_R, NEUT)
            t = 0.55 + 0.07 * a
            Q = tuck_limbs(Q, reg_fate, _curl_amt(t))       # flex limbs into the fetal tuck
            Q = fetal_curl(Q, _curl_amt(t), reg_fate)       # full fetal C through the handoff (spine outside)
            anchor = (1 - s) * Q.mean(0) + s * _chest(Q, reg_fate)   # ramp centroid(cloud)->chest(mesh)
            Q = Q - anchor
            out_frames.append(_emit(Q, fate, vm, "handoff",
                                   "handoff · cloud → human-embryo surface (coarse-graining)", t,
                                   panel_mesh(0.0, morphing=1.0)))

        # ---- Phase C: allometric maturation of the MakeHuman mesh, embryo -> adult ----
        print(f"[C] maturation: embryo -> fetus -> infant -> adult ({N_MESH} frames) ...")
        adult_len = 3.2
        labels = [(0.10, "early fetus"), (0.28, "fetus"), (0.46, "newborn"), (0.64, "infant"),
                  (0.80, "child"), (0.93, "adolescent"), (1.01, "adult")]
        for i in range(N_MESH):
            f = i / (N_MESH - 1)
            Vf = mature_body(Vsamp, f)
            Q = to_laid(Vf)
            tgt = 0.9 + (adult_len - 0.9) * f ** 1.2        # visible growth 0.9 -> 3.2 (convex)
            Q *= tgt / _long_axis_len(Q)
            t = 0.62 + 0.38 * f
            Q = tuck_limbs(Q, reg_fate, _curl_amt(t))       # flex limbs into the fetal tuck early
            Q = fetal_curl(Q, _curl_amt(t), reg_fate)       # fetal C (spine outside), unfurling toward the adult
            Q = Q - _chest(Q, reg_fate)                     # anchor the chest so it grows in place
            lab = next(l for thr, l in labels if f < thr)
            out_frames.append(_emit(Q, reg_fate, np.full(N_R, NEUT), "mesh",
                                   f"mesh · {lab} (allometric maturation)", t,
                                   panel_mesh(f, morphing=(1 - f))))
        Q_adult = Q                                         # the finished adult skin (laid, centred)

    else:
        # ================= MODEL-NATIVE path: mature the cloud's OWN cells (no MakeHuman) ==========
        # The final tetrapod cloud IS the embryo. Instead of resampling it onto a MakeHuman mesh, we
        # grow the model's own cells up by heads-tall allometry (mature_cloud) + limb outgrowth
        # (grow_limbs, now extending PAST the bud) + the fetal curl unfurling -- so infant and adult
        # carry the model's OWN fates and topology. There is no handoff: the same cells run all the
        # way through, coloured by their real cloud fate the whole time.
        reg_fate = S0col_fate                               # the cloud's real fates ARE the identity
        print(f"[C] model maturation: cloud embryo -> fetus -> infant -> adult ({N_MESH} frames) ...")
        from medic.skin_shell_head import mesh as _skin_mesh   # the epidermal-boundary skin SURFACE
        from medic.flesh_surface_head import flesh_skin as _flesh_skin   # skin draped over MUSCLE+FAT, not bone
        from medic.fine_relief_head import body_relief as _body_relief    # the surface-muscle silhouette + six-pack
        skin_faces = None
        adult_len = 3.2
        labels = [(0.10, "early fetus"), (0.28, "fetus"), (0.46, "newborn"), (0.64, "infant"),
                  (0.80, "child"), (0.93, "adolescent"), (1.01, "adult")]
        limbmask = (S0col_fate == LIMB)
        for i in range(N_MESH):
            f = i / (N_MESH - 1)
            Q = mature_cloud(S0, S0col_fate, f, MATURE_SEARCHED)   # searched allometry -> relaxes into the atlas
            t = 0.55 + 0.45 * f
            Q = grow_limbs(Q, limbmask, _limb_grow_model(t, MATURE_SEARCHED["limb_ext"]),
                           _limb_grow_model(t, MATURE_SEARCHED.get("leg_ext", MATURE_SEARCHED["limb_ext"])),
                           pose=f)   # arms/legs swing DOWN by maturation stage (decoupled from extension)
            tgt = 0.9 + (adult_len - 0.9) * f ** 1.2        # visible growth 0.9 -> 3.2 (convex)
            Q *= tgt / _long_axis_len(Q)
            Q = tuck_limbs(Q, S0col_fate, _curl_amt(t))     # fetal tuck early, releasing as it unfurls
            Q = fetal_curl(Q, _curl_amt(t), S0col_fate)     # fetal C (spine to the OUTSIDE), unfurling to adult
            Q = Q - _chest(Q, S0col_fate)                   # anchor the chest so it grows in place
            lab = next(l for thr, l in labels if f < thr)
            sv, sf = _flesh_skin(Q, S0col_fate)              # skin draped over the MUSCLE+FAT (not the bone), so
            #                                                  the muscle silhouette reads through the epidermal shell
            sv, _ = _body_relief(sv, Q, S0col_fate, amp=f)   # the surface-muscle RELIEF (deltoid/pec/lat/glute/
            #                                                  six-pack), blended by f -> smooth infant, muscled adult
            if skin_faces is None:
                skin_faces = sf                              # fixed topology -> store faces ONCE
            skin_op = 0.30 + 0.35 * f                        # skin firms up as the body matures
            out_frames.append(_emit(Q, S0col_fate, np.full(N_R, NEUT), "mesh",
                                   f"model · {lab} (allometric maturation of the cell cloud)", t,
                                   panel_mesh(f, morphing=(1 - f)), skin=sv, skin_op=skin_op))
        Q_adult = Q                                         # the finished adult cloud (laid, centred)
        skin_adult_v = _body_relief(_flesh_skin(Q_adult, S0col_fate)[0], Q_adult, S0col_fate, amp=1.0)[0]  # adult FLESHED + muscled skin (dissolves in the reveal)

    # ---- Phase D: reveal the MODEL'S OWN integrated anatomy (its genome-derived systems, not the atlas) --
    print(f"[D] anatomy: adult skin -> the model's OWN integrated anatomy ({N_REVEAL} reveal + {N_HOLD} hold) ...")
    anat_all, hi_all, atlas_layers2 = ANAT_LIST + ATLAS_COLS, HI_IDX + ATLAS_HI, ATLAS_LAYERS
    organ_surf = []
    try:
        from medic.integrated_body import anatomy_points, SYS as ANAT_SYS
        apos, asid, scols, snames, acounts, organ_meshes = anatomy_points(rng, N_R)
        # ONE consistent affine (centre apos, scale to the adult length, translate to the adult centroid) applied
        # to BOTH the anatomy points and the organ surfaces, so the solid organs sit exactly where the points reveal.
        _am0 = apos.mean(0); _asc = _long_axis_len(Q_adult) / (_long_axis_len(apos) + 1e-9)
        _align = lambda P: (np.asarray(P, float) - _am0) * _asc + Q_adult.mean(0)
        apos = _align(apos)
        # per-organ SURFACE solids (alpha-shape meshes on the matured condensed cloud), aligned the same way.
        def _hex(h):
            h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        organ_surf = [dict(name=nm, color=_hex(m["color"]),
                           V=_align(m["V"]).astype(np.float32).ravel().tolist(),
                           F=np.asarray(m["faces"], int).ravel().tolist())
                      for nm, m in organ_meshes.items()]
        # SPLAY FIX: bound the revealed anatomy inside Q_adult (the arms-down skin it dissolves from). The old
        # per-region _fit_atlas_to_model splayed the limbs (region mis-detection); clipping to the correct
        # arms-down silhouette keeps every part inside the body instead of radiating into a star.
        from medic.skin_shell_head import envelope_clip
        apos = envelope_clip(apos, Q_adult)
        base2 = len(ANAT_LIST) + len(ATLAS_COLS)
        T, Tf = apos, base2 + asid
        anat_all = ANAT_LIST + ATLAS_COLS + [list(c) for c in scols]
        hi_all = HI_IDX + ATLAS_HI + [base2 + ANAT_SYS.index("digit"), base2 + ANAT_SYS.index("bone")]
        ap_panel = [dict(name=k, layer=k.capitalize(), role="genome-derived — the model's own cells",
                         mag=0.7, added=False, firing="atlas") for k in ANAT_SYS]
        atlas_layers2 = [k.capitalize() for k in ANAT_SYS]
        stageD = ("anatomy · the model's OWN body — bone · muscle · digit · tendon · vessel · nerve "
                  f"({acounts['vertebrae']}, {acounts['digits']} digits, {acounts['tendons']} tendons)")
        print(f"    model anatomy: {acounts['vertebrae']} vertebrae · {acounts['digits']} digits · "
              f"{acounts['tendons']} tendons · {acounts['vessel_branch_points']} vessel branches")
    except Exception as _e:                                              # fall back to the Hill-organ atlas
        print(f"    [model-anatomy fold-in failed: {_e}; falling back to the atlas]")
        A_pos, A_fate, counts = build_human_atlas(rng)
        A_pos = _fit_atlas_to_model(A_pos - A_pos.mean(0) + Q_adult.mean(0), Q_adult, reg_fate)
        T, Tf = A_pos, A_fate
        ap_panel = panel_anatomy(counts)
        stageD = f"anatomy · Hill-organ atlas · {counts['bones']} bones · {counts['muscles']} muscles"
    # DISSOLVE the skin in place to reveal the internal parts (each point flips at its own random threshold).
    S, Sf = Q_adult, reg_fate
    thr = rng.random(N_R)
    for i in range(1, N_REVEAL + 1):
        s = i / N_REVEAL
        flip = thr < s
        Q = np.where(flip[:, None], T, S)
        fate = np.where(flip, Tf, Sf).astype(int)
        sk = dict(skin=skin_adult_v, skin_op=0.55 * (1 - s)) if skin_adult_v is not None else {}
        out_frames.append(_emit(Q, fate, np.full(N_R, NEUT), "anatomy",
                               "anatomy · skin dissolving to reveal the model's own parts", 1.0, ap_panel, "atlas", **sk))
    for _ in range(N_HOLD):
        out_frames.append(_emit(T, Tf, np.full(N_R, NEUT), "anatomy", stageD, 1.0, ap_panel, "atlas"))

    R = max(max(abs(float(v)) for v in fr["xyz"]) for fr in out_frames)
    print(f"    mesh magnitudes grounded: species={ {k: round(v,2) for k,v in SPECIES_MAG.items()} }")
    print(f"                              mature ={ {k: round(v,2) for k,v in MATURE_MAG.items()} }")
    disp = ("Genome → tetrapod → human · one clock — NCA+LGM cloud grown up (model-native), no MakeHuman"
            if source == "model" else
            "Genome → tetrapod → human · one clock — cloud generator + MakeHuman mesh readout")
    doc = dict(display=disp, source=source,
               vmin=VMIN, vmax=VMAX, R=round(R, 3), anat=anat_all,
               hi=hi_all, skin=FIDX["Skin"], spanning=SPANNING, layers=LAYER_ORDER,
               skin_faces=(skin_faces.tolist() if skin_faces is not None else None),
               organ_surfaces=organ_surf,
               atlas_layers=atlas_layers2, n_frames=len(out_frames), frames=out_frames)
    jp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(jp, "w"))
    print(f"saved {jp}  ({len(out_frames)} frames, {jp.stat().st_size/1e6:.1f} MB)")
    HTML.write_text(VIEWER, encoding="utf-8")
    print(f"saved {HTML}")


# ---------------------------------------------------------------- viewer
VIEWER = r"""<!doctype html><html><head><meta charset="utf-8"><title>Genome → tetrapod → human</title>
<style>
  html,body{margin:0;height:100%;background:#0d1017;color:#cbd5e1;font:13px system-ui;overflow:hidden}
  #ui{position:fixed;top:10px;left:12px;z-index:3;background:#0d1017cc;padding:9px 12px;border-radius:9px;max-width:48%}
  #ui b{color:#e8eef4}#stage{color:#7dd3fc;font-size:14px;margin-top:3px}#legend{color:#8091a8;margin-top:4px}
  #phase{margin-top:6px}#phase span{padding:2px 9px;border-radius:6px;margin-right:6px;background:#1b2130;color:#66748c}
  #phase span.on{background:#2b6cb0;color:#fff}
  #panel{position:fixed;top:10px;right:12px;bottom:12px;width:310px;z-index:3;background:#0d1017e8;
         padding:10px 12px;border-radius:9px;overflow-y:auto}
  #panel h3{margin:0 0 6px;font-size:13px;color:#e8eef4}
  #panel .hint{color:#6b7688;font-size:10px;margin:0 0 6px}
  #clk{margin:4px 0 8px}#clk .lab{font-size:11px;color:#8091a8;display:flex;justify-content:space-between}
  #clk .track{height:5px;background:#1b2130;border-radius:4px;margin-top:3px;overflow:hidden}
  #clk .fill{height:100%;background:#7dd3fc}
  #panel .sub{color:#cbd5e1;font-size:11px;font-weight:600;letter-spacing:.03em;margin:9px 0 3px;
              border-top:1px solid #263043;padding-top:6px}
  #panel .sub .q{color:#6b7688;font-weight:400}
  .row{margin:5px 0}
  .row .nm{font-size:12px;color:#dbe4ef}
  .row .rl{font-size:10px;color:#7f8ca3;line-height:1.25}
  .row .track{height:6px;background:#1b2130;border-radius:4px;margin-top:2px;overflow:hidden}
  .row .fill{height:100%;background:#4b9cff}
  .row.span .nm{color:#ffd23a}
  .row.span .fill{background:#ffd23a}
  .tag{font-size:9px;padding:1px 5px;border-radius:4px;margin-left:5px;vertical-align:1px}
  .tag.cloud{background:#14324a;color:#7dd3fc}.tag.mesh{background:#3a2a12;color:#ffcf7a}
  .tag.both{background:#3a3212;color:#ffd23a}
  #bar{position:fixed;bottom:12px;left:12px;right:334px;z-index:3;display:flex;align-items:center;gap:10px;
       background:#0d1017cc;padding:8px 12px;border-radius:9px}
  button{font:13px system-ui;background:#1b2130;color:#e2e8f0;border:1px solid #33405a;border-radius:6px;padding:4px 12px;cursor:pointer}
  button.on{background:#2b6cb0;border-color:#2b6cb0}input[type=range]{flex:1}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
  #key{margin-top:5px}#key span{margin-right:10px;white-space:nowrap;font-size:11px}
</style></head>
<body>
<div id="loadwrap" style="position:fixed;inset:0;z-index:20;background:#0d1017;display:flex;flex-direction:column;
     align-items:center;justify-content:center;color:#cbd5e1;font:14px system-ui">
  <div style="margin-bottom:14px">Loading the movie — genome → human…</div>
  <div style="width:340px;height:9px;background:#1b2130;border-radius:5px;overflow:hidden">
    <div id="loadbar" style="width:0;height:100%;background:linear-gradient(90deg,#2b6cb0,#7dd3fc);transition:width .12s"></div></div>
  <div id="loadpct" style="margin-top:9px;color:#7dd3fc;font-variant-numeric:tabular-nums">0%</div>
</div>
<div id="ui"><b>Genome → tetrapod → human — one clock, one clump of cells</b>
<div id="legend">CLOUD generates (morphogenesis) · handoff · MESH matures (allometry). Drag to rotate, scrub the timeline.</div>
<div id="phase"><span id="pA">A cloud</span><span id="pB">B handoff</span><span id="pC">C mesh</span><span id="pD">D anatomy</span></div>
<div id="key"></div>
<div id="stage">loading…</div></div>
<div id="panel"><h3>What is dialing — by layer</h3>
<div class="hint">gold = acts in BOTH phases (a kernel gene redeployed). genes are the knobs threaded through every layer. mesh-phase bar heights = real fitted knob moves (mouse→human builder: species adapter + maturation) and GWAS |β| (face/heart).</div>
<div id="clk"><div class="lab"><span>◷ clock τ</span><span id="clkv">0.00</span></div><div class="track"><div class="fill" id="clkf" style="width:0%"></div></div></div>
<div id="rows"></div></div>
<div id="bar">
  <button id="play">⏸ pause</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1">
  <button id="mode" class="on">anatomy</button>
  <button id="rot">↻ auto-rotate</button>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const params=new URLSearchParams(location.search);
let cur=0, playing=params.get('pause')?false:true, last=0, FRAME_MS=110, mode='anat';
const sc=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 500);
const rn=new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
rn.setSize(innerWidth,innerHeight); rn.setPixelRatio(devicePixelRatio); document.body.appendChild(rn.domElement);
sc.add(new THREE.AmbientLight(0xffffff,0.72));
const _dl=new THREE.DirectionalLight(0xffffff,0.85); _dl.position.set(0.6,1,0.8); sc.add(_dl);
const ctrl=new OrbitControls(cam, rn.domElement); ctrl.enableDamping=true; ctrl.autoRotateSpeed=1.1;
window.cam=cam; window.ctrl=ctrl;   // exposed for scripted camera stills (paper figures)
let DATA=null, pts=[], vmin=0, vmax=1, nf=0, skinMesh=null, skinIdx=null, showOrgans=true;
window.addEventListener('keydown',e=>{ if(e.key==='o'||e.key==='O'){ showOrgans=!showOrgans; if(DATA) build(cur); } });
const UNC=[0.5,0.55,0.6];
// the skin SURFACE: a translucent epidermal-boundary mesh (fixed topology; verts stream per frame). depthWrite
// off so the internal cell cloud shows THROUGH the skin -- a body with a silhouette, not a scatter.
function mkSkin(verts, op){
  const n=verts.length/3, P=new Float32Array(n*3);
  for(let i=0;i<n;i++){ P[3*i]=verts[3*i+2]; P[3*i+1]=verts[3*i]; P[3*i+2]=verts[3*i+1]; }
  if(!skinIdx){ skinIdx=[]; for(const f of DATA.skin_faces){ skinIdx.push(f[0],f[1],f[2]); } }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(P,3));
  g.setIndex(skinIdx); g.computeVertexNormals();
  const m=new THREE.MeshStandardMaterial({color:0xd8b49a, transparent:true, opacity:op,
    side:THREE.DoubleSide, roughness:0.9, metalness:0.0, depthWrite:false});
  return new THREE.Mesh(g,m);
}
// per-organ SURFACE solids: alpha-shape meshes, one per organ, coloured, shown during the anatomy reveal so an
// organ reads as a Gray's-crisp SOLID instead of a point scatter. Same AP->Y / ML->X / DV->Z remap as the skin.
let organMeshes=[];
function mkOrgan(V, F, color, op){
  const n=V.length/3, P=new Float32Array(n*3);
  for(let i=0;i<n;i++){ P[3*i]=V[3*i+2]; P[3*i+1]=V[3*i]; P[3*i+2]=V[3*i+1]; }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(P,3));
  g.setIndex(F); g.computeVertexNormals();
  const m=new THREE.MeshStandardMaterial({color:new THREE.Color(color[0],color[1],color[2]),
    transparent:true, opacity:op, side:THREE.DoubleSide, roughness:0.75, metalness:0.0, depthWrite:true});
  return new THREE.Mesh(g,m);
}
function clearOrgans(){ for(const o of organMeshes){ sc.remove(o); o.geometry.dispose(); o.material.dispose(); } organMeshes=[]; }
function vcol(vm){ let t=(vm-vmin)/(vmax-vmin+1e-9); t=Math.max(0,Math.min(1,t));
  let a=[0.23,0.32,0.78],b=[0.93,0.93,0.93],c=[0.82,0.14,0.16];
  if(t<0.5){let u=t*2;return[a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u,a[2]+(b[2]-a[2])*u];}
  let u=(t-0.5)*2;return[b[0]+(c[0]-b[0])*u,b[1]+(c[1]-b[1])*u,b[2]+(c[2]-b[2])*u]; }
function acol(fate){ return fate<0?UNC:DATA.anat[fate]; }
// UPRIGHT + FRONT remap (AP->Y up, ML->X, DV->Z, belly toward camera) baked into a cheap GPU POINT CLOUD
// (THREE.Points). Points render ~100k+ cells easily where instanced sphere-balls crash the iGPU at ~7k --
// so this is what lets the viewer actually show the high cell count, finely.
function mk(pos,col,size){
  const n=pos.length/3, P=new Float32Array(n*3), C=new Float32Array(n*3);
  for(let i=0;i<n;i++){ P[3*i]=pos[3*i+2]; P[3*i+1]=pos[3*i]; P[3*i+2]=pos[3*i+1];
    C[3*i]=col[3*i]; C[3*i+1]=col[3*i+1]; C[3*i+2]=col[3*i+2]; }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(P,3));
  g.setAttribute('color',new THREE.BufferAttribute(C,3));
  return new THREE.Points(g,new THREE.PointsMaterial({size:size,vertexColors:true,sizeAttenuation:true}));
}
function build(i){
  const fr=DATA.frames[i];
  for(const p of pts){sc.remove(p);p.geometry.dispose();p.material.dispose();}
  const N=fr.vm.length, hi=new Set(DATA.hi||[]);
  const bp=[],bc=[],hp=[],hc=[];
  for(let k=0;k<N;k++){
    const f=fr.fate[k], c=(mode==='anat')?acol(f):vcol(fr.vm[k]);
    if(mode==='anat' && hi.has(f)){ hp.push(fr.xyz[3*k],fr.xyz[3*k+1],fr.xyz[3*k+2]); hc.push(c[0],c[1],c[2]); }
    else { bp.push(fr.xyz[3*k],fr.xyz[3*k+1],fr.xyz[3*k+2]); bc.push(c[0],c[1],c[2]); }
  }
  pts=[mk(bp,bc,0.013)]; if(hp.length) pts.push(mk(hp,hc,0.022));   // point-cloud cell sizes
  for(const p of pts) sc.add(p);
  if(skinMesh){ sc.remove(skinMesh); skinMesh.geometry.dispose(); skinMesh.material.dispose(); skinMesh=null; }
  if(fr.skin && DATA.skin_faces && (fr.skin_op||0)>0.01){ skinMesh=mkSkin(fr.skin, fr.skin_op); sc.add(skinMesh); }
  clearOrgans();   // solid organ surfaces during the anatomy reveal (toggle with the 'o' key)
  if(showOrgans && fr.phase==='anatomy' && DATA.organ_surfaces){
    for(const os of DATA.organ_surfaces){ const o=mkOrgan(os.V, os.F, os.color, 0.85); organMeshes.push(o); sc.add(o); }
  }
  document.getElementById('stage').textContent=fr.stage;
  document.getElementById('slider').value=i;
  pA.className=fr.phase==='cloud'?'on':''; pB.className=fr.phase==='handoff'?'on':''; pC.className=fr.phase==='mesh'?'on':''; pD.className=fr.phase==='anatomy'?'on':'';
  document.getElementById('clkv').textContent=fr.t.toFixed(2)+'  ·  '+fr.phase;
  document.getElementById('clkf').style.width=Math.round(fr.t*100)+'%';
  renderPanel(fr.panel, fr.ptype);
}
const LAYER_Q={Clock:'when',Axes:'where',Heads:'what a cell becomes',Proportions:'the shape',
  Skeleton:'bones',Muscle:'Hill muscle groups',Viscera:'homologous organs',Teeth:'added parts'};
function renderPanel(rows, ptype){
  if(ptype==='atlas'){ renderAtlas(rows); return; }
  const line=r=>{
    const tag=r.span?`<span class="tag both">both · ${r.firing}</span>`
                     :`<span class="tag ${r.firing}">${r.firing}</span>`;
    return `<div class="row ${r.span?'span':''}"><div class="nm">${r.name}${tag}</div>`+
           `<div class="rl">${r.role}</div><div class="track"><div class="fill" style="width:${Math.round(r.mag*100)}%"></div></div></div>`;
  };
  const order=(DATA&&DATA.layers)||['Clock','Axes','Heads','Proportions'];
  const by={}; for(const r of (rows||[])){ (by[r.layer]=by[r.layer]||[]).push(r); }
  let html='';
  for(const L of order){
    html+=`<div class="sub">${L} <span class="q">— ${LAYER_Q[L]||''}</span></div>`;
    const rs=(by[L]||[]).sort((a,b)=>(b.span-a.span)||(b.mag-a.mag));
    html+= rs.length?rs.map(line).join(''):'<div class="rl" style="color:#5b6678">— quiet this frame —</div>';
  }
  document.getElementById('rows').innerHTML=html;
}
function renderAtlas(rows){
  const line=r=>{
    const tag=r.added?`<span class="tag both">added</span>`:`<span class="tag cloud">in situ</span>`;
    return `<div class="row ${r.added?'span':''}"><div class="nm">${r.name}${tag}</div>`+
           `<div class="rl">${r.role}</div><div class="track"><div class="fill" style="width:${Math.round(r.mag*100)}%"></div></div></div>`;
  };
  const order=(DATA&&DATA.atlas_layers)||['Skeleton','Muscle','Viscera','Teeth'];
  const by={}; for(const r of (rows||[])){ (by[r.layer]=by[r.layer]||[]).push(r); }
  let html='<div class="rl" style="color:#7dd3fc;margin-bottom:4px">Hill-organ atlas — one homologous roster, placed in situ (gold = ADDED, not in the roster)</div>';
  for(const L of order){
    if(!(by[L]&&by[L].length)) continue;
    html+=`<div class="sub">${L} <span class="q">— ${LAYER_Q[L]||''}</span></div>`;
    html+= by[L].map(line).join('');
  }
  document.getElementById('rows').innerHTML=html;
}
const pA=document.getElementById('pA'),pB=document.getElementById('pB'),pC=document.getElementById('pC'),pD=document.getElementById('pD');
const KEY=[['Head','#4d75f2'],['Eye','#33d8ff'],['Heart','#ee2938'],['Limb','#47db76'],
           ['Muscle','#db6b6b'],['Rib','#f0f0db'],['Spinal','#758de8'],['Jaw','#db9eb2'],
           ['Gut','#d19a66'],['Skin','#f5d1bd']];
const SRC=((params.get('src')==='mh')?'movie/human_movie_frames_mh.json':'movie/human_movie_frames.json')+'?v='+Date.now();
const _bar=document.getElementById('loadbar'),_pct=document.getElementById('loadpct'),_wrap=document.getElementById('loadwrap');
// streamed fetch so the loading bar tracks the download in real time (the JSON is several MB)
fetch(SRC).then(resp=>{
  const total=+resp.headers.get('Content-Length')||0; const reader=resp.body.getReader();
  let got=0; const chunks=[];
  const pump=()=>reader.read().then(({done,value})=>{
    if(done) return;
    chunks.push(value); got+=value.length;
    if(total){const p=got/total; _bar.style.width=(p*90)+'%'; _pct.textContent=Math.round(p*100)+'%';}
    else _pct.textContent=(got/1e6).toFixed(1)+' MB';
    return pump();
  });
  return pump().then(()=>{
    _pct.textContent='building…'; _bar.style.width='96%';
    const buf=new Uint8Array(got); let o=0; for(const c of chunks){buf.set(c,o); o+=c.length;}
    return JSON.parse(new TextDecoder().decode(buf));
  });
}).then(d=>{
  _bar.style.width='100%'; if(_wrap) _wrap.style.display='none';
  DATA=d; vmin=d.vmin; vmax=d.vmax; nf=d.frames.length;
  if(d.display) document.querySelector('#ui b').textContent=d.display;
  document.getElementById('slider').max=nf-1;
  document.getElementById('key').innerHTML=KEY.map(k=>`<span><i class="dot" style="background:${k[1]}"></i>${k[0]}</span>`).join('');
  const R=d.R||2.0; cam.position.set(R*0.42,R*0.10,R*3.4); ctrl.target.set(0,0,0);  // front-on; pulled back so a tall figure's head+feet are not cropped
  cur=params.get('f')?Math.min(nf-1,Math.max(0,parseInt(params.get('f')))):0;
  build(cur); syncPlay();
});
const playBtn=document.getElementById('play'),rotBtn=document.getElementById('rot'),
      modeBtn=document.getElementById('mode'),slider=document.getElementById('slider');
function syncPlay(){ playBtn.textContent=playing?'⏸ pause':'▶ play'; }
playBtn.onclick=()=>{playing=!playing;syncPlay();};
rotBtn.onclick=()=>{ctrl.autoRotate=!ctrl.autoRotate;rotBtn.classList.toggle('on',ctrl.autoRotate);};
modeBtn.onclick=()=>{ mode=(mode==='anat')?'volt':'anat'; modeBtn.textContent=(mode==='anat')?'anatomy':'voltage';
  modeBtn.classList.toggle('on',mode==='anat'); build(cur); };
slider.oninput=()=>{playing=false;syncPlay();cur=parseInt(slider.value);build(cur);};
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
function loop(t){ requestAnimationFrame(loop);
  if(DATA&&playing&&t-last>FRAME_MS){last=t;cur=(cur+1)%nf;build(cur);}
  ctrl.update(); rn.render(sc,cam); }
requestAnimationFrame(loop);
</script></body></html>"""


JSON_MH = Path("data/movie/human_movie_frames_mh.json")
COMPARE_PNG = Path("data/movie/_compare_model_vs_makehuman.png")


def compare_montage():
    """Side-by-side keyframe montage: the model-native maturation (top row) vs the MakeHuman
    readout (bottom row). Front view = ML (z) horizontal x AP (x) vertical, coloured by fate. Lets
    us judge whether growing the model's OWN cloud up produces a recognizable human without the
    mannequin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def load(p):
        return json.load(open(p))

    def pick(doc, targets):
        """Nearest morphogenesis/maturation frame to each target clock t (skip the atlas reveal)."""
        fr = [f for f in doc["frames"] if f["phase"] in ("cloud", "mesh", "handoff")]
        out = []
        for tt in targets:
            out.append(min(fr, key=lambda f: abs(f["t"] - tt)))
        return out

    targets = [0.53, 0.66, 0.78, 0.90, 1.00]
    labels = ["embryo", "fetus", "infant", "child", "adult"]
    rows = [("model-native (NCA+LGM cloud grown up)", load(JSON), "model"),
            ("MakeHuman readout (mannequin)", load(JSON_MH), "mh")]
    fig, axes = plt.subplots(2, len(targets), figsize=(3.0 * len(targets), 7.4), facecolor="#0d1017")
    for ri, (title, doc, _tag) in enumerate(rows):
        anat = doc["anat"]
        frames = pick(doc, targets)
        for ci, (fr, lab) in enumerate(zip(frames, labels)):
            ax = axes[ri, ci]
            ax.set_facecolor("#0d1017")
            xyz = np.array(fr["xyz"], float).reshape(-1, 3)
            fate = np.array(fr["fate"], int)
            col = np.array([anat[f] if 0 <= f < len(anat) else [0.5, 0.5, 0.5] for f in fate])
            # front view: ML (z) horizontal, AP (x=head+) vertical
            ax.scatter(xyz[:, 2], xyz[:, 0], c=np.clip(col, 0, 1), s=2.2, linewidths=0)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.9, 1.9)
            for sp in ax.spines.values():
                sp.set_color("#33405a")
            if ri == 0:
                ax.set_title(f"{lab}\nt={fr['t']:.2f}", color="#7dd3fc", fontsize=10)
        axes[ri, 0].set_ylabel(title, color="#e8eef4", fontsize=11)
    fig.suptitle("Maturing the model's own cells vs the MakeHuman mannequin  ·  front view, coloured by fate",
                 color="#e8eef4", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    COMPARE_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMPARE_PNG, dpi=120, facecolor="#0d1017")
    print(f"saved {COMPARE_PNG}")


if __name__ == "__main__":
    build(source="model", json_path=JSON)            # NCA+LGM native (the default the viewer loads)
    build(source="makehuman", json_path=JSON_MH)     # the mannequin, kept for comparison (?src=mh)
    compare_montage()
