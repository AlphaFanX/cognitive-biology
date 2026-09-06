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
from medic.dv_spread_head import spread as _dv_measured_spread
from medic.subhead_program import expand_names as _exn

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
# 2026-08-30: + Telencephalon (the bulk of the forebrain) -- the sub-head split post-dated this tuple,
# so the telencephalon ESCAPED the head-scale/rounding = the flat-wide "pancake brain" + the green
# crown wings at the reveal. Use sites expand sub-head children so the mask survives future splits.
_HEAD_FATES = ("Forebrain", "Telencephalon", "Eye", "Midbrain", "Hindbrain",
               "Retina", "OlfactoryBulb", "Cerebellum")


def _head_ids():
    from medic.subhead_program import expand_names
    return [FIDX[n] for n in expand_names(_HEAD_FATES) if n in FIDX]


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
    headf = _head_ids()
    hm = np.isin(fate, headf) if fate is not None else np.zeros(len(Q), bool)
    # the BRAIN mask (cephalic vesicles, no eye): the ML envelope may COMPRESS the brain but never
    # INFLATE it -- the two-sided head-slice conform was inflating the brain to fill the whole head
    # silhouette (ML x2 = the "pancake brain" / green crown cap), overwriting the braincase rounding.
    from medic.subhead_program import expand_names as _exn
    _bf = [FIDX[n] for n in _exn(("Forebrain", "Telencephalon", "Midbrain", "Hindbrain",
                                  "Cerebellum", "OlfactoryBulb")) if n in FIDX]
    brainm = np.isin(fate, _bf) if fate is not None else np.zeros(len(Q), bool)
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
    # HEAD ROUNDING (kill the "xenomorph"): the cranium must be a rounded ovoid, not an elongated ridge. Conform the
    # head cloud's three axis extents toward the human head aspect AP:DV:ML ~ 1.42:1.32:1.0 (height ~0.135 H, depth
    # ~0.125, breadth ~0.095) about its centroid -- shrinking whichever axis is over-long -> a braincase, not a snout.
    if hm.sum() >= 30 and f > 0:
        hc = Q[hm].mean(0)
        ext = np.ptp(Q[hm], axis=0) + 1e-9                       # AP, DV, ML extents
        aspect = np.array([1.42, 1.32, 1.0]); aspect = aspect / aspect.mean()
        tgt_ext = aspect * float(ext.mean())
        sc = np.clip(tgt_ext / ext, 0.6, 1.7)
        Q[hm] = hc + (Q[hm] - hc) * (1.0 + f * (sc - 1.0))
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
               + 0.040 * np.exp(-((u - 0.95) / 0.05) ** 2)                    # cranium ~0.135 (let the head fill FB)
               - 0.030 * np.exp(-((u - 0.855) / 0.022) ** 2))                 # NECK pinch (DV twin of the ML pinch:
        #                                       the skin field had no cervical waist -- head merged into shoulders)
        if depth > cap:
            s = 1.0 - f * (1.0 - cap / depth)
            idx = np.where(m)[0]
            Q[idx, 1] = axis_dv + (Q[idx, 1] - axis_dv) * s
    # ANTHROPOMETRIC ML ENVELOPE (regional): the mediolateral BREADTH also varies -- broad shoulders (~0.245 H)
    # and hips (~0.19), a pinched waist, but thin neck / head / limbs. ml_girth + the regional taper leave the
    # legs / hips / head too broad (the stout, "not-quite-standing" look), so clamp each AP slice's ML breadth to
    # the anthropometric envelope, compressing about the ML midline. Blended by f (f=0 identity). The ML twin of
    # the DV envelope above -- together they drive the front silhouette to real breadth (posture_silhouette scorer).
    mid_ml = float(np.median(Q[:, 2]))
    _lb = FIDX.get("Limb Bud")
    notlimb = (fate != _lb) if (fate is not None and _lb is not None) else np.ones(len(Q), bool)   # don't clamp the arms
    for i in range(nb):
        lo = xx.min() + i / nb * stature
        m = (xx >= lo) & (xx < lo + stature / nb) & notlimb       # the arm (Limb Bud) extends past the trunk -- exempt it
        if m.sum() < 8:
            continue
        breadth = (np.percentile(Q[m, 2], 95) - np.percentile(Q[m, 2], 5)) / stature
        u = (i + 0.5) / nb                                                    # AP fraction (0 = feet, 1 = crown)
        # anthropometric FULL ML breadth target (fraction of H): broad shoulders + chest + hips, pinched waist,
        # thin neck/head/limbs. TWO-SIDED conform (build the narrow ribbon-trunk OUT to human breadth AND slim the
        # wide hips/head) -- the ML twin of dv_girth building out the flat DV ribbon; anthropometric, not a fudge.
        tgt = (0.070                                                          # thin baseline (limbs / ankle)
               + 0.175 * np.exp(-((u - 0.80) / 0.040) ** 2)                   # shoulders ~0.245
               + 0.120 * np.exp(-((u - 0.70) / 0.115) ** 2)                   # chest ~0.19 (wider tail fills the waist)
               + 0.100 * np.exp(-((u - 0.47) / 0.140) ** 2)                   # hips ~0.19 (broader, less peaky)
               + 0.045 * np.exp(-((u - 0.30) / 0.110) ** 2)                   # thighs ~0.13
               + 0.028 * np.exp(-((u - 0.905) / 0.042) ** 2)                  # head ~0.10 (tighter -> crown tapers)
               - 0.030 * np.exp(-((u - 0.855) / 0.022) ** 2))                 # NECK pinch (thin neck below the head)
        if breadth > 1e-3:
            s = float(np.clip(1.0 + f * (tgt / breadth - 1.0), 0.4, 2.3))
            idx = np.where(m)[0]
            if s > 1.0:
                idx = idx[~brainm[idx]]                   # inflate the head SILHOUETTE, never the brain
            Q[idx, 2] = mid_ml + (Q[idx, 2] - mid_ml) * s
    # ORGAN ISOTROPY under trunk elongation (2026-08-30): trunk_e stretches SPACE; a discrete organ is a
    # cohesive body (the connexin capsule) whose POSITION rides the stretch while its own proportions hold.
    # Counter-scale each discrete organ family's internal AP by the trunk factor about its centroid -- the
    # general form of the heart's 4:1 residual and the "elongated viscera stack" in the canon comparison.
    # Per-family isotropy KNOBS (see _ORGAN_FAMILIES / ISO_LAMBDA -- searched by medic.curve_train):
    # lambda=1 fully counter-scales the trunk stretch inside the organ, lambda=0 rides it.
    te = 1.0 + p["trunk_e"] * f
    if fate is not None and te > 1.0:
        for _fname, _fam in _ORGAN_FAMILIES.items():
            lam = float(ISO_LAMBDA.get(_fname, 0.0))
            if lam <= 0:
                continue
            d = 1.0 + (te - 1.0) * lam
            fids = [FIDX[n] for n in _exn(_fam) if n in FIDX]
            fm2 = np.isin(fate, fids)
            if fm2.sum() >= 8:
                c0 = float(Q[fm2, 0].mean())
                Q[fm2, 0] = c0 + (Q[fm2, 0] - c0) / d
    # Per-family ADULT ASPECT knobs (knob set 2, searched by medic.curve_train): the isotropy knob above
    # counter-scales only the trunk's AP stretch; this is its general form -- each organ family (one connexin
    # capsule, moved/scaled as ONE body) relaxes toward its own adult proportions (a1, a2 = 2nd/3rd principal
    # axes vs the 1st) in its OWN principal frame, volume-preserving, blended by f. A fitted maturation anchor
    # in the g_K pattern: the per-organ allometric growth program, genome-derivable later.
    if fate is not None and f > 0 and ADULT_ASPECT:
        for _fname, (_a1, _a2) in ADULT_ASPECT.items():
            fam = _ORGAN_FAMILIES.get(_fname)
            if fam is None or (abs(_a1 - 1.0) < 1e-9 and abs(_a2 - 1.0) < 1e-9):
                continue
            fids = [FIDX[n] for n in _exn(fam) if n in FIDX]
            fm2 = np.isin(fate, fids)
            if fm2.sum() < 8:
                continue
            P2 = Q[fm2]
            c2 = P2.mean(0)
            C2 = P2 - c2
            Vt2 = np.linalg.svd(C2, full_matrices=False)[2]
            co = C2 @ Vt2.T
            s1 = 1.0 + f * (_a1 - 1.0)
            s2 = 1.0 + f * (_a2 - 1.0)
            co[:, 1] *= s1
            co[:, 2] *= s2
            co *= (s1 * s2) ** (-1.0 / 3.0)                # volume-preserving (D2 sees only the ratios)
            Q[fm2] = co @ Vt2 + c2                          # organ's own frame, centroid (address) kept
    # MEASURED ML WIDTH PROFILE (cycle 13): the width-profile audit against the canonical adult skin
    # measured the body 30-60% too WIDE through the upper torso (epaulette flare), 2.7x too wide at
    # the neck band (the no-neck look in ML), and too NARROW at the crown (the residual spike). The
    # per-height conform below is the ML anthropometric envelope CALIBRATED to the measured canonical
    # profile (the kidney/eye/liver measured-constants pattern, applied to the whole-body silhouette):
    # each height band's ML deviations scale toward the canon halfwidth, bounded, blended by f. Runs
    # LAST of the shape ops; the measured organ placements (kidney gutters, eye separation, chambers)
    # re-assert after it in the placement phase.
    if f > 0:
        _xc = Q[:, 0]
        _statc = np.ptp(_xc) + 1e-9
        _hc = (_xc - _xc.min()) / _statc
        # deep viscera + brain + eyes are EXEMPT: they have their own measured treatments (beans,
        # wedge, chambers, coil, braincase, eye separation) and sit inside the body -- the silhouette
        # the conform calibrates is carried by muscle/skin/fat/limb. Dragging the organs distorted
        # their principal frames faster than the placement phase could re-assert them.
        _vids = [FIDX[n] for n in _exn(("Heart", "Atrium", "Ventricle", "Left Ventricle",
                                        "Right Ventricle", "Outflow", "Kidney", "Nephron", "Liver",
                                        "LiverHaem", "Lung", "Spleen", "Stomach", "Duodenum",
                                        "Foregut", "Gut", "Hindgut", "Mucosa", "Forebrain",
                                        "Telencephalon", "Midbrain", "Hindbrain", "Cerebellum",
                                        "OlfactoryBulb", "Eye", "Retina")) if n in FIDX]
        _vex = np.isin(fate, _vids) if fate is not None else np.zeros(len(Q), bool)
        for _lo, _refw in _CANON_ML_HALFW.items():
            _mb = (_hc >= _lo - 0.025) & (_hc < _lo + 0.025)
            if _mb.sum() < 30:
                continue
            _midz = float(np.median(Q[_mb, 2]))
            _curw = float(np.percentile(np.abs(Q[_mb, 2] - _midz), 98)) / _statc
            if _curw < 1e-4:
                continue
            _sc = float(np.clip(_refw / _curw, 0.60, 1.30))
            _mb2 = _mb & ~_vex
            Q[_mb2, 2] = _midz + (Q[_mb2, 2] - _midz) * (1.0 + f * (_sc - 1.0))
        # MEASURED DV DEPTH PROFILE (cycle 53): the gross-sections scorecard found the body 1.4-3x TOO
        # DEEP front-to-back everywhere below the chest -- the ML profile was measured and conformed
        # (cycle 13) but its DV twin never was. The same treatment on the DV axis, ladder measured off
        # the bp3d canon skin (98th-pct half-depth / stature per height band; the 0.40-0.45 spike is
        # the hands' forward reach at thigh level, kept as measured). FULL height: a band-uniform DV
        # scale squeezes both legs front-to-back without merging them (the ML webbing risk does not
        # exist on this axis). Same exemptions, same bounds, same blend.
        for _lo, _refd in _CANON_DV_HALFW.items():
            _mb = (_hc >= _lo - 0.025) & (_hc < _lo + 0.025)
            if _mb.sum() < 30:
                continue
            _midy = float(np.median(Q[_mb, 1]))
            _curd = float(np.percentile(np.abs(Q[_mb, 1] - _midy), 98)) / _statc
            if _curd < 1e-4:
                continue
            _scd = float(np.clip(_refd / _curd, 0.55, 1.30))
            _mb2 = _mb & ~_vex
            Q[_mb2, 1] = _midy + (Q[_mb2, 1] - _midy) * (1.0 + f * (_scd - 1.0))
    # PLACEMENT RUNS LAST (the 08-09 law, cycle-2 reorder): shape first (envelopes/isotropy/aspect above),
    # then the AP register and the MEASURED DV re-assertion in the FINAL envelope -- the aspect knobs move
    # enough organ mass (the heart is 4.7k cells) to re-shape the band envelopes, which had left the heart
    # reading 0.25 vs its 0.38 target when the spread ran mid-chain.
    if register:
        Q = _visceral_ap_register(Q, fate, f)      # migrate each organ FAMILY to its Hox-addressed axial level
    # THE HEART TAKES ITS MEASURED FORM EARLY (cycle 31, 2026-09-04; the third organ with the
    # blend-by-f disease after the kidney bean and spleen tongue): at f52 the family was a flat
    # scattered pancake (principal sds 5.4:3.6:1 vs the canon's compact 1.24:1.09:1; 165/696 cells
    # strewn to 0.10 stature) that only condensed by f75 -- while the real heart is a compact
    # chambered organ by week 10. Family-as-one-body reshape toward the measured canon sds
    # (0.0165/0.0146/0.0133 of stature, clip 0.25-1.75) + PERICARDIAL CLOSURE (the pericardium IS
    # this organ's capsule; canon max radius 2.67x sds1, cap at 2.4): ellipsoid projection of
    # stragglers, full-strength whenever the mature chain runs, BEFORE the chamber arrangement so
    # the lobes re-establish inside the compact mass. A/B: fetus 76->89-91, newborn 80->90,
    # infant 83->93, adult unchanged end-to-end (the -1 in the raw A/B was ordering artifact).
    if fate is not None and f > 0:
        _hidsM = [FIDX[n] for n in _exn(_HEART_FAMILY) if n in FIDX]
        _hmM = np.isin(fate, _hidsM)
        if _hmM.sum() >= 40:
            _statM = float(np.ptp(Q[:, 0])) + 1e-9
            cM = Q[_hmM].mean(0)
            AM = Q[_hmM] - cM
            _, _, VtM = np.linalg.svd(AM, full_matrices=False)
            locM = AM @ VtM.T
            tgtM = np.array(_HEART_SDS_M) * _statM
            locM = locM * np.clip(tgtM / (locM.std(0) + 1e-9), 0.25, 1.75)
            capM = 2.4 * tgtM
            eM = np.sqrt(((locM / capM) ** 2).sum(1))
            _outM = eM > 1.0
            if _outM.any():
                locM[_outM] = locM[_outM] / eM[_outM, None]
            Q[np.where(_hmM)[0]] = cM + locM @ VtM
    # THE CHAMBERS ARRANGE (cycle 7): the heart family registers as ONE body, but its 74-trace ceiling is
    # chamber ARRANGEMENT -- within the family, each chamber sub-group moves as one body to its MEASURED
    # offset from the heart centre (_CHAMBER_OFFSETS, bp3d anchors in heart-lengths). Model axes: left = -ML
    # (the heart leans left at ML -0.13), dorsal = +DV. The family-level dv_spread runs after and moves the
    # family as one, so the internal arrangement survives. Blended by f.
    if fate is not None and f > 0:
        _hids7 = [FIDX[n] for n in _exn(_HEART_FAMILY) if n in FIDX]
        _hm7 = np.isin(fate, _hids7)
        if _hm7.sum() >= 40:
            hc7 = Q[_hm7].mean(0)
            # heart-length scale CAPPED at the canonical heart (cycle 17b): the family's own ptp ran
            # 2-4x the real heart, throwing the ventricles to |ML| ~0.11 of stature (canon ~0.045) --
            # the wide chest slab under the shoulders that fed the star-cape. Per-organ D2 is
            # scale-blind, so the heart trace never saw it; the cape census did.
            _statH7 = float(np.ptp(Q[:, 0])) + 1e-9
            hlen7 = min(float(np.ptp(Q[_hm7], axis=0).max()), _HEART_LEN_FRAC * _statH7) + 1e-9
            for _cn, (_dsi, _dlf, _ddo) in _CHAMBER_OFFSETS.items():
                _cids7 = [FIDX[n] for n in _exn((_cn,)) if n in FIDX]
                cm7 = np.isin(fate, _cids7) & _hm7
                if cm7.sum() < 8:
                    continue
                tgt7 = hc7 + hlen7 * np.array([_dsi, _ddo, -_dlf])   # (AP, DV, ML): dorsal=+DV, left=-ML
                Q[cm7] += f * (tgt7 - Q[cm7].mean(0))
    # THE CORD ASCENDS (ascensus medullae, 2026-08-30): the vertebral column outgrows the spinal cord, so the
    # adult cord occupies the canal from the foramen magnum (medulla, ~0.87) down to the conus at L1/L2
    # (~0.60) -- the embryonic cord fills the whole canal (our build, correctly). A TWO-POINT differential-
    # growth map (v2: anchoring only the cranial end collapsed the cord to a point, because the model cord's
    # cranial end sat at ~0.62, not at the medulla -- the cord as a whole was low AND long): both ends map to
    # their canonical canal levels, interior linearly, blended by f. The eye found it: cord ran into the legs.
    if fate is not None and f > 0:
        _cids = [FIDX[n] for n in _exn(("Spinal Cord",)) if n in FIDX]
        _cmk = np.isin(fate, _cids)
        if _cmk.sum() >= 40:
            xc = Q[_cmk, 0]
            cr = float(np.percentile(xc, 97))               # current cranial end
            ca = float(np.percentile(xc, 3))                # current caudal tip
            xmin2 = Q[:, 0].min(); stat2 = np.ptp(Q[:, 0]) + 1e-9
            conus = xmin2 + _CONUS_LEVEL * stat2
            medulla = xmin2 + _MEDULLA_LEVEL * stat2
            if cr - ca > 1e-9:
                u = (xc - ca) / (cr - ca)                   # 0 = caudal .. 1 = cranial
                tgt_x = conus + u * (medulla - conus)
                Q[_cmk, 0] = xc + f * (tgt_x - xc)
    # THE TAIL REGRESSES + THE YOLK RESORBS (cycle 5): the human embryonic tail (somites 35+) regresses by
    # week 8 and the yolk sac resorbs into the midgut -- but their cells were left dangling below the pelvic
    # floor at the MIDLINE, bridging the thighs into one fused column (measured: 1016 cells in the
    # inter-thigh band -- Yolk Syncytial Layer 236, Cartilage 224, Mesothelium 142, Connective 122,
    # Mesoderm 74, Notochord 32). Axial-residue fates below the pelvic floor near the midline retract up to
    # a thin perineal-floor layer; leg fates (Muscle/Skin/Adipose/Vessel/Limb Bud) and off-midline cells
    # (the knees) are untouched. The real events are embryonic, so this completes by mid-maturation (2f).
    if fate is not None and f > 0:
        # + Gonadal fates (cycle 17): the gonads were stranded at mid-thigh midline (53 cells in the
        # 26-46% band) -- gonadal DESCENT (INSL3/gubernaculum) ends at the perineal floor, not the
        # thigh; the same retract-to-the-floor map completes the descent at its anatomical level.
        _tids = [FIDX[n] for n in _exn(("Yolk Syncytial Layer", "Mesothelium", "Cavity", "Connective",
                                        "Mesoderm", "Cartilage", "Notochord",
                                        "Gonadal Cortex", "Gonadal Medulla")) if n in FIDX]
        _bl = [FIDX[n] for n in _exn(("Bladder",)) if n in FIDX]
        _blm = np.isin(fate, _bl)
        if _blm.sum() >= 20:
            statT = np.ptp(Q[:, 0]) + 1e-9
            floor_x = float(np.percentile(Q[_blm, 0], 5)) - 0.01 * statT       # just below the bladder
            midT = float(np.median(Q[:, 2]))
            _tm = (np.isin(fate, _tids) & (Q[:, 0] < floor_x)
                   & (np.abs(Q[:, 2] - midT) < 0.02 * statT))
            if _tm.sum():
                g5 = min(1.0, 2.0 * f)
                u5 = Q[_tm, 0]
                lo5 = u5.min()
                sq = (u5 - lo5) / max(floor_x - lo5, 1e-9)                     # 0..1 up the residue column
                tgt5 = floor_x - 0.02 * statT * (1.0 - sq)                     # -> a thin perineal layer
                Q[_tm, 0] = u5 + g5 * (tgt5 - u5)
    # KIDNEY SIZE + PAIR SEPARATION (2026-08-30, cycle 3 v2): the metanephroi are compact beans in the
    # PARAVERTEBRAL gutters. Measured from BodyParts3D (FMA7204/7205 in the FMA7163 skin frame): kidney SI
    # length 103mm = 0.062 of stature, pair centroid separation 114mm = 0.069 (ratio 1.1). The model kidney
    # was ~3x TOO LONG (side SI ~18% of stature) -- a fault per-organ D2 is scale-blind to; it made the pair
    # read as one midline mass relative to its size (the honest kidney-80, and why a length-keyed separation
    # v1 overshot to +-0.17 and scored WORSE, 63). Each side shrinks isotropically to measured size (the
    # GDNF-RET branching program's output, fitted-to-measured) and sits at its measured gutter. Blended by f.
    if fate is not None and f > 0:
        _kids = [FIDX[n] for n in _exn(("Kidney", "Nephron")) if n in FIDX]
        _km = np.isin(fate, _kids)
        if _km.sum() >= 40:
            stat3 = np.ptp(Q[:, 0]) + 1e-9
            # CENTRE ON THE COLUMN, not on the family (cycle 82): the paravertebral gutters are defined
            # about the vertebral column = the body's ML midline. The family's own median sat +0.017
            # right of it (an uneven pair), so the left kidney landed short of its gutter (-0.023 vs
            # +0.057) and straddled the midline; grays "paired" read midline mass 0.48 vs flank 0.50.
            mlm = float(np.median(Q[:, 2]))
            for _sgn in (-1.0, 1.0):
                sm = _km & ((Q[:, 2] - mlm) * _sgn >= 0)
                if sm.sum() < 20:
                    continue
                c3 = Q[sm].mean(0)
                # PER-SIDE BEAN (cycle 12): the isotropic shrink left each side a BALL; the canonical
                # kidney side is a flattened bean -- per-side sds measured off the adult reference
                # cloud: (0.0200, 0.0110, 0.0058) of stature (aspect ~3.4:1.9:1). Reshape in the
                # side's OWN principal frame (its placement/tilt stays the model's).
                A3k = Q[sm] - c3
                _, _, Vt3k = np.linalg.svd(A3k, full_matrices=False)
                loc3k = A3k @ Vt3k.T
                sds3k = loc3k.std(0) + 1e-9
                tgt3k = np.array(_KIDNEY_SDS) * stat3
                # shrink-only toward measured (the expand variant overshot the smaller side: the
                # reference sides differ, 0.0222 vs 0.0177, and _KIDNEY_SDS is their mean); the
                # cycle-13 conform exempts the kidney, so no upstream squeeze needs undoing.
                # FULL-STRENGTH shape (2026-09-04, cycle 25; the spleen-v3 idiom): the fetal/newborn
                # trace sat at 68-72 with the shape blended by f -- the metanephric reniform form is
                # established by ~week 10 (the GDNF-RET branching program), so the measured bean
                # applies whenever the mature chain runs. A/B on the shipped frames: full-bean 84-90
                # vs blended 68-72; forcing measured pair separation HURT (75) -- the model's own
                # fetal separation is honest, so separation KEEPS the f blend. f=1 adult unchanged.
                loc3k = loc3k * np.minimum(tgt3k / sds3k, 1.0)
                Q[np.where(sm)[0]] = c3 + loc3k @ Vt3k
                tgt = mlm + _sgn * 0.5 * _KIDNEY_SEP_FRAC * stat3
                Q[sm, 2] += f * (tgt - float(Q[sm, 2].mean()))
    # THE SPLEEN TAKES ITS MEASURED SHAPE (2026-09-02, cycle 24; the kidney treatment): the spleen
    # trace sat at ~71 with the f51 fetal cliff at 54 -- and per-organ D2 is position-blind, so the
    # deficit was SHAPE, not the queued placement suspicion: the canon spleen is a flattened tongue
    # (principal sds 0.0190/0.0110/0.0096 of stature) where ours was a near-isotropic blob. The
    # splenic condensation program (TLX1/BAPX1/NKX2-5 in the dorsal mesogastrium) fitted-to-measured:
    # the family reshapes as ONE body in its OWN principal frame (placement/tilt stay the model's),
    # each axis toward the measured sds, expansion capped 1.75x / shrink floored 0.5x, blended by f
    # so the fetal window benefits too. GATE f>0 IS CORRECT (v4 tried unconditional to reach the
    # handoff target E0=mature_body(0) and made it WORSE, 82->57 at f50, Miles's verdict: the
    # handoff frames LERP cells between two bodies, and condensing the target STRETCHES the
    # in-flight smear -- no static reshape can fix a body in transit; the residual f51-52 dip is
    # the morph itself being scored against a static reference, a scorer-annotation item).
    if fate is not None and f > 0:
        _spids = [FIDX[n] for n in _exn(("Spleen",)) if n in FIDX]
        _spm = np.isin(fate, _spids)
        if _spm.sum() >= 25:
            stat_sp = np.ptp(Q[:, 0]) + 1e-9
            c_sp = Q[_spm].mean(0)
            A_sp = Q[_spm] - c_sp
            _, _, Vt_sp = np.linalg.svd(A_sp, full_matrices=False)
            loc_sp = A_sp @ Vt_sp.T
            sds_sp = loc_sp.std(0) + 1e-9
            tgt_sp = np.array(_SPLEEN_SDS) * stat_sp
            ratio_sp = np.clip(tgt_sp / sds_sp, 0.25, 1.75)
            # FULL-STRENGTH from the first mesh frame (v3 of the blend; Miles: "i still see the
            # spleen dip to ~50 at ~50"): the f52 family measured a 4x-too-long STREAK (sds 0.0845
            # of stature vs the 0.0203 tongue -- the mesh build strings the young family out
            # axially), which a 0.5 shrink-floor + tiny early f could never reach. Biologically the
            # spleen is a compact condensation from CS21 on, so the measured form applies whenever
            # this block runs. f=1 at the adult either way -> frozen benchmark untouched.
            loc_sp = loc_sp * ratio_sp
            # CAPSULE CLOSURE (cycle 30, 2026-09-04): the family leaves the cloud as a DIFFUSE halo
            # (49/244 cells beyond 0.04 stature at f48) and an affine scale with a 0.25 shrink floor
            # can never retrieve the persistent ~8-17-cell satellite clump (+0.06 SI, f55->f85) --
            # the stragglers stretch the D2 normalization and cost ~10 points everywhere. The
            # TLX1/BAPX1 condensation's long-range recruitment: committed splenic cells beyond the
            # MEASURED capsule (canon max radius 0.0377 stature = 2.0x sds1) migrate onto it --
            # ellipsoid projection in the family's own principal frame. A/B: fetus 69->81,
            # newborn 73->82, infant 76->84, adult 76->83.
            _cap = 2.0 * tgt_sp
            _e = np.sqrt(((loc_sp / _cap) ** 2).sum(1))
            _out = _e > 1.0
            if _out.any():
                loc_sp[_out] = loc_sp[_out] / _e[_out, None]
            Q[np.where(_spm)[0]] = c_sp + loc_sp @ Vt_sp
    # EYE PAIR to measured anatomy (cycle 4, the kidney treatment): the skin field's "temple discs" were the
    # EYES -- the family spanned 17.3% of stature in ML (3x too wide, poking out at the temples; the otic
    # pinnae measured innocent at 1.6-3.3%). Measured: eyeball 24mm = 0.0145 of stature, interpupillary
    # 63mm = 0.038. Per-side isotropic shrink to eyeball size + measured separation, each side ONE body.
    if fate is not None and f > 0:
        _eids = [FIDX[n] for n in _exn(("Eye", "Retina")) if n in FIDX]
        _em2 = np.isin(fate, _eids)
        if _em2.sum() >= 20:
            stat4 = np.ptp(Q[:, 0]) + 1e-9
            emlm = float(np.median(Q[_em2, 2]))
            for _sgn in (-1.0, 1.0):
                sm = _em2 & ((Q[:, 2] - emlm) * _sgn >= 0)
                if sm.sum() < 10:
                    continue
                c4 = Q[sm].mean(0)
                elen = float(np.ptp(Q[sm], axis=0).max()) + 1e-9
                sc = float(np.clip(_EYE_DIAM_FRAC * stat4 / elen, 0.1, 1.0))
                Q[sm] = c4 + (Q[sm] - c4) * (1.0 + f * (sc - 1.0))
                tgt = emlm + _sgn * 0.5 * _EYE_SEP_FRAC * stat4
                Q[sm, 2] += f * (tgt - float(Q[sm, 2].mean()))
    # THE JAW DESCENDS (cycle 14): the Jaw family (6.9k cells) was smeared through the WHOLE head --
    # the anoikis/crowding deaths kept culling its crown strays (548-1983 per build = the fault
    # flag), but the disease is PLACEMENT: the mandible belongs at the lower-ventral face. The
    # family moves as ONE body (the law) to its address in the head's OWN frame -- centre at 12%
    # up the head's height, ventral of the head axis by 25% of the head's DV depth -- with a mild
    # compaction toward mandible proportions. The face (chin/jawline) emerges; the vault empties.
    if fate is not None and f > 0:
        _jids = [FIDX[n] for n in _exn(("Jaw",)) if n in FIDX]
        _jm = np.isin(fate, _jids)
        if _jm.sum() >= 60 and hm.sum() >= 30:
            _hx = Q[hm, 0]
            _hlen = float(np.ptp(_hx)) + 1e-9
            _hcy = float(np.median(Q[hm, 1]))
            _hdv = float(np.ptp(Q[hm, 1])) + 1e-9
            _vsn = _ventral_sign(Q, fate)                    # which DV sign is the face side
            _tgt = np.array([float(_hx.min()) + 0.12 * _hlen,
                             _hcy + _vsn * 0.25 * _hdv,
                             float(np.median(Q[hm, 2]))])
            _jc = Q[_jm].mean(0)
            _jext = float(np.ptp(Q[_jm], axis=0).max()) + 1e-9
            _jsc = float(np.clip(0.45 * _hlen / _jext, 0.3, 1.0))   # compact to mandible scale
            Q[np.where(_jm)[0]] = _jc + (Q[_jm] - _jc) * (1.0 + f * (_jsc - 1.0))
            Q[np.where(_jm)[0]] += f * (_tgt - _jc)
            # THE SENSORY ORGANS FACE FORWARD (cycle 56; Miles's eye: "eyes at the back of the head,
            # with the ears at the nose. looks like a duck head" -- measured: Eye dv -5.0 DORSAL,
            # OlfactoryBulb -4.4 dorsal, Otic +2.2 ventral, while the Jaw sat correctly ventral +3.0).
            # The eye treatment above sets only SIZE and ML separation; the eyes' DV was never placed
            # and the inherited build position flipped with the 09-02 convention recalibration. Each
            # sensory family takes its DV address in the head's own frame with the SAME face-side
            # sign the jaw uses: eyes upper-ventral face, olfactory bulb ventral under the frontal
            # lobe, otic just DORSAL of the head axis (the ear behind the temple). DV translation
            # only, family-as-one-body; ML separation and AP addresses stay measured.
            for _snames, _dvfrac in ((("Eye", "Retina"), +0.28), (("OlfactoryBulb",), +0.30),
                                     (("Otic",), -0.06)):
                _sids = [FIDX[n] for n in _exn(_snames) if n in FIDX]
                _sm2 = np.isin(fate, _sids)
                if _sm2.sum() >= 10:
                    _sdv = _hcy + _vsn * _dvfrac * _hdv
                    Q[np.where(_sm2)[0], 1] += f * (_sdv - float(np.median(Q[_sm2, 1])))
    # THE LIVER LOBATES (cycle 11) -- measured wedge within the registered family; runs before the
    # gut coil so the coil's belly frame (below the liver) reads the true hepatic border.
    if fate is not None and f > 0:
        Q = _liver_lobation(Q, fate, f)
        Q = _liverhaem_containment(Q, fate, f)
    # THE MIDGUT COILS + THE COLON FRAMES (cycle 10) -- internal arrangement of the registered gut
    # family (the chambers pattern); runs before the DV spread, which moves each family as one.
    if fate is not None and f > 0:
        Q = _gut_coil(Q, fate, f)
    # measured DV placement runs LAST of the movers (the ascensus relocates 9k dorsal cord cells into the
    # viscera bands, which re-shapes every band's DV envelope -- spread before it read organs ~0.1 too ventral)
    if register:
        Q = _dv_measured_spread(Q, fate, strength=f)   # re-assert each viscus's MEASURED bp3d depth
    # FINAL BRAINCASE CONFORM -- genuinely the LAST word (the register's per-fate AP translations pull the
    # six brain parts to their own addresses and re-squeeze the assembly's length, so it must follow them):
    # the brain relaxes to the cranial aspect AP:DV:ML ~ 1.42:1.32:1.0; the envelopes shape the head
    # silhouette, this shapes the organ inside it.
    if brainm.sum() >= 30 and f > 0:
        bc = Q[brainm].mean(0)
        bext = np.ptp(Q[brainm], axis=0) + 1e-9
        basp = np.array([1.42, 1.32, 1.0]); basp = basp / basp.mean()
        bsc = np.clip(basp * float(bext.mean()) / bext, 0.5, 2.0)
        Q[brainm] = bc + (Q[brainm] - bc) * (1.0 + f * (bsc - 1.0))
        # THE VAULT IS A HARD BOUNDARY (cycle 4): loose brain cells plumed above the case and the skin field
        # wrapped them into the CONE CROWN (crown-region fates measured: Forebrain/Midbrain/OlfactoryBulb).
        # The skull molds the brain -- clamp outliers onto the robust braincase ellipsoid shell, blended by f.
        bidx = np.where(brainm)[0]
        P5 = Q[bidx]
        half = np.array([np.percentile(np.abs(P5[:, a] - bc[a]), 98) for a in range(3)]) + 1e-9
        rn = np.linalg.norm((P5 - bc) / half, axis=1)
        out5 = rn > 1.0
        if out5.any():
            sc5 = 1.0 + f * (1.0 / rn[out5] - 1.0)
            Q[bidx[out5]] = bc + (P5[out5] - bc) * sc5[:, None]
    return Q


# the heart FAMILY (all chambers incl the L/R ventricle split): migrated/spread as ONE body everywhere.
_HEART_FAMILY = ("Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow")
# discrete-organ families for the ISOTROPY knob (lambda in [0,1]: 0 = ride the trunk stretch,
# 1 = full counter-scale). Searched by medic.curve_train against the staged-curve objective;
# data/organ_cascade/iso_lambda.json (when present) overrides the defaults.
_ORGAN_FAMILIES = {
    "heart": _HEART_FAMILY, "kidney": ("Kidney", "Nephron"), "liver": ("Liver", "LiverHaem"),
    "lung": ("Lung",), "spleen": ("Spleen",), "stomach": ("Foregut", "Stomach", "Duodenum"),
    "pancreas": ("Pancreas",), "bladder": ("Bladder",), "thymus": ("Thymus",), "adrenal": ("Adrenal",),
}
ISO_LAMBDA = {"heart": 1.0, "kidney": 1.0, "stomach": 0.5, "liver": 1.0}   # searched winners; rest default 0
# knob set 2: family -> (a1, a2) adult principal-frame aspect (2nd/3rd axes vs the 1st), volume-preserving.
# Searched by medic.curve_train; data/organ_cascade/adult_aspect.json overrides these measured winners.
# End-to-end verdict (curve adult window): lung +5.5 / stomach +1.8 / heart +1.6; liver's inner-loop gain
# did NOT survive (flat) -- liver needs a lobation/form mechanism, not aspect. kidney/spleen ~noise-level.
# KIDNEY REMOVED (cycle 3): on a properly SEPARATED pair the family PCA axis-1 IS the pair line, so the
# family-level aspect distorts pair geometry, not organ shape (A/B on the measured pair: 61.5 -> 67 without).
# Its (1.25,1.25) had been fitted to the old midline-fused geometry. Paired organs need per-side aspect.
# Winners re-searched 2026-08-30 under scorer v2 (surface-normalised, deterministic): liver's real headroom
# was invisible to v1 (solid-vs-surface artifact) -- iso 1.0 + aspect (0.8,1.25) took its trace 70.6 -> 77.6.
ADULT_ASPECT = {"liver": (0.8, 1.25), "heart": (1.25, 1.25),
                "lung": (1.25, 1.25), "spleen": (0.8, 1.25), "stomach": (1.25, 1.25)}


def _load_iso():
    import json as _json, os as _os
    p = "data/organ_cascade/iso_lambda.json"
    if _os.path.exists(p):
        try:
            ISO_LAMBDA.update({k: float(v) for k, v in _json.load(open(p)).items()})
            print(f"  [mature] iso_lambda loaded: {ISO_LAMBDA}")
        except Exception as e:
            print(f"  [mature] iso_lambda load failed: {e}")
    q = "data/organ_cascade/adult_aspect.json"
    if _os.path.exists(q):
        try:
            ADULT_ASPECT.update({k: (float(v[0]), float(v[1])) for k, v in _json.load(open(q)).items()})
            print(f"  [mature] adult_aspect loaded: {ADULT_ASPECT}")
        except Exception as e:
            print(f"  [mature] adult_aspect load failed: {e}")


_load_iso()


# Canonical antero-posterior level of each discrete organ (fraction of standing height, crown = 1, sole = 0),
# from standard adult anatomy -- the Hox-addressed segmental level the organ's condensed mass belongs at. Only
# discrete condensed organs are addressed; spanning structures (Gut/Notochord/Spinal Cord/Vessel) and the axial
# segmental series (Rib/Cartilage/Muscle/Somite) are NOT -- they occupy a range of levels by design.
# Keyed by ORGAN name (atlas_relax_search reads it by name); the register resolves each to its FAMILY.
ORGAN_AP_ADDRESS = {
    "Forebrain": 0.94, "Eye": 0.93, "Retina": 0.93, "Midbrain": 0.92, "Otic": 0.91, "Hindbrain": 0.90,
    "Cerebellum": 0.89, "Thymus": 0.78, "Lung": 0.74, "Heart": 0.715, "Spleen": 0.66, "Liver": 0.65,
    "Stomach": 0.64, "Pancreas": 0.63, "Adrenal": 0.62, "Kidney": 0.60, "Nephron": 0.60, "Bladder": 0.48,
    # OlfactoryBulb (cycle 17e): it had NO address and its 38 cells were stranded at the crown
    # (h 0.98-1.00) -- they WERE the witch-hat cone tip. The bulb lies under the frontal lobe.
    "OlfactoryBulb": 0.925,
}
_CONUS_LEVEL = 0.60      # conus medullaris at L1/L2 -- the adult caudal end of the cord (ascensus medullae)
_MEDULLA_LEVEL = 0.87    # foramen magnum -- the adult cranial end of the cord (just below Hindbrain 0.90)
_KIDNEY_LEN_FRAC = 0.062  # kidney SI length / stature (measured: 103/1655 mm, bp3d FMA7204/5 in FMA7163)
_KIDNEY_SDS = (0.0200, 0.0110, 0.0058)  # per-side principal sds / stature (canonical adult reference; the bean)
_SPLEEN_SDS = (0.0190, 0.0110, 0.0096)  # spleen principal sds / stature (canonical adult reference; the
# flattened tongue, aspect 1.97:1.15:1 -- measured 2026-09-02, cycle 24: the spleen trace sat at ~71 and
# per-organ D2 is position-blind, so the deficit was SHAPE: our spleen was a near-isotropic blob)
# Measured canonical adult skin: per-height ML 98th-pct halfwidth / stature (the width-profile audit,
# 2026-08-31). 0.40-0.45 includes the hanging hands; 0.85-0.95 is the neck-head window. TORSO AND UP
# ONLY: the leg bands were near-canon already, and a per-band uniform ML scale below the crotch drags
# the inner-leg cells toward the midline with the outer envelope -- the gap closes, the fat-fork test
# fails, and the legs web back into a column (the eyes caught it; crotch_gap 0.807 -> 0.738).
_CANON_ML_HALFW = {
    0.50: 0.170, 0.55: 0.164, 0.60: 0.164,
    0.65: 0.152, 0.70: 0.140, 0.75: 0.141, 0.80: 0.131, 0.85: 0.049, 0.90: 0.043, 0.95: 0.047,
}
# Measured canonical per-height DV half-DEPTH / stature (cycle 53; bp3d skin FMA7163, brain-oriented
# frame, 98th-pct |dv - band median|). The ML profile's twin -- the gross-sections scorecard found the
# body 1.4-3x too deep below the chest with no DV conform to answer it. 0.40-0.45 = the hands' forward
# reach at thigh level (kept as measured); 0.90 = the face+occiput window.
_CANON_DV_HALFW = {
    0.05: 0.028, 0.10: 0.034, 0.15: 0.050, 0.20: 0.045, 0.25: 0.053, 0.30: 0.053, 0.35: 0.047,
    0.40: 0.096, 0.45: 0.107, 0.50: 0.081, 0.55: 0.077, 0.60: 0.084, 0.65: 0.094, 0.70: 0.078,
    0.75: 0.083, 0.80: 0.058, 0.85: 0.074, 0.90: 0.103, 0.95: 0.062,
}
_KIDNEY_SEP_FRAC = 0.069  # kidney pair centroid separation / stature (measured: 114/1655 mm)
_EYE_DIAM_FRAC = 0.0145   # eyeball diameter / stature (24 mm)
_EYE_SEP_FRAC = 0.038     # interpupillary distance / stature (63 mm)
# Chamber centroid offsets from the heart centre, in HEART-LENGTHS (bp3d heart 115mm), body axes
# (dSI up+, dLEFT, dDORSAL). Measured from name-verified bp3d anchors: ventricles = their papillary
# muscles (+0.05 SI: papillaries sit low in the chamber), atria = their AV valves (+0.15 SI: the valve
# plane is the chamber's INFERIOR boundary), outflow = the pulmonary valve. Textbook signs confirmed:
# LV far left, LA posterior-superior, RA rightmost, RV anterior-inferior, outflow antero-superior-left.
_CHAMBER_OFFSETS = {
    "Right Ventricle": (-0.17, +0.05, -0.25),
    "Left Ventricle":  (-0.15, +0.47, -0.04),
    "Atrium":          (+0.13, +0.11, +0.04),   # generic atrium fate -> mean of LA/RA anchors (+0.15 SI)
    "Outflow":         (+0.25, +0.16, -0.20),
}
_COIL_ROWS = 4           # serpentine rows of the packed small-intestine mass (schematic jejunum-ileum)
_HEART_LEN_FRAC = 0.070  # canonical heart long axis / stature (115/1655 mm, bp3d -- the same source as
                         # _CHAMBER_OFFSETS; cycle 17b caps the offsets' scale at the real heart size)
_HEART_SDS_M = (0.0165, 0.0146, 0.0133)   # canonical heart principal sds / stature (cycle 31; the
                         # compact near-isotropic mass, measured off the canon adult reference cloud)
# Liver wedge constants, MEASURED from the canonical adult reference cloud (stature units):
# principal sds 0.0362/0.0277/0.0183 (aspect 1.98:1.51:1 -- ours scored a BALL, sphericity 0.68 vs
# canonical 0.43); long axis oblique in the SI-ML plane, THIN end (left-lobe tip) superior-LEFT
# toward the cardia; cross-section tapers ~1.37 -> 0.63 of mean from the thick right lobe to the tip.
_LIVER_SDS = (0.0362, 0.0277, 0.0183)
_LIVER_DIR = (0.52, 0.0, -0.85)   # thick(right,+ML) -> thin(left,-ML), rising +SI; model left = -ML
_LIVER_TAPER = (1.30, 0.65)       # cross-section scale g(u) = a - b*u along thick->thin


def _liverhaem_containment(Q, fate, f):
    """CYCLE 46 -- THE HAEM COMPARTMENT COMES HOME. Fetal hepatic haematopoiesis is INTERMIXED within
    the liver (the subhead_completion note), and in the adult the programme has ended -- there is no
    anatomy for LiverHaem OUTSIDE the capsule. The relational trace read Liver|LiverHaem as separated
    and the anoikis ledger had been killing the strays (174 in the 343k build: cells scattered from
    their family lose their neighbourhood) -- two independent instruments, one fault. The Mucosa-
    follows-wall idiom: each LiverHaem cell beyond 3 liver-spacings relocates to its nearest liver
    cell's neighbourhood (jittered to spacing), blended by f."""
    hid, lid = FIDX.get("LiverHaem"), FIDX.get("Liver")
    if fate is None or f <= 0 or hid is None or lid is None:
        return Q
    mH, mL = fate == hid, fate == lid
    if mH.sum() < 8 or mL.sum() < 60:
        return Q
    from scipy.spatial import cKDTree
    TL = cKDTree(Q[mL])
    dnn, _ = TL.query(Q[mL][:: max(1, mL.sum() // 1500)], k=2)
    spacing = float(np.median(dnn[:, 1])) + 1e-9
    d, idx = TL.query(Q[mH], k=1)
    # relocate anything beyond the CONTACT range (1.2 spacings), not a looser "stray" radius -- a
    # clump at 2 spacings is neither far nor touching (the dead zone the first pass fell into)
    far = d > 1.2 * spacing
    if not far.any():
        return Q
    hi = np.where(mH)[0][far]
    rng = np.random.default_rng(hid * 7919)
    tgt = Q[mL][idx[far]] + rng.normal(size=(len(hi), 3)) * spacing * 0.6
    Q[hi] += f * (tgt - Q[hi])
    return Q


def _liver_lobation(Q, fate, f):
    """THE LIVER LOBATES (cycle 11): the liver family matures from the hepatoblast ball into the
    measured oblique WEDGE -- thick right lobe under the right diaphragm dome, thin left-lobe tip
    crossing superior-left toward the cardia (_LIVER_SDS/_LIVER_DIR/_LIVER_TAPER, canonical adult
    reference). The kidney/chambers treatment: reshape about the registered family centroid in the
    measured frame (long axis set to the measured oblique direction, principal sds to measured
    absolute size, wedge taper on the cross-section), blended by f; the AP register and DV spread
    move the family as one, so the wedge survives placement."""
    from medic.subhead_program import expand_names as _exn3
    if fate is None or f <= 0:
        return Q
    ids = [FIDX[n] for n in _exn3(("Liver", "LiverHaem")) if n in FIDX]
    m = np.isin(fate, ids)
    if m.sum() < 60:
        return Q
    stat = np.ptp(Q[:, 0]) + 1e-9
    c = Q[m].mean(0)
    A = Q[m] - c
    e1 = np.array(_LIVER_DIR, float); e1 /= np.linalg.norm(e1)
    e2 = np.array([0.0, 1.0, 0.0]) - e1 * e1[1]; e2 /= np.linalg.norm(e2)   # DV-most cross axis
    e3 = np.cross(e1, e2)
    E = np.stack([e1, e2, e3])                            # rows = the measured wedge frame
    loc = A @ E.T                                         # family coords in the wedge frame
    sds = loc.std(0) + 1e-9
    tgt_sds = np.array(_LIVER_SDS) * stat
    loc2 = loc * (1.0 + f * (tgt_sds / sds - 1.0))        # anisotropic size to measured
    u = (loc2[:, 0] - loc2[:, 0].min()) / (np.ptp(loc2[:, 0]) + 1e-9)   # 0 thick .. 1 thin
    g = _LIVER_TAPER[0] - _LIVER_TAPER[1] * u
    gm = float(g.mean())
    loc2[:, 1] *= 1.0 + f * (g / gm - 1.0)                # wedge taper (volume-neutral about the mean)
    loc2[:, 2] *= 1.0 + f * (g / gm - 1.0)
    Q[np.where(m)[0]] = c + loc2 @ E
    return Q


def _gut_coil(Q, fate, f):
    """THE MIDGUT COILS + THE COLON FRAMES (cycle 10; viable since the cycle-9 tube law grew the gut
    from 70 to ~3700 cells). Physiological herniation, 270-degree rotation and return (wk 6-10) end
    with the small intestine as a PACKED MASS of coils framed by the colon -- embryonic events, so
    the head completes by mid-maturation (2f). The small intestine (Duodenum/Gut wall) lays along a
    layered serpentine centreline filling the belly frame (below the liver, above the bladder); each
    Mucosa cell follows its NEAREST wall cell (the lining stays inside its own wall -- the subhead
    relation survives); the colon (Hindgut) lays along the peripheral frame arc: ascending on the
    RIGHT (+ML: model left = -ML, the chambers' convention), transverse under the liver, descending
    left, sigmoid back to the midline. Cells keep their tube ORDER (rank along the wall's principal
    axis), so this is a re-arrangement of the registered family, not a scramble; the partial blend
    cap leaves the natural scatter as tube thickness. Runs inside the registered family, before the
    family-level DV spread (which moves each family as one, so the coil survives it)."""
    from medic.subhead_program import expand_names as _exn2
    w = float(np.clip(2.0 * f, 0.0, 1.0)) * 0.85
    if w <= 0 or fate is None:
        return Q
    wall_ids = [FIDX[n] for n in _exn2(("Gut", "Duodenum")) if n in FIDX]
    muc_ids = [FIDX[n] for n in _exn2(("Mucosa",)) if n in FIDX]
    col_ids = [FIDX[n] for n in _exn2(("Hindgut",)) if n in FIDX]
    wm = np.isin(fate, wall_ids)
    mm = np.isin(fate, muc_ids)
    cm = np.isin(fate, col_ids)
    if wm.sum() < 60 or cm.sum() < 30:
        return Q
    x = Q[:, 0]
    stat = np.ptp(x) + 1e-9
    # ---- the belly frame: below the liver, above the bladder, the body's own local width ----
    liv = [FIDX[n] for n in _exn2(("Liver",)) if n in FIDX]
    bla = [FIDX[n] for n in _exn2(("Bladder",)) if n in FIDX]
    lm = np.isin(fate, liv); bm = np.isin(fate, bla)
    gall = wm | mm | cm
    x_top = float(np.percentile(x[lm], 8)) if lm.sum() >= 20 else float(np.percentile(x[gall], 92))
    x_bot = (float(np.percentile(x[bm], 85)) + 0.015 * stat) if bm.sum() >= 20 else float(np.percentile(x[gall], 8))
    if x_top - x_bot < 0.04 * stat:                       # degenerate frame: leave the build alone
        return Q
    band = (x > x_bot) & (x < x_top)
    mid_ml = float(np.median(Q[:, 2]))
    half_ml = 0.55 * float(np.percentile(np.abs(Q[band, 2] - mid_ml), 90)) if band.sum() > 50 else 0.06 * stat
    dv0 = float(np.median(Q[gall, 1]))                    # the family's registered DV plane
    # ---- small intestine: serpentine rows stacked SI, alternating ML sweep ----
    P1 = Q[wm]
    c1 = P1.mean(0)
    A = P1 - c1
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    s_rank = np.argsort(np.argsort(A @ Vt[0])) / max(wm.sum() - 1, 1)   # 0..1 along the tube's own axis
    R = _COIL_ROWS
    row = np.minimum((s_rank * R).astype(int), R - 1)
    u = s_rank * R - row                                   # 0..1 within the row
    sweep = np.where(row % 2 == 0, u, 1.0 - u)             # alternate direction each row
    margin = 0.10
    tx = x_top - (row + 0.5) / R * (x_top - x_bot)
    tml = mid_ml + (sweep * 2.0 - 1.0) * half_ml * (1.0 - margin)
    tdv = dv0 + 0.12 * half_ml * np.sin(u * 2.0 * np.pi * 2.0)   # gentle DV undulation within each row
    tgt_w = np.stack([tx, tdv, tml], 1)
    idxw = np.where(wm)[0]
    Q[idxw] = Q[idxw] + w * (tgt_w - Q[idxw])
    # ---- mucosa follows its nearest wall cell (luminal lining) ----
    if mm.sum() >= 8:
        from scipy.spatial import cKDTree as _KD
        _, nnw = _KD(P1).query(Q[mm], k=1)
        idxm = np.where(mm)[0]
        Q[idxm] = Q[idxm] + w * (tgt_w[nnw] - Q[idxm])
    # ---- colon: the peripheral frame arc (ascending +ML, transverse top, descending -ML, sigmoid) ----
    P3 = Q[cm]
    c3 = P3.mean(0)
    A3 = P3 - c3
    _, _, Vt3 = np.linalg.svd(A3, full_matrices=False)
    proj3 = A3 @ Vt3[0]
    # PROXIMAL POLE (cycle 82d): the tube's principal axis has an arbitrary sign; the caecum end is the
    # CRANIAL end of the model's hindgut (it joins the midgut at a>=0.52) = high x in the registered
    # family -> t3 = 0 there, so the subhead labels (subhead_program: Hindgut axis split, rank 0 at
    # high x) and the arc segments (ascending < 0.30 < transverse < 0.62 < descending < 0.90 < sigmoid)
    # name the same cells; the caecum lands lower right, the rectum at the midline (the 270-degree
    # midgut rotation's end state).
    if np.corrcoef(proj3, P3[:, 0])[0, 1] > 0:
        proj3 = -proj3
    t3 = np.argsort(np.argsort(proj3)) / max(cm.sum() - 1, 1)
    edge = half_ml * (1.0 + margin)
    tx3 = np.empty(cm.sum()); tml3 = np.empty(cm.sum())
    a_seg = t3 < 0.30                                      # ascending: right edge, bottom -> top
    tx3[a_seg] = x_bot + (t3[a_seg] / 0.30) * (x_top - x_bot)
    tml3[a_seg] = mid_ml + edge
    t_seg = (t3 >= 0.30) & (t3 < 0.62)                     # transverse: along the top, right -> left
    uu = (t3[t_seg] - 0.30) / 0.32
    tx3[t_seg] = x_top
    tml3[t_seg] = mid_ml + edge - uu * 2.0 * edge
    d_seg = (t3 >= 0.62) & (t3 < 0.90)                     # descending: left edge, top -> bottom
    uu = (t3[d_seg] - 0.62) / 0.28
    tx3[d_seg] = x_top - uu * (x_top - x_bot)
    tml3[d_seg] = mid_ml - edge
    s_seg = t3 >= 0.90                                     # sigmoid: short leg back toward the midline
    uu = (t3[s_seg] - 0.90) / 0.10
    tx3[s_seg] = x_bot
    tml3[s_seg] = mid_ml - edge * (1.0 - uu)
    tgt3 = np.stack([tx3, np.full(cm.sum(), dv0), tml3], 1)
    idxc = np.where(cm)[0]
    Q[idxc] = Q[idxc] + w * (tgt3 - Q[idxc])
    return Q


def _visceral_ap_register(Q, fate, f, max_target=None):
    """THE AP-ADDRESS HEAD: slide each discrete organ's condensed mass along the body axis to its canonical
    Hox-addressed level, read-only on every other cell (the build cloud snaps organs onto coarse body-electric
    antinodes near the middle, so the viscera sag and scramble). A pure AP translation per ORGAN FAMILY, blended
    by f so f=0 is the identity.

    v2 (2026-08-30): EVERY family migrates as ONE body -- the heart's stale-fate bug #5, generalised. The old
    per-name loop could not see sub-head children (Lung -> lobes, Liver -> hepatic lobes, ...), so exactly the
    split organs never migrated (measured: lung stuck at 56% vs 74, liver at 32% vs 65, bladder at 14% vs 48,
    kidney TORN partway, while heart/spleen/thymus/adrenal -- unsplit names -- all landed). One family, one
    translation, expand_names every fate-keyed table; a done-mask so no cell moves twice. Stomach gained its
    missing address (0.64 -- it had NONE and drifted to heart level)."""
    if fate is None or f <= 0:
        return Q
    x = Q[:, 0]; xmin = x.min(); stat = np.ptp(x) + 1e-9
    done = np.zeros(len(Q), bool)

    def _move(names, target):
        fids = [FIDX[n] for n in _exn(tuple(names)) if n in FIDX]
        m = np.isin(fate, fids) & ~done
        if m.sum() < 8:
            return
        cur = float(((x[m] - xmin) / stat).mean())
        Q[m, 0] += f * (target - cur) * stat        # translate the family to its address (read-only elsewhere)
        done[m] = True

    for fam, members in _ORGAN_FAMILIES.items():    # viscera: family address = the parent organ's entry
        target = ORGAN_AP_ADDRESS.get(fam.capitalize())
        if target is not None and (max_target is None or target <= max_target):
            _move(members, target)
    for nm, target in ORGAN_AP_ADDRESS.items():     # head parts + anything not family-covered
        if max_target is None or target <= max_target:
            _move((nm,), target)
    return Q


def standing_register(Q, fate, f=1.0):
    """THE STANDING REGISTER (cycle 17c): ORGAN_AP_ADDRESS is defined as a fraction of STANDING
    height (crown = 1, sole = 0), but the placement phase runs inside mature_cloud, BEFORE
    grow_limbs extends the legs -- so once the legs added stature below, every address rode high
    in the finished body (measured on the shipped adult: heart 0.79 vs 0.715, liver 0.73 vs 0.65,
    bladder 0.61 vs 0.48; the atria/ventricle edges at 0.77-0.81 WERE the collar-spike mass under
    the shoulders). Per-organ D2 is translation-blind, so only the eyes and the census saw it.
    After the legs extend, the same registers re-assert on the standing frame: the same families,
    the same measured constants, AP translation only (DV/ML untouched, so the spread's arrangement
    survives); the cord re-maps to its canal levels; the perineal residue and the gonads re-anchor
    just below the re-registered bladder."""
    if fate is None or f <= 0:
        return Q
    # SUB-CRANIAL ONLY (the domeness collapse taught it): the head does not ride the leg
    # extension -- it defines the top of the stature -- and re-registering the cephalic entries
    # here pulled the brain ~2% down OUT of the vault the braincase conform had closed around it
    # (domeness 0.60 -> 0.215, the brain-not-in-vault lesson again). Everything with an address
    # below the neck re-asserts; the head keeps its build-time assembly.
    Q = _visceral_ap_register(Q, fate, f, max_target=0.85)
    from medic.subhead_program import expand_names as _exnS
    x = Q[:, 0]
    xmin = float(x.min())
    stat = float(np.ptp(x)) + 1e-9
    # the cord's two-point canal map, re-asserted on the standing stature
    _cids = [FIDX[n] for n in _exnS(("Spinal Cord",)) if n in FIDX]
    _cmk = np.isin(fate, _cids)
    if _cmk.sum() >= 40:
        xc = Q[_cmk, 0]
        cr = float(np.percentile(xc, 97))
        ca = float(np.percentile(xc, 3))
        if cr - ca > 1e-9:
            u = (xc - ca) / (cr - ca)
            tgt_x = (xmin + _CONUS_LEVEL * stat) + u * ((_MEDULLA_LEVEL - _CONUS_LEVEL) * stat)
            Q[_cmk, 0] = xc + f * (tgt_x - xc)
    # perineal residue + gonads: the mature-time retract anchored them to the PRE-extension floor,
    # which now sits mid-abdomen; re-anchor the midline residue band around the OLD floor into the
    # thin layer just below the re-registered bladder (order preserved).
    _tids = [FIDX[n] for n in _exnS(("Yolk Syncytial Layer", "Mesothelium", "Cavity", "Connective",
                                     "Mesoderm", "Cartilage", "Notochord",
                                     "Gonadal Cortex", "Gonadal Medulla")) if n in FIDX]
    _bl = [FIDX[n] for n in _exnS(("Bladder",)) if n in FIDX]
    _blm = np.isin(fate, _bl)
    if _blm.sum() >= 20:
        # floor from the bladder MEDIAN, not its p5 (cycle 17d): the family's lower tail hangs
        # ~0.05 below its centre, so the p5 floor landed at 0.42 and dragged the crotch with it;
        # the pubis bottom sits half a (canonical) bladder height below the bladder centre.
        floor_x = float(np.median(Q[_blm, 0])) - 0.035 * stat
        midT = float(np.median(Q[:, 2]))
        _tm = (np.isin(fate, _tids) & (np.abs(Q[:, 2] - midT) < 0.02 * stat)
               & (Q[:, 0] > floor_x - 0.06 * stat) & (Q[:, 0] < floor_x + 0.15 * stat))
        if _tm.sum() >= 8:
            xi = Q[_tm, 0]
            rk = np.argsort(np.argsort(xi)) / max(len(xi) - 1, 1)
            tgt5 = floor_x - 0.02 * stat * (1.0 - rk)
            Q[_tm, 0] = xi + f * (tgt5 - xi)
        # THE COCCYX CONDENSES (cycle 17d): the retract map had stacked the tail's skeletal
        # residue (Cartilage 486 + Notochord 148 measured) into a razor-thin MIDLINE wafer at
        # the floor -- but the embryonic tail skeleton is the sacrococcygeal column: it belongs
        # as a compact body at the DORSAL pelvic floor, behind the pelvic outlet, not spread
        # across it. Condense those fates (floor band, near-midline) to a small ball seated at
        # the floor level against the dorsal wall; the ventral midline empties for the crotch.
        _ccids = [FIDX[n] for n in _exnS(("Cartilage", "Notochord", "Mesoderm")) if n in FIDX]
        _ccm = (np.isin(fate, _ccids) & (np.abs(Q[:, 2] - midT) < 0.03 * stat)
                & (Q[:, 0] > floor_x - 0.05 * stat) & (Q[:, 0] < floor_x + 0.04 * stat))
        if _ccm.sum() >= 30:
            dvs = _ventral_sign(Q, fate)
            fb = np.abs(Q[:, 0] - floor_x) < 0.05 * stat
            y_band = Q[fb, 1]
            dorsal_wall = float(np.percentile(-dvs * y_band, 88))
            tgt_c = np.array([floor_x + 0.01 * stat, -dvs * (dorsal_wall - 0.01 * stat),
                              midT])
            P_c = Q[_ccm]
            c_c = P_c.mean(0)
            shrink = np.clip(0.012 * stat / (P_c.std(0) + 1e-9), 0.1, 1.0)
            Q[_ccm] = P_c + f * ((c_c + (P_c - c_c) * shrink) + (tgt_c - c_c) - P_c)
        # THE HIP SEATS AT THE PELVIC FLOOR (cycle 17d): with the viscera standing at their true
        # addresses, the legs' columns still topped out at ~0.43 of stature against a floor at
        # ~0.47 -- the missing 4% is the upper thigh, and it is why the figure read short-legged
        # (canon crotch 0.48). The limb attaches AT its girdle (the femoral head in the
        # acetabulum; the arms' seat-below-the-head rule, applied to the hips): everything below
        # the legs' old top stretches upward from the sole to meet the floor; the thin band of
        # perineal contents between old top and floor compresses onto the floor's underside.
        # Sole and crown unchanged, so the stature and every registered address hold.
        if "Limb Bud" in FIDX:
            leg2 = (fate == FIDX["Limb Bud"]) & (Q[:, 0] < floor_x)
            if leg2.sum() >= 100:
                t_leg = float(np.percentile(Q[leg2, 0], 97))
                sole = float(np.percentile(Q[:, 0], 0.2))
                if t_leg < floor_x - 0.005 * stat and t_leg - sole > 0.10 * stat:
                    s = min((floor_x - sole) / (t_leg - sole), 1.15)
                    low = Q[:, 0] < t_leg
                    Q[low, 0] = sole + (Q[low, 0] - sole) * (1.0 + f * (s - 1.0))
                    band = (Q[:, 0] >= t_leg) & (Q[:, 0] < floor_x) & ~low
                    Q[band, 0] = floor_x - (floor_x - Q[band, 0]) * (1.0 - f * 0.7)
    # THE CROWN DOMES (polish cycle, v2 -- the p95 trim failed: the tails are >5% of their
    # families, and clamping them DOWN would shrink the stature the registers just used). The
    # telencephalic vesicles expand LATERALLY under the vault -- the brain grows wide, not
    # pointed -- so cephalic cells above the collective 92nd percentile redistribute onto a
    # spherical cap of the head's own radius (deterministic golden-angle disks, height kept):
    # the midline column that the field tapered into the topknot becomes a closed dome.
    _bfam = [FIDX[n] for n in _exnS(("Forebrain", "Telencephalon", "Midbrain", "Hindbrain",
                                     "Cerebellum", "OlfactoryBulb", "Eye", "Retina")) if n in FIDX]
    _cm2 = np.isin(fate, _bfam)
    if _cm2.sum() >= 200:
        xh = Q[_cm2, 0]
        x_cap = float(np.percentile(xh, 92))
        x_top = float(np.percentile(xh, 99.8))
        span_c = max(x_top - x_cap, 1e-6)
        yc = float(np.median(Q[_cm2, 1]))
        zc = float(np.median(Q[_cm2, 2]))
        rr = np.hypot(Q[_cm2, 1] - yc, Q[_cm2, 2] - zc)
        R_head = float(np.percentile(rr, 85))
        hi = _cm2 & (Q[:, 0] > x_cap)
        idxh = np.where(hi)[0]
        if len(idxh) >= 20:
            u = np.clip((Q[idxh, 0] - x_cap) / span_c, 0.0, 1.0)
            allowed = 0.85 * R_head * np.sqrt(np.clip(1.0 - u ** 2, 0.05, 1.0))
            k = np.arange(len(idxh))
            gr = allowed * np.sqrt((k % 89) / 89.0)
            th = k * 2.399963
            Q[idxh, 1] += f * ((yc + gr * np.sin(th)) - Q[idxh, 1])
            Q[idxh, 2] += f * ((zc + gr * np.cos(th)) - Q[idxh, 2])
    # THE STANDING CONFORM (cycle 17e, the frame bug's third instance): the measured canonical
    # width profile is per-height of the STANDING body, but the mature-time conform runs before
    # the legs extend -- its neck band squeezed what became the crown, and the true neck kept
    # torso width (measured: neck-level mass at |ML| 0.084-0.105 vs the canonical 0.049 -- the
    # residual collar spikes). The same constants re-assert on the standing bands; the deep
    # viscera, brain and eyes stay exempt exactly as in the mature-time pass.
    _vids2 = [FIDX[n] for n in _exnS(("Heart", "Atrium", "Ventricle", "Left Ventricle",
                                      "Right Ventricle", "Outflow", "Kidney", "Nephron", "Liver",
                                      "LiverHaem", "Lung", "Spleen", "Stomach", "Duodenum",
                                      "Foregut", "Gut", "Hindgut", "Mucosa", "Forebrain",
                                      "Telencephalon", "Midbrain", "Hindbrain", "Cerebellum",
                                      "OlfactoryBulb", "Eye", "Retina")) if n in FIDX]
    _vex2 = np.isin(fate, _vids2)
    _xc2 = Q[:, 0]
    _statc2 = np.ptp(_xc2) + 1e-9
    _hc2 = (_xc2 - _xc2.min()) / _statc2
    # CONTINUOUS target profile (the Saturn-ring lesson): per-band constant targets make a hard
    # shelf wherever the canon profile steps (shoulder 0.131 -> neck 0.049 at 0.85 minted a
    # razor-thin brim in the render). The target interpolates smoothly in height, and the
    # current width is measured on fine smoothed bands, so the conform is shelf-free.
    from scipy.ndimage import uniform_filter1d as _uf1
    _keys = np.array(sorted(_CANON_ML_HALFW))
    _vals = np.array([_CANON_ML_HALFW[k] for k in _keys])
    _nb2 = 44
    _edges = np.linspace(0.475, 0.985, _nb2 + 1)
    _cen = 0.5 * (_edges[:-1] + _edges[1:])
    _curw_b = np.full(_nb2, np.nan)
    _midz_b = np.full(_nb2, 0.0)
    for _k in range(_nb2):
        _mb = (_hc2 >= _edges[_k]) & (_hc2 < _edges[_k + 1])
        if _mb.sum() < 20:
            continue
        _midz_b[_k] = float(np.median(Q[_mb, 2]))
        _curw_b[_k] = float(np.percentile(np.abs(Q[_mb, 2] - _midz_b[_k]), 98)) / _statc2
    _ok = ~np.isnan(_curw_b)
    if _ok.sum() >= 6:
        _curw_s = _curw_b.copy()
        _curw_s[_ok] = _uf1(_curw_b[_ok], 3)
        _tgt_b = np.interp(_cen, _keys, _vals)
        _sc_b = np.clip(_tgt_b / np.maximum(_curw_s, 1e-4), 0.60, 1.30)
        _in = (_hc2 >= 0.475) & (_hc2 < 0.985) & ~_vex2
        _ki = np.clip(((_hc2[_in] - 0.475) / (0.985 - 0.475) * _nb2).astype(int), 0, _nb2 - 1)
        _okc = _ok[_ki]
        _idx2 = np.where(_in)[0][_okc]
        _kk = _ki[_okc]
        Q[_idx2, 2] = (_midz_b[_kk] + (Q[_idx2, 2] - _midz_b[_kk])
                       * (1.0 + f * (_sc_b[_kk] - 1.0)))
    # ---- PLEURAL/PERITONEAL CONTAINMENT (cycle 70; Miles's eye: "cells outside the skin") ----
    # The conform narrows the WALL but exempts the viscera (their frames must not distort), so the
    # chest stack could protrude through the conformed envelope (census: 320 cells at h~0.77 incl
    # lung-lobe cells, 0.08-0.18 stature outside). The body wall is a HARD BOUNDARY (mesothelium):
    # any trunk-band cell beyond the measured canonical ML/DV halfwidth at its height -- exempt
    # viscera included -- CLAMPS to just inside the wall. Only the protruding tail moves (organ
    # frames keep their bulk); Limb Bud is excluded (arms/hands protrude laterally by design).
    _lb_id = FIDX.get("Limb Bud", -1)
    _dvk = np.array(sorted(_CANON_DV_HALFW))
    _dvv = np.array([_CANON_DV_HALFW[k] for k in _dvk])
    _mby_b = np.full(_nb2, 0.0)
    for _k in range(_nb2):
        _mb = (_hc2 >= _edges[_k]) & (_hc2 < _edges[_k + 1])
        if _mb.sum() >= 20:
            _mby_b[_k] = float(np.median(Q[_mb, 1]))
    _trunk = (_hc2 >= 0.475) & (_hc2 < 0.985) & (fate != _lb_id)
    _kt = np.clip(((_hc2[_trunk] - 0.475) / (0.985 - 0.475) * _nb2).astype(int), 0, _nb2 - 1)
    _ti = np.where(_trunk)[0]
    _wall_ml = np.interp(_cen, _keys, _vals)[_kt] * _statc2
    _wall_dv = np.interp(_cen, _dvk, _dvv)[_kt] * _statc2
    _offz = Q[_ti, 2] - _midz_b[_kt]
    _offy = Q[_ti, 1] - _mby_b[_kt]
    _ozc = np.abs(_offz) > 1.03 * _wall_ml
    _oyc = np.abs(_offy) > 1.03 * _wall_dv
    Q[_ti[_ozc], 2] = _midz_b[_kt[_ozc]] + np.sign(_offz[_ozc]) * 0.99 * _wall_ml[_ozc]
    Q[_ti[_oyc], 1] = _mby_b[_kt[_oyc]] + np.sign(_offy[_oyc]) * 0.99 * _wall_dv[_oyc]
    # LEG CONTAINMENT (cycle 71, Miles's eye: "leg cells outside the skin" -- the trunk clamp
    # starts at h 0.475, the legs never had a wall; census: Limb Bud strays at ankle height
    # |ML| 0.18 where the legs stand at 0.06). Below ANY hand (h < 0.25) the only lateral mass
    # is the two legs: each side's cells clamp ML to its own column line +- the tube wall
    # (thigh radius x flesh margin). ML only -- the feet keep their forward (DV) reach.
    _legb = _hc2 < 0.25
    if _legb.sum() > 200:
        _mid3 = float(np.median(Q[:, 2]))
        _rw = 1.35 * 0.040 * _statc2
        for _sgn in (-1.0, 1.0):
            _ms = _legb & (np.sign(Q[:, 2] - _mid3) == _sgn)
            if _ms.sum() < 50:
                continue
            _line = float(np.median(Q[_ms, 2]))
            _off3 = Q[_ms, 2] - _line
            _oc3 = np.abs(_off3) > _rw
            _mi3 = np.where(_ms)[0][_oc3]
            Q[_mi3, 2] = _line + np.sign(_off3[_oc3]) * 0.99 * _rw
    # KIDNEY GUTTERS RE-ASSERTED (cycle 82; the frame law: every measured constant is defined on the
    # STANDING body). The pair separation (114/1655 mm about the column) was applied in mature_cloud
    # on the pre-scale stature and centred on the family's own median, and the conform / containment
    # pulled the sides inward after it -- the left kidney ended at -0.045 vs the right at +0.111 and
    # grays "paired" failed on the standing body. Each side lands at +-0.5 * sep about the body's ML
    # midline (the column), moved as one body so the measured bean is kept; f-blended like every
    # measured placement here.
    _kids4 = [FIDX[n] for n in _exn(("Kidney", "Nephron")) if n in FIDX]
    _km4 = np.isin(fate, _kids4)
    if _km4.sum() >= 40:
        _mid4 = float(np.median(Q[:, 2]))
        for _sgn in (-1.0, 1.0):
            _ms4 = _km4 & ((Q[:, 2] - _mid4) * _sgn >= 0)
            if _ms4.sum() < 20:
                continue
            _tgt4 = _mid4 + _sgn * 0.5 * _KIDNEY_SEP_FRAC * _statc2
            Q[_ms4, 2] += f * (_tgt4 - float(Q[_ms4, 2].mean()))
    return Q


_DIGIT_NAMES = {"hand": ("Thumb", "Index Finger", "Middle Finger", "Ring Finger", "Little Finger"),
                "foot": ("Great Toe", "Second Toe", "Third Toe", "Fourth Toe", "Little Toe")}
_META_NAME = {"hand": "Metacarpal", "foot": "Metatarsal"}
_ORDINAL = ("First", "Second", "Third", "Fourth", "Fifth")


def populate_autopods(Q, fate, frac=1.0, labels_out=None):
    """THE AUTOPODS GET CELLS (Miles, frame 91: 'the hands are empty, no cells in them'). The
    hand/foot volumes were MESH-ONLY -- shells densified into the skin field -- so when the
    anatomy reveal dissolved the skin there were no cells to reveal: empty gloves. The autopod
    territory is real tissue: each limb's DISTAL cells extend into its own hand/foot volume
    (the Hox13 autopod populated by its own limb, the same cells that built the stylopod and
    zeugopod). Deterministic: distal-band cells land on the schematic autopod's shell vertices
    (cycled) with a small inward jitter; ~30% stay at the wrist/ankle for continuity. Returns
    (Q, hv, hf, fv, ff) -- the shells it used, so the skin field can reuse them EXACTLY (the
    field must not re-derive anchors from the now-populated limb: the hand would walk).
    Idempotent per frame: each movie frame rebuilds Q fresh, then populates once."""
    Q = Q.copy()
    hv, hf = hands_mesh(Q, fate, frac=frac)
    fv, ff = feet_mesh(Q, fate, frac=frac)
    if frac <= 0.05:
        return Q, hv, hf, fv, ff
    x, z = Q[:, 0], Q[:, 2]
    H = float(np.ptp(x)) + 1e-9
    apf = (x - x.min()) / H
    mid = float(np.median(z))
    limbm = np.asarray(fate) == LIMB
    for shell, m_band, FLOOR, kind in (
            (hv, limbm & (apf >= 0.45), 320, "hand"),       # arms -> hands
            (fv, limbm & (apf < 0.30), 420, "foot")):       # legs -> feet
        shell = np.asarray(shell, float)
        if len(shell) < 12 or m_band.sum() < 30:
            continue
        from medic.hand_foot_skin import NV as _NV, _NDIG as _ND, _NRING as _NRG, _NSIDE as _NS
        _blocks = [shell[b:b + _NV] for b in range(0, len(shell) - _NV + 1, _NV)]   # one fixed-layout block per side
        for sgn in (-1.0, 1.0):
            ms = m_band & (np.sign(z - mid) == sgn)
            blk = next((B for B in _blocks if np.sign(B[:, 2].mean() - mid) == sgn), None)
            if blk is None or ms.sum() < 20:
                continue
            xi = np.where(ms)[0]
            # THE AUTOPOD ALLOCATION (cycle 65, allocation-before-condensation -- the foot-completion
            # conviction: ~30-80 cells rode the whole ladder, below the level tests' noise floor).
            # The autopod draws its measured complement from the limb's OWN pool -- distal
            # proliferation supplies the Hox13 territory -- so instead of whatever the thin 12%% band
            # holds, each autopod takes its N MOST-DISTAL limb cells (feet 420, hands 320; ~3%% of
            # the limb column, negligible thinning).
            order = xi[np.argsort(Q[xi, 0])]
            band = order[: min(FLOOR, len(order))]
            if len(band) < 8:
                continue
            take = band[np.arange(len(band)) % 10 < 7]      # ~70% populate; 30% stay for continuity
            # THE SOX9 RAY HEAD (cycle 66): the old landing cycled cells across ALL shell verts
            # interleaved -- geometrically spread but without ray identity, so the completion
            # instrument read flickering noise. The condensation is made COHERENT: cells partition
            # into 5 ML-contiguous groups (the lateral-inhibition ray identities) matched to the
            # shell's 5 ML-contiguous digit tubes, and WITHIN each ray both cells and verts are
            # ordered along the digit's long axis (proximal -> distal) -- the ray becomes a real
            # condensation AND lays the ordered substrate the GDF5 segment interzones cut next.
            # THE DIGIT-LABELLED LANDING (cycle 82e; the ray head's open rung). The cycle-66 landing
            # split cells AND verts into five groups by ML order -- but the digit tubes overlap in ML
            # (and the hand's fan runs across DV, palm toward the thigh), so the groups smeared across
            # tubes and the completion instrument read PLATE at every stage, RAYS NEVER. hand_foot_skin
            # has a FIXED layout (5 digits x 3 rings x 6 sides, then a 12-vert palm), so every shell
            # vertex carries its digit and its ring by index. Per side: the most PROXIMAL quarter of
            # the cells (along the mean digit direction) becomes the palm/sole = the metacarpal/tarsal
            # mass; the rest split into five contiguous groups along the FAN axis (PCA-1 of the digit
            # centroids) and land on the digit at the same fan position, ordered along THAT digit's
            # own axis, ring 0 -> ring 2 = proximal -> distal. The rings are the phalangeal segments
            # the GDF5 interzones name next.
            dig = [blk[d * _NRG * _NS:(d + 1) * _NRG * _NS] for d in range(_ND)]
            palm = blk[_ND * _NRG * _NS:]
            cen = np.array([d_.mean(0) for d_ in dig])
            axes = np.array([d_[(_NRG - 1) * _NS:].mean(0) - d_[:_NS].mean(0) for d_ in dig])
            out_ax = axes.mean(0); out_ax = out_ax / (np.linalg.norm(out_ax) + 1e-12)
            _, _, Vf = np.linalg.svd(cen - cen.mean(0), full_matrices=False)
            fan_ax = Vf[0]
            jit = 0.004 * H
            prox = Q[take] @ out_ax
            o = np.argsort(prox)
            n_palm = max(4, int(0.25 * len(take)))
            palm_c, digit_c = take[o[:n_palm]], take[o[n_palm:]]
            side_nm = "Left" if sgn < 0 else "Right"          # model z: -1 left, +1 right (laterality)
            dorder = np.argsort(cen @ fan_ax)                  # digit index at each fan position
            # THE AUTOPOD NAMES (cycle 82g): the landing knows every cell's bone -- palm/sole cells take
            # the metacarpal / metatarsal at their fan position, digit cells the phalanx of their ring
            # (digit 1 has no middle phalanx) -- so the scored body's fates can carry the 76 named
            # bones the FMA ledger counts (subhead_program's deferred autopod roster).
            if labels_out is not None and len(palm_c) >= _ND:
                pgs = np.array_split(np.argsort(Q[palm_c] @ fan_ax), _ND)
                for g, d in zip(pgs, dorder):
                    fidn = FIDX.get(f"{side_nm} {_ORDINAL[d]} {_META_NAME[kind]} Bone")
                    if fidn is not None and len(g):
                        labels_out[palm_c[g]] = fidn
            pv = palm[np.argsort(palm @ out_ax)]
            tgtp = pv[np.linspace(0, len(pv) - 1, len(palm_c)).astype(int)]
            kk = np.arange(len(palm_c))
            offp = np.stack([np.sin(kk * 2.4), np.cos(kk * 1.7), np.sin(kk * 3.1)], 1) * jit
            Q[palm_c] = Q[palm_c] + frac * ((tgtp + offp) - Q[palm_c])
            if len(digit_c) >= 2 * _ND:
                cgs = np.array_split(np.argsort(Q[digit_c] @ fan_ax), _ND)
                for g, d in zip(cgs, dorder):
                    if len(g) < 1:
                        continue
                    ci = digit_c[g]
                    ax = axes[d] / (np.linalg.norm(axes[d]) + 1e-12)
                    ci = ci[np.argsort(Q[ci] @ ax)]                       # proximal -> distal along this digit
                    vv = dig[d]                                           # ring-major: proximal -> distal
                    vi = np.linspace(0, len(vv) - 1, len(ci)).astype(int)
                    tgt = vv[vi]
                    k = np.arange(len(ci))
                    off = np.stack([np.sin(k * 2.4), np.cos(k * 1.7), np.sin(k * 3.1)], 1) * jit
                    Q[ci] = Q[ci] + frac * ((tgt + off) - Q[ci])
                    if labels_out is not None:
                        ring = vi // _NS                                  # 0 proximal .. 2 distal
                        phal = (np.where(ring <= 1, "Proximal", "Distal") if d == 0
                                else np.array(("Proximal", "Middle", "Distal"))[ring])
                        dig_nm = _DIGIT_NAMES[kind][d]
                        for ph in np.unique(phal):
                            fidn = FIDX.get(f"{ph} Phalanx of {side_nm} {dig_nm}")
                            if fidn is not None:
                                labels_out[ci[phal == ph]] = fidn
    return Q, hv, hf, fv, ff


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

_RIGID_IDS = None        # cached compact-organ family id sets for the curl's organ-rigid transport


def _rigid_family_ids():
    global _RIGID_IDS
    if _RIGID_IDS is None:
        from medic.subhead_program import expand_names as _exn2
        fams = [["Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow"],
                ["Liver", "LiverHaem"], ["Kidney", "Nephron"], ["Lung"], ["Spleen"],
                ["Stomach", "Duodenum"], ["Eye"], ["Otic"], ["Pancreas"]]
        _RIGID_IDS = [[FIDX[n] for n in _exn2(f) if n in FIDX] for f in fams]
        _RIGID_IDS = [ids for ids in _RIGID_IDS if ids]
    return _RIGID_IDS


def _ce_elong(t):
    """Convergent-extension elongation schedule vs the global clock t (cycle 26, 2026-09-04).
    The Carnegie reference is a LONG THIN body through the somite window (whole-body principal aspect
    1:0.21-0.23 at CS10-11) where the model grew a fat cigar (1:0.40-0.48) -- Wnt-PCP convergent
    extension (mediolateral intercalation driving axial elongation, the presomitic-mesoderm CDX2/HOX
    program) peaks exactly there and the cloud growth lacked it. Volume-preserving AP stretch factor,
    measured by A/B sweep against the staged references on the shipped frames (optima: CS08 ~1.35,
    CS09 1.55, CS10 1.6-1.8, CS11 1.8-2.0, CS12 1.2-1.4, CS13 ~1.2): rises through the disc->somite
    transition, peaks 1.9 at CS11, relaxes as the C closes and lateral/organ growth catches up (the
    reference itself compacts to 1:0.40 by CS12). Deep-curl frames (t>0.24) taper conservatively --
    the post-curl affine A/B is unreliable there (the deep-curl instrument law)."""
    pts = [(0.000, 1.00), (0.045, 1.00), (0.090, 1.55), (0.112, 1.55), (0.168, 1.90),
           (0.191, 1.90), (0.213, 1.35), (0.236, 1.20), (0.300, 1.00), (1.000, 1.00)]
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


def ce_stretch(Q, k, fate=None):
    """Volume-preserving axial (AP) stretch: x*k, y,z/sqrt(k) about the centroid. ORGAN-RIGID
    (the fetal_curl transport idiom): compact organ families ride the stretch -- their centroid
    moves with the space but their internal shape is kept (CE elongates the axis by intercalation
    in the axial/paraxial tissue; the heart does not stretch). Spanning structures (CNS, gut, skin,
    somites) take the per-cell stretch -- they DO elongate. Measured on the shipped frames: affine
    stretch cost the heart 94->82 at CS11 while organ-rigid held it at 92; the CNS gained +4-8
    either way."""
    if abs(k - 1.0) < 1e-3:
        return Q
    c = Q.mean(0)
    S = (Q - c) * np.array([k, k ** -0.5, k ** -0.5]) + c
    if fate is not None:
        for ids in _rigid_family_ids():
            m = np.isin(fate, ids)
            if m.sum() < 8:
                continue
            S[m] = Q[m] - Q[m].mean(0) + S[m].mean(0)
    return S


_NEURULATE_IDS = None    # cached CNS ids for the neurulation tube contraction


_MESO_KIDS = None


def mesonephric_kidney(Q, fate, t):
    """THE KIDNEY'S EMBRYONIC LADDER (cycle 35, 2026-09-04). The reference kidney at CS16-17 is the
    MESONEPHROS -- a long thin paravertebral ridge (aspect 1:0.28:0.19 at CS16), a developmental
    predecessor the model never grew (its kidney was a fat blob-pair, 1:0.66:0.52, trace 73); by
    CS18 the reference compacts toward the metanephros (1:0.84:0.25) while the model was then TOO
    stringy. Two clock-gated windows, family-as-ONE-body volume-preserving aspect reshape (A/B:
    per-side loses to one-body at these stages -- the ridge complex reads as one mass):
      WT1/PAX2 nephrogenic-cord ridge (1:0.33:0.15) in the CS16 window (t 0.30-0.35), then
      GDNF-RET metanephric compaction (1:0.72:0.29) in the CS18 window (t 0.41-0.44).
    CS17 measured transitional (no reshape helps -- gates ~0 there); CS20+ already 92, untouched.
    A/B: CS16 73.1->82.5, CS18 83.3->87.8."""
    global _MESO_KIDS
    ga = max(0.0, 1.0 - abs(t - 0.325) / 0.035)             # CS16 window gate
    gb = max(0.0, 1.0 - abs(t - 0.425) / 0.030)             # CS18 window gate
    if ga < 0.05 and gb < 0.05:
        return Q
    if _MESO_KIDS is None:
        from medic.subhead_program import expand_names as _exn4
        _MESO_KIDS = [FIDX[n] for n in _exn4(["Kidney", "Nephron"]) if n in FIDX]
    m = np.isin(fate, _MESO_KIDS)
    if m.sum() < 30:
        return Q
    tgt = np.array((1.0, 0.33, 0.15)) if ga >= gb else np.array((1.0, 0.72, 0.29))
    g = max(ga, gb)
    Q = Q.copy()
    c = Q[m].mean(0)
    A = Q[m] - c
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    loc = A @ Vt.T
    sds = loc.std(0) + 1e-12
    sc = (tgt * sds[0]) / sds
    sc /= sc.prod() ** (1 / 3)                              # volume-preserving
    sc = 1.0 + g * (sc - 1.0)
    Q[np.where(m)[0]] = c + (loc * sc) @ Vt
    return Q


def _neurulate_c(t):
    """Neurulation tube-contraction schedule (cycle 27, 2026-09-04). The CS10 reference CNS is a thin
    closed neural TUBE (dense ring cross-section around the canal + flared cranial folds) where the
    model CNS is a fat solid slab -- the neural plate never narrowed. SHROOM3 apical constriction +
    PCP-driven fold fusion contract the plate to the tube in the CS09-11 window; the contraction
    RELEASES by CS13 as the brain vesicles expand (A/B: a persistent tube costs CS15 85->80).
    Measured optima: CS10 c~0.45-0.5 (brain 74->79), CS11 c~0.65 (93.6->96.0), CS12+ neutral->hurts.
    Returns the radial contraction factor (1.0 = no-op)."""
    pts = [(0.000, 1.00), (0.105, 1.00), (0.125, 0.45), (0.150, 0.48), (0.175, 0.62),
           (0.200, 0.75), (0.250, 1.00), (1.000, 1.00)]
    xs, ys = zip(*pts)
    return float(np.interp(t, xs, ys))


def neurulate(Q, c, fate, nb=24):
    """Contract CNS cells radially (DV+ML) toward their per-AP-bin centroid by factor c -- the
    neural plate folding into the tube. Per-bin so the tube follows the body axis."""
    global _NEURULATE_IDS
    if c >= 0.999 or fate is None:
        return Q
    if _NEURULATE_IDS is None:
        from medic.subhead_program import expand_names as _exn3
        _NEURULATE_IDS = [FIDX[n] for n in _exn3(["Forebrain", "Telencephalon", "Midbrain",
                          "Hindbrain", "Cerebellum", "OlfactoryBulb", "Spinal Cord",
                          "Nervous System", "DRG"]) if n in FIDX]
    m = np.isin(fate, _NEURULATE_IDS)
    if m.sum() < 40:
        return Q
    Q = Q.copy()
    x = Q[m, 0]
    edges = np.linspace(x.min() - 1e-9, x.max() + 1e-9, nb + 1)
    idxs = np.where(m)[0]
    for j in range(nb):
        bm = (x >= edges[j]) & (x < edges[j + 1])
        if bm.sum() < 5:
            continue
        ii = idxs[bm]
        cy, cz = Q[ii, 1].mean(), Q[ii, 2].mean()
        Q[ii, 1] = cy + (Q[ii, 1] - cy) * c
        Q[ii, 2] = cz + (Q[ii, 2] - cz) * c
    return Q


def fetal_curl(Q, curl, fate=None, sign=None):
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
        # ORGAN-RIGID TRANSPORT (cycle 21c): the per-AP-bin bend SHEARS any compact organ that spans
        # bins -- but the body curls AROUND its organs (families move as families, in the posture
        # layer). Each compact family rides the bend as one rigid body: its centroid moves to the bent
        # axis position, its cells keep their internal shape, rotated into the local (tangent, normal)
        # frame. Spanning structures (CNS, gut, skin, somites) keep the per-cell bend -- they DO curl.
        if fate is not None:
            for ids in _rigid_family_ids():
                m = np.isin(fate, ids)
                if m.sum() < 8:
                    continue
                xc, yc = float(Q[m, 0].mean()), float(Q[m, 1].mean())
                sxc = float(np.clip((xc - ctr[0]) / L, 0, 1)) * (nb - 1)
                j0 = int(sxc); j1 = min(j0 + 1, nb - 1); tc = sxc - j0
                Tc = T[j0] * (1 - tc) + T[j1] * tc; Tc /= (np.linalg.norm(Tc) + 1e-9)
                Nc0 = np.array([-Tc[1], Tc[0]])
                cbent = (C[j0] * (1 - tc) + C[j1] * tc
                         + (yc - float(np.interp(xc, ctr, axis))) * Nc0)
                out[m, :2] = cbent + (Q[m, :2] - (xc, yc)) @ np.stack([Tc, Nc0])
        return out

    if sign is not None:
        # EXPLICIT sign (cycle 22, the final word on the flip saga): every per-frame detection --
        # spine-reach (flickered) and then spine-side (unstable once proc's own flex() pre-bends the
        # body) -- proved frame-dependent. The sign is NOT a per-frame question: in the sim frame
        # dorsal = +y BY CONSTRUCTION of the fate map (neural fates commit at d > 0.46), so the belly
        # is -y and the curl is concave toward -y, always. The caller passes it.
        return _bend(float(sign))
    if fate is not None:
        # MESH-PHASE auto-sign: the ORIGINAL spine-reach rule, restored (cycle 23 -- the 21d spine-side
        # rewrite broke the historically-correct mesh/infant bend: Miles saw the infant arch BACKWARDS).
        # The cloud path never reaches here (explicit sign=-1.0); this serves the registered mesh frames
        # where the reach comparison has always picked the spine-outside bend correctly.
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


ARM_REACH_FRAC = 0.44      # anatomical upper-limb length (acromion->fingertip) as a fraction of stature H
                           # (Drillis: upper arm 0.186 + forearm 0.146 + hand 0.108). Caps the over-extended arm.
ARM_RADIUS_FRAC = 0.052    # upper-arm RADIUS as a fraction of H (a limb, not a blade); tapers to the wrist.


def _arm_tube(Q, m, r0, taper=0.45):
    """Reshape one arm's cells into a SOLID TAPERING CYLINDER about its own principal (shoulder->hand) axis, so
    the razor-thin lateral sheet (DV ~0.02 H) becomes a rounded limb the flesh can drape over. Keeps each cell's
    along-arm position; sets its cross-section to fill a disk of radius r0 (shoulder) tapering to r0*(1-taper)
    (wrist) via a deterministic golden-angle fill (no RNG). Read-only outside the arm."""
    idx = np.where(m)[0]
    pts = Q[idx]; c0 = pts.mean(0); X = pts - c0
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
    except Exception:
        return
    axis, e1, e2 = vt[0], vt[1], vt[2]
    along = X @ axis
    a0, a1 = float(along.min()), float(along.max())
    sf = (along - a0) / (a1 - a0 + 1e-9)                       # 0..1 along the axis
    med = np.median(along)                                     # orient sf=0 at the SHOULDER (higher AP x = up)
    if pts[along >= med, 0].mean() < pts[along < med, 0].mean():
        sf = 1.0 - sf
    rr = r0 * (1.0 - taper * sf)                               # taper shoulder -> wrist
    n = len(idx)
    theta = 2 * np.pi * ((np.arange(n) * 0.6180339887) % 1.0)  # golden-angle sunflower fill of the disk
    rad = rr * np.sqrt(((np.arange(n) * 0.7548776662 + 0.5) % 1.0))
    Q[idx] = c0 + along[:, None] * axis + (rad * np.cos(theta))[:, None] * e1 + (rad * np.sin(theta))[:, None] * e2


def mature_for_display(P, fate, adult_len=3.2):
    """Apply the movie's ADULT maturation (allometry + DV/ML anthropometric envelopes + head rounding + limb
    pose/length-cap) to a labelled point cloud, so the tabs (Gray's / NCA+LLM) show the SAME proportioned
    Vitruvian body the movie renders -- not the raw un-matured build_base cloud (the 'insect'). `fate` = per-cell
    fate index (Limb Bud cells get grow_limbs; head fates get the head scaling)."""
    P = np.asarray(P, float)
    headf = _head_ids()
    hm = np.isin(fate, headf)
    if hm.any() and P[hm, 0].mean() < np.median(P[:, 0]):        # orient head to +x (viewer convention)
        P = P.copy(); P[:, 0] = -P[:, 0]
    Q = mature_cloud(P, fate, 1.0, MATURE_SEARCHED)
    lb = FIDX.get("Limb Bud")
    if lb is not None:
        Q = grow_limbs(Q, fate == lb, _limb_grow_model(1.0, MATURE_SEARCHED["limb_ext"]),
                       _limb_grow_model(1.0, MATURE_SEARCHED.get("leg_ext", MATURE_SEARCHED["limb_ext"])),
                       pose=1.0, fate=fate)
    return Q * (adult_len / _long_axis_len(Q))


def grow_limbs(Q, limb, grow, leg_grow=None, pose=None, fate=None):
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
    # POSE (2026-08-08, Miles chose arms-DOWN): the Vitruvian arms-OUT pose CANNOT be skinned without webbing a
    # bat-wing membrane between the horizontal arm and the trunk (the "weirdo"). So the arms now swing DOWN to hang
    # at the sides by the maturation stage (pose_a follows `pose`): 0 = splayed bud (early), 1 = hanging (adult).
    # This removes the wings entirely. Legs still stand.
    # A-POSE cap (2026-08-09): fully-down (pose_a=1) tucks the arms against the torso and the skin mesh ABSORBS
    # them -> "no arms". Arms-out (0) webs bat-wings. pose_a~0.55 = an A-pose (arms down-and-out with a gap):
    # distinct visible arms + minimal webbing (tested per-pose on the adult frame).
    # A-pose cap 0.55 -> 0.88 (2026-08-31, cycle 16d): 0.55 = a 49.5-degree SPLAY that threw the
    # hands to |ML| ~0.40 at hip height -- "arms akimbo, hands floating at the hips" (Miles). The
    # 0.55 cap predates the field densification; a near-vertical arm now skins fine (own column
    # mass), and the canonical width profile (0.164 at the waist INCLUDING arms) assumes it.
    # 0.88 -> ~79 degrees = ~11-degree abduction: hands land beside the thighs (~0.19 ML).
    pose_a = 0.78 * float(np.clip(pose, 0.0, 1.0)) if pose is not None else float(np.clip(grow - 1.0, 0.0, 1.0))
    pose_l = float(np.clip(lg - 1.0, 0.0, 1.0)) if pose is None else float(np.clip(pose, 0.0, 1.0))
    if abs(grow - 1.0) < 1e-3 and abs(lg - 1.0) < 1e-3 and pose_a < 1e-3 and pose_l < 1e-3:
        return Q
    Q = Q.copy()
    x, z = Q[:, 0], Q[:, 2]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    Hh = np.ptp(x) + 1e-9
    w = np.percentile(np.abs(z[~limb]), 70) if (~limb).any() else 0.12 * (np.ptp(z) + 1e-9)
    # ARM = the limb-bud arm PLUS any cell splayed laterally beyond the shoulder girdle at shoulder height --
    # the arm's muscle/skin/connective COVERING is NOT limb-fate, so if we swing only F==LIMB it stays splayed
    # (the wide shoulder spike). lat_thr sits past the shoulder breadth (~0.12 H half) so the girdle is kept.
    lat_thr = max(0.16 * Hh, 1.8 * w)
    arm = (apf >= 0.5) & (limb | (np.abs(z) > lat_thr))
    # VISCERA NEVER RIDE THE ARM CAPTURE (2026-09-04, Miles: "that spleen is really stubborn at 50").
    # The lateral capture is deliberately fate-blind so the arm's muscle/skin COVERING swings with the
    # bud -- but it also catches any visceral cell a given build happens to place past lat_thr, and
    # the Vitruvian reach then flings it to 0.44 H: this build's f51-52 spleen streak (trace 83->42,
    # maxr 0.33 vs the 0.038 capsule; the chain audit pinned grow_limbs -- the family left
    # mature_cloud a perfect capsuled tongue). The same bistability WAS cycle 24's "f52 4x streak".
    # Compact organ families + the gut tube are excluded from the positional capture.
    if fate is not None:
        _visc = np.zeros(len(Q), bool)
        for _vids in _rigid_family_ids():
            _visc |= np.isin(fate, _vids)
        _gv = [FIDX[n] for n in _exn(("Gut", "Hindgut", "Foregut", "Duodenum")) if n in FIDX]
        _visc |= np.isin(fate, _gv)
        arm &= ~_visc
    leg = limb & (apf < 0.5)
    body = ~(arm | leg)
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
        Hh = np.ptp(x) + 1e-9
        arm_len = ARM_REACH_FRAC * Hh                            # anatomical arm length (acromion -> fingertip, ~0.44 H)
        for sgn in (1.0, -1.0):
            m = arm & (np.sign(z) == sgn)
            if m.sum() < 4:
                continue
            sh_x = np.percentile(x[m], 85)                       # shoulder = top (AP) of this arm
            sh_z = sgn * w                                       # shoulder sits at the trunk's side
            raw = np.abs(z[m] - sh_z)                            # each arm cell's lateral offset from the shoulder
            # TARGET the length: scale so the HAND (outer cells) reaches arm_len, regardless of the bud's own size
            # or limb_ext (the build_base arm bud is narrow, so a multiplier alone left the arm a stub) -> always ~0.44 H.
            g_eff = arm_len / (np.percentile(raw, 75) + 1e-9)   # top ~25% of arm cells reach full length (a solid arm, not a stub)
            reach = np.minimum(raw * g_eff, arm_len)             # along-arm reach, capped at the anatomical length
            out_z = sh_z + sgn * reach                           # arms-OUT: extended to the anatomical length
            # ROTATE the arm about the shoulder by pose_a*90deg (0 = lateral/out, 1 = straight down) -- this
            # PRESERVES the arm LENGTH. (The old linear interp between the out+down endpoints cut the corner and
            # shortened the arm to ~0.71x at the A-pose.) reach = per-cell along-arm distance from the shoulder.
            theta = pose_a * (np.pi / 2.0)
            Q[m, 0] = sh_x - reach * np.sin(theta)
            Q[m, 2] = sh_z + sgn * reach * np.cos(theta)
    if arm.any():                                            # TRIM the faint outermost arm tail (runs after EITHER
        for sgn in (1.0, -1.0):                              # branch above), clamping the lateral reach to the solid arm
            m = arm & (np.sign(Q[:, 2]) == sgn)
            if m.sum() < 4:
                continue
            sh_z = sgn * w
            off = np.abs(Q[m, 2] - sh_z)
            # only trim a genuine OUTLIER tail (97th pct): the arm extension already caps cleanly at the anatomical
            # length, so the old 88th-pct cap was chopping the real arm back to a stub (~0.25 H) -- keep the full arm.
            cap = 1.02 * np.percentile(off, 97)
            Q[m, 2] = sh_z + np.sign(Q[m, 2] - sh_z) * np.minimum(off, cap)
    # ARM GIRTH (2026-08-10): the positioned arm is a razor-thin lateral SHEET (DV ~0.02 H) -> a thin streak in
    # both the fate cloud and the skinned mesh (flesh adds no arm girth). Reshape each arm into a solid tapering
    # cylinder about its shoulder->hand axis so it is a rounded limb. Only once the arm is extended/posed (adult).
    if arm.any() and (pose_a > 1e-3 or grow > 1.0):
        Hh = np.ptp(Q[:, 0]) + 1e-9
        for sgn in (1.0, -1.0):
            m = arm & (np.sign(Q[:, 2]) == sgn)
            if m.sum() >= 12:
                _arm_tube(Q, m, r0=ARM_RADIUS_FRAC * Hh)
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
        # floor 0.055 -> 0.075 Hh (cycle 17): the measured line landed at 4.3% of FINAL stature --
        # canonical mid-thigh centres sit at ~5.5-6.5% -- so the columns hugged the midline and no
        # inter-thigh gap could beat the field smear no matter how clean the web. The floor now puts
        # the line at the canonical thigh centre; with the condensation wall at 0.30x the line, the
        # crotch gap is ~3%H = wide enough to surface.
        hipz = float(np.clip(hipz, 0.075 * Hh, 0.085 * Hh))     # FLOOR + cap: the two legs must stay a real hip-width
        #                                                         apart (~0.11-0.17 H bi-femoral), not collapse to a
        #                                                         central spike (the "gown"/cone that reads as no legs).
        for sgn in (1.0, -1.0):
            m = leg & (np.sign(z) == sgn)
            if m.sum() < 4:
                continue
            hipx = np.percentile(x[m], 88)                       # hip = top (AP) of this leg
            hz = sgn * hipz                                      # seat the leg at the hip width (not the trunk width)
            Q[m, 0] = hipx - (hipx - x[m]) * g                   # lengthen the leg downward on leg_ext
            # CONVERGE toward the ankle: the leg column drifts inward from hip to foot (feet closer than hips), so
            # the ankles/feet read narrow like a real standing figure instead of two parallel posts.
            lf = np.clip((hipx - Q[m, 0]) / (0.52 * Hh), 0.0, 1.0)           # 0 hip .. 1 foot
            hz_t = hz * (1.0 - 0.42 * lf * pose_l)
            # tighten each leg toward its (tapered) column centre, keeping real thigh thickness (0.5, not 0.85 which
            # collapsed the legs to thin lines meeting at the midline).
            Q[m, 2] = hz_t + (z[m] - hz) * (1.0 - 0.5 * pose_l)
            # LIMB CONDENSATION (cycle 17): the halved bud spread still left leg cells +-0.061 H about
            # their line (measured 85th-pct residual) -- the leg's OWN cells crossed the midline and
            # webbed the crotch shut up to ~25% of stature (the arm has _arm_tube for exactly this; the
            # leg had only the relative tighten). The limb is a mesenchymal condensation about its own
            # axis (SOX9): clamp the residual ML spread to 0.70x the column line -- the inner thigh
            # wall lands at 0.30x the line off the midline, a real inter-thigh gap, tapering with it.
            r_leg = 0.70 * np.abs(hz_t)
            dz_l = Q[m, 2] - hz_t
            Q[m, 2] = hz_t + dz_l + pose_l * (np.clip(dz_l, -r_leg, r_leg) - dz_l)
        # THE LEG TUBE (polish cycle; the _arm_tube law, at last, for the legs): the leg's cell
        # core was a twisted RIBBON -- per-side half-widths swinging 0.002-0.014 H with ML and DV
        # inverting by height -- so the fat wrapped a blade and the shins read paper-thin from
        # the side. The limb is a condensation about its own axis (SOX9): each leg redistributes
        # into a solid tapering cylinder about its hip->ankle line, thigh ~0.040 H to the ankle.
        if pose_l > 0.5:
            HhF = np.ptp(Q[:, 0]) + 1e-9
            for sgn in (1.0, -1.0):
                m = leg & (np.sign(Q[:, 2]) == sgn)
                if m.sum() >= 12:
                    _arm_tube(Q, m, r0=0.040 * HhF, taper=0.60)
        # THE THIGHS CLAIM THEIR TERRITORY (cycle 17): limb muscle is somitic myoblasts MIGRATING INTO
        # the limb bud (PAX3/LBX1/c-Met, HGF-guided streams), and the femoral vessels and dermis sprout
        # with the limb -- but the model's spanning leg-type fates stayed where the trunk envelope left
        # them: the inter-thigh midline (census: 1,676 cells at 26-46% of stature within |ML|<3%H --
        # Muscle 612, Vessel 257, Mesothelium 176, Skin 161...), so the crotch could not open above ~25%
        # and the legs read fused/short even though the leg fate itself reaches 0.43-0.50. Below the
        # groin, each cell of a migratory leg-type fate that is OUTSIDE both thigh columns joins its own
        # side's column at its height. The gonads (perineal by anatomy) and the groin band just under
        # the pelvic floor stay midline; the ankle/foot zone is left to the autopods.
        if fate is not None and pose_l > 1e-3:
            from medic.subhead_program import expand_names as _exnG
            _mig = [FIDX[n] for n in _exnG(("Muscle", "Skin", "Adipose", "Vessel", "Blood",
                                            "Connective", "Mesothelium", "Cavity")) if n in FIDX]
            xg, zg = Q[:, 0], Q[:, 2]
            groin = hipx0 - 0.02 * Hh
            cand = (np.isin(fate, _mig) & ~limb & (xg < groin)
                    & (xg > xg.min() + 0.10 * Hh))
            if cand.any():
                # The column line and its half-width are MEASURED off the placed leg cells (self-
                # reading): line(lf) = the tightening target hz_t; half-width = the 85th-pct residual
                # of the leg cells about their own line. (v1 guessed r=0.035H -- the guessed band
                # covered the inter-thigh midline, 2/3 of the web tested "already in the column",
                # and the spread floor even sent skirt cells INTO the midline. Measure, don't guess.)
                _lfL = np.clip((hipx0 - xg[leg]) / (0.52 * Hh), 0.0, 1.0)
                _lineL = hipz * (1.0 - 0.42 * _lfL * pose_l)
                _res = np.abs(np.abs(zg[leg]) - _lineL)
                r_th = float(np.clip(np.percentile(_res, 85), 0.010 * Hh, 0.035 * Hh))
                lf_c = np.clip((hipx0 - xg[cand]) / (0.52 * Hh), 0.0, 1.0)
                zc_mag = hipz * (1.0 - 0.42 * lf_c * pose_l)        # each side's leg line at this height
                r_c = np.minimum(r_th, 0.70 * zc_mag)               # column radius, never past the condensation wall
                inner_c = zc_mag - 1.05 * r_c                        # the compacted column's inner wall (>= 0.27 zc)
                outer_c = zc_mag + 1.05 * r_c
                zmag = np.abs(zg[cand])
                web = (zmag < inner_c) | (zmag > outer_c)           # inter-thigh web, or the skirt
                if web.any():
                    ci = np.where(cand)[0][web]
                    sgn_w = np.where(zg[ci] >= 0, 1.0, -1.0)
                    u_w = (ci % 89) / 89.0                           # deterministic spread across the column
                    tgt_mag = zc_mag[web] * (0.55 + 0.90 * u_w)     # land inside [0.55, 1.45] x the line
                    Q[ci, 2] = zg[ci] + pose_l * (sgn_w * tgt_mag - zg[ci])
    return Q


def _ventral_sign(Q, fate):
    """The body's own anterior (ventral) DV sign -- delegates to the SHARED robust rule (heart-vs-cord
    anchor, eye fallback) in flesh_surface_head, so the feet and the skin agree on which way is forward."""
    from medic.flesh_surface_head import ventral_sign
    return ventral_sign(Q, fate)


def feet_mesh(Q, fate, frac=1.0):
    """Genome-plausible pentadactyl autopod at each leg's distal tip (the Hox13 autopod territory) --
    the ONE pair of feet on the body (flesh_skin builds its skin shell with feet=False). Toes point
    VENTRAL by the body's own eye-derived anterior sign, and the big toe is MEDIAL on BOTH feet via the
    chirality flip in hand_foot_skin (the fan's spread axis is cross(out, up), so the side needing the
    mirror depends on the ventral sign). Digit COUNT is the pentadactyl default of the limb head (set by
    the Turing / lateral-inhibition wavelength elsewhere in the framework); full per-digit morphogenesis
    is future work -- this is a schematic autopod that does not violate the genome, NOT a MakeHuman graft.
    The thin, stumpy realisation is the kinematics-without-physics limitation (no soft-tissue settling).
    Laid frame (x=AP head+, y=DV, z=ML). Returns (verts, faces) with FIXED counts (2 * HFS.NV) so
    the movie can emit the skin faces once. `frac` in [0,1] ramps the autopod out with maturation."""
    from medic import hand_foot_skin as HFS
    x, z = Q[:, 0], Q[:, 2]
    H = float(np.ptp(x)) + 1e-9
    apf = (x - x.min()) / H
    leg = (np.asarray(fate) == LIMB) & (apf < 0.5)
    span = 0.055 * H
    # MEASURED foot (cycle 61, the gods-panel 48.3 row): canon foot length 0.152 of stature
    # heel-to-toe (252/1656mm) with the ankle at the anatomical quarter-point -- the old shell built
    # 0.14H FORWARD FROM THE ANKLE with no heel, and the landed cells read 0.073. The Hox13 autopod
    # territory takes its measured extent: base shifted back by the heel, length to the full measure.
    _ramp = float(np.clip(0.15 + 0.85 * frac, 0.15, 1.0))
    length = 0.152 * H * _ramp
    heel = 0.035 * H * _ramp
    dvsign = _ventral_sign(Q, fate)                          # toes FORWARD: anterior from the body's own eyes
    V, Fc, off = [], [], 0
    for sgn in (-1.0, 1.0):                                  # left (z<0), right (z>0)
        m = leg & (np.sign(z) == sgn)
        if m.sum() >= 6:
            legc = Q[m]
            # TRUE-BOTTOM ANCHOR (cycle 16): the distal leg column is sparse, so the old bottom-15%
            # mean landed at SHIN height (measured: feet at h 0.096-0.124, buried against the leg,
            # anchored dorsal of its axis). Anchor at the lowest 3% of the leg -- the real ankle.
            sel = legc[legc[:, 0] < np.percentile(legc[:, 0], 3)]
            if len(sel) < 4:
                sel = legc[np.argsort(legc[:, 0])[:6]]
            tip = np.array([sel[:, 0].mean(), np.median(sel[:, 1]), sel[:, 2].mean()])
        else:
            tip = np.array([x.min(), 0.0, sgn * 0.10 * H])   # fallback: bud tip on this side
        tip = tip - np.array([0.0, dvsign * heel, 0.0])      # the heel sits BEHIND the ankle
        v, f = HFS.build(tip, [0, dvsign, 0], [-1.0, 0, 0], span, length, "foot",
                         flip=(tip[2] * dvsign < 0))         # hallux medial on BOTH feet
        V.append(v); Fc.append(f + off); off += len(v)
    return np.vstack(V).astype(np.float32), np.vstack(Fc).astype(np.int32)


def hands_mesh(Q, fate, frac=1.0):
    """Pentadactyl autopod at each ARM's distal tip (cycle 16) -- the mirror of feet_mesh, using the
    same hand_foot_skin builder (kind='hand': thumb short, fingers long, wider fan). The arms hang
    down (A-pose), so the hand continues the arm: fingers point DOWN (-x), the dorsum faces LATERAL
    (palms toward the thigh -- the anatomical rest pose), and the thumb lands VENTRAL on both sides
    via the chirality flip. Same schematic-autopod honesty note as the feet: digit count is the limb
    head's pentadactyl default; per-digit morphogenesis is future work. Laid frame (x=AP head+,
    y=DV, z=ML). Returns (verts, faces); `frac` ramps the autopod out with maturation (Hox13-late)."""
    from medic import hand_foot_skin as HFS
    x, z = Q[:, 0], Q[:, 2]
    H = float(np.ptp(x)) + 1e-9
    apf = (x - x.min()) / H
    arm = (np.asarray(fate) == LIMB) & (apf >= 0.45)
    span = 0.045 * H
    length = 0.105 * H * float(np.clip(0.15 + 0.85 * frac, 0.15, 1.0))   # hand ~0.105 of stature
    dvsign = _ventral_sign(Q, fate)
    V, Fc, off = [], [], 0
    for sgn in (-1.0, 1.0):                                  # left (z<0), right (z>0)
        m = arm & (np.sign(z) == sgn)
        if m.sum() >= 6:
            armc = Q[m]
            tip = armc[armc[:, 0] < np.percentile(armc[:, 0], 12)].mean(0)   # the wrist (arm hangs down)
        else:
            continue
        # CLEAR OF THE THIGH (Miles: hands invisible): at 1cm voxels a hand AGAINST the thigh merges
        # into its density mass and the surface swallows it -- the feet read because the toes project
        # into open air. The relaxed hand hangs with a small gap beside the thigh: offset the anchor
        # laterally (+0.022 H) and slightly forward, so the hand and fingers make their OWN surface.
        tip = tip + np.array([0.0, dvsign * 0.010 * H, sgn * 0.015 * H])   # small clearance; near-vertical arm now
        v, f = HFS.build(tip, [-1.0, 0, 0], [0, 0, sgn], span, length, "hand",
                         flip=(sgn * dvsign < 0))            # thumb ventral on BOTH hands
        V.append(v); Fc.append(f + off); off += len(v)
    if not V:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.vstack(V).astype(np.float32), np.vstack(Fc).astype(np.int32)


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
    """Developmental curl schedule vs the global clock t. CYCLE 21b (2026-09-01, the CNS-slab look):
    real embryos curl EARLY -- the Carnegie plates are maximally C-curled from CS13 (the cephalic +
    caudal flexures close the C by t~0.26), and the old schedule only began curling at t=0.30, so the
    whole embryonic window scored a straight slab CNS against a curled reference tube (the thin CNS is
    the most curl-sensitive organ; the blob-like whole-body barely noticed). Rise CS10->CS13, full C
    held through the embryonic window + handoff + early fetus, then the unchanged unfurl to standing."""
    if t < 0.10:
        return 0.0                          # disc/early somite stages: still flat
    if t < 0.26:
        return (t - 0.10) / 0.16            # CS10 -> CS13: the embryonic C closes
    if t < 0.68:
        return 1.0                          # full C: embryonic window, end of cloud, handoff, early fetus
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
def _emit(Q, fate, vm, phase, stage, t, panel, ptype="genes", skin=None, skin_op=0.0, skin_f=None):
    # NO growth-window turn (2026-09-02 final): the mesh phase was ALREADY belly=+y like the
    # cloud (post continuity turn) and the anatomy reveal -- the -y reading that motivated a
    # turn here was a deep-curl instrument artifact (belly-gap and flexion stats both lie on
    # strongly curled frames; the straightening frames f72+ read correct). The real mesh
    # fault was only the CURL sign (now +1.0, bending toward the +y belly). Renders judge
    # curled frames; statistics only on straightened ones.
    # 3-DECIMAL EMIT (cycle 33, 2026-09-04, THE QUANTIZATION COURT): the old round(x, 2) crushed
    # small organs on small bodies -- at f51 the early-fetus body spans 0.93 units, the spleen's
    # sds1 is 0.0133, so the family occupied ~3 grid steps and its 140 display cells collapsed
    # onto coincident coordinates (median NN distance 0.0000; max exactly 0.0100 = the grid). The
    # scorer read the ROUNDING, not the anatomy: f51 spleen 62.9 actual vs 84.7 for a same-sds
    # gaussian control. The whole cloud phase (body 0.9 long) is in the same regime. ~12% JSON.
    d = dict(phase=phase, stage=stage, t=round(float(t), 3),
             xyz=[round(float(x), 3) for x in Q.ravel()],
             vm=[round(float(x)) for x in vm],
             fate=[int(x) for x in fate],
             panel=panel, ptype=ptype)
    if skin is not None:                                    # per-frame skin-surface mesh verts (faces in doc)
        d["skin"] = [round(float(x), 3) for x in np.asarray(skin).ravel()]
        d["skin_op"] = round(float(skin_op), 2)
        if skin_f is not None:                              # per-frame topology (the closed surface changes
            d["skin_f"] = np.asarray(skin_f, int).tolist()  # its mesh as the body grows)
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
    # (cycle 23e ATTEMPTED a belly-to-+DV rotation here and REVERTED it same-day: the standing mesh
    # frames measure viscera BELOW the CNS -- the mesh convention is ventral=-y, same as the sim frame,
    # and the rotation introduced the very mismatch it meant to fix. The "adult ventral=+DV" read came
    # from the heart-vs-skin-median statistic, which lies. Orientation verification lives in VIEWER
    # SCREENSHOTS now, not projections -- see _viewer_shots.py.)

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
        Q = ce_stretch(Q, _ce_elong(t), F)                   # CE axial elongation (cycle 26): the
        # somite-window body is long+thin (Wnt-PCP intercalation); organ-rigid, pre-curl.
        Q = neurulate(Q, _neurulate_c(t), F)                 # neural plate -> tube (cycle 27):
        # SHROOM3 apical constriction, CS09-11 window, released by CS13 (vesicles expand).
        Q = mesonephric_kidney(Q, F, t)                      # the kidney's embryonic ladder
        # (cycle 35): WT1/PAX2 mesonephric ridge at CS16, GDNF-RET compaction at CS18.
        Q = grow_limbs(Q, F == LIMB, _limb_grow(t))          # limb buds emerge small, not big paddles
        # THE DIGITAL RAYS + INTERZONES (cycle 82f): SOX9 condenses each paddle's distal cells into five
        # rays across the fan from CS17, the phalangeal anlagen condense (GDF5 interzones between them)
        # from CS18 -- the autopod completion ladder as a clock-gated head on the emitted embryo, with
        # mesonephric_kidney and neurulate above; foot_completion.run_full applies the SAME transform to
        # the full cloud (the movie path = the scored path). Each limb pair is patterned as one folded
        # paddle (the frames are symmetrised).
        from medic.digital_ray_head import apply_frame as _digital_rays_frame
        _digital_rays_frame(Q, F, prc2, LIMB)
        Q = fetal_curl(Q, _curl_amt(t), F, sign=-1.0); Q = Q - Q.mean(0)   # curl into the fetal C.
        # SIGN RECALIBRATED 2026-09-02 (Miles: "is our model curled with the heart inward or the
        # spine inward?"): the side-view court (_curl_vs_canon_court.png) against the POST-ordinal,
        # v3 skin-complete, organ-correspondence-ALIGNED canon showed +1.0 curling SPINE-INWARD --
        # opposite the real specimens (heart always inside the C). The old f26 flip to +1.0 was
        # calibrated against the pre-alignment canon and is superseded; -1.0 restores the
        # construction derivation (sim dorsal=+y => concavity toward the belly at -y) AND matches
        # the aligned canon. Mesh-phase calls stay -1.0 (one convention everywhere now).
        Q[:, 1:] *= -1.0
        # THE CONTINUITY TURN (2026-09-02, Miles: "the inward side of the model becomes the
        # back of the adult"): rigid 180-deg AP turn of the emitted cloud so its belly matches
        # the mesh phase's (+y) -- heart-inward is rotation-invariant so the recalibrated curl
        # survives; the embryo's inward side becomes the adult's FRONT. The embryonic canon
        # ladder turns in lockstep at export (see _cloud_canon_turn.py; canon_align runs after).
        stage = f"cloud · N={born:,} · {t_hpf:.0f} hpf · PRC2 {prc2:.2f}"
        out_frames.append(_emit(Q, F, V, "cloud", stage, t, panel_cloud(F, prc2, fi / (nfr - 1))))
    print(f"    {nfr} cloud frames")
    # THE HONEST CENSUS (cycle 19): per-cloud-frame family counts on the FULL simulate cloud -- the
    # frames' own fate arrays are the _rep_idx DISPLAY subsample (rare-fate lift alpha=0.32), which
    # biases composition instruments (the cycle-8 lying-instrument class). stage_composition reads this.
    _census = []
    for fi in range(nfr):
        _Ff = np.asarray(frames[fi][5])
        _cnt = np.bincount(_Ff[_Ff >= 0], minlength=len(FATES))   # fid -1 = uncommitted (no fate yet)
        _row = {FATES[k]: int(c) for k, c in enumerate(_cnt) if c}
        _row["_uncommitted"] = int((_Ff < 0).sum())
        _census.append(dict(frame=fi, born=int(frames[fi][0]), prc2=round(float(frames[fi][2]), 3),
                            counts=_row))
    json.dump(dict(note="full-cloud per-frame fate census (not the display subsample)",
                   census=_census), open("data/movie/cloud_census.json", "w"))
    print(f"    cloud census -> data/movie/cloud_census.json")

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
            Q = fetal_curl(Q, _curl_amt(t), reg_fate, sign=+1.0)       # fetal C toward the mesh belly (+y) -- RECALIBRATED 2026-09-02 (the old -1.0 bent the spine dorsally = Miles's wrong-way curl at the start of growth)
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
            Q = fetal_curl(Q, _curl_amt(t), reg_fate, sign=+1.0)       # fetal C toward the belly (+y), unfurling toward the adult -- RECALIBRATED 2026-09-02
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
        # ===== UNIFY the three views onto ONE proportioned cloud =====
        # Mature the SAME rich build_base cloud the anatomy reveal uses (girdles / organs / condensation + the
        # DV & ML anthropometric envelopes -> posture_silhouette ~85%), NOT the coarse simulate cloud, so the
        # adult SKIN and the REVEAL are one proportioned Vitruvian body instead of a blob + a splayed scatter.
        from medic.adult_persistence_audit import build_base as _build_base
        _Braw, _BFraw = _build_base(max(N_R, 30000))
        # GEOMETRY runs on the FULL cloud (2026-08-31): the _rep_idx display subsample lifts rare
        # fates (alpha=0.32), which over-draws the many small HEAD fates and under-draws the bulk
        # leg/trunk fates -- fine for the dot display it was designed for, but the closed-surface
        # skin turns that density distortion into SHAPE (giant head, thin legs, squat trunk), and
        # the percentile-based mechanisms (hip width, chest anchor) mis-measure the body. The full
        # cloud is the body; `sel` is only which cells get DRAWN as dots.
        B0 = _Braw.astype(float); BF = _BFraw
        _hmB = np.isin(BF, headf)                            # orient the head to +x (same convention as the cloud)
        if _hmB.any() and B0[_hmB, 0].mean() < np.median(B0[:, 0]):
            B0[:, 0] = -B0[:, 0]
        sel = _rep_idx(BF, N_R, rng)                         # DISPLAY subsample (rare organs stay visible)
        BFd = BF[sel]
        reg_fate = BFd                                       # the build_base cloud's real fates ARE the identity
        print(f"[C] model maturation: build_base cloud -> adult ({N_MESH} frames; unified with the reveal) ...")
        from medic.skin_shell_head import mesh as _skin_mesh   # the epidermal-boundary skin SURFACE
        from medic.flesh_surface_head import flesh_skin as _flesh_skin   # skin draped over MUSCLE+FAT, not bone
        from medic.fine_relief_head import body_relief as _body_relief    # the surface-muscle silhouette + six-pack
        skin_faces = None
        adult_len = 3.2
        labels = [(0.10, "early fetus"), (0.28, "fetus"), (0.46, "newborn"), (0.64, "infant"),
                  (0.80, "child"), (0.93, "adolescent"), (1.01, "adult")]
        limbmask = (BF == LIMB)
        for i in range(N_MESH):
            f = i / (N_MESH - 1)
            Q = mature_cloud(B0, BF, f, MATURE_SEARCHED)     # searched allometry on the FULL build_base cloud
            t = 0.55 + 0.45 * f
            Q = grow_limbs(Q, limbmask, _limb_grow_model(t, MATURE_SEARCHED["limb_ext"]),
                           _limb_grow_model(t, MATURE_SEARCHED.get("leg_ext", MATURE_SEARCHED["limb_ext"])),
                           pose=f, fate=BF)   # arms Vitruvian (out); legs stand; length capped
            tgt = 0.9 + (adult_len - 0.9) * f ** 1.2        # visible growth 0.9 -> 3.2 (convex)
            Q *= tgt / _long_axis_len(Q)
            Q = standing_register(Q, BF, f)                 # the addresses are fractions of STANDING height
            Q = tuck_limbs(Q, BF, _curl_amt(t))             # fetal tuck early, releasing as it unfurls
            Q = fetal_curl(Q, _curl_amt(t), BF, sign=+1.0)             # fetal C toward the belly (+y), unfurling to adult -- RECALIBRATED 2026-09-02
            Q = Q - _chest(Q, BF)                           # anchor the chest so it grows in place
            # the autopods get REAL cells (frame 91) -- LAST transform, so the shells it derives
            # and reuses in the skin field are anchored on the final frame geometry.
            Q, _hv, _hf, _fv, _ff = populate_autopods(Q, BF, frac=f)
            lab = next(l for thr, l in labels if f < thr)
            # CLOSED-SURFACE skin (2026-08-30, render-gated winner): marching cubes over the flesh density
            # field (body + bellies + fat + autopods in the field, so the skin wraps the toes) -- the slice
            # shell could not represent CONCAVITY (armpit/crotch/chin), giving the capes / skirt / cone.
            # Topology varies per frame -> emitted as fr.skin_f (mkSkin uses it over the global faces).
            from medic.skin_closed_surface import closed_surface as _closed_surf, flesh_cloud as _flesh_cloud
            # mid-frames at a lighter grid (the transitional surfaces peaked at 88k verts and blew the
            # frames JSON to ~190 MB = a 30-60 s black screen in the viewer); the adult keeps full res.
            # adult 170 -> 210 (cycle 17): the condensation crotch gap (~0.046 stature-units,
            # ~3.8 ML-voxels at 170) sits at the resolution limit -- the gaussian bridged it and
            # the legs fused above 24% no matter how clean the cells. At 210 the gap surfaces.
            # GRID SCALES WITH BODY LENGTH (cycle 74, Miles's pick): the fixed mid-frame grid 126
            # let the voxel grow 3.5x coarser as the body stretched to 3.2 -- cells/voxel dropped
            # and the iso surface cut inside the sparse periphery (the mid-movie head lag,
            # 18-30% of dots outside; the same-voxel adult was clean). The grid now holds the
            # ADULT's voxel size (3.2/210) at every frame, capped at the adult grid; costs ~40MB
            # of frames JSON (accepted over iso-widening's crotch-webbing risk).
            _grid = int(np.clip(round(tgt / (3.2 / 210.0)), 126, 210))
            if i >= N_MESH - 2:
                _grid = 210
            sv, sf = _closed_surf(_flesh_cloud(Q, BF, autopods=(_fv, _hv)),
                                  grid=_grid, sigma=1.35, iso_frac=0.38, smooth_iters=12)
            sv, _ = _body_relief(sv, Q, BF, amp=f)           # the surface-muscle RELIEF rides the new surface
            skin_op = 0.30 + 0.35 * f                        # skin firms up as the body matures
            out_frames.append(_emit(Q[sel], BFd, np.full(N_R, NEUT), "mesh",
                                   f"model · {lab} (allometric maturation of the cell cloud)", t,
                                   panel_mesh(f, morphing=(1 - f)), skin=sv, skin_op=skin_op, skin_f=sf))
            if i == N_MESH - 1:
                skin_faces = sf                              # the ADULT topology = the global (anatomy phase)
                skin_adult_v = sv
        Q_adult = Q[sel]                                    # the finished adult cloud (laid, centred; display cells)

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
        # PER-ORGAN RE-ANCHOR (frame-86, part 2 -- Miles caught the burst): the global affine
        # cannot reconcile the assembly's proportions with the movie body region by region; the
        # cephalic solids still stood 0.17-0.33 above the standing dome after the assembly was
        # standing-registered. Each solid translates so its centroid sits on ITS OWN fate-family
        # centroid in the revealed cloud (shape kept) -- every solid then pops in exactly where
        # the body's cells are, by construction.
        from medic.subhead_program import expand_names as _exnD
        for _os in organ_surf:
            _ids = [FIDX[n] for n in _exnD((_os["name"],)) if n in FIDX]
            _mD = np.isin(reg_fate, _ids)
            if _mD.sum() >= 8:
                _OV = np.asarray(_os["V"], float).reshape(-1, 3)
                _os["V"] = ((_OV - _OV.mean(0)) + Q_adult[_mD].mean(0)).astype(np.float32).ravel().tolist()
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
    # UNIFIED reveal: dissolve the skin to expose THIS proportioned cloud's OWN cells, coloured by their fate
    # (bone / muscle / organ / ...), NOT a separate scattered anatomy cloud -- so the reveal is the same Vitruvian
    # body with its skin off. The organ SOLID surfaces (aligned to Q_adult) stay overlaid.
    T, Tf = Q_adult, reg_fate
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
    try:
        from medic.viewer_tabs import augment as _augment_tabs
        _augment_tabs(HTML)                     # re-apply Gray's + NCA+LLM tabs + reveal toggle (survives re-render)
    except Exception as _e:
        print(f"  [viewer_tabs augment failed: {_e}]")
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
  button.on{background:#2b6cb0;border-color:#2b6cb0}
  #sliderwrap{flex:1;position:relative;padding-bottom:13px}
  #sliderwrap input[type=range]{width:100%;margin:0;display:block}
  #ruler{position:absolute;left:0;right:0;bottom:0;height:12px;font:10px system-ui;color:#8091a8;pointer-events:none}
  #ruler span{position:absolute;transform:translateX(-50%);font-variant-numeric:tabular-nums}
  #fnum{min-width:64px;text-align:right;color:#7dd3fc;font:13px system-ui;font-variant-numeric:tabular-nums}
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
  <div id="sliderwrap">
    <input id="slider" type="range" min="0" max="0" value="0" step="1">
    <div id="ruler"></div>
  </div>
  <span id="fnum">0 / 0</span>
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
function mkSkin(verts, op, ff){
  const n=verts.length/3, P=new Float32Array(n*3);
  for(let i=0;i<n;i++){ P[3*i]=verts[3*i+2]; P[3*i+1]=verts[3*i]; P[3*i+2]=verts[3*i+1]; }
  let idx;
  if(ff){ idx=[]; for(const f of ff){ idx.push(f[0],f[1],f[2]); } }       // per-frame topology (closed surface)
  else { if(!skinIdx){ skinIdx=[]; for(const f of DATA.skin_faces){ skinIdx.push(f[0],f[1],f[2]); } } idx=skinIdx; }
  const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.BufferAttribute(P,3));
  g.setIndex(idx); g.computeVertexNormals();
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
  if(fr.skin && (fr.skin_f||DATA.skin_faces) && (fr.skin_op||0)>0.01){ skinMesh=mkSkin(fr.skin, fr.skin_op, fr.skin_f); sc.add(skinMesh); }
  clearOrgans();   // solid organ surfaces during the anatomy reveal (toggle with the 'o' key)
  if(showOrgans && fr.phase==='anatomy' && DATA.organ_surfaces){
    for(const os of DATA.organ_surfaces){ const o=mkOrgan(os.V, os.F, os.color, 0.85); organMeshes.push(o); sc.add(o); }
  }
  document.getElementById('stage').textContent='frame '+i+' — '+fr.stage;
  document.getElementById('slider').value=i;
  document.getElementById('fnum').textContent=i+' / '+(nf-1);
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
  // numbered play bar (Miles): ticks every 10 frames so a strange frame can be named exactly
  { const rl=document.getElementById('ruler'); rl.innerHTML='';
    const step=nf>220?20:10;
    for(let k=0;k<nf;k+=step){ const s=document.createElement('span');
      s.style.left=(100*k/(nf-1))+'%'; s.textContent=k; rl.appendChild(s); }
    const e=document.createElement('span'); e.style.left='100%'; e.textContent=nf-1; rl.appendChild(e); }
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
