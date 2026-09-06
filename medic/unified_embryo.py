"""
The unified forward embryo: all four heads intercalated, one body, one clock, one movie.
=========================================================================================

A synthetic grow-from-one-cell embryo -- the anatomical compiler run forward -- in which every
timestep applies all four genome-emitted heads together on one substrate:

  1. CLOCK           cumulative divisions shorten the telomere -> PRC2 withdraws (division_head).
  2. DIVISION        the population grows toward the real generation count, roughly UNIFORM in
                     space with undifferentiated progenitors dividing a little more, declining as
                     the clock runs down (data-grounded vs MOSTA; no positional growth-zone bias).
  3. DIFFERENTIATION each cell's fate is set by body position AND gated by the clock; a cell
                     COMMITS the first time the clock unlocks its positional fate and keeps it.
  4. MIGRATION       convergent extension -- axial cells intercalate to the midline -> the body
                     narrows mediolaterally and elongates antero-posteriorly.
  5. SHAPE / COHESION  two mechanical coupling systems, the way real tissue has two:
       * CADHERINS (adherens junctions): SELECTIVE, same-fate cohesion, strength = the cadherin
         adhesion program (epithelial/neural high) -> tissues SORT into clean compartments.
       * INTEGRINS + ECM (the FASCIA): NON-SELECTIVE, longer-range cohesion binding ALL
         neighbours regardless of fate, strength = a mesenchymal ECM program (fibroblasts secrete
         the matrix, so it is high in mesoderm/crest, the complement of the cadherins) -> the
         sorted tissues stay bound into ONE mechanical continuum instead of fragmenting.
     plus the dorsal neural plate folds to the midline as apical constriction rises.

Cadherins vs connexins vs integrins: cadherins are cell--cell MECHANICAL adhesion (sorting);
integrins+ECM are cell--matrix / organ--organ MECHANICAL cohesion (the fascial continuum);
connexins are the ELECTRICAL coupling (the V_m operator). The adhesion carves the compartments
FIRST; the electrical coupling then follows within them (the field's gap-junction smoothing runs
over the sorted, same-fate neighbours) -- which is why the raw connexin transcript did not mark
the boundaries (the earlier operator null): the boundary is carved by adhesion, not conductance.

The fascia is demonstrated by its connectivity effect: WITH the ECM the body stays one connected
component with high cross-tissue binding; WITHOUT it, cadherin sorting alone fragments it.

LIMBS & THE AMPHIBIAN STEP (limb_buds=True). The four limb buds are NOT hand-placed. They are put
on the ANTINODES of the embryo's own ELECTRIC-BODY frame -- the low eigenmodes of the gap-junction
operator (the same frame that places the electric face, the mammary line and the six-pack, Paper #4):
the AP eigenmode gives the head->tail coordinate on which Hox sets the fore/hind levels, and the
LEFT-RIGHT eigenmode (its NODE is the midline) supplies the two bilateral sides; the buds sit where
an AP level meets an LR antinode. This ONLY works once the body has real medio-lateral WIDTH: in a
thin, convergent-extension-collapsed body the left-right mode is ABSENT (a limbless, fish-like body);
broadening the body drops that left-right mode into the accessible spectrum so bilateral limbs can be
placed. That geometric threshold -- width -> an LR eigenmode -> bilateral limbs -- IS the fish->tetrapod
transition, whose first members are the AMPHIBIANS. The width is a GENOME knob: the convergent-extension
strength `pcp` stands for the planar-cell-polarity / non-canonical Wnt pathway (Vangl2, Wnt5a/Wnt11), and
the per-step ML narrowing is `z *= 1 - 0.10*pcp` -- STRONG pcp narrows the body to a limbless fish, WEAK
pcp keeps it wide for a limbed tetrapod (the k=14 eigen-search reaches the LR mode in an elongated body).

HONEST SCOPE: synthetic, schematic geometry and reduced mechanics (the frontier-demo class), not
the real atlas. The rendered cells are a subsample of the true count the clock tracks (~1k->~57k).

Writes data/movie/zebrafish_unified_frames.json.  Run: python -m medic.unified_embryo
"""
from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from medic.differentiation_clock import FATE_PRC2
from medic.division_head import prc2_div, generations
from medic.zesta_temporal_4d import LAYER_VM
from medic.organ_sprouting import sprout_organs, bind_fates
from medic.body_electric_antinodes import body_electric_antinodes

OUT = Path("data/movie/zebrafish_unified_frames.json")
VMIN, VMAX = -70.0, -25.0
V_NEUTRAL = -50.0
N_START, N_END = 60, 9000

# Genome-anchored tissue-specific proliferation RATE (per-fate division weight), read from the AlphaGenome
# cell-cycle regulon by medic.proliferation_genome. The division head weights each cell's chance of being a
# dividing parent by its fate's rate; the differentiation clock sets how long each fate stays proliferative,
# so the simulation INTEGRATES rate x window into the final organ cell number. Falls back to uniform (all
# 1.0) if the table is absent, so the model still runs genome-table-free.
try:
    _PROLIF_W = json.load(open("data/organ_cascade/proliferation_weights.json"))["fate_weight"]
except Exception:
    _PROLIF_W = {}

# ---- LIMB PROGRAM KNOBS (searchable; each maps to a real limb-development gene) ------------------
# The limb program was hardcoded; these are the tweakable coefficients a von Dassow-Odell state-space
# search moves until the trajectory hits the stage targets. Defaults reproduce the prior behaviour.
#   allocation (WHERE + HOW MANY cells become bud) -- the limb FIELD:
#     dv_lo/dv_hi   lateral-plate DV competence band            (Wnt2b / lateral-plate mesoderm extent)
#     mll_thr       how far out the lateral edge must be         (Tbx5/Tbx4 field lateral limit)
#     hox_w         AP width of each fore/hind competence band   (Hox colinearity sharpness)
#     inhib, inhib_k  lateral-inhibition strength + reach        (BMP/Noggin Turing spacing -> 4 buds)
#     peak_thr      fraction of the sharpened peak kept as bud    (limb-field recruitment; LOWER = bigger buds)
#   proliferation (the AER makes the limb GROW by cell division -- the missing mechanism):
#     prolif        limb-bud division multiplier                 (AER FGF8/FGF10 proliferation)
#     inherit       limb-progenitor divisions stay limb (0/1)    (AER keeps the progenitor pool limb-fated)
#   outgrowth (the proximodistal SHOVE per step):
#     out_lat, out_ven   lateral projection + ventral drop       (Shh/Fgf proximodistal outgrowth)
#     grow_base, grow_slope  proximodistal gradient              (progress-zone distal bias)
LIMB_DEFAULTS = dict(dv_lo=0.26, dv_hi=0.62, mll_thr=0.55, hox_w=0.055,
                     inhib=1.25, inhib_k=45, peak_thr=0.35, gate_aspect=0.20,
                     prolif=1.0, inherit=0, fore_boost=1.9,
                     out_lat=0.120, out_ven=0.075, grow_base=0.30, grow_slope=0.70)
#   fore_boost: Tbx5 FORELIMB proliferation boost. The anterior (fore) bud is under-grown vs the
#     posterior (hind, Tbx4/Pitx1) pair because the anterior lateral plate is thinner; the forelimb
#     normally develops AHEAD, so boost its AER division to balance the two (so it resolves all 3 PD bones).
#   gate_aspect: the fish->tetrapod WIDTH threshold -- limbs form only if the body's ML/AP aspect
#     exceeds this (a wide tetrapod passes, a narrow fish does not). Raising it makes limb formation
#     more selective for a broad body (the faithfulness knob: keep the fish limbless).

# EYE FRONTAL-MIGRATION coupling: the eyes are born lateral on the electric-body antinodes, then a
# frontal-organizer morphogen gradient (Shh/Fgf8 at the frontonasal midline) makes the eye primordium
# chemotax frontally as the head grows -- the medialization that gives primates/humans binocular overlap
# (medic.field_driven_eye, trained to 85deg lateral -> 35deg frontal). The trained (motility, length-scale)
# are loaded here; the eye_frontation knob in [0,1] scales how frontal the eyes end up (0 = stays lateral,
# fish/mouse; 1 = full frontal, human/stereoscopic).
try:
    _EYEJ = json.load(open("data/organ_cascade/field_driven_eye.json"))
    EYE_MOT, EYE_LAM = float(_EYEJ["motility"]), float(_EYEJ["morphogen_lambda"])
except Exception:
    EYE_MOT, EYE_LAM = 0.30, 0.80
STEPS = 50
T0, T1 = 3.3, 26.0

# cadherin adhesion per fate (SELECTIVE, same-fate): epithelial/neural high -> sorts tissues.
ADH = {"Forebrain": 1.00, "Telencephalon": 1.00, "Eye": 0.90, "Nervous System": 0.94, "Spinal Cord": 0.92,
       "Neural Crest": 0.82, "Mesoderm": 0.42, "Somite": 0.45, "Epidermal": 0.88,
       "Hypoblast": 0.50, "Yolk Syncytial Layer": 0.58, "Blastodisc": 0.30,
       "Proliferative Like Cell": 0.30, "Limb Bud": 0.50, "Heart": 0.55, "Otic": 0.85,
       "Liver": 0.66, "Lung": 0.70, "Pancreas": 0.64, "Gut": 0.60, "Rib": 0.40,
       "Kidney": 0.70, "Muscle": 0.60, "Notochord": 0.72, "Skin": 0.86,
       "Cartilage": 0.55, "DRG": 0.78, "Sympathetic": 0.75, "Vessel": 0.66,
       # leaf heads added 2026-07-18 (MOSTA series): craniofacial crest, CNS envelope, filler mesenchyme
       "Meninges": 0.62, "Connective": 0.45, "Jaw": 0.60, "Choroid": 0.72, "Gonad": 0.60,
       # brain subheads (neural, high adhesion like Forebrain -> a tight neuroepithelial tube, not a spread cloud)
       "Midbrain": 0.95, "Hindbrain": 0.95, "Cerebellum": 0.95,
       # 6 remaining MOSTA heads added 2026-07-18: linings, craniofacial, haematopoietic
       "Mesothelium": 0.55, "Mesentery": 0.45, "Mucosa": 0.70, "HeadMes": 0.45,
       "Branchial": 0.60, "Blood": 0.20,
       # organ SUBHEADS (07-18) + new organ heads, grounded in the SEdb/AlphaGenome head registry
       "Atrium": 0.55, "Ventricle": 0.55, "Left Ventricle": 0.55, "Right Ventricle": 0.55,
       "Outflow": 0.55, "LiverHaem": 0.30, "Foregut": 0.60,
       "Hindgut": 0.60, "Nephron": 0.70, "Retina": 0.85, "Adrenal": 0.60, "Thymus": 0.60, "Spleen": 0.50,
       "Bladder": 0.65, "Adipose": 0.40, "OlfactoryBulb": 0.62, "Cavity": 0.05}
# integrin/ECM (fascia) per fate (NON-SELECTIVE, all neighbours): mesenchymal, so mesoderm/crest
# high, epithelia low -- the complement of the cadherins -> binds the body into one continuum.
ECM = {"Mesoderm": 1.00, "Somite": 0.90, "Neural Crest": 0.85, "Hypoblast": 0.60,
       "Yolk Syncytial Layer": 0.50, "Epidermal": 0.40, "Forebrain": 0.30, "Eye": 0.30,
       "Nervous System": 0.35, "Spinal Cord": 0.35, "Blastodisc": 0.45, "Telencephalon": 0.30,
       "Proliferative Like Cell": 0.45, "Limb Bud": 0.92, "Heart": 0.70, "Otic": 0.30,
       "Liver": 0.72, "Lung": 0.55, "Pancreas": 0.68, "Gut": 0.74, "Rib": 0.95,
       "Kidney": 0.55, "Muscle": 0.85, "Notochord": 0.60, "Skin": 0.42,
       "Cartilage": 0.90, "DRG": 0.55, "Sympathetic": 0.55, "Vessel": 0.60,
       "Meninges": 0.70, "Connective": 0.95, "Jaw": 0.80, "Choroid": 0.30, "Gonad": 0.60,
       "Midbrain": 0.34, "Hindbrain": 0.35, "Cerebellum": 0.33,
       "Mesothelium": 0.62, "Mesentery": 0.85, "Mucosa": 0.40, "HeadMes": 0.88,
       "Branchial": 0.82, "Blood": 0.30,
       "Atrium": 0.70, "Ventricle": 0.70, "Left Ventricle": 0.70, "Right Ventricle": 0.70,
       "Outflow": 0.70, "LiverHaem": 0.30, "Foregut": 0.70,
       "Hindgut": 0.70, "Nephron": 0.55, "Retina": 0.30, "Adrenal": 0.60, "Thymus": 0.50, "Spleen": 0.60,
       "Bladder": 0.50, "Adipose": 0.75, "OlfactoryBulb": 0.35, "Cavity": 0.02}
from medic.subhead_program import extend_value_maps as _sub_evm    # the sub-head PROGRAM (table-driven roster)
_sub_evm(ADH, ECM)                                                 # children inherit the parent's ADH/ECM
FATES = list(ADH.keys())
FIDX = {f: i for i, f in enumerate(FATES)}
bind_fates(FATES)                          # let organ_sprouting read a committed cell's germ context
# COMPACT point-organ primordia that CONDENSE into a coherent mass (mesenchymal condensation) rather
# than staying salt-and-pepper. NOT the spanning tubes (Gut/Notochord/Vessel) or segmented axial
# structures (Cartilage/Rib/Muscle/DRG), which are meant to be extended, not condensed to a point.
POINT_ORGANS = ("Eye", "Otic", "Heart", "Lung", "Liver", "Pancreas", "Kidney", "Spleen")
# Spleen joined 2026-09-01 (cycle 23): it had NO growth law (the gut disease) -- its size was whatever
# sprouting claimed, and the kidney-staged rebalance halved it (adult trace 84 -> 71). It condenses as
# a single left-flank mass; its staged target is the representability floor (growth_program.FLOOR).
# PAIRED point organs sit on the TWO LR antinodes (bilateral); they must condense as two separate
# side-primordia, not to one median (which is the midline and collapses the pair). Heart/Liver = midline.
PAIRED_ORGANS = ("Eye", "Otic", "Lung", "Pancreas", "Kidney")
PAIRED_ORGAN_IDS = np.array([FIDX[n] for n in PAIRED_ORGANS if n in FIDX])
# target size of each point organ as a fraction of the body (from the real E12.5 MOSTA fractions), so a
# primordium GROWS by recruiting nearby undifferentiated mesenchyme to a coherent size instead of a wisp.
ORG_TARGET = {"Liver": 0.028, "Heart": 0.085, "Kidney": 0.016, "Lung": 0.013,   # Heart -> ~HESTA 9.3%
              "Eye": 0.014, "Otic": 0.007, "Pancreas": 0.006}
# GUT TUBE TARGET (2026-08-31, Miles approved the allocation): HESTA says Primitive Gut = 5.33% of the
# embryo; the model captured 0.97% (the largest deficit in the composition table) because the gut is a
# spanning TUBE, deliberately excluded from the POINT_ORGANS condensation loop -- so it had NO growth
# law at all: point organs recruit to ORG_TARGET, the tube kept only what sprouting claimed (the 70-cell
# thread that capped the adult gut trace at ~60). The tube law below recruits generic cells NEAREST THE
# TUBE along its whole AP span (not toward a point), so the tube thickens instead of blobbing.
GUT_TARGET = 0.053                                  # HESTA Primitive Gut share (Gut+Mucosa+Foregut+Hindgut)
VM_OF = {**LAYER_VM, "Limb Bud": -45.0, "Heart": -32.0, "Otic": -58.0, "Liver": -45.0,
         "Lung": -48.0, "Pancreas": -43.0, "Gut": -46.0, "Rib": -62.0,
         "Kidney": -50.0, "Muscle": -80.0, "Notochord": -38.0, "Skin": -50.0,
         "Cartilage": -60.0, "DRG": -60.0, "Sympathetic": -58.0, "Vessel": -40.0,
         "Meninges": -55.0, "Connective": -55.0, "Jaw": -56.0, "Choroid": -44.0, "Gonad": -50.0,
         "Midbrain": -66.0, "Hindbrain": -66.0, "Cerebellum": -66.0, "Telencephalon": -64.0,
         "Mesothelium": -50.0, "Mesentery": -52.0, "Mucosa": -47.0, "HeadMes": -55.0,
         "Branchial": -56.0, "Blood": -25.0,
         "Atrium": -32.0, "Ventricle": -32.0, "Left Ventricle": -32.0, "Right Ventricle": -32.0,
         "Outflow": -34.0, "LiverHaem": -25.0, "Foregut": -46.0,
         "Hindgut": -46.0, "Nephron": -50.0, "Retina": -60.0, "Adrenal": -50.0, "Thymus": -55.0,
         "Spleen": -45.0, "Bladder": -48.0, "Adipose": -50.0, "OlfactoryBulb": -64.0,
         "Cavity": 0.0}  # bud Vm set-points; Cavity = fluid, ~0 mV (no membrane)
_sub_evm(VM_OF=VM_OF)                                              # sub-head children inherit the parent's Vm


# ---- FATE-MAP KNOBS (searchable; the joint HESTA differentiation search moves these) --------------
# The positional germ-layer/tissue thresholds of fate_of. They are COUPLED -- widening one pool steals
# from a neighbour (widening the neural tube starves the somite; widening the endoderm shifts the DV
# balance and collapses the somite) -- so hand-tuning one at a time backfires, and the joint search over
# ALL of them against the HESTA composition is the right tool (von Dassow-Odell). Defaults = current map.
FATE_DEFAULTS = dict(
    neural_d=0.46, neural_mln=0.44,     # neural TUBE: dorsal-midline competence (spinal cord + brain)
    brain_ap=0.25,                      # brain occupies the anterior a<brain_ap of the tube; rest = cord
    crest_d=0.56,                       # neural crest DV threshold
    endo_d=0.28,                        # ventral ENDODERM (the gut/lung/pancreas fuel)
    epid_mln=0.66,                      # epidermal (lateral surface ectoderm)
    somite_d=0.42, somite_lo=0.44, somite_hi=0.66,   # PARAXIAL somite band (flanks the neural tube)
)


def fate_of(a, d, mln, prc2, fp=None):
    """Positional fate map gated by the clock. a=AP[0..1] (0 anterior), d=DV[0..1] (1 dorsal), mln=|ML|.
    fp = the searchable FATE_DEFAULTS threshold knobs; the joint HESTA search moves them to balance the
    coupled tissue fractions (neural tube / somite / endoderm-gut / mesoderm)."""
    if fp is None:
        fp = FATE_DEFAULTS
    if d > fp["neural_d"] and mln < fp["neural_mln"]:
        # the neural TUBE = the dorsal-MIDLINE column, regionalised anterior->posterior into brain subheads
        # (Otx2/En1/Gbx2/Atoh1) over the anterior a<brain_ap, then Nervous System / Spinal Cord (Hox-ON).
        b = fp["brain_ap"]
        if a < 0.56 * b and 0.12 < mln < 0.38: f = "Eye"
        elif a < 0.30 * b: f = "Telencephalon"   # the forebrain's ANTERIOR antinode (Foxg1): cerebral
        #                                          hemispheres/cortex -- the sub-head the recursion cascade
        #                                          found (the forebrain still read as a chain, gap 3.2)
        elif a < 0.48 * b: f = "Forebrain"       # posterior forebrain (diencephalon: thalamus/hypothalamus)
        elif a < 0.64 * b: f = "Midbrain"
        elif a < 0.84 * b: f = "Hindbrain"
        elif a < b: f = "Cerebellum"
        elif a > 0.42: f = "Spinal Cord"
        else: f = "Nervous System"
    elif d > fp["crest_d"]: f = "Neural Crest"
    elif d < fp["endo_d"]: f = "Yolk Syncytial Layer" if a > 0.5 else "Hypoblast"
    elif mln > fp["epid_mln"]: f = "Epidermal"
    elif a > 0.16 and d > fp["somite_d"] and fp["somite_lo"] <= mln < fp["somite_hi"]:
        # PARAXIAL (somite) mesoderm FLANKING the neural tube (its ML window starts where the tube's ends,
        # so tube and somites don't compete for the midline) -- the source of vertebrae, ribs, axial muscle.
        f = "Somite"
    else: f = "Mesoderm"
    return f if FATE_PRC2.get(f, 0.0) > prc2 else None


def limb_bud_fate(a, d, mln, prc2, gate=0.42):
    """Four lateral-plate LIMB BUDS: paired (lateral, mln high) at fore (AP~0.30) and hind
    (AP~0.66) levels, mid-DV, unlocking LATE (after the clock has withdrawn PRC2 below `gate`)
    -- the SAME lateral-plate appendage field that later builds fins or limbs (deep homology)."""
    if prc2 > gate:
        return None
    if not (0.30 <= d <= 0.60) or mln < 0.42:
        return None
    if (0.22 <= a <= 0.38) or (0.58 <= a <= 0.74):
        return "Limb Bud"
    return None


def _norm(v):
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo + 1e-9)


def _flex(P, ang):
    """The FLEXURE HEAD: bend the antero-posterior axis into a circular arc of total angle `ang`
    (radians), pivoting at the head, so the body takes the cephalo-caudal C-curl of a real fetus.
    A DV (y) offset rides the LOCAL normal so cross-sections stay perpendicular to the curved axis;
    ML (z) is untouched, so bilateral symmetry and the fish<->tetrapod width are preserved. Applied
    LATE, after the straight-frame morphogenesis (eigenmodes, limbs, organ antinodes are all laid down
    on the straight body, exactly as the real embryo patterns straight then folds)."""
    if ang < 1e-6:
        return P
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    L = float(np.ptp(x)) + 1e-9
    b = ang / L
    s = x - x.min()
    th = b * s
    cx = np.sin(th) / b
    cy = (np.cos(th) - 1.0) / b
    nx, ny = -np.sin(th), np.cos(th)
    h = y - np.median(y)
    return np.stack([cx + h * nx, cy + h * ny, z], 1).astype(P.dtype)


def _apical_fold(P, fid, deg=140.0, nb=48):
    """APICAL CONSTRICTION FOLD -- the fetal curl EMERGES from a cell behaviour, not an imposed arc.
    Literature (2026-07 survey): apical constriction (Shroom3 -> ROCK/Rock1 -> Myosin II -> apical F-actin
    contraction) makes a cell WEDGE-shaped, which BENDS the epithelial sheet; the median hinge point wedges
    under the notochord's Shh. Here the VENTRAL-MIDLINE hinge cells (the notochord/floor-plate Shh zone)
    constrict, so the ventral surface SHORTENS and the antero-posterior axis bends VENTRALLY (ventral
    concave), with the curvature CONCENTRATED at the antero-posterior levels where the hinge is most active
    (the flexures) rather than a uniform arc. DV rides the local normal; ML (z) is untouched, so bilateral
    symmetry and the fish<->tetrapod width are preserved. `deg` = total curl angle; the SHAPE of the curl is
    set by the hinge-activity profile, the magnitude by `deg`."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    d = _norm(y); mln = np.abs(z) / (np.abs(z).max() + 1e-9)
    hinge = (d < 0.40) & (mln < 0.22)                          # ventral-midline hinge (notochord/floor plate)
    L = float(np.ptp(x)) + 1e-9
    edges = np.linspace(x.min(), x.max(), nb + 1); ctr = 0.5 * (edges[:-1] + edges[1:])
    act = np.zeros(nb)
    for i in range(nb):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() > 3:
            act[i] = float(hinge[m].mean())                   # apical-constriction activity per AP level
    for _ in range(3):
        act[1:-1] = 0.25 * act[:-2] + 0.5 * act[1:-1] + 0.25 * act[2:]
    if act.sum() < 1e-6 or deg < 1e-3:
        return P
    kap = act / (act.sum() + 1e-9)                             # curvature profile, sums to 1
    th_bin = np.cumsum(kap) * np.radians(deg)                  # cumulative bend angle, total = deg
    ds = L / nb
    cx_bin = np.cumsum(np.cos(th_bin)) * ds                    # centreline (tangent rotates ventrally)
    cy_bin = -np.cumsum(np.sin(th_bin)) * ds
    thc = np.interp(x, ctr, th_bin)
    cxc = np.interp(x, ctr, cx_bin); cyc = np.interp(x, ctr, cy_bin)
    nx, ny = np.sin(thc), np.cos(thc)                          # dorsal normal
    h = y - np.median(y)
    return np.stack([cxc + h * nx, cyc + h * ny, z], 1).astype(P.dtype)


def integrity(P, fid, r=0.05):
    """Body connectivity: connected components of the contact graph, and the fraction of
    cross-tissue (heterotypic) contacts. The fascia keeps it one component and binds tissues."""
    pairs = cKDTree(P).query_pairs(r, output_type="ndarray")
    if len(pairs) == 0:
        return len(P), 0.0
    n = len(P)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    ncomp, _ = connected_components(g, directed=False)
    het = float(np.mean(fid[pairs[:, 0]] != fid[pairs[:, 1]]))
    return int(ncomp), het


def simulate(use_ecm=True, seed=0, verbose=False, n_start=None, n_end=None, limb_buds=False,
             pcp=0.25, convergent_ext=None, relax=3, shape_target=None, flexure=0.0,
             head_expand=False, apical_fold=0.0, regional_growth=False, head_shape_relax=1.0,
             limb_params=None, eye_frontation=0.0, fate_params=None):
    rng = np.random.RandomState(seed)
    lpar = {**LIMB_DEFAULTS, **(limb_params or {})}       # searchable limb-program knobs (von Dassow-Odell)
    fpar = {**FATE_DEFAULTS, **(fate_params or {})}       # searchable fate-map knobs (joint HESTA search)
    # --- GENOME-GROUNDED limb frame -----------------------------------------------------------
    # fore/hind Hox AP levels from real Hox colinearity; width knob `pcp` from the ABC Wnt-PCP
    # convergent-extension tone x the species convergent_ext knob (medic.limb_genome_frame).
    # convergent_ext=None keeps the legacy hand-set frame (pcp param, Hox 0.20/0.44) for back-compat.
    fore_ap, hind_ap = 0.20, 0.44
    if convergent_ext is not None:
        from medic.limb_genome_frame import genome_limb_frame
        _gf = genome_limb_frame(convergent_ext)
        fore_ap, hind_ap, pcp = _gf["fore_ap"], _gf["hind_ap"], _gf["pcp"]
    ns = N_START if n_start is None else int(n_start)      # start count (1 = literal single cell)
    ne = N_END if n_end is None else int(n_end)
    pos = np.zeros((ne, 3), np.float32)
    vm = np.full(ne, V_NEUTRAL, np.float32)
    fid = np.full(ne, -1, np.int32)
    adhc = np.zeros(ne, np.float32)
    ecmc = np.zeros(ne, np.float32)
    born = ns
    sp = 0.22 if ns > 1 else 0.0                           # single cell -> at the origin
    pos[:born, 0] = rng.uniform(-sp, sp, born)
    pos[:born, 1] = rng.uniform(-sp * 0.64, sp * 0.64, born)
    pos[:born, 2] = rng.uniform(-sp * 0.64, sp * 0.64, born)
    apE = None; lrE = None; apE_tree = None                # electric-body frame (AP + LR modes), lazy
    antinode_levels = None; dv_levels = None               # AP + DV antinode ladders organs fill in clock order
    lr_quality = 0.0                                       # |corr| of the chosen LR mode with the ML axis
    lr_aspect = 0.0; lr_eigratio = 0.0                     # body width criterion at the eigenmode snapshot

    R_REP, R_ADH, R_ECM = 0.032, 0.060, 0.150   # integrin/ECM (fascia) reaches FURTHER than cadherin:
    #   the matrix is a long-range continuum, so it must span the gap an outgrowing limb opens.
    K_REP, K_ADH, K_ECM = 0.55, 0.22, 0.16       # integrins now BIND (was 0.10 -- the weak-integrin
    #   cause of limb fragmentation); applied over the relaxation loop below so it re-knits, not oscillates.
    RELAX = max(1, int(relax))                   # mechanical relaxation iterations per step (fascia re-knit)
    frames = []
    for s in range(STEPS):
        frac = s / (STEPS - 1)
        t_hpf = T0 + (T1 - T0) * frac
        prc2 = prc2_div(t_hpf)

        # ---- DIVISION ----
        target = int(ns + (ne - ns) * frac ** 1.3)
        if target > born:
            P = pos[:born]
            a = _norm(P[:, 0]); d = _norm(P[:, 1])
            # DIVISION HEAD (redesigned 2026-07-16, data-grounded vs MOSTA -- medic.investigate_division):
            # real proliferation is roughly UNIFORM in space and DECLINES globally as the clock runs
            # down (tissue-mean 0.5 -> 0.2 across E9.5->E13.5); undifferentiated PROGENITORS divide a
            # little more. The old posterior + dorsal-neural POSITIONAL bias was falsified at every
            # stage (prediction map r~0.01), so it is removed. Per-cell rate = a uniform baseline + a
            # progenitor (undifferentiated) boost + a mild broad-trunk term retained ONLY to offset the
            # fate head's anterior-neural over-assignment (a separate, flagged gap), not a division claim.
            # DIVISION HEAD (data-grounded vs MOSTA, medic.investigate_division): real proliferation
            # is roughly UNIFORM in space and DECLINES globally as the clock runs down (tissue-mean
            # ~0.5 E9.5 -> ~0.2 E13.5); undifferentiated PROGENITORS divide a little more. The old
            # posterior growth-zone + dorsal-neural POSITIONAL bias was falsified at every stage
            # (prediction map r~0.01), so it is gone. Axis ELONGATION is the migration head's job
            # (convergent extension), NOT division's -- any residual anterior-heaviness is a
            # migration/fate gap, not a reason to re-bias division.
            undiff = (fid[:born] < 0).astype(np.float32)
            prolif = 0.5 + 0.4 * undiff                               # uniform baseline + progenitor bonus
            # AER-DRIVEN LIMB PROLIFERATION (FGF8/FGF10): the real limb grows by cell division under the
            # apical ectodermal ridge, not by shoving the same cells outward. Boost the limb-bud pool's
            # division weight so the buds accumulate mesenchyme (lp['prolif']=1 = off = back-compatible).
            # AER REGRESSION (cycle 68): the AER boost + inheritance run only while the family is
            # BELOW its staged share (growth_program, HESTA 4.7%) -- FGF8 withdraws when the limb
            # reaches size and the progress zone closes. Without this gate the column-growth law's
            # recruits compound under prolif x inherit to ~24% of the body (measured, this cycle).
            from medic.growth_program import target as _aer_target
            lb = fid[:born] == FIDX["Limb Bud"]
            _aer_lt = _aer_target("Limb Bud", prc2, None)
            _aer_open = _aer_lt is None or (lb.sum() / max(born, 1)) < _aer_lt
            if limb_buds and lpar["prolif"] > 1.0 and _aer_open:
                prolif[lb] *= lpar["prolif"]
                # Tbx5 forelimb boost: extra AER division on the ANTERIOR (fore, a<0.5) bud so it grows to
                # parity with the posterior pair and its cartilage resolves stylopod/zeugopod/autopod.
                if lpar.get("fore_boost", 1.0) != 1.0:
                    prolif[lb & (a < 0.5)] *= lpar["fore_boost"]
            # ---- WINDOW-DRIVEN PROLIFERATION (genome-anchored via the differentiation clock) ----
            # Organ cell NUMBER is the proliferation rate INTEGRATED over the progenitor WINDOW, and the
            # window is the clock: a tissue that stays an undifferentiated, cycling PROGENITOR longer
            # accumulates more cells before its cells exit the cycle. That long window -- not a fast per-
            # step rate -- is why the big pools (CNS ~22%, muscle ~10%, meninges ~10%, cartilage ~7%,
            # connective ~5%) dominate while the organs (heart/liver ~3%) stay small: the neural and
            # mesenchymal progenitor pools cycle across the whole neurogenic/myogenic window, the organs
            # exit early. So each fate gets a WINDOW multiplier (WINDOW_CLASS, by progenitor persistence),
            # sustained while the clock runs (win gate) and, for the brain, graded rostrally so the cephalic
            # vesicles balloon. This must cover ALL long-window pools, not just the head -- boosting the
            # head alone starves the trunk of the fixed cell budget. WINDOW_CLASS magnitude is validated
            # against the MOSTA E9.5->E13.5 growth (g_K-anchor pattern: clock mechanism, atlas-tuned scale).
            if head_expand:
                from medic.proliferation_genome import WINDOW_CLASS
                fn_arr = np.array([FATES[j] if j >= 0 else "" for j in fid[:born]])
                wc = np.array([WINDOW_CLASS.get(f, 1.0) for f in fn_arr], np.float32)
                wc[fid[:born] < 0] = np.maximum(wc[fid[:born] < 0], 1.6)  # uncommitted progenitors cycle
                win = float(np.clip((prc2 - 0.16) / 0.58, 0.0, 1.0))     # 1 early, ->0 late (cells exit cycle)
                CEPH = {"Forebrain", "Midbrain", "Hindbrain", "Cerebellum", "OlfactoryBulb", "Brain"}
                ceph = np.isin(fn_arr, list(CEPH)) | ((fid[:born] < 0) & (a < 0.32))
                if regional_growth:
                    # PHASE 1b: the organizer field (ANR + isthmic FGF8 + Shh/Wnt DV) is the PRIMARY
                    # growth driver of the cephalic domain -- it REPLACES the uniform window-class boost
                    # there (so its spatial structure survives instead of being washed out), while the
                    # trunk/organ pools keep the validated window-class term.
                    from medic.regional_proliferation import region_growth
                    rg = region_growth(a, d, ceph.astype(np.float32))
                    boost = np.where(ceph, win * (rg - 1.0), win * (wc - 1.0))
                else:
                    head_grade = 1.0 + 1.2 * np.clip(1.0 - a / 0.38, 0.0, 1.0) * ceph.astype(np.float32)
                    boost = win * (wc - 1.0) * head_grade
                prolif = prolif * (1.0 + boost)
            prolif = prolif / prolif.sum()
            n_add = min(target - born, ne - born)
            par = rng.choice(born, n_add, p=prolif)
            off = rng.normal(0, 0.030, (n_add, 3)).astype(np.float32)
            off[:, 1] *= 0.55; off[:, 2] *= 0.55 * (1.0 - 0.62 * pcp)   # Wnt-PCP: new cells added in a
            #                                                            thinner ML band -> narrower body
            off[_norm(pos[par, 0]) > 0.78, 0] += 0.085
            pos[born:born + n_add] = pos[par] + off
            vm[born:born + n_add] = V_NEUTRAL
            fid[born:born + n_add] = -1
            # AER progenitor inheritance: a limb-bud cell's daughters STAY limb mesenchyme (the AER holds
            # the progenitor pool limb-fated), so the extra divisions FILL the limb instead of seeding
            # undifferentiated cells that drift off (lp['inherit']=0 = off = back-compatible).
            # CYCLE 69 (Miles's eye: "calves lost a lot of cells"): inheritance is UNGATED again.
            # Cycle 68 closed it with the AER regression gate -- but inheritance is FATE-KEEPING,
            # not growth: with it off, every limb daughter born during the outgrowth window
            # (prc2 0.40 -> 0.23, exactly when the leg extends) fell out of the family and the
            # calves emptied while recruitment replaced the mass proximally. Only the FGF8
            # proliferation BOOST is share-gated (that is what compounded to 24%); daughters of
            # limb cells dividing at the BASELINE rate stay limb -- the distal supply.
            if limb_buds and lpar["inherit"] and n_add:
                _lb = (fid[par] == FIDX["Limb Bud"])
                if _lb.any():
                    _o = (np.arange(born, born + n_add))[_lb]
                    fid[_o] = FIDX["Limb Bud"]; adhc[_o] = ADH["Limb Bud"]; ecmc[_o] = ECM["Limb Bud"]
            born += n_add

        P = pos[:born]
        kq = min(14, born)          # wider neighbourhood so a limb-base cell also sees TRUNK cells --
        #                             the integrin/ECM fascia can only bind neighbours it can reach.
        q_idx = cKDTree(P).query(P, k=kq)[1]
        nbr = q_idx[:, 1:] if kq > 1 else np.empty((born, 0), dtype=int)
        has_nbr = nbr.shape[1] > 0
        a = _norm(P[:, 0]); d = _norm(P[:, 1]); mln = np.abs(P[:, 2]) / (np.abs(P[:, 2]).max() + 1e-6)

        # ---- DIFFERENTIATION: commit on unlock, keep ----
        for i in np.where(fid[:born] < 0)[0]:
            f = fate_of(a[i], d[i], mln[i], prc2, fpar)
            if f is not None:
                fid[i] = FIDX[f]; adhc[i] = ADH[f]; ecmc[i] = ECM[f]
        # LIMB BUDS: once the clock unlocks (prc2 low), specify paired FORE + HIND lateral-plate
        # lobes -- convert lateral-plate mesoderm / uncommitted cells in the four bud zones. Done
        # before heavy convergent extension so the posterior still has lateral cells to recruit.
        # gate 0.46 -> 0.60 (cycle 20, emergence timing): the electric frame + organ sprouting open with
        # the FIRST organ (the heart's primary tube exists at CS09, prc2 ~0.59) -- the old single gate
        # held every organ to CS12+ regardless of its unlock. Limb CONVERSION keeps its own 0.46 clock
        # below; skin keeps 0.42; the gut tube law keeps the 0.46 condensation gate.
        if limb_buds and prc2 <= 0.60:
            # ==== ELECTRIC BODY: low eigenmodes of the kNN gap-junction operator = the body axes ====
            # (same frame as the electric face / mammary line / six-pack, Paper #4). The AP mode gives
            # the antero-posterior coordinate; the LR mode is the bilateral frame -- its NODE is the
            # midline, its two ANTINODES are the left/right sides. Limbs are placed on the antinodes.
            if apE is None:
                # ELECTRIC-BODY FRAME, read from the PHYSICAL AP/LR axes -- the robust fix the block's own
                # comments prescribed. The low kNN-Laplacian eigenmodes SELECT these same axes, but for a
                # near-symmetric body they are near-degenerate, so a multithreaded eigensolver returns a
                # different (rotated) basis each PROCESS launch -> the limb/organ frame flips -> limbs
                # form-or-don't run to run (measured: 159 vs 0 limb cells with identical inputs). We take
                # the AP coordinate as the rank along physical x and the LR coordinate as physical z (node
                # at the midline) -> deterministic + reproducible, the SAME body frame the eigenmode picked.
                # This is REQUIRED for any state-space (von Dassow-Odell) search: the objective must not be
                # noisy, or the optimiser cannot tell a good knob move from an eigenbasis flip.
                lr_aspect = float(P[:, 2].std() / (P[:, 0].std() + 1e-9))   # width criterion (physical)
                if verbose:
                    # eigenspectrum kept as a DIAGNOSTIC ONLY -- it no longer gates or positions anything
                    nb = cKDTree(P).query(P, k=min(11, born))[1][:, 1:]
                    rr = np.repeat(np.arange(born), nb.shape[1]); cc = nb.ravel()
                    Wk = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(born, born)).tocsr()
                    Wk = ((Wk + Wk.T) > 0).astype(float)
                    Lk = diags(np.asarray(Wk.sum(1)).ravel()) - Wk
                    try:
                        vv, _UU = eigsh(Lk, k=min(14, born - 1), which="SM"); vv = np.sort(vv)
                        print(f"    [limb frame] aspect={lr_aspect:.3f} eig[1:4]={np.round(vv[1:4], 4)}")
                    except Exception as _e:                                # pragma: no cover
                        print(f"    [limb frame] aspect={lr_aspect:.3f} (eig diag skipped: {_e})")
                apr = np.argsort(np.argsort(P[:, 0])).astype(np.float32) / max(1, born - 1)  # AP rank (physical)
                lrn = (P[:, 2] / (np.abs(P[:, 2]).max() + 1e-9)).astype(np.float32)          # LR, node at 0 (physical)
                apE = np.full(ne, -1.0, np.float32); apE[:born] = apr
                lrE = np.zeros(ne, np.float32); lrE[:born] = lrn
                apE_tree = (cKDTree(P.copy()), apr.copy(), lrn.copy())
                # the AP + DV antinode grid of THIS body -- organs fill it in clock order (Miles's law)
                _be = body_electric_antinodes(P)
                antinode_levels = _be["ap_levels"]; dv_levels = _be["dv_levels"]
            # width gate re-read EVERY step (cycle 20): the frame now freezes at prc2 0.60 on a narrower
            # pre-CE body -- a frozen lr_aspect would under-read the width and fail the limb gate (the
            # amphibian width-threshold law). The aspect is the LIVE body's width, the frame is the map.
            lr_aspect = float(P[:, 2].std() / (P[:, 0].std() + 1e-9))
            miss = np.where(apE[:born] < 0)[0]                         # cells born since -> NN on the frame
            if len(miss):
                tr, av, lv = apE_tree; j = tr.query(pos[miss], k=1)[1]
                apE[miss] = av[j]; lrE[miss] = lv[j]
            aE = apE[:born]; lr = lrE[:born]
            fi_ = fid[:born]
            # HOX sets the two AP levels (fore + hind) ON the electric-body AP axis; the LR mode's
            # ANTINODES (|lr| high, off-midline) give the bilateral sides, its NODE (lr~0) stays clear.
            # LIMBS FORM ONLY IF A GENUINE LR (bilateral) MODE EXISTS: a narrow, tapered body (fish) has
            # no left-right eigenmode (lr_quality low) -> no limbs; a wide body (tetrapod) does -> limbs.
            if lr_aspect > lpar["gate_aspect"] and prc2 <= 0.46:   # limb conversion keeps its own clock
                # The electric-body LR eigenmode gates WHETHER limbs form (lr_aspect: only a wide enough
                # body has the bilateral mode). WHERE they form uses a LOCAL mediolateral coordinate --
                # |z| relative to the body half-width at each AP slice -- so the lateral plate is found
                # along the WHOLE trunk despite the anteroposterior taper. Fore/hind = the genome Hox
                # levels in physical AP; left/right = the two sides -> the tetrapod's four limbs.
                apb = np.clip((a * 24).astype(int), 0, 23)
                locmax = np.ones(24, np.float32)
                for k in range(24):
                    mk = apb == k
                    if mk.sum() > 3:
                        locmax[k] = float(np.abs(P[mk, 2]).max()) + 1e-6
                mll = np.abs(P[:, 2]) / locmax[apb]                 # 0 midline .. 1 local lateral edge
                # the two genome Hox levels (physical AP) define two narrow competence BANDS with a GAP
                # between them, so a fore and a hind field are distinct along the axis to begin with.
                hox = np.exp(-((a - fore_ap) / lpar["hox_w"]) ** 2) + np.exp(-((a - hind_ap) / lpar["hox_w"]) ** 2)
                comp = ((mll > lpar["mll_thr"]) & (d >= lpar["dv_lo"]) & (d <= lpar["dv_hi"]) & (hox > 0.40)
                        & ((fi_ == FIDX["Mesoderm"]) | (fi_ < 0)))
                Aact = np.where(comp, hox * mll, 0.0).astype(np.float32)
                # LATERAL INHIBITION (reaction-diffusion Mexican hat): activator minus its long-range
                # neighbourhood mean -> the broad competent lateral plate CONDENSES into four discrete,
                # spaced limb buds (the tissue between them is inhibited). Same mechanism that resolves
                # the paired eyes -- without it the buds merge into a connected fin.
                nbL = cKDTree(P).query(P, k=min(int(lpar["inhib_k"]), born))[1]
                Aeff = Aact - lpar["inhib"] * Aact[nbL].mean(1)
                apmid = 0.5 * (fore_ap + hind_ap)
                zc = P[:, 2]
                conv_parts = []
                for apm in (a < apmid, a >= apmid):                 # fore (anterior) / hind (posterior)
                    for side in (zc > 0, zc < 0):                   # left / right
                        cs = comp & apm & side & (Aeff > 0.0)
                        if cs.sum() >= 4:
                            idxq = np.where(cs)[0]
                            thr = lpar["peak_thr"] * float(Aeff[cs].max())   # keep the sharpened peak = the bud
                            conv_parts.append(idxq[Aeff[cs] > thr])
                conv = np.concatenate(conv_parts) if conv_parts else np.array([], dtype=int)
                if verbose:
                    _mes = ((fi_ == FIDX["Mesoderm"]) | (fi_ < 0))
                    print(f"    [limb form] prc2={prc2:.2f} aspect={lr_aspect:.2f} hoxOK={(hox>0.40).sum()} "
                          f"mllOK={(mll>lpar['mll_thr']).sum()} dvOK={((d>=lpar['dv_lo'])&(d<=lpar['dv_hi'])).sum()} "
                          f"mesOK={_mes.sum()} comp={int(comp.sum())} Aeff+={int((Aeff>0).sum())} conv={len(conv)}")
                fid[conv] = FIDX["Limb Bud"]; adhc[conv] = ADH["Limb Bud"]; ecmc[conv] = ECM["Limb Bud"]
            # ORGAN BUDS: no longer typed windows -- each organ SPROUTS where the five
            # effectors (connexin/cadherin/integrin/differentiation) co-activate within its
            # address, condensed to discrete buds by lateral inhibition (medic.organ_sprouting;
            # the same five-factor law that marks the budding primordia in the E9.5 atlas).
            # AP = physical a, DV = physical d (same frame as the ladders); LR = electric LR mode
            _buds = sprout_organs(a, d, lrE[:born], fid[:born], prc2, antinode_levels, dv_levels, mln=mln)
            for _oname, _oidx in _buds.items():
                fid[_oidx] = FIDX[_oname]; adhc[_oidx] = ADH[_oname]; ecmc[_oidx] = ECM[_oname]
            # ORGAN CONDENSATION: a point-organ primordium CONDENSES into one coherent mass instead of
            # staying salt-and-pepper -- the kNN cadherin cannot gather cells that are not already
            # neighbours, so pull each point organ's cells toward their own median (mesenchymal
            # condensation), gated by the clock. Median (not mean) resists a few stray cells.
            _cond = float(np.clip((0.46 - prc2) / 0.24, 0.0, 1.0))
            # per-organ gate opening (cycle 20, emergence timing): the cardiac crescent condenses FIRST
            # (the heart functions from CS10) -- its gate opens with its sprout unlock; others keep 0.46.
            _COND_OPEN = {"Heart": 0.60}
            if _cond > 0 or prc2 <= max(_COND_OPEN.values()):
                _generic = {FIDX[f] for f in ("Mesoderm", "Somite", "Hypoblast", "Yolk Syncytial Layer",
                            "Blastodisc", "Proliferative Like Cell", "Neural Crest") if f in FIDX}
                _fib = fid[:born]
                _maxr = 0.16 * float(np.linalg.norm(pos[:born].max(0) - pos[:born].min(0)))
                for _on in POINT_ORGANS:
                    _cond_o = float(np.clip((_COND_OPEN.get(_on, 0.46) - prc2) / 0.24, 0.0, 1.0))
                    if _cond_o <= 0:
                        continue
                    _om = np.where(_fib == FIDX[_on])[0]
                    if len(_om) < 4:
                        continue
                    # a PAIRED organ is bilateral -> condense and grow EACH side as its own primordium, so the
                    # two LR-antinode buds stay separated. Condensing the whole organ to one median would pull
                    # both sides onto the midline (the median of a bilateral set), collapsing the pair -- the
                    # placement bug the audit found. Midline organs (heart, liver) stay one cluster.
                    if _on in PAIRED_ORGANS:
                        zc = pos[:born][_om, 2]
                        clusters = [_om[zc >= 0], _om[zc < 0]]
                    elif _on == "Heart" and prc2 <= 0.32 and len(_om) >= 40:
                        # THE CHAMBERS FORM (cycle 22): from the septation window (CS17, prc2~0.32) the
                        # heart condenses as THREE LOBES, not one ball -- the reference heart is
                        # four-lobed through CS17-23 while ours stayed a blob (the late-heart dip 66-80).
                        # Partition by the SAME rules the final subhead split uses (anterior=outflow,
                        # dorsal=atria, ventral=ventricles), so the end-of-run names land on the lobes;
                        # each lobe condenses to its own centroid, and the lobe centroids are pulled to
                        # the measured bp3d chamber offsets (heart-length units, ML left to laterality).
                        _ax = pos[:born][_om, 0]; _dy = pos[:born][_om, 1]
                        _athr = np.quantile(_ax, 0.72)
                        _ofl = _om[_ax >= _athr]
                        _rest = _om[_ax < _athr]
                        _dm = np.median(pos[:born][_rest, 1]) if len(_rest) else 0.0
                        _atr = _rest[pos[:born][_rest, 1] >= _dm]
                        _ven = _rest[pos[:born][_rest, 1] < _dm]
                        clusters = [c for c in (_ofl, _atr, _ven) if len(c) >= 4]
                        _hlen = float(np.ptp(pos[:born][_om], axis=0).max()) + 1e-9
                        _hc = pos[:born][_om].mean(0)
                        # gain 0.30 -> 0.45 (cycle 29-lite, 2026-09-04): the CS23 heart A/B measured
                        # tighter lobes worth +3 (77->80); hollowing was REFUTED (76 -> 75.7, the
                        # honest null -- the reference heart at this scale is not hollow-dominant).
                        _g = float(np.clip((0.32 - prc2) / 0.08, 0.0, 1.0)) * 0.45
                        for _cl, (_dx, _dyo) in zip((_ofl, _atr, _ven),
                                                    ((+0.25, -0.20), (+0.13, +0.15), (-0.16, -0.10))):
                            if len(_cl) >= 4:
                                tgt = _hc + _hlen * np.array([_dx, _dyo, 0.0])
                                pos[:born][_cl] += _g * (tgt - pos[:born][_cl].mean(0))
                    else:
                        clusters = [_om]
                    # THE GROWTH-PROGRAM HEAD (cycle 19): clock-gated staged targets for the wired
                    # families (heart declines as the body outgrows it, liver balloons for fetal
                    # haematopoiesis -- measured Carnegie ladder); unwired families keep ORG_TARGET.
                    from medic.growth_program import target as _staged_target
                    _tot = int(_staged_target(_on, prc2, ORG_TARGET.get(_on, 0.012)) * born)
                    for _cl in clusters:
                        if len(_cl) < 3:
                            continue
                        ctr = np.median(pos[:born][_cl], axis=0)
                        pos[:born][_cl] += 0.38 * _cond_o * (ctr - pos[:born][_cl])   # condense this side to its own centroid
                        # GROW + PURIFY: recruit the closest GENERIC (undifferentiated) cells into this side up
                        # to its (per-side) target size, converting foreign generic cells in its core.
                        need = _tot // len(clusters) - len(_cl)
                        if need > 0:
                            gi = np.where(np.isin(_fib, list(_generic)))[0]
                            if len(gi):
                                dd = np.linalg.norm(pos[:born][gi] - ctr, axis=1)
                                gi = gi[dd < _maxr]; dd = dd[dd < _maxr]
                                take = gi[np.argsort(dd)[:need]]
                                fid[take] = FIDX[_on]; adhc[take] = ADH[_on]; ecmc[take] = ECM[_on]
                                _fib = fid[:born]
                # GUT TUBE GROWTH (the tube-organ recruitment law; see GUT_TARGET above). The gut family
                # grows to its HESTA share by converting the generic cells nearest the EXISTING tube --
                # per-cell distance to the nearest gut cell, so recruitment follows the tube's whole AP
                # span. The pool it naturally drains is the ventral Hypoblast/YSL endoderm lineage (the
                # yolk sac IS resorbed into the midgut) plus adjacent ventral mesenchyme (the muscular
                # wall). Recruits enter as "Gut"; the existing Mucosa DV split and Foregut/Hindgut AP
                # regionalisation subheads then sort them. Committed organs are never touched.
                _gut_ids = [FIDX[n] for n in ("Gut", "Mucosa", "Foregut", "Hindgut") if n in FIDX]
                _gm = np.where(np.isin(_fib, _gut_ids))[0]
                _gneed = int(GUT_TARGET * born) - len(_gm)
                if _cond > 0 and _gneed > 0 and len(_gm) >= 8:   # the tube law keeps the 0.46 gate
                    gi = np.where(np.isin(_fib, list(_generic)))[0]
                    if len(gi):
                        d2tube, _ = cKDTree(pos[:born][_gm]).query(pos[:born][gi], k=1)
                        okr = d2tube < _maxr
                        gi, d2tube = gi[okr], d2tube[okr]
                        take = gi[np.argsort(d2tube)[:_gneed]]
                        fid[take] = FIDX["Gut"]; adhc[take] = ADH["Gut"]; ecmc[take] = ECM["Gut"]
                        _fib = fid[:born]
                # LIMB COLUMN GROWTH (cycle 68 -- the growth ladder's next customer; the cycle-66
                # pool correction). The limb family is a spanning COLUMN like the gut is a tube:
                # bud conversion is geometric (fitted at n=9k, limb 6.8%; drifted to 2.3% at 120k),
                # so it too needs a fraction-of-born law. Recruit the generic cells NEAREST the
                # existing limb columns (lateral-plate mesenchyme joining the bud -- recruitment is
                # proximal, where generic neighbours exist; the AER prolif and the maturation's
                # rank-uniform PD re-spacing carry the supply distally) up to the staged HESTA
                # target (growth_program: 4.7% at CS12-13, the single measured anchor).
                from medic.growth_program import target as _lim_target
                _lm = np.where(_fib == FIDX["Limb Bud"])[0]
                _ltar = _lim_target("Limb Bud", prc2, None)
                if _cond > 0 and _ltar is not None and len(_lm) >= 8:
                    _lneed = int(_ltar * born) - len(_lm)
                    if _lneed > 0:
                        gi = np.where(np.isin(_fib, list(_generic)))[0]
                        # LATERAL-PLATE COMPETENCE (cycle 69, Miles's eye: "feet stuck together"):
                        # nearest-to-column recruiting also converted the generic cells BETWEEN the
                        # legs (nearest to both columns) -- limb cells at the crotch midline dragged
                        # the foot anchors together. Recruits must sit in the Tbx5/Tbx4 lateral-
                        # plate territory: per-AP-slice local ML fraction above 0.45 (the bud
                        # conversion's own frame, looser than its 0.62 -- the sleeve around the
                        # bud), which excludes the inter-limb midline by construction.
                        if len(gi):
                            _Pb = pos[:born]
                            _apb = np.clip((_norm(_Pb[:, 0]) * 24).astype(int), 0, 23)
                            _locmax = np.ones(24, np.float32)
                            for _k in range(24):
                                _mk = _apb == _k
                                if _mk.sum() > 3:
                                    _locmax[_k] = float(np.abs(_Pb[_mk, 2]).max()) + 1e-6
                            _mll_g = np.abs(_Pb[gi, 2]) / _locmax[_apb[gi]]
                            gi = gi[_mll_g > 0.45]
                        if len(gi):
                            d2limb, _ = cKDTree(pos[:born][_lm]).query(pos[:born][gi], k=1)
                            okr = d2limb < _maxr
                            gi, d2limb = gi[okr], d2limb[okr]
                            take = gi[np.argsort(d2limb)[:_lneed]]
                            fid[take] = FIDX["Limb Bud"]
                            adhc[take] = ADH["Limb Bud"]; ecmc[take] = ECM["Limb Bud"]
                            # HONEST NULL (cycle 68, measured): a c-Met-style migration step
                            # (recruits pulled 0.6 toward their nearest column cell) was tried and
                            # REVERTED -- it piled a dense proximal sleeve on the column and the
                            # benchmark read it (canonical limb_bone 55.2 -> 52.4, muscle 78.4 ->
                            # 75.3, grays 0.964 -> 0.957 vs the in-place variant). Recruits stay
                            # in place as the lateral-plate sleeve; the AER prolif + the carve's
                            # rank-uniform PD re-spacing carry the supply distally.
            # SKIN: the epidermal envelope = the outer radial SHELL (surface ectoderm). Per AP slice,
            # cells in the outer rim become Skin -- over the deep organs, which stay internal.
            if prc2 <= 0.42:
                Pb = pos[:born]
                apb = np.clip((a * 24).astype(int), 0, 23)
                radial = np.zeros(born, np.float32)
                for k in range(24):
                    mkk = apb == k
                    if mkk.sum() > 5:
                        rr = np.hypot(Pb[mkk, 1] - Pb[mkk, 1].mean(), Pb[mkk, 2] - Pb[mkk, 2].mean())
                        radial[mkk] = rr / (rr.max() + 1e-9)
                _deep = {FIDX[f] for f in ("Eye", "Otic", "Heart", "Lung", "Liver", "Pancreas",
                         "Kidney", "Gut", "Rib", "Muscle", "Limb Bud", "Notochord", "Forebrain",
                         "Nervous System", "Spinal Cord", "Cartilage", "DRG", "Sympathetic", "Vessel",
                         "Jaw", "Choroid", "Gonad", "Midbrain", "Hindbrain", "Cerebellum",
                         "Branchial", "Blood", "Mesentery",
                         "Atrium", "Ventricle", "Outflow", "LiverHaem", "Foregut", "Hindgut",
                         "Nephron", "Retina", "Adrenal", "Thymus", "Spleen",
                         "Bladder", "Adipose", "OlfactoryBulb")}
                skinm = (radial > 0.88) & ~np.isin(fid[:born], list(_deep))   # thin epidermal shell (was 0.74 -> ~12%)
                sidx = np.where(skinm)[0]
                fid[sidx] = FIDX["Skin"]; adhc[sidx] = ADH["Skin"]; ecmc[sidx] = ECM["Skin"]
                # MENINGES: the neural-crest ENVELOPE of the central nervous system (master TF Zic1, the
                # E12.5 MOSTA "Meninges" head) -- a mesenchymal shell just INSIDE the skin, over the brain
                # and dorsal cord. Taken from the mid-outer radial band (0.55-0.74) in the head + dorsal
                # region, from cells not already a deep organ, neural core, or skin -> a covering layer.
                # gated at prc2<=0.40 (just AFTER the neural crest at 0.42) so the envelope wraps a CNS
                # that already exists -- the crest -> meninges branch of the cascade (medic.head_cascade).
                _cover = list(_deep) + [FIDX["Skin"]]
                mening = ((radial > 0.62) & (radial <= 0.74) & (a < 0.40) & (prc2 <= 0.40)
                          & ~np.isin(fid[:born], _cover))
                midx = np.where(mening)[0]
                fid[midx] = FIDX["Meninges"]; adhc[midx] = ADH["Meninges"]; ecmc[midx] = ECM["Meninges"]
                # CONNECTIVE TISSUE: the pervasive interstitial mesenchyme (master TF Twist2, present at
                # every MOSTA stage) filling BETWEEN the organs. This is the LARGE loose-mesenchyme pool
                # (real ~5-10%), NOT a thin dorsal strip -- it is the trunk interstitium that fills the
                # space the organs and segments leave. Runs AFTER sprout_organs, so it takes only the
                # LEFTOVER trunk Mesoderm (the committed organs/segments already claimed their cells and are
                # no longer Mesoderm); it therefore cannot steal an organ. Broadened 2026-07-19 across the
                # whole trunk DV so the interstitium is the bulk filler -- previously it was a decimated
                # dorsal strip (d 0.42-0.72, every-other cell), which left the interstitium empty and let the
                # midline Cartilage column become the nearest-fate filler (the cartilage-for-connective
                # mislabel). Head mesenchyme (a<0.30) is handled separately below.
                conn = np.where((fid[:born] == FIDX["Mesoderm"]) & (a >= 0.30) & (d > 0.40) & (d < 0.74) & (mln < 0.55))[0]
                cidx = conn                                        # ALL leftover interstitial mesoderm (was ::2); wider
#   bands re-introduce the residual bimodality (overlap the neural/head region), so kept to the trunk here.
                fid[cidx] = FIDX["Connective"]; adhc[cidx] = ADH["Connective"]; ecmc[cidx] = ECM["Connective"]
                # LEFTOVER SOMITE -> CONNECTIVE (the dermomyotome/interstitial fraction). After the
                # cartilage/rib/muscle segments have recruited their somite (they unlock at prc2 0.40-0.42),
                # the paraxial somite cells still left uncommitted late (prc2<=0.30) are the loose
                # dermomyotome-derived mesenchyme -- the pervasive interstitial CONNECTIVE (Twist2), real
                # ~5-10%. Committing them here BOTH populates the connective interstitium AND removes the
                # cream "Somite" pool that otherwise renders as cartilage and is over-sampled by the medial
                # sagittal slab. Safe for the organs: Somite is not the heart/liver germ, so heart cannot be
                # starved; muscle/cartilage already claimed their somite earlier in the clock.
                if prc2 <= 0.30:
                    leftover_som = np.where(fid[:born] == FIDX["Somite"])[0]
                    fid[leftover_som] = FIDX["Connective"]
                    adhc[leftover_som] = ADH["Connective"]; ecmc[leftover_som] = ECM["Connective"]
                # HEAD MESENCHYME (master Prrx1): the cranial crest/mesoderm mesenchyme filling the head
                # around the brain, anterior of the Hox trunk. Relabel the anterior generic mesoderm.
                hm = np.where((fid[:born] == FIDX["Mesoderm"]) & (a < 0.30) & (d > 0.20) & (d < 0.62))[0]
                fid[hm] = FIDX["HeadMes"]; adhc[hm] = ADH["HeadMes"]; ecmc[hm] = ECM["HeadMes"]
                # MESOTHELIUM (master Wt1): the serosal lining of the body cavity -- a thin VENTRAL-trunk
                # shell just inside the skin, bordering the coelom, the ventral counterpart of the meninges.
                # a THIN serosal lining (was radial 0.60-0.74 & d<0.36 = ~6% of cells, far too much and
                # eating the ventral cells the gut/heart need); tightened to the outer ventral rim only.
                meso = ((radial > 0.66) & (radial <= 0.74) & (a > 0.46) & (a < 0.86) & (d < 0.26)
                        & ~np.isin(fid[:born], _cover))
                mi = np.where(meso)[0]
                fid[mi] = FIDX["Mesothelium"]; adhc[mi] = ADH["Mesothelium"]; ecmc[mi] = ECM["Mesothelium"]
                # MUCOSAL EPITHELIUM (master Grhl3): the inner epithelial lining of the gut tube = a gut
                # SUBHEAD. Split the WHOLE gut pool (Gut + Mucosa) each step by DV: the ventral/luminal
                # half is the mucosa, the dorsal half stays gut wall. Idempotent -- reassigns both from the
                # union so it does not compound and eat the gut (the bug the naive half-relabel caused).
                gpool = np.where((fid[:born] == FIDX["Gut"]) | (fid[:born] == FIDX["Mucosa"]))[0]
                if len(gpool) >= 4:
                    dmed = np.median(d[gpool])
                    inner = gpool[d[gpool] <= dmed]; outer = gpool[d[gpool] > dmed]
                    fid[inner] = FIDX["Mucosa"]; adhc[inner] = ADH["Mucosa"]; ecmc[inner] = ECM["Mucosa"]
                    fid[outer] = FIDX["Gut"]; adhc[outer] = ADH["Gut"]; ecmc[outer] = ECM["Gut"]

        # ---- FATE PURITY: a morphological CLOSING that removes salt-and-pepper, run ONCE at the end so it
        # sharpens the organs without over-growing them. Only a GENERIC/undifferentiated (or uncommitted)
        # cell sitting inside a committed-organ neighbourhood is absorbed into that organ -- a committed
        # organ is NEVER eroded into generic OR into another organ, so scattered small organs (pancreas)
        # are protected and only the foreign cells within an organ's core are purged.
        if s >= STEPS - 3 and born > 50:
            _gen = np.array([FATES[i] in ("Mesoderm", "Somite", "Hypoblast", "Yolk Syncytial Layer",
                             "Blastodisc", "Proliferative Like Cell", "Neural Crest") for i in range(len(FATES))])
            Kp = min(12, born - 1)
            _, _nnp = cKDTree(pos[:born]).query(pos[:born], k=Kp + 1)
            for _ in range(3):
                nf = fid[:born][_nnp[:, 1:]]
                own = fid[:born]
                own_share = (nf == own[:, None]).mean(1)
                counts = np.zeros((born, len(FATES)), np.int32)
                for f in range(len(FATES)):
                    counts[:, f] = (nf == f).sum(1)
                maj = counts.argmax(1); maj_share = counts.max(1) / Kp
                own_generic = _gen[np.clip(own, 0, None)] | (own < 0)     # only generic/uncommitted cells move
                # only cells DEEP inside an organ (majority strongly one organ) are absorbed -> fills holes,
                # purifies cores, without expanding the organ border (so sizes stay put).
                sw = np.where(own_generic & (own_share < 0.5) & (maj_share > 0.62) & (maj != own) & ~_gen[maj])[0]
                if len(sw):
                    fid[sw] = maj[sw]
                    adhc[sw] = np.array([ADH.get(FATES[m], 0.4) for m in maj[sw]], np.float32)
                    ecmc[sw] = np.array([ECM.get(FATES[m], 0.4) for m in maj[sw]], np.float32)

        # ---- ORGAN SUBHEADS: split a fully-grown organ into its subtypes by position (the organ head's
        # differentiation subhead gating its subtypes -- the recursion the paper describes). Run at the
        # FINAL steps only, on mature organs, so it never disturbs the point-organ growth that keys on the
        # organ name each step. IDEMPOTENT union-pool reassignments (like the mucosa split); masters from
        # the SEdb/AlphaGenome head registry (medic.head_registry).
        if s >= STEPS - 2 and born > 50:
            fb = fid[:born]; a2 = _norm(pos[:born, 0]); d2 = _norm(pos[:born, 1])
            def _pool(names):
                return np.where(np.isin(fb, [FIDX[n] for n in names]))[0]
            def _set(cells, name):
                fid[cells] = FIDX[name]; adhc[cells] = ADH[name]; ecmc[cells] = ECM[name]
            # BRAIN VENTRICLES = the fluid CAVITY (real MOSTA "Cavity"): the neuroepithelial vesicles are
            # thin-walled around a large ventricle, so the inner core of each brain vesicle is carved to
            # Cavity, leaving a neural shell -- which also relieves the over-large brain toward the atlas
            # fraction. Plus a sparse ventral-midline coelom around the viscera.
            for _reg in ("Forebrain", "Midbrain", "Hindbrain", "Cerebellum", "Nervous System"):
                rc = _pool((_reg,))
                if len(rc) >= 16:
                    ctr = pos[rc].mean(0); rad = np.linalg.norm(pos[rc] - ctr, axis=1)
                    frac = 0.30 if _reg != "Nervous System" else 0.05          # spinal-cord central canal is THIN (was 0.14, eroding the cord)
                    _set(rc[rad < np.quantile(rad, frac)], "Cavity")
            coel = _pool(("Connective", "Mesoderm", "Mesentery"))
            if len(coel) >= 30:
                mlnc = np.abs(pos[coel, 2]) / (np.abs(pos[:born, 2]).max() + 1e-9)
                cm = coel[(d2[coel] < 0.46) & (a2[coel] > 0.40) & (a2[coel] < 0.82) & (mlnc < 0.34)]
                _set(cm[::3], "Cavity")                                       # the ventral coelomic space (sparse)
            # BLOOD VESSELS: the pervasive endothelial network (Cdh5/Pecam1) threads every tissue, so it is a
            # sparse fraction drawn across the structural pool rather than a single midline tube (real ~2.6%).
            vp = _pool(("Connective", "Muscle", "Cartilage", "Mesoderm", "HeadMes"))
            if len(vp) >= 40:
                _set(vp[::24], "Vessel")
            hp = _pool(("Heart", "Atrium", "Ventricle", "Outflow"))          # heart chambers
            if len(hp) >= 6:                                                 # anterior=outflow(Isl1)
                athr = np.quantile(a2[hp], 0.72); _set(hp[a2[hp] >= athr], "Outflow")
                rest = hp[a2[hp] < athr]
                if len(rest):                                               # dorsal=atria(Tbx5), ventral=ventricle(Irx4)
                    dm = np.median(d2[rest]); _set(rest[d2[rest] >= dm], "Atrium"); _set(rest[d2[rest] < dm], "Ventricle")
            lp = _pool(("Liver", "LiverHaem"))                              # hepatoblast(Foxa3) vs fetal haem(Gata1)
            if len(lp) >= 6:
                order = lp[np.argsort(a2[lp] + 0.31 * d2[lp])]; _set(order[::6], "LiverHaem"); _set(np.setdiff1d(order, order[::6]), "Liver")
            wp = _pool(("Gut", "Foregut", "Hindgut"))                       # gut wall regionalised by AP
            if len(wp) >= 6:
                aw = a2[wp]; _set(wp[aw < 0.42], "Foregut"); _set(wp[aw >= 0.52], "Hindgut"); _set(wp[(aw >= 0.42) & (aw < 0.52)], "Gut")
            kp = _pool(("Kidney", "Nephron"))                               # nephron/cortex (Six2, dorsal)
            if len(kp) >= 6:
                dm = np.median(d2[kp]); _set(kp[d2[kp] >= dm], "Nephron"); _set(kp[d2[kp] < dm], "Kidney")
            ep = _pool(("Eye", "Retina"))                                   # posterior sensory retina (Crx/Rax)
            if len(ep) >= 6:
                am = np.median(a2[ep]); _set(ep[a2[ep] >= am], "Retina"); _set(ep[a2[ep] < am], "Eye")
            op = _pool(("Forebrain", "OlfactoryBulb"))                      # rostral-most forebrain = olfactory bulb (Tbr1)
            if len(op) >= 8:
                oth = np.quantile(a2[op], 0.15); _set(op[a2[op] <= oth], "OlfactoryBulb"); _set(op[a2[op] > oth], "Forebrain")
            cp = _pool(("Connective", "Adipose"))                          # adipose (Pparg) = a scattered connective subtype
            if len(cp) >= 6:
                order = cp[np.argsort(a2[cp] + 0.31 * d2[cp])]; _set(order[::5], "Adipose"); _set(np.setdiff1d(order, order[::5]), "Connective")
        target_v = np.array([VM_OF.get(FATES[j], V_NEUTRAL) if j >= 0 else V_NEUTRAL for j in fid[:born]], np.float32)
        V = vm[:born].copy()
        for _ in range(3):
            gj = 0.15 * (V[nbr].mean(1) - V) if has_nbr else 0.0
            V = V + 0.5 * (target_v - V) + gj
        vm[:born] = V

        # ---- MIGRATION: convergent extension + flat sheet ----
        fnames = [FATES[j] if j >= 0 else None for j in fid[:born]]
        # Wnt-PCP coherent (convergent) extension of the trunk. EXEMPT the limb buds AND the placed paired
        # organs: like the limbs, a paired organ is a placed structure on the LR antinode, so the trunk's
        # mediolateral narrowing must not drag it back to the midline (the audit's systemic collapse).
        trunk = ((a > 0.10) & (a < 0.92) & (fid[:born] != FIDX["Limb Bud"])
                 & ~np.isin(fid[:born], PAIRED_ORGAN_IDS))
        pos[:born, 0] *= 1.006                        # axial elongation (AP)

        # ---- CONVERGENT EXTENSION as a CONTROLLED target width ----
        # Relax the trunk's mediolateral spread toward a Wnt-PCP-set target so the body WIDTH tracks
        # pcp DETERMINISTICALLY (robust to cell count), instead of emerging as a particle-dynamics
        # transient. This is what makes the fish<->tetrapod (limbless<->limbed) threshold reproducible:
        # strong PCP tone -> narrow trunk -> no left-right eigenmode; weak tone -> wide -> LR mode + limbs.
        idx = np.where(trunk)[0]
        if idx.size > 8:
            cur = float(pos[idx, 2].std()) + 1e-6
            target_std = max(0.02, 0.150 - 0.110 * pcp)   # pcp 0.25 -> 0.123 (wide) ; 0.95 -> 0.045 (narrow)
            pos[idx, 2] *= (1.0 + 0.30 * (target_std / cur - 1.0))

        # ---- INTRINSIC DORSO-VENTRAL GIRTH (no atlas silhouette) ----
        # The trunk must build HEIGHT as it elongates, otherwise convergent extension leaves a flat
        # ribbon (DV/AP ~0.26 vs the real embryo's ~0.7). This is the dorsal neural tube + notochord over
        # the ventral gut/heart/liver column giving the body its dorso-ventral bulk. Modelled like the ML
        # target above but on DV: the trunk relaxes toward a genome-set girth (fraction of body length),
        # so the rounded trunk EMERGES from the head and is not warped onto the mouse outline. DV-only, so
        # the fish<->tetrapod ML width criterion and the limb threshold are untouched. When an explicit
        # shape_target is supplied the per-AP silhouette block below owns DV instead of this default.
        if shape_target is None and idx.size > 8 and os.environ.get("NO_GIRTH") != "1":
            Lx0 = float(np.ptp(pos[:born, 0])) + 1e-9
            cy = float(np.median(pos[idx, 1]))
            cur_dv = float(np.percentile(np.abs(pos[idx, 1] - cy), 95) * 2 / Lx0) + 1e-6
            dv_target = 0.50                               # intrinsic trunk girth (frac of length); real ~0.5-0.7
            # Build the girth LATE, as the clock runs down, AFTER the organ founders are placed (the point
            # organs sprout early, prc2~0.42-0.46). Inflating DV during that window stretched the ventral
            # mesenchyme out from under the heart's ventral-midline capture and the (marginal) heart sprout
            # failed. Gating the girth to the late clock lets the organs found in a stable body, then fills
            # the dorso-ventral bulk around them. Also excludes already-committed point organs so a sprouted
            # organ keeps its place while the body wall grows around it.
            gclk = float(np.clip((0.44 - prc2) / 0.34, 0.0, 1.0))  # 0 early (organs sprout), ->1 late (fill girth)
            movable = idx[~np.isin(fid[idx], [FIDX[o] for o in POINT_ORGANS if o in FIDX])]
            if movable.size > 8 and gclk > 0:
                pos[movable, 1] = cy + (pos[movable, 1] - cy) * (1.0 + 0.30 * gclk * (dv_target / cur_dv - 1.0))

        # ---- MIGRATION HEAD: per-AP GIRTH matched to the real MOSTA silhouette ----
        # The motility head gives shape (convergent extension sets the ML width above; here it also
        # matches the per-AP DORSO-VENTRAL girth to the real embryo's silhouette g(a), the shape-training
        # target = a MOSTA anchor like g_K). Applied FORWARD, gated by the CE clock, accumulating over the
        # run -- the outline EMERGES from the head, not a post-hoc warp. DV only, so the fish<->tetrapod ML
        # width criterion is untouched. shape_target = dict(a=[...], g=[...]) girth as fraction of body length.
        if shape_target is not None and idx.size > 30:
            ta = np.asarray(shape_target["a"], float); tg = np.asarray(shape_target["g"], float)
            ce_clk = float(np.clip((prc2 - 0.20) / 0.60, 0.0, 1.0))    # shape set during axis elongation
            if ce_clk > 0:
                nbb = 40
                be = np.linspace(0.0, 1.0, nbb + 1)
                wbin = np.clip(np.digitize(a, be) - 1, 0, nbb - 1)
                Lx = float(np.ptp(pos[:born, 0])) + 1e-9
                for b in range(nbb):
                    mb = np.where((wbin == b) & trunk)[0]
                    if mb.size < 6:
                        continue
                    ac = 0.5 * (be[b] + be[b + 1])
                    tgt = float(np.interp(ac, ta, tg))                # real DV girth here (frac of length)
                    cy = pos[mb, 1].mean()
                    og = float(np.percentile(np.abs(pos[mb, 1] - cy), 95) * 2 / Lx)  # our DV girth here
                    s = float(np.clip(tgt / (og + 1e-9), 0.5, 3.0))
                    # PHASE 1c: loosen the silhouette pull over the HEAD (a<0.32) so the brain proportions
                    # EMERGE from the organizer growth field instead of being warped to the outline.
                    relax = 1.0 if ac > 0.32 else head_shape_relax
                    pos[mb, 1] = cy + (pos[mb, 1] - cy) * (1.0 + 0.22 * ce_clk * relax * (s - 1.0))

        # ---- CONVERGENT EXTENSION as AXIAL REDISTRIBUTION (the true elongation) ----
        # Extension is the flip side of convergence: intercalating cells to the midline SPREADS them
        # ALONG the axis into a uniform rod. The old model only scaled AP uniformly (*1.006), which
        # never de-clumps an anterior cell pile -- so uniform division left the body anterior-heavy.
        # Here CE moves each trunk cell (monotonically, order preserved -> organ AP order intact)
        # toward where it would sit if axial density were UNIFORM, gated by the CE clock (strong early,
        # during axis elongation; fades late). This is the migration head doing the elongation that
        # division must NOT: it flattens the anterior pile without any positional division bias. The
        # ML-width knob above is convergence; this is the coupled extension.
        if idx.size > 30:                                 # extension is AP-only (order-preserving); it
            # never touches ML width, so a strong-PCP fish stays narrow/limbless while ALSO elongating
            # into a proper rod -- indeed the strongest-CE body should be the longest and thinnest.
            x = pos[idx, 0]
            order = np.argsort(x)
            ranks = np.empty(len(order)); ranks[order] = np.linspace(0.0, 1.0, len(order))
            x_uniform = x.min() + ranks * (x.max() - x.min())         # uniform-density AP target
            ce_clock = float(np.clip((prc2 - 0.20) / 0.60, 0.0, 1.0))  # strong early, fades as clock runs down
            # strengthened so the axis fills to a fairly UNIFORM trunk density instead of the anterior pile
            # the anterior-biased division leaves; a populated posterior trunk is what lets the somite-derived
            # spine (vertebrae, ribs, axial muscle, spinal cord) run the full length rather than truncate.
            pos[idx, 0] = x + 0.45 * ce_clock * (x_uniform - x)

        # ---- SHAPE: TAIL -- elongate the posterior into a proper tail (mouse-like). Posterior cells
        # are pushed backward, graded by how posterior they are (a tapering tail), and thinned in DV+ML,
        # accumulating over the run into a long thin tail beyond the last organs.
        tail_prog = float(np.clip((0.50 - prc2) / 0.50, 0, 1))
        tailm = a > 0.85                                                     # a SHORTER tail region (was 0.78)
        if tailm.any():
            pos[:born][tailm, 0] += (a[tailm] - 0.85) * 0.20 * tail_prog     # gentler push -> tail ~14% of axis, not 40%
            pos[:born][tailm, 1] *= (1 - 0.035 * tail_prog)                  # thin the tail DV
            pos[:born][tailm, 2] *= (1 - 0.045 * tail_prog)                  # thin the tail ML

        # ---- SHAPE: dorsal neural fold ----
        fs = float(np.clip((0.52 - prc2) / 0.30, 0, 1))
        neural = np.array([f in ("Forebrain", "Eye", "Nervous System", "Spinal Cord",
                                  "Midbrain", "Hindbrain", "Cerebellum") for f in fnames])
        if fs > 0:
            pos[:born][neural, 2] *= (1 - 0.10 * fs)
            pos[:born][neural, 1] += 0.012 * fs

        # ---- SHAPE: limb-bud OUTGROWTH -- the buds EXTEND into projecting limbs as the clock runs down.
        # The outgrowth is DISTAL-GRADED: a cell already further from the midline grows out more, so each
        # bud stretches into a limb that projects laterally and drops ventrally (a proximodistal axis).
        # It is applied BEFORE the mechanical relaxation below so the integrin/ECM fascia re-knits the
        # just-extended limb to the body WITHIN the same step, instead of a single force pass forever
        # lagging the shove (the cause of the limbs tearing off). Limb COUNT is untouched (set at
        # formation), so the fish stays limbless.
        if limb_buds:
            isbud = np.array([f == "Limb Bud" for f in fnames])
            if isbud.any():
                prog = float(np.clip((0.42 - prc2) / 0.42, 0, 1))
                bz = np.abs(pos[:born][:, 2])
                uu = bz / (bz[isbud].max() + 1e-9)                    # proximal 0 .. distal 1 within the limb
                grow = (lpar["grow_base"] + lpar["grow_slope"] * uu) * prog   # distal cells extend more (proximodistal)
                sgn = np.sign(pos[:born][:, 2] + 1e-9)
                pos[:born][isbud, 2] += sgn[isbud] * lpar["out_lat"] * grow[isbud]   # project laterally
                pos[:born][isbud, 1] -= lpar["out_ven"] * grow[isbud]               # drop ventrally -> a limb

        # ---- EYE MIGRATION: born lateral, then field-driven frontal medialization -> stereoscopic vision
        # (Miles). The optic vesicles are BORN LATERAL on the head (the electric-body LR antinodes); a
        # frontal-organizer morphogen gradient at the frontonasal midline then makes them medialise as the
        # head grows -- the trained trajectory of medic.field_driven_eye (85deg lateral -> 35deg frontal),
        # the medialization that gives primates/humans binocular overlap. Here that whole sequence is placed
        # on the eye CELLS on the head sphere: each step seats the two eyes on the head at the current angle
        # phi from the frontal (anterior) pole, phi carried from 85deg down toward 85-50*eye_frontation over
        # the head-growth clock. eye_frontation in [0,1]: 1 -> ~35deg frontal (human/stereoscopic), 0 -> the
        # eyes are left alone (lateral, like the fish/mouse). The current model seats the eye primordium near
        # the midline, so this term supplies both the lateral BIRTH and the frontal migration.
        if eye_frontation > 0.0 and prc2 <= 0.44:
            eye = np.where(fid[:born] == FIDX["Eye"])[0]
            if len(eye) >= 4:
                Pb = pos[:born]
                headf = [FIDX[n] for n in ("Forebrain", "Eye", "Midbrain", "Hindbrain") if n in FIDX]
                hmask = np.isin(fid[:born], headf)
                hc = Pb[hmask].mean(0) if hmask.sum() >= 8 else Pb[eye].mean(0)
                ant = 1.0 if hc[0] >= Pb[:, 0].mean() else -1.0       # frontal (anterior) pole direction in AP
                Rh = 0.6 * (np.percentile(np.abs(Pb[hmask, 2]), 80) if hmask.sum() >= 8 else 0.10) + 0.05
                clockg = float(np.clip((0.44 - prc2) / 0.34, 0, 1))   # head-growth gate: 0 birth -> 1 grown
                phi = np.radians(85.0 - 50.0 * float(np.clip(eye_frontation, 0, 1)) * clockg)  # 85 -> 35 (human)
                order = np.argsort(Pb[eye, 2])                        # split the pool into a left and right eye
                side = np.empty(len(eye), np.float32)
                side[order[:len(eye) // 2]] = -1.0; side[order[len(eye) // 2:]] = 1.0
                dv = Pb[eye, 1] - Pb[eye, 1].mean()                   # keep each eye a small disc, not a point
                Pb[eye, 0] = hc[0] + ant * Rh * np.cos(phi)           # anterior-pole component (frontal)
                Pb[eye, 2] = side * Rh * np.sin(phi) + 0.15 * dv      # lateral component: converges as phi->small
                Pb[eye, 1] = hc[1] + 0.6 * dv

        # ---- LAPLACIAN LR-ANTINODE ATTRACTION for the paired organs: the LR low mode of the electric body
        # has its two ANTINODES off the midline (its node), and a paired organ belongs on them. The gate only
        # SELECTS antinode cells; here they are ATTRACTED to the antinode -- eased toward |z| = a fraction of
        # the LOCAL body half-width at their AP level (the antinode position of the fundamental LR mode; local,
        # so a head organ goes to the narrow head's edge, not the wide trunk's), sign(z) keeping each side on
        # its own antinode. With coherent extension now exempting these organs, the attraction holds.
        if prc2 <= 0.42:
            _apb = np.clip((_norm(pos[:born][:, 0]) * 24).astype(int), 0, 23)
            _lw = np.ones(24, np.float32)
            for _k in range(24):
                _mk = _apb == _k
                if _mk.sum() > 4:
                    _lw[_k] = np.percentile(np.abs(pos[:born][_mk, 2]), 80) + 1e-6
            for _on in PAIRED_ORGANS:
                if _on not in FIDX:
                    continue
                _oc = np.where(fid[:born] == FIDX[_on])[0]
                if len(_oc) < 4:
                    continue
                _z = pos[:born][_oc, 2]
                _sgn = np.where(_z >= 0, 1.0, -1.0)
                _tgt = 0.70 * _lw[_apb[_oc]]
                pos[:born][_oc, 2] += 0.35 * (_sgn * _tgt - _z)       # ease toward its lateral antinode

        # ---- MECHANICS: repulsion + cadherin (selective) + integrin/ECM fascia (non-selective).
        # Run as a short RELAXATION (RELAX inner iterations on the fixed neighbour list). A single pass
        # cannot re-bind a limb the outgrowth just shoved a step away; iterating lets the fascia pull the
        # tissue back into ONE bound continuum each frame. The same-fate cadherin and the cross-tissue
        # integrin bonds are precomputed once (fate is fixed this step); only geometry re-evaluates.
        fi = fid[:born]
        same = (fi[:, None] == fi[nbr]) & (fi[:, None] >= 0)
        cad = np.minimum(adhc[:born][:, None], adhc[:born][nbr]) * same                 # cadherin: same-fate only
        ecm_bond = 0.5 * (ecmc[:born][:, None] + ecmc[:born][nbr]) if use_ecm else None  # integrin: all neighbours
        # ---- EMT / MET CLOCK (cadherins -> move -> integrins). The genome sequences adhesion through the
        # epithelial-mesenchymal transition and its reverse: EARLY, the EMT factors (Snail/Slug/Twist/Zeb)
        # REPRESS cadherins, so cells release and are motile and the tissue rearranges freely (convergent
        # extension, organ positioning, crest migration); LATE, the reverse transition (MET) re-expresses
        # cadherins and the cells CONDENSE, and the integrin/ECM fascia firms to stabilise the final form.
        # Gated by the differentiation clock (prc2): so we migrate WHILE adhesion is low and let integrins
        # engage AFTER, instead of applying full adhesion throughout. Integrins keep a traction floor during
        # migration (they are the motile cell's grip on the matrix), then firm late.
        if os.environ.get("EMT_OFF") != "1":
            met = float(np.clip((0.44 - prc2) / 0.30, 0.0, 1.0))                       # 0 early (EMT) -> 1 late (MET)
            cad = cad * (0.55 + 0.45 * met)                                            # cadherin softened early, firmer late
            if use_ecm:
                ecm_bond = ecm_bond * (0.72 + 0.28 * met)                              # integrin: traction throughout, firmer late
        lat_damp = max(0.03, 1.0 - 1.0 * pcp ** 2)                                      # Wnt-PCP lateral damping
        for _relax in range(RELAX):
            dvec = pos[:born][nbr] - pos[:born][:, None, :]
            dist = np.linalg.norm(dvec, axis=2) + 1e-9
            u = dvec / dist[..., None]
            rep = np.maximum(R_REP - dist, 0.0)
            force = -K_REP * rep + K_ADH * np.clip(dist - R_REP, 0, R_ADH) * cad
            if use_ecm:
                force = force + K_ECM * np.clip(dist - R_REP, 0, R_ECM) * ecm_bond
            disp = (force[..., None] * u).sum(axis=1)
            disp[:, 2] *= lat_damp                       # damp the LATERAL (ML) spread -> narrow body at high pcp
            pos[:born] += disp

        # FLEXURE HEAD: the cephalo-caudal C-curl, developing LATE (CE clock) toward the target angle.
        # Direction is dorsal-convex (our dorsal = +y, so _flex curls the tail ventrally), magnitude is
        # driven by the same convergent-extension program as the width/elongation, its angle anchored to
        # the real per-stage MOSTA curl. MOSTA-validated: the curl is NOT dorsal over-proliferation
        # (cell-cycle dorsal/ventral 0.87) but a CE + DV-tension fold -> a geometric bend, not diff growth.
        Pc = pos[:born].copy()
        if flexure > 0:
            gate = float(np.clip((frac - 0.35) / 0.65, 0.0, 1.0))    # curl grows late, as the axis folds
            Pc = _flex(Pc, np.radians(flexure) * gate)
        frames.append((born, t_hpf, prc2, Pc, vm[:born].copy(), fid[:born].copy()))
        if verbose and (s % 12 == 0 or s == STEPS - 1):
            print(f"    step {s:2d} t={t_hpf:4.1f} N={born:5d} PRC2={prc2:.2f} AP={np.ptp(P[:,0]):.2f} ML={np.ptp(P[:,2]):.2f}")
    ncomp, het = integrity(pos[:born], fid[:born])
    out_pos = _flex(pos[:born].copy(), np.radians(flexure)) if flexure > 0 else pos[:born].copy()
    return frames, dict(ncomp=ncomp, het=het, n=born, pos=out_pos, fid=fid[:born].copy())


def _symmetrize(P, V, F=None):
    """Bilateral symmetry about the ML midline (z=0): the symmetric bioelectric frame makes both
    sides develop as mirror images. FOLD every cell to one half (|z|) and reflect, so structures
    that grew on EITHER side of the stochastic body -- e.g. a hind limb that happened to form on
    the left -- are kept and mirrored, giving a proper mirror-symmetric embryo. Optionally carries
    a per-cell fate array F through the same reflection."""
    Pf = P.copy()
    Pf[:, 2] = np.abs(P[:, 2])                                    # fold both sides onto z>=0
    mm = Pf[:, 2] > 1e-6                                          # don't duplicate midline cells
    Ps = np.vstack([Pf, Pf[mm] * np.array([1.0, 1.0, -1.0])])
    Vs = np.concatenate([V, V[mm]])
    if F is None:
        return Ps.astype(np.float32), Vs.astype(np.float32)
    Fs = np.concatenate([F, F[mm]])
    return Ps.astype(np.float32), Vs.astype(np.float32), Fs


def _export(frames):
    sym = [_symmetrize(P, V) for (_, _, _, P, V, _) in frames]
    maxn = max(len(s[0]) for s in sym)
    Pf = sym[-1][0]
    c = Pf.mean(0); c[2] = 0.0                                    # keep the midline at z=0
    scale = 1.7 / (0.5 * max(np.ptp(Pf[:, 0]), np.ptp(Pf[:, 1]), np.ptp(Pf[:, 2])))
    out = []
    for (born, t_hpf, prc2, _, _, _), (Ps, Vs) in zip(frames, sym):
        n = len(Ps)
        Q = (Ps - c) * scale
        xyz = np.full((maxn, 3), [0.0, -9999.0, 0.0])
        xyz[:n] = Q
        v = np.full(maxn, VMIN, np.float32); v[:n] = Vs
        out.append(dict(stage=f"{t_hpf:.0f} hpf · N={n} · {generations(t_hpf):.0f} div · PRC2 {prc2:.2f}",
                        n_cells=int(n),
                        xyz=[[round(float(x), 3) for x in p] for p in xyz],
                        vm=[round(float(x), 1) for x in v]))
    doc = dict(display="Zebrafish · unified embryo (all 4 heads, one forward pass)",
               source="grow-from-one-cell: division + telomere/PRC2 differentiation + convergent extension + cadherin sorting + integrin/ECM fascia + neural fold; bilaterally symmetric",
               accent="#7ab8ff", vmin=VMIN, vmax=VMAX, n_points=maxn, open_frame=0,
               setpoints={k.replace(",", ""): float(v) for k, v in LAYER_VM.items()},
               frames=out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w"))
    print(f"\nsaved {OUT}  ({len(out)} frames, grows {frames[0][0]}->{frames[-1][0]} cells, {OUT.stat().st_size/1e6:.1f} MB)")


def main():
    print("simulating WITH integrin/ECM fascia ...")
    frames, m_ecm = simulate(use_ecm=True, verbose=True)
    print(f"  -> {m_ecm['n']} cells: connected components {m_ecm['ncomp']}, cross-tissue (heterotypic) contact fraction {m_ecm['het']:.2f}")
    print("simulating WITHOUT ECM (cadherin sorting only) for the contrast ...")
    _, m_no = simulate(use_ecm=False)
    print(f"  -> {m_no['n']} cells: connected components {m_no['ncomp']}, heterotypic contact fraction {m_no['het']:.2f}")
    print(f"\nFASCIA EFFECT: with the ECM the body is {m_ecm['ncomp']} component(s), cross-tissue binding {m_ecm['het']:.2f};")
    print(f"  without it, cadherin sorting alone gives {m_no['ncomp']} components and {m_no['het']:.2f} -- the fascia is the continuum.")
    _export(frames)


if __name__ == "__main__":
    main()
