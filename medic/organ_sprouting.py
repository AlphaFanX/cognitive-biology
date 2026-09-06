"""
Auto-sprouting: organs FILL the antinode GRID of the body electric, in clock order.
===================================================================================

Miles's law (2026-07-15): organs are not positioned by Vm. They take up the ANTINODES
of the body-electric eigenmodes -- now a 3-axis GRID: AP x DV x LR (see
medic.body_electric_antinodes). The MASTER ORGAN CLOCK (PRC2 / Hox withdrawal) sets the
ORDER; each unlocked organ fills its grid cell:

    AP antinode  (ap_i-th of the AP ladder)   -- antero-posterior level
    DV antinode  (ventral / mid / dorsal)     -- the dorso-ventral standing wave
    LR eigenmode: node = midline, antinode = a side
        midline organ (heart, liver, pancreas, gut)  -> the LR node
        paired  organ (eye, otic, lung)              -> the two LR antinodes

Lateral inhibition = EXCLUSIVITY: a grid cell claimed by one organ cannot be taken by
another. Germ context (committed fate) gates which cells are competent (carries the layer).

RIBS are the SEGMENTED case: not one antinode but one pair per SOMITE (the high-frequency
AP harmonic = the segmentation clock) x the two LR antinodes, at a lateral-dorsal DV level,
recruited from sclerotome/somite. The metameric version of the same antinode law.

ORGAN_SCHEDULE is the organ tier of the head tree (medic.head_tree), clock-ordered.
"""
from __future__ import annotations
import numpy as np

# (name, unlock PRC2, ap_i = which AP antinode, dv band, LR placement, germ context)
ORGAN_SCHEDULE = [
    # Eye: the optic vesicles evaginate LATERALLY from the forebrain, so they must sit out on the sides,
    # not hug the midline neural tube. An explicit lateral lr band (with the physical mediolateral fallback)
    # pushes the two eyes to the lateral edge of the forebrain neuroectoderm; germ stays neural (adding head
    # mesenchyme ballooned the eye medially).
    dict(name="Eye",      unlock=0.46, ap_i=0, dv="mid",     place="paired", lr=(0.26, 0.60), germ={"Forebrain", "Nervous System", "Eye"}),
    dict(name="Otic",     unlock=0.45, ap_i=1, dv="dorsal",  place="paired", lr=(0.16, 0.55), germ={"Neural Crest", "Epidermal", "Nervous System", "Otic"}),
    # Heart unlock 0.42 -> 0.58 (cycle 20, emergence timing): the Carnegie ladder has the primary heart
    # tube at CS09 (prc2 ~0.59, 4.7% of the embryo) -- the heart is the FIRST organ to function; the old
    # 0.42 made it emerge at CS13, four stages late (the composition instrument's CS09-12 zero row).
    dict(name="Heart",    unlock=0.58, ap_i=2, dv="ventral", place="midline", germ={"Mesoderm", "Heart"}),
    # Lung: the lung buds sprout from the VENTRAL foregut endoderm and grow laterally, so it must germinate
    # where the endoderm actually is (ventral, near the midline) -- a paired mid-DV LR-antinode placement
    # found no endoderm cells (they are all ventral-midline) and the lung failed to sprout. An lr band (with
    # the physical fallback) at a ventral level, on foregut/endoderm germ, buds the two lungs off the midline.
    dict(name="Lung",     unlock=0.40, ap_i=2, dv="ventral", place="paired", lr=(0.08, 0.32), germ={"Hypoblast", "Yolk Syncytial Layer", "Liver", "Gut", "Lung"}),
    dict(name="Liver",    unlock=0.38, ap_i=3, dv="ventral", place="midline", germ={"Hypoblast", "Yolk Syncytial Layer", "Liver"}),
    # pancreas buds from the foregut just off the midline, so it does NOT collide with the strict-midline
    # liver (both otherwise snap to the same coarse antinode). A SMALL foregut organ: tight para-midline
    # band + foregut germ (no Mesoderm) so it stays a compact bud, not a greedy scattered mass.
    dict(name="Pancreas", unlock=0.40, ap_i=4, dv="ventral", place="midline", lr=(0.04, 0.16), germ={"Hypoblast", "Yolk Syncytial Layer", "Gut", "Pancreas"}),  # single near-midline organ (dorsal+ventral buds fuse), NOT bilateral -- matches ORGAN_PLAN + the line-13 docstring; the lr= band already places it, so generation is unchanged. unlock BEFORE liver so it claims its foregut cells first
    dict(name="Kidney",   unlock=0.36, ap_i=4, dv="mid",     place="paired",  lr=(0.20, 0.52), germ={"Mesoderm", "Kidney"}),
]
# NOTOCHORD = the axial ORGANISER rod (master TF Brachyury/T) = the LR NODE made physical: the strict
# midline, mid-DV (ventral to the neural tube, dorsal to the gut), spanning the trunk AP. Placed FIRST
# -- it is the real molecular identity of the electric-body midline everything else reads from.
NOTOCHORD = dict(name="Notochord", unlock=0.50, ap_lo=0.14, ap_hi=0.90, dv="mid",
                 germ={"Mesoderm", "Nervous System", "Somite", "Notochord"})  # Somite = midline source posteriorly
# GUT = a ventral-MIDLINE endoderm TUBE spanning several AP antinodes (not the whole ventral width, so a
# thin tube, not a slab); the LR band keeps it near the midline like the real gut.
GUT = dict(name="Gut", unlock=0.34, ap_lo=0.28, ap_hi=0.74, dv="ventral", lr=(0.00, 0.30),
           germ={"Hypoblast", "Yolk Syncytial Layer", "Gut"})
# SEGMENTED bilateral structures: one block per SOMITE (the her1 segmentation clock), PARAXIAL
# (medial to the lateral-plate limbs). Sclerotome -> RIBS (ventro-medial, thoracic); myotome ->
# MUSCLE (dorso-lateral, whole trunk). Periodicity is inherited from the segmentation clock.
# separated mediolaterally: sclerotome (ribs) MEDIAL, myotome (muscle) more LATERAL (both still
# medial to the lateral-plate limbs at |LR|>0.7) -- so they don't compete for the same cells.
RIBS   = dict(name="Rib",    unlock=0.42, ap_lo=0.22, ap_hi=0.66, n_seg=13, dv="mid", dv_tol=0.10,
              lr=(0.14, 0.32), germ={"Somite", "Rib"})   # thoracic only; SCLEROTOME germ only (dropping
#   generic Mesoderm/Neural Crest -- with them the rib grabbed the whole thoracic paraxial region, ~13% of
#   the body, a thick white cartilage band; ribs are thin, so restrict to somite + a narrow DV/LR footprint.
MUSCLE = dict(name="Muscle", unlock=0.40, ap_lo=0.16, ap_hi=0.88, n_seg=20, dv="mid",
              lr=(0.34, 0.70), germ={"Somite", "Mesoderm", "Muscle"})                 # whole-trunk axial muscle
#   LR band widened 0.40-0.66 -> 0.34-0.70 into the unused paraxial gaps (DRG/rib <0.32, limbs >0.7). Real
#   muscle ~10.5% vs our ~5%. Safe now that (a) the LR frame is physically anchored [deterministic] and
#   (b) Connective claims ALL leftover interstitial mesoderm, so the vacated cells no longer get absorbed
#   into the Cartilage column (the earlier ~16% cartilage blow-up).
# CARTILAGE = the segmented AXIAL skeleton (the vertebral column): one block per somite straddling the
# midline around the notochord (the centra), mid-DV, spanning the trunk. Master TF Sox9 (Cartilage
# primordium is the E12.5 MOSTA skeleton head). Runs BEFORE the ribs so the MEDIAL column forms first
# and the ribs project laterally from the remaining sclerotome (medial cartilage < lateral rib < muscle).
CARTIL = dict(name="Cartilage", unlock=0.42, ap_lo=0.16, ap_hi=0.86, n_seg=20, dv="mid", dv_tol=0.12,
              lr=(0.00, 0.12), germ={"Somite", "Mesoderm", "Neural Crest", "Cartilage"})  # midline vertebral column
# DRG = dorsal root ganglia: neural-crest SENSORY ganglia, one pair per somite, DORSOLATERAL (beside the
# dorsal neural tube, medial to the muscle). Master TF Sox10. The segmental PNS sensory head (MOSTA E12.5).
DRG    = dict(name="DRG", unlock=0.40, ap_lo=0.30, ap_hi=0.86, n_seg=16, dv="dorsal",
              lr=(0.14, 0.32), germ={"Neural Crest", "DRG"})
# SYMPATHETIC chain: neural-crest AUTONOMIC ganglia, one pair per somite, VENTRAL and near the midline
# (flanking the dorsal aorta). Master TF Phox2b. The segmental PNS autonomic head (MOSTA E12.5).
SYMPATH = dict(name="Sympathetic", unlock=0.38, ap_lo=0.34, ap_hi=0.82, n_seg=11, dv="ventral",
               lr=(0.00, 0.16), germ={"Neural Crest", "Mesoderm", "Sympathetic"})
# Ordered medial->lateral and by DV so they claim distinct grid cells: axial cartilage column, then the
# dorsolateral DRG and ventromedial sympathetic ganglia, then the lateral ribs and the lateralmost muscle.
SEGMENTS = [CARTIL, DRG, SYMPATH, RIBS, MUSCLE]
# VESSEL = the dorsal aorta + AGM/blood: a midline vessel just ventral of the notochord and dorsal to the
# gut, spanning the trunk AP. Master TF Etv2 / Tal1 (endothelium + haematopoiesis). A midline tube like the
# gut, but higher in the DV stack (MOSTA "Blood vessel" + "AGM" heads).
VESSEL = dict(name="Vessel", unlock=0.44, ap_lo=0.20, ap_hi=0.90, dv=0.40,
              germ={"Mesoderm", "Vessel"})
# --- CRANIOFACIAL / MISC LEAF HEADS (present in the MOSTA series, added 2026-07-18) --------------------
# EXTRA blocks placed AFTER the trunk organs + segments, so they take the cells those did not claim.
# JAW & TOOTH = the craniofacial neural-crest mesenchyme of the first branchial arch (mandible/maxilla):
# ANTERIOR + ventral, off the midline on both sides, from crest/head-mesenchyme. Master TF Msx1 (E12.5
# MOSTA "Jaw and tooth" head, Msx1/Prrx2/Msx2). A paired anterior mass (place='tube' -> both sides).
#   germ deliberately EXCLUDES committed organ/neural fates: `claimed` resets each call, so a germ set
#   that named e.g. Forebrain/Kidney would re-recruit already-committed organ cells on later steps (the
#   theft that collapsed brain/kidney). Only progenitor + loosely-related mesenchyme/crest are competent.
JAW     = dict(name="Jaw", unlock=0.40, ap_lo=0.02, ap_hi=0.16, dv=0.24, dv_tol=0.22,
               lr=(0.10, 0.55), place="tube", germ={"Neural Crest", "Mesoderm", "Hypoblast"})
# CHOROID PLEXUS = the CSF-secreting epithelium in the brain ventricle roof: deep in the ANTERIOR neural
# tissue, dorsal + near the midline. Master TF Rfx2 (E12.5 MOSTA "Choroid plexus"). Small; from neural germ.
CHOROID = dict(name="Choroid", unlock=0.36, ap_lo=0.04, ap_hi=0.24, dv=0.84, dv_tol=0.10,
               lr=(0.00, 0.16), place="tube", germ={"Nervous System"})
# GONAD = the genital ridge on the intermediate mesoderm, just medial/ventral of the kidney in the
# posterior trunk. Master TF Sohlh2 (E12.5 MOSTA "Ovary"/"Gonad" -- the cleanest master in the atlas, 105x).
GONAD   = dict(name="Gonad", unlock=0.34, ap_lo=0.64, ap_hi=0.80, dv=0.38, dv_tol=0.14,
               lr=(0.02, 0.22), place="tube", germ={"Mesoderm", "Hypoblast"})
# BRANCHIAL ARCH = the pharyngeal-arch crest mesenchyme, anterior-ventral, just POSTERIOR to the jaw
# (the arches that build the jaw, ear ossicles, and neck). Master Dlx2. Paired (both sides).
BRANCHIAL = dict(name="Branchial", unlock=0.42, ap_lo=0.18, ap_hi=0.28, dv=0.24, dv_tol=0.09,
                 lr=(0.08, 0.30), place="tube", germ={"Neural Crest", "Mesoderm", "Hypoblast"})
# MESENTERY = the dorsal mesentery suspending the gut: a thin MIDLINE sheet dorsal to the gut tube.
# Master Wt1 (coelomic/serosal). germ excludes the committed Gut (no cross-step theft).
MESENTERY = dict(name="Mesentery", unlock=0.34, ap_lo=0.38, ap_hi=0.68, dv=0.30, dv_tol=0.15,
                 lr=(0.00, 0.18), place="tube", germ={"Mesoderm", "Hypoblast"})
# BLOOD (AGM) = the intra-aortic haematopoietic cells along the dorsal aorta (and fetal liver). A thin
# midline strand at the aortic DV level. Master Runx1. germ = mesoderm only (does not deplete the vessel).
BLOOD = dict(name="Blood", unlock=0.40, ap_lo=0.45, ap_hi=0.82, dv=0.38, dv_tol=0.08,
             lr=(0.00, 0.10), place="tube", germ={"Mesoderm"})
# --- NEW ORGAN HEADS (predicted by AlphaGenome/SEdb, not in MOSTA's 35; grounded in the head registry) ----
# ADRENAL gland: the adrenogonadal mesoderm just medial/superior of the kidney. Master Nr5a1 (SF1).
ADRENAL = dict(name="Adrenal", unlock=0.36, ap_lo=0.58, ap_hi=0.78, dv=0.48, dv_tol=0.16,
               lr=(0.02, 0.28), place="tube", germ={"Mesoderm", "Hypoblast"})
# THYMUS: the 3rd-pharyngeal-pouch epithelium, anterior-ventral (pharyngeal/neck). Master Foxn1.
THYMUS  = dict(name="Thymus", unlock=0.36, ap_lo=0.14, ap_hi=0.26, dv=0.22, dv_tol=0.14,
               lr=(0.04, 0.28), place="tube", germ={"Hypoblast", "Mesoderm"})
# SPLEEN: the dorsal mesogastrium mesenchyme beside the stomach. Master Tlx1/Nkx2-5-independent (Tlx1).
SPLEEN  = dict(name="Spleen", unlock=0.38, ap_lo=0.46, ap_hi=0.60, dv=0.32, dv_tol=0.13,
               lr=(0.10, 0.30), place="tube", germ={"Mesoderm", "Hypoblast"})
# BLADDER: the urogenital-sinus endoderm/mesenchyme, caudal-ventral on the midline. Master Foxa1.
BLADDER = dict(name="Bladder", unlock=0.34, ap_lo=0.78, ap_hi=0.90, dv=0.20, dv_tol=0.14,
               lr=(0.00, 0.16), place="tube", germ={"Hypoblast", "Mesoderm"})
EXTRA = [JAW, CHOROID, GONAD, BRANCHIAL, MESENTERY, BLOOD, ADRENAL, THYMUS, SPLEEN, BLADDER]

_FATES = None
_SEG_CENTERS = {}
_MOSTA = None
_GENOME = None


def _mosta_anchors():
    """Cached real-atlas organ AP/DV anchors (medic.mosta_organ_anchors). {} if unavailable."""
    global _MOSTA
    if _MOSTA is None:
        try:
            from medic.mosta_organ_anchors import organ_anchor_table
            _MOSTA = organ_anchor_table()
        except Exception:
            _MOSTA = {}
    return _MOSTA


def _genome_anchors():
    """Cached GENOME-derived organ AP addresses (medic.genome_organ_address): the Hox-code address
    read through the head layer, the wiring of the genome<->MOSTA bridge. These take priority over the
    fitted MOSTA anchor where present (fitted-to-MOSTA -> read-from-the-genome-code). {} if unavailable."""
    global _GENOME
    if _GENOME is None:
        try:
            from medic.genome_organ_address import genome_organ_ap_table
            _GENOME = genome_organ_ap_table()
        except Exception:
            _GENOME = {}
    return _GENOME


def bind_fates(fates):
    global _FATES
    _FATES = list(fates)


def _somite_centers(lo, hi, n):
    """Segment AP positions = the SEGMENTATION-CLOCK somite centers (her1/Hes clock x wavefront;
    the vertebrate segment-polarity module, von Dassow-Odell-style) -- so rib/muscle periodicity is
    inherited from the somite clock, not hand-set. Cached per (lo,hi,n); falls back to linspace."""
    key = (round(lo, 3), round(hi, 3), int(n))
    if key not in _SEG_CENTERS:
        try:
            from medic.body_plan_morphogenesis import clock_somite_centers
            centers, _ = clock_somite_centers(lo, hi, n)
            _SEG_CENTERS[key] = list(np.asarray(centers, float))
        except Exception:
            _SEG_CENTERS[key] = list(np.linspace(lo, hi, n))
    return _SEG_CENTERS[key]


def _germ_mask(fid, germ):
    m = fid < 0
    if _FATES is not None:
        gi = {i for i, f in enumerate(_FATES) if f in germ}
        for i in gi:
            m |= (fid == i)
    return m


def _dv_target(tag, dv_levels):
    if isinstance(tag, (int, float)):          # explicit DV fraction (e.g. the aorta's mid-ventral level)
        return float(tag)
    if not dv_levels:
        return {"ventral": 0.15, "mid": 0.5, "dorsal": 0.85}[tag]
    return {"ventral": dv_levels[0], "mid": dv_levels[len(dv_levels) // 2],
            "dorsal": dv_levels[-1]}[tag]


def sprout_organs(a, d, lrE, fid, prc2, ap_levels, dv_levels, mln=None,
                  ap_tol=0.06, dv_tol=0.20, lr_gate=0.35):
    """Return {organ_name: indices}. Each unlocked organ fills its (AP, DV, LR) grid
    cell; a claimed cell can't be reused (lateral-inhibition exclusivity).

    `mln` = the PHYSICAL mediolateral coordinate |z| in [0,1]. The LR placement normally reads the
    electric LR eigenmode `lrE`, but that mode LOSES its midline node in the posterior trunk (no ML
    amplitude there), so a midline organ read purely off `lrE` truncates anteriorly. Where `mln` is given,
    every LR band falls back to the PHYSICAL midline/paraxial position -- the sclerotome converging on the
    notochord physically, not on an eigenmode -- so the axial column runs the full length."""
    if not ap_levels:
        return {}
    levels = np.asarray(ap_levels, float)
    nearest = np.argmin(np.abs(a[:, None] - levels[None, :]), axis=1)   # each cell -> nearest AP antinode
    claimed = np.zeros(len(a), bool)
    out = {}

    def _lr(lo, hi):
        """|LR| in [lo,hi] by the electric eigenmode OR (fallback) the physical mediolateral position,
        so the band still selects cells where the LR eigenmode has lost its node (the posterior trunk)."""
        band = (np.abs(lrE) >= lo) & (np.abs(lrE) <= hi)
        if mln is not None:
            band = band | ((mln >= lo) & (mln <= hi))
        return band

    def _place(name, ap_t, dv_t, place, germ, ap_i=None, ap_band=None, lr=None, atol=None, dtol=None):
        at = ap_tol if atol is None else atol
        dt = dv_tol if dtol is None else dtol
        if ap_band is not None:
            near_ap = (a >= ap_band[0]) & (a <= ap_band[1])
        else:
            near_ap = (np.abs(a - ap_t) < at) & (nearest == ap_i)
        near = near_ap & (np.abs(d - dv_t) < dt) & _germ_mask(fid, germ) & ~claimed
        if lr is not None:                                             # explicit |LR| band (e.g. kidney = intermediate mesoderm)
            sel = near & _lr(lr[0], lr[1])
        elif place == "midline":
            sel = near & _lr(0.0, lr_gate)
        elif place == "paired":
            # LATERAL INHIBITION (the same Mexican-hat that resolves the four limb buds): within the
            # bilateral competent band (|LR| past the antinode gate, on both sides), keep the |LR| PEAK on
            # each side and inhibit the cells nearer the midline, so the organ resolves into TWO DISCRETE
            # bilateral buds on the LR antinodes instead of one band the condensation/coherent-extension
            # collapse onto the axis. Numpy-only (no positions here): the |LR| median splits each side, the
            # more-lateral half is the antinode bud, the inner half is inhibited.
            band = near & (np.abs(lrE) >= lr_gate)
            sel = np.zeros(len(a), bool)
            for _sd in (lrE > 0, lrE < 0):
                _bs = band & _sd
                if _bs.sum() >= 3:
                    _med = np.median(np.abs(lrE[_bs]))
                    sel |= _bs & (np.abs(lrE) >= _med)
            if sel.sum() < 4:                                           # too sparse after inhibition -> keep the band
                sel = band
        else:                                                          # 'tube' -> both node and sides
            sel = near
        idx = np.where(sel)[0]
        if len(idx) >= 4:
            out[name] = idx
            claimed[idx] = True
            return True
        return False

    # NOTOCHORD first: the axial organiser rod on the strict LR node (the midline made physical).
    if prc2 <= NOTOCHORD["unlock"]:
        dv_t = _dv_target(NOTOCHORD["dv"], dv_levels)
        # a THIN axial rod (real notochord <1%): tight LR node + tight DV band (was 0.09/0.09 -> ~8%)
        noto = ((a > NOTOCHORD["ap_lo"]) & (a < NOTOCHORD["ap_hi"]) & _lr(0.0, 0.045)
                & (np.abs(d - dv_t) < 0.055) & _germ_mask(fid, NOTOCHORD["germ"]) & ~claimed)
        ii = np.where(noto)[0]
        if len(ii) >= 4:
            out[NOTOCHORD["name"]] = ii; claimed[ii] = True

    # point organs, in clock order. The AP antinode each fills is ANCHORED to the organ's address and
    # snapped to the nearest antinode of the genome-derived ladder. Priority: (1) the GENOME Hox-code
    # address read through the head layer (medic.genome_organ_address -- the wired genome<->MOSTA
    # bridge), (2) the fitted MOSTA anchor for any organ the genome read does not cover, (3) the
    # hand-set ap_i. This is the g_K progression: fitted-to-MOSTA -> read-from-the-genome-code.
    anchors = {**_mosta_anchors(), **_genome_anchors()}
    for org in ORGAN_SCHEDULE:
        if prc2 > org["unlock"]:
            continue
        dv_t = _dv_target(org["dv"], dv_levels)
        placed = False
        if org["name"] in anchors:                                   # snap to the MOSTA-anchored antinode
            ai = int(np.argmin(np.abs(levels - anchors[org["name"]]["ap"])))
            placed = _place(org["name"], levels[ai], dv_t, org["place"], org["germ"], ap_i=ai, lr=org.get("lr"))
        if not placed and org["ap_i"] < len(levels):                 # fallback: hand-set antinode (body too anterior for the MOSTA position)
            placed = _place(org["name"], levels[org["ap_i"]], dv_t, org["place"], org["germ"], ap_i=org["ap_i"], lr=org.get("lr"))
        # ROBUST retry for the marginal HEART sprout: it is a small ventral-midline organ that fails to reach
        # its cell threshold on some seeds/stages (E13.5 seed 0). Widen the AP/DV capture progressively until
        # it catches its ventral mesoderm, so the heart is present on every seed.
        if not placed and org["name"] == "Heart":
            for atol, dtol in [(0.09, 0.28), (0.12, 0.34)]:
                ai = int(np.argmin(np.abs(levels - anchors.get("Heart", {"ap": levels[org["ap_i"]]})["ap"])))
                if _place("Heart", levels[ai], dv_t, "midline", org["germ"], ap_i=ai, atol=atol, dtol=dtol):
                    break

    # GUT tube (ventral midline, spans AP)
    if prc2 <= GUT["unlock"]:
        _place(GUT["name"], None, _dv_target(GUT["dv"], dv_levels), "tube",
               GUT["germ"], ap_band=(GUT["ap_lo"], GUT["ap_hi"]), lr=GUT.get("lr"))

    # VESSEL (dorsal aorta + AGM/blood): a strict-midline tube just ventral of the notochord and dorsal
    # to the gut -- higher in the DV stack than the gut, so it uses its own mid-ventral DV level.
    if prc2 <= VESSEL["unlock"]:
        vt = ((a > VESSEL["ap_lo"]) & (a < VESSEL["ap_hi"]) & _lr(0.0, 0.06)
              & (np.abs(d - _dv_target(VESSEL["dv"], dv_levels)) < 0.055)
              & _germ_mask(fid, VESSEL["germ"]) & ~claimed)
        iv = np.where(vt)[0]
        if len(iv) >= 4:
            out[VESSEL["name"]] = iv; claimed[iv] = True

    # SEGMENTED structures (ribs, muscle): one bilateral block per somite (seg clock), PARAXIAL
    # (medial to the limbs at the LR antinode). Ribs run first (ventro-medial sclerotome), then
    # muscle (dorso-lateral myotome) on the cells ribs did not claim.
    for seg in SEGMENTS:
        if prc2 > seg["unlock"]:
            continue
        dv_t = _dv_target(seg["dv"], dv_levels)
        dvt = seg.get("dv_tol", dv_tol)                               # per-segment DV width (centra = thin)
        germ = _germ_mask(fid, seg["germ"])
        band = _lr(seg["lr"][0], seg["lr"][1])                         # electric OR physical LR band
        midline_seg = seg["lr"][0] == 0.0                             # the vertebral centra are ONE midline block
        seg_idx = []
        for ap_t in _somite_centers(seg["ap_lo"], seg["ap_hi"], seg["n_seg"]):
            ap_hit = (np.abs(a - ap_t) < 0.025) & (np.abs(d - dv_t) < dvt) & band & germ & ~claimed
            sides = [np.ones(len(a), bool)] if midline_seg else [lrE > 0, lrE < 0]
            for side in sides:                                        # midline: one block; paraxial: left/right
                ii = np.where(ap_hit & side)[0]
                if len(ii) >= 3:
                    seg_idx.append(ii); claimed[ii] = True
        if seg_idx:
            out[seg["name"]] = np.concatenate(seg_idx)

    # CRANIOFACIAL / MISC leaf heads (jaw, choroid plexus, gonad): placed last on an AP band, taking the
    # cells the trunk organs and segments left free (lateral-inhibition exclusivity via `claimed`).
    for ex in EXTRA:
        if prc2 > ex["unlock"]:
            continue
        _place(ex["name"], None, _dv_target(ex["dv"], dv_levels), ex["place"], ex["germ"],
               ap_band=(ex["ap_lo"], ex["ap_hi"]), lr=ex.get("lr"))
    return out
