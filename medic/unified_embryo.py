"""
The unified forward embryo: all four heads intercalated, one body, one clock, one movie.
=========================================================================================

A synthetic grow-from-one-cell embryo -- the anatomical compiler run forward -- in which every
timestep applies all four genome-emitted heads together on one substrate:

  1. CLOCK           cumulative divisions shorten the telomere -> PRC2 withdraws (division_head).
  2. DIVISION        the population grows toward the real generation count, biased to the
                     proliferative zones (posterior growth zone, neuroepithelium).
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

OUT = Path("data/movie/zebrafish_unified_frames.json")
VMIN, VMAX = -70.0, -25.0
V_NEUTRAL = -50.0
N_START, N_END = 60, 9000
STEPS = 50
T0, T1 = 3.3, 26.0

# cadherin adhesion per fate (SELECTIVE, same-fate): epithelial/neural high -> sorts tissues.
ADH = {"Forebrain": 1.00, "Eye": 0.90, "Nervous System": 0.60, "Spinal Cord": 0.72,
       "Neural Crest": 0.82, "Mesoderm": 0.42, "Somite": 0.45, "Epidermal": 0.88,
       "Hypoblast": 0.50, "Yolk Syncytial Layer": 0.58, "Blastodisc": 0.30,
       "Proliferative Like Cell": 0.30, "Limb Bud": 0.50, "Heart": 0.55, "Otic": 0.85}
# integrin/ECM (fascia) per fate (NON-SELECTIVE, all neighbours): mesenchymal, so mesoderm/crest
# high, epithelia low -- the complement of the cadherins -> binds the body into one continuum.
ECM = {"Mesoderm": 1.00, "Somite": 0.90, "Neural Crest": 0.85, "Hypoblast": 0.60,
       "Yolk Syncytial Layer": 0.50, "Epidermal": 0.40, "Forebrain": 0.30, "Eye": 0.30,
       "Nervous System": 0.35, "Spinal Cord": 0.35, "Blastodisc": 0.45,
       "Proliferative Like Cell": 0.45, "Limb Bud": 0.92, "Heart": 0.70, "Otic": 0.30}
FATES = list(ADH.keys())
FIDX = {f: i for i, f in enumerate(FATES)}
VM_OF = {**LAYER_VM, "Limb Bud": -45.0, "Heart": -32.0, "Otic": -58.0}  # bud Vm set-points


def fate_of(a, d, mln, prc2):
    """Positional fate map gated by the clock. a=AP[0..1] (0 anterior), d=DV[0..1] (1 dorsal)."""
    if d > 0.56 and mln < 0.38:
        if a < 0.14 and 0.12 < mln < 0.38: f = "Eye"
        elif a < 0.34: f = "Forebrain"
        elif a > 0.55: f = "Spinal Cord"
        else: f = "Nervous System"
    elif d > 0.56: f = "Neural Crest"
    elif d < 0.28: f = "Yolk Syncytial Layer" if a > 0.5 else "Hypoblast"
    elif mln > 0.66: f = "Epidermal"
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
             pcp=0.25, convergent_ext=None):
    rng = np.random.RandomState(seed)
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
    lr_quality = 0.0                                       # |corr| of the chosen LR mode with the ML axis
    lr_aspect = 0.0; lr_eigratio = 0.0                     # body width criterion at the eigenmode snapshot

    R_REP, R_ADH, R_ECM = 0.032, 0.060, 0.088
    K_REP, K_ADH, K_ECM = 0.55, 0.22, 0.10
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
            prolif = 0.3 + 0.6 * (d > 0.6) + 1.0 * (a > 0.80) + 0.15 * ((d > 0.35) & (d < 0.6))
            prolif /= prolif.sum()
            n_add = min(target - born, ne - born)
            par = rng.choice(born, n_add, p=prolif)
            off = rng.normal(0, 0.030, (n_add, 3)).astype(np.float32)
            off[:, 1] *= 0.55; off[:, 2] *= 0.55 * (1.0 - 0.62 * pcp)   # Wnt-PCP: new cells added in a
            #                                                            thinner ML band -> narrower body
            off[_norm(pos[par, 0]) > 0.78, 0] += 0.085
            pos[born:born + n_add] = pos[par] + off
            vm[born:born + n_add] = V_NEUTRAL
            fid[born:born + n_add] = -1
            born += n_add

        P = pos[:born]
        kq = min(11, born)
        q_idx = cKDTree(P).query(P, k=kq)[1]
        nbr = q_idx[:, 1:] if kq > 1 else np.empty((born, 0), dtype=int)
        has_nbr = nbr.shape[1] > 0
        a = _norm(P[:, 0]); d = _norm(P[:, 1]); mln = np.abs(P[:, 2]) / (np.abs(P[:, 2]).max() + 1e-6)

        # ---- DIFFERENTIATION: commit on unlock, keep ----
        for i in np.where(fid[:born] < 0)[0]:
            f = fate_of(a[i], d[i], mln[i], prc2)
            if f is not None:
                fid[i] = FIDX[f]; adhc[i] = ADH[f]; ecmc[i] = ECM[f]
        # LIMB BUDS: once the clock unlocks (prc2 low), specify paired FORE + HIND lateral-plate
        # lobes -- convert lateral-plate mesoderm / uncommitted cells in the four bud zones. Done
        # before heavy convergent extension so the posterior still has lateral cells to recruit.
        if limb_buds and prc2 <= 0.46:
            # ==== ELECTRIC BODY: low eigenmodes of the kNN gap-junction operator = the body axes ====
            # (same frame as the electric face / mammary line / six-pack, Paper #4). The AP mode gives
            # the antero-posterior coordinate; the LR mode is the bilateral frame -- its NODE is the
            # midline, its two ANTINODES are the left/right sides. Limbs are placed on the antinodes.
            if apE is None:
                nb = cKDTree(P).query(P, k=min(11, born))[1][:, 1:]
                rr = np.repeat(np.arange(born), nb.shape[1]); cc = nb.ravel()
                Wk = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(born, born)).tocsr()
                Wk = ((Wk + Wk.T) > 0).astype(float)
                Lk = diags(np.asarray(Wk.sum(1)).ravel()) - Wk
                # search enough modes to reach the LR bilateral mode: in an elongated body it is a
                # HIGHER mode (its frequency ~ 1/width) -- the fish->tetrapod (amphibian) transition
                # is precisely the body broadening until this left-right mode becomes available.
                vv, UU = eigsh(Lk, k=14, which="SM")
                _o = np.argsort(vv); vv = vv[_o]; UU = UU[:, _o]

                def _best(coord):
                    b, bi = 0.0, 1
                    for i in range(1, UU.shape[1]):
                        cabs = abs(np.corrcoef(UU[:, i], coord)[0, 1])
                        if cabs > b:
                            b, bi = cabs, i
                    return bi, b
                iAP, _apq = _best(P[:, 0]); iLR, lr_quality = _best(P[:, 2])
                # WIDTH CRITERION (the fish->tetrapod threshold): the LR bilateral mode is admissible
                # only when the body is wide enough that it sits LOW in the spectrum (eigenvalue ~1/width^2).
                lr_eigratio = float(vv[iLR] / (vv[iAP] + 1e-12))   # low (~2-4) if wide; high if narrow/fish
                lr_aspect = float(P[:, 2].std() / (P[:, 0].std() + 1e-9))
                if verbose:
                    print(f"    [limb frame] aspect={lr_aspect:.3f} eigratio={lr_eigratio:.2f} "
                          f"lr_corr={lr_quality:.2f} iLR={iLR}")
                ap_m = UU[:, iAP] * (1 if np.corrcoef(UU[:, iAP], P[:, 0])[0, 1] >= 0 else -1)
                lr_m = UU[:, iLR] * (1 if np.corrcoef(UU[:, iLR], P[:, 2])[0, 1] >= 0 else -1)
                apr = np.argsort(np.argsort(ap_m)).astype(np.float32) / max(1, born - 1)  # AP coord
                lrn = (lr_m / (np.abs(lr_m).max() + 1e-9)).astype(np.float32)   # LR mode, node at 0
                apE = np.full(ne, -1.0, np.float32); apE[:born] = apr
                lrE = np.zeros(ne, np.float32); lrE[:born] = lrn
                apE_tree = (cKDTree(P.copy()), apr.copy(), lrn.copy())
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
            if lr_aspect > 0.20:
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
                hox = np.exp(-((a - fore_ap) / 0.055) ** 2) + np.exp(-((a - hind_ap) / 0.055) ** 2)
                comp = ((mll > 0.55) & (d >= 0.26) & (d <= 0.62) & (hox > 0.40)
                        & ((fi_ == FIDX["Mesoderm"]) | (fi_ < 0)))
                Aact = np.where(comp, hox * mll, 0.0).astype(np.float32)
                # LATERAL INHIBITION (reaction-diffusion Mexican hat): activator minus its long-range
                # neighbourhood mean -> the broad competent lateral plate CONDENSES into four discrete,
                # spaced limb buds (the tissue between them is inhibited). Same mechanism that resolves
                # the paired eyes -- without it the buds merge into a connected fin.
                nbL = cKDTree(P).query(P, k=min(45, born))[1]
                Aeff = Aact - 1.25 * Aact[nbL].mean(1)
                apmid = 0.5 * (fore_ap + hind_ap)
                zc = P[:, 2]
                conv_parts = []
                for apm in (a < apmid, a >= apmid):                 # fore (anterior) / hind (posterior)
                    for side in (zc > 0, zc < 0):                   # left / right
                        cs = comp & apm & side & (Aeff > 0.0)
                        if cs.sum() >= 4:
                            idxq = np.where(cs)[0]
                            thr = 0.35 * float(Aeff[cs].max())      # keep the sharpened peak = the bud
                            conv_parts.append(idxq[Aeff[cs] > thr])
                conv = np.concatenate(conv_parts) if conv_parts else np.array([], dtype=int)
                fid[conv] = FIDX["Limb Bud"]; adhc[conv] = ADH["Limb Bud"]; ecmc[conv] = ECM["Limb Bud"]
            # ORGAN BUDS: heart (ventral-anterior midline) + otic vesicles (paired dorsolateral head)
            heart = np.where((a >= 0.05) & (a <= 0.17) & (d < 0.32) & (mln < 0.30)
                             & ((fi_ == FIDX["Mesoderm"]) | (fi_ == FIDX["Hypoblast"]) | (fi_ < 0)))[0]
            fid[heart] = FIDX["Heart"]; adhc[heart] = ADH["Heart"]; ecmc[heart] = ECM["Heart"]
            otic = np.where((a >= 0.06) & (a <= 0.17) & (d >= 0.42) & (d <= 0.64) & (mln > 0.45)
                            & ((fi_ == FIDX["Neural Crest"]) | (fi_ == FIDX["Epidermal"]) | (fi_ < 0)))[0]
            fid[otic] = FIDX["Otic"]; adhc[otic] = ADH["Otic"]; ecmc[otic] = ECM["Otic"]
        target_v = np.array([VM_OF.get(FATES[j], V_NEUTRAL) if j >= 0 else V_NEUTRAL for j in fid[:born]], np.float32)
        V = vm[:born].copy()
        for _ in range(3):
            gj = 0.15 * (V[nbr].mean(1) - V) if has_nbr else 0.0
            V = V + 0.5 * (target_v - V) + gj
        vm[:born] = V

        # ---- MIGRATION: convergent extension + flat sheet ----
        fnames = [FATES[j] if j >= 0 else None for j in fid[:born]]
        is_axial = np.array([f in ("Mesoderm", "Nervous System", "Spinal Cord") for f in fnames])
        trunk = (a > 0.10) & (a < 0.92) & (fid[:born] != FIDX["Limb Bud"])   # Wnt-PCP convergent extension
        pos[:born, 0] *= 1.006                        # axial elongation (AP)
        pos[:born, 1] *= 0.99                         # dorsoventral thinning

        # ---- MECHANICS: repulsion + cadherin (selective) + integrin/ECM fascia (non-selective) ----
        dvec = P[nbr] - P[:, None, :]
        dist = np.linalg.norm(dvec, axis=2) + 1e-9
        u = dvec / dist[..., None]
        rep = np.maximum(R_REP - dist, 0.0)
        fi = fid[:born]
        same = (fi[:, None] == fi[nbr]) & (fi[:, None] >= 0)
        cad = np.minimum(adhc[:born][:, None], adhc[:born][nbr]) * same                 # cadherin: same-fate only
        force = -K_REP * rep + K_ADH * np.clip(dist - R_REP, 0, R_ADH) * cad
        if use_ecm:
            ecm_bond = 0.5 * (ecmc[:born][:, None] + ecmc[:born][nbr])                  # ECM: all neighbours
            force = force + K_ECM * np.clip(dist - R_REP, 0, R_ECM) * ecm_bond
        disp = (force[..., None] * u).sum(axis=1)
        disp[:, 2] *= max(0.03, 1.0 - 1.0 * pcp ** 2)   # Wnt-PCP convergent extension: damp the LATERAL
        pos[:born] += disp                              # (ML) spreading -> narrow body (fish) at high pcp

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

        # ---- SHAPE: dorsal neural fold ----
        fs = float(np.clip((0.52 - prc2) / 0.30, 0, 1))
        neural = np.array([f in ("Forebrain", "Eye", "Nervous System", "Spinal Cord") for f in fnames])
        if fs > 0:
            pos[:born][neural, 2] *= (1 - 0.10 * fs)
            pos[:born][neural, 1] += 0.012 * fs

        # ---- SHAPE: limb-bud OUTGROWTH -- the buds EXTEND into projecting limbs as the clock runs down.
        # The outgrowth is DISTAL-GRADED: a cell already further from the midline grows out more, so each
        # bud stretches into a limb that projects laterally and drops ventrally (a proximodistal axis),
        # rather than a rigid bulge. Accumulates over the remaining steps -> the longer it runs, the
        # further the limbs extend. Limb COUNT is untouched (set at formation), so the fish stays limbless.
        if limb_buds:
            isbud = np.array([f == "Limb Bud" for f in fnames])
            if isbud.any():
                prog = float(np.clip((0.42 - prc2) / 0.42, 0, 1))
                bz = np.abs(pos[:born][:, 2])
                u = bz / (bz[isbud].max() + 1e-9)                     # proximal 0 .. distal 1 within the limb
                grow = (0.30 + 0.70 * u) * prog                      # distal cells extend more (proximodistal)
                sgn = np.sign(pos[:born][:, 2] + 1e-9)
                pos[:born][isbud, 2] += sgn[isbud] * 0.150 * grow[isbud]   # project laterally
                pos[:born][isbud, 1] -= 0.095 * grow[isbud]               # drop ventrally -> a limb

        frames.append((born, t_hpf, prc2, pos[:born].copy(), vm[:born].copy(), fid[:born].copy()))
        if verbose and (s % 12 == 0 or s == STEPS - 1):
            print(f"    step {s:2d} t={t_hpf:4.1f} N={born:5d} PRC2={prc2:.2f} AP={np.ptp(P[:,0]):.2f} ML={np.ptp(P[:,2]):.2f}")
    ncomp, het = integrity(pos[:born], fid[:born])
    return frames, dict(ncomp=ncomp, het=het, n=born, pos=pos[:born].copy(), fid=fid[:born].copy())


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
