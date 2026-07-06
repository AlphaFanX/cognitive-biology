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

HONEST SCOPE: synthetic, schematic geometry and reduced mechanics (the frontier-demo class), not
the real atlas. The rendered cells are a subsample of the true count the clock tracks (~1k->~57k).

Writes data/movie/zebrafish_unified_frames.json.  Run: python -m medic.unified_embryo
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
       "Proliferative Like Cell": 0.30}
# integrin/ECM (fascia) per fate (NON-SELECTIVE, all neighbours): mesenchymal, so mesoderm/crest
# high, epithelia low -- the complement of the cadherins -> binds the body into one continuum.
ECM = {"Mesoderm": 1.00, "Somite": 0.90, "Neural Crest": 0.85, "Hypoblast": 0.60,
       "Yolk Syncytial Layer": 0.50, "Epidermal": 0.40, "Forebrain": 0.30, "Eye": 0.30,
       "Nervous System": 0.35, "Spinal Cord": 0.35, "Blastodisc": 0.45, "Proliferative Like Cell": 0.45}
FATES = list(ADH.keys())
FIDX = {f: i for i, f in enumerate(FATES)}


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


def simulate(use_ecm=True, seed=0, verbose=False):
    rng = np.random.RandomState(seed)
    pos = np.zeros((N_END, 3), np.float32)
    vm = np.full(N_END, V_NEUTRAL, np.float32)
    fid = np.full(N_END, -1, np.int32)
    adhc = np.zeros(N_END, np.float32)
    ecmc = np.zeros(N_END, np.float32)
    born = N_START
    pos[:born, 0] = rng.uniform(-0.22, 0.22, born)
    pos[:born, 1] = rng.uniform(-0.14, 0.14, born)
    pos[:born, 2] = rng.uniform(-0.14, 0.14, born)

    R_REP, R_ADH, R_ECM = 0.032, 0.060, 0.088
    K_REP, K_ADH, K_ECM = 0.55, 0.22, 0.10
    frames = []
    for s in range(STEPS):
        frac = s / (STEPS - 1)
        t_hpf = T0 + (T1 - T0) * frac
        prc2 = prc2_div(t_hpf)

        # ---- DIVISION ----
        target = int(N_START + (N_END - N_START) * frac ** 1.3)
        if target > born:
            P = pos[:born]
            a = _norm(P[:, 0]); d = _norm(P[:, 1])
            prolif = 0.3 + 0.6 * (d > 0.6) + 1.0 * (a > 0.80) + 0.15 * ((d > 0.35) & (d < 0.6))
            prolif /= prolif.sum()
            n_add = min(target - born, N_END - born)
            par = rng.choice(born, n_add, p=prolif)
            off = rng.normal(0, 0.030, (n_add, 3)).astype(np.float32)
            off[:, 1] *= 0.55; off[:, 2] *= 0.55
            off[_norm(pos[par, 0]) > 0.78, 0] += 0.085
            pos[born:born + n_add] = pos[par] + off
            vm[born:born + n_add] = V_NEUTRAL
            fid[born:born + n_add] = -1
            born += n_add

        P = pos[:born]
        nbr = cKDTree(P).query(P, k=min(11, born))[1][:, 1:]
        a = _norm(P[:, 0]); d = _norm(P[:, 1]); mln = np.abs(P[:, 2]) / (np.abs(P[:, 2]).max() + 1e-6)

        # ---- DIFFERENTIATION: commit on unlock, keep ----
        for i in np.where(fid[:born] < 0)[0]:
            f = fate_of(a[i], d[i], mln[i], prc2)
            if f is not None:
                fid[i] = FIDX[f]; adhc[i] = ADH[f]; ecmc[i] = ECM[f]
        target_v = np.array([LAYER_VM[FATES[j]] if j >= 0 else V_NEUTRAL for j in fid[:born]], np.float32)
        V = vm[:born].copy()
        for _ in range(3):
            V = V + 0.5 * (target_v - V) + 0.15 * (V[nbr].mean(1) - V)
        vm[:born] = V

        # ---- MIGRATION: convergent extension + flat sheet ----
        fnames = [FATES[j] if j >= 0 else None for j in fid[:born]]
        is_axial = np.array([f in ("Mesoderm", "Nervous System", "Spinal Cord") for f in fnames])
        ce = is_axial & (a > 0.12) & (a < 0.94)
        pos[:born][ce, 2] *= 0.90
        pos[:born, 0] *= 1.006
        pos[:born, 1] *= 0.99

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
        pos[:born] += (force[..., None] * u).sum(axis=1)

        # ---- SHAPE: dorsal neural fold ----
        fs = float(np.clip((0.52 - prc2) / 0.30, 0, 1))
        neural = np.array([f in ("Forebrain", "Eye", "Nervous System", "Spinal Cord") for f in fnames])
        if fs > 0:
            pos[:born][neural, 2] *= (1 - 0.10 * fs)
            pos[:born][neural, 1] += 0.012 * fs

        frames.append((born, t_hpf, prc2, pos[:born].copy(), vm[:born].copy()))
        if verbose and (s % 12 == 0 or s == STEPS - 1):
            print(f"    step {s:2d} t={t_hpf:4.1f} N={born:5d} PRC2={prc2:.2f} AP={np.ptp(P[:,0]):.2f} ML={np.ptp(P[:,2]):.2f}")
    ncomp, het = integrity(pos[:born], fid[:born])
    return frames, dict(ncomp=ncomp, het=het, n=born, pos=pos[:born].copy(), fid=fid[:born].copy())


def _symmetrize(P, V):
    """Bilateral symmetry about the ML midline (z=0): the symmetric bioelectric frame makes both
    sides develop as mirror images. Keep the right half (z>=0) and reflect it to the left, so the
    body renders as a proper mirror-symmetric embryo instead of a stochastically asymmetric one."""
    right = P[:, 2] >= 0.0
    Pr, Vr = P[right], V[right]
    mm = Pr[:, 2] > 1e-6                                          # don't duplicate midline cells
    Ps = np.vstack([Pr, Pr[mm] * np.array([1.0, 1.0, -1.0])])
    Vs = np.concatenate([Vr, Vr[mm]])
    return Ps.astype(np.float32), Vs.astype(np.float32)


def _export(frames):
    sym = [_symmetrize(P, V) for (_, _, _, P, V) in frames]
    maxn = max(len(s[0]) for s in sym)
    Pf = sym[-1][0]
    c = Pf.mean(0); c[2] = 0.0                                    # keep the midline at z=0
    scale = 1.7 / (0.5 * max(np.ptp(Pf[:, 0]), np.ptp(Pf[:, 1]), np.ptp(Pf[:, 2])))
    out = []
    for (born, t_hpf, prc2, _, _), (Ps, Vs) in zip(frames, sym):
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
