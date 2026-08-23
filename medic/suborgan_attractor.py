"""suborgan_attractor.py -- the heart-chamber relational attractor, GENERALIZED to any suborgan CHAIN (tube).

The heart proof (medic/heart_chamber_attractor.py) showed a coordinate cut fails on a looped tube while a
relational attractor -- geodesic (position) on the Physiome-weighted gap-junction operator + the stored
developmental chain (Gray's form) + constriction landmarks -- recovers it. That tool is topology-specific:
it is the CHAIN/TUBE family (a 1-D sequence along a curved axis). This module makes it spec-driven so it
drops onto every tube/chain suborgan:

    GUT      esophagus -> stomach -> small intestine -> colon -> rectum   (the coiled midgut = the hard case)
    NEPHRON  glomerulus -> proximal -> loop of Henle -> distal -> duct    (the loop of Henle = a hairpin)
    BRAIN    forebrain -> midbrain -> hindbrain -> spinal   (the neuraxis; cerebellum is a BRANCH = family 2)
    COCHLEA / ear canal / oviduct / epididymis ...

A ChainSpec = (section names, boundary fractions, section centres, per-section g_gj). Point it at the organ's
cells; it returns the subhead assignment. Compared vs the straight-AP coordinate cut on ground truth.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.suborgan_attractor [gut|heart]
Out:  data/organ_cascade/suborgan_attractor_<name>.{json,png}
"""
from __future__ import annotations
import os, sys, json
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.gut_tube import gut_centerline
from medic.heart_loop import loop_heart, CHAMBER_COL

RNG = np.random.default_rng(0)


def _gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


# ============================================================================= spec
@dataclass
class ChainSpec:
    name: str
    chain: list                 # section names, venous/proximal -> arterial/distal
    fract: np.ndarray           # boundary fractions along the tube (len = n_sections - 1)
    centers: np.ndarray         # section-centre fractions (len = n_sections)
    ggj: dict                   # per-section g_gj (Physiome; relative)
    rad: np.ndarray             # relative radius per section (the stored form signature -> orientation anchor)
    col: dict                   # per-section colour
    align_pow: float = 0.0      # >0 = the LUMINAL (tangent-consistent) operator, for self-approaching tubes (hairpins)


# ============================================================================= operator + geodesic
def gj_laplacian(pts, node_g=None, lab_idx=None, gate_low=0.04, k=10, align_pow=0.0, align_floor=0.15):
    """kNN gap-junction Laplacian. With align_pow>0 (the LUMINAL operator): each edge is scaled by how PARALLEL
    its displacement is to the local tube tangent (PCA of the neighbourhood). Along-tube edges (parallel) keep
    full weight; cross-limb / cross-wall edges (perpendicular) are down-weighted -- so the geodesic follows the
    lumen through a hairpin instead of short-circuiting across touching limbs."""
    n = len(pts); k = min(k, n - 1)
    d, idx = cKDTree(pts).query(pts, k=k + 1)
    sig = np.median(d[:, 1:]) + 1e-9
    if node_g is None:
        node_g = np.ones(n)
    T = None
    if align_pow > 0:
        T = np.zeros((n, pts.shape[1]))
        for i in range(n):
            nb = pts[idx[i, 1:]] - pts[idx[i, 1:]].mean(0)
            T[i] = np.linalg.svd(nb, full_matrices=False)[2][0]     # local tube tangent (unsigned)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j, dist in zip(idx[i, 1:], d[i, 1:]):
            g = np.sqrt(max(node_g[i], 1e-6) * max(node_g[j], 1e-6))
            w = g * np.exp(-(dist / sig) ** 2)
            if T is not None:
                dv = pts[j] - pts[i]; align = abs(dv @ T[i]) / (np.linalg.norm(dv) + 1e-9)   # 1 along, 0 perp
                w *= align_floor + (1 - align_floor) * align ** align_pow
            if lab_idx is not None and abs(int(lab_idx[i]) - int(lab_idx[j])) > 1:
                w *= gate_low                                       # identity gate: no shortcut across coils
            rows += [i, j]; cols += [j, i]; vals += [w, w]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


def fiedler(L):
    try:
        vals, vecs = eigsh(L.astype(float), k=3, sigma=1e-8, which="LM")
        v = vecs[:, np.argsort(vals)][:, 1]
    except Exception:
        vals, vecs = np.linalg.eigh(L.toarray())
        v = vecs[:, np.argsort(vals)][:, 1]
    return (v - v.min()) / (np.ptp(v) + 1e-9)


def orient(s, pts, sp_):
    """Orient the geodesic by matching the tube's RADIUS SIGNATURE to the stored per-section template (e.g. the
    stomach bulge sits near the proximal end). AP poles are unreliable on a folded/coiled tube; the radius
    signature is not. Flip s if it matches the reversed template better."""
    sbin, prof = radius_along_s(pts, s)
    edges = np.quantile(s, np.concatenate([[0.0], sp_.fract, [1.0]]))
    sec = np.array([prof[(sbin >= edges[i]) & (sbin <= edges[i + 1])].mean() if
                    ((sbin >= edges[i]) & (sbin <= edges[i + 1])).any() else np.nan
                    for i in range(len(sp_.chain))])
    ok = ~np.isnan(sec)
    if ok.sum() >= 2 and np.std(sec[ok]) > 1e-9:
        tmpl = np.asarray(sp_.rad, float)
        cf = np.corrcoef(sec[ok], tmpl[ok])[0, 1]
        cr = np.corrcoef(sec[ok], tmpl[::-1][ok])[0, 1]
        if cr > cf:
            return 1.0 - s
    return s


def segment(s, sp_, means=None):
    chain = sp_.chain
    if means is None:
        bnd = np.quantile(s, sp_.fract)
    else:
        m = np.sort(means); bnd = (m[:-1] + m[1:]) / 2
    idx = np.searchsorted(bnd, s)
    return np.array([chain[i] for i in idx]), idx


def luminal_radius(pts, k=10):
    """Per-cell LUMEN radius = perpendicular distance to the LOCAL tube tangent line (PCA of the neighbourhood).
    A local measure -> immune to the tube doubling back (unlike the spread about an s-bin centroid, which reads
    the gap between two limbs of a hairpin as 'radius')."""
    k = min(k, len(pts) - 1)
    _, idx = cKDTree(pts).query(pts, k=k + 1)
    r = np.zeros(len(pts))
    for i in range(len(pts)):
        nb = pts[idx[i, 1:]]; c = nb.mean(0)
        T = np.linalg.svd(nb - c, full_matrices=False)[2][0]
        v = pts[i] - c
        r[i] = np.linalg.norm(v - (v @ T) * T)                   # perpendicular distance to the tangent line
    return r


def radius_along_s(pts, s, nbin=44, smooth=3):
    r = luminal_radius(pts)
    edges = np.quantile(s, np.linspace(0, 1, nbin + 1)); edges[-1] += 1e-9
    binid = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, nbin - 1)
    sbin = np.zeros(nbin); prof = np.full(nbin, np.nan)
    for b in range(nbin):
        m = binid == b
        if m.sum() >= 3:
            prof[b] = r[m].mean(); sbin[b] = s[m].mean()
        else:
            sbin[b] = 0.5 * (edges[b] + edges[b + 1])
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(nbin), np.where(ok)[0], prof[ok])
    prof = np.convolve(prof, np.ones(smooth) / smooth, mode="same")
    return sbin, prof


def snap_boundaries(s, sp_, sbin, prof, win=0.08, depth=0.9):
    """Boundary = the stored developmental fraction, REFINED to a local radius minimum (constriction) ONLY where
    a genuine dip exists (prof < depth x local mean). Featureless sections (thin esophagus / small intestine)
    keep the stored fraction; ballooned ones (chambers, stomach) snap to the real constriction. General across
    the chain family -- does not require a bulge per section (which the watershed did)."""
    span = np.ptp(sbin) + 1e-9
    base = np.quantile(s, sp_.fract)
    lo, hi = np.quantile(s, 0.12), np.quantile(s, 0.88)          # exclude the pole tapers (radius->0 at the tips)
    out = []
    for b in base:
        if b < lo or b > hi:
            out.append(b); continue
        near = np.where(np.abs(sbin - b) <= win * span)[0]
        cand = None
        if len(near) >= 3:
            j = near[np.argmin(prof[near])]
            if near[0] < j < near[-1] and prof[j] < depth * np.mean(prof[near]):   # a BRACKETED real dip
                cand = sbin[j]
        out.append(cand if cand is not None else b)
    return np.sort(np.array(out))


def segment_bnds(s, sp_, bnds):
    idx = np.searchsorted(bnds, s)
    return np.array([sp_.chain[i] for i in idx]), idx


# ============================================================================= attractor + baseline
def relational_attractor(pts, sp_, iters=4):
    """Co-relaxation for a STABLE geodesic: the identity gate (built from the current fixed-fraction estimate)
    keeps the geodesic on the chain across coils; re-segment by fixed fraction each pass (stable -- means-based
    boundaries drift when one section dominates, e.g. the small intestine at 50% of the gut)."""
    ap = sp_.align_pow
    if ap > 0:                                                   # LUMINAL tube (hairpin): single tangent-consistent
        s = orient(fiedler(gj_laplacian(pts, np.ones(len(pts)), align_pow=ap)), pts, sp_)   # pass; the coil identity
        lab, li = segment(s, sp_)                                # gate hurts a hairpin, so skip it
        return s, lab, li
    node_g = np.ones(len(pts))
    s = orient(fiedler(gj_laplacian(pts, node_g)), pts, sp_)
    lab, li = segment(s, sp_)
    for _ in range(iters):
        node_g = np.array([sp_.ggj.get(c, 1.0) for c in lab])
        s = orient(fiedler(gj_laplacian(pts, node_g, lab_idx=li)), pts, sp_)
        lab, li = segment(s, sp_)
    return s, lab, li


def coordinate_cut(pts, sp_):
    """The straight-axis cut (what unified_embryo does): threshold the raw AP axis at the stored fractions."""
    x = pts[:, 0]; bnd = np.quantile(x, sp_.fract)
    return np.array([sp_.chain[i] for i in np.searchsorted(bnd, x)])


def accuracy(pred, truth):
    return float((np.array(pred) == np.array(truth)).mean())


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def basin(s, li, sp_, sig_max=0.45, nsig=12, reps=6):
    means = np.array([s[li == i].mean() if (li == i).any() else (i + 0.5) / len(sp_.chain)
                      for i in range(len(sp_.chain))])
    sigs = np.linspace(0, sig_max, nsig)

    def sweep(assign, m):
        out = np.zeros(nsig)
        for k, sg in enumerate(sigs):
            acc = 0.0
            for _ in range(reps):
                _, ri = segment(np.clip(s + sg * RNG.standard_normal(len(s)), 0, 1), sp_, m)
                acc += (ri == assign).mean()
            out[k] = acc / reps
        return out
    real = sweep(li, means)
    nl = RNG.permutation(li)
    nm = np.array([s[nl == i].mean() if (nl == i).any() else (i + 0.5) / len(sp_.chain) for i in range(len(sp_.chain))])
    nm = 0.5 * nm + 0.5 * np.full(len(sp_.chain), s.mean())
    nul = sweep(nl, nm)

    def width(c):
        for i in range(1, nsig):
            if c[i] < 0.5 <= c[i - 1]:
                t = (c[i - 1] - 0.5) / (c[i - 1] - c[i] + 1e-9)
                return float(sigs[i - 1] + t * (sigs[i] - sigs[i - 1]))
        return float(sigs[-1] if c[-1] >= 0.5 else 0.0)
    return sigs, real, nul, width(real), width(nul)


# ============================================================================= substrates
def _loop_gut(straight, girth):
    """Loop a straight gut strand onto the gut_tube coiling centreline, radius = per-cell girth."""
    x = straight[:, 0]; L = np.ptp(x) + 1e-9
    t = (x - x.min()) / L
    ts = np.linspace(0, 1, 240)
    C = gut_centerline(ts) * L
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    up = np.array([0, 0, 1.0]); N = np.cross(up, T); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    B = np.cross(T, N)
    idx = np.clip((t * (len(ts) - 1)).astype(int), 0, len(ts) - 1)
    ang = t * 40.0
    return C[idx] + (girth * np.cos(ang))[:, None] * N[idx] + (girth * np.sin(ang))[:, None] * B[idx]


def gut_substrate(n=1700, seed=0):
    sp_ = ChainSpec(
        name="gut",
        chain=["Esophagus", "Stomach", "SmallIntestine", "Colon", "Rectum"],
        fract=np.array([0.10, 0.30, 0.80, 0.92]),
        centers=np.array([0.05, 0.20, 0.55, 0.86, 0.965]),
        ggj={"Esophagus": 0.4, "Stomach": 0.35, "SmallIntestine": 0.35, "Colon": 0.3, "Rectum": 0.3},  # ICC slow-wave
        rad=np.array([0.10, 0.95, 0.22, 0.55, 0.35]),          # stored radius signature (stomach big, rectum bulge)
        col={"Esophagus": "#38bdf8", "Stomach": "#10b981", "SmallIntestine": "#f59e0b",
             "Colon": "#a78bfa", "Rectum": "#ef4444"})
    rng = np.random.default_rng(seed)
    t = np.sort(rng.random(n))
    # per-section radius: stomach balloons, colon/cecum + rectal ampulla bulge, thin long small intestine; the
    # minima between are the sphincter constrictions (cardia/pylorus/ileocecal/rectosigmoid). Kept SMALL vs the
    # coil spacing so windings do not touch (the luminal condition -- else the geodesic short-circuits).
    rp = 0.008 + 0.11 * (0.10 * _gauss(t, 0.05, 0.04) + 0.95 * _gauss(t, 0.20, 0.05)
                         + 0.22 * _gauss(t, 0.55, 0.18) + 0.55 * _gauss(t, 0.86, 0.05)
                         + 0.35 * _gauss(t, 0.965, 0.025))
    # loose FOLDING SPIRAL centreline: ~1.5 turns, radius shrinking inward -> AP (x) is NON-monotonic (folds
    # back and forth), so the straight-axis cut must fail; windings ~0.25 apart >> tube radius, so resolved.
    theta = t * 3.0 * np.pi
    R = 0.62 - 0.44 * t
    C = np.stack([R * np.cos(theta), R * np.sin(theta), 0.14 * np.sin(2 * np.pi * t)], 1)
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    up = np.array([0, 0, 1.0]); N = np.cross(up, T); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    B = np.cross(T, N)
    ang = rng.random(n) * 2 * np.pi; rr = rp * np.sqrt(rng.random(n))
    looped = C + (rr * np.cos(ang))[:, None] * N + (rr * np.sin(ang))[:, None] * B
    lab_true = np.array([sp_.chain[i] for i in np.searchsorted(sp_.fract, t)])
    return sp_, looped, t, lab_true


def heart_substrate(n=1400, seed=0):
    from medic.heart_chamber_attractor import synthetic_tube
    sp_ = ChainSpec(
        name="heart",
        chain=["inflow", "Atrium", "Ventricle", "Outflow"],
        fract=np.array([0.22, 0.45, 0.78]),
        centers=np.array([0.11, 0.335, 0.615, 0.89]),
        ggj={"inflow": 0.6, "Atrium": 0.8, "Ventricle": 1.0, "Outflow": 0.9},
        rad=np.array([0.55, 1.0, 1.35, 0.70]),                 # inflow / atrium / ventricle(biggest) / outflow
        col=CHAMBER_COL)
    looped, t, cham = loop_heart(synthetic_tube(n, seed=seed))
    return sp_, looped.astype(float), t, np.array([str(c) for c in cham])


def _wrap(t, C, rp, seed):
    """Place cells around a centre-line C(t) with per-cell radius rp, on the local normal frame."""
    rng = np.random.default_rng(seed)
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    up = np.array([0, 0, 1.0]); N = np.cross(up, T); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    B = np.cross(T, N)
    ang = rng.random(len(t)) * 2 * np.pi; rr = rp * np.sqrt(rng.random(len(t)))
    return C + (rr * np.cos(ang))[:, None] * N + (rr * np.sin(ang))[:, None] * B


def nephron_substrate(n=1500, seed=0):
    """Nephron: glomerulus -> proximal -> LOOP OF HENLE (a hairpin: descend into the medulla, turn, ascend) ->
    distal -> collecting duct (descend to the papilla). Axis 0 = cortico-medullary depth, which FOLDS (down at
    the loop, up, then down again) -> the straight-depth cut must fail."""
    sp_ = ChainSpec("nephron", ["Glomerulus", "Proximal", "LoopHenle", "Distal", "Collecting"],
                    np.array([0.12, 0.30, 0.62, 0.80]), np.array([0.06, 0.21, 0.46, 0.71, 0.90]),
                    {"Glomerulus": 0.5, "Proximal": 0.4, "LoopHenle": 0.4, "Distal": 0.4, "Collecting": 0.45},
                    np.array([1.0, 0.28, 0.22, 0.28, 0.5]),
                    {"Glomerulus": "#ef4444", "Proximal": "#f59e0b", "LoopHenle": "#38bdf8",
                     "Distal": "#a78bfa", "Collecting": "#10b981"}, align_pow=2.0)
    rng = np.random.default_rng(seed); t = np.sort(rng.random(n))
    u = np.clip((t - 0.30) / 0.32, 0, 1)                          # 0..1 across the loop of Henle
    inloop = (t >= 0.30) & (t < 0.62)
    depth = np.where(t < 0.30, 0.82,
             np.where(inloop, 0.80 - 0.72 * np.sin(np.pi * u),                                    # hairpin down + up
             np.where(t < 0.80, 0.78, 0.78 * (1 - np.clip((t - 0.80) / 0.20, 0, 1)))))            # collecting descent
    conv = 0.07 * np.sin(6 * np.pi * t) * (((t > 0.12) & (t < 0.30)) | ((t > 0.62) & (t < 0.80)))  # gentle S-tubule (developing)
    loop_sep = np.where(inloop, -0.16 * np.cos(np.pi * u), 0.0)   # descend on the LEFT, ascend on the RIGHT (resolved)
    coll = np.where(t >= 0.80, 0.24, 0.0)                         # collecting duct offset clear of the loop
    C = np.stack([depth, conv + loop_sep + coll, 0.05 * np.sin(2 * np.pi * t)], 1)
    rp = 0.007 + 0.10 * (0.55 * _gauss(t, 0.06, 0.035) + 0.25 * _gauss(t, 0.21, 0.08) + 0.20 * _gauss(t, 0.46, 0.12)
                         + 0.25 * _gauss(t, 0.71, 0.06) + 0.50 * _gauss(t, 0.90, 0.08))
    looped = _wrap(t, C, rp, seed + 1)
    return sp_, looped, t, np.array([sp_.chain[i] for i in np.searchsorted(sp_.fract, t)])


def cochlea_substrate(n=1400, seed=0):
    """Cochlea: the cochlear duct SPIRALS ~2.25 turns, tonotopic base -> middle -> apex. Axis 0 = x on the
    spiral -> folds every turn -> the straight cut fails; windings ~0.19 apart >> tube radius (resolved)."""
    sp_ = ChainSpec("cochlea", ["Base", "Middle", "Apex"], np.array([0.34, 0.67]),
                    np.array([0.17, 0.50, 0.83]), {"Base": 0.5, "Middle": 0.45, "Apex": 0.4},
                    np.array([0.6, 0.45, 0.35]),
                    {"Base": "#38bdf8", "Middle": "#f59e0b", "Apex": "#ef4444"})
    rng = np.random.default_rng(seed); t = np.sort(rng.random(n))
    theta = t * 2.25 * 2 * np.pi; R = 0.60 - 0.42 * t
    C = np.stack([R * np.cos(theta), R * np.sin(theta), 0.06 * t], 1)
    rp = 0.006 + 0.075 * (0.60 * _gauss(t, 0.15, 0.10) + 0.45 * _gauss(t, 0.50, 0.12) + 0.35 * _gauss(t, 0.85, 0.10))
    looped = _wrap(t, C, rp, seed + 1)
    return sp_, looped, t, np.array([sp_.chain[i] for i in np.searchsorted(sp_.fract, t)])


def brain_substrate(n=1600, seed=0):
    """Brain neuraxis: forebrain -> midbrain -> hindbrain -> spinal, along the neural tube bent by the cephalic
    FLEXURES (an ~300 deg arc). Axis 0 = x on the arc -> folds -> the straight cut fails. (Cerebellum is a
    dorsal BRANCH off the hindbrain = the tree family, not this chain.)"""
    sp_ = ChainSpec("brain", ["Forebrain", "Midbrain", "Hindbrain", "Spinal"], np.array([0.30, 0.50, 0.75]),
                    np.array([0.15, 0.40, 0.62, 0.88]),
                    {"Forebrain": 0.25, "Midbrain": 0.22, "Hindbrain": 0.20, "Spinal": 0.2},
                    np.array([0.9, 0.5, 0.7, 0.25]),
                    {"Forebrain": "#38bdf8", "Midbrain": "#10b981", "Hindbrain": "#f59e0b", "Spinal": "#a78bfa"})
    rng = np.random.default_rng(seed); t = np.sort(rng.random(n))
    ang = np.pi * (1.75 - 1.55 * t)                              # ~300 deg sweep -> x folds
    C = np.stack([np.cos(ang), np.sin(ang), 0.05 * np.sin(2 * np.pi * t)], 1)
    rp = 0.009 + 0.12 * (0.90 * _gauss(t, 0.14, 0.07) + 0.50 * _gauss(t, 0.40, 0.06)
                         + 0.70 * _gauss(t, 0.62, 0.08) + 0.20 * _gauss(t, 0.88, 0.10))
    looped = _wrap(t, C, rp, seed + 1)
    return sp_, looped, t, np.array([sp_.chain[i] for i in np.searchsorted(sp_.fract, t)])


SUBSTRATE = {"gut": gut_substrate, "heart": heart_substrate, "nephron": nephron_substrate,
             "cochlea": cochlea_substrate, "brain": brain_substrate}


# ============================================================================= run
def run(name):
    sp_, looped, t, truth = SUBSTRATE[name]()
    print(f"[{name}] {len(looped)} cells  |  chain {sp_.chain}")
    print("   ground truth: " + ", ".join(f"{c}={int((truth==c).sum())}" for c in sp_.chain))

    base = coordinate_cut(looped, sp_)
    s, lab, li = relational_attractor(looped, sp_)
    sbin, prof = radius_along_s(looped, s)
    bnds = snap_boundaries(s, sp_, sbin, prof)
    lab_lm, li_lm = segment_bnds(s, sp_, bnds)

    acc_base, acc_frac, acc_lm = accuracy(base, truth), accuracy(lab, truth), accuracy(lab_lm, truth)
    sp_r = spearman(s, t)
    sigs, real, nul, w_real, w_null = basin(s, li_lm, sp_)

    res = {"name": name, "n": len(looped), "chain": sp_.chain,
           "accuracy_coordinate_cut": round(acc_base, 3),
           "accuracy_attractor_fixed_fraction": round(acc_frac, 3),
           "accuracy_attractor_landmark": round(acc_lm, 3),
           "geodesic_spearman_vs_true_arclength": round(sp_r, 3),
           "basin_width_real": round(w_real, 3), "basin_width_null": round(w_null, 3),
           "per_section": {c: round(accuracy(np.array(lab_lm)[truth == c], truth[truth == c]), 3)
                           for c in sp_.chain if (truth == c).any()}}
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(res, open(f"data/organ_cascade/suborgan_attractor_{name}.json", "w"), indent=1)
    print(f"   coordinate cut (straight AP)      acc = {acc_base:.2f}")
    print(f"   attractor, fixed-fraction         acc = {acc_frac:.2f}")
    print(f"   attractor + landmarks             acc = {acc_lm:.2f}")
    print(f"   geodesic Spearman vs arc-length   = {sp_r:.2f}   basin {w_real:.2f} vs null {w_null:.2f}")
    print("   per-section:", res["per_section"])

    def col(l):
        return [sp_.col.get(c, "#556") for c in l]
    fig, ax = plt.subplots(1, 5, figsize=(24, 5.2), facecolor="#0d1017")
    for a in ax[:3]:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(looped[:, 0], looped[:, 1], s=8, c=col(truth)); ax[0].set_title(f"{name}: ground truth (looped)", color="#cbd5e1", fontsize=9)
    ax[1].scatter(looped[:, 0], looped[:, 1], s=8, c=col(base)); ax[1].set_title(f"coordinate cut (straight AP)\nacc {acc_base:.2f}", color="#f87171", fontsize=9)
    ax[2].scatter(looped[:, 0], looped[:, 1], s=8, c=col(lab_lm)); ax[2].set_title(f"relational attractor + landmarks\nacc {acc_lm:.2f}", color="#a5f3c0", fontsize=9)
    a = ax[3]; a.set_facecolor("#0d1017"); a.plot(sbin, prof, color="#e2e8f0", lw=2.0)
    for b in bnds:
        a.axvline(b, color="#f59e0b", lw=1.5)
    a.set_xlabel("geodesic s"); a.set_ylabel("radius"); a.set_title("boundaries = constrictions", color="#cbd5e1", fontsize=9)
    for sn in a.spines.values(): sn.set_color("#334155")
    a.tick_params(colors="#94a3b8")
    a = ax[4]; a.set_facecolor("#0d1017")
    a.plot(sigs, real, color="#7dd3fc", lw=2.4, label=f"attractor (w={w_real:.2f})")
    a.plot(sigs, nul, color="#f87171", lw=1.8, ls="--", label=f"null (w={w_null:.2f})")
    a.axhline(0.5, color="#475569", lw=0.8, ls=":"); a.set_xlabel("perturbation σ"); a.set_ylabel("returned")
    a.set_title("positional basin", color="#cbd5e1", fontsize=9); a.legend(fontsize=8, facecolor="#0d1017", labelcolor="#cbd5e1")
    for sn in a.spines.values(): sn.set_color("#334155")
    a.tick_params(colors="#94a3b8")
    hs = [plt.Line2D([0], [0], marker="o", ls="", mfc=sp_.col[k], mec="none", label=k) for k in sp_.chain]
    ax[0].legend(handles=hs, loc="lower left", fontsize=7, facecolor="#0d1017", labelcolor="#cbd5e1", framealpha=0.3)
    fig.suptitle(f"Suborgan relational attractor — {name}: geodesic + Physiome + chain + landmarks vs the coordinate cut",
                 color="#e2e8f0", fontsize=12)
    fig.tight_layout(); out = f"data/organ_cascade/suborgan_attractor_{name}.png"
    fig.savefig(out, dpi=125, facecolor="#0d1017"); print(f"   saved {out}")
    return res


def main():
    for name in (sys.argv[1:] or ["gut"]):
        run(name)


if __name__ == "__main__":
    main()
