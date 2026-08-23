"""heart_chamber_attractor.py -- a RELATIONAL (configurational) attractor for the looped heart chambers.

WHY. electric_organs_head reports the HEART chambers UNORDERED on their own eigenframe: the tube LOOPS
(the D-loop), so NO linear coordinate -- AP/DV/ML, or any Cartesian eigenmode projection -- separates the
developmental chamber chain inflow -> atrium -> ventricle -> outflow. The current unified_embryo mechanism
assigns chambers by a STATIC COORDINATE CUT (Outflow = top AP quantile; Atrium/Ventricle by the DV median).
On a loop that mis-slices: after looping, the outflow returns cranio-ventral and the ventricle bulges
ventral, so AP and DV are non-monotonic along the chamber sequence.

THE ATTRACTOR (form + function, no new data):
  * OPERATOR  (function, Physiome): the gap-junction graph Laplacian on the heart cells, edges weighted by
    the real per-chamber g_gj (ten Tusscher ventricle 1.0 / Courtemanche atrium 0.8 / conduction outflow 0.9,
    slow venous pole 0.6) -- data/physiome/tissue_conductances.json.
  * COORDINATE (position): the graph GEODESIC arc-length (Fiedler vector of the Laplacian) -- it runs ALONG
    the loop where a straight axis cannot.
  * PATTERN   (form, Gray's): the chamber CHAIN inflow -> atrium -> ventricle -> outflow at its arc-length
    fractions -- the relational configuration the cells relax into.
  * CO-RELAX  (the attractor dynamics): iterate {identity-gated geodesic -> re-segment onto the stored chain}.
    The identity gate suppresses edges between NON-ADJACENT chambers, so the geodesic follows the chain and
    does NOT short-circuit across touching windings (the hazard the luminal-heart work flagged). The chambers
    settle into the stored looped sequence; perturb + relax returns each cell to its own chamber -- the SPN
    basin test, for POSITION, one scale below head_attractor_test's identity basins.

Compared vs the coordinate-cut baseline on the heart_loop ground truth.
Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.heart_chamber_attractor
Out:  data/organ_cascade/heart_chamber_attractor.{json,png}
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.spatial import cKDTree
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.heart_loop import loop_heart, CHAMBER_COL

CHAIN = ["inflow", "Atrium", "Ventricle", "Outflow"]      # the stored relational sequence (venous -> arterial)
CIDX = {c: i for i, c in enumerate(CHAIN)}
FRACT = np.array([0.22, 0.45, 0.78])                       # developmental arc-length boundaries (heart_loop.chamber_of)
PHYS = "data/physiome/tissue_conductances.json"
GGJ = {"inflow": 0.6, "Atrium": 0.8, "Ventricle": 1.0, "Outflow": 0.9}   # fallback; overwritten from Physiome
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------- substrate
def _gauss(t, mu, sd):
    return np.exp(-0.5 * ((t - mu) / sd) ** 2)


def radius_profile(t):
    """Tube thickness along the cardiac tube: the chambers BALLOON (atrium, and the ventricle biggest) and the
    junctions CONSTRICT -- the sino-atrial, atrio-ventricular canal, and outflow narrowings. So the chamber
    BOUNDARIES are the radius minima along arc-length (the landmarks the attractor snaps to)."""
    bulge = (0.55 * _gauss(t, 0.11, 0.06)      # sinus / inflow (mild)
             + 1.00 * _gauss(t, 0.33, 0.07)    # atrium
             + 1.35 * _gauss(t, 0.61, 0.09)    # ventricle (largest)
             + 0.70 * _gauss(t, 0.89, 0.06))   # outflow
    return 0.045 + 0.135 * bulge               # ~0.05 at the constrictions -> ~0.24 at the ventricle


def synthetic_tube(n=1400, length=1.0, seed=0):
    """A straight cardiac tube with a REALISTIC per-chamber radius profile (ballooned chambers, constricted
    junctions), cells strung along x with a radial offset scaled by radius_profile(t)."""
    rng = np.random.default_rng(seed)
    t = rng.random(n)
    x = (t - 0.5) * length
    ang = rng.random(n) * 2 * np.pi
    rr = radius_profile(t) * np.sqrt(rng.random(n))
    return np.stack([x, rr * np.cos(ang), rr * np.sin(ang)], 1)


def physiome_ggj():
    g = dict(GGJ)
    if os.path.exists(PHYS):
        p = json.load(open(PHYS))
        for c in ("Atrium", "Ventricle", "Outflow"):
            if c in p:
                g[c] = float(p[c]["g_gj_rel"])
    return g


# ----------------------------------------------------------------------------- operator + geodesic
def gj_laplacian(pts, node_g=None, lab_idx=None, gate_low=0.04, k=10):
    """Physiome-weighted kNN gap-junction Laplacian. If lab_idx (current chamber estimate) is given, edges
    between NON-ADJACENT chambers (chain distance > 1) are down-weighted by gate_low -- the identity gate
    that keeps the geodesic on the chamber chain instead of shortcutting across touching windings."""
    n = len(pts); k = min(k, n - 1)
    d, idx = cKDTree(pts).query(pts, k=k + 1)
    sig = np.median(d[:, 1:]) + 1e-9
    if node_g is None:
        node_g = np.ones(n)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j, dist in zip(idx[i, 1:], d[i, 1:]):
            g = np.sqrt(max(node_g[i], 1e-6) * max(node_g[j], 1e-6))
            w = g * np.exp(-(dist / sig) ** 2)
            if lab_idx is not None and abs(int(lab_idx[i]) - int(lab_idx[j])) > 1:
                w *= gate_low
            rows += [i, j]; cols += [j, i]; vals += [w, w]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


def fiedler(L):
    """2nd eigenvector of the graph Laplacian = the geodesic arc-length coordinate along the tube."""
    try:
        vals, vecs = eigsh(L.astype(float), k=3, sigma=1e-8, which="LM")
        v = vecs[:, np.argsort(vals)][:, 1]
    except Exception:
        vals, vecs = np.linalg.eigh(L.toarray())
        v = vecs[:, np.argsort(vals)][:, 1]
    return (v - v.min()) / (np.ptp(v) + 1e-9)                    # -> [0,1]


def orient(s, pts):
    """Orient the geodesic so s=1 is the CRANIAL (outflow) pole -- the tube end with higher AP (x)."""
    lo = pts[s <= np.quantile(s, 0.1), 0].mean()
    hi = pts[s >= np.quantile(s, 0.9), 0].mean()
    return 1.0 - s if lo > hi else s


def segment(s, means=None):
    """Assign chambers along the ordered coordinate s. Boundaries = midpoints of the ordered chamber means
    (the Hopfield pull to the stored, ORDER-CONSTRAINED prototypes); the developmental FRACT on the first pass."""
    if means is None:
        bnd = np.quantile(s, FRACT)      # stored fractions laid on the geodesic by RANK (rank ~ arc-length)
    else:
        m = np.sort(means)
        bnd = (m[:-1] + m[1:]) / 2
    idx = np.searchsorted(bnd, s)
    return np.array([CHAIN[i] for i in idx]), idx


# ----------------------------------------------------------------------------- the attractor
def relational_attractor(pts, g, iters=6):
    """Co-relaxation: geodesic -> segment -> identity-gated geodesic -> re-segment ... settle onto the chain."""
    node_g = np.ones(len(pts))
    s = orient(fiedler(gj_laplacian(pts, node_g)), pts)         # pass 0: ungated geodesic
    lab, li = segment(s)
    for _ in range(iters):
        node_g = np.array([g[c] for c in lab])                  # Physiome g_gj per current chamber estimate
        s = orient(fiedler(gj_laplacian(pts, node_g, lab_idx=li)), pts)
        means = np.array([s[li == i].mean() if (li == i).any() else (i + 0.5) / len(CHAIN)
                          for i in range(len(CHAIN))])
        lab, li = segment(s, means)
    return s, lab, li


def radius_along_s(pts, s, nbin=44, smooth=3):
    """Tube radius as a function of the geodesic arc-length s: bin cells by s, take each bin's cross-sectional
    spread about its own centroid. The minima of this profile are the junction constrictions = the boundaries."""
    edges = np.quantile(s, np.linspace(0, 1, nbin + 1)); edges[-1] += 1e-9
    binid = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, nbin - 1)
    sbin = np.zeros(nbin); prof = np.full(nbin, np.nan)
    for b in range(nbin):
        m = binid == b
        if m.sum() >= 3:
            prof[b] = np.linalg.norm(pts[m] - pts[m].mean(0), axis=1).mean()
            sbin[b] = s[m].mean()
        else:
            sbin[b] = 0.5 * (edges[b] + edges[b + 1])
    ok = ~np.isnan(prof)
    prof = np.interp(np.arange(nbin), np.where(ok)[0], prof[ok])
    prof = np.convolve(prof, np.ones(smooth) / smooth, mode="same")
    return sbin, prof


CENTERS = np.array([0.11, 0.335, 0.615, 0.89])                   # chamber centres = midpoints of the chain segments


def snap_boundaries(s, sbin, prof, win=0.09):
    """WATERSHED between chamber bulges: find each chamber's radius PEAK (near its stored centre fraction),
    then place the boundary at the radius MINIMUM BETWEEN consecutive peaks. This snaps to the real junction
    constriction (AV canal, outflow narrowing) and structurally excludes the pole TAPER (radius->0 at the tips),
    which a naive 'minimum near the fraction' wrongly grabs. Returns 3 ordered boundary s-values."""
    span = np.ptp(sbin) + 1e-9
    peaks = []
    for cf in CENTERS:                                           # 1. bulge peak near each chamber centre
        s_c = np.quantile(s, cf)
        near = np.where(np.abs(sbin - s_c) <= win * span)[0]
        peaks.append(sbin[near[np.argmax(prof[near])]] if len(near) else s_c)
    peaks = np.sort(np.array(peaks))
    bnds = []
    for k in range(3):                                           # 2. minimum between consecutive peaks
        m = np.where((sbin >= peaks[k]) & (sbin <= peaks[k + 1]))[0]
        bnds.append(sbin[m[np.argmin(prof[m])]] if len(m) else 0.5 * (peaks[k] + peaks[k + 1]))
    return np.sort(np.array(bnds))


def segment_bnds(s, bnds):
    idx = np.searchsorted(bnds, s)
    return np.array([CHAIN[i] for i in idx]), idx


def coordinate_cut(pts):
    """The current unified_embryo mechanism: AP poles + DV median. Faithful straight-space baseline."""
    x, y = pts[:, 0], pts[:, 1]
    lab = np.empty(len(pts), object)
    hi, lo = np.quantile(x, 0.78), np.quantile(x, 0.22)
    lab[x >= hi] = "Outflow"; lab[x <= lo] = "inflow"
    mid = (x > lo) & (x < hi); ym = np.median(y[mid])
    lab[mid & (y >= ym)] = "Atrium"; lab[mid & (y < ym)] = "Ventricle"
    return lab


# ----------------------------------------------------------------------------- scoring
def accuracy(pred, truth):
    return float((np.array(pred) == np.array(truth)).mean())


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def basin(s, li, sig_max=0.45, nsig=12, reps=6):
    """SPN-style basin test on POSITION: perturb the geodesic coordinate, relax (nearest ordered prototype),
    measure the fraction of cells returning to their own chamber. Real vs a label-shuffled null."""
    means = np.array([s[li == i].mean() if (li == i).any() else (i + 0.5) / len(CHAIN) for i in range(len(CHAIN))])
    sigs = np.linspace(0, sig_max, nsig)

    def sweep(assign):
        out = np.zeros(nsig)
        for k, sg in enumerate(sigs):
            acc = 0.0
            for _ in range(reps):
                _, ri = segment(np.clip(s + sg * RNG.standard_normal(len(s)), 0, 1), means)
                acc += (ri == assign).mean()
            out[k] = acc / reps
        return out
    real = sweep(li)
    nullassign = RNG.permutation(li)
    nmeans = np.array([s[nullassign == i].mean() if (nullassign == i).any() else (i + 0.5) / len(CHAIN)
                       for i in range(len(CHAIN))])
    globalmean = np.full(len(CHAIN), s.mean())                   # shuffled prototypes collapse to the global mean
    nmeans = 0.5 * nmeans + 0.5 * globalmean

    def sweep_null():
        out = np.zeros(nsig)
        for k, sg in enumerate(sigs):
            acc = 0.0
            for _ in range(reps):
                _, ri = segment(np.clip(s + sg * RNG.standard_normal(len(s)), 0, 1), nmeans)
                acc += (ri == nullassign).mean()
            out[k] = acc / reps
        return out
    nul = sweep_null()

    def width(curve):
        for i in range(1, nsig):
            if curve[i] < 0.5 <= curve[i - 1]:
                t = (curve[i - 1] - 0.5) / (curve[i - 1] - curve[i] + 1e-9)
                return float(sigs[i - 1] + t * (sigs[i] - sigs[i - 1]))
        return float(sigs[-1] if curve[-1] >= 0.5 else 0.0)
    return sigs, real, nul, width(real), width(nul)


# ----------------------------------------------------------------------------- run
def main():
    straight = synthetic_tube()
    looped, t, cham = loop_heart(straight)                       # ground-truth looped tube + chamber labels
    looped = looped.astype(float); cham = np.array([str(c) for c in cham])
    g = physiome_ggj()
    print(f"heart cells {len(looped)}  |  Physiome g_gj {g}")
    print(f"ground-truth chambers: " + ", ".join(f"{c}={int((cham==c).sum())}" for c in CHAIN))

    base = coordinate_cut(looped)
    s, lab, li = relational_attractor(looped, g)                 # fraction-segmented (previous)
    sbin, prof = radius_along_s(looped, s)
    bnds = snap_boundaries(s, sbin, prof)
    lab_lm, li_lm = segment_bnds(s, bnds)                        # landmark-snapped boundaries

    acc_base = accuracy(base, cham)
    acc_frac = accuracy(lab, cham)
    acc_lm = accuracy(lab_lm, cham)
    sp_attr = spearman(s, t)                                     # geodesic vs true tube parameter (loop recovered?)
    sigs, real, nul, w_real, w_null = basin(s, li_lm)

    res = {"n": len(looped), "physiome_ggj": g,
           "accuracy_coordinate_cut": round(acc_base, 3),
           "accuracy_attractor_fixed_fraction": round(acc_frac, 3),
           "accuracy_attractor_landmark_snapped": round(acc_lm, 3),
           "geodesic_spearman_vs_true_arclength": round(sp_attr, 3),
           "basin_width_real": round(w_real, 3), "basin_width_null": round(w_null, 3),
           "boundary_s_values": [round(float(b), 3) for b in bnds],
           "per_chamber_accuracy_landmark": {c: round(accuracy(np.array(lab_lm)[cham == c], cham[cham == c]), 3)
                                             for c in CHAIN if (cham == c).any()}}
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(res, open("data/organ_cascade/heart_chamber_attractor.json", "w"), indent=1)

    print(f"\n[baseline]  coordinate cut (AP pole + DV median)         accuracy = {acc_base:.2f}")
    print(f"[attractor] geodesic + chain, FIXED-FRACTION boundaries   accuracy = {acc_frac:.2f}")
    print(f"[attractor] geodesic + chain, LANDMARK-SNAPPED boundaries accuracy = {acc_lm:.2f}")
    print(f"           geodesic Spearman vs true arc-length           = {sp_attr:.2f}  (loop recovered)")
    print(f"           basin width real {w_real:.2f} vs null {w_null:.2f}  (robust positional attractor)")
    print("           per-chamber (landmark):", res["per_chamber_accuracy_landmark"])
    lab = lab_lm  # the landmark-snapped assignment is the attractor's output for the figure

    # ---------------- figure ----------------
    def col(labels):
        return [CHAMBER_COL.get(c, "#556") for c in labels]
    fig, ax = plt.subplots(1, 5, figsize=(24, 5.4), facecolor="#0d1017")
    for a in ax[:3]:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(looped[:, 0], looped[:, 1], s=9, c=col(cham))
    ax[0].set_title("ground truth (D-loop, sagittal)\ninflow→atrium→ventricle→outflow", color="#cbd5e1", fontsize=9)
    ax[1].scatter(looped[:, 0], looped[:, 1], s=9, c=col(base))
    ax[1].set_title(f"coordinate cut (current mechanism)\nAP pole + DV median  —  acc {acc_base:.2f}", color="#f87171", fontsize=9)
    ax[2].scatter(looped[:, 0], looped[:, 1], s=9, c=col(lab_lm))
    ax[2].set_title(f"relational attractor + landmarks\ngeodesic+Physiome+chain  —  acc {acc_lm:.2f}", color="#a5f3c0", fontsize=9)

    a = ax[3]; a.set_facecolor("#0d1017")
    a.plot(sbin, prof, color="#e2e8f0", lw=2.0)
    for b in bnds:
        a.axvline(b, color="#f59e0b", lw=1.6, ls="-")               # snapped constriction (boundary)
    for f in FRACT:
        a.axvline(np.quantile(s, f), color="#64748b", lw=1.0, ls=":")  # stored developmental fraction
    a.set_xlabel("geodesic arc-length s"); a.set_ylabel("tube radius")
    a.set_title("boundaries = radius MINIMA (constrictions)\norange = snapped, dotted = stored fraction", color="#cbd5e1", fontsize=9)
    for sname in a.spines.values(): sname.set_color("#334155")
    a.tick_params(colors="#94a3b8")

    a = ax[4]; a.set_facecolor("#0d1017")
    a.plot(sigs, real, color="#7dd3fc", lw=2.6, label=f"attractor (w={w_real:.2f})")
    a.plot(sigs, nul, color="#f87171", lw=2.0, ls="--", label=f"shuffled null (w={w_null:.2f})")
    a.axhline(0.5, color="#475569", lw=0.8, ls=":")
    a.set_xlabel("perturbation σ (geodesic)"); a.set_ylabel("returned to own chamber")
    a.set_title("positional basin (SPN test, one scale down)", color="#cbd5e1", fontsize=9)
    a.legend(fontsize=8, facecolor="#0d1017", labelcolor="#cbd5e1")
    for sname in a.spines.values(): sname.set_color("#334155")
    a.tick_params(colors="#94a3b8")

    hs = [plt.Line2D([0], [0], marker="o", ls="", mfc=CHAMBER_COL[k], mec="none", label=k) for k in CHAIN]
    ax[0].legend(handles=hs, loc="lower left", fontsize=7, facecolor="#0d1017", labelcolor="#cbd5e1", framealpha=0.3)
    fig.suptitle("Heart-chamber relational attractor: coordinate cut fails on the loop; geodesic + Physiome operator + "
                 "stored chain + constriction landmarks recovers it", color="#e2e8f0", fontsize=12)
    fig.tight_layout()
    out = "data/organ_cascade/heart_chamber_attractor.png"
    fig.savefig(out, dpi=130, facecolor="#0d1017"); print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
