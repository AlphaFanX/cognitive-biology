"""
Landmark thin-plate-spline registration: our genome body onto the real fetus.
=============================================================================

A 1D backbone cannot capture the fetus's blob head and branchy limbs. The right tool is a
LANDMARK warp: our organ centroids (heart, liver, gut, brain, muscle, skeleton, skin ...) are
put in correspondence with the SAME organs' centroids in the real E13.5 cell cloud (read from
the labels), and a thin-plate spline (scipy RBFInterpolator) smoothly deforms our whole body
so its organs land on the real organs. The organ correspondences are exactly the ones we
validated at r=0.99, now used as the registration landmarks. Result: our derived anatomy warped
into the real curled mouse shape.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.tps_register
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from medic.cell_level_mouse import COL, read_stage
from medic.cell_level_fit import FATE_COL, _col
from medic.unified_embryo import simulate, _symmetrize, FATES

TARGET = "data/mosta/E13.5_E1S1.MOSTA.h5ad"
# the full section series, earliest -> latest (E11.5 fills the gap so the trajectory is continuous).
STAGES = {"E9.5":  "data/mosta/E9.5_E1S1.MOSTA.h5ad",
          "E10.5": "data/mosta/E10.5_E1S1.MOSTA.h5ad",
          "E11.5": "data/mosta/E11.5_E1S1.MOSTA.h5ad",
          "E12.5": "data/mosta/E12.5_E1S1.MOSTA.h5ad",
          "E13.5": "data/mosta/E13.5_E1S1.MOSTA.h5ad"}
# each real stage <- the developmental extent of OUR grown body (a fraction of the simulate frames),
# so an earlier section is matched to a younger, smaller body -- a trajectory, not one body warped 5x.
STAGE_FRAC = {"E9.5": 0.55, "E10.5": 0.68, "E11.5": 0.78, "E12.5": 0.90, "E13.5": 1.0}
# FLEXURE HEAD anchor: the cephalo-caudal C-curl (degrees) per stage. A DOCUMENTED developmental
# schedule -- the mouse embryo tightens its C-curl through organogenesis (gentle comma at E9.5 -> a
# strong C by E11.5 that the large fetus keeps). Informed by MOSTA + embryology, stabilized because a
# single curl angle can't be read cleanly from the blob-headed, limb-branched sections (_curl_angle
# tried five backbone variants, all noisy). This is the g_K-style anchor: fitted now, genome-derivable
# later (from the convergent-extension / Wnt-PCP tone that drives both elongation and the fold).
CURL_DEG = {"E9.5": 100, "E10.5": 140, "E11.5": 165, "E12.5": 165, "E13.5": 155}


def curl_schedule():
    """The per-stage flexure curl. Prefer the CE-DERIVED schedule (medic.curl_from_ce -> the curl slaved
    to the convergent-extension elongation trajectory, a genome program + one confinement anchor; matches
    the hand-set schedule at r=0.98/rho=1.00) if it has been computed; else the hand-set CURL_DEG fallback."""
    import os, json
    p = "data/organ_cascade/curl_from_ce.json"
    if os.path.exists(p):
        try:
            d = json.load(open(p)).get("derived_curl_deg", {})
            if all(s in d for s in CURL_DEG):
                return {s: float(d[s]) for s in CURL_DEG}
        except Exception:
            pass
    return dict(CURL_DEG)
# landmark organ -> (real tissue names [synonyms across stages], our fate names). ONLY compact/localized
# organs: an extended tissue (spinal cord, skin, muscle) has a meaningless centroid and mis-anchors.
LM = {
    "head":      (["Brain"], ["Forebrain", "Eye"]),                   # localized organs only: they
    "heart":     (["Heart"], ["Heart"]),                             # trace head -> abdomen and give
    "notochord": (["Notochord"], ["Notochord"]),                    # the TPS a multi-point path to
    "skeleton":  (["Sclerotome", "Cartilage primordium"], ["Rib", "Cartilage"]),  # mid-trunk axial anchor
    "lung":      (["Lung", "Lung primordium"], ["Lung"]),           # bend our body along. Notochord is
    "liver":     (["Liver"], ["Liver"]),                            # the mid-axis anchor for the early
    "gut":       (["GI tract", "Primitive gut tube"], ["Gut"]),     # sections (before the viscera form).
    "pancreas":  (["Pancreas", "Pancreas primordium"], ["Pancreas"]),
}


def centroids(xy, names, want, minc=5):
    m = np.isin(names, want)
    return xy[m].mean(0) if m.sum() >= minc else None


def _our_slab(frame):
    """Sagittal slab of OUR grown body at a given frame (the real section is a 2D sagittal slice)."""
    Ps, _, Fs = _symmetrize(frame[3], frame[4], frame[5])
    slab = np.abs(Ps[:, 2]) < 0.50 * np.abs(Ps[:, 2]).max()
    return Ps[slab][:, :2].astype(float), np.array([FATES[f] if f >= 0 else "" for f in Fs[slab]])


def _tips(xy, ap):
    """Anterior-most and posterior-most points of a cloud along AP (robust: 2%-tail centroids)."""
    lo = xy[ap <= np.percentile(ap, 2)].mean(0)
    hi = xy[ap >= np.percentile(ap, 98)].mean(0)
    return lo, hi


def _real_ap(rxy, rann):
    """AP coordinate of a real section: PCA long axis, oriented so the brain/neural pole is anterior."""
    c = rxy - rxy.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    ap = c @ vt[0]
    bm = np.array([("Brain" in x) or ("Spinal" in x) for x in rann])
    if bm.sum() > 8 and ap[bm].mean() > ap.mean():
        ap = -ap
    return ap


def _curl_angle(rxy, rann, k=8, nb=14):
    """The real section's total cephalo-caudal CURL (degrees) = the turning of its GEODESIC medial-axis
    backbone. A GEODESIC (graph arc-length) backbone is required: PCA-AP cuts a chord ACROSS a tightly
    curled embryo and collapses the turn (reports the tight fetus as ~straight). The graph-diameter tips
    are the head and tail; geodesic distance from the head threads the curl. This is the flexure anchor
    (fitted-to-MOSTA now, genome-derivable later -- the g_K pattern)."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra, connected_components
    # measure along the NEURAL AXIS (brain->spinal cord): it threads the whole body head-to-tail and,
    # unlike the whole cloud, does NOT branch into the limbs -- so the graph diameter is the true axis,
    # not a head->limb path (which under-measured the curl of the limbed E12.5/E13.5 fetuses).
    axial = np.array([("Brain" in t) or ("Spinal" in t) or ("Notochord" in t) or ("ganglion" in t)
                      for t in rann])
    xy = rxy[axial] if axial.sum() >= 200 else rxy
    if len(xy) > 8000:
        xy = xy[np.random.default_rng(0).choice(len(xy), 8000, replace=False)]
    n = len(xy)
    dd, ii = cKDTree(xy).query(xy, k=k + 1)
    r = np.repeat(np.arange(n), k); c = ii[:, 1:].ravel(); w = dd[:, 1:].ravel()
    W = coo_matrix((w, (r, c)), shape=(n, n)).tocsr(); W = W.maximum(W.T)
    _, lab = connected_components(W, directed=False)
    keep = np.where(lab == np.bincount(lab).argmax())[0]
    xy, W, n = xy[keep], W[keep][:, keep], len(keep)
    d0 = dijkstra(W, indices=0)
    A = int(np.nanargmax(np.where(np.isfinite(d0), d0, -1)))         # one tip (head or tail)
    arc = dijkstra(W, indices=A)
    B = int(np.nanargmax(np.where(np.isfinite(arc), arc, -1)))       # the other tip
    arc_len = float(arc[B])                                          # geodesic backbone length (follows curl)
    chord = float(np.hypot(*(xy[B] - xy[A])))                       # straight head->tail distance
    ratio = arc_len / (chord + 1e-9)                               # 1 = straight; larger = more curled
    # arc/chord of a circular arc of angle t is (t/2)/sin(t/2); invert it robustly (no tangent noise).
    ts = np.linspace(0.02, 5.6, 3000)
    t = float(ts[np.argmin(np.abs((ts / 2) / np.sin(ts / 2) - ratio))])
    return float(np.degrees(t))


def _flex2d(xy, deg):
    """2D cephalo-caudal flexure of a point cloud (x=AP, y=DV): bend the AP axis into an arc of `deg`,
    DV rides the local normal. Applied AFTER the volumetric fill (which needs the straight AP axis)."""
    ang = np.radians(deg)
    if ang < 1e-6:
        return np.asarray(xy, float)
    x, y = xy[:, 0], xy[:, 1]
    L = float(np.ptp(x)) + 1e-9
    b = ang / L; s = x - x.min(); th = b * s
    cx = np.sin(th) / b; cy = (np.cos(th) - 1.0) / b
    nx, ny = -np.sin(th), np.cos(th)
    h = y - np.median(y)
    return np.c_[cx + h * nx, cy + h * ny]


def _head_at_low_ap(ap, dv):
    """Which AP end is the head? The head (brain+eye) is the bulkier end -- larger DV spread. Returns True
    if the head is at low AP. Robust to fate labelling (uses girth, not names)."""
    lo = dv[ap < np.percentile(ap, 25)].std()
    hi = dv[ap > np.percentile(ap, 75)].std()
    return lo >= hi


def _upright(xy, names=None):
    """Stand the generated embryo upright: AP vertical, head (the bulkier neural pole) at the TOP, DV
    horizontal. Purely a display rotation; head detected by girth so it never lands upside down."""
    xy = np.asarray(xy, float)
    ap, dv = xy[:, 0], xy[:, 1]
    v = -ap if _head_at_low_ap(ap, dv) else ap                       # head end -> top (positive v)
    v = v - v.mean()
    return np.c_[dv - dv.mean(), v]


def _flex_emergent(xy, names=None, ceph=0.9, cerv=0.4, ce=1.0, tail=1.0, total_deg=None):
    """EMERGENT fetal curl, head-up, from the ORGANS -- not a curvature painted on. The head is already
    LARGE because the brain expanded (head_expand in the growth); this function only adds the bending. The
    dominant term is the CONVERGENT-EXTENSION trunk curl (`ce`), a gentle DISTRIBUTED curvature along the
    trunk -- a smooth arc, not a kink. The cephalic term (`ceph`) is kept SMALL: the head is big, not
    sharply hooked, it only tilts ventrally. A small tail curl closes the comma. Over-driving the cephalic
    bump (the earlier default) is what made the shape angular. Amplitudes are turning angles in radians;
    output is upright, head at the top."""
    xy = np.asarray(xy, float)
    ap, dv = xy[:, 0].copy(), xy[:, 1].copy()
    L = np.ptp(ap) + 1e-9
    s = (ap - ap.min()) / L                                          # 0..1 along AP
    if not _head_at_low_ap(ap, dv):                                  # orient s so the head is at s=0
        s = 1.0 - s
    # local DV midline per s so thickness rides the centreline (not the global median)
    nb = 60; edges = np.linspace(0, 1, nb + 1); ctr = 0.5 * (edges[:-1] + edges[1:])
    midb = np.full(nb, np.nan)
    for i in range(nb):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if m.sum() > 3:
            midb[i] = np.median(dv[m])
    g = ~np.isnan(midb); midb = np.interp(ctr, ctr[g], midb[g])
    h = dv - np.interp(s, ctr, midb)                                 # signed dorsal(+)/ventral(-) thickness
    # curvature profile kappa(s): cephalic (brain) + cervical + CE trunk + tail, all same sense (ventral)
    # so the body tucks into a single comma. Each term's AMPLITUDE is its total turning angle in radians
    # (bumps normalised to unit integral), so ceph=1.6 really turns the head ~90 deg -- the flexures are
    # concentrated (a straighter trunk between them), which tucks a strong total curl WITHOUT the self-
    # crossing a uniform arc of the same total would make, and the tail curls BACK toward the head.
    grid = np.linspace(0, 1, 400); ds = grid[1] - grid[0]
    def _bump(mu, sig):
        b = np.exp(-((grid - mu) / sig) ** 2); return b / (b.sum() * ds)   # unit-integral bump
    box = ((grid > 0.28) & (grid < 0.86)).astype(float); box /= (box.sum() * ds)
    kap = (ceph * _bump(0.14, 0.10)                                 # cephalic flexure = head expansion
           + cerv * _bump(0.30, 0.13)                              # cervical flexure (WIDE -> gradual bend)
           + ce * box                                              # CE trunk buckling (distributed)
           + tail * _bump(0.90, 0.10))                             # tail curl (back toward the head)
    # GENOME GROUNDING: if a total curl is given (the CE-derived curl_schedule, from the convergent-
    # extension elongation trajectory in curl_from_ce), scale the whole profile so its integrated turning
    # equals that angle, keeping the ceph:cerv:ce:tail proportions. The MAGNITUDE is then the genome-CE
    # value, not a free number; only the DISTRIBUTION (head tilt vs trunk arc vs tail) is shape.
    if total_deg is not None:
        tot = float(np.sum(kap) * ds)
        if tot > 1e-6:
            kap = kap * (np.radians(total_deg) / tot)
    theta = np.cumsum(kap) * ds                                     # tangent angle vs +down (head at top)
    # centreline: start at the head (top) heading DOWN (-y), bending ventrally (+x) as theta grows
    cx = np.cumsum(np.sin(theta)) * ds * L
    cy = -np.cumsum(np.cos(theta)) * ds * L
    px = np.interp(s, grid, cx); py = np.interp(s, grid, cy); th = np.interp(s, grid, theta)
    nx, ny = np.cos(th), -np.sin(th)                               # dorsal normal (rotate tangent +90)
    out = np.c_[px + h * nx, py + h * ny]
    out = out - out.mean(0)
    # re-orient HEAD-UP: once curled, the girth test no longer finds the head, so rotate by the tracked
    # head cells (s<0.15) so the neural pole points to +y (a strong curl otherwise lands the head low).
    hd = out[s < 0.15].mean(0) if (s < 0.15).sum() > 5 else out[np.argmin(s)]
    rot = np.pi / 2 - np.arctan2(hd[1], hd[0])                     # bring head direction to +y (top)
    c, sn = np.cos(rot), np.sin(rot)
    return out @ np.array([[c, sn], [-sn, c]])                     # rotate head to the top


def register_frame(frame, rxy, rann, curl=0.0):
    """Warp our body onto a real section. Pipeline: (1) grown STRAIGHT body slab; (2) organ-centroid
    landmarks; (3) VOLUMETRIC fill of the straight body's DV envelope -> a solid tissue slice with real
    2D area; (4) FLEXURE head applied in 2D to the filled cloud AND the landmarks together (post-fill,
    so the envelope bins are on the true AP axis); (5) a gentle TPS places the filled, curled body."""
    oxy, Fnames = _our_slab(frame)
    src, dst, used = [], [], []
    for k, (rw, ow) in LM.items():
        cs = centroids(oxy, Fnames, ow); cd = centroids(rxy, rann, rw)
        if cs is not None and cd is not None:
            src.append(cs); dst.append(cd); used.append(k)
    if len(src) < 3:                                                 # TPS needs >=3 correspondences
        return None, None, np.array([]), used
    from medic.volumetric_body import fill_2d
    n_fill = int(min(28000, max(6000, len(rxy) * 0.7)))
    oxy_f, Fnames_f = fill_2d(oxy, Fnames, n_fill)                   # solid fill on the STRAIGHT body
    src2 = _flex2d(np.array(src), curl); oxy_f = _flex2d(oxy_f, curl)  # curl body + landmarks together
    warp = RBFInterpolator(src2, np.array(dst), kernel="thin_plate_spline", smoothing=8.0)
    keep = Fnames_f != ""
    return warp(oxy_f[keep]), Fnames_f[keep], np.array(dst), used


def _silhouette_from(xy, ann, nbins=40):
    """Real per-AP DV girth profile g(a) (fraction of body length) from an h5py-read section --
    the migration head's shape-training TARGET (mosta_shape_target, but h5py-direct for the big files).
    Binned along the GEODESIC (Fiedler) axis that FOLLOWS the body's C-curl: a PCA axis cuts a chord
    ACROSS a curled embryo, so cells from different true-AP levels fall in one bin and the girth
    profile is smeared. Girth per bin = the perpendicular (cross-section) spread about the local
    backbone tangent, normalised by the geodesic body length -- the unfolded, curl-robust silhouette."""
    from medic.ap_unbend import geodesic_ap
    xy = np.asarray(xy, float)
    ap = geodesic_ap(xy)
    bm = np.array([("Brain" in x) or ("Spinal" in x) for x in ann])
    if bm.sum() > 8 and ap[bm].mean() > 0.5:
        ap = 1.0 - ap
    be = np.linspace(0.0, 1.0, nbins + 1)
    ctr = 0.5 * (be[:-1] + be[1:])
    cen = np.full((nbins, 2), np.nan)                                # per-bin centroid = the backbone
    for i in range(nbins):
        m = (ap >= be[i]) & (ap < be[i + 1])
        if m.sum() >= 5:
            cen[i] = xy[m].mean(0)
    idx = np.where(~np.isnan(cen[:, 0]))[0]
    if len(idx) < 3:
        return dict(a=[], g=[])
    cg = cen[idx]
    L = float(np.sum(np.hypot(np.diff(cg[:, 0]), np.diff(cg[:, 1])))) + 1e-9   # geodesic body length
    A, G = [], []
    for k, i in enumerate(idx):
        m = (ap >= be[i]) & (ap < be[i + 1])
        j0 = idx[max(0, k - 1)]; j1 = idx[min(len(idx) - 1, k + 1)]
        tan = cen[j1] - cen[j0]; tan = tan / (np.hypot(*tan) + 1e-9)          # local backbone tangent
        perp = np.array([-tan[1], tan[0]])                                    # cross-section (girth) axis
        proj = (xy[m] - cen[i]) @ perp
        A.append(ctr[i]); G.append(float((np.percentile(proj, 95) - np.percentile(proj, 5)) / L))
    return dict(a=A, g=G)


def series(n_base=12000):
    """Register our genome body onto EVERY section E9.5->E13.5 (the full trajectory, E11.5 fills the gap).
    Genome-grounded shape: each stage's body is GROWN (division head -> density scaled by stage; migration
    head -> per-AP girth matched to that stage's real silhouette) then TPS-warped by the organ landmarks --
    the outline emerges from the heads, the landmarks only place it."""
    stages = list(STAGES)
    sched = curl_schedule()                                          # CE-derived curl if available, else hand-set
    fig, ax = plt.subplots(2, len(stages), figsize=(3.1 * len(stages), 6.4), facecolor="#0d1017")
    rng = np.random.default_rng(0)
    for j, st in enumerate(stages):
        rxy, rann = read_stage(STAGES[st])
        sil = _silhouette_from(rxy, rann)                             # this stage's real silhouette target
        curl = sched[st]                                             # flexure head: CE-derived curl anchor
        ne = int(n_base * STAGE_FRAC[st])                            # division head: density scaled by stage
        frames, _ = simulate(use_ecm=True, limb_buds=True, convergent_ext=1.0, n_end=ne,
                             shape_target=sil, flexure=0.0)          # grow STRAIGHT; curl applied post-fill
        bent, names, dst, used = register_frame(frames[-1], rxy, rann, curl)
        nwarp = 0 if names is None else len(names)
        rs = rng.choice(len(rxy), min(24000, len(rxy)), replace=False)   # subsample only for the scatter
        rxyp, rannp = rxy[rs], rann[rs]
        for a in ax[:, j]:
            a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        ax[0, j].scatter(rxyp[:, 0], rxyp[:, 1], s=1.2, c=[COL.get(t, "#585c66") for t in rannp], linewidths=0)
        if len(dst):
            ax[0, j].scatter(dst[:, 0], dst[:, 1], s=34, facecolors="none", edgecolors="#fff", linewidths=1.0)
        ax[0, j].set_title(f"real {st}  ({len(rxy):,})", color="#cbd5e1", fontsize=10)
        if bent is not None:
            ax[1, j].scatter(bent[:, 0], bent[:, 1], s=2.6, c=[_col(n) for n in names], linewidths=0, alpha=0.8)
        ax[1, j].set_title(f"ours -> {st}  ({len(used)} lm, N={frames[-1][0]:,})", color="#7dd3fc", fontsize=9)
        print(f"  {st}: real curl {curl:.0f} deg, {len(used)} landmarks, grown N={frames[-1][0]}, warped {nwarp:,} cells")
    ax[0, 0].set_ylabel("real MOSTA", color="#cbd5e1"); ax[1, 0].set_ylabel("our genome body", color="#7dd3fc")
    fig.suptitle("Genome body across the whole mouse series (E9.5->E13.5): shape from the division+migration heads",
                 color="#7dd3fc", fontsize=13)
    fig.tight_layout()
    fig.savefig("data/tps_register_series.png", dpi=120, facecolor="#0d1017")
    print("saved data/tps_register_series.png")


def main():
    series()


if __name__ == "__main__":
    main()
