"""
skin_shell_head.py -- the SKIN SURFACE SHELL: form the body's outer surface from its own outer cells.

The one thing that turns the point cloud into a body you RECOGNISE -- a point cloud has no occlusion, so it
reads as a scatter; a surface has a silhouette. Genome-honest: the epidermis (K5/K14, ectoderm) IS the outer
germ layer, so the shell is just the ENVELOPE of the body's outermost cells. Reads the body, writes the
surface -- read-only, cannot oppose the body plan.

Method: slice along AP; in each slice take the OUTERMOST cell in each angular sector around the slice
centroid (the skin lies on this envelope) -> a ring of surface points; stack the rings -> a closed outer
shell. Densify by interpolating around each ring so the surface is continuous (a silhouette, not dots).

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.skin_shell_head
Out: data/organ_cascade/skin_shell_head.png
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base


def shell(P, n_slice=80, n_sec=40, dens=3):
    """Outer envelope of a body point cloud P (laid: x=AP, y=DV, z=ML). Returns dense shell points."""
    x = P[:, 0]
    edges = np.linspace(x.min(), x.max(), n_slice + 1)
    rings = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 8:
            continue
        sl = P[m]
        c = sl.mean(0)
        ang = np.arctan2(sl[:, 2] - c[2], sl[:, 1] - c[1])         # angle in the DV-ML cross-section
        r = np.hypot(sl[:, 1] - c[1], sl[:, 2] - c[2])
        sec = ((ang + np.pi) / (2 * np.pi) * n_sec).astype(int) % n_sec
        ring = np.full((n_sec, 3), np.nan)
        for s in range(n_sec):
            ms = sec == s
            if ms.any():
                ring[s] = sl[np.where(ms)[0][r[ms].argmax()]]      # outermost cell in this sector = the skin
        # fill empty sectors by angular interpolation so the ring is closed
        good = ~np.isnan(ring[:, 0])
        if good.sum() < 6:
            continue
        idx = np.arange(n_sec)
        for k in range(3):
            ring[~good, k] = np.interp(idx[~good], idx[good], ring[good, k], period=n_sec)
        rings.append(ring)
    if not rings:
        return np.zeros((0, 3))
    S = np.vstack(rings)
    # densify between adjacent ring points (small) so the shell reads as a surface
    if dens > 1:
        extra = []
        for R in rings:
            for t in np.linspace(0, 1, dens, endpoint=False)[1:]:
                extra.append(R * (1 - t) + np.roll(R, -1, axis=0) * t)
        S = np.vstack([S] + extra)
    return S


def envelope_clip(P, body, n_slice=64, n_sec=24, margin=1.12):
    """Pull any point in P that lies OUTSIDE the body's own outer envelope back onto it (radius * margin),
    per AP slice and angular sector. Kills anatomy that SPLAYS beyond the skin (limb/vessel/nerve spikes
    sticking out) WITHOUT disturbing points already inside -- a properly-posed arm is inside the envelope, an
    over-warped spike is outside and gets clipped in. Also clamps AP to the body's AP range (no spike above
    the head). Angle-aware because the body is elliptical (wider in ML than DV)."""
    P = np.asarray(P, float).copy()
    x = body[:, 0]
    lo, hi = x.min(), x.max()
    P[:, 0] = np.clip(P[:, 0], lo, hi)                              # AP clamp (no spike above the head / below feet)
    edges = np.linspace(lo, hi, n_slice + 1)
    px = P[:, 0]
    cy, cz = np.median(body[:, 1]), np.median(body[:, 2])          # global midline (DV, ML) fallback centre
    r_all = np.hypot(body[:, 1] - cy, body[:, 2] - cz)
    r_global = np.percentile(r_all, 97)                            # global fallback cap for sparse slices
    for a, b in zip(edges[:-1], edges[1:]):
        mp = (px >= a) & (px < b)
        if not mp.any():
            continue
        mb = (x >= a) & (x < b)
        Ps = P[mp]
        if mb.sum() >= 8:                                          # per-sector cap from this slice's own skin
            sl = body[mb]; c = sl.mean(0)
            ang_b = np.arctan2(sl[:, 2] - c[2], sl[:, 1] - c[1])
            r_b = np.hypot(sl[:, 1] - c[1], sl[:, 2] - c[2])
            sec_b = ((ang_b + np.pi) / (2 * np.pi) * n_sec).astype(int) % n_sec
            rmax = np.array([np.percentile(r_b[sec_b == s], 92) if (sec_b == s).any() else 0.0
                             for s in range(n_sec)])
            good = rmax > 0
            idx = np.arange(n_sec)
            rmax[~good] = np.interp(idx[~good], idx[good], rmax[good], period=n_sec) if good.any() else r_global
        else:                                                     # sparse slice -> global circular fallback
            c = np.array([px[mp].mean() if False else 0.0, cy, cz]); c[0] = 0.5 * (a + b)
            rmax = np.full(n_sec, r_global)
        ang_p = np.arctan2(Ps[:, 2] - c[2], Ps[:, 1] - c[1])
        r_p = np.hypot(Ps[:, 1] - c[1], Ps[:, 2] - c[2])
        sec_p = ((ang_p + np.pi) / (2 * np.pi) * n_sec).astype(int) % n_sec
        cap = rmax[sec_p] * margin
        over = r_p > cap
        if over.any():
            scale = np.where(over, cap / (r_p + 1e-9), 1.0)
            Ps[:, 1] = c[1] + (Ps[:, 1] - c[1]) * scale
            Ps[:, 2] = c[2] + (Ps[:, 2] - c[2]) * scale
            P[mp] = Ps
    return P


def _tube(P, n_slice=48, n_sec=24, close_ends=True):
    """ONE closed FIXED-TOPOLOGY tube from a point cloud P (laid: x=AP, y=DV, z=ML). Stacks n_slice rings of
    n_sec outer-envelope points (empty sectors interpolated angularly, empty rings interpolated along AP, any
    remainder filled from the centroid so there are never NaNs / holes), triangulates adjacent rings, and fans
    end caps. Topology (faces) is identical for any P at the same (n_slice,n_sec). Returns (verts, faces int)."""
    P = np.asarray(P, float)
    x = P[:, 0]
    lo, hi = x.min(), x.max()
    edges = np.linspace(lo, hi, n_slice + 1)
    rings = np.full((n_slice, n_sec, 3), np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (x >= a) & (x <= b) if i == n_slice - 1 else (x >= a) & (x < b)
        apx = 0.5 * (a + b)
        if m.sum() >= 6:
            sl = P[m]; c = sl.mean(0)
            ang = np.arctan2(sl[:, 2] - c[2], sl[:, 1] - c[1])
            r = np.hypot(sl[:, 1] - c[1], sl[:, 2] - c[2])
            sec = ((ang + np.pi) / (2 * np.pi) * n_sec).astype(int) % n_sec
            for s in range(n_sec):
                ms = sec == s
                if ms.any():
                    rr = r[ms]
                    rt = np.percentile(rr, 88)               # ROBUST outer radius: reject splay outliers so a
                    j = np.where(ms)[0][np.abs(rr - rt).argmin()]   # few stray cells don't spike the surface
                    rings[i, s] = [apx, sl[j, 1], sl[j, 2]]
            good = ~np.isnan(rings[i, :, 0])
            if good.sum() >= 4:                              # fill empty sectors by angular interpolation
                idx = np.arange(n_sec)
                for k in (1, 2):
                    rings[i, ~good, k] = np.interp(idx[~good], idx[good], rings[i, good, k], period=n_sec)
        rings[i, :, 0] = apx                                 # every vert in the ring shares the slice AP
    # fill entirely-empty rings by interpolating each sector column along AP
    ai = np.arange(n_slice)
    for s in range(n_sec):
        col = rings[:, s, 1]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            for k in (1, 2):
                bad = np.isnan(rings[:, s, k])
                rings[bad, s, k] = np.interp(ai[bad], ai[ok], rings[ok, s, k])
    ctr = np.nanmean(rings.reshape(-1, 3), 0)               # final fallback for any remaining NaN
    rings[np.isnan(rings)] = 0.0
    bad = (rings[:, :, 1] == 0) & (rings[:, :, 2] == 0)
    rings[bad] = ctr
    # SMOOTH the envelope along AP (median-3 per sector): a localized 1-2-slice jut (a T-posed arm spiking
    # laterally at the shoulder) is replaced by its torso neighbours -> the surface reads as a smooth shoulder
    # taper instead of a thin horizontal spike, without flattening the gradual body contours.
    from scipy.ndimage import median_filter
    for k in (1, 2):
        rings[:, :, k] = median_filter(rings[:, :, k], size=(3, 1), mode="nearest")
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(n_slice - 1):
        for s in range(n_sec):
            s2 = (s + 1) % n_sec
            aa, bb = i * n_sec + s, i * n_sec + s2
            cc, dd = (i + 1) * n_sec + s, (i + 1) * n_sec + s2
            faces.append([aa, bb, dd]); faces.append([aa, dd, cc])
    if close_ends:
        # DOMED caps: the apex sits at the true extreme AP (x.min/max of the cloud), not at the last ring's slice
        # height -- so the crown rounds off to the top of the head instead of a FLAT fan disc one slice below it
        # (a wide head + flat lid reads as "chopped off"). The apex takes the ring's DV/ML centre + the real crown x.
        cap0 = rings[0].mean(0).copy(); cap0[0] = lo
        c0 = len(verts); verts = np.vstack([verts, cap0])
        for s in range(n_sec):
            faces.append([c0, (s + 1) % n_sec, s])
        cap1 = rings[-1].mean(0).copy(); cap1[0] = hi
        c1 = len(verts); verts = np.vstack([verts, cap1])
        base = (n_slice - 1) * n_sec
        for s in range(n_sec):
            faces.append([c1, base + s, base + (s + 1) % n_sec])
    return verts.astype(np.float32), np.array(faces, dtype=np.int32)


def mesh(P, n_slice=48, n_sec=24, crotch_f=0.42, ventral=None):
    """MULTI-TUBE skin surface: a torso+head tube, a sleeve per ARM, and a tube per LEG, unioned. The ARM sleeves
    are essential in the Vitruvian arms-OUT pose: the torso tube's median-filter erases a horizontal arm as a
    T-pose splay spike, so without a dedicated sleeve the arms (and the hands + fingers at their ends) render
    BARE. Each arm sleeve is a tube stacked along the arm's own ML long axis (so its end cap covers the hand);
    the arms are excluded from the torso so the shoulder is not a wide disc. Fixed topology (faces identical for a
    given n_slice/n_sec): the region split ALWAYS emits five tubes (a region with too few cells becomes a hidden
    collapsed stub), so faces emit ONCE and verts stream per frame. Returns (verts, faces int), laid (x=AP,y=DV,z=ML)."""
    P = np.asarray(P, float)
    x, z = P[:, 0], P[:, 2]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    n_leg = max(10, n_slice // 2); n_arm = max(10, n_slice // 2)
    rng = np.random.default_rng(0)
    # ARM split: outboard of the trunk in the shoulder AP band. Threshold from the arm-free lower trunk's ML width.
    low = P[(apf >= 0.48) & (apf < 0.60)]
    trunk_half = float(np.percentile(np.abs(low[:, 2]), 85)) if len(low) > 10 else 0.15 * (np.ptp(z) + 1e-9)
    armthr = max(0.55 * trunk_half, 0.08 * (np.ptp(z) + 1e-9))   # start the sleeve INSIDE the shoulder so it OVERLAPS
    is_arm = (apf >= 0.56) & (apf <= 0.94) & (np.abs(z) > armthr)  # the torso (no floating-arm gap), out to the hand
    armR = P[is_arm & (z > 0)]; armL = P[is_arm & (z < 0)]
    # keep the arm base in the torso too (overlap join); only DROP the far arm from the torso so the shoulder is not
    # a wide disc but the sleeves still meet the trunk.
    torso = P[(apf >= crotch_f) & ~(is_arm & (np.abs(z) > 1.4 * armthr))]
    if len(torso) < n_sec:
        torso = P
    legR = P[(apf < crotch_f) & (z > 0)]; legL = P[(apf < crotch_f) & (z < 0)]

    def _stub(anchor):                                          # a hidden collapsed tube (keeps topology fixed)
        return anchor + rng.normal(size=(2 * n_sec, 3)) * 0.004

    def _leg(leg):
        return leg if len(leg) >= n_sec else _stub(np.array([x.min(), torso[:, 1].mean(), 0.0]))

    def _arm_tube(arm):
        # stack the sleeve along the arm's ML long axis: remap (x,y,z)->(z,y,x) so _tube stacks on z and caps the
        # shoulder + HAND ends, then remap the verts back. A short arm collapses to a hidden stub.
        if len(arm) < n_sec:
            v, f = _tube(_stub(np.array([torso[:, 0].mean(), torso[:, 1].mean(), 0.0])), n_arm, n_sec)
            return v, f
        v, f = _tube(arm[:, [2, 1, 0]], n_arm, n_sec)
        return v[:, [2, 1, 0]], f

    tubes = [_tube(torso, n_slice, n_sec), _arm_tube(armR), _arm_tube(armL),
             _tube(_leg(legR), n_leg, n_sec), _tube(_leg(legL), n_leg, n_sec)]
    V, Fc, off = [], [], 0
    for v, f in tubes:
        V.append(v); Fc.append(f + off); off += len(v)
    # HANDS + FEET: a five-digit hand at each arm tip, a five-toe foot at each leg tip (fixed topology). Reads as
    # a hand/foot with fingers/toes instead of a smooth stump. The tips track the moving cloud each frame.
    from medic import hand_foot_skin as HFS

    def _tip(cells, col, sign, pct):                            # the extreme end of a limb along axis `col`
        if len(cells) < 8:
            return None, 0.0
        v = sign * cells[:, col]; sel = cells[v > np.percentile(v, pct)]
        return sel.mean(0), float(np.ptp(cells[:, col]))

    appendages = []
    for arm, sgn in ((armR, 1.0), (armL, -1.0)):               # hands: fingers point outward along +-ML
        tip, reach = _tip(arm, 2, sgn, 90)
        if tip is None:
            tip = np.array([torso[:, 0].mean(), torso[:, 1].mean(), 0.0]); reach = 0.3
        appendages.append(HFS.build(tip, [0, 0, sgn], [0, 1, 0], 0.18 * reach + 0.03, 0.30 * reach + 0.03, "hand"))
    # anterior (toes point FORWARD): use the ventral sign from the fated cloud if given, else estimate from DV skew.
    dvsign = float(ventral) if ventral is not None else (1.0 if torso[:, 1].mean() >= P[:, 1].mean() else -1.0)
    H = float(np.ptp(x)) + 1e-9                                 # stature: size the feet off the body, not the thin leg
    for leg in (legR, legL):                                   # feet: toes point anterior, sole down, at the SOLE
        if len(leg) >= 8:
            lo = leg[leg[:, 0] < np.percentile(leg[:, 0], 12)]  # the lowest slice of the leg
            tip = np.array([leg[:, 0].min(), lo[:, 1].mean(), lo[:, 2].mean()])  # ANCHOR at the true leg bottom
        else:
            tip = np.array([x.min(), torso[:, 1].mean(), 0.0])
        appendages.append(HFS.build(tip, [0, dvsign, 0], [-1, 0, 0], 0.055 * H, 0.14 * H, "foot"))
    for v, f in appendages:
        V.append(v); Fc.append(f + off); off += len(v)
    Vout = np.vstack(V).astype(np.float32)
    # a thin/degenerate tube can leave NaNs; scrub them (NaN -> body centroid) so the emitted JSON is valid
    if np.isnan(Vout).any():
        Vout = np.nan_to_num(Vout, nan=float(np.nanmean(P)))
    return Vout, np.vstack(Fc).astype(np.int32)


def silhouette_bounds(P, n=80):
    """Per-AP-band left/right ML extent -- for a filled body silhouette in the front view."""
    x, z = P[:, 0], P[:, 2]
    edges = np.linspace(x.min(), x.max(), n + 1)
    xs, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 6:
            continue
        xs.append(0.5 * (a + b)); lo.append(np.percentile(z[m], 3)); hi.append(np.percentile(z[m], 97))
    return np.array(xs), np.array(lo), np.array(hi)


def main():
    base, F = build_base()
    S = shell(base)
    xs, lo, hi = silhouette_bounds(base)
    fig, ax = plt.subplots(1, 2, figsize=(11, 8), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(base[:, 2], base[:, 0], s=1, c="#2b3340", alpha=0.4)
    ax[0].scatter(S[:, 2], S[:, 0], s=2, c="#caa894", alpha=0.5)
    ax[0].set_title(f"skin shell = outer envelope ({len(S)} surface points)", color="#e8c9a8", fontsize=10)
    ax[1].fill_betweenx(xs, lo, hi, color="#caa894", alpha=0.85)
    ax[1].set_title("filled silhouette (the body reads as a body)", color="#e8c9a8", fontsize=10)
    fig.suptitle("Skin surface shell: the body's own outer cells enveloped into a surface -- gives the cloud "
                 "a silhouette", color="#e2e8f0", fontsize=10)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/skin_shell_head.png", dpi=120, facecolor="#0d1017")
    print(f"skin shell: {len(S)} surface points enveloping {len(base)} cells")
    print("saved data/organ_cascade/skin_shell_head.png")


if __name__ == "__main__":
    main()
