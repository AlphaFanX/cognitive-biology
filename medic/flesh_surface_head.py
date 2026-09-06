"""
flesh_surface_head.py -- THE FLESH: drape the skin over the MUSCLE+FAT layer, not the bone.

The Vitruvian proportions are the frame; the flesh is what makes a body appreciable -- the bulge of the biceps,
the geometry of the six-pack, the flare of the gluteus into the sweep of the quadriceps, the taper of the calf.
Those are absent when the skin is the envelope of the OUTER cells (bone + skin layer): nothing pushes the skin
out from underneath. This head fixes that at the source. It gives each muscle a fusiform BELLY with real mass,
adds a subcutaneous FAT shell that smooths the bellies into a continuous contour (marble, not raw anatomy), and
computes the skin SURFACE as the envelope of THAT layer -- so the muscle relief reads through the skin.

One mechanism, hundreds of surface-relief metrics (the Economy principle): belly-shaping x fat x skin-over-muscle.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.flesh_surface_head
Out: data/organ_cascade/flesh_surface_head.png
"""
from __future__ import annotations
import os
import numpy as np
from scipy.spatial import cKDTree

from menagerie.targets import reference_genome
from menagerie.skeleton import build_skeleton
from menagerie.muscles import build_muscles, render_points as _muscle_pts
from medic.skin_shell_head import shell, mesh as _skin_mesh


def _to_laid(P):
    """Atlas frame (col2 = height, col1 = ML, col0 = DV) -> laid frame (x=AP/height head+x, y=DV, z=ML)."""
    P = np.atleast_2d(P)
    return np.stack([P[:, 2], P[:, 0], P[:, 1]], axis=1)


def _dense_bellies(muscles, per=140, thick=0.26, rng=None):
    """A DENSE, volumetric fusiform belly per muscle (not a sparse line): fill the muscle's spindle -- points
    along the origin->insertion axis, at a radius that swells in the middle (sin^0.6, the fusiform belly) and
    tapers to the tendons, filling the perpendicular disk. `thick` = peak belly radius as a fraction of length.
    Returns the belly cells (laid frame). This is the mass that pushes the skin out into the muscle silhouette."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = []
    for m in muscles:
        o = _to_laid(m.origin)[0]; i = _to_laid(m.insertion)[0]
        seg = i - o; L = float(np.linalg.norm(seg)) + 1e-9
        axis = seg / L
        p1 = np.cross(axis, [0, 0, 1.0]); n1 = np.linalg.norm(p1)
        p1 = p1 / n1 if n1 > 1e-6 else np.array([1.0, 0, 0])
        p2 = np.cross(axis, p1)
        n = max(24, int(per * L / 0.25))                             # more cells for a longer muscle
        t = rng.random(n)
        r_prof = thick * L * np.sin(np.pi * t) ** 0.6                 # fusiform: swollen middle, tendon taper
        rr = r_prof * np.sqrt(rng.random(n)); th = 2 * np.pi * rng.random(n)
        pts = (o[None] + (t * L)[:, None] * axis[None]
               + (rr * np.cos(th))[:, None] * p1[None] + (rr * np.sin(th))[:, None] * p2[None])
        out.append(pts)
    return np.vstack(out) if out else np.zeros((0, 3))


def ventral_sign(body, F):
    """The body's anterior (ventral) DV sign, read off the cloud -- SHARED by flesh_skin + human_movie
    (feet point the way the face looks). PRIORITY ANCHOR (2026-08-30, Miles "the feet point the wrong way"):
    KIDNEY-vs-HEART -- the kidney is placed DORSAL of the heart by measured construction (dv_spread), both
    compact localized viscera, and the relation SURVIVES the movie's laying/maturation transform. (The first
    fix used heart-vs-spinal-cord, but the cord spans the whole curved trunk and its DV MEDIAN flipped sides
    in the laid frame: heart +0.01 vs cord +0.06 vs kidney -0.10 -- kidney-heart right, cord wrong.) The old
    eye-vs-brain rule flips when the eyes sit near the head's DV midline (~1%H dorsal of it = the original
    wrong-way feet). Fallbacks keep the old rules. Returns +1.0 or -1.0 on the laid-frame y axis."""
    from medic.unified_embryo import FIDX
    from medic.subhead_program import expand_names
    body = np.asarray(body, float); F = np.asarray(F)
    heart_ids = [FIDX[n] for n in ("Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle",
                                   "Outflow") if n in FIDX]
    kid_ids = [FIDX[n] for n in expand_names(["Kidney", "Nephron"]) if n in FIDX]
    hm, km = np.isin(F, heart_ids), np.isin(F, kid_ids)
    if hm.sum() > 20 and km.sum() > 20:
        return 1.0 if np.median(body[hm, 1]) >= np.median(body[km, 1]) else -1.0
    eye_ids = [FIDX[n] for n in ("Eye", "Retina") if n in FIDX]
    brain_ids = [FIDX[n] for n in ("Forebrain", "Midbrain", "Hindbrain") if n in FIDX]
    if eye_ids and brain_ids and np.isin(F, eye_ids).sum() > 8 and np.isin(F, brain_ids).sum() > 8:
        return 1.0 if np.median(body[np.isin(F, eye_ids), 1]) >= np.median(body[np.isin(F, brain_ids), 1]) else -1.0
    for nm in ("Notochord", "Spinal Cord"):
        if nm in FIDX and (F == FIDX[nm]).sum() > 20:
            return -1.0 if np.median(body[F == FIDX[nm], 1]) >= np.median(body[:, 1]) else 1.0
    return 1.0


def _fat_shell(cells, frac, rng):
    """The fat core: envelope of `cells` around ITS OWN axis, pushed radially out by frac of its radius."""
    S = shell(cells, n_slice=60, n_sec=40, dens=2)
    if not len(S):
        return cells[:0]
    cy, cz = np.median(cells[:, 1]), np.median(cells[:, 2])
    R = np.percentile(np.hypot(cells[:, 1] - cy, cells[:, 2] - cz), 95) + 1e-9
    d = np.hypot(S[:, 1] - cy, S[:, 2] - cz) + 1e-9
    push = 1.0 + frac * R / d
    F = S.copy()
    F[:, 1] = cy + (S[:, 1] - cy) * push
    F[:, 2] = cz + (S[:, 2] - cz) * push
    return F


def _subcutaneous_fat(muscle, frac=0.06, n=6000, rng=None, arm_mask=None):
    """A thin fat SHELL just outside the muscle mass -> the panniculus adiposus that rounds the muscle
    bellies into a smooth contour. Returns the fat points.

    v2 (2026-08-30, cycle 6): fat forms around EACH LEG separately below the fork. The v1 angular envelope
    was built around the WHOLE-BODY axis, so at leg level the sectors aiming between the legs closed over
    the gap -- a fat WEB across the crotch that re-bridged the thighs even after the tail-regression head
    cleared the cell residue. Fork detection is self-reading: AP slices in the lower body whose midline
    band is empty of layer cells are 'forked' -> per-side shells around each leg's own axis. On a body
    whose legs are not yet extended (the build frame) no slice tests forked and v1 behaviour is unchanged."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = muscle[:, 0]
    stat = np.ptp(x) + 1e-9
    apf = (x - x.min()) / stat
    mid = float(np.median(muscle[:, 2]))
    # v3 (2026-08-31): the v2 per-slice test was BISTABLE (the ~2% midline band sat on
    # a knife edge -> one build forked, the next fell back to the v1 skirt) and the
    # per-leg + torso shells shared NO cells at the fork line -> a fat seam at the hip
    # that marching-cubes could pinch into disconnected legs. v3: the fork is the
    # CONTIGUOUS run of empty-midline slices from the bottom up (one borderline slice
    # moves the crotch by one slice instead of flipping the whole mode), and the leg
    # and torso shells OVERLAP across a band at the fork so the fat is continuous
    # through the hip -- the panniculus does not stop at the perineum.
    # scan ceiling 0.42 -> 0.50 (cycle 17): the canonical crotch tops out at ~0.44-0.48 of stature;
    # with the ceiling at 0.42 the torso envelope started at 0.38 and draped fat across the open gap
    # (measured: the last 68+68 midline flesh points at 0.38-0.46 were ALL torso-shell fat). The
    # bridged-run stop (the perineal layer is midline-occupied) ends the fork at the true crotch.
    nb = 28
    slice_h = 0.50 / nb
    fork_hi = 0.0                                                   # fork top (apf); 0 = no fork
    bridged_run = 0
    for k in range(nb):
        lo = k * slice_h                                            # only the lower body can fork
        sl = (apf >= lo) & (apf < lo + slice_h)
        if sl.sum() < 40:
            fork_hi = lo + slice_h                                  # sparse slice: pass through
            continue
        # band 0.02 -> 0.01 stat (cycle 17): the limb-condensation wall puts the columns' inner
        # edges at ~0.011-0.021 stat -- a 0.02 band counts legitimate inner-column cells as
        # "bridging" and the fork never rises past ~25%. The test band must be NARROWER than the
        # condensation gap it is looking for.
        midcnt = (sl & (np.abs(muscle[:, 2] - mid) < 0.01 * stat)).sum()
        if midcnt < 0.02 * sl.sum():
            fork_hi = lo + slice_h
            bridged_run = 0
        else:
            bridged_run += 1                                        # one bridged slice may be debris;
            if bridged_run >= 2:                                    # two consecutive end the fork
                break
    forked = apf < fork_hi
    parts = []
    if fork_hi > slice_h and forked.sum() >= 100:
        band = 0.04                                                 # hip overlap band (frac of stature)
        for sgn in (-1.0, 1.0):
            leg = (apf < fork_hi + band) & ((muscle[:, 2] - mid) * sgn > 0)
            if leg.sum() >= 50:
                parts.append(_fat_shell(muscle[leg], frac, rng))
        # the torso shell starts AT the fork (cycle 17): its old downward band re-draped fat across
        # the open gap just below the crotch; hip continuity is already carried by the per-leg
        # shells extending UP through [fork_hi, fork_hi + band].
        up = apf > fork_hi
        am = arm_mask if arm_mask is not None else np.zeros(len(muscle), bool)
        if (up & am).sum() >= 100:
            # THE ARM FORK (cycle 17b, the leg law's sibling): the panniculus forms around each
            # ARM separately -- the single upper-body envelope swept its angular sectors across
            # the axilla, filling the armpit notch and the wing tips with fat (census: 912
            # cape-band + 22 wing-tip fat points). Per-arm shells around each arm's own axis;
            # the torso shell keeps the arm's top band for shoulder continuity (the hip pattern).
            tops = muscle[up & am, 0]
            sh_band = float(np.percentile(tops, 95)) - 0.04 * stat
            for sgn in (-1.0, 1.0):
                a = up & am & ((muscle[:, 2] - mid) * sgn > 0)
                if a.sum() >= 50:
                    parts.append(_fat_shell(muscle[a], frac, rng))
            parts.append(_fat_shell(muscle[up & (~am | (muscle[:, 0] > sh_band))], frac, rng))
        else:
            parts.append(_fat_shell(muscle[up], frac, rng))
    else:
        parts.append(_fat_shell(muscle, frac, rng))
    F = np.vstack([p for p in parts if len(p)]) if parts else muscle[:0]
    if len(F) > n:
        F = F[rng.choice(len(F), n, replace=False)]
    return F + rng.normal(size=F.shape) * 0.004


def build(species="human_male"):
    g = reference_genome(species)
    bones = build_skeleton(g)
    muscles = build_muscles(g, bones)
    muscle = _dense_bellies(muscles)                               # DENSE volumetric fusiform bellies
    fat = _subcutaneous_fat(muscle)
    layer = np.vstack([muscle, fat])                               # the flesh = muscle bellies + subcutaneous fat
    return dict(muscle=muscle, fat=fat, layer=layer, n_muscles=len(muscles))


def _silhouette(P, axis, n=90, plo=4, phi=96, smooth=3):
    """Per-AP-band low/high extent along `axis` (2=ML front, 1=DV side): a filled body outline in which the
    muscle bellies read as the widening of the contour. Smoothed along AP for a clean skin line."""
    from scipy.ndimage import uniform_filter1d
    x = P[:, 0]
    edges = np.linspace(x.min(), x.max(), n + 1)
    xs, lo, hi = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 6:
            continue
        xs.append(0.5 * (a + b)); lo.append(np.percentile(P[m, axis], plo)); hi.append(np.percentile(P[m, axis], phi))
    xs, lo, hi = np.array(xs), np.array(lo), np.array(hi)
    if smooth > 1 and len(xs) > smooth:
        lo = uniform_filter1d(lo, smooth); hi = uniform_filter1d(hi, smooth)
    return xs, lo, hi


def _model_muscle_bellies(body, F, n_per=60, rng=None):
    """Grow a dense fusiform belly for each named muscle from the body's OWN skeleton (carve_by_action_line gives
    O/I in the model frame) -- the mass that pushes the skin into the muscle silhouette. Self-contained (no
    integrated_body import -> no circular dep); returns muscle cells in the body frame, or empty on any failure."""
    if rng is None:
        rng = np.random.default_rng(0)
    try:
        from medic.limb_muscle_head import carve_by_action_line
        _mus, _assign, muscles, O, I = carve_by_action_line(body, F)
    except Exception:
        return np.zeros((0, 3))
    out = []
    for m in range(len(muscles)):
        o, i = np.asarray(O[m], float), np.asarray(I[m], float)
        seg = i - o; L = float(np.linalg.norm(seg))
        if L < 1e-6:
            continue
        u = seg / L
        ref = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(u, ref); e1 /= np.linalg.norm(e1) + 1e-9; e2 = np.cross(u, e1)
        t = rng.random(n_per)
        r = 0.09 * L * np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1)) * np.sqrt(rng.random(n_per))
        th = rng.random(n_per) * 2 * np.pi
        out.append(o + t[:, None] * seg + (r * np.cos(th))[:, None] * e1 + (r * np.sin(th))[:, None] * e2)
    return np.vstack(out) if out else np.zeros((0, 3))


def flesh_skin(body, F, push=0.06, layers=3, rng=None, with_bellies=True):
    """The MODEL-NATIVE flesh: given a model body cloud (laid frame x=AP,y=DV,z=ML) and its fates, drape the skin
    over the MUSCLE+FAT rather than the bone. The model's own Muscle-fate cells are thickened into SUPERFICIAL
    bellies (pushed radially toward the surface + jittered into mass) and a subcutaneous fat shell is added over
    them, then the model skin mesh envelopes the fleshed body -- so where the muscle is thick the skin bulges.
    Returns (skin_verts, skin_faces). Falls back to the plain body skin if there is no muscle."""
    from medic.unified_embryo import FIDX
    from medic.skin_shell_head import mesh as _model_skin
    if rng is None:
        rng = np.random.default_rng(0)
    mus_ids = [FIDX[n] for n in ("Muscle",) if n in FIDX]
    add = []
    if mus_ids:
        mus = body[np.isin(F, mus_ids)]
        if len(mus) > 20:
            cy, cz = float(np.median(body[:, 1])), float(np.median(body[:, 2]))
            d = np.hypot(mus[:, 1] - cy, mus[:, 2] - cz) + 1e-9
            R = np.percentile(np.hypot(body[:, 1] - cy, body[:, 2] - cz), 90) + 1e-9
            fac = 1.0 + push * R / d                                  # push muscle toward the surface (belly bulge)
            b = mus.copy(); b[:, 1] = cy + (mus[:, 1] - cy) * fac; b[:, 2] = cz + (mus[:, 2] - cz) * fac
            span = float(np.ptp(body)) + 1e-9
            for _ in range(layers):
                add.append(b + rng.normal(size=b.shape) * 0.012 * span)   # thicken into a belly of mass
    if with_bellies:                                             # grow real per-muscle bellies from the skeleton
        b = _model_muscle_bellies(body, F, rng=rng)              # (fixes thin legs: base has little leg Muscle-fate)
        if len(b):
            add.append(b)
    layer = np.vstack([body] + add) if add else body
    fat = _subcutaneous_fat(layer, frac=0.04, n=max(2000, len(layer) // 6), rng=rng)
    ventral = ventral_sign(body, F)
    # feet=False: the movie supplies its own genome-plausible autopods (human_movie.feet_mesh); building the
    # skin shell's schematic feet too gave the body FOUR feet.
    return _model_skin(np.vstack([layer, fat]), ventral=ventral, feet=False)


def _figure(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    muscle, layer = R["muscle"], R["layer"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 9), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(muscle[:, 2], muscle[:, 0], s=2, c="#b0403a", alpha=0.5)
    ax[0].set_title(f"muscle bellies ({R['n_muscles']} muscles)", color="#cbd5e1", fontsize=9)
    xs, lo, hi = _silhouette(layer, axis=2)                          # front (ML): muscle bulges widen the outline
    ax[1].fill_betweenx(xs, lo, hi, color="#d8b49a")
    ax[1].scatter(muscle[:, 2], muscle[:, 0], s=1, c="#a83c36", alpha=0.18)   # muscle showing through, faint
    ax[1].set_title("skin over the flesh — front", color="#e8c9a8", fontsize=9)
    xs2, lo2, hi2 = _silhouette(layer, axis=1)                       # side (DV)
    ax[2].fill_betweenx(xs2, lo2, hi2, color="#d8b49a")
    ax[2].set_title("skin — side", color="#e8c9a8", fontsize=9)
    fig.suptitle("The flesh: skin draped over the MUSCLE+FAT layer (not the bone) — the muscle silhouette reads "
                 "through the skin (deltoid, biceps, six-pack, gluteus, calf)", color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/flesh_surface_head.png", dpi=130, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/flesh_surface_head.png")


def main():
    R = build()
    print(f"flesh: {R['n_muscles']} muscles -> {len(R['muscle'])} belly cells + {len(R['fat'])} fat cells "
          f"-> skin silhouette over the flesh layer ({len(R['layer'])} cells)")
    _figure(R)


if __name__ == "__main__":
    main()
