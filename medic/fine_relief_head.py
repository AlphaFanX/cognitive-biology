"""
fine_relief_head.py -- THE FINE RELIEF: the specific muscle SHAPES that read as a body -- the six-pack and the
biceps peak. Generic fusiform bellies give gross bulk; the recognisable surface comes from each muscle's OWN
form and, as much as the bulges, the GROOVES between them.

  * RECTUS ABDOMINIS (the six-pack): NOT one bulge but two vertical columns flanking the midline, each cut into
    blocks. The relief is made by three tethers: the LINEA ALBA (the midline vertical groove where the skin is
    bound down to the deep fascia), the TENDINOUS INTERSECTIONS (three horizontal grooves that segment each
    column into four packs), and the SEMILUNAR line (the lateral border against the obliques). The packs are the
    bulges BETWEEN those grooves -- so it is the grooves that carve the six-pack, not the muscle.
  * BICEPS BRACHII (the peak): a fusiform belly whose apex is drawn to a rounded PEAK at the mid-upper arm when
    the muscle shortens -- the height field rises to a single crest, not an even swell.

Modelled as a skin-DEPTH height field h(u,v) (the surface's rise toward the viewer) = bulges minus groove
tethers, then hill-shaded so the relief reads. This is the shape the fleshed skin must take; the flesh head
(medic.flesh_surface_head) supplies the mass, this supplies the FORM.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.fine_relief_head
Out: data/organ_cascade/fine_relief_head.png
"""
from __future__ import annotations
import os
import numpy as np


def _g(u, v, u0, v0, su, sv):
    return np.exp(-(((u - u0) / su) ** 2 + ((v - v0) / sv) ** 2))


def sixpack(nu=260, nv=340):
    """Skin-depth height field of the anterior abdomen: the rectus six-pack. u = ML (-1..1), v = AP (0 pubis ..
    1 sternum). Bulges = the eight packs; troughs = the linea alba + three tendinous intersections + the two
    semilunar borders."""
    u = np.linspace(-1, 1, nu); v = np.linspace(0, 1, nv)
    U, V = np.meshgrid(u, v)
    h = 0.10 * _g(U, V, 0, 0.5, 1.6, 0.9)                         # the abdomen's own gentle dome
    # eight packs: two columns at u = +-0.28, four rows (larger + more separated toward the top)
    cols = (-0.28, 0.28)
    rows = (0.30, 0.47, 0.63, 0.80)                              # pubis-side small, sternum-side large
    for j, v0 in enumerate(rows):
        amp = 0.16 + 0.05 * j                                    # upper packs stand prouder
        sv = 0.055 + 0.006 * j
        for u0 in cols:
            h += amp * _g(U, V, u0, v0, 0.20, sv)
    # the two lowest "packs" fuse below the navel into a single mound -> soften the bottom pair
    h -= 0.04 * _g(U, V, -0.28, 0.30, 0.20, 0.05) + 0.04 * _g(U, V, 0.28, 0.30, 0.20, 0.05)
    # GROOVES (skin tethered down to the fascia): linea alba (midline) + 3 tendinous intersections + semilunar
    h -= 0.13 * np.exp(-((U - 0.0) / 0.045) ** 2)                # linea alba: the central vertical groove
    for vt in (0.385, 0.55, 0.715):                              # tendinous intersections: horizontal grooves
        h -= 0.11 * np.exp(-((V - vt) / 0.028) ** 2) * (np.abs(U) < 0.52)
    for us in (-0.52, 0.52):                                     # semilunar lines: the lateral rectus borders
        h -= 0.08 * np.exp(-((U - us) / 0.05) ** 2) * (V > 0.2) * (V < 0.9)
    h += 0.05 * (np.abs(U) > 0.55) * (0.6 - 0.4 * V)             # the flanking obliques slope away
    return u, v, h


def biceps(nu=200, nv=320):
    """Skin-depth height field of the anterior upper arm: the biceps peak. u = arm width, v = shoulder(0)..elbow(1)."""
    u = np.linspace(-1, 1, nu); v = np.linspace(0, 1, nv)
    U, V = np.meshgrid(u, v)
    h = 0.06 * _g(U, V, 0, 0.5, 1.4, 1.1)                        # the arm cylinder
    h += 0.42 * _g(U, V, 0, 0.55, 0.62, 0.20)                    # the biceps belly, apex mid-upper-arm
    h += 0.10 * _g(U, V, 0, 0.42, 0.34, 0.10)                    # the PEAK: a proud crest on the belly
    h -= 0.05 * np.exp(-((V - 0.86) / 0.05) ** 2)                # the tendon dips toward the elbow
    h -= 0.04 * (np.abs(U) > 0.6)                                # fall away at the arm's sides
    return u, v, h


def _height(u, v, hu, hv, h):
    """Sample the height field h (on grid hu x hv) at scattered (u,v), 0 outside the patch."""
    iu = np.clip(np.searchsorted(hu, u) - 1, 0, len(hu) - 1)
    iv = np.clip(np.searchsorted(hv, v) - 1, 0, len(hv) - 1)
    out = h[iv, iu]
    out[(u < hu.min()) | (u > hu.max()) | (v < hv.min()) | (v > hv.max())] = 0.0
    return out


def apply_to_body(skin_v, body, F, amp=0.11, abd=(0.44, 0.66)):
    """Sculpt the six-pack into the body's ventral-abdomen SKIN: map the height field onto the abdomen patch and
    displace each ventral surface vertex along DV -- OUT over the packs, IN at the linea alba / tendinous-
    intersection grooves (the skin tethered to the fascia). Returns (skin_v', relief-per-vertex). Laid frame
    (x=AP head+x, y=DV, z=ML). Ventral is read as the DV side OPPOSITE the spinal cord."""
    from medic.unified_embryo import FIDX
    V = np.asarray(skin_v, float).copy()
    Hb = float(np.ptp(body[:, 0])) + 1e-9
    apf = (V[:, 0] - body[:, 0].min()) / Hb
    # ventral sign: opposite the dorsal neural tube (spinal cord); fallback to the body's own DV skew
    dsn = 1.0
    for nm in ("Spinal Cord", "Notochord"):
        if nm in FIDX and (F == FIDX[nm]).sum() > 20:
            dsn = -1.0 if np.median(body[F == FIDX[nm], 1]) >= np.median(body[:, 1]) else 1.0
            break
    ymid = float(np.median(body[:, 1]))
    ventral = (dsn * (V[:, 1] - ymid) > 0)                       # verts on the belly side
    half = np.percentile(np.abs(V[:, 2]), 88) + 1e-9            # abdomen half-width (ML)
    band = ventral & (apf > abd[0]) & (apf < abd[1]) & (np.abs(V[:, 2]) < 1.15 * half)
    hu, hv, h = sixpack()
    u = np.clip(V[band, 2] / half, -1, 1)                       # ML -> u
    v = np.clip((apf[band] - abd[0]) / (abd[1] - abd[0]), 0, 1)  # AP -> v
    hv_rel = _height(u, v, hu, hv, h) - 0.14                    # subtract the abdominal dome so grooves go NEGATIVE
    relief = np.zeros(len(V))
    relief[band] = hv_rel
    V[band, 1] += dsn * amp * Hb * hv_rel                       # displace along DV: out at packs, in at grooves
    return V, relief


# ---- the surface-muscle relief library: every major muscle from ONE primitive (an anatomical bulge lobe) ------
# Each lobe is (apf0, dv_sign, ml_band, amp, s_ap, s_ml) -- a fusiform swelling centred at axial level apf0, on
# the ventral(+1)/dorsal(-1)/either(0) side, over an ML band (|z|/half in [lo,hi]), pushing the skin radially OUT
# by amp (fraction of body radius). Bilateral muscles are one lobe applied to both ML signs. This is the Economy
# principle: ~one mechanism (the lobe) instantiates the whole surface musculature.
_MUSCLE_LOBES = [
    # name              apf0  dv   ml_band     amp    s_ap  s_ml
    ("deltoid",         0.80, 0.0, (0.55, 1.15), 0.12, 0.05, 0.35),   # the rounded shoulder cap (bilateral)
    ("pectoralis",      0.73, +1.0, (0.05, 0.62), 0.10, 0.05, 0.34),  # the chest slab (ventral, bilateral)
    ("trapezius",       0.79, -1.0, (0.00, 0.45), 0.07, 0.07, 0.40),  # the upper-back sheet (dorsal, para-midline)
    ("latissimus",      0.64, -1.0, (0.30, 0.95), 0.07, 0.09, 0.40),  # the dorsal wing (back, lateral)
    ("erector_spinae",  0.58, -1.0, (0.02, 0.28), 0.06, 0.22, 0.16),  # the two paraspinal columns (long, dorsal)
    ("gluteus",         0.47, -1.0, (0.10, 0.85), 0.13, 0.06, 0.45),  # the buttock flare (dorsal, bilateral)
]


def _radial_relief(skin_v, body, F, amp=1.0):
    """Add every surface-muscle bulge to the trunk skin as a RADIAL push from the body axis: for each vertex,
    sum the muscle lobes it falls under and push it out from the DV-ML axis by that amount. Bilateral, ventral/
    dorsal-aware. Returns (skin_v', relief-per-vertex). Laid frame (x=AP head+x, y=DV, z=ML)."""
    from medic.unified_embryo import FIDX
    V = np.asarray(skin_v, float).copy()
    x = body[:, 0]; xmin = x.min(); H = np.ptp(x) + 1e-9
    apf = (V[:, 0] - xmin) / H
    cy, cz = float(np.median(body[:, 1])), float(np.median(body[:, 2]))
    # ventral sign = opposite the dorsal spinal cord / notochord
    dsn = 1.0
    for nm in ("Spinal Cord", "Notochord"):
        if nm in FIDX and (F == FIDX[nm]).sum() > 20:
            dsn = -1.0 if np.median(body[F == FIDX[nm], 1]) >= np.median(body[:, 1]) else 1.0
            break
    dv = dsn * (V[:, 1] - cy)                                  # >0 ventral, <0 dorsal
    half = np.percentile(np.abs(V[:, 2] - cz), 90) + 1e-9
    mlf = np.abs(V[:, 2] - cz) / half                          # |ML| as a fraction of half-width
    Rrad = np.hypot(V[:, 1] - cy, V[:, 2] - cz) + 1e-9
    relief = np.zeros(len(V))
    for _nm, apf0, dvs, (mlo, mhi), a, sap, sml in _MUSCLE_LOBES:
        g = np.exp(-((apf - apf0) / sap) ** 2)                 # axial lobe
        g *= (mlf > mlo) & (mlf < mhi)                         # ML band (both sides -> bilateral)
        mlc = 0.5 * (mlo + mhi)
        g = g * np.exp(-((mlf - mlc) / sml) ** 2)              # taper within the band -> a rounded belly
        if dvs != 0:                                           # ventral / dorsal gating (soft)
            g = g * np.clip(dvs * dv / (0.35 * half) + 0.5, 0.0, 1.0)
        relief += a * g
    push = amp * relief * H / Rrad
    V[:, 1] = cy + (V[:, 1] - cy) * (1.0 + push)
    V[:, 2] = cz + (V[:, 2] - cz) * (1.0 + push)
    return V, relief


def _limb_relief(skin_v, body, F, amp=1.0):
    """The LIMB muscles, as radial bulges around each limb's OWN axis (not the body axis): the BICEPS peak on the
    upper arm, the QUADRICEPS sweep + HAMSTRING on the thigh, and the GASTROCNEMIUS belly that swells the upper
    calf then TAPERS to the ankle. Arms hang at the sides, legs below, so the along-limb coordinate is apf; each
    limb is found by its ML sign and pushed out around its local centre. Returns (skin_v', relief). Laid frame."""
    V = np.asarray(skin_v, float).copy()
    x = body[:, 0]; xmin = x.min(); H = np.ptp(x) + 1e-9
    apf = (V[:, 0] - xmin) / H
    cy, cz = float(np.median(body[:, 1])), float(np.median(body[:, 2]))
    half = np.percentile(np.abs(V[:, 2] - cz), 90) + 1e-9
    relief = np.zeros(len(V))
    def _push(m, relief_m):
        """Displace limb verts OUTWARD from the limb's OWN axis by a BOUNDED absolute amount (relief_m*H), so a
        thin limb doesn't blow up the way the trunk's multiplicative radial push would (Rr can be tiny here)."""
        idx = np.where(m)[0]
        aly, alz = float(np.median(V[idx, 1])), float(np.median(V[idx, 2]))
        dy, dz = V[idx, 1] - aly, V[idx, 2] - alz
        Rr = np.hypot(dy, dz) + 1e-9
        disp = amp * relief_m * H                             # absolute outward magnitude (bounded)
        V[idx, 1] += disp * dy / Rr; V[idx, 2] += disp * dz / Rr
        relief[idx] += relief_m
    # ---- ARMS: outboard of the trunk, mid-AP; biceps/triceps belly on the upper arm
    arm = (apf > 0.42) & (apf < 0.80) & (np.abs(V[:, 2] - cz) > 0.55 * half)
    for s in (+1.0, -1.0):
        m = arm & (np.sign(V[:, 2] - cz) == s)
        if m.sum() < 6:
            continue
        along = np.clip((0.80 - apf[m]) / 0.38, 0, 1)         # 0 shoulder -> 1 hand
        _push(m, 0.09 * np.exp(-((along - 0.32) / 0.16) ** 2))   # biceps peak on the upper arm
    # ---- LEGS: below the crotch; quad+hamstring on the thigh, gastrocnemius bulging then tapering on the calf
    leg = apf < 0.36
    for s in (+1.0, -1.0):
        m = leg & (np.sign(V[:, 2] - cz) == s)
        if m.sum() < 6:
            continue
        along = np.clip((0.36 - apf[m]) / 0.36, 0, 1)         # 0 hip -> 1 foot
        thigh = 0.11 * np.exp(-((along - 0.22) / 0.16) ** 2)  # quadriceps/hamstring sweep (upper thigh, broad)
        calf = 0.09 * np.exp(-((along - 0.60) / 0.10) ** 2)   # gastrocnemius belly (upper calf)
        calf *= np.clip(1.0 - (along - 0.60) / 0.30, 0.2, 1.0)  # then TAPER toward the ankle
        _push(m, thigh + calf)
    return V, relief


def body_relief(skin_v, body, F, amp=1.0):
    """THE FULL SURFACE RELIEF: drape all the major surface muscles + the six-pack grooves onto the body skin so
    the musculature reads through -- the deltoid cap, the pectoral slab, the trapezius/latissimus of the back,
    the paraspinal columns, the gluteal flare (radial bulges), plus the rectus six-pack (a DV height field with
    its carving grooves). One call, the whole silhouette. Returns (skin_v', total-relief-per-vertex)."""
    V, r1 = _radial_relief(skin_v, body, F, amp=amp)
    V, r2 = apply_to_body(V, body, F, amp=0.11 * amp)          # the six-pack packs + grooves, on top
    return V, r1 + np.abs(r2)


def _shade(ax, u, v, h, title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=210, altdeg=42)
    rgb = ls.shade(h, cmap=plt.cm.copper, vert_exag=6.0, blend_mode="soft")
    ax.imshow(rgb, extent=[u.min(), u.max(), v.min(), v.max()], origin="lower", aspect="equal")
    ax.axis("off"); ax.set_title(title, color="#e8c9a8", fontsize=10)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 8), facecolor="#0d1017")
    u1, v1, h1 = sixpack(); _shade(ax[0], u1, v1, h1, "rectus abdominis — the six-pack\n(packs + linea alba + tendinous intersections)")
    u2, v2, h2 = biceps(); _shade(ax[1], u2, v2, h2, "biceps brachii — the peak")
    fig.suptitle("The fine relief: the specific muscle SHAPES + the GROOVES between them (skin-depth, hill-shaded)",
                 color="#e2e8f0", fontsize=10)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/fine_relief_head.png", dpi=140, facecolor="#0d1017", bbox_inches="tight")
    print("saved data/organ_cascade/fine_relief_head.png")


if __name__ == "__main__":
    main()
