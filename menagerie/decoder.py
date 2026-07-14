"""
Decode a genome into a body: the conserved tetrapod Bauplan the knobs deform.

Two paths:
  morphometrics(g) / feature_vector(g)  -- analytic, fast; used by the evolution strategy
  build_parcels(g)                      -- the 3-D tissue-conglomerate point cloud, for figures

The axial counts are FROZEN (7 cervical, 13 thoracic, 6 lumbar, 4 sacral, ~12 caudal x tail_count);
species differ by the PER-SEGMENT scale, never the count -- the giraffe-neck invariant.
"""
from __future__ import annotations
import numpy as np
from .genome import (Genome, CERVICAL_COUNT, THORACIC_COUNT, LUMBAR_COUNT,
                     SACRAL_COUNT, BASE_CAUDAL)

# base per-segment lengths (world units ~ metres for a generic ~1.8 m tetrapod)
L_CERV, L_TRUNK, L_SACR, L_CAUD = 0.11, 0.105, 0.09, 0.085
SKULL_BASE = 0.28
LIMB_BASE = 0.62
GIRTH_BASE = 0.17


# ----------------------------------------------------------------------------- analytic
def morphometrics(g: Genome) -> dict:
    bs = g.body_size
    neck = CERVICAL_COUNT * L_CERV * g.cervical_elong * bs
    trunk = (THORACIC_COUNT + LUMBAR_COUNT) * L_TRUNK * g.trunk_len * bs
    n_caud = max(1, round(BASE_CAUDAL * g.tail_count))
    tail = n_caud * L_CAUD * g.tail_len * bs
    limb = LIMB_BASE * g.limb_len * bs
    girth = GIRTH_BASE * g.trunk_girth * bs
    skull = SKULL_BASE * g.skull_size * bs
    snout = skull * g.snout_len
    # withers height: leg + trunk half-girth; head height adds the raised neck
    neck_angle = np.deg2rad(8 + 74 * g.neck_raise)
    head_z = limb + girth + neck * np.sin(neck_angle)
    withers = limb + 2 * girth
    height = max(withers, head_z)
    total_len = snout + neck * np.cos(neck_angle) + trunk + tail
    mass = (np.pi * girth ** 2 * trunk) + 0.15 * (np.pi * (0.09 * g.limb_gracility * bs) ** 2 * limb) * 4
    return dict(neck=neck, trunk=trunk, tail=tail, limb=limb, girth=girth, skull=skull,
                snout=snout, withers=withers, head_z=head_z, height=height,
                total_len=total_len, mass=mass, n_caudal=n_caud,
                horn=g.horn_size * bs, tusk=g.tusk_len * bs, proboscis=g.proboscis_len * bs,
                gracility=g.limb_gracility)


# feature vector used for fitness (world-scale measures; ratios kept implicit via the set)
FEATURES = ["height", "neck", "trunk", "limb", "girth", "tail", "snout",
            "horn", "tusk", "proboscis", "gracility"]


def feature_vector(g: Genome) -> np.ndarray:
    m = morphometrics(g)
    return np.array([m[k] for k in FEATURES], dtype=float)


# ----------------------------------------------------------------------------- geometry
def _spine(g: Genome):
    """Build the axial midline as a list of (center, radius, part) in the AP(x)-DV(z) plane,
    y = 0 on the midline. Returns points anchored so the feet touch z = 0."""
    m = morphometrics(g)
    bs = g.body_size
    # shoulder at origin-x; build trunk backward, neck forward.
    shoulder_z = m["limb"] + m["girth"]
    hip_z = m["limb"] + m["girth"]                      # (kept level; slope emerges via legs)
    seg = []  # (x, z, r, part)

    # --- trunk: shoulder (x=0) backward to hip ---
    x = 0.0
    r_tr = m["girth"]
    n_trunk = THORACIC_COUNT + LUMBAR_COUNT
    for i in range(n_trunk + 1):
        f = i / n_trunk
        z = shoulder_z + (hip_z - shoulder_z) * f
        # girth swells at the belly (mid-trunk), tapers at both ends
        r = r_tr * (0.72 + 0.5 * np.sin(np.pi * min(1.0, max(0.0, f))) )
        seg.append((x, z, r, "trunk"))
        x -= L_TRUNK * g.trunk_len * bs
    hip_x = x
    # --- sacrum + tail: from hip backward and drooping down ---
    ta = np.deg2rad(28)
    xc, zc = hip_x, hip_z
    n_caud = m["n_caudal"]
    for i in range(SACRAL_COUNT + n_caud):
        L = (L_SACR if i < SACRAL_COUNT else L_CAUD * g.tail_len) * bs
        xc -= L * np.cos(ta)
        zc -= L * np.sin(ta)
        rr = m["girth"] * (0.55 * (1 - i / (SACRAL_COUNT + n_caud)) + 0.06)
        seg.append((xc, max(0.02, zc), rr, "tail"))

    # --- neck: shoulder forward and up ---
    na = np.deg2rad(8 + 74 * g.neck_raise)
    xc, zc = 0.0, shoulder_z
    neck_pts = []
    for i in range(CERVICAL_COUNT):
        L = L_CERV * g.cervical_elong * bs
        xc += L * np.cos(na)
        zc += L * np.sin(na)
        rr = m["girth"] * (0.34 + 0.10 * (1 - i / CERVICAL_COUNT))
        neck_pts.append((xc, zc, rr, "neck"))
    seg += neck_pts
    head_anchor = (xc, zc)
    return seg, head_anchor, m


def _rot_y(vec, ang):
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([c * x + s * z, y, -s * x + c * z], -1)


def _coat_color(P, g: Genome, base_rgb):
    """Procedural reaction-diffusion-style coat over body-wall parcels."""
    n = len(P)
    col = np.tile(np.array(base_rgb, float), (n, 1))
    w = 1.0 / max(0.25, g.coat_wavelength)
    t = g.coat_type
    dark = np.array([0.20, 0.15, 0.10])
    if t == "plain":
        return col
    if t == "spot":     # leopard-ish small spots / lion cub rosette-free
        u = (np.sin(P[:, 0] * 20 * w) * np.sin(P[:, 1] * 20 * w) * np.sin(P[:, 2] * 20 * w))
        col[u > 0.35] = dark
    elif t == "rosette":  # leopard rosettes (ring clusters)
        u = np.sin(P[:, 0] * 16 * w) + np.sin(P[:, 1] * 16 * w) + np.sin(P[:, 2] * 16 * w)
        ring = (np.abs(u) > 1.4) & (np.abs(u) < 2.1)
        col[ring] = dark
    elif t == "patch":   # giraffe reticulated patches (large polygonal)
        u = (np.sin(P[:, 0] * 7 * w) * np.cos(P[:, 1] * 7 * w)
             + np.sin(P[:, 2] * 7 * w) * np.cos(P[:, 0] * 7 * w))
        col[u > 0.15] = np.array([0.55, 0.32, 0.12])
        col[np.abs(u) <= 0.15] = np.array([0.93, 0.86, 0.70])   # cream seams
    elif t == "stripe":
        u = np.sin(P[:, 0] * 26 * w)
        col[u > 0] = dark
    elif t == "armor":   # crocodile osteoderm segmentation
        u = np.sin(P[:, 0] * 30 * w)
        col[u > 0.4] = np.array([0.28, 0.33, 0.24])
    return col


def build_parcels(g: Genome, seed: int = 0, base_rgb=(0.74, 0.62, 0.47)):
    """Return (P Nx3, colors Nx3, parts list) tissue-conglomerate cloud for rendering."""
    rng = np.random.default_rng(seed)
    seg, head, m = _spine(g)
    bs = g.body_size
    body_rgb = list(base_rgb)
    P, C, parts = [], [], []

    # body wall: ring of parcels around each axial center (in the x-z plane, extruded in y)
    for (x, z, r, part) in seg:
        npts = max(8, int(46 * r / GIRTH_BASE))
        th = rng.uniform(0, 2 * np.pi, npts)
        rr = r * (1 + rng.normal(0, 0.05, npts))
        yy = rr * np.cos(th)
        zz = z + rr * np.sin(th)
        # ventral belly sag on the trunk
        if part == "trunk":
            zz -= 0.10 * r * (np.sin(th) < 0)
        xx = x + rng.normal(0, 0.02 * bs, npts)
        pts = np.stack([xx, yy, zz], 1)
        P.append(pts); parts += [part] * npts
        C.append(_coat_color(pts, g, body_rgb))

    # skull at the neck tip (ellipsoid), plus snout extension forward
    hx, hz = head
    na = np.deg2rad(8 + 74 * g.neck_raise)
    sk = m["skull"]
    fwd = np.array([np.cos(na), 0, np.sin(na)])
    d = rng.normal(size=(360, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    scale = np.array([0.34 * sk * g.snout_len, 0.22 * sk, 0.24 * sk])
    skull = np.array([hx, 0, hz]) + (d * scale) + fwd * 0.30 * sk
    P.append(skull); C.append(np.tile([0.78, 0.66, 0.5], (len(skull), 1))); parts += ["skull"] * len(skull)

    def cone(base, tip, r0, npts, jitter=0.02):
        t = np.sqrt(rng.random(npts))
        axis = base[None] + t[:, None] * (tip - base)[None]
        rad = r0 * (1 - 0.85 * t)
        return axis + rng.normal(0, 1, (npts, 3)) * (rad[:, None] + jitter * 0.0)

    snout_tip = np.array([hx, 0, hz]) + fwd * (0.55 * sk * g.snout_len)

    # --- cranial appendages (discrete) ---
    if g.horn_mode == "single_median" and g.horn_size > 0:  # rhino: on the snout midline (LR node)
        base = snout_tip + np.array([0, 0, 0.10 * sk])
        tip = base + np.array([0.15, 0, 0.55]) * g.horn_size * bs
        h = cone(base, tip, 0.06 * bs, 260)
        P.append(h); C.append(np.tile([0.85, 0.82, 0.74], (len(h), 1))); parts += ["horn"] * len(h)
    elif g.horn_mode == "paired" and g.horn_size > 0:        # buffalo: paired, lateral (frame+lat.inhib.)
        for s in (+1, -1):
            base = np.array([hx, s * 0.18 * sk, hz + 0.14 * sk])
            tip = base + np.array([-0.05, s * 0.5, 0.45]) * g.horn_size * bs
            h = cone(base, tip, 0.05 * bs, 200)
            P.append(h); C.append(np.tile([0.16, 0.14, 0.13], (len(h), 1))); parts += ["horn"] * len(h)

    if g.dentition == "tusks" and g.tusk_len > 0:            # elephant: elongated incisors
        for s in (+1, -1):
            base = np.array([hx, s * 0.09 * sk, hz - 0.10 * sk]) + fwd * 0.2 * sk
            tip = base + fwd * (0.8 * g.tusk_len * bs) + np.array([0, 0, -0.25 * g.tusk_len * bs])
            tk = cone(base, tip, 0.03 * bs, 150)
            P.append(tk); C.append(np.tile([0.92, 0.90, 0.82], (len(tk), 1))); parts += ["tusk"] * len(tk)

    if g.nose == "trunk" and g.proboscis_len > 0:            # elephant trunk: hanging muscular hydrostat
        npts = 320
        t = np.linspace(0, 1, npts)
        curl = 0.35 * g.proboscis_len * bs
        base = np.array([hx, 0, hz]) + fwd * (0.5 * sk)
        path = base[None] + t[:, None] * (fwd * (0.7 * g.proboscis_len * bs))[None]
        path[:, 2] -= (t ** 1.6) * (g.proboscis_len * bs)          # droops down
        path[:, 0] += (t ** 2) * curl
        rad = 0.07 * bs * (1 - 0.6 * t)
        tr = path + rng.normal(0, 1, (npts, 3)) * rad[:, None]
        P.append(tr); C.append(np.tile([0.55, 0.46, 0.36], (len(tr), 1))); parts += ["trunk_nose"] * len(tr)

    # --- limbs: fore at shoulder (x~0), hind at hip; sprawling posture splays them out ---
    limb = m["limb"]; thick = 0.09 * g.limb_gracility * bs
    hip_x = min(s[0] for s in seg if s[3] in ("trunk",))
    sprawl = 0.55 if g.posture == "sprawling" else 0.0
    for (ax, shoulder_z) in [(0.0, seg[0][1]), (hip_x, seg[THORACIC_COUNT + LUMBAR_COUNT][1])]:
        for s in (+1, -1):
            top = np.array([ax, s * 0.12 * bs, shoulder_z])
            foot = np.array([ax + rng.normal(0, 0.03), s * (0.12 + sprawl) * bs, 0.02])
            npts = 220
            t = np.sqrt(rng.random(npts))
            axis = top[None] + t[:, None] * (foot - top)[None]
            rad = thick * (1.15 - 0.5 * t)
            lp = axis + rng.normal(0, 1, (npts, 3)) * rad[:, None]
            P.append(lp); C.append(np.tile([0.62, 0.52, 0.4], (len(lp), 1))); parts += ["limb"] * len(lp)

    P = np.vstack(P).astype(np.float32)
    C = np.clip(np.vstack(C), 0, 1).astype(np.float32)
    return P, C, parts
