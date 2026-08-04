"""
Canonical organ shapes as a shared basis + tissue SUBHEADS.

Each organ is a master-transcription-factor identity HEAD (Paper #4); its SHAPE is not an
ellipsoid but the arrangement of its constituent tissues, each of which is a SUBHEAD gated by the
organ head. This is the recursive-operator structure (organism -> organ -> tissue -> cell): the
organ head sets the envelope, the tissue subheads set the internal composition that gives the
right shape. The mean shape is the shared kernel (homologous across mammals); the genome (via the
outer LGM) supplies only the low-rank per-species DEFORMATION -- here, the size/aspect from the
organ's radii -- never the raw shape.

`shape_points(organ)` returns (points, colours, subhead-list). `SUBHEADS` documents the tissue
subheads per organ (the payload / the answer to "are we adding subheads for the tissues").
"""
from __future__ import annotations
import numpy as np

# organ -> tissue subheads (each a sub-identity the organ head gates)
SUBHEADS = {
    "heart":  ["myocardium", "left ventricle", "right ventricle", "left atrium", "right atrium"],
    "kidney": ["cortex", "medulla", "pelvis"],
    "lung":   ["cranial lobe", "middle lobe", "caudal lobe"],
    "liver":  ["left lobe", "right lobe", "caudate lobe"],
    "brain":  ["cerebrum (L)", "cerebrum (R)", "cerebellum", "brainstem"],
    "stomach":["fundus", "body", "pylorus"],
}


def _shade(rgb, f):
    return tuple(np.clip(np.array(rgb) * f, 0, 1))


def _blob(center, radii, n, rng):
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    rr = rng.random(n) ** (1 / 3)
    return np.asarray(center) + d * np.asarray(radii) * rr[:, None]


def shape_points(o, seed=2):
    """Dispatch to a canonical shape with tissue subheads; fall back to an ellipsoid."""
    rng = np.random.default_rng(seed + abs(hash(o.name)) % 1000)
    base = o.name.split(" ")[0] if o.name.split(" ")[0] in SUBHEADS else o.name
    c, r, col = o.center, o.radii, o.color
    P, C, subs = [], [], []

    def add(pts, color, sub):
        P.append(pts); C.append(np.tile(color, (len(pts), 1))); subs.append(sub)

    if base == "heart":
        # myocardial envelope + four chambers (subheads)
        add(_blob(c, r * 1.05, 260, rng), _shade(col, 0.75), "myocardium")
        off = r[1] * 0.45
        for (dz, dy, nm, sh) in [(+0.35, +off, "left ventricle", 1.0), (+0.35, -off, "right ventricle", 0.85),
                                 (-0.4, +off, "left atrium", 1.15), (-0.4, -off, "right atrium", 0.95)]:
            cc = c + np.array([r[0] * 0.2, dy, r[2] * dz])
            add(_blob(cc, r * 0.42, 120, rng), _shade(col, sh), nm)
    elif base == "kidney":
        add(_blob(c, r * np.array([1.0, 0.7, 1.15]), 200, rng), _shade(col, 1.0), "cortex")
        add(_blob(c, r * 0.55, 120, rng), _shade(col, 0.7), "medulla")
        add(_blob(c + np.array([0, -r[1] * 0.6, 0]), r * 0.3, 40, rng), _shade(col, 1.3), "pelvis")
    elif base == "lung":
        for k, (dz, sh, nm) in enumerate([(0.5, 1.0, "cranial lobe"), (0.0, 0.9, "middle lobe"),
                                          (-0.5, 0.8, "caudal lobe")]):
            add(_blob(c + np.array([r[0] * dz, 0, 0]), r * np.array([0.5, 0.9, 0.9]), 110, rng),
                _shade(col, sh), nm)
    elif base == "liver":
        for (dy, sh, nm) in [(+0.5, 1.0, "left lobe"), (-0.5, 0.85, "right lobe"), (0.0, 1.15, "caudate lobe")]:
            add(_blob(c + np.array([0, r[1] * dy, 0]), r * np.array([0.9, 0.7, 0.8]), 150, rng),
                _shade(col, sh), nm)
    elif base == "brain":
        for (dy, nm) in [(+0.45, "cerebrum (L)"), (-0.45, "cerebrum (R)")]:
            add(_blob(c + np.array([r[0] * 0.15, r[1] * dy, r[2] * 0.1], ), r * np.array([0.8, 0.55, 0.8]), 130, rng),
                _shade(col, 1.0), nm)
        add(_blob(c + np.array([-r[0] * 0.7, 0, -r[2] * 0.2]), r * 0.4, 70, rng), _shade(col, 0.85), "cerebellum")
        add(_blob(c + np.array([-r[0] * 0.9, 0, -r[2] * 0.6]), r * np.array([0.25, 0.25, 0.5]), 40, rng),
            _shade(col, 0.7), "brainstem")
    elif base == "stomach":
        for (dz, sh, nm) in [(0.5, 1.1, "fundus"), (0.0, 1.0, "body"), (-0.5, 0.9, "pylorus")]:
            add(_blob(c + np.array([r[0] * dz, 0, r[2] * dz * 0.4]), r * np.array([0.5, 0.9, 0.9]), 90, rng),
                _shade(col, sh), nm)
    else:
        add(_blob(c, r, 220, rng), col, base)      # single-tissue organ

    return (np.vstack(P).astype(np.float32),
            np.clip(np.vstack(C), 0, 1).astype(np.float32), subs)


def organ_subheads(name):
    base = name.split(" ")[0]
    return SUBHEADS.get(base if base in SUBHEADS else name, [name])
