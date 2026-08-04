"""
The visceral organ layer: the ~20 canonical mammalian organs, placed in the body cavity.

The third bucket of the homologous parts atlas (bones + Hill-muscles + organs). The viscera are
shared homologously across mammals and placed by anatomical address along the trunk cavity and
the head; each is deformed by the genome (body size, and organ-specific scaling) and carries the
liftable payload (genome parameters, cell-fate germ layer, boundary ellipsoid). Reuses the
organ-placement-on-the-body-frame idea of the embryo/clocks papers at gross-anatomy resolution.

Run: venv_win_new/Scripts/python.exe -m menagerie.organs
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .genome import Genome
from .skeleton import build_skeleton

# (name, ap_frac[0=shoulder..1=hip; <0 = head], dv[-1 ventral..+1 dorsal], lat[0 mid, ±paired],
#  rel_size, germ-layer fate, colour, paired)
ORGAN_PLAN = [
    ("brain",           -0.9, 0.0, 0.0, 0.55, "neural ectoderm",              (0.75, 0.75, 0.80), False),
    ("eye",             -0.8, 0.2, 0.6, 0.16, "neural ectoderm + crest",      (0.20, 0.35, 0.75), True),
    ("heart",            0.12, -0.15, 0.1, 0.45, "cardiac (splanchnic) mesoderm", (0.80, 0.12, 0.14), False),
    ("lung",             0.10, 0.15, 0.55, 0.5, "foregut endoderm + mesoderm",   (0.90, 0.55, 0.60), True),
    ("liver",            0.30, -0.35, 0.1, 0.7, "foregut endoderm",             (0.45, 0.22, 0.16), False),
    ("stomach",          0.36, 0.0, 0.45, 0.4, "foregut endoderm",             (0.78, 0.60, 0.42), False),
    ("spleen",           0.40, 0.25, 0.7, 0.22, "mesoderm",                    (0.5, 0.10, 0.14), False),
    ("pancreas",         0.42, 0.0, 0.2, 0.2, "foregut endoderm",              (0.85, 0.78, 0.45), False),
    ("kidney",           0.56, 0.55, 0.5, 0.3, "intermediate mesoderm",        (0.42, 0.18, 0.40), True),
    ("small intestine",  0.55, -0.15, 0.0, 0.75, "midgut endoderm",            (0.90, 0.65, 0.35), False),
    ("large intestine",  0.72, 0.05, 0.35, 0.5, "hindgut endoderm",            (0.80, 0.55, 0.30), False),
    ("bladder",          0.92, -0.35, 0.0, 0.25, "hindgut/urogenital endoderm",(0.85, 0.82, 0.40), False),
]


@dataclass
class Organ:
    name: str
    center: np.ndarray
    radii: np.ndarray
    fate: str
    color: tuple
    knobs: dict
    side: str = "axial"

    def boundary(self):
        return dict(center=self.center.tolist(), radii=self.radii.tolist())


def _cavity(g: Genome, bones):
    bmap = {b.name: b for b in bones}
    shoulder = bmap["T1"].a.copy()
    lumbars = [b for b in bones if b.name.startswith("L") and b.name[1:].isdigit()]
    hip = lumbars[-1].b.copy() if lumbars else shoulder + np.array([-1.0, 0, 0])
    cran = bmap.get("cranium")
    head = (cran.a + cran.b) / 2 if cran else shoulder + np.array([1.0, 0, 0])
    girth = 0.17 * g.trunk_girth * g.body_size
    return shoulder, hip, head, girth


def build_organs(g: Genome, bones=None):
    if bones is None:
        bones = build_skeleton(g)
    shoulder, hip, head, girth = _cavity(g, bones)
    trunk_vec = hip - shoulder
    up = np.array([0, 0, 1.0])
    out = []
    kn = lambda: {k: getattr(g, k) for k in ("body_size", "trunk_girth")}
    for (name, ap, dv, lat, size, fate, col, paired) in ORGAN_PLAN:
        if ap < 0:                       # head organ, placed relative to the skull
            base = head + np.array([ (abs(ap) - 0.8) * 0.2, 0, 0])
            anchor = base + up * (dv * girth * 0.6)
        else:
            anchor = shoulder + ap * trunk_vec + up * (dv * girth * 0.9)
        r = size * girth * np.array([1.15, 0.85, 0.85])
        for s in ([+1, -1] if paired else [0]):
            c = anchor + np.array([0, s * lat * girth, 0]) if paired else anchor + np.array([0, lat * girth, 0])
            out.append(Organ(name + (" R" if s > 0 else " L" if s < 0 else ""),
                             c.astype(float), r.astype(float), fate, col, kn(),
                             side="R" if s > 0 else "L" if s < 0 else "axial"))
    return out


def render_points(organs, seed=2, shaped=True):
    """Render organs. shaped=True uses the canonical shape basis + tissue subheads
    (organ_shapes.shape_points); shaped=False falls back to plain ellipsoids."""
    if shaped:
        from .organ_shapes import shape_points
        P, C = [], []
        for o in organs:
            p, c, _subs = shape_points(o, seed=seed)
            P.append(p); C.append(c)
        return np.vstack(P).astype(np.float32), np.vstack(C).astype(np.float32)
    rng = np.random.default_rng(seed)
    P, C = [], []
    for o in organs:
        n = min(600, max(40, int(220 * float(np.prod(o.radii)) / 0.002)))
        d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        rr = rng.random(n) ** (1 / 3)
        pts = o.center[None] + d * (o.radii[None]) * rr[:, None]
        P.append(pts); C.append(np.tile(o.color, (n, 1)))
    return np.vstack(P).astype(np.float32), np.clip(np.vstack(C), 0, 1).astype(np.float32)


if __name__ == "__main__":
    from .targets import reference_genome
    for sp in ["base_vertebrate", "giraffe", "elephant", "crocodile"]:
        g = reference_genome(sp) if sp != "base_vertebrate" else Genome()
        orgs = build_organs(g)
        print(f"{sp:16s} {len(orgs):2d} organs: {[o.name for o in orgs]}")
