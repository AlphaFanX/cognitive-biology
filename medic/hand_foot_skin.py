"""
hand_foot_skin.py -- skin-covered HANDS (five fingers) and FEET (five toes) at the limb tips.

The tube skin caps a limb with a smooth dome, which reads as a stump, not a hand. Real hands and feet are five
articulated digits fanning from a palm/sole -- the same five Shh-patterned rays the limb chondrogenesis head
already grows. This head builds that shape as skin: a palm (or sole) slab plus five short digit tubes fanned
across it, oriented to the limb's own axes, so the hand/foot is COVERED and its fingers/toes READ. Fixed
topology (a constant vert/face count) so it can be attached to the streamed movie mesh and track the moving tip.

Frame: laid (x=AP head+x, y=DV, z=ML). A hand/foot is placed by (tip, out_dir, up_dir, span, length).
"""
from __future__ import annotations
import numpy as np

_NSIDE = 6            # sides per digit tube
_NRING = 3            # rings along a digit
_NDIG = 5            # fingers / toes


def _digit(base, axis, up, length, radius, taper=0.6):
    """One finger/toe: a short tapering tube from `base` along `axis`, `up` sets the tube's cross-section frame.
    Returns (verts[_NRING*_NSIDE], faces) with a rounded tip cap folded into the last ring."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    u = up - axis * np.dot(up, axis); u = u / (np.linalg.norm(u) + 1e-9)
    w = np.cross(axis, u)
    verts = []
    for i in range(_NRING):
        t = i / (_NRING - 1)
        c = base + axis * (length * t)
        r = radius * (1.0 - taper * t) if i < _NRING - 1 else radius * 0.18   # taper, near-point tip
        for s in range(_NSIDE):
            a = 2 * np.pi * s / _NSIDE
            verts.append(c + r * (np.cos(a) * u + np.sin(a) * w))
    verts = np.array(verts)
    faces = []
    for i in range(_NRING - 1):
        for s in range(_NSIDE):
            s2 = (s + 1) % _NSIDE
            a, b = i * _NSIDE + s, i * _NSIDE + s2
            c, d = (i + 1) * _NSIDE + s, (i + 1) * _NSIDE + s2
            faces.append([a, b, d]); faces.append([a, d, c])
    return verts, np.array(faces)


def build(tip, out_dir, up_dir, span, length, kind="hand", flip=False):
    """A hand or foot at `tip`, fanning five digits from a palm/sole. `out_dir` points along the limb (fingers
    point away from the wrist), `up_dir` is the palm normal, `span` the palm width, `length` the digit length.
    `flip` reverses the digit fan across the palm (chirality) so the thumb / big toe sits on the opposite
    side -- needed to mirror a hand/foot to the contralateral side (big toes medial on BOTH feet).
    Returns (verts, faces) with FIXED counts (independent of inputs)."""
    out_dir = np.asarray(out_dir, float); out_dir = out_dir / (np.linalg.norm(out_dir) + 1e-9)
    up_dir = np.asarray(up_dir, float); up_dir = up_dir / (np.linalg.norm(up_dir) + 1e-9)
    spread = np.cross(out_dir, up_dir); spread = spread / (np.linalg.norm(spread) + 1e-9)  # across the palm
    if flip:
        spread = -spread                                   # mirror the fan for the contralateral side
    V, Fc, off = [], [], 0
    # digits fanned across the palm, slight length variation (thumb/big-toe shorter, middle longest)
    rel = np.array([0.72, 0.95, 1.0, 0.92, 0.75]) if kind == "hand" else np.array([1.0, 0.9, 0.78, 0.66, 0.55])
    rad = 0.20 * span
    for k in range(_NDIG):
        off_s = (k - (_NDIG - 1) / 2) / ((_NDIG - 1) / 2)     # -1..1 across the palm
        fan = 0.28 if kind == "hand" else 0.14                # fingers splay more than toes
        axis = out_dir + spread * off_s * fan
        base = np.asarray(tip, float) + spread * (off_s * span * 0.42)
        v, f = _digit(base, axis, up_dir, length * rel[k], rad)
        V.append(v); Fc.append(f + off); off += len(v)
    # palm / sole: a flat slab (two rings: wrist -> knuckles) so the digits do not float off the limb
    palm = []
    for i in range(2):
        c = np.asarray(tip, float) + out_dir * (i * 0.35 * length)
        for s in range(_NSIDE):
            a = 2 * np.pi * s / _NSIDE
            palm.append(c + span * 0.5 * np.cos(a) * spread + span * 0.28 * np.sin(a) * up_dir)
    palm = np.array(palm); pf = []
    for s in range(_NSIDE):
        s2 = (s + 1) % _NSIDE
        pf.append([s, s2, _NSIDE + s2]); pf.append([s, _NSIDE + s2, _NSIDE + s])
    V.append(palm); Fc.append(np.array(pf) + off)
    return np.vstack(V).astype(np.float32), np.vstack(Fc).astype(np.int32)


# fixed vert/face counts (so the movie can emit faces once)
NV = _NDIG * _NRING * _NSIDE + 2 * _NSIDE
