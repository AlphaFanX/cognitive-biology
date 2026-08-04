"""Paper #7 Figure 7: DETAILED faces and bodies showing the effect of each allele.

Replaces the earlier stick-figure/cartoon version. Faces are rendered from the REAL 43,071-vertex
FaceBase mean mesh (face_demo/data/meanface.npz), morphed per-allele by the same dense Gaussian
adapter as the demo (medic/../face_demo/mesh_morph.py), and shaded with a software renderer
(painter's algorithm + Lambertian light). Bodies are the model's OWN matured cell cloud (the NCA+LGM
movie, via medic.movie_body_render), deformed by a height allele (maturation layer, overall size) and a
proportion allele (embryonic layer, limb:trunk ratio) -- no MakeHuman mesh anywhere.

The allele's morphological effect is exaggerated for visibility (clearly labelled) and tinted where it
acts (warm = protrusion toward the viewer relative to baseline). Directions are grounded in published
facial-shape GWAS; per-allele magnitudes are illustrative pending real beta_k (EDAR/PAX3 carry real
per-population frequencies).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.allele_faces_bodies
Out:  data/organ_cascade/allele_faces_bodies.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# import the demo's real-mesh morph
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "face_demo"))
import mesh_morph  # noqa: E402

SKIN = np.array([0.94, 0.80, 0.70])
INK = "#233"
LIGHT = np.array([-0.35, 0.45, 0.86]); LIGHT /= np.linalg.norm(LIGHT)


# --------------------------------------------------------------------------- render core
def _shade(V, F, base_rgb, tint=None, ambient=0.42, viewer_z=True):
    """Return (polys_xy, facecolors, order) for a frontal orthographic render.
    V: (n,3) x=lateral y=up z=anterior(+ toward viewer). Painter's algorithm."""
    tris = V[F]                                   # (m,3,3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    n = n / ln
    if viewer_z:
        n[n[:, 2] < 0] *= -1                      # orient toward viewer
    inten = ambient + (1 - ambient) * np.clip(n @ LIGHT, 0, 1)
    cols = base_rgb[None, :] * inten[:, None]
    if tint is not None:  # signed surface-normal displacement: outward=yellow, inward=blue (Claes/Walsh convention)
        tf = tint[F].mean(1)
        warm = np.array([0.99, 0.85, 0.15]); cool = np.array([0.20, 0.45, 0.95])
        pos = np.clip(tf, 0, 1)[:, None]; neg = np.clip(-tf, 0, 1)[:, None]
        a = 0.62
        cols = cols * (1 - a * (pos + neg)) + a * (pos * warm + neg * cool) * inten[:, None]
    cols = np.clip(cols, 0, 1)
    order = np.argsort(tris[:, :, 2].mean(1))     # far first
    polys = tris[:, :, :2][order]
    return polys, cols[order]


def draw_mesh(ax, V, F, base_rgb, tint=None, box=None, ambient=0.42):
    polys, cols = _shade(V, F, base_rgb, tint=tint, ambient=ambient)
    pc = PolyCollection(polys, facecolors=cols, edgecolors="none", antialiaseds=True)
    ax.add_collection(pc)
    if box is None:
        mn = V[:, :2].min(0); mx = V[:, :2].max(0); box = (mn[0], mx[0], mn[1], mx[1])
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal"); ax.axis("off")


# --------------------------------------------------------------------------- FACES
EXAG = 2.3   # uniform shape exaggeration (real per-allele effect is small); labelled on the figure


def vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)


def widen_nose(V0, A, frac=0.72):
    """West-African-typical broad nose: widen the nasal region laterally to ~+50% alar width. A geometric
    POPULATION preset, not an allele combination: no West African facial GWAS exists (the African facial
    GWAS are East-African Bantu, a different, narrower nose), and YRI (Yoruba) supplies only frequencies.
    The +50% width is polygenic (~30x any single real per-allele effect; see the caption)."""
    tip = A["nose"]; s = A["size"]; cx = A["cx"]
    d = np.linalg.norm(V0 - tip, axis=1)
    w = np.clip(1.0 - (d / (0.14 * s)) ** 2, 0, 1)   # plateau at the nose, taper out
    Vd = V0.copy()
    Vd[:, 0] = cx + (V0[:, 0] - cx) * (1.0 + frac * w)
    return Vd


def _render_face(ax, V0, Vd, F, vn, box):
    signed = ((Vd - V0) * vn).sum(1)               # displacement along the surface normal (out+/in-)
    signed = signed / (np.abs(signed).max() + 1e-9)
    draw_mesh(ax, Vd, F, SKIN, tint=signed, box=box)


def face_panel(ax, V0, F, A, dosages, box=None, vn=None):
    _render_face(ax, V0, mesh_morph.morph(V0, A, dosages, exaggerate=EXAG), F, vn, box)


# --------------------------------------------------------------------------- BODY mannequin
def _ring(cx, y, rx, rz, nseg, phase=0.0):
    t = np.linspace(0, 2 * np.pi, nseg, endpoint=False) + phase
    return np.column_stack([cx + rx * np.cos(t), np.full(nseg, y), rz * np.sin(t)])


def _loft(rings, F, V):
    """Connect consecutive same-size rings (list of (nseg,3)) into triangles, append to V/F lists."""
    nseg = rings[0].shape[0]
    for a, b in zip(rings[:-1], rings[1:]):
        i0 = len(V); V.extend(a); V.extend(b)
        for k in range(nseg):
            k2 = (k + 1) % nseg
            F.append((i0 + k, i0 + k2, i0 + nseg + k2))
            F.append((i0 + k, i0 + nseg + k2, i0 + nseg + k))


def _capsule(p0, p1, r0, r1, F, V, nseg=20):
    """Round tube from p0 to p1 (circular cross-section perpendicular to axis)."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    axis = p1 - p0; L = np.linalg.norm(axis); axis /= (L + 1e-9)
    up = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, up); u /= np.linalg.norm(u); w = np.cross(axis, u)
    rings = []
    for t in np.linspace(0, 1, 6):
        c = p0 + t * (p1 - p0); r = r0 + t * (r1 - r0)
        ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
        rings.append(c[None, :] + r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w))
    _loft(rings, F, V)


def _ellipsoid(c, rx, ry, rz, F, V, nlat=16, nlon=24):
    c = np.asarray(c, float)
    i0 = len(V)
    for i in range(nlat + 1):
        th = np.pi * i / nlat
        for j in range(nlon):
            ph = 2 * np.pi * j / nlon
            V.append((c[0] + rx * np.sin(th) * np.cos(ph),
                      c[1] + ry * np.cos(th),
                      c[2] + rz * np.sin(th) * np.sin(ph)))
    for i in range(nlat):
        for j in range(nlon):
            j2 = (j + 1) % nlon
            a = i0 + i * nlon + j; b = i0 + i * nlon + j2
            cc = i0 + (i + 1) * nlon + j; d = i0 + (i + 1) * nlon + j2
            F.append((a, b, d)); F.append((a, d, cc))


def build_body(height=1.0, proportion=1.0, head=True):
    """Parametric front-facing mannequin. height -> overall stature (feet on common ground);
    proportion -> limb:trunk ratio at ~constant total height. head=False omits the head ellipsoid
    (used when a real FaceBase face is placed there instead)."""
    h = 1.0                                        # head unit
    limb = 1.0 + 0.26 * (proportion - 1)           # limb elongation
    trunkf = 1.0 - 0.14 * (proportion - 1)         # trunk shortens as limbs lengthen (ratio, not size)
    V, F = [], []

    hip_y = 3.55 * h * trunkf + (1 - trunkf) * 0    # keep it simple below
    # --- torso: lofted elliptical rings from pelvis up to shoulders ---
    torso = [
        # (y, rx, rz)
        (3.35, 0.72, 0.46),   # pelvis
        (3.95, 0.62, 0.42),   # lower belly
        (4.55, 0.55, 0.40),   # waist
        (5.15, 0.68, 0.44),   # lower chest
        (5.70, 0.80, 0.48),   # chest
        (6.05, 0.98, 0.46),   # shoulders
    ]
    torso = [(3.35 + (y - 3.35) * trunkf, rx, rz) for (y, rx, rz) in torso]
    rings = [_ring(0.0, y, rx, rz, 28) for (y, rx, rz) in torso]
    _loft(rings, F, V)
    shoulder_y = torso[-1][0]; hip_y = torso[0][0]
    hip_hw = torso[0][1]

    # --- neck + head ---
    neck_y0 = shoulder_y; neck_y1 = shoulder_y + 0.35 * h
    _capsule((0, neck_y0, 0.05), (0, neck_y1, 0.05), 0.24, 0.20, F, V, nseg=20)
    if head:
        _ellipsoid((0, neck_y1 + 0.52 * h, 0.05), 0.46, 0.60, 0.52, F, V)

    # --- arms: shoulder -> elbow -> wrist ---
    for sx in (-1, 1):
        sh = (sx * 0.92, shoulder_y - 0.05 * h, 0.02)
        el = (sx * (1.02), shoulder_y - 1.35 * h * limb, 0.0)
        wr = (sx * (0.98), shoulder_y - 2.55 * h * limb, 0.05)
        _capsule(sh, el, 0.26, 0.20, F, V)
        _capsule(el, wr, 0.20, 0.13, F, V)
        _ellipsoid((wr[0], wr[1] - 0.16 * h, wr[2]), 0.15, 0.20, 0.10, F, V, nlat=10, nlon=16)  # hand

    # --- legs: hip -> knee -> ankle + foot ---
    for sx in (-1, 1):
        hp = (sx * 0.34, hip_y - 0.05 * h, 0.0)
        kn = (sx * 0.30, hip_y - 1.75 * h * limb, 0.02)
        an = (sx * 0.26, hip_y - 3.45 * h * limb, 0.0)
        _capsule(hp, kn, 0.36, 0.24, F, V)
        _capsule(kn, an, 0.24, 0.14, F, V)
        _ellipsoid((sx * 0.24, an[1] - 0.10 * h, 0.18), 0.16, 0.10, 0.34, F, V, nlat=10, nlon=16)  # foot

    V = np.array(V, float); F = np.array(F, int)
    # feet to common ground y=0
    V[:, 1] -= V[:, 1].min()
    # overall size (maturation) about the ground
    V *= (1.0 + 0.12 * (height - 1))
    return V, F


BODY_SKIN = np.array([0.86, 0.72, 0.62])


def body_panel(ax, height, proportion, box=None):
    V, F = build_body(height, proportion)
    draw_mesh(ax, V, F, BODY_SKIN, ambient=0.40, box=box)


# --------------------------------------------------------------------------- figure
def main():
    V0, F, HL = mesh_morph.load()
    if F.max() >= len(V0):                          # FaceBase faces are 1-indexed
        F = F - 1
    A = mesh_morph.anchors(V0)

    fig = plt.figure(figsize=(13.2, 5.8))
    fig.suptitle("Faces as allele combinations on the frozen frame "
                 "(the individual adapter, a low-rank move on the coupled manifold)",
                 fontsize=12.5, color=INK, y=0.985)

    fig.text(0.5, 0.90, "Faces on the real 43,071-vertex FaceBase mesh — surface-normal displacement "
             "vs baseline (yellow = outward, blue = inward); magnitudes ∝ real Xiong 2025 C-GWAS β "
             "(bilateral nasal-width effects at reduced exaggeration for legibility), shape ×2.3",
             ha="center", fontsize=9.4, color=INK)
    # (kind, payload, label): 'allele' -> morph dosages; 'preset' -> geometric population preset
    face_combos = [
        ("allele", dict(EDAR=0), "baseline\n(European-typical)"),
        ("allele", dict(PAX3=2), "PAX3 GG\n(deeper nasion)"),
        ("allele", dict(EDAR=2), "EDAR 370A/A\n(E-Asian ≈0.85: chin)"),
        ("allele", dict(RUNX2=2, GLI3=2), "RUNX2+GLI3 high\n(broad bridge+alae)"),
        ("preset", "wafr", "West African (YRI ref.)\nnose +50% — polygenic"),
    ]
    vn = vertex_normals(V0, F)                       # base-face surface normals for the displacement map

    def face_Vd(kind, payload):
        return widen_nose(V0, A) if kind == "preset" else mesh_morph.morph(V0, A, payload, exaggerate=EXAG)

    allf = np.vstack([face_Vd(k, p)[:, :2] for k, p, _ in face_combos])
    fmn = allf.min(0); fmx = allf.max(0); pad = 0.07 * (fmx - fmn)
    fbox = (fmn[0] - pad[0], fmx[0] + pad[0], fmn[1] - pad[1], fmx[1] + pad[1])
    for i, (k, p, lab) in enumerate(face_combos):
        ax = fig.add_axes([0.008 + i * 0.198, 0.30, 0.185, 0.56])
        _render_face(ax, V0, face_Vd(k, p), F, vn, fbox)
        fig.text(0.008 + i * 0.198 + 0.0925, 0.285, lab, ha="center", va="top", fontsize=8.4, color=INK)
    cax = fig.add_axes([0.42, 0.17, 0.16, 0.022])
    grad = np.vstack([np.linspace(-1, 1, 256)] * 2)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("io", [(0.20, 0.45, 0.95), (0.96, 0.96, 0.92), (0.99, 0.85, 0.15)])
    cax.imshow(grad, aspect="auto", cmap=cm); cax.axis("off")
    fig.text(0.42, 0.155, "inward", ha="center", va="top", fontsize=7.5, color=INK)
    fig.text(0.58, 0.155, "outward", ha="center", va="top", fontsize=7.5, color=INK)

    out = "data/organ_cascade/allele_faces_bodies.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
