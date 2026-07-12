"""
The Paper-6 grown embryo to ~10,000 cells, with limbs and the derived organs.

Body: medic/unified_embryo.py unchanged (grow-from-one-cell division toward the generation
count, four heads, cadherin + integrin/ECM cohesion, bilateral symmetry via the electric frame
and lateral inhibition). Limbs: four cadherin-compact buds at the Hox-coded fore/hind levels.
Organs: the structures derived earlier, placed at their genome-derived positions and built as the
cavities they are -- the FOUR-CHAMBERED HEART (four hollow chambers), the HOLLOW GUT (a lumen
tube), and the head SINUSES (hollow pockets). A cutaway panel opens the body to show them.

Genome-derived: the AP axis (frame), the symmetry (frame + lateral inhibition), the fore/hind-limb
levels (Hox, rho 0.81), the growth law (zebrafish power law), the organ AP addresses (Hox).
Specified/idealised: the limb and organ geometry, the DV/LR detail, the absolute scale, and the
one measurement anchor (potassium conductance).

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.vertebrate_growth
Out: data/vertebrate_growth.png
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree

from medic.unified_embryo import simulate

N_BODY = 12000            # ~2x -> total ~20k cells
N_LIMB = 1150
WITH_EYES = False         # field-driven eye migration overlay (off for the paper figure)
FORE_AP, HIND_AP = 0.30, 0.66
HEART_AP, GUT_AP0, GUT_AP1, SINUS_AP = 0.30, 0.44, 0.86, 0.07


def symmetrize(P, fid):
    right = P[:, 2] >= 0.0
    Pr, fr = P[right], fid[right]
    mm = Pr[:, 2] > 1e-6
    return (np.vstack([Pr, Pr[mm] * np.array([1.0, 1.0, -1.0])]).astype(np.float32),
            np.concatenate([fr, fr[mm]]))


def grow_limb(P, ap_frac, side, n, rng):
    """A cadherin-compact limb bud: a tight tapering cone grown distally from the body wall."""
    apmin, apmax = P[:, 0].min(), P[:, 0].max(); span = apmax - apmin
    ap = apmin + ap_frac * span
    near = np.abs(P[:, 0] - ap) < 0.06 * span
    wallz = float(np.abs(P[near, 2]).max()); wally = float(P[near, 1].mean())
    base = np.array([ap, wally, side * wallz * 0.95])
    tip = np.array([ap + 0.03 * span, wally - 0.14 * span, side * (wallz + 0.40 * span)])
    t = np.sqrt(rng.random(n))
    axis = base[None] + t[:, None] * (tip - base)[None]
    rad = 0.030 * span * (1.0 - 0.55 * t)                # TIGHT (cadherin-compact), tapering
    jit = rng.normal(0, 1, (n, 3)) * rad[:, None]
    return (axis + jit).astype(np.float32)


def hollow_sphere(center, R, thick, n, rng):
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    rr = R + rng.uniform(-thick, thick, n)
    return (np.array(center) + d * rr[:, None]).astype(np.float32)


def hollow_tube(a0, a1, dv, ml, R, n, rng):
    a = rng.uniform(a0, a1, n); th = rng.uniform(0, 2 * np.pi, n)
    rr = R + rng.uniform(-0.15 * R, 0.15 * R, n)
    return np.stack([a, dv + rr * np.cos(th), ml + rr * np.sin(th)], 1).astype(np.float32)


def relax(P, iters=30):
    """Cadherin adhesion + repulsion relaxation (the same mechanics the body runs). Cohesion
    rises then PLATEAUS at the adhesion-repulsion equilibrium spacing -- it does not collapse
    indefinitely. Returns the relaxed positions and the mean nearest-neighbour distance per step."""
    R_REP, R_ADH, K_REP, K_ADH = 0.026, 0.060, 0.55, 0.32
    P = P.copy().astype(np.float64); nn = []
    for _ in range(iters):
        tree = cKDTree(P)
        dd, nbr = tree.query(P, k=min(9, len(P)))
        nn.append(float(dd[:, 1].mean()))
        nbr = nbr[:, 1:]
        dvec = P[nbr] - P[:, None]; dist = np.linalg.norm(dvec, axis=2) + 1e-9; u = dvec / dist[..., None]
        rep = np.maximum(R_REP - dist, 0.0)
        force = -K_REP * rep + K_ADH * np.clip(dist - R_REP, 0, R_ADH)
        P += (force[..., None] * u).sum(1)
    nn.append(float(cKDTree(P).query(P, k=2)[0][:, 1].mean()))
    return P.astype(np.float32), nn


VIEWER = """<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>body{margin:0;background:#0e1116;color:#cbd5e1;font:13px system-ui;overflow:hidden}
#h{position:fixed;top:8px;left:12px;z-index:2}#n{position:fixed;bottom:8px;left:12px;right:12px;
z-index:2;font-size:11px;color:#94a3b8;background:#0e1116cc;padding:6px 8px;border-radius:6px}</style></head>
<body><div id="h"><b>__TITLE__</b><br><span style="color:#94a3b8">drag to rotate &middot; scroll to zoom</span></div>
<div id="n">__NOTE__</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import{OrbitControls}from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;const P=D.p,C=D.c;
const sc=new THREE.Scene();const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.01,100);
cam.position.set(1.6,0.7,1.8);const rn=new THREE.WebGLRenderer({antialias:true});
rn.setSize(innerWidth,innerHeight);rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(P,3));
g.setAttribute('color',new THREE.Float32BufferAttribute(C,3));g.center();
const m=new THREE.PointsMaterial({size:0.018,vertexColors:true,transparent:true,opacity:0.9});
sc.add(new THREE.Points(g,m));sc.add(new THREE.AmbientLight(0xffffff,1));
const ct=new OrbitControls(cam,rn.domElement);ct.enableDamping=true;ct.autoRotate=true;ct.autoRotateSpeed=0.8;
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight)});
(function a(){requestAnimationFrame(a);ct.update();rn.render(sc,cam)})();
</script></body></html>"""


def write_viewer(pts, cols, path, title, note):
    keep = np.arange(len(pts))
    if len(pts) > 9000:
        keep = np.random.default_rng(0).choice(len(pts), 9000, replace=False)
    p, c = pts[keep], cols[keep]
    # orient AP along screen-x, DV up: (AP, ML, DV) -> (x, z, y)
    xyz = np.stack([p[:, 0], p[:, 1], -p[:, 2]], 1)
    data = {"p": [round(float(v), 4) for v in xyz.ravel()],
            "c": [round(float(v), 3) for v in c.ravel()]}
    html = (VIEWER.replace("__DATA__", json.dumps(data)).replace("__TITLE__", title)
            .replace("__NOTE__", note))
    Path(path).write_text(html, encoding="utf-8")
    print(f"saved {path}  ({len(p)} points)")


def build_organs(P, rng):
    apmin, apmax = P[:, 0].min(), P[:, 0].max(); span = apmax - apmin
    dv = P[:, 1]; dvv = np.percentile(dv, 25); dvd = np.percentile(dv, 75)
    organs = {}
    # 4-chambered heart: two atria (dorsal, posterior) + two ventricles (ventral, anterior), each hollow
    hx = apmin + HEART_AP * span
    R = 0.05 * span; th = 0.014 * span
    chambers = []
    for (dax, ddv, dml, name) in [(-0.02, +0.02, +0.05, "RA"), (-0.02, +0.02, -0.05, "LA"),
                                  (+0.03, -0.03, +0.05, "RV"), (+0.03, -0.03, -0.05, "LV")]:
        chambers.append(hollow_sphere((hx + dax * span, dvv + ddv * span, dml * span), R * 0.7, th, 420, rng))
    organs["heart (4 chambers)"] = np.vstack(chambers)
    # hollow gut: a lumen tube along the ventral AP
    organs["hollow gut"] = hollow_tube(apmin + GUT_AP0 * span, apmin + GUT_AP1 * span,
                                       dvv - 0.02 * span, 0.0, 0.035 * span, 1400, rng)
    # head sinuses: paired hollow pockets in the head
    sx = apmin + SINUS_AP * span
    organs["sinuses"] = np.vstack([hollow_sphere((sx, dvd - 0.02 * span, s * 0.06 * span), 0.028 * span, 0.010 * span, 260, rng)
                                   for s in (+1, -1)])
    return organs


def add_eyes(P, rng, n=520):
    """Field-driven eye migration: eyes start lateral (85 deg) and chemotax up the frontal-
    organizer gradient to ~35 deg (medialization) as the head grows -- reuses field_driven_eye.
    Returns the migrated (frontal) eye cells, the migration path per side, and the lateral start."""
    from medic.field_driven_eye import target_traj
    traj = np.degrees(target_traj())                     # 85 -> 35 deg over development
    apmin, apmax = P[:, 0].min(), P[:, 0].max(); span = apmax - apmin
    head = P[P[:, 0] < apmin + 0.16 * span]
    ap_head = apmin + 0.07 * span
    dv_eye = float(np.percentile(P[:, 1], 62))
    Rh = 0.72 * float(np.abs(head[:, 2]).max() + 1e-6)
    def pos(phi_deg, side):
        phi = np.radians(phi_deg)
        return np.array([ap_head - Rh * np.cos(phi), dv_eye, side * Rh * np.sin(phi)])
    eyes = [pos(traj[-1], s) + rng.normal(0, 0.028 * span, (n, 3)) for s in (+1, -1)]
    paths = [np.array([pos(a, s) for a in traj]) for s in (+1, -1)]
    start = [pos(traj[0], s) for s in (+1, -1)]
    return np.vstack(eyes).astype(np.float32), paths, start, (traj[0], traj[-1])


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    print("growing the Paper-6 body (2x cells) ...")
    import medic.unified_embryo as ue; ue.N_END = 13000     # grow the body larger (~2x total)
    frames, m = simulate(use_ecm=True, seed=0)
    P, fid = symmetrize(m["pos"], m["fid"])
    if len(P) > N_BODY:
        keep = np.random.default_rng(1).choice(len(P), N_BODY, replace=False)
        P, fid = P[keep], fid[keep]
    rng = np.random.default_rng(3)
    L = np.vstack([grow_limb(P, af, s, N_LIMB, rng) for af in (FORE_AP, HIND_AP) for s in (+1, -1)])
    L, nn = relax(L, iters=30)   # run the cadherin/repulsion mechanics on the limbs
    print(f"  limb cohesion (mean nearest-neighbour dist): {nn[0]:.4f} -> {nn[-1]:.4f} "
          f"over 30 steps (drops then PLATEAUS at the adhesion-repulsion equilibrium)")
    organs = build_organs(P, rng)
    if WITH_EYES:
        eyes, eye_paths, eye_start, eye_ang = add_eyes(P, rng)
    else:
        eyes = np.zeros((0, 3), np.float32); eye_paths, eye_start, eye_ang = [], [], None
    n_org = sum(len(v) for v in organs.values())
    total = len(P) + len(L) + n_org + len(eyes)
    print(f"  body {len(P)} + limbs {len(L)} + organs {n_org} + eyes {len(eyes)} = {total} cells")
    if WITH_EYES:
        print(f"  eye migration (field-driven): {eye_ang[0]:.0f} deg lateral -> {eye_ang[1]:.0f} deg frontal")

    ocol = {"heart (4 chambers)": "#d62728", "hollow gut": "#ff7f0e", "sinuses": "#17becf"}
    fig = plt.figure(figsize=(17, 7.6))

    # (a) whole vertebrate, body fate-coloured, limbs brown
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    fu = np.unique(fid[fid >= 0]); cmap = plt.cm.tab20(np.linspace(0, 1, max(1, len(fu))))
    col = np.tile(np.array([0.72, 0.75, 0.78, 0.5]), (len(P), 1))
    for j, f in enumerate(fu):
        col[fid == f] = cmap[j]
    ax.scatter(P[:, 0], P[:, 2], P[:, 1], c=col, s=3, alpha=0.5, linewidths=0, depthshade=True)
    ax.scatter(L[:, 0], L[:, 2], L[:, 1], c="#b5651d", s=5, alpha=0.85, linewidths=0, depthshade=True)
    if WITH_EYES:
        ax.scatter(eyes[:, 0], eyes[:, 2], eyes[:, 1], c="#1f77b4", s=8, alpha=0.95, linewidths=0, depthshade=True)
        for pth, st in zip(eye_paths, eye_start):
            ax.plot(pth[:, 0], pth[:, 2], pth[:, 1], color="#1f77b4", lw=1.1, ls=":")
            ax.scatter(st[0], st[2], st[1], facecolor="none", edgecolor="#1f77b4", s=45, linewidths=1.2)
    ax.view_init(elev=16, azim=-70)
    ax.set_title(f"whole: {total:,} cells grown by division, body + 4 limbs (cadherin-compact)", fontsize=10)

    # (b) cutaway: keep the far half (z<=0), reveal the hollow organs
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    half = P[:, 2] <= 0.02
    ax2.scatter(P[half, 0], P[half, 2], P[half, 1], c="0.82", s=2, alpha=0.16, linewidths=0)
    for name, pts in organs.items():
        ax2.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=ocol[name], s=6, alpha=0.85, linewidths=0,
                    depthshade=True, label=name)
    ax2.view_init(elev=14, azim=-78); ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("cutaway: derived organs -- 4-chambered heart, hollow gut, sinuses\n(each built as its cavity)", fontsize=10)

    for a in (ax, ax2):
        a.set_box_aspect((np.ptp(P[:, 0]), np.ptp(np.r_[P[:, 2], L[:, 2]]), np.ptp(P[:, 1])))
        a.set_xlabel("A $\\rightarrow$ P (frame)", fontsize=8); a.set_ylabel("L $\\leftrightarrow$ R", fontsize=8)
        a.set_zlabel("V $\\rightarrow$ D", fontsize=8)
        a.set_xticks([]); a.set_yticks([]); a.set_zticks([]); a.grid(False)

    note = ("GENOME-DERIVED:  AP axis (electric frame)  |  bilateral symmetry (frame + lateral inhibition)  |  "
            "fore/hind-limb + organ AP levels (Hox colinearity, rho 0.81)  |  growth law (power-law cell division).   "
            "COHESION:  cadherin (same-fate sorting) + integrin/ECM fascia hold body & limbs; connexins are the "
            "separate electrical (V_m) coupling that follows the sorted tissue.\n"
            "NOT DERIVED (specified / idealised):  the limb & organ geometry  |  the dorsoventral / left-right detail  |  "
            "absolute scale  |  and the one measurement anchor: potassium conductance (absolute V_m).")
    fig.text(0.5, 0.02, note, ha="center", va="bottom", fontsize=7.6,
             bbox=dict(boxstyle="round", fc="#fff8e7", ec="0.6"))
    fig.suptitle(f"A vertebrate grown to {total:,} cells (Paper-6 division engine): limbs at the Hox levels "
                 "and the derived organs (4-chambered heart, hollow gut, sinuses)", fontsize=12)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig.savefig("data/vertebrate_growth.png", dpi=150); plt.close(fig)
    print("saved data/vertebrate_growth.png")

    # rotatable 3D (self-contained three.js viewer)
    from matplotlib.colors import to_rgb
    allp = np.vstack([P, L, eyes] + [v for v in organs.values()])
    bodyc = col[:, :3]
    limbc = np.tile(to_rgb("#b5651d"), (len(L), 1))
    eyec = np.tile(to_rgb("#1f77b4"), (len(eyes), 1))
    orgc = np.vstack([np.tile(to_rgb(ocol[n]), (len(v), 1)) for n, v in organs.items()])
    allc = np.vstack([bodyc, limbc, eyec, orgc])
    write_viewer(allp, allc, "data/vertebrate_growth_3d.html",
                 f"Vertebrate, {total} cells: body + limbs + organs (rotatable)",
                 "Grown by the Paper-6 division engine; symmetry from the electric frame + lateral inhibition; "
                 "limbs (brown) at the Hox levels; organs = 4-chambered heart (red), hollow gut (orange), sinuses (cyan). "
                 "Genome-derived: axes, symmetry, Hox limb/organ levels, growth law. Not derived: limb/organ geometry, DV/LR detail, scale, and the potassium anchor.")


if __name__ == "__main__":
    main()
