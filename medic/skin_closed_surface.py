"""skin_closed_surface.py -- PROTOTYPE (render-gated): a TRUE closed skin surface over the flesh cloud,
replacing the per-AP-slice radial envelope whose inability to represent CONCAVITY is the root of the
shoulder capes, the hip skirt shelf, the cone crown, and the blocky foot stubs (the 2026-08-30 visual
audit). Method: voxelise the flesh cloud (body + muscle bellies + fat + foot autopods) into a gaussian
density field, marching-cubes the iso-surface, Laplacian-smooth. Output: a side-by-side render of the
OLD slice shell vs the NEW closed surface on the SAME matured adult cloud, front + side.

Wiring into the movie (next session): the movie stores ONE faces array with per-frame verts, so either
emit per-frame topology or reconstruct at the adult only; this prototype settles whether the surface
is worth that plumbing.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.skin_closed_surface
Out: data/organ_cascade/skin_closed_vs_slice.png
"""
from __future__ import annotations
import numpy as np


def closed_surface(P, grid=140, sigma=1.6, iso_frac=0.30, smooth_iters=8):
    """Marching-cubes closed surface over a point cloud. Returns (verts, faces) in cloud coords."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    P = np.asarray(P, float)
    lo = P.min(0) - 0.04 * np.ptp(P, 0)
    hi = P.max(0) + 0.04 * np.ptp(P, 0)
    span = hi - lo
    res = span / span.max()
    dims = np.maximum((grid * res).astype(int), 24)
    idx = np.clip(((P - lo) / span * (dims - 1)).astype(int), 0, dims - 1)
    vol = np.zeros(dims, np.float32)
    np.add.at(vol, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    vol = gaussian_filter(vol, sigma)
    iso = iso_frac * vol[vol > 0].mean()
    V, F, _, _ = marching_cubes(vol, level=iso)
    V = lo + V / (dims - 1) * span
    # The skin is ONE closed surface: isolated density islands (fat/relief specks)
    # marching-cubes as floating debris blobs -- keep the largest component only.
    parent = np.arange(len(V))
    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b, c in F:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb
        rb, rc = _find(b), _find(c)
        if rb != rc:
            parent[rb] = rc
    roots = np.array([_find(i) for i in range(len(V))])
    ids, cts = np.unique(roots, return_counts=True)
    keep = roots == ids[np.argmax(cts)]
    if not keep.all():
        remap = -np.ones(len(V), int)
        remap[keep] = np.arange(keep.sum())
        V = V[keep]
        F = remap[F]
        F = F[(F >= 0).all(1)]
    # TAUBIN smoothing (cycle 73 — the mid-movie head lag conviction): the plain Laplacian
    # (0.5 blend x12) SHRINKS the mesh, and the shrinkage scales with edge length — on the
    # elongated mid-frames (14k verts over a 3.2-long body) the skin contracted DEEP inside the
    # cloud (outside-cell fraction tracked vert count, 4% at 74k verts -> 24-27% at 14k, clean
    # again at the 46k adult). Taubin's lambda|mu pair (0.5 / -0.53) smooths without shrinking:
    # the same sheen, the surface stays ON the flesh at every frame.
    n = len(V)
    nbr = [[] for _ in range(n)]
    for a, b, c in F:
        nbr[a] += [b, c]; nbr[b] += [a, c]; nbr[c] += [a, b]
    nbr = [np.unique(x) for x in nbr]

    def _lap(Vv, w):
        L = np.array([Vv[k].mean(0) if len(k) else Vv[i] for i, k in enumerate(nbr)])
        return Vv + w * (L - Vv)

    for _ in range(smooth_iters):
        V = _lap(V, 0.5)
        V = _lap(V, -0.53)
    return V, F


def flesh_cloud(Q, F, autopods=None):
    """The full flesh stack the skin should wrap: cloud + per-muscle bellies + subcutaneous fat + feet.
    `autopods=(fv, hv)`: precomputed foot/hand shells from populate_autopods -- REQUIRED when the
    cloud has been autopod-populated (re-deriving anchors from the populated limb walks the hand)."""
    from medic.flesh_surface_head import _model_muscle_bellies, _subcutaneous_fat
    from medic.human_movie import feet_mesh, LIMB
    rng = np.random.default_rng(0)
    parts = [Q]
    b = _model_muscle_bellies(Q, F, rng=rng)
    if len(b):
        parts.append(b)
    # ARM FIELD REINFORCEMENT (2026-08-31, cycle 15): the arm CELLS are already canon (shoulder
    # 0.84 -> fingertip 0.45, length 0.39 of stature) but ~1.9k cells per arm is too sparse for the
    # density field -- the iso surface truncated the arm into a stub partway down (the same
    # density-not-anatomy disease as the gut thread; the feet already solve it by joining the field
    # as explicit points). Clone the distal arm cells with small jitter so the arm column carries
    # body-core density; radius stays the cells' own (jitter << arm radius).
    x = Q[:, 0]
    h = (x - x.min()) / (np.ptp(x) + 1e-9)
    armm = (np.asarray(F) == LIMB) & (h > 0.45) if F is not None else np.zeros(len(Q), bool)
    n_rep = 0
    if armm.sum() > 100:
        A = Q[armm]
        reps = np.vstack([A + rng.normal(size=A.shape) * 0.012 for _ in range(3)])
        parts.append(reps)
        n_rep = len(reps)
    layer = np.vstack(parts)
    # the layer's ARM mask (cycle 17b): cells' arm flag + bellies (not arm) + the arm reps -- so
    # the fat head can shell each arm separately (the arm fork, the leg law's sibling).
    am_layer = np.concatenate([armm, np.zeros(len(layer) - len(Q) - n_rep, bool),
                               np.ones(n_rep, bool)])
    parts.append(_subcutaneous_fat(layer, frac=0.04, n=max(2000, len(layer) // 6), rng=rng,
                                   arm_mask=am_layer))
    # LEG FIELD REINFORCEMENT (cycle 72, Miles's eye x3: cells outside the skin DOWN THE LEGS --
    # the census probe put them AT the column, |ML| med 0.058 vs the line 0.06, depth 0.03-0.06,
    # densest at the feet): the ARM got the x3 clone treatment for exactly this disease (sparse
    # column -> weak field -> iso surface cuts inside the outermost cells) but the leg never did.
    # Same treatment, the distal leg (below the knee band, where the column is sparsest).
    # Appended AFTER layer/fat so the arm mask alignment is untouched.
    legm = (np.asarray(F) == LIMB) & (h < 0.30) if F is not None else np.zeros(len(Q), bool)
    if legm.sum() > 100:
        L = Q[legm]
        parts.append(np.vstack([L + rng.normal(size=L.shape) * 0.012 for _ in range(3)]))
    if autopods is not None:
        fv = np.asarray(autopods[0], float)               # shells from populate_autopods (fixed anchors)
    else:
        fv, _ = feet_mesh(Q, F, frac=1.0)                 # the autopods join the field -> skin wraps the toes
        fv = np.asarray(fv, float)
    if len(fv):
        # densified like the hands: the bare 204-vert shell registered so weakly the surface capped
        # at the ankle (measured: max 0.12 stature from any surface vertex -- the blocky foot stubs).
        parts.append(np.vstack([fv + rng.normal(size=fv.shape) * 0.008 for _ in range(8)]))
    from medic.human_movie import hands_mesh
    if autopods is not None:
        hv = np.asarray(autopods[1], float)
    else:
        hv, _ = hands_mesh(Q, F, frac=1.0)                # the HANDS join the field too (cycle 16)
    if len(hv):
        hv = np.asarray(hv, float)
        # 204 shell verts are too sparse to register in the density field (the surface missed the
        # hand by up to 0.03 stature) -- densify like the arm column so the skin wraps the hand.
        # reps 8 -> 12 (polish): at grid 210 the hand lobe needs core density or it beads.
        reps = np.vstack([hv + rng.normal(size=hv.shape) * 0.007 for _ in range(12)])  # fused mitt lobe
        parts.append(reps)
        # THE WRIST BRIDGE (cycle 16c, Miles: hands not connected to the forearms): the lateral
        # hang-out that made the hands visible opened a diagonal gap between the forearm's tip and
        # the hand base that the field cannot span. Bridge each side with a short dense column of
        # points -- the wrist -- so arm, wrist and hand skin as one body.
        x = Q[:, 0]
        H = np.ptp(x) + 1e-9
        apf = (x - x.min()) / H
        armm2 = (np.asarray(F) == LIMB) & (apf >= 0.45)
        mid = float(np.median(Q[:, 2]))
        for sgn in (-1.0, 1.0):
            am = armm2 & (np.sign(Q[:, 2] - mid) == sgn)
            hm2 = hv[np.sign(hv[:, 2] - mid) == sgn]
            if am.sum() < 6 or len(hm2) < 6:
                continue
            armc = Q[am]
            wrist = armc[armc[:, 0] < np.percentile(armc[:, 0], 12)].mean(0)   # forearm true tip
            handtop = hm2[hm2[:, 0] > np.percentile(hm2[:, 0], 80)].mean(0)    # hand base (palm heel)
            # 240 -> 600 pts, jitter 0.010 -> 0.012 (polish): at grid 210 the thin bridge thread
            # pinched between arm tip and palm -- the bead-chain look. A real wrist's worth of mass.
            t = rng.random(600)[:, None]
            bridge = wrist + t * (handtop - wrist) + rng.normal(size=(600, 3)) * 0.012
            parts.append(bridge)
    return np.vstack(parts)


def run():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from medic.adult_persistence_audit import build_base
    import medic.human_movie as HM
    from medic.flesh_surface_head import flesh_skin

    print("building + maturing ...")
    B0, BF = build_base(30000)
    Q = HM.mature_cloud(B0.copy(), BF, 1.0, HM.MATURE_SEARCHED)
    Q = HM.grow_limbs(Q, BF == HM.LIMB, HM._limb_grow_model(1.0, HM.MATURE_SEARCHED["limb_ext"]),
                      HM._limb_grow_model(1.0, HM.MATURE_SEARCHED.get("leg_ext", HM.MATURE_SEARCHED["limb_ext"])),
                      pose=1.0, fate=BF)
    Q = HM.standing_register(Q, BF, 1.0)
    print("old slice shell ...")
    sv_old, sf_old = flesh_skin(Q, BF)
    print("new closed surface ...")
    cloud = flesh_cloud(Q, BF)
    sv_new, sf_new = closed_surface(cloud)
    print(f"  slice shell {len(sv_old)} verts / closed surface {len(sv_new)} verts, {len(sf_new)} faces")

    np.savez_compressed("data/organ_cascade/_skin_surf_cache.npz",
                        sv_old=sv_old, sf_old=sf_old, sv_new=sv_new, sf_new=sf_new, cloud=cloud)
    _plot(sv_old, sf_old, sv_new, sf_new)


def tune(grid=170, sigma=1.1, iso_frac=0.45):
    """Fast re-surface of the CACHED flesh cloud with new params (crotch separation / feet definition)."""
    d = np.load("data/organ_cascade/_skin_surf_cache.npz")
    sv, sf = closed_surface(d["cloud"], grid=grid, sigma=sigma, iso_frac=iso_frac)
    print(f"tuned: grid={grid} sigma={sigma} iso={iso_frac} -> {len(sv)} verts {len(sf)} faces")
    _plot(d["sv_old"], d["sf_old"], sv, sf)


def _plot(sv_old, sf_old, sv_new, sf_new):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(18, 11), facecolor="#0d1017")
    views = [("slice shell · front", sv_old, sf_old, -90), ("closed surface · front", sv_new, sf_new, -90),
             ("slice shell · side", sv_old, sf_old, 0), ("closed surface · side", sv_new, sf_new, 0)]
    for i, (title, V, Fc, azim) in enumerate(views):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.set_facecolor("#0d1017")
        tri = np.asarray(V)[np.asarray(Fc)][:, :, [2, 1, 0]]     # plot frame: x=ML, y=DV, z=AP (upright)
        ax.add_collection3d(Poly3DCollection(tri, alpha=1.0, facecolor="#c9a189", edgecolor="none"))
        m = np.asarray(V).mean(0)
        r = 0.55 * np.ptp(np.asarray(V)[:, 0])
        ax.set_xlim(m[2] - r, m[2] + r); ax.set_ylim(m[1] - r, m[1] + r); ax.set_zlim(m[0] - r, m[0] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=6, azim=azim)
        ax.set_axis_off(); ax.set_title(title, color="#e2e8f0", fontsize=12)
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0)
    fig.savefig("data/organ_cascade/skin_closed_vs_slice.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/skin_closed_vs_slice.png")


def replot():
    d = np.load("data/organ_cascade/_skin_surf_cache.npz")
    _plot(d["sv_old"], d["sf_old"], d["sv_new"], d["sf_new"])


if __name__ == "__main__":
    run()
