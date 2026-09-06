"""outside_skin.py -- THE OUTSIDE-SKIN CENSUS, a standing loop instrument (cycle 70; Miles:
'can we sort out the skin issues as part of the normal loop?').

Scores the SHIPPED frames (the viewer's own object): which displayed cells sit OUTSIDE the
closed skin surface, by fate, height band, protrusion depth and spatial cluster. Runs with the
curve + visual board in every LOOK; the per-frame outside fractions rank the next skin target
exactly as the curve ranks organ targets. Appends a summary to data/organ_cascade/outside_skin.json.

CAVEAT (declared): inside/outside = sign of (p - nearest skin vert) . (kNN-averaged vertex
normal). At CONCAVITIES (armpit, neck-shoulder, crotch) the test misreads -- treat depths
< ~0.04 stature near junctions as noise; clusters with depth >= 0.05 are real. _wall_probe.py
is the cross-check against the measured canonical envelope.

Fault ledger: mid-movie HEAD LAG (skin trails the head through the skin_op ramp, 21-27 pct
outside there), foot-level Limb Bud strays (below-floor landing selections), left-forearm
cluster. Adult after cycle-70 containment: chest band beyond-wall 320 -> 12.

Run:  venv_win_new/Scripts/python.exe -m medic.outside_skin
"""
import json
import numpy as np
from scipy.spatial import cKDTree

fr = json.load(open("data/movie/human_movie_frames.json"))
frames = fr["frames"]
FACES = np.asarray(fr["skin_faces"], int).reshape(-1, 3)

# the adult skin frames the viewer shows: skin mesh + cloud, NOT the exploded reveal frames
# (ptype marks the reveal; 'panel' is just the gene-annotation strip)
cand = [f for f in frames if isinstance(f, dict) and f.get("skin") and "xyz" in f
        and "fate" in f and f.get("phase") == "mesh"]
print(f"{len(cand)} non-panel frames carry skin+cloud; scoring first + middle + last")
for fi in (0, len(cand) // 2, len(cand) - 1):
    f = cand[fi]
    V = np.asarray(f["skin"], float).reshape(-1, 3)
    Fc = FACES[(FACES < len(V)).all(1)]
    P = np.asarray(f["xyz"], float).reshape(-1, 3)
    fate = np.asarray(f["fate"])
    # vertex normals from faces
    N = np.zeros_like(V)
    e1 = V[Fc[:, 1]] - V[Fc[:, 0]]
    e2 = V[Fc[:, 2]] - V[Fc[:, 0]]
    fn = np.cross(e1, e2)
    for k in range(3):
        np.add.at(N, Fc[:, k], fn)
    # orient outward: normal at each vert should point away from the local centroid
    ctr = V.mean(0)
    flip = ((V - ctr) * N).sum(1) < 0
    # closed surface with consistent winding -> global majority decides
    if flip.mean() > 0.5:
        N = -N
    N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-12)
    tree = cKDTree(V)
    dd, jj = tree.query(P, k=6)
    nbar = N[jj].mean(1)
    nbar /= (np.linalg.norm(nbar, axis=1, keepdims=True) + 1e-12)
    disp = P - V[jj[:, 0]]
    signed = (disp * nbar).sum(1)
    out = signed > 0.02 * float(np.ptp(V[:, 0]))           # DEFINITE: beyond surface by >2% stature
    #                                                        (the 0.5% band is concavity test-noise)
    x = P[:, 0]
    h = (x - V[:, 0].min()) / (np.ptp(V[:, 0]) + 1e-9)
    st = str(f.get("stage", fi))[:30]
    print(f"\nframe [{st}]: {len(P):,d} cells shown, OUTSIDE {int(out.sum()):,d} "
          f"({100.0*out.mean():.1f}%), max protrusion {signed[out].max() if out.any() else 0:.4f}")
    if out.any():
        from medic.unified_embryo import FATES
        names, counts = np.unique(fate[out], return_counts=True)
        top = sorted(zip(counts, names), reverse=True)[:12]
        for c, n in top:
            hh = h[out & (fate == n)]
            ss = signed[out & (fate == n)]
            nm = FATES[int(n)] if str(n).lstrip("-").isdigit() and 0 <= int(n) < len(FATES) else str(n)
            print(f"    {nm:24s} {c:>5d}  h [{hh.min():.2f},{hh.max():.2f}] med {np.median(hh):.2f}"
                  f"  depth med {np.median(ss):.3f} max {ss.max():.3f}")
    if out.any() and fi == len(cand) - 1:
        # localize the adult stragglers: connected clusters by proximity (2% stature linkage)
        from scipy.cluster.hierarchy import fcluster, linkage
        Po = P[out]
        if len(Po) > 2:
            lab = fcluster(linkage(Po, method="single"), t=0.02 * float(np.ptp(V[:, 0])),
                           criterion="distance")
            print("  ADULT CLUSTERS (pos = h, ML, DV medians):")
            for cl in np.unique(lab):
                m = lab == cl
                if m.sum() < 15:
                    continue
                hq = h[out][m]
                print(f"    cluster n={int(m.sum()):>4d}  h {np.median(hq):.2f}  "
                      f"ML {np.median(Po[m, 2]):+.3f}  DV {np.median(Po[m, 1]):+.3f}  "
                      f"fates {[str(FATES[int(v)]) for v in np.unique(fate[out][m])[:5]]}")

