"""A lightweight topology-aware shape metric (no TDA library needed).

Outer-moment descriptors (elongation/bend) cannot see a loop or a coil, because a filled looped organ is
overall compact. Two cheap features capture the internal topology instead:

  tortuosity  = geodesic backbone length / straight end-to-end distance  -- high for a COIL or LOOP (the path
                through the organ winds), ~1 for a blob or a straight rod.
  hollowness  = distance from the centroid to the nearest cell / organ size -- high when the organ ENCLOSES a
                void (a tube lumen, a cup, the hole of a loop), ~0 for a solid blob.

Both are ~0 for a shapeless blob, so together they reward exactly the shapes the outer moments miss -- IF the
real reconstructed organ actually shows the topology (this file lets us check).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components

def topo3d(P, k=8, cap=600):
    P = np.asarray(P, float)
    if len(P) > cap:
        P = P[np.random.default_rng(0).choice(len(P), cap, replace=False)]   # fresh seeded RNG -> deterministic
    if len(P) < 20:
        return {"tortuosity": 1.0, "hollowness": 0.0}
    c = P.mean(0); size = np.sqrt(((P - c) ** 2).sum(1).mean()) + 1e-9
    hollow = float(np.min(np.linalg.norm(P - c, axis=1)) / size)          # centroid in a void?
    # kNN graph, restricted to the largest connected component
    dist, idx = cKDTree(P).query(P, k=min(k + 1, len(P)))
    n = len(P); rows = np.repeat(np.arange(n), idx.shape[1] - 1)
    G = csr_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n))
    G = G.maximum(G.T)
    ncomp, lab = connected_components(G, directed=False)
    if ncomp > 1:
        keep = lab == np.bincount(lab).argmax()
        if 20 < keep.sum() < n:
            return topo3d(P[keep], k, cap)
    d0 = shortest_path(G, method="D", indices=0)
    u = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    du = shortest_path(G, method="D", indices=u)
    v = int(np.argmax(np.where(np.isfinite(du), du, -1)))
    gd = du[v]; ed = np.linalg.norm(P[u] - P[v]) + 1e-9
    return {"tortuosity": float(gd / ed), "hollowness": hollow}


if __name__ == "__main__":
    from medic.forward_organs_solid import heart_solid, gut_solid, neural_solid, spinal_cord_solid
    from medic.organ_3d_vs_real import blob
    d = np.load("data/mosta/mouse_e125_3d.npz", allow_pickle=True)
    xyz, tis = d["xyz"], d["tissue"]
    FWD = {"Heart": heart_solid, "GI tract": gut_solid, "Brain": neural_solid, "Spinal cord": spinal_cord_solid}
    print(f"{'organ':12s} | {'REAL tort/holl':>16s} | {'BLOB tort/holl':>16s} | {'FORWARD tort/holl':>18s}")
    for t, gen in FWD.items():
        R = xyz[tis == t]
        tr = topo3d(R); size = np.sqrt(((R - R.mean(0)) ** 2).sum(1).mean())
        tb = topo3d(blob(len(R), size * 2)); tf = topo3d(gen())
        print(f"{t:12s} | {tr['tortuosity']:6.2f} {tr['hollowness']:6.2f}   | "
              f"{tb['tortuosity']:6.2f} {tb['hollowness']:6.2f}   | {tf['tortuosity']:7.2f} {tf['hollowness']:7.2f}")
