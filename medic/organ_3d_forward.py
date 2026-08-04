"""The four organs, grown in 3D by the same forward engine.

The 2D cross-sections proved the mechanism; here each is lifted to its true 3D form, because a section
through a looped or coiled organ hides the shape a metric needs to see:

  optic cup   -- the 2D invagination profile REVOLVED about the optic axis -> a 3D two-layer bowl.
  neural tube -- the 2D rolled ring EXTRUDED along the antero-posterior axis -> a 3D cylinder with a lumen.
  gut tube    -- an over-long rod anchored at both ends buckles in 3D into a HELICAL coil (the midgut loop is
                 a 3D coil, not a planar one) -> a swept tube.
  heart       -- a handed 3D curvature loops the straight tube out of plane into the D-loop; flip the sign
                 (Pitx2) and it mirrors -> the L-loop (situs inversus), which is genuinely a 3D handedness.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.organ_3d_forward
Out:  data/organ_cascade/organ_3d_forward.png  + organ_3d_points.npz (for the viewer)
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import medic.optic_cup_forward as cup
import medic.neural_tube_forward as neural

RNG = np.random.default_rng(0)


# ---------- sheet organs: revolve / extrude the 2D shape ----------
def optic_cup_3d(nrev=64):
    A0, B0, ds, h = cup.build_sheet(); c = cup.constriction_profile(len(A0) - 1)
    A1, B1 = cup.relax(A0.copy(), B0.copy(), ds, h, c)          # 2D cup cross-section (apical + basal)
    n = len(A1); lo, hi = int(0.20 * n), int(0.80 * n)          # keep the invaginated centre, drop flat arms
    A1, B1 = A1[lo:hi], B1[lo:hi]
    prof = np.vstack([A1, B1]); prof = prof - prof.mean(0)
    r = prof[:, 0] - prof[:, 0].min()                          # radius from the optic axis
    z = prof[:, 1]
    layer = np.r_[np.zeros(len(A1)), np.ones(len(B1))]         # 0 = retina (inner), 1 = RPE (outer)
    th = np.linspace(0, 2 * np.pi, nrev)
    P = np.concatenate([np.c_[r * np.cos(t), r * np.sin(t), z] for t in th])
    lay = np.tile(layer, nrev)
    return P, lay


def neural_tube_3d(nap=48, L=6.0):
    _, _, An, Bn, *_ = neural.run(0.17)                        # 2D closed ring (apical + basal)
    ring = np.vstack([An, Bn]); ring = ring - ring.mean(0)
    ring = ring / (np.abs(ring).max() + 1e-9)                  # cross-section in (u, v)
    lumen = np.r_[np.zeros(len(An)), np.ones(len(Bn))]         # 0 = apical (lumen side), 1 = basal
    ap = np.linspace(0, L, nap)
    P = np.concatenate([np.c_[np.full(len(ring), x), ring[:, 0], ring[:, 1]] for x in ap])
    return P, np.tile(lumen, nap)


# ---------- tube organs: 3D rod buckling + tube sweep ----------
def rod_3d(N, excess, span, handed=1.0, turns=3.0, iters=2500, bend=0.28, amp=0.12):
    """A rod anchored at both ends, its rest length grown beyond the span, buckles in 3D. A helical seed sets
    the mode and handedness; the excess sets how tightly it coils."""
    s = np.linspace(0, 1, N + 1)
    P = np.c_[s * span, amp * np.sin(2 * np.pi * turns * s), handed * amp * np.cos(2 * np.pi * turns * s)]
    P[:, 2] -= P[0, 2]
    rest = (span / N) * (1.0 + excess)
    left, right = P[0].copy(), P[-1].copy()
    for _ in range(iters):
        for k in range(N):
            d = P[k + 1] - P[k]; Ln = np.linalg.norm(d) + 1e-9; corr = (Ln - rest) / Ln * d
            P[k] += 0.5 * corr; P[k + 1] -= 0.5 * corr
        P[0], P[-1] = left, right
        P[1:-1] += bend * (0.5 * (P[:-2] + P[2:]) - P[1:-1])
        P[0], P[-1] = left, right
    return P


def sweep_tube(C, radius=0.16, nseg=10):
    """Sweep a circular cross-section along a 3D centre-line C (parallel-transport frame)."""
    T = np.gradient(C, axis=0); T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    up = np.array([0, 0, 1.0]); N0 = np.cross(up, T[0]); N0 /= np.linalg.norm(N0) + 1e-9
    normals = [N0]
    for i in range(1, len(C)):                                # transport the normal along the curve
        n = normals[-1] - T[i] * (normals[-1] @ T[i]); n /= np.linalg.norm(n) + 1e-9; normals.append(n)
    normals = np.array(normals); B = np.cross(T, normals)
    ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    pts = [C + radius * (np.cos(a) * normals + np.sin(a) * B) for a in ang]
    P = np.concatenate(pts)
    frac = np.tile(np.linspace(0, 1, len(C)), nseg)           # inflow->outflow / foregut->hindgut
    return P, frac


def gut_3d():
    C = rod_3d(N=160, excess=1.15, span=3.0, turns=4.0, bend=0.3)
    return sweep_tube(C, radius=0.13)


def heart_3d(handed=1.0):
    C = rod_3d(N=120, excess=0.55, span=2.2, handed=handed, turns=1.15, bend=0.5, amp=0.25)
    return sweep_tube(C, radius=0.22)


def _ax(fig, i, title, col):
    a = fig.add_subplot(1, 5, i, projection="3d"); a.set_facecolor("#0d1017")
    a.set_title(title, color=col, fontsize=10, pad=0)
    a.set_axis_off(); a.set_box_aspect((1, 1, 1)); return a


def main():
    cupP, cupL = optic_cup_3d()
    ntP, ntL = neural_tube_3d()
    gutP, gutF = gut_3d()
    hdP, hdF = heart_3d(+1.0)
    hlP, _ = heart_3d(-1.0)
    np.savez("data/organ_cascade/organ_3d_points.npz",
             cup=cupP, cup_layer=cupL, neural=ntP, neural_layer=ntL,
             gut=gutP, gut_frac=gutF, heart=hdP, heart_frac=hdF, heart_inv=hlP)

    fig = plt.figure(figsize=(21, 4.6), facecolor="#0d1017")
    a = _ax(fig, 1, "optic cup (revolved)\nretina + RPE", "#f59e0b")
    a.scatter(cupP[:, 0], cupP[:, 1], cupP[:, 2], s=2, c=cupL, cmap="autumn", linewidths=0)
    a.view_init(elev=18, azim=-60)
    a = _ax(fig, 2, "neural tube (extruded)\ncylinder + lumen", "#38bdf8")
    a.scatter(ntP[:, 0], ntP[:, 1], ntP[:, 2], s=1.5, c=ntL, cmap="cool", linewidths=0)
    a.view_init(elev=22, azim=-70)
    a = _ax(fig, 3, "gut tube (3D buckle)\nhelical midgut coil", "#22c55e")
    a.scatter(gutP[:, 0], gutP[:, 1], gutP[:, 2], s=2, c=gutF, cmap="viridis", linewidths=0)
    a.view_init(elev=24, azim=-60)
    a = _ax(fig, 4, "heart (handed 3D loop)\nD-loop", "#ef4444")
    a.scatter(hdP[:, 0], hdP[:, 1], hdP[:, 2], s=2, c=hdF, cmap="plasma", linewidths=0)
    a.view_init(elev=20, azim=-60)
    a = _ax(fig, 5, "heart, Pitx2 flipped\nL-loop (situs inversus)", "#f472b6")
    a.scatter(hlP[:, 0], hlP[:, 1], hlP[:, 2], s=2, c=hdF, cmap="plasma", linewidths=0)
    a.view_init(elev=20, azim=-60)
    fig.suptitle("The four organs grown in 3D by the forward engine: revolve/extrude the sheet organs, "
                 "3D buckling for the tubes -- the shapes a 2D section hides", color="#e2e8f0", fontsize=13)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/organ_3d_forward.png", dpi=130, facecolor="#0d1017")
    print(f"cup {len(cupP)}  neural {len(ntP)}  gut {len(gutP)}  heart {len(hdP)} points")
    print("saved data/organ_cascade/organ_3d_forward.png + organ_3d_points.npz")


if __name__ == "__main__":
    main()
