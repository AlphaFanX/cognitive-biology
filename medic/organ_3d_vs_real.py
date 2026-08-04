"""3D before/after, on the REAL E9.5 3D MOSTA reconstruction (mouse_3d.npz).

For the organs that are already SHAPED at E9.5 (the heart is looping, the neural tube is closed), compare
three 3D shapes on scale/rotation-invariant descriptors: the REAL organ cells (from the stacked E9.5
sections), the CLOUD the current model places (an isotropic blob at the right spot), and the FORWARD 3D
organ shape. If the forward shape sits closer to the real than the blob, the 3D pipeline works and building
organs by the forward engine gets the computed embryo closer to real -- now in a frame that can see shape.

Note (E9.5): organ forms are gentle; the gut has not coiled yet, so it is excluded. The dramatic loop/coil
test needs a dense late-stage 3D reconstruction (E12.5 has 5 serial sections, E16.5 has 13).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.organ_3d_vs_real
Out:  data/organ_cascade/organ_3d_vs_real.{json,png}
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.organ_3d_forward import heart_3d, neural_tube_3d

RNG = np.random.default_rng(0)


def desc3d(P):
    """Scale/rotation-invariant 3D shape descriptors: elongation and flatness (from the PCA singular values)
    and the bend of the principal axis. A blob -> ~(1, 1, 0); a shaped tube/loop -> elongated and/or curved."""
    P = np.asarray(P, float); P = P - P.mean(0)
    w = np.linalg.svd(P, full_matrices=False)[1] + 1e-9
    u = np.linalg.svd(P, full_matrices=False)[2][0]
    s = P @ u                                              # along the main axis
    perp = P - np.outer(s, u)
    t = np.linalg.norm(perp, axis=1) * np.sign(perp @ (np.linalg.svd(P, full_matrices=False)[2][1]))
    span = np.ptp(s) + 1e-9
    bend = abs(np.polyfit(s / span, t / span, 2)[0])       # normalised parabolic curvature of the axis
    return {"elongation": float(w[0] / w[1]), "flatness": float(w[1] / w[2]), "bend": float(bend)}


def blob(n, extent):
    """The current model's placement: an isotropic cloud (no shape) matched to the organ's size."""
    return RNG.standard_normal((n, 3)) * (extent / 3.0)


def rel(d1, d2):
    ks = ["elongation", "flatness", "bend"]
    return float(np.mean([abs(d1[k] - d2[k]) / (abs(d1[k]) + abs(d2[k]) + 1e-6) for k in ks]))


def main():
    d = np.load("data/mosta/mouse_3d.npz", allow_pickle=True)
    xyz, tissue = d["xyz"], d["tissue"]
    forward = {"heart": heart_3d(+1.0)[0], "neural tube": neural_tube_3d()[0]}
    real_of = {"heart": "Heart", "neural tube": "Brain"}

    rows = {}
    print(f"{'organ':12s} {'d(real,cloud)':>14s} {'d(real,forward)':>16s}  closer?")
    reals = {}
    for name, tname in real_of.items():
        R = xyz[tissue == tname]
        reals[name] = R
        dr = desc3d(R)
        dc = desc3d(blob(len(R), (R.max(0) - R.min(0)).mean()))
        df = desc3d(forward[name])
        distc, distf = rel(dr, dc), rel(dr, df)
        rows[name] = {"real": {k: round(dr[k], 3) for k in dr}, "cloud": {k: round(dc[k], 3) for k in dc},
                      "forward": {k: round(df[k], 3) for k in df},
                      "dist_cloud": round(distc, 3), "dist_forward": round(distf, 3), "closer": bool(distf < distc)}
        print(f"{name:12s} {distc:14.3f} {distf:16.3f}  {'YES' if distf < distc else 'no'}")

    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(rows, open("data/organ_cascade/organ_3d_vs_real.json", "w"), indent=1)

    # ---------------- figure: real / cloud / forward per organ, in 3D ----------------
    fig = plt.figure(figsize=(16, 8.5), facecolor="#0d1017")
    cols = {"heart": "#ef4444", "neural tube": "#38bdf8"}
    for r, name in enumerate(real_of):
        R = reals[name]; C = blob(len(R), (R.max(0) - R.min(0)).mean()); F = forward[name]
        for c, (lab, P, col) in enumerate([("real E9.5 (MOSTA 3D)", R, cols[name]),
                                           ("cloud (current model)", C, "#64748b"),
                                           ("forward organ model", F, cols[name])]):
            a = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d"); a.set_facecolor("#0d1017")
            Q = P - P.mean(0); sub = RNG.choice(len(Q), min(2500, len(Q)), replace=False)
            a.scatter(Q[sub, 0], Q[sub, 1], Q[sub, 2], s=3, c=col, linewidths=0)
            a.set_axis_off(); a.set_box_aspect((1, 1, 1)); a.view_init(elev=18, azim=-70)
            a.set_title(f"{name}: {lab}", color="#cbd5e1", fontsize=9)
    d0 = rows["heart"]; d1 = rows["neural tube"]
    fig.suptitle(f"3D before/after on the real E9.5 MOSTA reconstruction -- forward shape vs cloud, distance to "
                 f"real.  heart {d0['dist_cloud']}->{d0['dist_forward']}, neural {d1['dist_cloud']}->"
                 f"{d1['dist_forward']}", color="#e2e8f0", fontsize=12)
    fig.tight_layout()
    fig.savefig("data/organ_cascade/organ_3d_vs_real.png", dpi=125, facecolor="#0d1017")
    print("\nsaved data/organ_cascade/organ_3d_vs_real.{json,png}")


if __name__ == "__main__":
    main()
