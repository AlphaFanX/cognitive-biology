"""Stitch the NCA+LGM vertebrate embryo to the MakeHuman body in ONE continuous point-cloud sequence:
1 cell -> vertebrate (grown by the NCA+LGM) -> (morph) -> the MakeHuman body as a point cloud.

The vertebrate's antero-posterior axis is stood upright to the body's head->foot axis, then the cloud is
morphed (height-ranked correspondence) onto points sampled from the MakeHuman body, its membrane-voltage
colour fading to skin as it becomes the adult form. Overwrites data/organ_cascade/nca_growth/nca_00..12
so the viewer's right panel now runs the whole journey in the generated representation.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.nca_to_human_stitch
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import medic.basic_vertebrate_movie as bvm

OUT = "data/organ_cascade/nca_growth"
NG, NM = 8, 5                                            # growth frames, morph frames  (total 13)
SKIN = np.array([0.90, 0.76, 0.66])


def _standup(P):
    """rotate so the longest (AP) axis -> y, then centre."""
    Pc = P - P.mean(0)
    ax = int(np.argmax(Pc.max(0) - Pc.min(0)))
    order = [0, 1, 2]; order[1], order[ax] = order[ax], order[1]
    return Pc[:, order]


def _normy(P):
    return P / (P[:, 1].max() - P[:, 1].min() + 1e-9) * 2.0


def main():
    os.makedirs(OUT, exist_ok=True)
    frames, _ = bvm.simulate(use_ecm=True, seed=0, n_start=1, n_end=2500, verbose=True)
    sym = [bvm._symmetrize(P, V) for (_, _, _, P, V, _) in frames]

    # common orientation + scale from the final vertebrate; the human box is the target
    d = np.load("data/bodybase/makehuman_decimated.npz"); Ph = _normy(_standup(d["V"].astype(float)))
    # keep the human head at top: MakeHuman y is already up, standup may flip -> ensure head (max y) up
    Pv_last = _normy(_standup(sym[-1][0]))
    # box = union of the stood-up vertebrate and the human
    allP = np.vstack([Pv_last, Ph])
    # plot (x, z, y): data-y (height) -> matplotlib's Z (up), so the figure stands upright
    xl = (allP[:, 0].min() * 1.15, allP[:, 0].max() * 1.15)
    yl = (allP[:, 2].min() * 1.25, allP[:, 2].max() * 1.25)
    zl = (allP[:, 1].min() * 1.08, allP[:, 1].max() * 1.08)
    boxaspect = (np.ptp(allP[:, 0]), np.ptp(allP[:, 2]) + 1e-3, np.ptp(allP[:, 1]))
    rng = np.random.default_rng(0)

    def draw(Q, C, k, label):
        n = len(Q)
        if n > 9000:
            sel = rng.choice(n, 9000, replace=False); Q, C = Q[sel], C[sel]
        s = float(np.clip(2600.0 / np.sqrt(len(Q)), 4.0, 80.0))
        fig = plt.figure(figsize=(5.0, 6.4), dpi=150); fig.patch.set_facecolor(bvm.BG)
        ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(bvm.BG)
        ax.scatter(Q[:, 0], Q[:, 2], Q[:, 1], c=C, s=s, linewidths=0, depthshade=False)  # height -> up
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl); ax.set_box_aspect(boxaspect, zoom=1.5)
        ax.set_axis_off(); ax.view_init(elev=6, azim=-90)
        fig.text(0.5, 0.045, label, ha="center", color="#7dd3fc", fontsize=11)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(f"{OUT}/nca_{k:02d}.png", facecolor=bvm.BG); plt.close(fig)

    # --- growth frames 0..NG-1: 1 cell -> vertebrate (stood up) ---
    nfr = len(frames)
    for k in range(NG):
        si = round(k / (NG - 1) * (nfr - 1))
        (born, t_hpf, prc2, _, _, _), (P, V) = frames[si], sym[si]
        Q = _normy(_standup(P))
        draw(Q, bvm.CMAP(bvm.NORM(V)), k, f"NCA+LGM · N={len(P):,} cells · {t_hpf:.0f} hpf")
        print(f"growth {k}/{NG-1} N={len(P)}")

    # --- morph frames NG..NG+NM-1: vertebrate cloud -> MakeHuman body cloud (voltage -> skin) ---
    Pv, Vv = _normy(_standup(sym[-1][0])), sym[-1][1]
    n = min(len(Pv), len(Ph), 5000)
    iv = rng.choice(len(Pv), n, replace=False); ih = rng.choice(len(Ph), n, replace=False)
    Pvs, Vvs, Phs = Pv[iv], Vv[iv], Ph[ih]
    # height-ranked correspondence (head->head, tail->feet)
    ov = np.lexsort((Pvs[:, 0], -Pvs[:, 1])); oh = np.lexsort((Phs[:, 0], -Phs[:, 1]))
    Pvs, Vvs, Phs = Pvs[ov], Vvs[ov], Phs[oh]
    volt = bvm.CMAP(bvm.NORM(Vvs))[:, :3]
    for j in range(NM):
        t = (j + 1) / NM
        Q = (1 - t) * Pvs + t * Phs
        C = (1 - t) * volt + t * SKIN
        draw(Q, C, NG + j, "maturing into the human body" if t < 1 else "MakeHuman body (generated form)")
        print(f"morph {j}/{NM-1} t={t:.2f}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
