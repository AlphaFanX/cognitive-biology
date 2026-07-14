"""
Basic-vertebrate development MOVIE (GIF) via the NCA+LGM engine.
================================================================
Runs the unified four-head embryo (medic.unified_embryo.simulate) and renders the growth as a
rotating 3D point cloud coloured by membrane voltage, then writes an animated GIF (Pillow).

  heads: telomere/PRC2 clock · division · differentiation · convergent-extension migration
         · cadherin sorting + integrin/ECM fascia + neural fold  (all four, one forward pass)

  --start N   population start count (1 = literal single cell; 60 = blastula blob)
  --end   N   final cell count (e.g. 20000 for the fuller general-vertebrate form)
  --out   stem   output filename stem (data/movie/<stem>.gif)

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.basic_vertebrate_movie --start 1 --end 20000
"""
from __future__ import annotations
import argparse
import io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image

from medic.unified_embryo import simulate, _symmetrize, VMIN, VMAX

BG = "#0d1017"
CMAP = plt.get_cmap("coolwarm")
NORM = colors.Normalize(vmin=VMIN, vmax=VMAX)
MAX_RENDER = 9000                      # subsample the scatter for speed at high cell counts


def render(n_start=1, n_end=4000, out_stem="basic_vertebrate_development"):
    origin = "one cell" if n_start == 1 else f"the {n_start}-cell blastula"
    print(f"growing the basic vertebrate from {origin} -> {n_end} cells (unified 4-head NCA+LGM) ...")
    frames, _ = simulate(use_ecm=True, seed=0, n_start=n_start, n_end=n_end, verbose=True)

    sym = [_symmetrize(P, V) for (_, _, _, P, V, _) in frames]
    Pf = sym[-1][0]
    c = Pf.mean(0); c[2] = 0.0
    scale = 1.7 / (0.5 * max(np.ptp(Pf[:, 0]), np.ptp(Pf[:, 1]), np.ptp(Pf[:, 2])) + 1e-9)
    Qf = (Pf - c) * scale
    pad = 1.12
    xl = (Qf[:, 0].min() * pad, Qf[:, 0].max() * pad)
    yl = (Qf[:, 1].min() * pad, Qf[:, 1].max() * pad)
    zl = (Qf[:, 2].min() * pad, Qf[:, 2].max() * pad)
    box = (np.ptp(Qf[:, 0]), np.ptp(Qf[:, 1]) + 1e-3, np.ptp(Qf[:, 2]))
    rng = np.random.default_rng(0)

    imgs = []
    nfr = len(frames)
    for fi, ((born, t_hpf, prc2, _, _, _), (Ps, Vs)) in enumerate(zip(frames, sym)):
        frac = fi / (nfr - 1)
        Q = (Ps - c) * scale
        n = len(Q)
        if n > MAX_RENDER:                                  # subsample the render (keep the true N in the HUD)
            sel = rng.choice(n, MAX_RENDER, replace=False)
            Qd, Vd = Q[sel], Vs[sel]
        else:
            Qd, Vd = Q, Vs
        s = float(np.clip(1600.0 / np.sqrt(len(Qd)), 2.5, 55.0))
        fig = plt.figure(figsize=(7.2, 5.7), dpi=110)
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(BG)
        ax.scatter(Qd[:, 0], Qd[:, 1], Qd[:, 2], c=CMAP(NORM(Vd)), s=s, linewidths=0, depthshade=False)
        ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
        ax.set_box_aspect(box, zoom=1.6)
        ax.set_axis_off()
        ax.view_init(elev=14, azim=-72 + 52 * frac)
        fig.text(0.5, 0.95, f"Basic vertebrate — grown from {origin} by the NCA + LGM",
                 ha="center", va="top", color="#e8eef4", fontsize=13, weight="bold")
        fig.text(0.5, 0.905, "telomere/PRC2 clock · division · differentiation · convergent extension · cadherin + fascia · neural fold",
                 ha="center", va="top", color="#8aa0b4", fontsize=7.5)
        fig.text(0.5, 0.055, f"N = {n:,} {'cell' if n == 1 else 'cells'}   ·   {t_hpf:.0f} hpf   ·   PRC2 {prc2:.2f}",
                 ha="center", va="bottom", color="#7dd3fc", fontsize=11)
        fig.text(0.5, 0.02, "membrane voltage:  blue = hyperpolarised (neural)   →   red = depolarised (yolk / epidermis)",
                 ha="center", va="bottom", color="#68788a", fontsize=7.5)
        fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.9)
        buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig)
        buf.seek(0); imgs.append(Image.open(buf).convert("RGB"))
        if fi % 10 == 0 or fi == nfr - 1:
            print(f"  frame {fi:2d}/{nfr-1}  N={n:6d}  t={t_hpf:4.1f} hpf")

    out = Path(f"data/movie/{out_stem}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [110] * len(imgs)
    durations[0] = 700
    durations[-1] = 2200
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True)
    print(f"\nsaved {out}  ({len(imgs)} frames, {out.stat().st_size/1e6:.1f} MB)")
    try:
        dsk = Path.home() / "Desktop" / out.name
        dsk.write_bytes(out.read_bytes())
        print(f"copied -> {dsk}")
    except Exception as e:
        print("desktop copy skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=4000)
    ap.add_argument("--out", type=str, default="basic_vertebrate_development")
    a = ap.parse_args()
    render(n_start=a.start, n_end=a.end, out_stem=a.out)
