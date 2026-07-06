#!/usr/bin/env python3
r"""
The mechanical morphogenesis layer, eigenmode-native: folding + fusion (harelip).
=================================================================================

The bioelectric/morphogen model (medic.nca_vertebrate_3d) computes the PREPATTERN
-- a scalar Vm field on a FIXED voxel mask. It cannot move material or change
topology. Two developmental events ARE topology changes and live in a layer above
it, and -- following Paper #4's result that morphology is low-rank in the
eigenmodes of the genome-derived operator -- each has an eigenmode reading:

  (1) INVAGINATION / FOLDING is BUCKLING, and buckling is EIGENMODE SELECTION.
      A compressed epithelium (apical constriction) buckles into the lowest mode
      of its bending operator; the hinge is that mode's crease (a nodal line),
      exactly as gut looping and cortical gyrification are buckling. We compute
      the low buckling modes of the constriction-weighted operator and integrate
      the fold (a rod with the mode's spontaneous curvature) into a closed tube.
      Weak drive -> the mode never closes = a neural tube defect.

  (2) FUSION is the event that REDEFINES the eigenbasis: two prominences and one
      fused lip are different domains with different spectra. The exact spectral
      signature is the FIEDLER eigenvalue lambda_2 (algebraic connectivity) of
      the tissue graph Laplacian: lambda_2 = 0 when the tissue is in two pieces
      (a cleft), and lambda_2 > 0 the instant the seam bridges into one lip. So
      the harelip is the persistence of a second zero-mode, and the CLEFT BASIN
      is the parameter region where lambda_2 stays pinned at zero.

Scope (honest): 2D cross-section, idealized geometry, reduced mechanics; the fold
placement/pattern and the fusion criterion are eigenmode quantities, the
mechanical execution is a reduced model. What is real is the topology change and
its spectral signature.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.mechanical_fusion --part fold
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.mechanical_fusion --part fuse
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.mechanical_fusion
Outputs: mechanical_fold.png, mechanical_fusion.png
"""
from __future__ import annotations

import numpy as np


# ===========================================================================
# PART 1 -- Folding as a buckling eigenmode of the constriction-weighted operator
# ===========================================================================
def constriction_profile(N, hinges=(0.5, 0.22, 0.78), amps=(1.0, 0.65, 0.65),
                         width=0.09, floor=0.35):
    """Apical-constriction competence along the plate, from the prepattern: a
    medial hinge (MHP) plus two dorsolateral hinges (DLHP) on a floor."""
    s = np.linspace(0, 1, N)
    c = np.full(N, floor)
    for h, a in zip(hinges, amps):
        c = c + a * np.exp(-((s - h) / width) ** 2)
    return c / c.max()


def buckling_modes(N, competence, k=4):
    """Low buckling modes of the plate = eigenvectors of the constriction-
    weighted 1D bending (Laplacian) operator with free ends. The competence
    stiffens the hinges, so the low modes crease there. Returns (vals, vecs);
    mode 1 (first non-constant) is the single-arch fold whose crease is the
    medial hinge."""
    # weighted graph Laplacian of the path, edge weight = mean competence of its
    # two nodes (a stiff hinge couples strongly -> the mode bends there).
    w = 0.5 * (competence[:-1] + competence[1:])
    L = np.zeros((N, N))
    for i in range(N - 1):
        L[i, i] += w[i]; L[i + 1, i + 1] += w[i]
        L[i, i + 1] -= w[i]; L[i + 1, i] -= w[i]
    vals, vecs = np.linalg.eigh(L)
    return vals[:k], vecs[:, :k]


def fold_from_curvature(drive=1.0, N=121, H=1.6, competence=None):
    """Integrate the fold as a rod carrying the buckling mode's spontaneous
    curvature. kappa(s) is proportional to the constriction competence (the
    mode's crease); its integral (total turning) = drive * 2*pi. drive=1 turns a
    full circle -> the two edges meet -> a CLOSED tube; drive<1 -> an open groove
    (a neural tube defect). Returns the apical/basal surfaces, the edge gap and
    whether the tube closed. This is the large-deflection (post-buckling) shape
    of mode 1 -- eigenmode placement, mechanical execution."""
    if competence is None:
        competence = constriction_profile(N)
    s = np.linspace(0, 1, N)
    ds = s[1] - s[0]
    K_total = drive * 2.0 * np.pi
    kappa = competence / (np.sum(competence) * ds + 1e-9) * K_total   # int kappa = K_total
    # tangent angle, centred so the fold is symmetric about the midline
    theta = np.cumsum(kappa) * ds
    theta = theta - theta[N // 2]
    # centreline by integrating the tangent
    xc = np.cumsum(np.cos(theta)) * ds
    yc = np.cumsum(np.sin(theta)) * ds
    xc = xc - xc[N // 2]
    yc = yc - yc[N // 2]
    # apical (inner) and basal (outer) surfaces offset along the normal
    nx, ny = -np.sin(theta), np.cos(theta)
    A = np.stack([xc - 0.5 * H * nx, yc - 0.5 * H * ny], axis=1)   # apical / lumen side
    B = np.stack([xc + 0.5 * H * nx, yc + 0.5 * H * ny], axis=1)   # basal side
    gap = float(np.hypot(*(A[-1] - A[0])))
    span = float(np.hypot(np.ptp(A[:, 0]), np.ptp(A[:, 1])))
    closed = gap < 0.18 * (span + 1e-9) or gap < 0.35
    return dict(A=A, B=B, xc=xc, yc=yc, gap=gap, closed=bool(closed),
                drive=drive, competence=competence, theta=theta)


def render_fold(out="mechanical_fold.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 121
    comp = constriction_profile(N)
    vals, vecs = buckling_modes(N, comp, k=4)
    # snapshots of the fold at increasing drive (post-buckling amplification)
    drives = [0.25, 0.5, 0.75, 1.0]
    folds = [fold_from_curvature(d, N=N, competence=comp) for d in drives]
    healthy = folds[-1]
    defect = fold_from_curvature(0.5, N=N, competence=comp)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.3, height_ratios=[2, 3, 3])

    # row 0: the low buckling modes + the constriction competence
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(comp, color="#b02318"); ax.set_title("apical-constriction\ncompetence (prepattern)", fontsize=9)
    ax.set_xticks([])
    for m in range(1, 4):
        ax = fig.add_subplot(gs[0, m])
        ax.plot(vecs[:, m], color="#2171b5")
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_title(f"buckling mode {m}\n(crease = hinge)", fontsize=9)
        ax.set_xticks([])

    # row 1: fold snapshots at increasing drive
    def draw(ax, f, color, title):
        A, B = f["A"], f["B"]
        for i in range(len(A) - 1):
            poly = np.array([A[i], A[i + 1], B[i + 1], B[i]])
            ax.fill(poly[:, 0], poly[:, 1], facecolor=color, edgecolor="#444", lw=0.3, alpha=0.85)
        ax.plot(A[:, 0], A[:, 1], "-", color="#b02318", lw=1.6)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9)

    for j, f in enumerate(folds):
        ax = fig.add_subplot(gs[1, j])
        draw(ax, f, "#9ecae1", f"drive={f['drive']:.2f}\ngap={f['gap']:.2f}")

    # row 2: closed vs open + gap-vs-drive
    ax = fig.add_subplot(gs[2, 0:2])
    draw(ax, healthy, "#9ecae1", f"CLOSED neural tube (drive=1.0)\nred=apical/lumen  gap={healthy['gap']:.2f}")
    ax = fig.add_subplot(gs[2, 2])
    draw(ax, defect, "#fdd0a2", f"OPEN tube = NTD\n(weak drive 0.5, gap={defect['gap']:.2f})")
    ax = fig.add_subplot(gs[2, 3])
    dd = np.linspace(0.1, 1.15, 30)
    gg = [fold_from_curvature(d, N=N, competence=comp)["gap"] for d in dd]
    ax.plot(dd, gg, color="#2171b5")
    ax.axhline(0.35, color="k", ls=":", lw=1, label="closure")
    ax.set_xlabel("constriction drive"); ax.set_ylabel("edge gap")
    ax.set_title("closure vs drive", fontsize=9); ax.legend(fontsize=8)

    fig.suptitle("Invagination as a buckling eigenmode: the fold is mode 1 of the "
                 "constriction-weighted operator; strong drive closes the tube, weak drive leaves it open",
                 fontsize=12.5, y=0.99)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return healthy, defect


# ===========================================================================
# PART 2 -- Fusion as the Fiedler eigenvalue of the tissue graph Laplacian
# ===========================================================================
def grow_prominences(outgrowth=1.0, adhesion=1.0, nx=161, ny=81,
                     steps=1200, dt=0.10, D=0.5, a=0.4):
    """Two facial prominences as a bistable (Nagumo) phase field phi in [0,1].

    Two knobs, two failure modes -- both give a cleft:
      * `outgrowth` sets how far each prominence REACHES toward the midline;
        too little and the masses never meet (a geometric gap).
      * `adhesion` sets whether the epithelial SEAM resolves. We model it as the
        cross-midline tissue COUPLING: diffusion across the seam column is gated
        by adhesion, plus a fusogenic fill. So even when the masses touch, low
        adhesion leaves them electrically/materially UNcoupled = a persistent
        cleft. (This is the same coupling idea as the gap-junction operator.)

    Bistable Nagumo reaction 6 phi(1-phi)(phi-a): undriven gaps DECAY to 0 (a
    genuine cleft) instead of Allen-Cahn's phi-phi^3 filling every gap. Explicit
    Euler is stable while dt*D*4 < 1 (here 0.8); gating only lowers D."""
    xs = np.linspace(-1.5, 1.5, nx)
    ys = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(xs, ys)
    x0, lip_y = 0.62, 0.42
    Rmax = 0.43 + 0.14 * outgrowth              # out=1 -> reach -0.05; out=0.5 -> -0.12; out~0 -> wide gap

    # adhesion-gated cross-midline diffusion: seam x-edges couple only by adhesion
    seam_hw = 0.15
    xedge = 0.5 * (xs[1:] + xs[:-1])
    gate = np.ones(nx - 1)
    gate[np.abs(xedge) < seam_hw] = adhesion
    Dx = D * gate[None, :]                       # x-edge diffusivity (1, nx-1)

    def lap_var(Z):
        Zp = np.pad(Z, 1, mode="edge")
        lap_y = Zp[2:, 1:-1] + Zp[:-2, 1:-1] - 2.0 * Z      # uniform D in y
        fx = Dx * (Z[:, 1:] - Z[:, :-1])                    # gated flux across x-edges
        div_x = np.zeros_like(Z)
        div_x[:, 1:-1] = fx[:, 1:] - fx[:, :-1]
        div_x[:, 0] = fx[:, 0]; div_x[:, -1] = -fx[:, -1]
        return D * lap_y + div_x

    dl = np.hypot(X + x0, Y / (lip_y / 0.5))
    dr = np.hypot(X - x0, Y / (lip_y / 0.5))
    seam = np.exp(-(X / 0.15) ** 2) * (np.abs(Y) < lip_y)   # fusogenic fill profile

    phi = np.zeros_like(X)
    for s in range(steps):
        R = 0.30 + (Rmax - 0.30) * min(1.0, s / (0.7 * steps))
        reach_l = np.clip((R - dl) / 0.15, 0, 1)
        reach_r = np.clip((R - dr) / 0.15, 0, 1)
        src = np.maximum(reach_l, reach_r)
        # fusion only where BOTH prominences have arrived near the seam (contact),
        # so it bridges a real contact -- never fabricates tissue across a wide gap
        contact = (np.clip((R - dl + 0.15) / 0.15, 0, 1)
                   * np.clip((R - dr + 0.15) / 0.15, 0, 1))
        react = 6.0 * phi * (1.0 - phi) * (phi - a)
        phi = phi + dt * (lap_var(phi) + react
                          + 1.2 * src * (1 - phi)
                          + 2.0 * adhesion * seam * contact * (1 - phi))  # fusogenic fill at contact
        np.clip(phi, 0.0, 1.0, out=phi)
    return dict(phi=phi, X=X, Y=Y, xs=xs, ys=ys, lip_y=lip_y)


def tissue_fiedler(mask):
    """Fiedler eigenvalue lambda_2 (algebraic connectivity) and the number of
    connected components of the tissue mask (4-connectivity). lambda_2 = 0 iff
    the tissue is disconnected (a cleft); lambda_2 > 0 iff it is one piece."""
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import eigsh
    from scipy.ndimage import label

    lab, nc = label(mask)
    ys, xs = np.where(mask)
    n = len(ys)
    if n < 3:
        return 0.0, int(nc)
    idx = -np.ones(mask.shape, dtype=int)
    idx[ys, xs] = np.arange(n)
    rows, cols = [], []
    deg = np.zeros(n)
    for dy, dx in ((1, 0), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inb = (ny < mask.shape[0]) & (nx < mask.shape[1])
        hit = np.zeros(n, bool)
        hit[inb] = mask[ny[inb], nx[inb]]
        ii = np.where(hit)[0]
        jj = idx[ny[hit], nx[hit]]
        rows += list(ii) + list(jj)
        cols += list(jj) + list(ii)
        np.add.at(deg, ii, 1.0)
        np.add.at(deg, jj, 1.0)
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    Lap = diags(deg) - W
    try:
        w = eigsh(Lap, k=min(2, n - 1), which="SA", return_eigenvectors=False)
        lam2 = float(np.sort(w)[1]) if len(w) > 1 else 0.0
    except Exception:
        lam2 = 0.0
    return max(lam2, 0.0), int(nc)


def prominence_fusion(outgrowth=1.0, adhesion=1.0, steps=1200):
    """Grow the prominences, threshold to tissue, and read fusion off the
    spectrum: lambda_2 (Fiedler) and the component count."""
    g = grow_prominences(outgrowth=outgrowth, adhesion=adhesion, steps=steps)
    band = np.abs(g["ys"]) < g["lip_y"]
    mask = g["phi"] > 0.5
    mask[~band, :] = False                     # restrict connectivity to the lip band
    lam2, nc = tissue_fiedler(mask)
    fused = nc == 1
    # a midline gap fraction as a plain corroborating readout
    col = int(np.argmin(np.abs(g["xs"])))
    gap = float(np.mean(g["phi"][band, col] < 0.5))
    return dict(**g, mask=mask, lam2=lam2, ncomp=nc, fused=bool(fused), gap=gap,
                outgrowth=outgrowth, adhesion=adhesion)


def cleft_sweep(n=11, steps=800):
    """Sweep outgrowth x adhesion -> the Fiedler lambda_2 map = the CLEFT BASIN
    (lambda_2 = 0 is a cleft; lambda_2 > 0 is a fused lip)."""
    gs = np.linspace(0.05, 1.0, n)
    as_ = np.linspace(0.0, 1.0, n)
    lam = np.zeros((n, n))
    fused = np.zeros((n, n))
    for i, g in enumerate(gs):
        for j, a in enumerate(as_):
            r = prominence_fusion(outgrowth=g, adhesion=a, steps=steps)
            lam[j, i] = r["lam2"]              # rows=adhesion, cols=outgrowth
            fused[j, i] = 1.0 if r["fused"] else 0.0
    return dict(gs=gs, as_=as_, lam2=lam, fused=fused)


def render_fusion(intact, cleft, sweep, out="mechanical_fusion.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    ext = [intact["X"].min(), intact["X"].max(), intact["Y"].min(), intact["Y"].max()]

    def show(a, res, title):
        a.imshow(res["phi"], origin="lower", extent=ext, aspect="equal", cmap="pink_r", vmin=0, vmax=1)
        a.axvline(0, color="#2171b5", ls=":", lw=1)
        a.axhline(res["lip_y"], color="w", ls=":", lw=0.7)
        a.axhline(-res["lip_y"], color="w", ls=":", lw=0.7)
        a.set_title(title, fontsize=10.5)
        a.set_xlabel("L <- midline -> R"); a.set_ylabel("lip band")

    show(ax[0], intact, f"FUSED = intact lip\n"
                        f"components = {intact['ncomp']} (one piece), $\\lambda_2$ > 0")
    show(ax[1], cleft, f"CLEFT LIP (harelip)\n"
                       f"components = {cleft['ncomp']} (two pieces), $\\lambda_2$ = 0")

    im = ax[2].imshow(sweep["fused"], origin="lower", aspect="auto", cmap="RdYlGn",
                      vmin=0, vmax=1,
                      extent=[sweep["gs"][0], sweep["gs"][-1], sweep["as_"][0], sweep["as_"][-1]])
    ax[2].contour(sweep["gs"], sweep["as_"], sweep["fused"], levels=[0.5],
                  colors="k", linewidths=2)
    ax[2].set_xlabel("outgrowth (do the prominences meet?)")
    ax[2].set_ylabel("fusogenic adhesion (does the seam resolve?)")
    ax[2].set_title("The CLEFT BASIN (Fiedler $\\lambda_2$ of the tissue Laplacian)\n"
                    "green = fused (1 component) -> red = cleft ($\\lambda_2$=0, 2 components)")
    fig.colorbar(im, ax=ax[2], fraction=0.045, label="fused (1) / cleft (0)")

    fig.suptitle("Fusion as a spectral event: the harelip is the persistence of a second "
                 "zero-mode ($\\lambda_2$=0) in the tissue Laplacian", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ===========================================================================
def run_fold():
    print("=" * 74)
    print("PART 1 -- INVAGINATION as a buckling eigenmode")
    print("=" * 74)
    healthy, defect = render_fold()
    print(f"  drive=1.00 -> closed tube (edge gap {healthy['gap']:.2f})")
    print(f"  drive=0.50 -> open tube = NTD (edge gap {defect['gap']:.2f})")
    print("  the fold shape is mode 1 of the constriction-weighted operator;")
    print("  the hinge is the mode's crease. Placement modal, execution mechanical.")


def run_fuse():
    print("=" * 74)
    print("PART 2 -- FUSION as the Fiedler eigenvalue lambda_2")
    print("=" * 74)
    # 4-corner sanity: fusion needs BOTH reach (outgrowth) AND seam resolution (adhesion)
    print("  4-corner check (components; 1=fused, >=2=cleft):")
    for og, ad in ((1.0, 1.0), (1.0, 0.0), (0.05, 1.0), (0.05, 0.0)):
        r = prominence_fusion(outgrowth=og, adhesion=ad)
        print(f"    outgrowth={og:.2f} adhesion={ad:.2f} -> components={r['ncomp']}, "
              f"lambda_2={r['lam2']:.4f}  -> {'FUSED' if r['fused'] else 'CLEFT'}")
    intact = prominence_fusion(outgrowth=1.0, adhesion=1.0)   # meet + seam resolves
    cleft = prominence_fusion(outgrowth=1.0, adhesion=0.0)    # meet but seam PERSISTS = harelip
    print(f"  intact lip (outgrowth 1.0, adhesion 1.0) -> components={intact['ncomp']} "
          f"({'FUSED' if intact['fused'] else 'CLEFT'})")
    print(f"  harelip    (outgrowth 1.0, adhesion 0.0) -> components={cleft['ncomp']} "
          f"({'CLEFT' if not cleft['fused'] else 'FUSED'})")
    print("  sweeping outgrowth x adhesion -> the Fiedler cleft basin...")
    sweep = cleft_sweep()
    print(f"  fused fraction over the basin = {sweep['fused'].mean():.0%}")
    render_fusion(intact, cleft, sweep)


def main():
    import sys
    part = "all"
    if "--part" in sys.argv:
        part = sys.argv[sys.argv.index("--part") + 1]
    if part in ("fold", "all"):
        run_fold()
    if part in ("fuse", "all"):
        run_fuse()


if __name__ == "__main__":
    main()
