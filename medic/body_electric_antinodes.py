"""
Antinodes of the body electric -- the discrete loci where organs (and limbs) sprout.
====================================================================================

The embryo's own gap-junction operator (kNN Laplacian) has low STANDING-WAVE modes
whose antinodes are the body's organizing loci -- the same electric-body frame that
places the limbs, the face and the mammary line (Paper #4). This module extracts the
antinode LADDER:

  * the ANTERO-POSTERIOR harmonics (mode 1 = fundamental with antinodes at head & tail,
    mode 2 adds a mid-body antinode, mode 3 adds two, ...) give a sequence of AP levels;
  * the LEFT-RIGHT mode gives the bilateral frame -- its NODE is the midline (where a
    MIDLINE organ sits) and its two ANTINODES are the left/right sides (a PAIRED organ).

Organs do NOT read Vm to find their place; they FILL these antinodes in the order set
by the master organ clock (PRC2/Hox withdrawal). This module only computes the antinode
positions; medic.organ_sprouting does the clock-ordered filling.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.body_electric_antinodes
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh


def _laplacian(P, k=12):
    n = len(P)
    nb = cKDTree(P).query(P, k=min(k, n))[1][:, 1:]
    rr = np.repeat(np.arange(n), nb.shape[1]); cc = nb.ravel()
    W = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)).tocsr()
    W = ((W + W.T) > 0).astype(float)
    return diags(np.asarray(W.sum(1)).ravel()) - W


def _profile_extrema(coord, u, nb_bins=40, min_prom=0.15):
    """1-D standing-wave profile of mode u along `coord`: bin, average, find the
    interior extrema (antinodes) and the zero-crossings (nodes). Returns antinode
    coords (in [0,1] over the coord range) with their |amplitude|."""
    c = (coord - coord.min()) / (np.ptp(coord) + 1e-9)
    edges = np.linspace(0, 1, nb_bins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(nb_bins, np.nan)
    for b in range(nb_bins):
        m = (c >= edges[b]) & (c < edges[b + 1])
        if m.sum() >= 3:
            prof[b] = u[m].mean()
    ok = ~np.isnan(prof)
    if ok.sum() < 5:
        return []
    x = mid[ok]; y = prof[ok]
    y = (y - y.mean())
    amp = np.abs(y).max() + 1e-9
    anti = []
    for i in range(1, len(y) - 1):                       # interior local extrema of |y|
        if (abs(y[i]) >= abs(y[i - 1])) and (abs(y[i]) > abs(y[i + 1])) and abs(y[i]) > min_prom * amp:
            anti.append((float(x[i]), float(abs(y[i]) / amp)))
    # include the two ends (fundamental antinodes) if the profile is strong there
    if abs(y[0]) > 0.6 * amp:
        anti.insert(0, (float(x[0]), float(abs(y[0]) / amp)))
    if abs(y[-1]) > 0.6 * amp:
        anti.append((float(x[-1]), float(abs(y[-1]) / amp)))
    return anti


def body_electric_antinodes(P, k_modes=16, verbose=False):
    """Return the antinode ladder of the body-electric operator.

    dict(ap_levels=[sorted AP fractions of antinodes, anterior->posterior],
         ap_from_mode={mode_rank: [ap fractions]},
         lr_node=0.0, lr_sign present, has_lr=bool)
    AP = long axis (x), DV = y, LR = z.
    """
    n = len(P)
    L = _laplacian(P)
    kk = min(k_modes, n - 2)
    # DETERMINISTIC START (cycle 75, the regen-to-regen drift conviction; REVISED cycle 79):
    # eigsh without v0 draws its start from numpy's per-process GLOBAL state -- on
    # near-degenerate body modes the returned basis then flips per launch (identical-code
    # regens differed at every mesh frame while the 9k cloud checksummed identical). The
    # cycle-75 v0=ones was WRONG for a Laplacian: ones IS the null eigenvector (L.1=0), so
    # ARPACK's start deflates to zero (error -9, caught at build_base 120k -- it had survived
    # the movie regens by iteration luck). A SEEDED random vector is deterministic AND generic.
    vv, UU = eigsh(L, k=kk, which="SM", v0=np.random.default_rng(0).standard_normal(n))
    o = np.argsort(vv); vv, UU = vv[o], UU[:, o]
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    def axis_power(u):
        # how much of the mode's variation lies along each axis (binned variance)
        out = []
        for coord in (x, y, z):
            c = (coord - coord.min()) / (np.ptp(coord) + 1e-9)
            b = np.clip((c * 20).astype(int), 0, 19)
            means = np.array([u[b == j].mean() if (b == j).sum() > 2 else 0.0 for j in range(20)])
            out.append(means.var())
        return np.array(out)                              # [AP, DV, LR]

    ap_modes, dv_modes, lr_modes = [], [], []
    for i in range(1, UU.shape[1]):                       # skip the constant mode 0
        p = axis_power(UU[:, i])
        dom = np.argmax(p)
        if dom == 0 and p[0] > 1.5 * (p[1] + 1e-9):       # AP-dominant standing wave
            ap_modes.append(i)
        elif dom == 1 and p[1] > 1.2 * (p[0] + 1e-9):     # DV-dominant standing wave
            dv_modes.append(i)
        elif dom == 2 and p[2] > 1.2 * (p[0] + 1e-9):     # LR-dominant (bilateral)
            lr_modes.append(i)

    def _ladder(coord, modes, n_harm=4, merge=0.06):
        from_mode, allx = {}, []
        for i in modes[:n_harm]:
            anti = _profile_extrema(coord, UU[:, i])
            from_mode[i] = [round(a, 3) for a, _ in anti]
            allx += anti
        allx.sort()
        merged = []
        for xf, amp in allx:
            if merged and abs(xf - merged[-1][0]) < merge:
                if amp > merged[-1][1]:
                    merged[-1] = (xf, amp)
            else:
                merged.append((xf, amp))
        return [round(a, 3) for a, _ in merged], from_mode

    ap_levels, ap_from_mode = _ladder(x, ap_modes)
    dv_levels, dv_from_mode = _ladder(y, dv_modes)
    # DV named bands: ventral (low y) / mid / dorsal (high y) = the ventral/dorsal DV antinodes
    dv_ventral = dv_levels[0] if dv_levels else 0.15
    dv_dorsal = dv_levels[-1] if dv_levels else 0.85
    dv_mid = dv_levels[len(dv_levels) // 2] if dv_levels else 0.5

    res = dict(ap_levels=ap_levels, ap_from_mode=ap_from_mode,
               dv_levels=dv_levels, dv_ventral=dv_ventral, dv_mid=dv_mid, dv_dorsal=dv_dorsal,
               has_lr=bool(lr_modes), lr_mode=lr_modes[0] if lr_modes else None,
               n_ap_harmonics=len(ap_modes), n_dv_harmonics=len(dv_modes))
    if verbose:
        print(f"  {len(ap_modes)} AP harmonics {ap_modes[:6]}, {len(dv_modes)} DV harmonics "
              f"{dv_modes[:4]}, {len(lr_modes)} LR mode(s) {lr_modes[:3]}")
        print(f"  AP antinode ladder (ant->post): {ap_levels}")
        print(f"  DV antinode ladder (ventral->dorsal): {dv_levels}  "
              f"(ventral={dv_ventral} mid={dv_mid} dorsal={dv_dorsal})")
    return res


def main():
    import warnings; warnings.filterwarnings("ignore")
    from medic.unified_embryo import simulate
    print("growing a tetrapod to read its body-electric antinodes ...")
    _, m = simulate(use_ecm=True, limb_buds=True, convergent_ext=1.0)
    print(f"N={m['n']}  (antinode ladder of the grown body):")
    body_electric_antinodes(m["pos"], verbose=True)


if __name__ == "__main__":
    main()
