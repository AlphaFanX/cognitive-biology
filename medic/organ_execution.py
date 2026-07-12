"""
Integrated per-organ execution forward pass.

For each organ this runs the four effectors of the organ head in order, on one tissue
domain, and reports what each stage produced:

  1. CONNEXIN FRAME   -- low eigenmodes of the gap-junction Laplacian on the domain.
                         mode 1 is the organ's primary (AP) axis; an antisymmetric mode
                         is the midline. (frame = orientation + symmetry)
  2. MORPHOGEN RD     -- an oriented Turing field at a genome-set wavelength, laid in the
                         frame's axes, so the spacing comes from the wavelength and the
                         orientation/symmetry from the frame. (spacing)
  3. LATERAL INHIBITION -- Delta-Notch winner-take-all resolves the graded field into
                         discrete, spaced units. (discretization)
  4. CADHERIN COHESION -- cells are typed by their unit; differential adhesion (majority
                         smoothing) sorts them into clean domains and binds each unit into
                         one connected component; the whole stays one body. (cohesion)

Honest scope: a schematic 2D forward pass that assembles the series' four operators; each
operator is validated separately elsewhere. The absolute geometry and the causal
prepattern-precedes-form claim are not asserted here.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.organ_execution
Out: data/organ_execution.{json,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage

N = 80  # grid

# organ -> (domain shape, genome-set spacing n_ap x n_lat, label)
ORGANS = [
    ("heart",       dict(shape="ellipse",  ax=0.42, ay=0.30, n_ap=2, n_lat=1)),
    ("gut tube",    dict(shape="ellipse",  ax=0.46, ay=0.13, n_ap=6, n_lat=1)),
    ("kidney",      dict(shape="ellipse",  ax=0.40, ay=0.30, n_ap=5, n_lat=3)),
    ("limb (digits)", dict(shape="ellipse", ax=0.42, ay=0.22, n_ap=1, n_lat=5)),
]


def domain(cfg):
    ys, xs = np.mgrid[0:N, 0:N]
    x = (xs - N / 2) / N
    y = (ys - N / 2) / N
    if cfg["shape"] == "ellipse":
        m = (x / cfg["ax"]) ** 2 + (y / cfg["ay"]) ** 2 <= 1.0
    else:  # face: a rounded wider-than-tall domain
        m = (x / cfg["ax"]) ** 2 + (y / cfg["ay"]) ** 2 <= 1.0
    return m, x, y


def frame_eigenmodes(mask, k=8):
    """Low eigenmodes of the gap-junction (graph) Laplacian on the masked domain."""
    idx = np.flatnonzero(mask.ravel())
    remap = -np.ones(N * N, int); remap[idx] = np.arange(len(idx))
    rows, cols = [], []
    g = mask.ravel()
    for d in (1, -1, N, -N):
        nb = np.roll(np.arange(N * N), -d)
        # valid within-grid, both endpoints in-mask, and not wrapping rows for +/-1
        ok = g & g[nb]
        if abs(d) == 1:
            same_row = (np.arange(N * N) // N) == (nb // N)
            ok = ok & same_row
        for a, b in zip(np.flatnonzero(ok), nb[np.flatnonzero(ok)]):
            rows.append(remap[a]); cols.append(remap[b])
    n = len(idx)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    vals, vecs = spla.eigsh(L.tocsc(), k=min(k, n - 1), sigma=0, which="LM")
    order = np.argsort(vals)
    vecs = vecs[:, order]
    modes = np.full((k, N * N), np.nan)
    for j in range(vecs.shape[1]):
        modes[j].ravel()[idx] = vecs[:, j]
    return modes.reshape(k, N, N), idx


def pca_axes(mask, x, y):
    pts = np.stack([x[mask], y[mask]], 1)
    c = pts.mean(0); p = pts - c
    _, _, V = np.linalg.svd(p, full_matrices=False)
    ap = p @ V[0]; lat = p @ V[1]
    AP = np.zeros((N, N)); LAT = np.zeros((N, N))
    AP[mask] = (ap - ap.min()) / (np.ptp(ap) + 1e-9)     # 0..1 along long axis
    LAT[mask] = lat / (np.abs(lat).max() + 1e-9)        # -1..1 across, 0 = midline
    return AP, LAT


def oriented_turing(mask, AP, LAT, n_ap, n_lat):
    """Spacing from the wavelength (n_ap x n_lat half-waves), orientation from the frame."""
    f = np.cos(np.pi * n_ap * AP) * np.cos(np.pi * n_lat * LAT)
    f = f * mask
    return f


def lateral_inhibition(field, mask):
    """Delta-Notch winner-take-all -> discrete unit labels + centers."""
    u = np.clip(field, 0, None) * mask
    fp = ndimage.maximum_filter(u, size=7)
    peaks = (u == fp) & (u > 0.35 * (u.max() + 1e-9)) & mask
    # merge near-duplicate peaks, label units by watershed-free nearest-peak on thresholded field
    lbl, nun = ndimage.label(peaks)
    centers = ndimage.center_of_mass(np.ones_like(u), lbl, range(1, nun + 1))
    return peaks, centers


def cohesion(mask, centers, x, y, noise=0.12, seed=0):
    """Type cells by nearest unit; inject noise; differential-adhesion (majority) sort."""
    rng = np.random.default_rng(seed)
    cen = np.array([[x[int(r), int(c)], y[int(r), int(c)]] for r, c in centers])
    xy = np.stack([x, y], -1)
    d = np.linalg.norm(xy[..., None, :] - cen[None, None], axis=-1)
    typ = np.argmin(d, -1) + 1
    typ[~mask] = 0
    # inject noise
    noisy = typ.copy()
    flip = (rng.random((N, N)) < noise) & mask
    noisy[flip] = rng.integers(1, len(cen) + 1, size=flip.sum())
    comp_before = _components(noisy, len(cen), mask)
    # adhesion = iterated majority vote among same/neighbours
    cur = noisy.copy()
    for _ in range(12):
        nxt = cur.copy()
        for t in range(1, len(cen) + 1):
            score = sum(np.roll(np.roll((cur == t).astype(float), dy, 0), dx, 1)
                        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)])
            if t == 1:
                best = score; arg = np.ones((N, N), int)
            else:
                take = score > best; best = np.where(take, score, best)
                arg = np.where(take, t, arg)
        cur = np.where(mask, arg, 0)
    comp_after = _components(cur, len(cen), mask)
    whole_after = int(ndimage.label(cur > 0)[1])
    return noisy, cur, comp_before, comp_after, whole_after


def _components(lab, ntypes, mask):
    tot = 0
    for t in range(1, ntypes + 1):
        tot += ndimage.label(lab == t)[1]
    return tot


def bilateral_symmetry(field, mask):
    """Mirror-symmetry of the feature pattern (magnitude), max over the two mirror axes."""
    a = np.abs(field) * mask
    best = float("nan")
    for axis in (0, 1):
        fm = np.flip(a, axis); m = mask & np.flip(mask, axis)
        if m.sum() < 10:
            continue
        u, v = a[m], fm[m]
        if np.std(u) < 1e-9 or np.std(v) < 1e-9:
            continue
        c = float(np.corrcoef(u, v)[0, 1])
        best = c if (best != best or c > best) else best
    return best


def run_organ(name, cfg):
    mask, x, y = domain(cfg)
    modes, idx = frame_eigenmodes(mask)
    AP, LAT = pca_axes(mask, x, y)
    # validate the frame: mode 1 tracks the AP axis
    m1 = modes[1].ravel()[idx]; apv = AP.ravel()[idx]
    ap_corr = abs(np.corrcoef(m1, apv)[0, 1])
    field = oriented_turing(mask, AP, LAT, cfg["n_ap"], cfg["n_lat"])
    sym = bilateral_symmetry(field, mask)
    peaks, centers = lateral_inhibition(field, mask)
    n_units = len(centers)
    noisy, sorted_, comp_b, comp_a, whole = cohesion(mask, centers, x, y)
    res = dict(organ=name, ap_mode_corr=round(ap_corr, 3), rd_symmetry=round(sym, 3),
               n_units=n_units, components_noisy=comp_b, components_sorted=comp_a,
               whole_body_components=whole)
    return res, dict(mask=mask, modes=modes, AP=AP, LAT=LAT, field=field,
                     peaks=peaks, centers=centers, noisy=noisy, sorted=sorted_)


def main():
    Path("data").mkdir(exist_ok=True)
    results, panels = [], []
    for name, cfg in ORGANS:
        r, p = run_organ(name, cfg)
        results.append(r); panels.append((name, p))
        print(f"{name:10s}  AP-mode corr {r['ap_mode_corr']:.2f}  RD sym {r['rd_symmetry']:.2f}  "
              f"units {r['n_units']:2d}  cohesion comps {r['components_noisy']}->{r['components_sorted']}"
              f"  (target {r['n_units']})  whole-body {r['whole_body_components']}")
    ok = sum(r["components_sorted"] == r["n_units"] for r in results)
    print(f"\ncohesion bound each organ to exactly its unit count: {ok}/{len(results)}")
    json.dump(results, open("data/organ_execution.json", "w"), indent=2)
    print("saved data/organ_execution.json")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nrow = len(panels)
        fig, ax = plt.subplots(nrow, 4, figsize=(12, 3 * nrow))
        titles = ["1. connexin frame (mode 1 = axis)", "2. morphogen RD (spacing)",
                  "3. lateral inhibition (units)", "4. cadherin cohesion (bound)"]
        for i, (name, p) in enumerate(panels):
            m = p["mask"]
            fr = np.where(m, p["modes"][1], np.nan)
            ax[i, 0].imshow(fr, cmap="coolwarm");
            ax[i, 1].imshow(np.where(m, p["field"], np.nan), cmap="viridis")
            ax[i, 2].imshow(np.where(m, p["field"], np.nan), cmap="gray")
            cy = [c[0] for c in p["centers"]]; cx = [c[1] for c in p["centers"]]
            ax[i, 2].scatter(cx, cy, c="red", s=40, edgecolor="w")
            ax[i, 3].imshow(np.where(m, p["sorted"], np.nan), cmap="tab20")
            for j in range(4):
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(titles[j], fontsize=9)
            ax[i, 0].set_ylabel(name, fontsize=11)
        fig.suptitle("Integrated per-organ execution: organ head -> connexin frame -> morphogen RD -> "
                     "lateral inhibition -> cadherin cohesion", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig("data/organ_execution.png", dpi=140); plt.close(fig)
        print("saved data/organ_execution.png")
    except Exception as ex:
        print("figure skipped:", repr(ex))


if __name__ == "__main__":
    main()
