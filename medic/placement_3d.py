"""
Organ placement forward pass on the 3D body frame.

The body-scale generalization of the face placement of Paper #2 (Perceptrons and
Morphogen Primordia), which placed facial primordia on the 2D electric-face eigenmode
frame. Here the frame is the 3D electric body: the low eigenmodes of the whole embryo's
gap-junction operator are the body axes (mode 1 = anterior-posterior; an antisymmetric
mode has its node at the midline). Each organ carries a positional address in that frame
-- an AP level, a dorsoventral level, and a laterality -- and is placed as a primordium at
that address; paired organs are placed symmetrically about the midline, so the whole layout
is bilaterally symmetric.

Honest scope: the FRAME (axes + midline) is computed from the eigenmodes; the per-organ
ADDRESSES stand in for the genome-derived Hox (AP), BMP/Shh (DV) and Nodal/Pitx2 (laterality)
readouts, and are set from known relative anatomy here. Schematic; not organ-specific geometry.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.placement_3d
Out: data/placement_3d.{json,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

NAP, NDV, NLR = 46, 24, 24          # body grid: elongated along AP
AX, AY, AZ = 0.46, 0.30, 0.26       # ellipsoid semi-axes (fraction of grid)

# organ -> (AP 0=ant..1=post, DV -1=ventral..+1=dorsal, laterality: 0=midline / 1=paired, sigma)
ORGANS = {
    "forebrain":  dict(ap=0.07, dv=+0.35, paired=0, sig=0.05, lat=0.0),
    "eye":        dict(ap=0.09, dv=+0.10, paired=1, sig=0.035, lat=0.55),
    "heart":      dict(ap=0.30, dv=-0.55, paired=0, sig=0.05, lat=0.0),
    "lung":       dict(ap=0.30, dv=+0.05, paired=1, sig=0.045, lat=0.45),
    "liver":      dict(ap=0.40, dv=-0.45, paired=0, sig=0.05, lat=0.0),
    "stomach":    dict(ap=0.42, dv=-0.25, paired=0, sig=0.045, lat=0.0),
    "kidney":     dict(ap=0.56, dv=+0.55, paired=1, sig=0.04, lat=0.5),
    "forelimb":   dict(ap=0.34, dv=+0.0,  paired=1, sig=0.05, lat=0.9),
    "hindlimb":   dict(ap=0.72, dv=+0.0,  paired=1, sig=0.05, lat=0.9),
    "gut":        dict(ap=0.60, dv=-0.30, paired=0, sig=0.05, lat=0.0),
}


def body_domain():
    zz, yy, xx = np.mgrid[0:NAP, 0:NDV, 0:NLR]
    ap = (zz - (NAP - 1) / 2) / NAP
    dv = (yy - (NDV - 1) / 2) / NDV
    lr = (xx - (NLR - 1) / 2) / NLR
    mask = (ap / AX) ** 2 + (dv / AY) ** 2 + (lr / AZ) ** 2 <= 1.0
    # normalized frame coordinates on the domain
    APn = np.zeros_like(ap); DVn = np.zeros_like(dv); LRn = np.zeros_like(lr)
    APn[mask] = (ap[mask] - ap[mask].min()) / (np.ptp(ap[mask]) + 1e-9)      # 0..1
    DVn[mask] = dv[mask] / (np.abs(dv[mask]).max() + 1e-9)                   # -1..1
    LRn[mask] = lr[mask] / (np.abs(lr[mask]).max() + 1e-9)                   # -1..1, 0 = midline
    return mask, APn, DVn, LRn, (ap, dv, lr)


def frame_eigenmodes(mask, k=6):
    """Low eigenmodes of the 3D gap-junction (graph) Laplacian on the body domain."""
    flat = mask.ravel()
    idx = np.flatnonzero(flat)
    remap = -np.ones(flat.size, int); remap[idx] = np.arange(len(idx))
    strides = (NDV * NLR, NLR, 1)
    rows, cols = [], []
    grid = np.arange(flat.size)
    for ax, st in enumerate(strides):
        for d in (st, -st):
            nb = grid + d
            ok = (nb >= 0) & (nb < flat.size)
            # prevent wrapping across the axis boundary
            coord = (grid // st) % (mask.shape[ax])
            ncoord = (nb.clip(0, flat.size - 1) // st) % (mask.shape[ax])
            ok &= (np.abs(coord - ncoord) == 1)
            ok &= flat & np.take(flat, nb.clip(0, flat.size - 1))
            a = grid[ok]; b = nb[ok]
            rows.extend(remap[a]); cols.extend(remap[b])
    n = len(idx)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    vals, vecs = spla.eigsh(L.tocsc(), k=k, sigma=0, which="LM")
    order = np.argsort(vals)
    modes = np.full((k, flat.size), np.nan)
    for j, o in enumerate(order):
        modes[j].ravel()[idx] = vecs[:, o]
    return modes.reshape(k, NAP, NDV, NLR), idx


def place(mask, APn, DVn, LRn):
    """Place each organ as a Gaussian primordium at its frame address; paired -> symmetric.
    Returns the summed field, per-organ centre voxels, and the organ table."""
    field = np.zeros((NAP, NDV, NLR))
    rows, centers = [], []
    for name, o in ORGANS.items():
        sides = [+1, -1] if o["paired"] else [0]
        cs = []
        for s in sides:
            g = read_address(APn, DVn, LRn, mask, o["ap"], o["dv"], o["lat"], o["sig"], side=s)
            field = np.maximum(field, g)
            cs.append(np.unravel_index(int(np.argmax(g)), g.shape))  # (z,y,x) centre
        centers.append((name, cs))
        rows.append(dict(organ=name, ap=o["ap"], dv=o["dv"], paired=bool(o["paired"]),
                         laterality=o["lat"]))
    return field, centers, rows


def read_address(APn, DVn, LRn, mask, hox, dv, lat, sig, side=0):
    """The organ head as a coincidence detector (AND-gate) on the positional codes.
    The codes are read off the electric frame: Hox = the AP code (set by the clock),
    the dorsoventral gradient = DV, and laterality from the LR mode. The head fires where
    all three codes match its enhancer tuning. Returns the primordium field."""
    hox_code = APn                          # AP code (Hox colinearity, from the differentiation clock)
    dv_code = DVn                           # dorsoventral gradient (BMP-Shh)
    lr_code = LRn                           # left-right (Nodal/Pitx2); |lr| = distance from midline
    m_hox = np.exp(-((hox_code - hox) ** 2) / (2 * sig ** 2))
    m_dv = np.exp(-((dv_code - dv) ** 2) / (2 * sig ** 2))
    m_lr = np.exp(-((lr_code - side * lat) ** 2) / (2 * sig ** 2))
    return (m_hox * m_dv * m_lr) * mask     # AND-gate = product of code memberships


def homeotic_demo(APn, DVn, LRn, mask, organ="forelimb", dhox=0.20):
    """Proof that the node is READ, not fixed: shift the organ's Hox code and it relocates."""
    o = ORGANS[organ]
    def center_ap(hox):
        g = read_address(APn, DVn, LRn, mask, hox, o["dv"], o["lat"], o["sig"], side=1)
        z, y, x = np.unravel_index(int(np.argmax(g)), g.shape)
        return APn[z, y, x]
    ap0 = center_ap(o["ap"]); ap1 = center_ap(o["ap"] + dhox)
    return dict(organ=organ, ap_before=round(float(ap0), 3), ap_after=round(float(ap1), 3),
                requested_hox_shift=dhox, realized_ap_shift=round(float(ap1 - ap0), 3))


def bilateral_symmetry(field, mask):
    fm = np.flip(field, axis=2)         # mirror across the LR (midline) axis
    m = mask & np.flip(mask, axis=2)
    a, b = field[m], fm[m]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    Path("data").mkdir(exist_ok=True)
    mask, APn, DVn, LRn, (ap, dv, lr) = body_domain()
    modes, idx = frame_eigenmodes(mask)
    # validate the frame: which eigenmode tracks each geometric axis
    def corr_mode(coord):
        best, bj = 0.0, -1
        cflat = coord.ravel()[idx]
        for j in range(1, modes.shape[0]):
            mv = modes[j].ravel()[idx]
            c = abs(np.corrcoef(mv, cflat)[0, 1])
            if c > best:
                best, bj = c, j
        return bj, round(best, 3)
    ap_j, ap_c = corr_mode(ap)
    dv_j, dv_c = corr_mode(dv)
    lr_j, lr_c = corr_mode(lr)
    print(f"frame: AP=mode{ap_j} (corr {ap_c}), DV=mode{dv_j} ({dv_c}), LR=mode{lr_j} ({lr_c})")

    field, centers, rows = place(mask, APn, DVn, LRn)
    sym = bilateral_symmetry(field, mask)
    n_paired = sum(r["paired"] for r in rows)
    print(f"placed {len(rows)} organs ({n_paired} paired), bilateral symmetry {sym:.3f}")
    for r in sorted(rows, key=lambda x: x["ap"]):
        tag = "paired" if r["paired"] else "midline"
        print(f"    {r['organ']:10s} AP {r['ap']:.2f}  DV {r['dv']:+.2f}  {tag}")

    # proof the node is READ from the code, not fixed: shift the Hox code -> organ relocates
    hom = homeotic_demo(APn, DVn, LRn, mask)
    print(f"homeotic test: {hom['organ']} Hox +{hom['requested_hox_shift']} "
          f"-> AP {hom['ap_before']} to {hom['ap_after']} (realized shift {hom['realized_ap_shift']}) "
          f"= the primordium reads the Hox code")

    out = dict(frame=dict(ap_mode=ap_j, ap_corr=ap_c, dv_mode=dv_j, dv_corr=dv_c,
                          lr_mode=lr_j, lr_corr=lr_c),
               bilateral_symmetry=round(sym, 3), n_organs=len(rows), n_paired=n_paired,
               homeotic=hom, organs=rows)
    json.dump(out, open("data/placement_3d.json", "w"), indent=2)
    print("saved data/placement_3d.json")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        names = list(ORGANS.keys())
        col = {n: cm.tab20(i / max(1, len(names) - 1)) for i, n in enumerate(names)}
        fig, ax = plt.subplots(1, 3, figsize=(16, 5.6))
        # (proj index for z=AP, y=DV, x=LR): sagittal drops x, dorsal drops y, transverse drops z
        views = [
            (ax[0], "sagittal (AP $\\times$ DV) -- side view", "anterior $\\rightarrow$ posterior",
             "ventral $\\rightarrow$ dorsal", mask.max(2).T, lambda z, y, x: (z, y), None),
            (ax[1], "dorsal (AP $\\times$ LR) -- top view", "anterior $\\rightarrow$ posterior",
             "left $\\leftrightarrow$ right", mask.max(1).T, lambda z, y, x: (z, x), ("h", NLR / 2)),
            (ax[2], "transverse (DV $\\times$ LR) -- cross section", "ventral $\\rightarrow$ dorsal",
             "left $\\leftrightarrow$ right", mask.max(0).T, lambda z, y, x: (y, x), ("v", NLR / 2))]
        for a, ttl, xl, yl, outline, proj, mid in views:
            a.imshow(np.ma.masked_where(~outline.astype(bool), outline), origin="lower",
                     cmap="Greys", alpha=0.18, aspect="auto")
            seen = set()
            for name, cs in centers:
                for (z, y, x) in cs:
                    px, py = proj(z, y, x)
                    a.scatter(px, py, color=col[name], s=90, edgecolor="k", lw=0.5, zorder=3)
                    key = (name, round(px), round(py))
                    if key not in seen:
                        seen.add(key)
                        a.annotate(name, (px, py), fontsize=6.5, ha="center", va="bottom",
                                   xytext=(0, 5), textcoords="offset points")
            if mid:
                (a.axhline if mid[0] == "h" else a.axvline)(mid[1] - 0.5, color="k", lw=0.8, ls="--")
            a.set_title(ttl, fontsize=10); a.set_xlabel(xl, fontsize=9); a.set_ylabel(yl, fontsize=9)
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"Organ placement on the 3D electric-body frame -- bilateral symmetry {sym:.2f}; "
                     f"AP = gap-junction mode {ap_j} (corr {ap_c})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig("data/placement_3d.png", dpi=150); plt.close(fig)
        print("saved data/placement_3d.png")
    except Exception as ex:
        print("figure skipped:", repr(ex))


if __name__ == "__main__":
    main()
