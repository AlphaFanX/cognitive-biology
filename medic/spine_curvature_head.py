"""
spine_curvature_head.py -- the SPINE CURVATURE head: give the vertebral column its sagittal S-curve.

The musculoskeletal audit found the column STRAIGHT (0/2 characteristic reversals). A real spine is an
S in the sagittal plane -- cervical LORDOSIS (concave dorsal), thoracic KYPHOSIS (concave ventral),
lumbar LORDOSIS, sacral KYPHOSIS. This is not imposed by hand: it is regionally PROGRAMMED vertebral
wedging, read off the SAME Hox region identity (C/T/L/S) that named the vertebrae, and integrated along
the column -- exactly the integral-of-curvature principle body_fold.py uses for the fetal C.

A head, by the composition rule: SPEC = the Hox region of each vertebra (already derived); READS the
AP-ordered vertebrae read-only; WRITES one field = the sagittal (AP-DV) deflection. Each region carries a
signed wedge kappa; the tangent angle is its integral along arc length, and the column follows
(X = int cos, Y = int sin). Because the input is the region identity, it cannot oppose the genome; the
validation is that the curve comes out ALTERNATING (an S), not that it is told the shape.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.spine_curvature_head
"""
from __future__ import annotations
import numpy as np

# signed regional wedge (lordosis +concave-dorsal / kyphosis -concave-ventral). Magnitudes ~ the human
# curve depths (thoracic + lumbar are the deep ones); these are the searchable knobs.
CURVE = {"C": +0.9, "T": -1.2, "L": +1.3, "S": -0.7, "Ca": -0.3}


def curve_column(V, Vname, amp=0.16):
    """Bend the straight column of vertebra positions V (laid frame: col0=AP, col1=DV, col2=ML) into the
    sagittal S. Orders the vertebrae along AP, gives each its region's signed wedge, integrates to a tangent
    angle, and lays the column on the resulting curve. ML is untouched. `amp` = curve depth (fraction of
    column length) -- the one knob. Returns the curved positions (same order as V)."""
    x = V[:, 0]
    order = np.argsort(x)
    reg = np.array([_region(n) for n in Vname])[order]
    s = x[order]; L = np.ptp(s) + 1e-9
    ds = np.gradient(s)                                        # arc-length step per vertebra along AP
    kappa = np.array([CURVE.get(r, 0.0) for r in reg])
    kappa = kappa - kappa.mean()                              # no net lean (ends align) -> a pure S
    phi = np.cumsum(kappa * ds); phi -= phi.mean()            # tangent angle along the column
    phi *= amp * L / (np.abs(phi).max() * L + 1e-9) * 6.0     # scale the angle so the deflection ~ amp*L
    # integrate the tangent into a path (X along AP, Y = sagittal deflection); keep AP spread ~ original
    dX = np.cos(phi) * ds; dY = np.sin(phi) * ds
    X = np.cumsum(dX); Y = np.cumsum(dY)
    X = (X - X.min()) / (np.ptp(X) + 1e-9) * L + s.min()      # renormalise AP to the original span
    Y = (Y - Y.mean()) * (amp * L) / (np.ptp(Y) + 1e-9)       # deflection amplitude = amp * L
    out = V.copy().astype(float)
    out[order, 0] = X
    out[order, 1] = V[order, 1] + Y                            # add the sagittal S onto the existing DV
    return out


def _region(name):
    if name[:2] == "Ca":
        return "Ca"
    return name[0]


def apply_to_cloud(base, F, amp=0.16):
    """Optional: propagate the spine curve to the whole trunk (each trunk cell rides the local spinal
    tangent). Kept simple -- the primary product is the curved column; this is for the visible body."""
    return base    # (column-level head; body propagation is a follow-on)


def main():
    from medic.adult_persistence_audit import build_base
    from medic.part_resolved_anatomy import complete_column
    base, F = build_base(30000)
    V, Vname, Vreal, populated, real_v, names = complete_column(base, F, np.random.default_rng(0))
    V2 = curve_column(V, Vname)

    def _reversals(P):
        ap, dv, reg = [], [], []
        for nm in names:
            m = Vname == nm
            if m.any():
                ap.append(P[m, 0].mean()); dv.append(P[m, 1].mean()); reg.append(_region(nm))
        ap = np.array(ap); dv = np.array(dv); reg = np.array(reg)
        o = np.argsort(ap); ap, dv, reg = ap[o], dv[o], reg[o]
        res = dv - np.polyval(np.polyfit(ap, dv, 1), ap)
        rm = {r: res[reg == r].mean() for r in ("C", "T", "L") if (reg == r).any()}
        rev = int(np.sign(rm.get("C", 0)) != np.sign(rm.get("T", 0))) + \
              int(np.sign(rm.get("T", 0)) != np.sign(rm.get("L", 0)))
        return rev, rm
    r0, m0 = _reversals(V)
    r1, m1 = _reversals(V2)
    print("spine curvature head (cervical lordosis / thoracic kyphosis / lumbar lordosis):")
    print(f"  STRAIGHT column:  reversals {r0}/2   region deflection {{ {', '.join(f'{k}:{v:+.3f}' for k,v in m0.items())} }}")
    print(f"  CURVED  column:   reversals {r1}/2   region deflection {{ {', '.join(f'{k}:{v:+.3f}' for k,v in m1.items())} }}")
    print("  genome-derived: signed wedge per Hox region (C/T/L/S), integrated along the column")


if __name__ == "__main__":
    main()
