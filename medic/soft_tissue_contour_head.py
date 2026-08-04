"""
soft_tissue_contour_head.py -- sculpt the limb soft tissue into fusiform bellies at anthropometric girth.

The audit found the thigh 2.7x too thick: after the migration head put muscle mass in the limbs, that mass
is an unsculpted blob filling the whole limb-bud volume. Real limbs are not: each muscle belly is FUSIFORM
(Myf5/MyoD myotubes bundle and taper to tendons at the joints), and total limb girth is genetically scaled
to body size (muscle CSA ~ body_size). So the contour is a genome-anchored reshape, not an arbitrary trim.

A head, by the composition rule: SPEC = the anthropometric girth (muscle-mass ~ body_size) + the joint
positions (tendon tapers); READS the limb soft-tissue cells + their proximodistal axis, read-only on
identity; WRITES one field = each cell's RADIAL position about the limb axis, on a fusiform profile scaled
so mid-belly diameter ~ target x height. It never changes which muscle a cell belongs to, only the girth.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.soft_tissue_contour_head
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX

SOFT = ("Muscle", "Limb Bud", "Cartilage", "Connective")


def _limb_masks(base, F):
    """The 4 appendicular soft-tissue masks (fore/hind x L/R): soft cells that lie OUT beyond the trunk
    half-width at their AP level (the arm/leg), so the trunk is left alone."""
    soft = np.isin(F, [FIDX[n] for n in SOFT if n in FIDX])
    x, z = base[:, 0], base[:, 2]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    # trunk half-width per AP bin (non-limb soft = axial/trunk reference is hard; use a robust global body width)
    body = ~np.isin(F, [FIDX["Limb Bud"]]) if "Limb Bud" in FIDX else np.ones(len(F), bool)
    hw = np.percentile(np.abs(z[body]), 70) + 1e-9
    out = {}
    for side, zs in (("L", z >= 0), ("R", z < 0)):
        out[f"fore{side}"] = soft & (apf >= 0.55) & zs & (np.abs(z) > 0.9 * hw)  # lateral ARM (outboard of trunk)
        out[f"hind{side}"] = soft & (apf < 0.32) & zs                            # whole LEG below the hip (full CSA)
    return out


def contour(base, F, target=0.11, rng=None):
    """Reshape each limb's soft tissue to a fusiform belly whose mid diameter ~ target x height. Returns a
    new positions array (soft-tissue radii rescaled about each limb's PD axis; everything else unchanged)."""
    P = base.astype(float).copy()
    H = float(np.ptp(P[:, 0]))
    for name, m in _limb_masks(base, F).items():
        if m.sum() < 12:
            continue
        idx = np.where(m)[0]
        Q = P[idx]; c = Q.mean(0)
        u = np.linalg.svd(Q - c, full_matrices=False)[2][0]       # proximodistal axis of this limb
        s = (Q - c) @ u                                            # position along the limb
        t = (s - s.min()) / (np.ptp(s) + 1e-9)                    # 0 = proximal joint, 1 = distal joint
        rad = (Q - c) - np.outer(s, u)                            # radial (perpendicular-to-axis) vector
        r = np.linalg.norm(rad, axis=1)
        tgt = 0.5 * target * H                                    # anthropometric target radius
        # FUSIFORM, per axial STATION: bin along the limb and rescale each cross-section's girth to the
        # target profile (full through the belly, tapering to the tendons at both joints). Per-station so the
        # mid actually reaches the target (a single whole-limb reference under-shrinks the wide belly).
        nb = 10; scale = np.ones(len(Q))
        for b in range(nb):
            sel = (np.clip((t * nb).astype(int), 0, nb - 1) == b)
            if sel.sum() < 4:
                continue
            tc = (b + 0.5) / nb
            prof = 0.35 + 0.65 * np.sin(np.pi * tc) ** 0.6
            cur_b = np.percentile(r[sel], 85) + 1e-9
            scale[sel] = (tgt * prof) / cur_b
        P[idx] = c + np.outer(s, u) + rad * scale[:, None]
    return P


def _thigh_diam(base, F):
    """Quick anthropometric read: mid-hindlimb soft-tissue diameter / height (matches the audit's metric)."""
    H = float(np.ptp(base[:, 0]))
    x, z = base[:, 0], base[:, 2]
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    soft = np.isin(F, [FIDX[n] for n in SOFT if n in FIDX])
    body = ~np.isin(F, [FIDX["Limb Bud"]]) if "Limb Bud" in FIDX else np.ones(len(F), bool)
    hw = np.percentile(np.abs(z[body]), 70) + 1e-9
    leg = soft & (apf < 0.35) & (np.abs(z) > 0.9 * hw)
    if leg.sum() < 8:
        return 0.0
    Q = base[leg]; c = Q.mean(0)
    u = np.linalg.svd(Q - c, full_matrices=False)[2][0]
    s = (Q - c) @ u; t = (s - s.min()) / (np.ptp(s) + 1e-9)
    mid = np.abs(t - 0.5) < 0.2
    rad = (Q[mid] - c) - np.outer(s[mid], u)
    return float(2 * np.percentile(np.linalg.norm(rad, axis=1), 85) / H) if mid.sum() >= 6 else 0.0


def main():
    from medic.adult_persistence_audit import build_base
    from medic.limb_myoblast_migration import augment
    base, F = build_base(30000)
    base, F, _ = augment(base, F)                                 # limbs must be populated first
    d0 = _thigh_diam(base, F)
    P = contour(base, F, target=0.11)
    d1 = _thigh_diam(P, F)
    print("soft-tissue contour head (fusiform bellies at anthropometric girth):")
    print(f"  thigh diameter / height:  before {d0:.2f}  ->  after {d1:.2f}   (canon ~0.11)")
    print("  genome-derived: myotube fusiform taper (Myf5/MyoD) + muscle mass ~ body_size, tendon tapers at joints")


if __name__ == "__main__":
    main()
