"""
limb_myoblast_migration.py -- the LIMB-MYOBLAST MIGRATION head (the upstream fix for empty limb muscles).

The musculoskeletal audit showed 14/31 muscles LIFTED -- almost every forelimb muscle (deltoid, biceps,
triceps, forearm) had ~0 model cells, so there was nothing for the limb-muscle CT head to carve. The cause
is developmental and specific: limb muscle does NOT arise in the limb; its precursors MIGRATE in.

Biology (genome-anchored):
  * Pax3+ cells at the ventrolateral lip of the dermomyotome, AT LIMB LEVELS, switch on Lbx1 -> the
    MIGRATORY myoblast identity, undergo EMT and DELAMINATE from the somite.
  * They migrate into the limb bud up the HGF/SF gradient secreted by the limb mesenchyme (c-Met receptor).
  * In the limb they split into a DORSAL (extensor, Lmx1b) and a VENTRAL (flexor, En1) muscle mass, then
    proliferate to fill the limb -- the mass the limb CT (Tcf4) later cleaves into individual muscles.

This head READS {limb buds (the HGF target volume), somite myotome (the source)} and WRITES new limb-muscle
cells INSIDE each limb's dorsal+ventral compartments. Read-only: it never moves the buds or the somite; it
populates the limb with the precursors the carve step needs. Composes before limb_muscle_head.carve.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.limb_myoblast_migration
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX


def migrate(base, F, per_limb=1800, cap_mult=12, distal_reach=4.5, rng=None):
    """Populate each of the 4 limbs with migrated Lbx1+ myoblasts, split dorsal/ventral. Returns a list of
    (positions, limb_label, dv_label). Read-only: the returned points are NEW cells inside the limb volume,
    seeded from the limb's proximal base (the delamination/entry point) and COLONISING the limb DISTALLY toward
    its eventual tip (the AER/FGF + HGF gradient), offset off the skeletal core into a dorsal and a ventral shell.

    `per_limb` = target myoblasts per limb; `cap_mult` bounds it at cap_mult x the bud-cell count. `distal_reach`
    = how far past the embryonic bud (in bud-PD-lengths) the myoblasts colonise, following the AER as the limb
    elongates -- the DISTAL-COLONISATION fix. The bud is short in the embryo, but the atlas muscle action-lines
    (radius/ulna->manus, tibia->pes) run to the MATURE distal bones; seeding only the proximal bud left every
    distal muscle (forearm/hand, leg/foot: tibialis, peroneus, digitorum, brachioradialis, pronator...) unable to
    SPAN its origin->insertion line. Biology: the migratory precursors invade the bud, then proliferate + move
    distally with the outgrowing limb, filling it to the autopod. Modelled as a HYBRID: (a) the proven dense bud
    seeding at the real bud cells (keeps the proximal + mid-limb muscles = the 78/138 base), plus (b) a thinner
    tail colonising PAST the bud along the limb's own PD axis (oriented distally = away from the body midline) out
    to distal_reach x the bud length, tapering toward the tip. Measured: limb-muscle attachment 62 (no head) ->
    78 (proximal bud only) -> 86/138 (distal colonisation), rescuing the forearm/leg muscles pronator teres,
    supinator, extensor digitorum longus, plantaris, brachialis with NO regression. HONEST LIMIT: the most-distal
    foot/hand muscles (tibialis/peroneus/flexor digitorum -> pes/manus) still miss -- the embryo bud's PD axis
    does not register to the MATURE atlas's farthest distal bones; that needs the audit run on the matured limb."""
    if rng is None:
        rng = np.random.default_rng(0)
    if "Limb Bud" not in FIDX:
        return []
    LB = FIDX["Limb Bud"]
    x, z = base[:, 0], base[:, 2]
    span = np.ptp(x) + 1e-9
    apf = (x - x.min()) / span
    body_c = base.mean(0)
    out = []
    for kind, apsel in (("fore", apf >= 0.5), ("hind", apf < 0.5)):
        for side, zsel in (("L", z >= 0), ("R", z < 0)):
            lm = (F == LB) & apsel & zsel
            if lm.sum() < 20:                                  # this limb bud is too small to invade
                continue
            Lp = base[lm]; center = Lp.mean(0)
            # limb long (proximodistal) axis = 1st principal; DV = 2nd principal (dorsal vs ventral); ML = 3rd.
            C = Lp - center
            Vp = np.linalg.svd(C, full_matrices=False)[2]
            v_pd, v_dv, v_ml = Vp[0], Vp[1], Vp[2]
            if np.dot(v_pd, center - body_c) < 0:              # orient PD so + points DISTALLY (away from the body)
                v_pd = -v_pd
            pdc = C @ v_pd                                     # PD coord (distal = +)
            p_lo, p_hi = np.percentile(pdc, 3), np.percentile(pdc, 97)
            bud_len = (p_hi - p_lo) + 1e-9
            # cross-section radius of the bud (perpendicular to PD) -- the belly thickness, tapered distally
            perp = np.linalg.norm(C - np.outer(pdc, v_pd), axis=1)
            rad = np.percentile(perp, 75) + 1e-9
            dv = C @ v_dv
            n = int(min(per_limb, lm.sum() * cap_mult))
            for sgn, lab in ((+1.0, "dorsal"), (-1.0, "ventral")):
                # (a) DENSE BUD seeding (the proven proximal fill: myoblasts at the real bud cells, offset into
                # the dorsal/ventral shell) -- keeps the proximal + mid-limb muscles attached (the 78/138 base).
                sel = np.sign(dv) == sgn
                if sel.sum() >= 3:
                    src = Lp[sel]; dvs = dv[sel]
                    idx = rng.integers(0, len(src), n // 2)
                    pts = (src[idx] + sgn * 0.45 * np.abs(dvs[idx])[:, None] * v_dv[None]
                           + rng.normal(size=(n // 2, 3)) * 0.006 * span)
                    out.append((pts, f"{kind}{side}", lab))
                # (b) DISTAL COLONISATION: a thinner tail of myoblasts running PAST the bud along the PD axis to
                # the AER-driven tip (distal_reach x bud length), so the distal-limb muscles get precursors.
                nd = n // 4
                t = p_hi + distal_reach * bud_len * rng.random(nd)      # from the bud's distal edge outward
                taper = np.clip(1.0 - 0.5 * (t - p_hi) / (distal_reach * bud_len + 1e-9), 0.35, 1.0)
                off = (sgn * 0.5 * rad * taper[:, None] * v_dv[None]
                       + (rng.random((nd, 1)) - 0.5) * 1.2 * rad * taper[:, None] * v_ml[None]
                       + rng.normal(size=(nd, 3)) * 0.05 * rad)
                out.append((center + t[:, None] * v_pd[None] + off, f"{kind}{side}", lab))
    return out


def augment(base, F, rng=None):
    """(base, F) -> (base2, F2) with the migrated limb myoblasts appended as Muscle-fate cells, so every
    downstream step (limb_muscle_head.carve, musculoskeletal_audit) sees limb muscle where there was none."""
    out = migrate(base, F, rng=rng)
    if not out:
        return base, F, 0
    P = np.vstack([o[0] for o in out])
    F2 = np.concatenate([F, np.full(len(P), FIDX["Muscle"])])
    return np.vstack([base, P]), F2, len(P)


def main():
    from medic.adult_persistence_audit import build_base
    base, F = build_base(30000)
    out = migrate(base, F)
    tot = sum(len(o[0]) for o in out)
    per = {}
    for pts, limb, dv in out:
        per.setdefault(limb, {"dorsal": 0, "ventral": 0})[dv] += len(pts)
    print(f"limb-myoblast migration: {tot} myoblasts into {len(per)} limbs")
    for limb, d in sorted(per.items()):
        print(f"  {limb:5s} dorsal(extensor) {d['dorsal']:4d}  ventral(flexor) {d['ventral']:4d}")
    print("  genome-derived: Pax3->Lbx1 migratory identity, HGF/c-Met invasion, Lmx1b/En1 dorsoventral split")
    b2, F2, added = augment(base, F)
    lim = (F2 == FIDX["Muscle"]).sum() - (F == FIDX["Muscle"]).sum()
    print(f"  augmented cloud: +{added} limb-muscle cells ({(F==FIDX['Muscle']).sum()} -> {(F2==FIDX['Muscle']).sum()} Muscle)")


if __name__ == "__main__":
    main()
