"""abdominal_packing_head.py -- wire the abdominal soft-body PACKING MECHANISM into the build. (2026-08-10)

The MECHANISM (medic.abdominal_packing_3d) that EARNS the viscera's dorso-ventral depth instead of
dv_spread_head PLACING each at its measured BodyParts3D depth. Miles's insight: the abdominal organs "fit
together because they are fluid" -- deformable bags packed into a closed cavity, so DV is a packing
equilibrium set by the PERITONEAL RELATIONSHIP (a 3-class membrane fact), not 8 independent measured depths.

This head runs AFTER dv_spread (which keeps the THORACIC organs -- heart, lung -- on their measured depths)
and RE-RELAXES only the ABDOMINAL viscera: it pulls each organ's local-trunk DV toward its peritoneal-class
target (intraperitoneal 0.32 / secondarily-retroperitoneal 0.47 / retroperitoneal 0.66) under organ-organ
NON-PENETRATION (contact) and body-wall containment. It works in the SAME local-trunk-DV envelope as
dv_spread (per-AP-band 5th-95th percentile of the axial body), moving each organ as a coherent rigid DV body
so its shape, its AP (Hox) level, and its ML (laterality) are all kept -- only DV changes. So it cannot break
heart-in-thorax / rib articulation any more than dv_spread can.

TRADEOFF (honest): the class mechanism gives ~0.67 DV rank-correlation vs the measured depths dv_spread hits
near-exactly (~0.95). This is the physics-boundary choice made explicit -- a glass-box mechanism vs a fit.
Wired REVERSIBLY (git checkout adult_persistence_audit.py). Self-test prints both + Gray's integrity.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.abdominal_packing_head
"""
from __future__ import annotations
import json
import numpy as np

from medic.unified_embryo import FIDX
from medic.dv_spread_head import _dorsal_sign, _limb_mask

TARGETS_PATH = "data/organ_cascade/bp3d_insitu_dv.json"

# the 3 peritoneal-relationship classes -> a local-trunk DV target (0 ventral .. 1 dorsal). 3 class values.
DV_CLASS = {"intra": 0.32, "secondary": 0.47, "retro": 0.66}
# abdominal organ -> (member fates, peritoneal relationship). Thoracic organs (Heart/Lung) are NOT here --
# dv_spread keeps them on their measured depth.
ABD_ORGANS = {
    "Liver":          (("Liver", "LiverHaem"), "intra"),
    "Stomach":        (("Foregut", "Stomach"), "intra"),
    "Pancreas":       (("Pancreas",),           "intra"),
    "SmallIntestine": (("Gut",),                "intra"),
    "Bladder":        (("Bladder",),            "intra"),
    "LargeIntestine": (("Hindgut",),            "secondary"),
    "Spleen":         (("Spleen",),             "retro"),
    "Kidney":         (("Kidney", "Nephron"),   "retro"),
}


def _envelope(base, F, s, nb=24):
    """Per-AP-band local-trunk DV envelope (olo, ohi) from the axial body -- identical to dv_spread's."""
    x = base[:, 0]; oriented = base[:, 1] * s
    axial = ~_limb_mask(F)
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    band = np.clip((apf * nb).astype(int), 0, nb - 1)
    olo = np.array([np.percentile(oriented[axial & (band == k)], 5)
                    if (axial & (band == k)).sum() > 20 else oriented.min() for k in range(nb)])
    ohi = np.array([np.percentile(oriented[axial & (band == k)], 95)
                    if (axial & (band == k)).sum() > 20 else oriented.max() for k in range(nb)])
    return apf, band, olo, ohi, oriented


def pack(base, F, iters=200, k_pull=0.12, contact=False, verbose=False):   # contact=False: the DV-stacking gate
    # over-fires on the real clouds (kidney pushed to 0.89) -> WIP; class-pull-only is the better mechanism for now.
    """Relax the abdominal viscera's DV toward their peritoneal-class targets under contact + containment.
    Read-only on non-abdominal cells; each organ moved as a coherent rigid DV body."""
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    s = _dorsal_sign(base, F)
    nb = 24
    apf, band, olo, ohi, oriented = _envelope(base, F, s, nb)
    z = base[:, 2]

    org = {}                                                       # per-organ scalars
    for name, (fates, rel) in ABD_ORGANS.items():
        fids = [FIDX[f] for f in fates if f in FIDX]
        m = np.isin(F, fids)
        if m.sum() < 8:
            continue
        b = int(np.clip(apf[m].mean() * nb, 0, nb - 1))
        span = (ohi[b] - olo[b]) + 1e-9
        ldv = (oriented[m] - olo[b]) / span                        # local DV of every cell
        org[name] = dict(m=m, b=b, span=float(span), rel=rel,
                         cur=float(ldv.mean()), ldv0=float(ldv.mean()),
                         half=float(max(0.04, ldv.std() * 1.3)),   # DV half-extent (for contact)
                         ap=float(apf[m].mean()), aph=float(max(0.03, apf[m].std() * 1.3)),
                         ml=float(z[m].mean()), mlh=float(max(0.03, z[m].std() * 1.3)))
    names = list(org)

    for _ in range(iters):
        for n in names:                                            # 1. peritoneal-class pull (the mechanism)
            o = org[n]; o["cur"] += k_pull * (DV_CLASS[o["rel"]] - o["cur"])
        for i in range(len(names)) if contact else []:            # 2. non-penetration for organs that STACK
            for j in range(i + 1, len(names)):
                a, b = org[names[i]], org[names[j]]
                if abs(a["ap"] - b["ap"]) < a["aph"] + b["aph"] and abs(a["ml"] - b["ml"]) < a["mlh"] + b["mlh"]:
                    gap = (a["half"] + b["half"]) - abs(a["cur"] - b["cur"])
                    if gap > 0:
                        d = 0.5 * gap * (1.0 if a["cur"] >= b["cur"] else -1.0)
                        a["cur"] += d; b["cur"] -= d
        for n in names:                                            # 3. body-wall containment
            org[n]["cur"] = float(np.clip(org[n]["cur"], 0.05, 0.95))

    for n in names:                                                # apply the rigid DV shift per organ
        o = org[n]
        d_oriented = (o["cur"] - o["ldv0"]) * o["span"]
        base[o["m"], 1] += d_oriented * s
        if verbose:
            print(f"  {n:15s} local DV {o['ldv0']:.2f} -> {o['cur']:.2f}  ({o['rel']})")
    return base


def _score_vs_bp3d(base, F):
    """Local-trunk DV of each abdominal organ vs the measured bp3d depth (rank correlation)."""
    tgt = {k: v["dv_local"] for k, v in json.load(open(TARGETS_PATH))["organs"].items()}
    s = _dorsal_sign(base, F)
    apf, band, olo, ohi, oriented = _envelope(base, F, s)
    emg, mea = [], []
    for name, (fates, rel) in ABD_ORGANS.items():
        fids = [FIDX[f] for f in fates if f in FIDX]
        m = np.isin(F, fids)
        if m.sum() < 8 or name not in tgt:
            continue
        b = int(np.clip(apf[m].mean() * 24, 0, 23))
        emg.append(float((oriented[m].mean() - olo[b]) / (ohi[b] - olo[b] + 1e-9))); mea.append(tgt[name])
    emg, mea = np.array(emg), np.array(mea)
    ra, rb = np.argsort(np.argsort(emg)), np.argsort(np.argsort(mea))
    return float(np.corrcoef(ra, rb)[0, 1]), emg, mea


def main():
    from medic.adult_persistence_audit import build_base
    from medic.dv_spread_head import spread as _dv_spread
    from medic import gray_integrity_audit as GIA

    # build WITHOUT the final dv_spread by rebuilding + placing ourselves would require refactor; instead compare
    # dv_spread-only (the current default) against dv_spread + abdominal packing (the mechanism).
    base, F = build_base(30000)                                    # already includes dv_spread (the default)
    sp0, e0, mea = _score_vs_bp3d(base, F)
    packed = pack(base, F, verbose=True)
    sp1, e1, _ = _score_vs_bp3d(packed, F)
    sp2, e2, _ = _score_vs_bp3d(pack(base, F, contact=False), F)   # class-pull only (contact off)

    print(f"\n[DV rank-corr vs measured bp3d]  dv_spread(default) {sp0:.3f}  ->  +packing {sp1:.3f}"
          f"  (class-pull only, no contact {sp2:.3f})")
    print(f"{'organ':16s} {'measured':>8s} {'dvspread':>8s} {'packed':>8s}")
    for name, mv, a, b in zip([n for n in ABD_ORGANS], mea, e0, e1):
        print(f"{name:16s} {mv:8.2f} {a:8.2f} {b:8.2f}")
    for tag, cloud in (("dv_spread", base), ("packed", packed)):
        rows = GIA.audit(cloud, F); npass = sum(1 for r in rows if r.get("ok"))
        fails = [f"{r['relation']}" for r in rows if not r.get("ok")]
        print(f"[integrity {tag:9s}] {npass}/{len(rows)}" + (f"  FAIL: {fails}" if fails else ""))


if __name__ == "__main__":
    main()
