"""
rib_cage_head.py -- the RIB-CAGE head: 12 pairs of ribs + the sternum, per Gray's.

Reads the thoracic vertebrae (the twelve T-levels the column names) + the ventral midline, read-only, and
writes a rib arcing from each thoracic vertebra around the thorax to the sternum. It moves no existing cell.

Biology / Gray's (all from fields already produced):
  * COUNT = one rib pair per thoracic vertebra -> 12 pairs (the column now carries T1..T12).
  * TRUE ribs 1-7 reach the sternum by their costal cartilage; FALSE ribs 8-10 join the rib above and do not
    reach the midline; FLOATING ribs 11-12 end in the lateral wall.
  * SHAPE = each rib sweeps from its vertebra (dorsal, at that AP level) laterally to the mid-axillary line and
    then ventrally, sloping caudally as it comes forward (Hox-regionalised sclerotome + the costal field).
  * STERNUM = a ventral-midline bar (manubrium + body + xiphoid) over the upper thoracic levels, to which the
    true ribs converge.
So the head reads {thoracic vertebrae, midline, notochord = dorsal ref, H} and writes the cage.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.rib_cage_head
Out: data/organ_cascade/rib_cage_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.tuned_knobs import tuned


def _dorsal_sign(base, F):
    if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8:
        return 1.0 if np.median(base[F == FIDX["Notochord"]][:, 1]) >= np.median(base[:, 1]) else -1.0
    return 1.0


def build(base, F, span=None, per=26):
    """The rib-cage head. `span` = chest HALF-width as a fraction of body length H (spec anchor; None -> tuned,
    default 0.14). Returns per-part dicts: 24 ribs (rib1..rib12 x L/R) + the sternum, all bone."""
    if span is None:
        span = tuned("rib_cage", {"span": 0.14})["span"]
    from medic.part_resolved_anatomy import complete_column
    H = float(np.ptp(base[:, 0]))
    dsgn = _dorsal_sign(base, F)
    rng = np.random.default_rng(0)      # SAME seed as integrated_body + the audits, so the ribs attach to THE
    V, Vname, _, _, _, _ = complete_column(base, F, rng)   # canonical column (lifted vertebrae agree, not a 2nd copy)
    thor = [(nm, V[Vname == nm].mean(0)) for nm in set(Vname.tolist()) if nm.startswith("T")]
    thor.sort(key=lambda t: int(t[0][1:]))                       # T1..T12 in order
    if not thor:
        return dict(parts={}, H=H, span=span)
    depth = 1.35 * span * H                                      # chest DV depth (dorsal vertebra -> ventral sternum)
    parts = {}
    vent_pts = []                                               # ventral ends of the true ribs -> the sternum line
    for i, (nm, vc) in enumerate(thor):
        n = i + 1
        th_end = np.pi if n <= 7 else (0.72 * np.pi if n <= 10 else 0.5 * np.pi)   # true / false / floating
        x_i, dv0 = float(vc[0]), float(vc[1])                    # this vertebra's AP and (dorsal) DV
        for sgn, sd in ((1.0, "R"), (-1.0, "L")):
            th = np.linspace(0.06, th_end, per)
            z = sgn * span * H * np.sin(th)                      # 0 at spine -> lateral -> back toward midline
            dv = dv0 - dsgn * depth * (1.0 - np.cos(th)) / 2.0   # dorsal (vertebra) -> ventral (sternum)
            xr = x_i - 0.05 * H * (th / np.pi)                   # slope caudally as it comes forward
            P = np.c_[xr, dv, z] + rng.normal(size=(per, 3)) * 0.004
            parts[f"rib{n}-{sd}"] = dict(kind="bone", part="rib", rib=n, side=sd, P=P)
            if n <= 7:
                vent_pts.append(P[-1])                           # the sternal end of this true rib
    # STERNUM: a midline bar down the ventral thorax over the true-rib span, carved into Gray's three named
    # parts -- MANUBRIUM (superior, articulates with clavicle + rib 1), BODY (the long middle, ribs 2-7),
    # XIPHOID process (the small inferior tip). Superior = high AP (head-forward), inferior = low AP.
    if vent_pts:
        vp = np.array(vent_pts)
        sdv = float(np.median(vp[:, 1]))
        x_top, x_bot = float(vp[:, 0].max()), float(vp[:, 0].min())
        x_lo = x_bot - 0.03 * H                                 # extend a little below for the xiphoid
        wid = 0.020 * H                                         # sternal WIDTH (ML) -- a FLAT bar, not a line
        # manubrium (superior) + body (long middle) + xiphoid (small inferior tip), each a proper flat plate
        # with ~30 cells so the sub-segments are non-degenerate (the Gray's scorecard found them n<20 lines).
        for nm, (a0, a1, wsc, nps) in (("xiphoid", (0.0, 0.16, 0.5, 22)),
                                       ("body", (0.16, 0.80, 1.0, 40)),
                                       ("manubrium", (0.80, 1.0, 1.5, 30))):
            xa = x_lo + (x_top - x_lo) * (a0 + (a1 - a0) * rng.random(nps))
            za = (rng.random(nps) - 0.5) * wid * wsc            # manubrium widest, xiphoid narrowest
            Ps = np.c_[xa, np.full(nps, sdv), za] + rng.normal(size=(nps, 3)) * 0.003
            parts[f"sternum-{nm}"] = dict(kind="bone", part="sternum", bone=nm, side="M", P=Ps)
    return dict(parts=parts, H=H, span=span)


def _validate(res):
    parts = res["parts"]
    ribs = sorted({p["rib"] for p in parts.values() if p["part"] == "rib"})
    pairs = sum(1 for n in ribs if f"rib{n}-R" in parts and f"rib{n}-L" in parts)
    sternum = sorted({p["bone"] for p in parts.values() if p["part"] == "sternum" and "bone" in p})
    return dict(parts=len(parts), rib_pairs=pairs, distinct_rib_levels=len(ribs),
                has_sternum=int(len(sternum) > 0), sternum_parts=sternum,
                n_bone=sum(p["kind"] == "bone" for p in parts.values()))


def _figure(res, base):
    parts = res["parts"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), facecolor="#0d1017")
    for j, (i, k, ttl) in enumerate([(2, 0, "front (ML x AP)"), (1, 0, "side (DV x AP)")]):
        a = ax[j]; a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        a.scatter(base[::7, i], base[::7, k], s=2, c="#2a3140", alpha=0.35)
        for p in parts.values():
            c = "#fbbf24" if p["part"] == "sternum" else "#cbd5e1"
            a.scatter(p["P"][:, i], p["P"][:, k], s=6, c=c, alpha=0.85)
        a.set_title(ttl, color="#cbd5e1", fontsize=9)
    v = _validate(res)
    fig.suptitle(f"Rib-cage head: {v['rib_pairs']} rib pairs (true 1-7 to the sternum, false 8-10, floating "
                 f"11-12) + sternum (gold) -- reads the thoracic vertebrae, read-only", color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/rib_cage_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/rib_cage_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res)
    print(f"rib cage: {v['parts']} parts, {v['rib_pairs']} rib pairs, sternum={bool(v['has_sternum'])}")
    print("  genome-derived: one rib per thoracic vertebra (T1..T12); true 1-7 reach the sternum, 8-10 false, 11-12 floating")
    _figure(res, base)
    json.dump(v, open("data/organ_cascade/rib_cage_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/rib_cage_head.json")


if __name__ == "__main__":
    main()
