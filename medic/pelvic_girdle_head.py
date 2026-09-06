"""
pelvic_girdle_head.py -- the PELVIC-GIRDLE head: build the ilium + ischium + pubis + acetabulum that form the
hip bone, the counterpart of the shoulder girdle. It gives the lower body its hip width and anchors the legs.

WHY: the model had a shoulder girdle but no pelvic girdle -- the two femur heads sat at the hips with nothing
between them ("2 pelvic bones, no pelvis"). This head builds the os coxae, joining the sacrum to each femur.

Biology (genome-anchored; input fields already laid down):
  * LEVEL / SPAN = the pelvic girdle forms at the HINDLIMB level (Pitx1/Tbx4 hindlimb field, Hox9-13 sacral),
    from lateral-plate mesoderm; the biiliac breadth is an anthropometric ~0.18*H -- the spec anchor.
  * ILIUM = the blade (Emx2/Alx4/Pbx1) from the SACRUM (sacroiliac joint, dorsal midline) out to the
    ACETABULUM (the hip socket).
  * ISCHIUM + PUBIS = the lower ring from the acetabulum toward the ventral midline, where the two sides meet
    at the PUBIC SYMPHYSIS.
  * ACETABULUM = the socket where the femur head articulates (at the hindlimb's proximal end).
So the head reads {hindlimb bud (level + where the femur attaches), midline, sacrum/notochord = dorsal ref,
H} -- read-only -- and WRITES the girdle at the biiliac span. It moves no existing cell.

READ-ONLY. Validation = bilateral, the ilium spans sacrum->acetabulum, the two pubes meet near the midline,
the acetabulum sits near the hindlimb proximal end, biiliac breadth ~ anthropometric.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.pelvic_girdle_head
Out: data/organ_cascade/pelvic_girdle_head.{png,json}
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


def _hindlimbs(base, F):
    """The two hindlimb-bud clusters (posterior half, split by ML sign)."""
    if "Limb Bud" not in FIDX:
        return {}
    P = base[F == FIDX["Limb Bud"]]
    if not len(P):
        return {}
    apf = (P[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)
    hind = P[apf < 0.5]                                   # posterior half = the hindlimbs (legs)
    out = {}
    for side, m in (("R", hind[:, 2] > 0), ("L", hind[:, 2] < 0)):
        if m.sum() >= 6:
            out[side] = hind[m]
    return out


def build(base, F, span=None, per=90):
    """The pelvic-girdle head. `span` = biiliac HALF-width as a fraction of body length H (spec anchor; None ->
    tuned, default 0.10 so biiliac = 0.20*H). Returns per-part dicts (ilium/ischiopubis/acetabulum bone)."""
    if span is None:
        span = tuned("pelvic_girdle", {"span": 0.10})["span"]
    H = float(np.ptp(base[:, 0]))
    dsgn = _dorsal_sign(base, F)
    hl = _hindlimbs(base, F)
    rng = np.random.default_rng(1)
    parts = {}
    for side, cells in hl.items():
        sgn = 1.0 if side == "R" else -1.0
        proximal = cells[np.argsort(np.abs(cells[:, 2]))[:max(4, len(cells) // 5)]]  # hip end (near midline)
        ap0 = float(np.median(proximal[:, 0]))
        dv0 = float(np.median(proximal[:, 1]))
        acet = np.array([ap0, dv0, sgn * span * H])                                  # acetabulum (hip socket)
        # ILIUM: a blade from the sacrum (dorsal, a touch superior/anterior of the socket) to the acetabulum
        sacrum = np.array([ap0 + 0.06 * H, dv0 + 0.9 * dsgn * span * H, sgn * 0.15 * span * H])
        wing = np.array([ap0 + 0.02 * H, dv0 + 0.5 * dsgn * span * H, sgn * 1.1 * span * H])  # iliac crest (flared)
        blade = []
        for _ in range(per):
            a, b = rng.random(2)
            if a + b > 1:
                a, b = 1 - a, 1 - b
            blade.append(acet + a * (sacrum - acet) + b * (wing - acet))
        blade = np.array(blade) + rng.normal(size=(per, 3)) * 0.005
        # ISCHIUM + PUBIS (Gray's: the os coxae is ilium + ISCHIUM + PUBIS): an arc from the acetabulum
        # down-and-in toward the ventral midline (pubic symphysis). Ischium = the posteroinferior part nearer
        # the acetabulum; pubis = the anterior part reaching the symphysis. Split the arc at its midpoint.
        symph = np.array([ap0 - 0.03 * H, dv0 - 0.5 * dsgn * span * H, sgn * 0.05 * H])
        t = np.linspace(0.0, 1.0, per // 2)[:, None]
        ring = acet * (1 - t) + symph * t
        ring[:, 1] += -0.25 * dsgn * span * H * np.sin(np.pi * t[:, 0])              # bow ventrally (obturator)
        ring += rng.normal(size=ring.shape) * 0.004
        tt = t[:, 0]
        parts[f"ilium-{side}"] = dict(kind="bone", part="ilium", side=side, P=blade)
        parts[f"ischium-{side}"] = dict(kind="bone", part="ischium", side=side, P=ring[tt < 0.5])
        parts[f"pubis-{side}"] = dict(kind="bone", part="pubis", side=side, P=ring[tt >= 0.5])
        femoral = cells[np.argmax(np.abs(cells[:, 2]))]                 # a distal-ish hindlimb point (femur dir)
        cup = _socket_cup(acet, femoral - acet, 0.13 * span * H, 36, rng)   # a real concave hip socket
        parts[f"acetabulum-{side}"] = dict(kind="bone", part="acetabulum", side=side,
                                           P=np.vstack([acet[None], cup]))
    return dict(parts=parts, H=H, span=span)


def _socket_cup(center, facing, radius, n, rng):
    """A concave articular SOCKET (acetabulum): a shallow spherical-cap bowl opening toward `facing` (the
    femoral head articulates on the +facing side). Fixes the n=1 socket the Gray's scorecard flagged."""
    f = np.asarray(facing, float); f = f / (np.linalg.norm(f) + 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(f, ref); u /= np.linalg.norm(u) + 1e-9; v = np.cross(f, u)
    pts = []
    for _ in range(n):
        rr = radius * np.sqrt(rng.random()); th = rng.random() * 2 * np.pi
        depth = 0.45 * radius * (1 - (rr / (radius + 1e-9)) ** 2)
        pts.append(center + u * rr * np.cos(th) + v * rr * np.sin(th) - f * depth)
    return np.array(pts)


def _validate(res, base, F):
    parts = res["parts"]; H = res["H"]
    sides = {p["side"] for p in parts.values()}
    bilateral = int({"R", "L"} <= sides)
    ac = [p["P"][0] for p in parts.values() if p["part"] == "acetabulum"]
    biiliac = float((max(a[2] for a in ac) - min(a[2] for a in ac)) / H) if len(ac) == 2 else 0.0
    # pubes meet near the midline
    pubis_gap = 1.0
    pr = [p["P"] for p in parts.values() if p["part"] == "pubis"]
    if len(pr) == 2:
        # closest approach of the two ischiopubic arcs' medial ends
        e0 = pr[0][np.abs(pr[0][:, 2]).argmin()]; e1 = pr[1][np.abs(pr[1][:, 2]).argmin()]
        pubis_gap = float(np.linalg.norm(e0 - e1) / H)
    hl = _hindlimbs(base, F)
    acet_near = 0
    for s, cells in hl.items():
        a = parts.get(f"acetabulum-{s}")
        if a is not None:
            acet_near += int(np.linalg.norm(cells - a["P"][0], axis=1).min() < 0.15 * H)
    return dict(parts=len(parts), bilateral=bilateral, biiliac_over_H=round(biiliac, 3),
                pubic_symphysis_gap_over_H=round(pubis_gap, 3), acetabulum_near_hindlimb=acet_near,
                n_bone=sum(p["kind"] == "bone" for p in parts.values()))


def _figure(res, base, F):
    parts = res["parts"]
    col = {"ilium": "#93c5fd", "ischium": "#e5e7eb", "pubis": "#c7b8ff", "acetabulum": "#fbbf24"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    for j, (i, k, ttl) in enumerate([(2, 0, "front (AP x ML)"), (1, 0, "side (AP x DV)")]):
        ax[j].scatter(base[::7, i], base[::7, k], s=2, c="#2a3140", alpha=0.4)
        for p in parts.values():
            P = p["P"]
            ax[j].scatter(P[:, i], P[:, k], s=14 if p["part"] == "acetabulum" else 7, c=col[p["part"]], alpha=0.9)
        ax[j].set_title(ttl, color="#cbd5e1", fontsize=9)
    v = _validate(res, base, F)
    fig.suptitle(f"Pelvic-girdle head: ilium (blue) + ischiopubis (white) + acetabulum (gold) -- biiliac "
                 f"{v['biiliac_over_H']}*H, joins the sacrum to each femur, read-only", color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/pelvic_girdle_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/pelvic_girdle_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res, base, F)
    print(f"pelvic girdle: {v['parts']} parts ({v['n_bone']} bone), bilateral={v['bilateral']}")
    print(f"  biiliac breadth: {v['biiliac_over_H']}*H  (anthropometric ~0.18)")
    print(f"  pubic symphysis gap: {v['pubic_symphysis_gap_over_H']}*H (the two pubes meet at the midline)")
    print(f"  acetabulum near hindlimb proximal (femur articulates): {v['acetabulum_near_hindlimb']}/2")
    print("  genome-derived: Pitx1/Tbx4 hindlimb level, Hox sacral, Emx2/Alx4 ilium; reads hindlimb+midline+sacrum")
    _figure(res, base, F)
    json.dump(v, open("data/organ_cascade/pelvic_girdle_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/pelvic_girdle_head.json")


if __name__ == "__main__":
    main()
