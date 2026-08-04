"""
shoulder_girdle_head.py -- the SHOULDER-GIRDLE (pectoral girdle) head: build the clavicle + scapula + glenoid
+ deltoid cap that give the body its biacromial (shoulder) WIDTH.

WHY THIS HEAD EXISTS (the 2026-07-31 knob-tuning finding): the maturation `shoulder_w` knob could not widen
the rendered shoulder past biacromial ~0.16*H (canon 0.25*H) -- shoulder width turned out to be MECHANISM-
limited, not range-limited: there simply were no cells out at the shoulder span to widen. A knob cannot buy
width that has no cells. This head ADDS them, at the anthropometric span, structurally.

Biology (genome-anchored; all input fields already laid down upstream):
  * SPAN / LEVEL = the pectoral girdle forms at the FORELIMB level (Tbx5 forelimb field, Hoxc6 brachial),
    lateral to the upper thoracic column. The SCAPULAR blade is an Emx2 / Tbx15 / Pax1 field; the biacromial
    breadth is an anthropometric constant ~0.25*H -- the spec anchor (like the limb bones are named by Hox).
  * CLAVICLE = a dermal strut from the STERNOCLAVICULAR joint (ventral midline, upper thorax) laterally to
    the ACROMION (Runx2 intramembranous). It is the strut that HOLDS the shoulder out at width.
  * SCAPULA = a flat triangular blade DORSOLATERAL over the upper ribs, from a medial border near the spine
    out to the GLENOID, where the humerus head articulates.
  * DELTOID = the muscle cap over the acromion + humerus head; it is the lateral soft-tissue mass that the
    shoulder silhouette actually reads.
So the head reads {forelimb bud (level + where the humerus attaches), midline, notochord = the dorsal ref,
body scale H} -- read-only -- and WRITES the girdle bones + deltoid at the biacromial span. It moves no
existing cell, so it cannot oppose the limb placement or the axial skeleton.

READ-ONLY on the base. Validation = the girdle is bilateral, the clavicle spans midline->acromion, the
acromion sits at ~the anthropometric biacromial width, and the glenoid lands near the forelimb's proximal
(shoulder) end so the humerus can articulate.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.shoulder_girdle_head
Out: data/organ_cascade/shoulder_girdle_head.{png,json}
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
    """+1 or -1 along DV (axis 1) toward DORSAL, read off the notochord (the dorsal-midline reference the
    vasculature head also uses); fall back to the body's own DV spread if no notochord."""
    if "Notochord" in FIDX and (F == FIDX["Notochord"]).sum() > 8:
        noto_dv = np.median(base[F == FIDX["Notochord"]][:, 1])
        return 1.0 if noto_dv >= np.median(base[:, 1]) else -1.0
    return 1.0


def _forelimbs(base, F):
    """The two forelimb-bud clusters (anterior half, split by ML sign). Returns {side: cells}."""
    if "Limb Bud" not in FIDX:
        return {}
    P = base[F == FIDX["Limb Bud"]]
    if not len(P):
        return {}
    apf = (P[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)
    fore = P[apf >= 0.5]                                  # anterior half = the forelimbs (fore = high AP)
    out = {}
    for side, m in (("R", fore[:, 2] > 0), ("L", fore[:, 2] < 0)):
        if m.sum() >= 6:
            out[side] = fore[m]
    return out


def _arc(p0, p1, bow, n):
    """A gentle arc of n points from p0 to p1, bowed by `bow` (in world units) along +DV (the clavicle /
    scapular spine curve, not a straight rod)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    line = p0 * (1 - t) + p1 * t
    line[:, 1] += bow * np.sin(np.pi * t[:, 0])          # bow along DV
    return line


def build(base, F, span=None, per=90):
    """The shoulder-girdle head. `span` = the biacromial HALF-width as a fraction of body length H (the spec
    anchor; None -> tuned, default 0.125 so full biacromial = 0.25*H = the canon). Returns per-part dicts."""
    if span is None:
        # 0.125 = biacromial 0.25*H (the canon). NB the maturation envelope clamps the RENDERED shoulder canon
        # at ~0.23 regardless of a wider span or a higher shoulder_w -- the last 0.02 is envelope-limited, not
        # girdle- or knob-limited (a wider span only distorted the silhouette without moving the shoulder).
        span = tuned("shoulder_girdle", {"span": 0.125})["span"]
    H = float(np.ptp(base[:, 0]))                         # body length (AP long axis) = height scale
    dsgn = _dorsal_sign(base, F)
    mid_z = 0.0                                           # ML midline
    fl = _forelimbs(base, F)
    rng = np.random.default_rng(0)
    parts = {}
    for side, cells in fl.items():
        sgn = 1.0 if side == "R" else -1.0
        # SHOULDER LEVEL = the forelimb's proximal (near-trunk) end: the cells with the smallest |ML|.
        proximal = cells[np.argsort(np.abs(cells[:, 2]))[:max(4, len(cells) // 5)]]
        ap0 = float(np.median(proximal[:, 0]))           # shoulder AP level
        dv0 = float(np.median(proximal[:, 1]))           # shoulder DV level
        glenoid = np.array([ap0, dv0, sgn * span * H])   # acromion/glenoid at the biacromial span
        sc = np.array([ap0, dv0 - 0.15 * dsgn * span * H, mid_z + sgn * 0.02 * H])  # sternoclavicular (ventral midline)
        # CLAVICLE: dermal strut sternoclavicular -> acromion, bowed ventrally (Runx2)
        clav = _arc(sc, glenoid, bow=-0.05 * dsgn * span * H, n=per // 2)
        clav += rng.normal(size=clav.shape) * 0.004
        # SCAPULA: a flat triangular blade DORSAL to the ribs, medial border near the spine -> glenoid.
        medial = np.array([ap0 + 0.05 * H, dv0 + 0.9 * dsgn * span * H, sgn * 0.25 * span * H])  # dorsomedial corner
        inferior = np.array([ap0 - 0.10 * H, dv0 + 0.6 * dsgn * span * H, sgn * 0.4 * span * H])  # inferior angle
        blade = []
        for _ in range(per):
            a, b = rng.random(2)
            if a + b > 1:                                # barycentric fill of the (medial, inferior, glenoid) triangle
                a, b = 1 - a, 1 - b
            blade.append(glenoid + a * (medial - glenoid) + b * (inferior - glenoid))
        blade = np.array(blade) + rng.normal(size=(per, 3)) * 0.005
        # DELTOID cap: the lateral muscle mass draping the acromion over the humerus head (the width the
        # silhouette reads). A shell just lateral + distal of the glenoid.
        humeral = cells[np.argmax(np.abs(cells[:, 2]))]  # a distal-ish forelimb point (arm direction)
        armdir = humeral - glenoid; armdir = armdir / (np.linalg.norm(armdir) + 1e-9)
        delt = []
        for _ in range(per):
            t = rng.random() * 0.55                       # along the upper arm from the acromion
            r = (0.35 + 0.65 * rng.random()) * span * H * 0.45
            th = rng.random() * 2 * np.pi
            perp1 = np.cross(armdir, [0, 0, 1.0]); perp1 /= np.linalg.norm(perp1) + 1e-9
            perp2 = np.cross(armdir, perp1)
            delt.append(glenoid + armdir * t * span * H * 2.2 + (np.cos(th) * perp1 + np.sin(th) * perp2) * r)
        delt = np.array(delt)
        parts[f"clavicle-{side}"] = dict(kind="bone", part="clavicle", side=side, P=clav)
        parts[f"scapula-{side}"] = dict(kind="bone", part="scapula", side=side, P=blade)
        # Gray's: the glenoid is a FOSSA OF THE SCAPULA, not a separate bone -> label it as scapula, tagged as
        # the glenoid landmark (the pectoral girdle bones are just clavicle + scapula).
        parts[f"glenoid-{side}"] = dict(kind="bone", part="scapula", landmark="glenoid", side=side, P=glenoid[None])
        parts[f"deltoid-{side}"] = dict(kind="muscle", part="deltoid", side=side, P=delt)
    return dict(parts=parts, H=H, span=span, glenoids={s: p["P"][0] for s, p in
                {f"glenoid-{k}": parts[f"glenoid-{k}"] for k in fl}.items()})


def augment(base, F, per=None):
    """Inject the girdle CELLS into the base cloud (not just parts), so the maturation, the Vitruvian canon
    shoulder metric, the anthropometric silhouette, and the skin surface all see the biacromial WIDTH -- the
    knob search proved the shoulder was MECHANISM-limited (no cells out at the span), so a knob can't widen it;
    these cells are the mass. Clavicle/scapula -> Cartilage (skeletal), deltoid -> Muscle. Read-only; returns
    (base2, F2, n_added). `per` (density per part per side) SCALES WITH CLOUD SIZE so the girdle stays a
    consistent FRACTION of the shoulder band -- the canon reads the 90th-percentile ML, so the acromial cells
    must be more than the top decile of the band (a fixed count is diluted below that at high cell counts)."""
    if per is None:
        per = int(np.clip(len(base) / 140, 200, 3000))       # ~436 at 60k, ~1745 at 240k
    res = build(base, F, per=per)
    parts = res["parts"]
    if not parts:
        return base, F, 0
    # label the girdle bone cells CONNECTIVE, not Cartilage: the axial-skeleton carving (complete_column /
    # rib_cage_head) reads Cartilage, so girdle Cartilage at the shoulder polluted the axial pool and broke the
    # rib<->vertebra articulation (24/24 -> 8/24). Connective is inert to the skeletal carving; the real girdle
    # BONES are still rendered from the parts in integrated_body -- these cloud cells only supply WIDTH.
    cart = FIDX.get("Connective", FIDX.get("Muscle"))
    mus = FIDX["Muscle"]
    addP, addF = [], []
    for p in parts.values():
        P = np.atleast_2d(p["P"])
        addP.append(P)
        addF.append(np.full(len(P), mus if p["kind"] == "muscle" else cart))
    P = np.vstack(addP); Fa = np.concatenate(addF)
    return np.vstack([base, P]), np.concatenate([F, Fa]), len(P)


def _validate(res, base, F):
    parts = res["parts"]
    H = res["H"]
    sides = {p["side"] for p in parts.values()}
    bilateral = int({"R", "L"} <= sides)
    # biacromial breadth = span across the two acromions (glenoids), over H
    gl = [p["P"][0] for k, p in parts.items() if p.get("landmark") == "glenoid"]
    biac = float((max(g[2] for g in gl) - min(g[2] for g in gl)) / H) if len(gl) == 2 else 0.0
    # clavicle spans midline -> acromion (min|z| near midline, max|z| near the span)
    clav_ok = 0
    for p in parts.values():
        if p["part"] == "clavicle":
            z = np.abs(p["P"][:, 2])
            clav_ok += int(z.min() < 0.05 * H and z.max() > 0.5 * res["span"] * H)
    # glenoid lands near the forelimb proximal end (so the humerus articulates)
    fl = _forelimbs(base, F)
    glen_near = 0
    for s, cells in fl.items():
        g = parts.get(f"glenoid-{s}")
        if g is not None:
            d = np.linalg.norm(cells - g["P"][0], axis=1).min()
            glen_near += int(d < 0.15 * H)
    return dict(parts=len(parts), bilateral=bilateral, biacromial_over_H=round(biac, 3),
                clavicle_spans=clav_ok, glenoid_near_forelimb=glen_near,
                n_bone=sum(p["kind"] == "bone" for p in parts.values()),
                n_muscle=sum(p["kind"] == "muscle" for p in parts.values()))


def _figure(res, base, F):
    parts = res["parts"]
    col = {"clavicle": "#e5e7eb", "scapula": "#93c5fd", "glenoid": "#fbbf24", "deltoid": "#ef4444"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 7), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    # front view AP(x) x ML(z); the biacromial width shows here
    ax[0].scatter(base[::7, 0], base[::7, 2], s=2, c="#2a3140", alpha=0.4)
    for p in parts.values():
        P = p["P"]
        ax[0].scatter(P[:, 0], P[:, 2], s=14 if p["part"] == "glenoid" else 7, c=col[p["part"]], alpha=0.9)
    v = _validate(res, base, F)
    ax[0].set_title(f"front (AP x ML): biacromial {v['biacromial_over_H']}*H (canon 0.25)", color="#cbd5e1", fontsize=9)
    # side view AP(x) x DV(y)
    ax[1].scatter(base[::7, 0], base[::7, 1], s=2, c="#2a3140", alpha=0.4)
    for p in parts.values():
        P = p["P"]
        ax[1].scatter(P[:, 0], P[:, 1], s=14 if p["part"] == "glenoid" else 7, c=col[p["part"]], alpha=0.9)
    ax[1].set_title("side (AP x DV)", color="#cbd5e1", fontsize=9)
    fig.suptitle("Shoulder-girdle head: clavicle (white) + scapula (blue) + glenoid (gold) + deltoid (red) built at "
                 "the biacromial span from the forelimb+midline -- read-only, gives the shoulder its WIDTH",
                 color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/shoulder_girdle_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/shoulder_girdle_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res, base, F)
    print(f"shoulder girdle: {v['parts']} parts ({v['n_bone']} bone + {v['n_muscle']} muscle), bilateral={v['bilateral']}")
    print(f"  biacromial breadth: {v['biacromial_over_H']}*H  (anthropometric canon 0.25)")
    print(f"  clavicle spans midline->acromion: {v['clavicle_spans']}/2")
    print(f"  glenoid near forelimb proximal (humerus articulates): {v['glenoid_near_forelimb']}/2")
    print("  genome-derived: Tbx5/Hoxc6 level, Emx2/Tbx15 scapula, Runx2 clavicle; reads forelimb+midline, read-only")
    _figure(res, base, F)
    json.dump(v, open("data/organ_cascade/shoulder_girdle_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/shoulder_girdle_head.json")


if __name__ == "__main__":
    main()
