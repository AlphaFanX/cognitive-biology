"""pbd_viscera.py -- position-based-dynamics packing of the viscera. (2026-08-19, prototype)

THE EXPERIMENT: physics instead of placement. dv_spread_head MOVES each viscus to its measured
BodyParts3D depth (a kinematic re-target that broke once when later heads shifted the envelope).
Here the measured depths are VALIDATION ONLY. Inputs are the anatomy Gray's actually specifies:
  - attachment TOPOLOGY (which wall each organ is tethered to: kidney/pancreas retroperitoneal,
    heart mediastinum, liver diaphragm, gut on a mesentery root, bladder pelvic floor),
  - volume exclusion (organs are soft bodies that cannot interpenetrate),
  - the body cavity (trunk envelope) as containment.
Organs start SCRAMBLED to a DV-agnostic mid-cavity stack; PBD relaxation (shape matching +
contacts + attachments + containment) finds the equilibrium. PASS = the measured local-trunk
DV depths EMERGE at the equilibrium instead of being imposed.

This is the relational attractor made literal: Gray's relations = constraints, anatomy = the
energy minimum. If it works, dv_spread_head becomes a solver pass, not a re-target.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.pbd_viscera [--ne 30000] [--iters 300]
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from medic.unified_embryo import FIDX
from medic.dv_spread_head import DV_GROUP, _targets, _dorsal_sign, _limb_mask

OUT_JSON = "data/organ_cascade/pbd_viscera.json"
OUT_PNG = "data/organ_cascade/pbd_viscera.png"

N_PART = 220          # particles per organ (downsampled)
NB = 24               # AP bands (same frame as dv_spread_head)
MARGIN = 0.05         # containment margin, fraction of band DV span
# -- compliant belly wall (2026-08-29): the VENTRAL wall yields where there is no rib cage.
# Real anatomy: the thoracic wall is rigid (ribs), the abdominal wall is muscle/aponeurosis
# and BULGES; the omentum drapes ventrally and the midgut returns ventrally at herniation
# reduction. Compliance is derived from RIB PRESENCE per AP band -- anatomy, not a target.
BULGE = 0.15          # bulge budget: unribbed wall may yield ventrally by this fraction
                      # of the band DV span beyond the resting (measured) envelope
K_WALL = 0.10         # per-iteration restitution of the compliant wall (soft spring)
# -- the peritoneal lining (Gray's CATEGORICAL topology, no measured depth): kidney and
# pancreas lie BEHIND the posterior parietal peritoneum; stomach/intestines/liver/spleen
# lie IN FRONT of it. The lining = the ventral face of the retro layer, per AP band;
# intraperitoneal organs are softly confined ventral of it (belly bands only).
RETRO = ("Kidney", "Pancreas")
INTRA = ("Stomach", "SmallIntestine", "LargeIntestine", "Liver", "Spleen")
K_PERIT = 0.25        # per-iteration restitution of the peritoneal lining
# -- diaphragm dome (2026-08-29): musculotendinous partition attached on the COSTAL RING
# (the rigid/compliant wall boundary), arching CRANIALLY (dome peak central -- the liver
# reaches up under it peripherally-centrally, Gray's). Thoracic organs stay cranial of the
# sheet, abdominal viscera caudal. The esophageal hiatus = the stomach's tether passes
# through unimpeded (the tether is exempt, only cell positions are constrained).
THORACIC = ("Heart", "Lung")
H_DOME = 1.5          # dome apex height above the costal ring, in AP band widths
K_DIA = 0.30          # per-iteration restitution of the diaphragm sheet

# attachment TOPOLOGY (the Gray's input): organ -> (wall DV fraction, rest length as span
# fraction, stiffness). DV fraction is which WALL the tether reaches (0.9=dorsal body wall,
# 0.15=ventral wall) -- schematic topology, NOT the measured depths (those stay validation).
ANCHORS = {
    "Kidney":         (0.90, 0.02, 0.9),   # retroperitoneal: pressed to the dorsal wall
    "Pancreas":       (0.60, 0.02, 0.8),   # retroperitoneal but the ANTERIOR pararenal lamina
                                           # (Gray's: lies on the posterior wall IN FRONT of
                                           # the kidneys/great vessels)
    "Spleen":         (0.80, 0.06, 0.5),   # dorsal-left, splenorenal ligament
    "Lung":           (0.70, 0.05, 0.5),   # hilum / posterior mediastinum
    "Heart":          (0.35, 0.06, 0.6),   # sternopericardial ligaments: pericardium tethers
                                           # to the STERNUM (ventral), heart sits anterior
                                           # between the lungs (Gray's middle mediastinum)
    "Stomach":        (0.55, 0.12, 0.3),   # esophageal hiatus, then hangs
    "Liver":          (0.35, 0.10, 0.4),   # coronary/falciform ligaments to diaphragm, ventral
    "SmallIntestine": (0.70, 0.35, 0.15),  # mesentery ROOT is dorsal; long, loose -> hangs ventral
    "LargeIntestine": (0.70, 0.25, 0.25),  # mesocolon, partly retro
    "Bladder":        (0.25, 0.05, 0.35),  # pelvic floor, retropubic (ventral, loose)
}


def band_envelope(base, F):
    """Per-AP-band trunk envelope in oriented DV and ML (same frame as dv_spread_head)."""
    s = _dorsal_sign(base, F)
    x = base[:, 0]
    ody = base[:, 1] * s
    z = base[:, 2]
    axial = ~_limb_mask(F)
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    band = np.clip((apf * NB).astype(int), 0, NB - 1)
    olo = np.zeros(NB); ohi = np.zeros(NB); zlo = np.zeros(NB); zhi = np.zeros(NB)
    for k in range(NB):
        m = axial & (band == k)
        if m.sum() > 20:
            olo[k] = np.percentile(ody[m], 5); ohi[k] = np.percentile(ody[m], 95)
            zlo[k] = np.percentile(z[m], 5); zhi[k] = np.percentile(z[m], 95)
        else:
            olo[k] = ody.min(); ohi[k] = ody.max(); zlo[k] = z.min(); zhi[k] = z.max()
    return s, apf, band, olo, ohi, zlo, zhi


def organ_groups(F, tgt):
    groups = {}
    for fate, organ in DV_GROUP.items():
        fid = FIDX.get(fate)
        if fid is not None and organ in tgt:
            groups.setdefault(organ, []).append(fid)
    return {o: np.isin(F, fids) for o, fids in groups.items()
            if np.isin(F, fids).sum() >= 8}


def measure_dv(base, F, masks):
    """Local-trunk DV per organ, dv_spread_head's exact formula."""
    s, apf, band, olo, ohi, _, _ = band_envelope(base, F)
    ody = base[:, 1] * s
    out = {}
    for organ, m in masks.items():
        b = int(np.clip(apf[m].mean() * NB, 0, NB - 1))
        span = (ohi[b] - olo[b]) + 1e-9
        out[organ] = float((ody[m].mean() - olo[b]) / span)
    return out


def pack(base, F, masks, iters=300, seed=0, verbose=True):
    """Scramble organ DV, then PBD-relax: shape match + contacts + attachments + containment.
    Returns the updated cloud (organ cells rigidly transformed to the equilibrium)."""
    rng = np.random.default_rng(seed)
    base = np.asarray(base, float).copy()
    s, apf, band, olo, ohi, zlo, zhi = band_envelope(base, F)

    # -- scramble: every organ to mid-cavity DV (+ jitter) => DV-agnostic start
    for organ, m in masks.items():
        b = int(np.clip(apf[m].mean() * NB, 0, NB - 1))
        span = ohi[b] - olo[b]
        cur = (base[m, 1] * s).mean()
        want = olo[b] + (0.5 + rng.uniform(-0.08, 0.08)) * span
        base[m, 1] += (want - cur) * s

    # -- particles: downsample each organ
    P, R, org_of, rest, cent0 = [], [], [], {}, {}
    scr_cells = {}
    for oi, (organ, m) in enumerate(masks.items()):
        cells = base[m]
        scr_cells[organ] = cells.copy()
        idx = rng.choice(len(cells), size=min(N_PART, len(cells)), replace=False)
        pts = cells[idx]
        d = cKDTree(pts).query(pts, k=2)[0][:, 1]
        P.append(pts); R.append(np.full(len(pts), np.median(d) * 1.0))
        org_of.append(np.full(len(pts), oi))
        c = pts.mean(0)
        rest[organ] = pts - c
        cent0[organ] = c
    P = np.vstack(P); R = np.concatenate(R); org_of = np.concatenate(org_of)
    organs = list(masks.keys())
    P0 = P.copy()

    # -- static colliders: the body wall IS geometry. The dorsal trunk mass (spine, back
    # muscles) is solid -- the peritoneal cavity is what remains. This, not any target,
    # is what forces the gut ventral.
    any_visc = np.any([masks[o] for o in masks], axis=0)
    axial = ~_limb_mask(F)
    bmin = min(int(np.clip(apf[masks[o]].mean() * NB, 0, NB - 1)) for o in masks) - 1
    bmax = max(int(np.clip(apf[masks[o]].mean() * NB, 0, NB - 1)) for o in masks) + 1
    inband = (band >= bmin) & (band <= bmax)
    # only STRUCTURAL tissue is rigid (spine, somite-derived back mass, ribs, wall muscle,
    # skin); fat/connective/mesoderm yield and are not colliders
    struct_ids = [FIDX[n] for n in ("Spinal Cord", "Notochord", "Somite", "Rib",
                                    "Cartilage", "Muscle", "Skin", "Epidermal") if n in FIDX]
    structural = np.isin(F, struct_ids)
    wall = base[axial & ~any_visc & inband & structural]
    if len(wall) > 5000:
        wall = wall[rng.choice(len(wall), 5000, replace=False)]
    r_stat = float(np.median(cKDTree(wall).query(wall, k=2)[0][:, 1])) * 0.9
    stat_tree = cKDTree(wall)

    # -- per-band wall compliance: rigid where the rib cage spans, compliant caudal of it.
    # Rib fate cells are absent in the reduced solver build, so the costal margin is proxied
    # by the LUNG's caudal extent (the thoracic cage encloses the lungs; Gray's).
    rib_ids = [FIDX[n] for n in ("Rib",) if n in FIDX]
    ribbed = np.zeros(NB, bool)
    if rib_ids:
        rb = band[np.isin(F, rib_ids)]
        for k in range(NB):
            ribbed[k] = (rb == k).sum() >= 10
    if not ribbed.any() and "Lung" in masks:
        # head is at +x (high apf band), so the thorax occupies the HIGH bands: rigid from
        # the lung's caudal (low-band) edge to the head end, compliant belly caudal of it
        lung_b = np.clip((apf[masks["Lung"]] * NB).astype(int), 0, NB - 1)
        ribbed[int(np.percentile(lung_b, 10)):] = True
    if verbose:
        print(f"[pbd] rigid (ribbed/thoracic) bands: {np.where(ribbed)[0].tolist()}  "
              f"compliant: {np.where(~ribbed)[0].tolist()}")

    # -- cavity fill: the coelom has NO empty space (omentum/mesentery fat packs it).
    # Inflate effective INTER-ORGAN contact radii so the viscera fill the cavity; anchored
    # organs then displace free ones into the compliant wall instead of all floating
    # mid-cavity. Wall collisions keep true radii (organs touch walls directly).
    dx_band = (np.ptp(base[:, 0]) + 1e-9) / NB
    kk = np.arange(NB)
    vband = (kk >= bmin) & (kk <= bmax)
    cav_vol = float(np.sum((ohi - olo)[vband] * (zhi - zlo)[vband]) * dx_band)
    p_vol = float(np.sum(4.0 / 3.0 * np.pi * R ** 3))
    fill = p_vol / (cav_vol + 1e-12)
    PACK = float(np.clip((0.50 / max(fill, 1e-6)) ** (1.0 / 3.0), 1.0, 2.0))
    if verbose:
        print(f"[pbd] cavity fill {fill:.3f} -> contact-radius inflation x{PACK:.2f}")

    # -- diaphragm dome geometry: costal ring = the first rigid band (thorax at high x)
    dia_on = ribbed.any() and any(o in organs for o in THORACIC)
    if dia_on:
        mb = int(np.where(ribbed)[0].min())                  # costal-margin band
        x_margin = base[:, 0].min() + mb * dx_band
        dv_mid = 0.5 * (olo[mb] + ohi[mb]); dv_half = 0.5 * (ohi[mb] - olo[mb]) + 1e-9
        z_mid = 0.5 * (zlo[mb] + zhi[mb]); z_half = 0.5 * (zhi[mb] - zlo[mb]) + 1e-9
        tho_pi = np.isin(org_of, [organs.index(o) for o in THORACIC if o in organs])
        if verbose:
            print(f"[pbd] diaphragm: costal ring at band {mb}, dome apex +{H_DOME} bands")

    # -- attachments: anchor point(s) per organ on its wall. Bilateral organs (two bodies in
    # one fate group) get one anchor per side; the tether acts on the whole organ side (a
    # near-rigid body hangs from its ligament -- organ-level, not single-particle).
    BILATERAL = {"Kidney", "Lung"}
    anchors = []   # (particle_indices_of_side, tether_particle, anchor_xyz, rest, k)
    ody_all = base[:, 1] * s
    for oi, organ in enumerate(organs):
        if organ not in ANCHORS:
            continue
        frac, rlen, k_att = ANCHORS[organ]
        m = masks[organ]
        b = int(np.clip(apf[m].mean() * NB, 0, NB - 1))
        span = ohi[b] - olo[b]
        # NOTE (08-29, tried and REVERTED): deriving the dorsal-wall anchor from the
        # "inner face of the dorsal structural mass" fails at ne=30k -- the dorsal half
        # holds only spinal cord + skin shell (6 muscle cells at the kidney band): the
        # psoas/erector mass the kidney really rests on is NOT GROWN in this build. The
        # kidney's residual error is missing tissue, not solver physics.
        pi = np.where(org_of == oi)[0]
        if organ in BILATERAL:
            zmid = P[pi][:, 2].mean()
            sides = [pi[P[pi][:, 2] < zmid], pi[P[pi][:, 2] >= zmid]]
        else:
            sides = [pi]
        for side in sides:
            if len(side) < 4:
                continue
            a = np.array([P[side][:, 0].mean(),
                          (olo[b] + frac * span) * s,   # back to raw y
                          P[side][:, 2].mean()])
            j = side[np.argmin(np.linalg.norm(P[side] - a, axis=1))]
            anchors.append((side, j, a, rlen * span, k_att))

    # -- PBD relaxation with GROWTH ANNEALING: organs inflate 30% -> 100% during the solve
    # (as in development: small organs find their attachment homes first, growing volume
    # then locks the arrangement -- and it removes the jamming multi-stability)
    for it in range(iters):
        g = min(1.0, 0.3 + 0.7 * it / max(1, int(0.6 * iters)))
        # attachments (organ-level: the tether translates the hanging body, not one particle)
        for side, j, a, rl, k in anchors:
            d = P[j] - a; dist = np.linalg.norm(d)
            if dist > rl:
                corr = k * (dist - rl) * (d / dist)
                P[side] -= 0.15 * corr          # whole side drifts on its ligament
                P[j] -= corr                    # local pull at the attachment point
        # shape matching (soft rigid per organ, at current growth scale)
        for oi, organ in enumerate(organs):
            pi = org_of == oi
            q = rest[organ] * g; p = P[pi]
            c = p.mean(0)
            H = q.T @ (p - c)
            U, _, Vt = np.linalg.svd(H)
            Rm = (U @ Vt).T
            if np.linalg.det(Rm) < 0:
                U[:, -1] *= -1; Rm = (U @ Vt).T
            goal = c + q @ Rm.T
            P[pi] += 0.85 * (goal - p)
        # inter-organ contacts (radii at current growth scale, INFLATED by the cavity-fill
        # factor: the packing tissue between organs transmits the push)
        Rg = R * g
        Rgc = Rg * PACK
        tree = cKDTree(P)
        pairs = tree.query_pairs(2.0 * Rgc.max(), output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            diff = org_of[i] != org_of[j]
            i, j = i[diff], j[diff]
            d = P[i] - P[j]
            dist = np.linalg.norm(d, axis=1) + 1e-12
            pen = (Rgc[i] + Rgc[j]) - dist
            hit = pen > 0
            if hit.any():
                push = (0.5 * pen[hit] / dist[hit])[:, None] * d[hit]
                np.add.at(P, i[hit], push)
                np.add.at(P, j[hit], -push)
        # static body-wall collision (spine + back mass exclude the dorsal trunk)
        dist_s, si = stat_tree.query(P)
        pen_s = (Rg + r_stat) - dist_s
        hit_s = pen_s > 0
        if hit_s.any():
            d = P[hit_s] - wall[si[hit_s]]
            P[hit_s] += (pen_s[hit_s] / (np.linalg.norm(d, axis=1) + 1e-12))[:, None] * d
        # diaphragm dome: thoracic organs cranial of the sheet, abdominal viscera caudal.
        # The sheet: x_dia = costal ring + dome arching cranially toward the section centre.
        if dia_on:
            r2 = (((P[:, 1] * s) - dv_mid) / dv_half) ** 2 + ((P[:, 2] - z_mid) / z_half) ** 2
            x_dia = x_margin + H_DOME * dx_band * np.clip(1.0 - r2, 0.0, None)
            lowx = tho_pi & (P[:, 0] < x_dia)                # thoracic below the sheet
            P[lowx, 0] += K_DIA * (x_dia[lowx] - P[lowx, 0])
            hix = ~tho_pi & (P[:, 0] > x_dia)                # abdominal above the sheet
            P[hix, 0] += K_DIA * (x_dia[hix] - P[hix, 0])

        # peritoneal lining: intraperitoneal organs stay VENTRAL of the retro layer's face
        pb = np.clip(((P[:, 0] - base[:, 0].min()) / (np.ptp(base[:, 0]) + 1e-9) * NB)
                     .astype(int), 0, NB - 1)
        retro_pi = np.isin(org_of, [organs.index(o) for o in RETRO if o in organs])
        intra_pi = np.isin(org_of, [organs.index(o) for o in INTRA if o in organs])
        if retro_pi.any() and intra_pi.any():
            oyP = P[:, 1] * s
            for k in np.unique(pb[intra_pi]):
                if ribbed[k]:
                    continue    # belly bands only. (Tried extending under the costal margin
                                # -- the stomach bed relation -- but at ne=30k the stomach's
                                # band holds no pancreas cells so it never engages, and the
                                # extra coupling destabilized the thorax: 0.077/0.60 vs
                                # 0.069/0.66. Stomach = the open failure; needs the true
                                # stomach-bed contact or the diaphragm dome.)
                rm = retro_pi & (pb == k)
                if rm.sum() < 6:
                    continue
                rf = np.percentile(oyP[rm], 10)               # ventral face of the retro layer
                im = intra_pi & (pb == k) & (oyP > rf)
                oyP[im] += K_PERIT * (rf - oyP[im])
            P[:, 1] = oyP * s

        # containment in the trunk envelope -- dorsal HARD (spine/back mass), ventral
        # hard only in ribbed bands; unribbed bands get the compliant belly wall
        span_b = ohi[pb] - olo[pb]
        lo = olo[pb] + MARGIN * span_b; hi = ohi[pb] - MARGIN * span_b
        oy = np.minimum(P[:, 1] * s, hi)                     # dorsal wall: always rigid
        rigid_v = ribbed[pb]
        oy[rigid_v] = np.maximum(oy[rigid_v], lo[rigid_v])   # rib cage: rigid ventral wall
        below = ~rigid_v & (oy < lo)                         # compliant wall engaged
        oy[below] += K_WALL * (lo[below] - oy[below])        # soft restoring spring
        lo_soft = olo[pb] - BULGE * span_b                   # absolute bulge floor
        oy[~rigid_v] = np.maximum(oy[~rigid_v], lo_soft[~rigid_v])
        P[:, 1] = oy * s
        zspan = zhi[pb] - zlo[pb]
        P[:, 2] = np.clip(P[:, 2], zlo[pb] + MARGIN * zspan, zhi[pb] - MARGIN * zspan)

    # -- map back: per-organ rigid transform (initial particles -> final) applied to cells
    out = base.copy()
    for oi, organ in enumerate(organs):
        pi = org_of == oi
        p0 = P0[pi]; p1 = P[pi]
        c0, c1 = p0.mean(0), p1.mean(0)
        H = (p0 - c0).T @ (p1 - c1)
        U, _, Vt = np.linalg.svd(H)
        Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            U[:, -1] *= -1; Rm = (U @ Vt).T
        m = masks[organ]
        out[m] = (scr_cells[organ] - c0) @ Rm.T + c1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ne", type=int, default=30000)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from medic.adult_persistence_audit import build_base
    base, F = build_base(args.ne)
    tgt = _targets()
    masks = organ_groups(F, tgt)
    print(f"[pbd] {len(masks)} viscera groups: {sorted(masks)}")

    built = measure_dv(base, F, masks)              # the build's placed depths (~= targets)
    packed = pack(base, F, masks, iters=args.iters, seed=args.seed)
    emergent = measure_dv(packed, F, masks)

    rows = []
    print(f"{'organ':16s} {'measured':>8s} {'built':>7s} {'emergent':>8s} {'|err|':>6s}")
    errs = []
    for organ in sorted(masks):
        t = tgt[organ]; e = emergent[organ]
        err = abs(e - t); errs.append(err)
        rows.append({"organ": organ, "measured": round(t, 3), "built": round(built[organ], 3),
                     "emergent": round(e, 3), "abs_err": round(err, 3)})
        print(f"{organ:16s} {t:8.3f} {built[organ]:7.3f} {e:8.3f} {err:6.3f}")
    mean_err = float(np.mean(errs))
    # rank agreement: does the dorsal<->ventral ORDER emerge?
    ms = [tgt[o] for o in sorted(masks)]; es = [emergent[o] for o in sorted(masks)]
    rho = float(np.corrcoef(np.argsort(np.argsort(ms)), np.argsort(np.argsort(es)))[0, 1])
    print(f"[pbd] mean |err| = {mean_err:.3f}   DV rank correlation (emergent vs measured) = {rho:.2f}")

    try:
        from medic import gray_integrity_audit as GIA
        rows_a = GIA.audit(packed, F)
        npass = sum(1 for r in rows_a if r.get("ok"))
        print(f"[integrity after packing] {npass}/{len(rows_a)}")
    except Exception as ex:
        print(f"[integrity audit skipped: {ex}]")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump({"mean_abs_err": mean_err, "rank_corr": rho, "iters": args.iters,
               "anchors": {k: {"wall_frac": v[0], "rest": v[1], "k": v[2]}
                           for k, v in ANCHORS.items()},
               "rows": rows}, open(OUT_JSON, "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, cloud, title in ((ax[0], base, "built (dv_spread placement)"),
                            (ax[1], packed, "PBD equilibrium (scrambled start)")):
        bg = ~np.any([masks[o] for o in masks], axis=0)
        a.scatter(cloud[bg, 0], cloud[bg, 1], s=0.2, c="0.85")
        for organ in sorted(masks):
            m = masks[organ]
            a.scatter(cloud[m, 0], cloud[m, 1], s=0.8, label=organ)
        a.set_title(title); a.set_xlabel("AP"); a.set_ylabel("DV"); a.set_aspect("equal")
    ax[0].legend(fontsize=6, markerscale=6, loc="lower left")
    names = sorted(masks)
    yy = np.arange(len(names))
    ax[2].barh(yy - 0.2, [tgt[o] for o in names], height=0.4, label="measured (BP3D)")
    ax[2].barh(yy + 0.2, [emergent[o] for o in names], height=0.4, label="emergent (PBD)")
    ax[2].set_yticks(yy); ax[2].set_yticklabels(names, fontsize=8)
    ax[2].set_xlabel("local-trunk DV (0=ventral, 1=dorsal)")
    ax[2].set_title(f"mean |err| {mean_err:.3f}, rank corr {rho:.2f}")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110)
    print(f"[pbd] wrote {OUT_JSON} + {OUT_PNG}")


if __name__ == "__main__":
    main()
