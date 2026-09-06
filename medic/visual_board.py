"""visual_board.py -- TIER D: the VISUAL benchmark (Miles 2026-08-30: "visual evaluation is now dominant").

Turns the visual audit into numbers, computed on the ADULT skin mesh straight from the shipped movie
frames (verts + per-frame topology), so every artifact class the eye found today has a detector:

  * shelf_score    -- spikes in the d(width)/d(AP) silhouette derivative (the hip-skirt class). Lower = better.
  * armpit_depth   -- the concavity between shoulder peak and arm: depth of the width-profile dip
                      below the shoulder (the cape class -- a cape fills the dip to ~0). Higher = better.
  * crotch_gap     -- fraction of the upper-thigh band whose midline is EMPTY of mesh (leg separation).
  * crown_taper    -- head-top width taper rate; a spike-crown tapers abruptly (high). Lower = better.
  * smoothness     -- 1 - variance of adjacent-face normal angles (surface sheen). Higher = better.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.visual_board
Out: data/organ_cascade/visual_board.json (+ printed scorecard). The proxies are the training loop's
visual objective; David's eyes on the shot board remain the final court.
"""
from __future__ import annotations
import json
import numpy as np


def _adult_skin():
    d = json.load(open("data/movie/human_movie_frames.json"))
    fr = next(f for f in reversed(d["frames"]) if f.get("phase") == "mesh" and f.get("skin"))
    V = np.asarray(fr["skin"], float).reshape(-1, 3)
    F = np.asarray(fr.get("skin_f") or d.get("skin_faces"), int)
    return V, F


def run():
    V, F = _adult_skin()
    ap = V[:, 0]; H = np.ptp(ap) + 1e-9
    apf = (ap - ap.min()) / H
    nb = 48
    width = np.zeros(nb)
    for k in range(nb):
        m = (apf >= k / nb) & (apf < (k + 1) / nb)
        if m.sum() > 6:
            width[k] = np.percentile(V[m, 2], 97) - np.percentile(V[m, 2], 3)
    width /= width.max() + 1e-9
    dw = np.abs(np.diff(width))
    shelf = float(np.sort(dw)[-3:].mean())                      # the 3 sharpest silhouette steps

    sh_band = slice(int(0.72 * nb), int(0.86 * nb))             # shoulders
    ax_band = slice(int(0.62 * nb), int(0.74 * nb))             # below them: the armpit dip
    armpit = float(max(0.0, width[sh_band].max() - width[ax_band].min()))

    thigh = (apf > 0.18) & (apf < 0.34)                         # upper thigh: is the midline empty?
    mid = np.abs(V[thigh, 2] - np.median(V[:, 2])) < 0.04 * np.ptp(V[:, 2])
    crotch = float(1.0 - mid.mean()) if thigh.sum() > 20 else 0.0

    crown = width[int(0.92 * nb):]
    crown_taper = float(np.abs(np.diff(crown)).max()) if len(crown) > 1 else 0.0

    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    smooth = float(1.0 - np.var(n, axis=0).sum() / 3.0)

    # ---- v2 detectors (2026-08-30, both v1 detectors were caught lying by the eye) ----
    # crotch_v2: the TRUE between-legs surface gap -- v1 counted off-midline verts, which a webbed crotch
    # still passed at 0.77 while the eye saw one fused column. Per thigh-band slice: verts split L/R of the
    # midline; gap = (min right ML) - (max left ML); report the fraction of slices with a POSITIVE gap.
    crotch_v2 = 0.0
    try:
        mid_ml = float(np.median(V[:, 2]))
        rows_open = []
        for k in range(int(0.10 * nb), int(0.34 * nb)):
            m = (apf >= k / nb) & (apf < (k + 1) / nb)
            if m.sum() < 12:
                continue
            ml = V[m, 2]
            L, Rr = ml[ml < mid_ml], ml[ml >= mid_ml]
            if len(L) > 5 and len(Rr) > 5:
                rows_open.append(1.0 if (np.percentile(Rr, 5) - np.percentile(L, 95)) > 0 else 0.0)
        crotch_v2 = float(np.mean(rows_open)) if rows_open else 0.0
    except Exception:
        pass
    # domeness: v1 crown_taper preferred the smooth CONE over the dome (it measures taper rate, not shape).
    # Fit the crown cap's radial profile r(h) to a hemisphere (r = R*sqrt(1-(h/H)^2)) vs a cone
    # (r = R*(1-h/H)); report SSE_cone / (SSE_cone + SSE_dome): > 0.5 = more dome than cone.
    domeness = 0.0
    try:
        head = V[apf > 0.90]
        if len(head) > 60:
            hx = head[:, 0]
            wl = np.array([np.percentile(np.abs(head[(hx >= q0) & (hx < q1), 2]
                                                 - np.median(head[:, 2])), 90)
                           if ((hx >= q0) & (hx < q1)).sum() > 4 else np.nan
                           for q0, q1 in zip(np.linspace(hx.min(), hx.max(), 13)[:-1],
                                             np.linspace(hx.min(), hx.max(), 13)[1:])])
            ok = ~np.isnan(wl)
            h = np.linspace(0, 1, 12)[ok]
            r = wl[ok] / (np.nanmax(wl) + 1e-9)
            sse_d = float(np.sum((r - np.sqrt(np.clip(1 - h ** 2, 0, 1))) ** 2))
            sse_c = float(np.sum((r - (1 - h)) ** 2))
            domeness = sse_c / (sse_c + sse_d + 1e-9)
    except Exception:
        pass

    out = dict(shelf_score=round(shelf, 3), armpit_depth=round(armpit, 3), crotch_gap=round(crotch, 3),
               crown_taper=round(crown_taper, 3), smoothness=round(smooth, 3),
               crotch_v2=round(crotch_v2, 3), domeness=round(domeness, 3),
               n_verts=int(len(V)), n_faces=int(len(F)))
    print("VISUAL BOARD (tier D) -- the adult skin, from the shipped frames")
    for k, v in out.items():
        print(f"  {k:14s} {v}")
    json.dump(out, open("data/organ_cascade/visual_board.json", "w"), indent=1)
    print("saved data/organ_cascade/visual_board.json")


if __name__ == "__main__":
    run()
