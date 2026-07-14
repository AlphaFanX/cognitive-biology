"""
limb_shape -- the proximodistal limb CLOCK (segments) + tightening of limbs and body.
=====================================================================================

The trunk is segmented by the her1 somite clock along the AP axis; a limb is segmented the same
way along its OWN proximodistal (PD) axis. As the limb bud extends, a distal timer lays down three
segments -- STYLOPOD (upper arm / thigh) -> ZEUGOPOD (fore-arm / shank) -> AUTOPOD (hand / foot) --
with a JOINT (constriction) at each boundary. The segment count (3) and proportions are the genome's
limb program (here fixed to the tetrapod plan); the clock just reads them out along the PD axis.

Also tightens the rough cell cloud: each limb is compacted toward its own PD axis and tapered; the
trunk is drawn toward its per-slice medial axis so the body reads as a solid form rather than a haze.

Pure geometry on a grown body (medic.unified_embryo): does not change cell fate or limb count.
"""
from __future__ import annotations
import numpy as np

# PD segment boundaries (fractions of limb length) and joint positions.
SEG_BOUNDS = (0.44, 0.80)          # stylopod | zeugopod | autopod (autopod = compact distal plate)
JOINTS = (0.44, 0.80)              # shoulder/elbow-ish constrictions
SEG_NAMES = ("stylopod", "zeugopod", "autopod")


def tighten_body(pos, fid, limb_id, keep=0.72, nbins=44):
    """Draw the trunk toward its per-AP-slice medial axis so the body is a solid form, not a haze.
    Limb cells are left alone (they get their own tightening)."""
    out = pos.copy()
    body = fid != limb_id
    a = (pos[:, 0] - pos[:, 0].min()) / (np.ptp(pos[:, 0]) + 1e-9)
    b = np.clip((a * nbins).astype(int), 0, nbins - 1)
    for k in range(nbins):
        m = body & (b == k)
        if m.sum() < 5:
            continue
        c = pos[m][:, 1:].mean(0)
        out[np.where(m)[0][:, None], [1, 2]] = c + (pos[m][:, 1:] - c) * keep
    return out


def segment_limbs(pos, fid, limb_id, fore_ap, hind_ap):
    """Tighten each of the four limbs to its PD axis, taper it, add joint constrictions, and tag each
    cell with a PD segment id (0 stylopod / 1 zeugopod / 2 autopod; -1 = not a limb).

    Returns (new_pos, seg) with seg an int array over all cells.
    """
    out = pos.copy()
    seg = np.full(len(pos), -1, np.int8)
    isb = fid == limb_id
    if not isb.any():
        return out, seg
    a = (pos[:, 0] - pos[:, 0].min()) / (np.ptp(pos[:, 0]) + 1e-9)
    apmid = 0.5 * (fore_ap + hind_ap)
    zc_all = pos[:, 2]
    for apm in (a < apmid, a >= apmid):                        # fore / hind
        for side in (zc_all > 0, zc_all < 0):                  # left / right
            m = isb & apm & side
            if m.sum() < 8:
                continue
            idx = np.where(m)[0]
            L = pos[idx].copy()
            # PD axis: |z| from the proximal root (near the body) to the distal tip.
            zc = np.abs(L[:, 2])
            z0, z1 = np.percentile(zc, 5), np.percentile(zc, 98)
            pd = np.clip((zc - z0) / (z1 - z0 + 1e-9), 0.0, 1.0)   # 0 root .. 1 tip
            xc = np.median(L[:, 0]); yc0 = np.median(L[:, 1])
            # TIGHTEN: compact the AP (x) cross-section toward the limb line, and taper distally.
            taper = 1.0 - 0.45 * pd
            L[:, 0] = xc + (L[:, 0] - xc) * 0.42 * taper
            # keep a mild dorsoventral thickness that also tapers
            L[:, 1] = yc0 + (L[:, 1] - yc0) * (0.6 * taper)
            # JOINT constrictions: pinch the cross-section at each PD boundary -> visible joints.
            for jb in JOINTS:
                pinch = 1.0 - 0.55 * np.exp(-((pd - jb) / 0.045) ** 2)
                L[:, 0] = xc + (L[:, 0] - xc) * pinch
                L[:, 1] = yc0 + (L[:, 1] - yc0) * pinch
            # AUTOPOD flare: the hand/foot plate widens slightly at the very tip.
            flare = 1.0 + 0.9 * np.clip((pd - 0.85) / 0.15, 0, 1)
            L[:, 0] = xc + (L[:, 0] - xc) * flare
            seg[idx] = np.digitize(pd, SEG_BOUNDS).astype(np.int8)   # 0,1,2
            out[idx] = L
    return out, seg


def shape(pos, fid, limb_id, fore_ap, hind_ap, tighten=True):
    """Convenience: tighten the body, then segment+tighten the limbs. Returns (pos, seg)."""
    p = tighten_body(pos, fid, limb_id) if tighten else pos
    return segment_limbs(p, fid, limb_id, fore_ap, hind_ap)
