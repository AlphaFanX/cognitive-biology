"""dv_spread_head.py -- restore the dorso-ventral SPREAD of the viscera. (2026-08-09)

The assembled model bunches every viscus ventral (whole-body DV ~0.28-0.34, no spread), because the
placement had no dorso-ventral differentiation beyond the schematic. medic.bp3d_insitu_dv measured the REAL
in-situ depth of each organ from the BodyParts3D meshes in the canonical-human (FMA7163) frame, and it has a
genuine spread: the kidney, spleen, and lung sit DORSAL (retroperitoneal / posterior mediastinal), the heart,
liver, stomach, and small bowel sit VENTRAL. This head places each viscus at its MEASURED local-trunk depth.

It is the dorso-ventral analogue of the laterality head (which places each organ on its measured left/right
side): a placement driven by measured anatomy, read-only per organ, each organ moved as a coherent body so
its shape and its antero-posterior level are kept. The dorsal direction is read from the model's own spine
(the Spinal Cord fate is dorsal), so the head is self-orienting and makes no assumption about the y sign.
Wired in build_base after the organ shape/placement heads.

Run (self-test): cd cognimed && venv_win_new/Scripts/python.exe -m medic.dv_spread_head
"""
from __future__ import annotations
import json
import numpy as np

from medic.unified_embryo import FIDX

TARGETS_PATH = "data/organ_cascade/bp3d_insitu_dv.json"

# our fate -> the BodyParts3D-measured organ whose in-situ depth it should take (local-trunk DV, 0=ventral..1=dorsal)
DV_GROUP = {
    "Heart": "Heart", "Atrium": "Heart", "Ventricle": "Heart", "Outflow": "Heart",
    "Lung": "Lung",
    "Liver": "Liver", "LiverHaem": "Liver",
    "Kidney": "Kidney", "Nephron": "Kidney",
    "Pancreas": "Pancreas",
    "Spleen": "Spleen",
    "Foregut": "Stomach", "Stomach": "Stomach",
    "Gut": "SmallIntestine", "Hindgut": "LargeIntestine",
    "Bladder": "Bladder",
}


def _targets():
    d = json.load(open(TARGETS_PATH))["organs"]
    return {k: v["dv_local"] for k, v in d.items()}


def _dorsal_sign(base, F):
    """+1 if larger y is dorsal, -1 otherwise -- read from the spine (Spinal Cord is dorsal)."""
    sc = FIDX.get("Spinal Cord")
    y = base[:, 1]
    trunk_mid = float(np.median(y))
    if sc is not None:
        m = F == sc
        if m.sum() >= 8:
            return 1.0 if float(np.median(y[m])) > trunk_mid else -1.0
    noto = FIDX.get("Notochord")                                  # fallback: notochord is dorsal-axial
    if noto is not None and (F == noto).sum() >= 8:
        return 1.0 if float(np.median(y[F == noto])) > trunk_mid else -1.0
    return 1.0


def spread(base, F, strength=1.0, targets=None, verbose=False):
    """Move each viscus in DV to its measured local-trunk depth. Read-only on non-organ cells."""
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    tgt = targets if targets is not None else _targets()
    s = _dorsal_sign(base, F)
    x = base[:, 0]; oriented = base[:, 1] * s                     # oriented DV: larger = dorsal
    axial = ~_limb_mask(F)                                        # trunk+head reference for the DV envelope
    apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    nb = 24; band = np.clip((apf * nb).astype(int), 0, nb - 1)
    olo = np.array([np.percentile(oriented[axial & (band == k)], 5)
                    if (axial & (band == k)).sum() > 20 else oriented.min() for k in range(nb)])
    ohi = np.array([np.percentile(oriented[axial & (band == k)], 95)
                    if (axial & (band == k)).sum() > 20 else oriented.max() for k in range(nb)])

    # group our fates by the organ they target, so a multi-fate organ (heart, kidney) moves as one body
    groups = {}
    for fate, organ in DV_GROUP.items():
        fid = FIDX.get(fate)
        if fid is not None and organ in tgt:
            groups.setdefault(organ, []).append(fid)

    for organ, fids in groups.items():
        m = np.isin(F, fids)
        if m.sum() < 8:
            continue
        b = int(np.clip(apf[m].mean() * nb, 0, nb - 1))
        span = (ohi[b] - olo[b]) + 1e-9
        cur = (float(oriented[m].mean()) - olo[b]) / span
        target = float(np.clip(tgt[organ], 0.05, 0.95))
        d_or = (target - cur) * span * strength
        base[m, 1] += d_or * s                                    # shift back to raw y
        if verbose:
            print(f"  {organ:15s} local DV {cur:.2f} -> {target:.2f}  (shift {d_or:+.3f})")
    return base


def _limb_mask(F):
    ids = [FIDX[n] for n in ("Limb Bud",) if n in FIDX]
    return np.isin(F, ids) if ids else np.zeros(len(F), bool)


def main():
    from medic.adult_persistence_audit import build_base
    from medic import gray_integrity_audit as GIA
    base, F = build_base(30000)
    tgt = _targets()
    print(f"[targets] {tgt}")
    print(f"[dorsal sign] {_dorsal_sign(base, F):+.0f} (spine vs trunk-median y)")
    print("[before -> after] local-trunk DV per organ:")
    out = spread(base, F, verbose=True)
    for tag, cloud in (("before", base), ("after", out)):
        rows = GIA.audit(cloud, F); npass = sum(1 for r in rows if r.get("ok"))
        fails = [f"{r['relation']} ({r['value']})" for r in rows if not r.get("ok")]
        print(f"[integrity {tag}] {npass}/{len(rows)}" + (f"  FAIL: {fails}" if fails else ""))


if __name__ == "__main__":
    main()
