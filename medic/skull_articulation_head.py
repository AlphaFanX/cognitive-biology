"""skull_articulation_head.py -- make the cranial vault a real interlocking dome, MOLDED BY THE BRAIN.
(2026-08-09, Miles) The vault bones (frontal/parietal/temporal/occipital) are labelled on a diffuse blob, so
they do not interlock (articulation ~44%) and the head reads funny. Rather than PLACE each bone at a measured
coordinate (a lookup that would just redraw the atlas), grow the vault as a thin shell around the brain: the
brain is the physical mold (its growth presses the vault outward in vivo), so the vault cells are projected onto
an ellipsoid hugging the brain, sized by the brain's own extent. The MEASURED BodyParts3D territories
(medic.skull_head.PROTO/COV, frontal .21 / parietals .46 / temporals .16 / occipital .18) then only decide which
patch of that dome is which bone -- the partition, not the position. This is the honest version: the shape is
molded by a body part (the brain), and measurement sets the labelling, not the geometry.

Read-only on the brain, eyes, face (viscerocranium), and body. Wired in build_base.

Run (self-test): cd cognimed && venv_win_new/Scripts/python.exe -m medic.skull_articulation_head
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX

BRAIN = ("Forebrain", "Midbrain", "Hindbrain", "Cerebellum")


def _ellip_R(dirs, a, rmax=None):
    R = 1.0 / np.sqrt(np.sum((dirs / a) ** 2, axis=1) + 1e-12)
    return np.clip(R, 0.0, rmax) if rmax is not None else R


def build(base, F, margin=0.08, thickness=0.04, densify=4, jitter=0.10, seed=0):
    """Grow the cranial vault as a DENSE shell molded by the brain. Returns (base, F): the vault cells are
    projected onto the brain-hugging ellipsoid and each is densified by tangential copies so the dome reads as a
    solid shell rather than a sparse scatter. Read-only on the brain/eyes/face/body."""
    base = np.asarray(base, float).copy(); F = np.asarray(F)
    bids = [FIDX[n] for n in BRAIN if n in FIDX]
    bm = np.isin(F, bids)
    if bm.sum() < 30:
        return base, F
    B = base[bm]; bc = B.mean(0)
    a = np.array([(np.percentile(B[:, k], 95) - np.percentile(B[:, k], 5)) * 0.5 for k in range(3)]) + 1e-6
    a = a * (1.0 + margin)                                               # the mold the vault hugs

    x = base[:, 0]; apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    cranial_floor = float(apf[bm].min()) - 0.02
    inner = [FIDX[n] for n in (*BRAIN, "Eye", "Retina", "OlfactoryBulb", "Otic") if n in FIDX]
    d0 = base - bc; r0 = np.linalg.norm(d0, axis=1)
    R0 = _ellip_R(d0 / (r0[:, None] + 1e-9), a)                          # each cell's shell radius in its direction
    # VAULT = cranial, non-inner, near/beyond the shell (the bone crust, not deep mesenchyme), superior dome only
    dv_ap = (base[:, 0] - bc[0]) / a[0]
    vault = (apf >= cranial_floor) & (~np.isin(F, inner)) & (dv_ap > -0.30) & (r0 > 0.55 * R0)
    if vault.sum() < 20:
        return base, F

    rng = np.random.default_rng(seed)
    idx = np.where(vault)[0]
    rmax = float(a.max()) * 1.15                                         # no cell flies past the brain-mold shell
    un = (base[idx] - bc) / (r0[idx][:, None] + 1e-9)
    R = _ellip_R(un, a, rmax)
    base[idx] = bc + un * (R * (1.0 + thickness * (rng.random(len(idx)) - 0.5)))[:, None]   # project onto the dome

    # DENSIFY: tangential copies on the shell so the dome is contiguous (the vault is genuinely thin -> resolution
    # matters here). For each vault cell, spread `densify` copies in the tangent plane and re-project to the shell.
    newP, newF = [], []
    for j, i in enumerate(idx):
        u = un[j]
        t1 = np.cross(u, [0, 0, 1.0]);
        if np.linalg.norm(t1) < 1e-6:
            t1 = np.cross(u, [0, 1.0, 0])
        t1 /= np.linalg.norm(t1) + 1e-9; t2 = np.cross(u, t1)
        ang = rng.random(densify) * 2 * np.pi; mag = jitter * np.sqrt(rng.random(densify))
        dirs = u[None] + mag[:, None] * (np.cos(ang)[:, None] * t1 + np.sin(ang)[:, None] * t2)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
        RR = _ellip_R(dirs, a, rmax) * (1.0 + thickness * (rng.random(densify) - 0.5))
        newP.append(bc + dirs * RR[:, None]); newF.append(np.full(densify, F[i]))
    if newP:
        base = np.vstack([base, np.vstack(newP)]); F = np.concatenate([F, np.concatenate(newF)])
    return base, F


def main():
    from medic.adult_persistence_audit import build_base
    from medic import gray_integrity_audit as GIA
    from medic import skull_head as SK
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    base, F = build_base(30000)
    out, Fout = build(base, F)
    print(f"[cells] {len(base)} -> {len(out)} (+{len(out)-len(base)} dome)")

    def artic(cloud, Fc):
        p = SK.build(cloud, Fc)["parts"]
        neuro = {k: v for k, v in p.items() if v.get("part") in ("frontal", "parietal", "temporal", "occipital")
                 or k in SK.NEURO}
        cents = {k: v["P"].mean(0) for k, v in neuro.items() if len(v.get("P", []))}
        if len(cents) < 3:
            return float("nan"), len(cents)
        C = np.array(list(cents.values()))
        gaps = []
        for i in range(len(C)):
            gaps.append(min(np.linalg.norm(C[i] - C[j]) for j in range(len(C)) if j != i))
        return float(np.mean(gaps)), len(cents)

    for tag, c, Fc in (("before", base, F), ("after", out, Fout)):
        rows = GIA.audit(c, Fc); npass = sum(1 for r in rows if r.get("ok"))
        fails = [r["relation"] for r in rows if not r.get("ok")]
        g, n = artic(c, Fc)
        print(f"[{tag}] integrity {npass}/{len(rows)}  vault-bone spacing {g:.3f} ({n} bones)"
              + (f"  FAIL {fails}" if fails else ""))

    x = base[:, 0]; stat = np.ptp(x); lo = x.min() + 0.80 * stat
    fig, ax = plt.subplots(1, 2, figsize=(7, 6), facecolor="#0d1017")
    for a_, (t, c) in zip(ax, (("before", base), ("after", out))):
        U = c[c[:, 0] >= lo]
        a_.set_facecolor("#0d1017"); a_.set_aspect("equal"); a_.axis("off")
        a_.scatter(U[:, 1], U[:, 0], s=4, c=("#f0a" if t == "before" else "#9fe6b0"), alpha=.5)  # DV vs AP (side)
        a_.set_title(f"{t} (side)", color="#cbd5e1")
    fig.suptitle("skull_articulation v2: dense brain-molded dome (head, side)", color="#e2e8f0")
    fig.savefig("data/organ_cascade/_skull.png", dpi=120, facecolor="#0d1017"); print("saved _skull.png")


if __name__ == "__main__":
    main()
