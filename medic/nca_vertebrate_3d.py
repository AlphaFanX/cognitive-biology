#!/usr/bin/env python3
"""
3D NCA+LGM forward run -> the vertebrate Bauplan, with the her1 clock LIVE.
==========================================================================

The body field is grown by the SAME local NCA rule over a 3D vertebrate body:

    dV/dt = k_relax*(V_target(t) - V) + k_gj*laplacian_3d(V)   (no-flux in body)

Two axes are combined:

  DV axis : the germ-layer voltage floor (kernel-derived; dorsal/neural
            hyperpolarized -> ventral/gut depolarized).
  AP axis : the conserved program -- Wnt/Hox head->tail polarity, a
            hyperpolarized dorsal-midline neural tube, an anterior-ventral
            depolarized heart field, AND the somites.

The somites are NOT painted at precomputed centers. They are produced by running
the her1 segmentation clock (Her1Oscillator delayed-feedback period T) against a
posteriorly regressing determination wavefront (SegmentationClock, S = v*T)
THROUGH the NCA's developmental time: as t advances the front regresses, the PSM
oscillates ahead of it, and one somite freezes per clock period -- so the trunk
segments anterior->posterior while the NCA relaxes onto the moving target.

Run (clock-driven somitogenesis movie + final Bauplan, the default):
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.nca_vertebrate_3d
Outputs: nca_somitogenesis.gif, nca_somitogenesis_kymograph.png,
         nca_vertebrate_3d.png, nca_vertebrate_3d_sagittal.png
"""
from __future__ import annotations

import numpy as np

try:
    from .tissue.genomic_nca import GenomicNCA, V_ZYGOTE
    from .zebrafish_somitogenesis import Her1Oscillator, SegmentationClock, V_SURROUND
except ImportError:  # pragma: no cover
    from medic.tissue.genomic_nca import GenomicNCA, V_ZYGOTE
    from medic.zebrafish_somitogenesis import Her1Oscillator, SegmentationClock, V_SURROUND

# Grid (AP x DV x LR). Margin of empty voxels keeps the body off the grid edge.
NAP, NDV, NLR = 124, 34, 26
TRUNK_START, TRUNK_END = 0.24, 0.82
N_SOMITES = 14
# Hox-derived limb AP levels (body_plan_morphogenesis.hox_limb_levels, from HOX
# cluster colinearity): forelimb = Hox6 anterior boundary, hindlimb = Hox10.
FORELIMB_AP, HINDLIMB_AP = 0.543, 0.767


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _grids():
    ap = np.linspace(0, 1, NAP)[:, None, None]
    dv = np.linspace(0, 1, NDV)[None, :, None]
    lr = np.linspace(-1, 1, NLR)[None, None, :]
    return np.broadcast_arrays(ap, dv, lr)


def body_mask(ap, dv, lr):
    """Vertebrate body: spans the full DV column at every AP station; only the
    LR width tapers (rounded head lobe -> trunk -> thin tail). Cross-section is
    an ellipse in LR that closes toward the dorsal and ventral edges, so axial
    and near-axial structures (neural tube, somites, gut) stay inside the whole
    trunk."""
    head = 0.55 * np.exp(-((ap - 0.12) / 0.12) ** 2)              # head LR lobe
    trunk = 0.50 * np.clip(1.0 - (ap - 0.12) / 1.15, 0.0, 1.0)    # LR taper to tail
    half_w = np.maximum.reduce([head, trunk, np.where(ap < 0.97, 0.10, 0.0)])
    dv_off = (dv - 0.5) / 0.5                                     # -1 ventral .. +1 dorsal
    lr_max = half_w * np.sqrt(np.clip(1.0 - dv_off ** 2, 0.0, 1.0))
    base = (np.abs(lr) <= lr_max) & (half_w > 0.04)
    # Limb-bud protrusions: lateral outgrowths at the two Hox AP levels
    # (forelimb Hox6 ~0.543, hindlimb Hox10 ~0.767), bilateral, at lateral-plate DV.
    buds = None
    for ap0 in (FORELIMB_AP, HINDLIMB_AP):
        for s in (-1.0, 1.0):
            e = (((ap - ap0) / 0.05) ** 2 + ((dv - 0.48) / 0.13) ** 2
                 + ((lr - s * 0.45) / 0.13) ** 2) <= 1.0
            buds = e if buds is None else (buds | e)
    return base | buds


def _paraxial(DV, LR):
    """Paraxial trunk band beside the axis (bilateral, near-midline) where the
    somites form -- kept close to the axis so it stays inside the tapering body."""
    return (np.exp(-((DV - 0.64) / 0.09) ** 2)
            * (np.exp(-((LR - 0.16) / 0.10) ** 2) + np.exp(-((LR + 0.16) / 0.10) ** 2)))


def _organ_rd_fields():
    """Eyes / heart / neural-tube fields from morphogen reaction-diffusion
    (medic.morphogen_rd), resampled onto the body grid as (NAP, NLR), max-1
    normalized. These REPLACE the prescribed Gaussian source terms."""
    try:
        from . import morphogen_rd as mrd
    except ImportError:  # pragma: no cover
        from medic import morphogen_rd as mrd
    _, neural = mrd.neural_tube()
    eyes = mrd.eye_field(split=True)
    heart = mrd.heart_field()
    limbs = mrd.limb_field()

    def rs(f):                                   # f indexed [lr, ap] -> (NAP, NLR)
        lr_idx = np.round(np.linspace(0, mrd.NLR - 1, NLR)).astype(int)
        ap_idx = np.round(np.linspace(0, mrd.NAP - 1, NAP)).astype(int)
        out = f[np.ix_(lr_idx, ap_idx)].T
        m = out.max()
        return out / m if m > 0 else out

    return dict(neural=rs(neural), eyes=rs(eyes), heart=rs(heart), limbs=rs(limbs))


def build_static_target(use_rd=True, ectopic_eye_ap=None):
    """Genome-sourced 3D target WITHOUT the somites (DV floor + AP program).

    use_rd=True : eyes/heart/neural-tube placed by morphogen reaction-diffusion
                  (emergent). use_rd=False : the prescribed Gaussian placeholders.
    ectopic_eye_ap : if set (an AP coordinate, e.g. 0.88 = tail), add a
                  hyperpolarizing eye source at that station -- the Levin
                  bioelectric instruction placing an ectopic eye outside the
                  anterior eye field (see medic.ectopic_eye).
    """
    AP, DV, LR = _grids()

    # DV germ-layer floor from the kernel (dorsal hyperpol -> ventral depol).
    nca2d = GenomicNCA(height=NDV, width=8, use_real_strata=True)
    dv_prepattern = nca2d.V_adult_target.mean(axis=1)
    if dv_prepattern[-1] > dv_prepattern[0]:
        dv_prepattern = dv_prepattern[::-1]
    Vdv = dv_prepattern[None, :, None] * np.ones_like(AP)

    # AP polarity (head depolarized -> tail hyperpolarized): the validated floor.
    Vap = 16.0 * (0.5 - AP)
    T = Vdv + Vap

    # Nodal/Activin ventral gradient. The kernel Vm floor resolves ectoderm/neural
    # but leaves mesoderm and endoderm nearly isopotential (the phylotypic_form
    # finding); the conserved Nodal/Activin gradient (high ventral) depolarizes
    # endoderm beyond mesoderm and so resolves the two. (De Robertis & Sasai 1996;
    # Schier, Nodal signaling.)
    T = T + 8.0 * np.clip((0.40 - DV) / 0.40, 0.0, 1.0)

    # Organ source terms (neural tube / heart / eyes). The Gaussians are
    # placeholders; their mechanistic origin is reaction-diffusion. With use_rd
    # the emergent morphogen_rd fields supply the AP/LR placement and only the
    # DV level is banded; otherwise the prescribed Gaussians are used.
    if use_rd:
        rd = _organ_rd_fields()
        T = T + (-9.0) * rd["neural"][:, None, :] * np.exp(-((DV - 0.82) / 0.10) ** 2)
        T = T + (-6.0) * rd["eyes"][:, None, :] * np.exp(-((DV - 0.70) / 0.10) ** 2)
        # heart sits in cardiac mesoderm (DV~0.42), amplitude trimmed so its local
        # depolarization does not lift the meso-band germ-layer statistics.
        T = T + 8.0 * rd["heart"][:, None, :] * np.exp(-((DV - 0.42) / 0.08) ** 2)
        # limb buds (lateral-plate mesoderm, DV~0.48), Hox-gated placement from RD.
        T = T + 7.0 * rd["limbs"][:, None, :] * np.exp(-((DV - 0.48) / 0.13) ** 2)
    else:
        T = T + (-9.0) * np.exp(-((DV - 0.82) / 0.10) ** 2) * np.exp(-(LR / 0.22) ** 2)
        T = T + 12.0 * np.exp(-((AP - 0.27) / 0.05) ** 2) * np.exp(-((DV - 0.34) / 0.10) ** 2) \
            * np.exp(-(LR / 0.30) ** 2)
        T = T + (-6.0) * np.exp(-((AP - 0.07) / 0.03) ** 2) * np.exp(-((DV - 0.70) / 0.10) ** 2) \
            * (np.exp(-((LR - 0.40) / 0.14) ** 2) + np.exp(-((LR + 0.40) / 0.14) ** 2))
        # limb buds: 4 Gaussian peaks = 2 Hox AP levels x 2 (L/R), lateral-plate DV.
        for ap0 in (FORELIMB_AP, HINDLIMB_AP):
            for s in (-1.0, 1.0):
                T = T + 7.0 * np.exp(-((AP - ap0) / 0.05) ** 2) \
                    * np.exp(-((DV - 0.48) / 0.13) ** 2) * np.exp(-((LR - s * 0.45) / 0.13) ** 2)

    # Ectopic eye (Pai-Levin 2012): a hyperpolarizing eye source imposed at an
    # off-target AP station. The SAME eye signature (hyperpolarized, DV~0.70) as
    # the anterior eyes -- only its position differs, because the eye is a
    # relocatable attractor placed by the field, not a genomic AP address.
    if ectopic_eye_ap is not None:
        # stronger than the anterior eyes' -6 mV: the tail is thin, so gap-junction
        # diffusion bleeds a local source more than in the broad head -- the eye
        # signature is the same, the instruction is scaled for the tail geometry.
        T = T + (-12.0) * np.exp(-((AP - ectopic_eye_ap) / 0.03) ** 2) \
            * np.exp(-((DV - 0.70) / 0.10) ** 2) * np.exp(-(LR / 0.18) ** 2)

    mask = body_mask(AP, DV, LR)
    paraxial = _paraxial(DV, LR)
    return T, mask, AP, DV, LR, paraxial


# ---------------------------------------------------------------------------
# The her1 clock (Cooke-Zeeman) driving the somite field over time
# ---------------------------------------------------------------------------
def get_clock():
    """her1 oscillator -> emergent period T -> segmentation clock (S = v*T)."""
    T, _sol = Her1Oscillator().period()
    clock = SegmentationClock(period_min=T, n_somites=N_SOMITES,
                              wavefront_velocity=1.0 / N_SOMITES)  # somites span [0,1]
    return T, clock


def somite_ap_center(k, clock):
    """Body-AP coordinate of somite k (mapped into the trunk band)."""
    span = TRUNK_END - TRUNK_START
    return TRUNK_START + (k + 0.5) * clock.S * span


def front_ap(t, clock):
    """Determination-front body-AP coordinate at developmental time t."""
    span = TRUNK_END - TRUNK_START
    return TRUNK_START + min(clock.v * (t / clock.T), 1.0) * span


def somite_field_dynamic(t, AP, paraxial, clock):
    """Somite Vmem contribution at developmental time t.

    Frozen somites (anterior to the front) carry their Silic-matched, maturation-
    dependent hyperpolarization; the PSM (posterior to the front) carries the
    travelling her1 oscillation. The pattern grows anterior->posterior as t rises.
    """
    span = TRUNK_END - TRUNK_START
    ap1d = AP[:, 0, 0]
    stripe = np.zeros_like(ap1d)

    live = [s for s in clock.somites if s.freeze_time <= t]
    max_age = max((t - s.freeze_time for s in live), default=1.0) or 1.0
    for k, s in enumerate(clock.somites):
        if s.freeze_time > t:
            continue
        c = somite_ap_center(k, clock)
        u = (t - s.freeze_time) / max_age
        mean, _amp = clock.somite_vmem(u)
        rel = 0.7 * (mean - V_SURROUND)        # hyperpolarized vs surround
        stripe = stripe + rel * np.exp(-((ap1d - c) / 0.010) ** 2)

    # PSM travelling wave ahead of the front (the clock visibly ticking).
    fr = front_ap(t, clock)
    ahead = (ap1d > fr) & (ap1d < TRUNK_END)
    if ahead.any():
        ap01 = (ap1d - TRUNK_START) / span
        fr01 = (fr - TRUNK_START) / span
        phase = 2 * np.pi * (t / clock.T) - 2 * np.pi * (ap01 - fr01) / clock.S
        stripe = stripe + np.where(ahead, -5.0 * np.cos(phase), 0.0)

    return stripe[:, None, None] * paraxial


def laplacian_noflux(V, mask):
    """6-neighbour Laplacian with no-flux (only flux between in-body voxels)."""
    lap = np.zeros_like(V)
    for axis in (0, 1, 2):
        for shift in (1, -1):
            nb = np.roll(V, shift, axis=axis)
            valid = np.roll(mask, shift, axis=axis) & mask
            lap += np.where(valid, nb - V, 0.0)
    return lap


def run_clock(steps=130, k_relax=0.14, k_gj=0.045, n_capture=40, ectopic_eye_ap=None):
    """Forward NCA driven through developmental time by the her1 clock."""
    T_static, mask, AP, DV, LR, paraxial = build_static_target(ectopic_eye_ap=ectopic_eye_ap)
    period, clock = get_clock()
    t_max = clock.n * clock.T                    # all somites frozen by t_max

    V = np.full(T_static.shape, V_ZYGOTE, dtype=float)
    cap_steps = set(np.linspace(0, steps, n_capture).astype(int).tolist())
    frames, kymo, kymo_t, kymo_front = [], [], [], []

    # paraxial sampling bands for the dorsal projection / kymograph
    dv_band = (DV[0, :, 0] > 0.56) & (DV[0, :, 0] < 0.72)
    lr_pos = (LR[0, 0, :] > 0.06) & (LR[0, 0, :] < 0.30)

    for s in range(steps + 1):
        t = (s / steps) * t_max
        target = T_static + somite_field_dynamic(t, AP, paraxial, clock)
        if s in cap_steps:
            Vm = np.where(mask, V, np.nan)
            with np.errstate(invalid="ignore"):
                slab = np.nanmean(Vm[:, dv_band, :], axis=1)    # (NAP, NLR) dorsal view
                prof = np.nanmean(Vm[:, dv_band, :][:, :, lr_pos], axis=(1, 2))
            frames.append((slab.copy(), t, front_ap(t, clock),
                           sum(1 for so in clock.somites if so.freeze_time <= t)))
            kymo.append(prof); kymo_t.append(t); kymo_front.append(front_ap(t, clock))
        dV = k_relax * (target - V) + k_gj * laplacian_noflux(V, mask)
        V = V + np.where(mask, dV, 0.0)

    V = np.where(mask, V, np.nan)
    return dict(V=V, mask=mask, clock=clock, period=period, t_max=t_max,
                frames=frames, kymo=np.array(kymo), kymo_t=np.array(kymo_t),
                kymo_front=np.array(kymo_front))


# ---------------------------------------------------------------------------
# Bauplan scoring (read out of the produced 3D field)
# ---------------------------------------------------------------------------
def bauplan_checklist(V):
    ap = np.linspace(0, 1, NAP)[:, None, None] * np.ones_like(V)
    dv = np.linspace(0, 1, NDV)[None, :, None] * np.ones_like(V)
    lr = np.linspace(-1, 1, NLR)[None, None, :] * np.ones_like(V)
    fin = np.isfinite(V)

    def mw(cond):
        sel = fin & cond
        return float(np.nanmean(V[sel])) if sel.any() else np.nan

    checks = []
    checks.append(("AP axis: head depolarized anterior of hyperpolarized tail",
                   mw(ap < 0.18) > mw(ap > 0.85)))
    checks.append(("DV axis: neural (dorsal) hyperpolarized of gut (ventral)",
                   mw(dv > 0.72) < mw(dv < 0.30)))
    e, m, n = mw(dv > 0.66), mw((dv >= 0.33) & (dv <= 0.66)), mw(dv < 0.33)
    checks.append(("germ layers DV-ordered (ecto < meso < endo in Vmem)", e < m < n))
    checks.append(("dorsal-midline neural tube hyperpolarized vs dorsal-lateral",
                   mw((dv > 0.74) & (np.abs(lr) < 0.12)) < mw((dv > 0.74) & (np.abs(lr) > 0.25))))
    band = fin & (dv > 0.56) & (dv < 0.72) & (lr > 0.06) & (lr < 0.30)
    prof = np.array([np.nanmean(V[i][band[i]]) if band[i].any() else np.nan
                     for i in range(NAP)])
    pf = prof[np.isfinite(prof)]
    dips = int(np.sum((pf[1:-1] < pf[:-2]) & (pf[1:-1] < pf[2:]))) if pf.size > 3 else 0
    checks.append((f"metameric somites along paraxial band ({dips} periodic minima >= 8)",
                   dips >= 8))
    Vsym = V[:, :, ::-1]
    both = np.isfinite(V) & np.isfinite(Vsym)
    asym = float(np.nanmean(np.abs(V[both] - Vsym[both])))
    checks.append((f"bilateral (L-R) symmetry (mean |dV| = {asym:.1f} mV < 3)", asym < 3.0))
    checks.append(("anterior-ventral depolarized heart field",
                   mw((ap > 0.18) & (ap < 0.36) & (dv < 0.45))
                   > mw((ap > 0.18) & (ap < 0.36) & (dv > 0.45))))
    # two bilateral limb buds at the Hox AP levels (fore Hox6 anterior of hind Hox10):
    # the lateral-plate bud is depolarized vs the axial mesoderm AT THE SAME AP LEVEL
    # (comparing at-level removes the AP-gradient confound).
    def bud_vs_axial(a_lo, a_hi):
        bud = mw((ap > a_lo) & (ap < a_hi) & (dv > 0.40) & (dv < 0.56) & (np.abs(lr) > 0.35))
        axial = mw((ap > a_lo) & (ap < a_hi) & (dv > 0.40) & (dv < 0.56) & (np.abs(lr) < 0.22))
        return np.isfinite(bud) and np.isfinite(axial) and bud > axial
    checks.append(("two limb buds (Hox6 fore + Hox10 hind), depolarized lateral plate",
                   bud_vs_axial(0.49, 0.60) and bud_vs_axial(0.72, 0.82)))
    return checks


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_3d_and_sagittal(V, clock):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    ii, jj, kk = np.where(np.isfinite(V))
    ap = ii / (NAP - 1); dv = jj / (NDV - 1); lr = (kk / (NLR - 1)) * 2 - 1
    vals = V[ii, jj, kk]
    vmin, vmax = np.nanpercentile(V, 2), np.nanpercentile(V, 98)

    fig = plt.figure(figsize=(15, 6))
    for col, (az, el, ttl) in enumerate([(-60, 18, "lateral / 3-quarter"),
                                         (-90, 90, "dorsal (top-down)")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        p = ax.scatter(ap, lr, dv, c=vals, cmap="viridis", vmin=vmin, vmax=vmax,
                       s=6, alpha=0.55, edgecolors="none")
        ax.set_xlabel("anterior -> posterior (AP)"); ax.set_ylabel("L <- LR -> R")
        ax.set_zlabel("ventral -> dorsal (DV)")
        ax.set_title(f"NCA+LGM 3D Vmem  ({ttl})"); ax.view_init(elev=el, azim=az)
        try:
            ax.set_box_aspect((3.0, 1.1, 1.1))
        except Exception:
            pass
    cb = fig.colorbar(p, ax=fig.axes, fraction=0.015, pad=0.02); cb.set_label("Vmem (mV)")
    fig.suptitle("3D NCA+LGM forward run = the vertebrate Bauplan (Vmem heatmap)",
                 fontsize=13, y=0.98)
    fig.savefig("nca_vertebrate_3d.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print("Saved: nca_vertebrate_3d.png")

    mid = NLR // 2
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    im = ax2.imshow(V[:, :, mid].T, origin="lower", aspect="auto", cmap="viridis",
                    extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    ax2.set_xlabel("anterior <- AP -> posterior"); ax2.set_ylabel("ventral <- DV -> dorsal")
    ax2.set_title("Midsagittal Vmem slice: neural tube dorsal, gut ventral, head anterior, tail posterior")
    fig2.colorbar(im, ax=ax2, fraction=0.025, label="Vmem (mV)")
    fig2.tight_layout(); fig2.savefig("nca_vertebrate_3d_sagittal.png", dpi=120, bbox_inches="tight")
    plt.close(fig2); print("Saved: nca_vertebrate_3d_sagittal.png")


def render_somitogenesis(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    frames = res["frames"]; clock = res["clock"]
    allslab = np.concatenate([f[0][np.isfinite(f[0])] for f in frames])
    vmin, vmax = np.nanpercentile(allslab, 2), np.nanpercentile(allslab, 98)

    imgs = []
    for slab, t, fr, nsom in frames:
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.imshow(slab.T, origin="lower", aspect="auto", cmap="viridis",
                  extent=[0, 1, -1, 1], vmin=vmin, vmax=vmax)
        ax.axvline(fr, color="white", lw=2, alpha=0.8)            # determination front
        ax.text(fr + 0.01, 0.8, "front", color="white", fontsize=8)
        ax.set_xlabel("anterior -> posterior (AP)"); ax.set_ylabel("L <- LR -> R")
        ax.set_title(f"her1 clock in the NCA: dorsal view   t={t:.0f} min   "
                     f"somites formed: {nsom}/{clock.n}")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=92, bbox_inches="tight")
        plt.close(fig); buf.seek(0); imgs.append(Image.open(buf).convert("RGB"))
    imgs[-1:] = imgs[-1:] * 8                                     # hold final frame
    imgs[0].save("nca_somitogenesis.gif", save_all=True, append_images=imgs[1:],
                 duration=120, loop=0)
    print("Saved: nca_somitogenesis.gif")

    # kymograph: AP x developmental time of the paraxial Vm + the front line.
    fig, ax = plt.subplots(figsize=(11, 5))
    K = res["kymo"]
    im = ax.imshow(K, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, 1, res["kymo_t"][0], res["kymo_t"][-1]], vmin=vmin, vmax=vmax)
    ax.plot(res["kymo_front"], res["kymo_t"], color="white", lw=2, label="determination front")
    for k in range(clock.n):
        ax.plot(somite_ap_center(k, clock), (k + 1) * clock.T, "o", color="#39d353", ms=4)
    ax.set_xlabel("anterior -> posterior (AP)"); ax.set_ylabel("developmental time (min)")
    ax.set_title("Somitogenesis kymograph from the NCA run: front regresses, one somite per her1 period")
    ax.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, label="paraxial Vmem (mV)")
    fig.tight_layout(); fig.savefig("nca_somitogenesis_kymograph.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("Saved: nca_somitogenesis_kymograph.png")


def run_ectopic_demo(ectopic_ap=0.88):
    """Run the whole-body NCA with and without the tail eye instruction and show
    the ectopic eye in the 3D Vmem field (the body-level companion of
    medic.ectopic_eye)."""
    print("=" * 72)
    print("ECTOPIC EYE IN THE 3D BODY  --  relocatable attractor (Pai-Levin 2012)")
    print("=" * 72)
    wt = run_clock()
    ec = run_clock(ectopic_eye_ap=ectopic_ap)

    # tail eye = hyperpolarized spot at the instruction site vs the same site WT.
    ap = np.linspace(0, 1, NAP)[:, None, None] * np.ones_like(wt["V"])
    dv = np.linspace(0, 1, NDV)[None, :, None] * np.ones_like(wt["V"])
    lr = np.linspace(-1, 1, NLR)[None, None, :] * np.ones_like(wt["V"])
    site = (np.abs(ap - ectopic_ap) < 0.03) & (dv > 0.60) & (dv < 0.80) & (np.abs(lr) < 0.12)

    def m(V, c):
        sel = np.isfinite(V) & c
        return float(np.nanmean(V[sel])) if sel.any() else np.nan
    v_wt, v_ec = m(wt["V"], site), m(ec["V"], site)
    print(f"\n  Vmem at the tail site (AP={ectopic_ap}):")
    print(f"    WT       = {v_wt:+.1f} mV  (ordinary tail tissue)")
    print(f"    ectopic  = {v_ec:+.1f} mV  (hyperpolarized eye spot)")
    print(f"    delta    = {v_ec - v_wt:+.1f} mV  -> a 3rd eye where the field instructs it")

    # also confirm the rest of the body is unchanged (the instruction is local)
    elsewhere = np.isfinite(wt["V"]) & np.isfinite(ec["V"]) & ~site
    drift = float(np.nanmean(np.abs(ec["V"][elsewhere] - wt["V"][elsewhere])))
    print(f"    body elsewhere mean |dV| = {drift:.2f} mV (instruction is local)")

    render_3d_and_sagittal(ec["V"], ec["clock"])
    import os
    for src, dst in [("nca_vertebrate_3d.png", "nca_ectopic_eye_3d.png"),
                     ("nca_vertebrate_3d_sagittal.png", "nca_ectopic_eye_sagittal.png")]:
        if os.path.exists(src):
            os.replace(src, dst); print(f"  saved {dst}")


def main():
    print("=" * 72)
    print("3D NCA+LGM FORWARD RUN with the her1 CLOCK -> vertebrate Bauplan")
    print("=" * 72)
    res = run_clock()
    V, clock = res["V"], res["clock"]
    print(f"her1 emergent period T = {res['period']:.1f} min;  "
          f"{clock.n} somites; S = v*T = {clock.S:.4f} AP/period")
    print(f"developmental time run 0 -> {res['t_max']:.0f} min over the NCA forward pass")
    print(f"Body volume: {int(np.isfinite(V).sum())} voxels; "
          f"Vmem range [{np.nanmin(V):+.1f}, {np.nanmax(V):+.1f}] mV")

    print("\nFinal vertebrate Bauplan checklist (read out of the 3D field):")
    checks = bauplan_checklist(V)
    for name, ok in checks:
        print(f"   [{'OK' if ok else 'XX'}] {name}")
    print(f"\n   {sum(ok for _, ok in checks)}/{len(checks)} Bauplan features present.")

    render_somitogenesis(res)
    render_3d_and_sagittal(V, clock)


if __name__ == "__main__":
    import sys
    if "--ectopic" in sys.argv:
        run_ectopic_demo()
    else:
        main()
