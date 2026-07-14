"""
The fine model wired in: grow the body with the real NCA (unified_embryo), then the Vm field places
the tissues and restriction gives the coarse organs.

Where grow_from_kernel used a genome-parametric body domain, this uses the actual cell-by-cell NCA
growth of the companion embryo paper (medic/unified_embryo: division + the four heads + cohesion +
bilateral symmetry) as the genuine fine model. The grown cell cloud is then given a medial-axis
frame (AP + radial), the radial Vm-shell coordinate places bone (core along the axis) / muscle /
fat / viscera with the local tissue mass setting regional muscularity, and the cells are RESTRICTED
into named coarse organs -- the Engquist organs, now obtained from the NCA+LGM as intended.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m menagerie.grow_nca
Out: data/menagerie_nca_grow.png
"""
from __future__ import annotations
import numpy as np

from .genome import CERVICAL_COUNT, THORACIC_COUNT, LUMBAR_COUNT, SACRAL_COUNT, BASE_CAUDAL
from .grow_from_kernel import FATE_COL


def species_convergent_ext(g):
    """The species Wnt-PCP convergent-extension setting, GENOME-GROUNDED: a finned body_plan uses the
    measured zebrafish/mouse Wnt-PCP ratio (ZESTA vs MOSTA), a tetrapod uses the mouse base. This is
    the knob whose value decides whether the body is wide enough to admit the left-right electric-body
    eigenmode -- i.e. whether limbs (tetrapod) or no limbs (finned fish) emerge."""
    from medic.limb_genome_frame import convergent_ext_for
    if g is not None and getattr(g, "body_plan", "tetrapod") == "finned":
        return convergent_ext_for("fish")
    return convergent_ext_for("tetrapod")


def nca_body(g=None, seed=0, limbs=True):
    """Grow the fine NCA body. With limbs=True the genome-grounded limb frame is active: fore/hind
    Hox levels + the body width from the species' convergent_ext -> limbs emerge on the electric-body
    LR antinodes for a tetrapod, and stay absent for a finned fish."""
    import medic.unified_embryo as ue
    ce = species_convergent_ext(g) if limbs else None
    frames, m = ue.simulate(use_ecm=True, seed=seed, limb_buds=limbs, convergent_ext=ce)
    return np.asarray(m["pos"], float), np.asarray(m["fid"])


def species_deform(pos, g):
    """The species LoRA on the generic NCA body (von Baer: the embryo is generic, it deforms late).
    Warp the AP axis to the genome's neck/trunk/tail proportions (cervical_elong etc.) and scale the
    girth -- so the giraffe's NCA-grown body carries the giraffe's long neck."""
    from .decoder import L_CERV, L_TRUNK, L_CAUD
    from .genome import CERVICAL_COUNT, THORACIC_COUNT, LUMBAR_COUNT, BASE_CAUDAL
    x = pos[:, 0]; ap = (x - x.min()) / (np.ptp(x) + 1e-9)
    neck = CERVICAL_COUNT * L_CERV * g.cervical_elong
    trunk = (THORACIC_COUNT + LUMBAR_COUNT) * L_TRUNK * g.trunk_len
    tail = max(1, round(BASE_CAUDAL * g.tail_count)) * L_CAUD * g.tail_len
    tot = neck + trunk + tail
    fn, ft = neck / tot, (neck + trunk) / tot
    # generic body regions: head/neck 0-0.22, trunk 0.22-0.86, tail 0.86-1 -> species fractions
    ap_new = np.interp(ap, [0, 0.22, 0.86, 1.0], [0, fn, ft, 1.0])
    out = pos.copy()
    out[:, 0] = ap_new * tot * g.body_size
    out[:, 1] *= g.trunk_girth * g.body_size          # mediolateral girth
    out[:, 2] *= g.trunk_girth * g.body_size
    return out


def frame(pos, nbins=48):
    """Medial-axis frame of the grown cell cloud: AP (along the body) + radial (per AP slice)."""
    x = pos[:, 0]; ap = (x - x.min()) / (np.ptp(x) + 1e-9)
    b = np.clip((ap * nbins).astype(int), 0, nbins - 1)
    cen = np.zeros((nbins, 2)); rad = np.ones(nbins)
    for k in range(nbins):
        mm = b == k
        if mm.sum() > 3:
            c = pos[mm, 1:].mean(0); cen[k] = c
            rad[k] = np.linalg.norm(pos[mm, 1:] - c, axis=1).max() + 1e-6
        elif k > 0:
            cen[k] = cen[k - 1]; rad[k] = rad[k - 1]
    d = np.linalg.norm(pos[:, 1:] - cen[b], axis=1)
    return dict(ap=ap, radial=np.clip(d / rad[b], 0, 1.3), locrad=rad[b], cz=cen[b, 1], pos=pos)


def fates(F):
    radial, ap, cz, pos = F["radial"], F["ap"], F["cz"], F["pos"]
    musc = F["locrad"] / (F["locrad"].max() + 1e-9)
    fat_thr = 0.22 * (1.35 - 0.85 * musc)
    musc_outer = 1.0 - fat_thr
    fate = np.empty(len(pos), object); fate[:] = "muscle"
    fate[radial >= musc_outer] = "fat"
    fate[radial < 0.24] = "bone"
    ventral = pos[:, 2] < cz
    organ = (ap > 0.30) & (ap < 0.72) & ventral & (radial > 0.24) & (radial < musc_outer)
    fate[organ] = "organ"
    return fate


def restrict(F, fate):
    ap = F["ap"]; parts = {}
    counts = [("cervical", CERVICAL_COUNT), ("thoracic", THORACIC_COUNT), ("lumbar", LUMBAR_COUNT),
              ("sacral", SACRAL_COUNT), ("caudal", BASE_CAUDAL)]
    edges = np.array([0] + list(np.cumsum([c for _, c in counts])), float); edges /= edges[-1]
    bone = fate == "bone"; apb = ap[bone]
    for k, (nm, c) in enumerate(counts):
        for s in range(c):
            lo = edges[k] + (edges[k+1]-edges[k]) * s / c
            hi = edges[k] + (edges[k+1]-edges[k]) * (s+1) / c
            parts[f"{nm}_vertebra_{s+1}"] = int(((apb >= lo) & (apb < hi)).sum())
    musc = fate == "muscle"; apm = ap[musc]
    for k, (nm, c) in enumerate(counts):
        parts[f"muscle_{nm}"] = int(((apm >= edges[k]) & (apm < edges[k+1])).sum())
    parts["subcutaneous_fat"] = int((fate == "fat").sum())
    parts["viscera"] = int((fate == "organ").sum())
    return parts


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .targets import reference_genome
    gir = reference_genome("giraffe")
    pos0, fid = nca_body(gir)                                   # NCA body WITH genome-grounded limbs
    pos = species_deform(pos0, gir)                             # generic NCA body + giraffe LoRA
    F = frame(pos)
    fate = fates(F)
    parts = restrict(F, fate)
    col = np.array([FATE_COL[f] for f in fate])

    fig = plt.figure(figsize=(17, 5.6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    fu = np.unique(fid); cmap = plt.cm.tab10(np.linspace(0, 1, len(fu)))
    fc = np.zeros((len(fid), 3))
    for j, f in enumerate(fu):
        fc[fid == f] = cmap[j][:3]
    ax0.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=fc, s=4, alpha=0.6, linewidths=0)
    ax0.set_title(f"(a) NCA-grown cells ({len(pos)})\nmedic/unified_embryo — the fine model", fontsize=10)

    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    half = pos[:, 1] >= -0.02
    ax1.scatter(pos[half, 0], pos[half, 1], pos[half, 2], c=col[half], s=5, alpha=0.85, linewidths=0)
    ax1.set_title("(b) Vm field places the tissues\nbone core · muscle · fat · viscera", fontsize=10)

    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    bone = fate == "bone"
    ax2.scatter(pos[bone, 0], pos[bone, 1], pos[bone, 2], c=F["ap"][bone], cmap="turbo", s=6, alpha=0.9, linewidths=0)
    ax2.scatter(pos[~bone, 0], pos[~bone, 1], pos[~bone, 2], c="0.8", s=1, alpha=0.06, linewidths=0)
    nb = sum(1 for k, v in parts.items() if "vertebra" in k and v > 0)
    ax2.set_title(f"(c) RESTRICTION → organs\n{nb} vertebrae (AP) · muscle · fat · viscera", fontsize=10)

    for a in (ax0, ax1, ax2):
        a.set_box_aspect((np.ptp(pos[:, 0]), np.ptp(pos[:, 1]) + 1e-3, np.ptp(pos[:, 2]) + 1e-3), zoom=1.5)
        a.view_init(elev=16, azim=-72); a.set_axis_off()
    handles = [Line2D([0], [0], marker="o", ls="", mfc=FATE_COL[k], mec="none", ms=9, label=k)
               for k in ["bone", "muscle", "fat", "organ"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The fine model wired in: the NCA grows the cells; the Vm field places bone, muscle "
                 "and fat; restriction gives the coarse organs", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig("data/menagerie_nca_grow.png", dpi=150, bbox_inches="tight")
    props = {k: round(100 * float((fate == k).mean())) for k in ("bone", "muscle", "fat", "organ")}
    named = sum(1 for k, v in parts.items() if v > 0)
    print(f"NCA body {len(pos)} cells -> %{props} -> restricted to {named} named organs")
    print("saved data/menagerie_nca_grow.png")


if __name__ == "__main__":
    main()
