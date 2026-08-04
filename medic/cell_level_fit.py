"""
Run the shape-training at cell level: our genome-grown body into the real fetus shape.
======================================================================================

The real E13.5 cell cloud is the target shape. We grow our body from the genome (with its
heads on the electric-body antinodes), read the real fetus outline (per antero-posterior
slice, the dorso-ventral extent of the real cells), and MAP our body's cells into that
outline -- each of our cells keeps its antero-posterior level and its fractional dorso-ventral
position but is placed inside the real body's silhouette at that level. The result: our
derived anatomy, coloured by our heads, sitting in the real mouse's shape, beside the real
cells coloured by their tissue. The loop is closed: genome -> our body -> the real form.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medic.ap_unbend import unbend
from medic.cell_level_mouse import COL, read_stage
from medic.unified_embryo import simulate, _symmetrize, FATES

TARGET = "data/mosta/E13.5_E1S1.MOSTA.h5ad"
# our fate -> a display tissue name that hits the COL palette
FATE_COL = {"Forebrain": "Brain", "Nervous System": "Brain", "Spinal Cord": "Spinal cord",
            "Eye": "Eye", "Otic": "#ffd23a", "Heart": "Heart", "Liver": "Liver", "Lung": "Lung",
            "Pancreas": "Pancreas", "Gut": "Gut", "Kidney": "Kidney", "Rib": "Cartilage primordium",
            "Muscle": "Muscle", "Limb Bud": "#47db76", "Notochord": "Notochord", "Skin": "Epidermis",
            "Neural Crest": "Dorsal root ganglion", "Mesoderm": "Connective tissue",
            "Somite": "Cartilage primordium", "Epidermal": "Epidermis",
            # brain subheads -> one CNS mass; organ subheads -> their organ colour; new leaf heads
            "Midbrain": "Brain", "Hindbrain": "Brain", "Cerebellum": "Brain", "OlfactoryBulb": "Brain",
            "Retina": "Eye", "Choroid": "Choroid plexus", "Meninges": "Meninges",
            "Connective": "Connective tissue", "HeadMes": "Head mesenchyme", "Cartilage": "Cartilage primordium",
            "DRG": "Dorsal root ganglion", "Sympathetic": "Dorsal root ganglion", "Jaw": "Jaw and tooth",
            "Vessel": "Blood vessel", "Blood": "Blood", "Atrium": "Heart", "Ventricle": "Heart",
            "Outflow": "Heart", "LiverHaem": "Liver", "Foregut": "Gut", "Hindgut": "Gut", "Mucosa": "Gut",
            "Nephron": "Kidney", "Mesothelium": "Connective tissue", "Mesentery": "Connective tissue",
            "Branchial": "Jaw and tooth"}


def _col(name):
    v = FATE_COL.get(name, name)
    return COL.get(v, v if isinstance(v, str) and v.startswith("#") else "#585c66")


def outline(ap, dv, nb=48):
    edges = np.linspace(0, 1, nb + 1)
    lo = np.full(nb, np.nan); hi = np.full(nb, np.nan)
    for i in range(nb):
        m = (ap >= edges[i]) & (ap < edges[i + 1])
        if m.sum() >= 5:
            lo[i] = np.percentile(dv[m], 3); hi[i] = np.percentile(dv[m], 97)
    ok = ~np.isnan(lo)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return mid[ok], lo[ok], hi[ok]


def main():
    # real fetus target
    rxy, rann = read_stage(TARGET)
    rng = np.random.default_rng(0)
    rs = rng.choice(len(rxy), min(30000, len(rxy)), replace=False)
    rap, rdv = unbend(rxy[rs]); rann = rann[rs]
    neural = np.array([("Brain" in t) or ("Spinal" in t) for t in rann])
    if neural.sum() > 8 and rap[neural].mean() > 0.5:
        rap = 1.0 - rap
    mid, lo, hi = outline(rap, rdv)

    # our genome-grown body
    frames, _ = simulate(use_ecm=True, limb_buds=True, convergent_ext=1.0)
    Ps, _, Fs = _symmetrize(frames[-1][3], frames[-1][4], frames[-1][5])
    oap = (Ps[:, 0] - Ps[:, 0].min()) / (np.ptp(Ps[:, 0]) + 1e-9)
    # our DV fraction within our body at each AP slice
    odvf = np.zeros(len(Ps))
    ob = np.clip((oap * 48).astype(int), 0, 47)
    for k in range(48):
        m = ob == k
        if m.sum() > 3:
            y = Ps[m, 1]
            odvf[m] = (y - y.min()) / (np.ptp(y) + 1e-9)
    # map our cells into the real outline
    rlo = np.interp(oap, mid, lo); rhi = np.interp(oap, mid, hi)
    ody = rlo + odvf * (rhi - rlo)
    names = np.array([FATES[f] if f >= 0 else "" for f in Fs])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].scatter(rap, rdv, s=1.4, c=[COL.get(t, "#585c66") for t in rann], linewidths=0)
    ax[0].set_title("real E13.5 fetus  ·  real tissues", color="#cbd5e1")
    keep = (names != "") & np.isfinite(ody)
    ax[1].scatter(oap[keep], ody[keep], s=2.0, c=[_col(n) for n in names[keep]], linewidths=0)
    ax[1].set_title("our genome-grown anatomy  ·  fitted to the real shape", color="#cbd5e1")
    fig.suptitle("Cell-level shape-training: genome → our body → the real mouse form (E13.5)",
                 color="#7dd3fc", fontsize=13)
    fig.tight_layout()
    fig.savefig("data/cell_level_fit.png", dpi=118, facecolor="#0d1017")
    print(f"real {len(rs)} cells; our body {int(keep.sum())} placed. saved data/cell_level_fit.png")


if __name__ == "__main__":
    main()
