"""
The real mouse at CELL resolution, E9.5 -> E13.5 (cell-level shape).
===================================================================

Read the real MOSTA cell clouds directly (h5py, bypassing the anndata parser these files
break), colour every cell by its tissue, and render the stages side by side: the actual
mouse embryo taking shape at cell resolution, from the comma-shaped E9.5 to the recognizable
E13.5 fetus. This is the cell-level target the shape-training fits to -- the real thing, not
a silhouette. Corrupt stages (download resume errors) are skipped with a note.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = [("E9.5", "data/mosta/E9.5_E2S2.MOSTA.h5ad"),
          ("E10.5", "data/mosta/E10.5_E1S1.MOSTA.h5ad"),
          ("E11.5", "data/mosta/E11.5_E1S1.MOSTA.h5ad"),
          ("E12.5", "data/mosta/E12.5_E1S1.MOSTA.h5ad"),
          ("E13.5", "data/mosta/E13.5_E1S1.MOSTA.h5ad")]

# tissue -> colour (our head palette + fetal tissues)
COL = {
    "Brain": "#5b78e8", "Spinal cord": "#6a86e0", "Dorsal root ganglion": "#8a6fe0",
    "Choroid plexus": "#7ad0ff", "Eye": "#33d8ff", "Heart": "#ee2938", "Liver": "#b857a3",
    "Lung": "#8cc7e6", "Lung primordium": "#8cc7e6", "Pancreas": "#ccd14d", "Gut": "#d19a66",
    "GI tract": "#d19a66", "Kidney": "#a8497a", "Meninges": "#c0a0d0",
    "Muscle": "#db6b6b", "Cartilage primordium": "#f0f0db", "Cartilage": "#f0f0db",
    "Ossification": "#eaeada", "Sclerotome": "#e8e8cf",
    "Epidermis": "#f5d1bd", "Surface ectoderm": "#f5d1bd", "Blood vessel": "#8a1a1a",
    "Blood": "#a11414", "AGM": "#a8497a", "Connective tissue": "#c8b48c",
    "Mesenchyme": "#b8a684", "Head mesenchyme": "#c8b490", "Notochord": "#99d1b8",
    "Cavity": "#20242c", "Jaw and tooth": "#e0c060",
}


def read_stage(path):
    h = h5py.File(path, "r")
    xy = h["obsm/spatial"][:].astype(float)
    codes = h["obs/annotation"][:]
    cats = [c.decode() if isinstance(c, bytes) else str(c) for c in h["obs/__categories/annotation"][:]]
    ann = np.array([cats[c] if 0 <= c < len(cats) else "" for c in codes])
    h.close()
    return xy, ann


def orient(xy):
    """Head to the left, dorsal up: PCA long axis = horizontal."""
    c = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    q = c @ vt.T
    return q


def main():
    ok = []
    for name, path in STAGES:
        try:
            xy, ann = read_stage(path)
            ok.append((name, orient(xy), ann))
            print(f"{name}: {len(xy)} cells, {len(set(ann))} tissues")
        except Exception as e:
            print(f"{name}: SKIP (corrupt) {str(e)[:50]}")
    n = len(ok)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5), facecolor="#0d1017")
    if n == 1:
        axes = [axes]
    for ax, (name, q, ann) in zip(axes, ok):
        ax.set_facecolor("#0d1017")
        cols = np.array([COL.get(t, "#585c66") for t in ann])
        # subsample for a crisp scatter
        rng = np.random.default_rng(0)
        sel = rng.choice(len(q), min(30000, len(q)), replace=False)
        ax.scatter(q[sel, 0], q[sel, 1], s=1.4, c=cols[sel], linewidths=0)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{name}  ·  {len(q):,} cells", color="#cbd5e1", fontsize=12)
    fig.suptitle("The real mouse at cell resolution (MOSTA), coloured by tissue", color="#7dd3fc", fontsize=14)
    fig.tight_layout()
    fig.savefig("data/cell_level_mouse.png", dpi=115, facecolor="#0d1017")
    print("saved data/cell_level_mouse.png")


if __name__ == "__main__":
    main()
