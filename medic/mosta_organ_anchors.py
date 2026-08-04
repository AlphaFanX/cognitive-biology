"""
Train the organ addresses against the real MOSTA atlas -> genome-derivable ANCHORS.
==================================================================================

Miles's principle (2026-07-15): rather than HAND-SET the organ addresses (which antinode
each organ fills, its DV band), FIT them to the real mouse atlas and treat the fitted
values as ANCHOR VARIABLES -- measurement-anchored now, genome-derivable later (exactly
like the g_K conductance anchor). This "trains between the genome and MOSTA" to extract
variables that COULD be genome-based even before we trace the exact genomic mechanism.

This module reads the real E9.5 mouse organogenesis atlas (Stereo-seq MOSTA, spatial +
annotated primordia), builds a normalized body frame (PCA: AP = long axis, DV = second),
and returns each organ's REAL (AP, DV) centroid -- the anchor. `organ_anchor_table()`
maps the MOSTA annotations onto our organ names and caches the result; organ_sprouting can
then snap each organ to the antinode NEAREST its MOSTA AP (data-anchored, not hand-set).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.mosta_organ_anchors
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
import os
import numpy as np

ATLAS = "data/mosta/E9.5_E2S2.MOSTA.h5ad"
CACHE = "data/organ_cascade/mosta_organ_anchors.json"

# MOSTA annotation -> our organ/tissue name (the ones we place)
MAP = {
    "Heart": "Heart", "Lung primordium": "Lung", "Liver": "Liver",
    "Pancreas primordium": "Pancreas", "AGM": "Kidney",          # AGM = intermediate-mesoderm region
    "Brain": "Eye", "Sclerotome": "Rib", "Surface ectoderm": "Skin",
    "Spinal cord": "SpinalCord", "Notochord": "Notochord",
}


def _body_frame(xy):
    """Canonical AP x DV frame of the embryo section, UNBENT (medic.ap_unbend): AP = arc-length
    along the medial-axis backbone, DV = signed perpendicular -- so a curved embryo is straightened
    and the axial positions aren't compressed by the fold. Orientation set by the caller."""
    try:
        from medic.ap_unbend import unbend
        return unbend(xy)
    except Exception:
        c = xy - xy.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        ap = c @ vt[0]; dv = c @ vt[1]
        return ((ap - ap.min()) / (np.ptp(ap) + 1e-9), (dv - dv.min()) / (np.ptp(dv) + 1e-9))


def extract(verbose=False):
    import anndata as ad
    a = ad.read_h5ad(ATLAS)
    ann = a.obs["annotation"].astype(str).values
    xy = np.asarray(a.obsm["spatial"], float)
    ap, dv = _body_frame(xy)

    # orient AP so the brain is anterior (ap small) and DV so the brain/spinal (neural) is dorsal (dv large)
    if "Brain" in ann:
        if ap[ann == "Brain"].mean() > 0.5:
            ap = 1.0 - ap
        if dv[ann == "Brain"].mean() < 0.5:
            dv = 1.0 - dv

    rows = {}
    for label in sorted(set(ann)):
        m = ann == label
        if m.sum() < 8:
            continue
        rows[label] = dict(ap=round(float(np.median(ap[m])), 3), dv=round(float(np.median(dv[m])), 3),
                           ap_sd=round(float(ap[m].std()), 3), n=int(m.sum()))
    # map to our organs
    anchors = {}
    for label, organ in MAP.items():
        if label in rows:
            anchors[organ] = {**rows[label], "mosta_label": label}
    if verbose:
        print(f"MOSTA E9.5: {a.n_obs} spots, body frame from PCA (AP=long axis, DV=2nd).")
        print(f"\n {'organ':10s} {'MOSTA label':20s} {'AP':>5s} {'DV':>5s}  n")
        for organ, r in sorted(anchors.items(), key=lambda kv: kv[1]["ap"]):
            print(f" {organ:10s} {r['mosta_label']:20s} {r['ap']:5.2f} {r['dv']:5.2f}  {r['n']}")
        print("\nAll annotated regions (AP-sorted) -- the full anchor set available:")
        for label, r in sorted(rows.items(), key=lambda kv: kv[1]["ap"]):
            print(f"   {label:22s} AP {r['ap']:.2f}  DV {r['dv']:.2f}  (n={r['n']})")
    return anchors, rows


def organ_anchor_table(rebuild=False):
    """Cached {organ: {ap, dv, ...}} MOSTA anchors."""
    if not rebuild and os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    anchors, _ = extract()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(anchors, f, indent=2)
    return anchors


def main():
    anchors, _ = extract(verbose=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(anchors, f, indent=2)
    print(f"\nsaved {CACHE}  ({len(anchors)} organ anchors)")
    print("These AP/DV positions are the MOSTA ANCHORS: fitted to real data now, "
          "genome-derivable later (the g_K-anchor pattern).")


if __name__ == "__main__":
    main()
