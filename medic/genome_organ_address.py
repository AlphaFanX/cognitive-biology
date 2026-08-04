"""
Genome-derived organ AP addresses -> the anchor organ_sprouting places on.
==========================================================================
This is the wiring step of the genome<->MOSTA bridge (medic.genome_head_mosta): it turns the
genome's Hox address per organ head -- the Hox-regulon colinear number read at the head layer --
into an antero-posterior address the embryo can place on, replacing the FITTED atlas anchor
(medic.mosta_organ_anchors) with a genome-code read. It reuses the existing machinery rather than
re-deriving anything:

  * genome_head_mosta  supplies the per-head genome Hox address (Link 2) and the trunk/cranial split.
  * limb_genome_frame  already does this for the LIMBS from the fossil Hox enhancers; organs are the
                       same idea generalised to the trunk head set.

One documented calibration maps the Hox colinear number to body AP -- a single linear anchor
(AP = b0 + b1*hox) fit once to the trunk organs, exactly the g_K / PCP_K anchor pattern: the code is
read, one constant is fit. Trunk organs get their AP from the genome Hox code; the cranial heads are
the Hox-free anterior default and keep a fixed anterior address (they carry no colinear signal).

organ_sprouting calls genome_organ_ap_table() and snaps each organ to the antinode nearest its
genome AP; it falls back to the MOSTA anchor, then the hand-set antinode, if the genome read is absent.

Cache: data/organ_cascade/genome_organ_address.json.
Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.genome_organ_address
"""
from __future__ import annotations
import os, json, glob
import numpy as np

CACHE = "data/organ_cascade/genome_organ_address.json"

# MOSTA tissue (head) -> our organ-schedule name. Only the placed heads.
TISSUE_TO_ORGAN = {
    "Heart": "Heart", "Liver": "Liver", "Lung primordium": "Lung", "Lung": "Lung",
    "Pancreas": "Pancreas", "Kidney": "Kidney", "Urogenital ridge": "Kidney",
    "GI tract": "Gut", "Dorsal root ganglion": "DRG", "Sympathetic nerve": "Sympathetic",
    "Cartilage primordium": "Cartilage", "Muscle": "Muscle", "Blood vessel": "Vessel",
}
# anterior, Hox-free heads -> fixed anterior address. The Hox code is a TRUNK/posterior code; the
# eye and otic vesicle (cranial) and the heart and lung (anterior lateral plate / foregut, anterior
# to the Hox domain) are the anterior default and are NOT placed by the Hox read -- their Hox-regulon
# signal is noise. The genome Hox address places only the genuinely Hox-controlled posterior organs.
ANTERIOR_ORGANS = {"Eye": 0.10, "Otic": 0.15, "Heart": 0.38, "Lung": 0.47}


def _stage_path(stage):
    return sorted(glob.glob(os.path.join("data/mosta", f"{stage}_*.MOSTA.h5ad")))[0]


def build(stage="E12.5", verbose=False):
    """Fit the one Hox->AP calibration on the trunk heads and emit the genome AP per organ."""
    from medic.genome_head_mosta import _load, _body_ap, link2_hox_position
    from medic.genome.hox_ap_address import colinear_fraction
    cat, ann, xy, R, tfs = _load(_stage_path(stage))
    ap = _body_ap(xy, ann, cat)
    pos = link2_hox_position(R, tfs, ann, cat, ap)

    # GENOME-derived Hox->AP: the predictor is the mm9 COLINEAR fraction h(paralog) (the genome's own
    # antero-posterior gene-spacing code), NOT the raw number. Register it onto the body through the
    # FIXED tail anchor (h=1 -> AP=1.0; the posterior body end is definitional), leaving ONE free
    # anatomical anchor -- the anterior Hox boundary -- instead of a 2-param blind fit. That anchor
    # self-consistently resolves to AP~0.29 = the real anterior Hox limit (= the fossil-limb Hox9 0.294).
    trunk = [r for r in pos["rows"] if r["trunk"]]
    hh = np.array([colinear_fraction(r["genome_hox"]) for r in trunk])
    ay = np.array([r["mosta_ap"] for r in trunk])
    b1 = float(np.sum((hh - 1.0) * (ay - 1.0)) / (np.sum((hh - 1.0) ** 2) + 1e-12))
    b0 = 1.0 - b1                                   # AP = 1.0 + b1*(h-1) = b0 + b1*h ; anterior Hox limit = b0

    organs = {}
    for r in pos["rows"]:
        organ = TISSUE_TO_ORGAN.get(r["head"])
        if organ is None or not r["trunk"] or organ in ANTERIOR_ORGANS:
            continue
        ap_g = float(np.clip(b0 + b1 * colinear_fraction(r["genome_hox"]), 0.02, 0.98))
        # keep the most anterior mapping if two tissues map to one organ (e.g. Kidney<-Urogenital)
        if organ not in organs or ap_g < organs[organ]["ap"]:
            organs[organ] = dict(ap=round(ap_g, 3), genome_hox=r["genome_hox"],
                                 mosta_tissue=r["head"], source="genome-Hox")
    for organ, ap_a in ANTERIOR_ORGANS.items():
        organs[organ] = dict(ap=ap_a, genome_hox=None, mosta_tissue=None, source="anterior-default")

    out = dict(stage=stage, calibration=dict(b0=round(float(b0), 4), b1=round(float(b1), 4),
               note="AP = b0 + b1*colinear_fraction(genome_hox); predictor = mm9 genome colinear geometry, "
                    "registered through the fixed tail anchor -> 1 anatomical anchor (anterior Hox limit=b0)"),
               anterior_hox_limit=round(float(b0), 3),
               trunk_fit=dict(pearson=pos["trunk_pearson"], spearman=pos["trunk_spearman"]),
               organs=organs)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"), indent=1)
    if verbose:
        print(f"{stage}: Hox->AP calibration AP = {b0:.3f} + {b1:.3f}*hox  "
              f"(trunk fit r={pos['trunk_pearson']:.2f})\n")
        print(f" {'organ':10s} {'AP(genome)':>10s} {'Hox#':>6s}  source (MOSTA tissue)")
        for o, r in sorted(organs.items(), key=lambda kv: kv[1]["ap"]):
            hx = f"{r['genome_hox']:.2f}" if r["genome_hox"] is not None else "  -"
            print(f" {o:10s} {r['ap']:>10.3f} {hx:>6s}  {r['source']} ({r['mosta_tissue']})")
    return out


def genome_organ_ap_table(rebuild=False):
    """Cached {organ: {ap, ...}} genome-derived AP addresses for organ_sprouting. {} if unavailable."""
    if not rebuild and os.path.exists(CACHE):
        try:
            return json.load(open(CACHE))["organs"]
        except Exception:
            pass
    try:
        return build()["organs"]
    except Exception:
        return {}


def main():
    build(verbose=True)
    print(f"\nsaved {CACHE}")
    print("These AP addresses are read from the genome Hox code (one fitted calibration anchor); "
          "organ_sprouting consumes them in place of the fitted MOSTA anchor.")


if __name__ == "__main__":
    main()
