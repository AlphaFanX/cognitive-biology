"""Zero-shot cross-species transfer: FROZEN mouse-tuned organ models scored on the real human embryo.

The dense-human benchmark (hesta_dense_match_score) re-calibrates each organ's magnitudes to the human
fingerprints, so it measures whether the shared mechanism REPERTOIRE transfers. This is the stronger test: take
the organ models with their mouse E16.5 magnitudes FROZEN -- not one parameter adjusted for the human -- and
score them directly against the human dense reconstruction. If a mouse-tuned form still beats a shapeless-cloud
null on the human, the organ is conserved not only in mechanism but in magnitude across the divergence.

Result (see paragraph in Paper #9 sec:form): the HEART transfers untouched and by a wide margin (0.71 vs 0.42) --
the ancient cardiac loop is conserved in magnitude; the brain and spinal cord transfer weakly; the GUT fails
(0.28 vs 0.40). The gut failure has an immediate ontogenetic cause -- at Carnegie stage 12-13 the human
primitive gut is a nearly straight tube that has not yet coiled, while the mouse form was fitted at E16.5, a
coiled midgut -- and, on top of it, the gut is the organ most reduced in the human lineage (the expensive-tissue
hypothesis, Aiello & Wheeler 1995). A stage-matched human gut is what would separate the two.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.zeroshot_mouse_to_human
"""
import numpy as np
from medic.embryo_match_score import full_desc, fingerprint, shape_match
from medic.organ_3d_vs_real import blob
from medic.e165_match_score import e165_heart, e165_brain, e165_spinal, e165_gut   # mouse E16.5, params frozen

TARGET = "data/hesta/hesta_dense_3d.npz"
# (human organ, frozen mouse-tuned model)
PAIRS = [("Heart", e165_heart), ("Brain", e165_brain), ("Spinal Cord", e165_spinal), ("Primitive Gut", e165_gut)]


def main():
    d = np.load(TARGET, allow_pickle=True); xyz, tis = d["xyz"], d["tissue"]
    print("ZERO-SHOT: frozen mouse-E16.5 organ models scored on the real human dense embryo (no human re-fit)")
    print(f"{'human organ':16s} {'blob':>6s} {'mouse-frozen':>13s}  beats null?")
    bs, fs = [], []
    for organ, gen in PAIRS:
        R = xyz[tis == organ]; fp = fingerprint(R)
        sb = shape_match(full_desc(blob(len(R), fp["size"] * 2)), fp)
        sf = shape_match(full_desc(gen()), fp)
        bs.append(sb); fs.append(sf)
        print(f"  {organ:16s} {sb:6.2f} {sf:13.2f}   {'YES' if sf > sb else 'no'}")
    print(f"\nfrozen-mouse mean: null {np.mean(bs):.3f} -> {np.mean(fs):.3f}  "
          f"(heart {fs[0]:.2f} vs null {bs[0]:.2f}; gut {fs[3]:.2f} vs null {bs[3]:.2f})")


if __name__ == "__main__":
    main()
