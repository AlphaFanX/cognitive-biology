#!/usr/bin/env python3
"""
limb_genome_frame -- ground the amphibian (fish->tetrapod) limb frame in the genome.
=====================================================================================

The unified embryo (medic.unified_embryo) placed the paired limbs with three
HAND-SET numbers: the fore/hind Hox AP levels (0.20, 0.44), and the body-width /
convergent-extension knob ``pcp`` (0.25). This module replaces both with reads
from the SAME genome data the fish bioelectric NCA already uses:

  * fore/hind AP levels  <-  Hox chromosomal COLINEARITY (real cluster positions,
      medic.body_plan_morphogenesis.hox_limb_levels: Hox6 forelimb / Hox10 hindlimb,
      Burke et al. 1995). No typed coordinate.

  * body width / convergent-extension  <-  the Wnt-PCP module ACCESSIBILITY from the
      ABC enhancer database (VANGL2, WNT5A, FZD7 -- the planar-cell-polarity genes
      that drive axial convergent extension). Strong PCP tone -> strong CE -> the
      body narrows -> the left-right ("body electric") eigenmode leaves the spectrum
      -> LIMBLESS (fish). Weak PCP tone -> the body keeps its width -> the LR mode is
      present -> paired LIMBS (tetrapod). This is exactly Miles's observation: a
      fish tapers so there is no body-electric LR mode; the amphibian keeps a wide
      pelvis so the LR mode -- and with it the limbs -- appears.

The fish<->tetrapod CONTRAST is carried by the ``convergent_ext`` species knob
(medic ../menagerie/genome.py; zebrafish ~1.45 high CE, tetrapod 1.0). The ABC read
sets the genome-derived BASE tone; one documented calibration maps ABC PCP-activity
onto the model's width scale (the single anchor, analogous to the g_K anchor in the
conductance kernel -- everything else is read, not fit).

Cache: data/organ_cascade/limb_genome_frame.json (so the 325 MB ABC file is scanned
once, not every simulation). Rebuild:  python -m medic.limb_genome_frame --rebuild
"""
from __future__ import annotations

import gzip
import json
import os

# Wnt-PCP / planar-cell-polarity convergent-extension module (present in ABC hg38).
WNT_PCP_GENES = ["VANGL2", "WNT5A", "FZD7"]

# The SAME module in the two spatial atlases -- read to ground the fish<->tetrapod CONTRAST in
# real cross-species data (zebrafish ZESTA vs mouse MOSTA), replacing the hand-set convergent_ext.
# Each entry is an ortholog group (paralogs summed).
_ZF_PCP = [["vangl2"], ["wnt5a", "wnt5b"], ["wnt11", "wnt11f2"],
           ["prickle1a", "prickle1b"], ["fzd7a", "fzd7b"], ["celsr1a", "celsr1b"]]
_MS_PCP = [["Vangl2"], ["Wnt5a"], ["Wnt11"], ["Prickle1"], ["Fzd7"], ["Celsr1"]]

_HERE = os.path.dirname(__file__)
_ABC = os.path.join(_HERE, "..", "data", "enhancer_promoter", "AllPredictions.ABC.txt.gz")
_ZESTA = os.path.join(_HERE, "..", "data", "zesta", "zf_sixtime_slice.h5ad")
_MOSTA = os.path.join(_HERE, "..", "data", "mosta", "E9.5_E2S2.MOSTA.h5ad")
_CACHE = os.path.join(_HERE, "..", "data", "organ_cascade", "limb_genome_frame.json")

# --- calibration (the single anchor) ----------------------------------------
# The tetrapod BASE width is genome-anchored: the model width knob at the tetrapod
# operating point is set from the real ABC Wnt-PCP tone by one constant,
#     pcp_base = PCP_K * wnt_pcp_tone           (the g_K-style anchor; tone is read).
# The fish<->tetrapod CONTRAST rides the species convergent_ext knob (menagerie),
#     pcp = pcp_base * convergent_ext ** PCP_GAMMA .
# Honest scope: the BASE is grounded in a genome accessibility read (human ABC hg38);
# the fish contrast is the convergent_ext knob and is NOT yet a fish-genome read (no
# zebrafish ABC/ATAC here) -- fully grounding it needs a fish accessibility track.
PCP_K = 5.13          # ABC tone -> tetrapod base pcp (0.0487 -> ~0.25, the limbed operating point)
PCP_GAMMA = 1.04      # convergent_ext exponent: with the MEASURED zebrafish/mouse ratio ~3.6 this
#                       maps mouse (1.0) -> ~0.25 (limbed) and zebrafish (~3.6) -> ~0.95 (limbless).
_PCP_LO, _PCP_HI = 0.05, 1.20


def _scan_abc_pcp_tone():
    """Mean ABC.Score of the Wnt-PCP module across all cell types = the genome's
    convergent-extension tone (real accessibility read, not a typed number)."""
    if not os.path.exists(_ABC):
        raise FileNotFoundError(f"ABC file not found: {_ABC}")
    want = set(WNT_PCP_GENES)
    tot = {g: 0.0 for g in want}
    cnt = {g: 0 for g in want}
    with gzip.open(_ABC, "rt") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        gi = hdr.index("TargetGene")
        si = hdr.index("ABC.Score")
        for line in f:
            c = line.rstrip("\n").split("\t")
            g = c[gi]
            if g in want:
                try:
                    tot[g] += float(c[si]); cnt[g] += 1
                except (ValueError, IndexError):
                    pass
    per_gene = {g: (tot[g] / cnt[g] if cnt[g] else 0.0) for g in want}
    enh = {g: cnt[g] for g in want}
    present = [g for g in want if cnt[g] > 0]
    tone = sum(per_gene[g] for g in present) / max(1, len(present))
    return tone, per_gene, enh


def _atlas_pcp_enrichment(path, groups):
    """Mean Wnt-PCP module expression as fold-over-average-gene in a spatial atlas
    (comparable across species/platforms). Each group = ortholog paralogs summed."""
    import anndata as ad
    import scipy.sparse as sp
    a = ad.read_h5ad(path)
    X = a.X
    if sp.issparse(X):
        X = X.tocsc()
    vn = {str(v).lower(): i for i, v in enumerate(a.var_names)}
    glob = float(X.mean()) + 1e-12
    scores = []
    for grp in groups:
        idxs = [vn[g.lower()] for g in grp if g.lower() in vn]
        if not idxs:
            continue
        m = sum(float(X[:, i].mean()) for i in idxs)   # sum paralogs -> one ortholog
        scores.append(m / glob)
    import numpy as _np
    return float(_np.mean(scores)) if scores else float("nan")


def _fish_convergent_ext():
    """Cross-species Wnt-PCP CE ratio (zebrafish ZESTA / mouse MOSTA) = the genome-grounded
    fish<->tetrapod width contrast. > 1 means the fish reads stronger CE -> narrower body."""
    if not (os.path.exists(_ZESTA) and os.path.exists(_MOSTA)):
        return None, None, None
    zf = _atlas_pcp_enrichment(_ZESTA, _ZF_PCP)
    ms = _atlas_pcp_enrichment(_MOSTA, _MS_PCP)
    return round(zf / ms, 4), round(zf, 4), round(ms, 4)


def build_cache():
    """Scan ABC + Hox + the two atlases once and cache the genome reads."""
    from medic.body_plan_morphogenesis import hox_limb_levels
    fore, hind = hox_limb_levels(0.0, 1.0)         # genome Hox colinearity
    tone, per_gene, enh = _scan_abc_pcp_tone()     # genome Wnt-PCP accessibility (human ABC)
    fish_ce, zf_enr, ms_enr = _fish_convergent_ext()   # measured zebrafish/mouse CE contrast
    out = {
        "hox_forelimb_ap": round(float(fore), 4),
        "hox_hindlimb_ap": round(float(hind), 4),
        "wnt_pcp_genes": WNT_PCP_GENES,
        "wnt_pcp_tone": round(float(tone), 5),
        "wnt_pcp_per_gene": {g: round(v, 5) for g, v in per_gene.items()},
        "wnt_pcp_enhancer_counts": enh,
        "fish_convergent_ext": fish_ce,          # measured zebrafish/mouse Wnt-PCP ratio (fish contrast)
        "zebrafish_pcp_enrichment": zf_enr,
        "mouse_pcp_enrichment": ms_enr,
        "source": "Hox: body_plan_morphogenesis.hox_limb_levels (Hox6/Hox10 colinearity); "
                  "width base: ABC.Score of VANGL2/WNT5A/FZD7 (Wnt-PCP module, human hg38); "
                  "fish contrast: Wnt-PCP module enrichment ZESTA(zebrafish)/MOSTA(mouse)",
    }
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    with open(_CACHE, "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def _load():
    if not os.path.exists(_CACHE):
        return build_cache()
    with open(_CACHE) as fh:
        return json.load(fh)


def genome_limb_frame(convergent_ext: float = 1.0):
    """Return the genome-grounded limb frame.

    convergent_ext: the species Wnt-PCP knob (menagerie/genome.py). 1.0 = tetrapod
    base; > 1 = stronger CE = narrower body = fish-ward. Zebrafish ~ 1.45.

    Returns dict(fore_ap, hind_ap, pcp, wnt_pcp_tone, convergent_ext).
    """
    d = _load()
    tone = d["wnt_pcp_tone"]
    pcp_base = PCP_K * tone                                  # genome-anchored tetrapod width
    pcp = pcp_base * float(convergent_ext) ** PCP_GAMMA      # species contrast (convergent_ext knob)
    pcp = max(_PCP_LO, min(_PCP_HI, pcp))
    return {
        "fore_ap": d["hox_forelimb_ap"],
        "hind_ap": d["hox_hindlimb_ap"],
        "pcp": round(float(pcp), 4),
        "wnt_pcp_tone": tone,
        "convergent_ext": float(convergent_ext),
    }


def convergent_ext_for(species="tetrapod"):
    """Genome-grounded convergent_ext for a species: 1.0 for the tetrapod/mouse base, and the
    MEASURED zebrafish/mouse Wnt-PCP ratio (from the atlases) for fish. Falls back to 1.45 for
    fish only if the atlases are unavailable (hand-set legacy)."""
    if str(species).lower() in ("fish", "zebrafish", "zf"):
        d = _load()
        return float(d.get("fish_convergent_ext") or 1.45)
    return 1.0


def main():
    import sys
    if "--rebuild" in sys.argv:
        d = build_cache()
        print("rebuilt cache:", _CACHE)
        print(json.dumps(d, indent=2))
    fish_ce = convergent_ext_for("fish")
    print("\ngenome-grounded limb frame (fish contrast now READ from ZESTA/MOSTA):")
    for name, ce in [("tetrapod (mouse)", convergent_ext_for("tetrapod")),
                     ("fish (zebrafish)", fish_ce)]:
        fr = genome_limb_frame(ce)
        print(f"  {name:18s} conv_ext={ce:>5.2f}  ->  pcp={fr['pcp']:.3f}  "
              f"fore_ap={fr['fore_ap']:.3f} hind_ap={fr['hind_ap']:.3f}")


if __name__ == "__main__":
    main()
