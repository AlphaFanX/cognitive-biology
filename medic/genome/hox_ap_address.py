#!/usr/bin/env python3
"""
Derive the limb AP address from the posterior-Hox ENHANCERS in the fossil record.
=================================================================================

Replaces the colinear AP levels that were read from IDEALISED HUMAN Hox cluster
coordinates (medic.body_plan_morphogenesis.hox_colinear_ap, feeding the sim's
fore/hind limb levels) with the REAL mouse fossil-record enhancer positions
(Jadhav mm9 LMR archive, see medic.genome.hox_fossil_record).

Colinearity in one line: a Hox gene's position along the cluster IS its
anterior->posterior body address (3' anterior -> 5' posterior; Kmita & Duboule
2003). So we build a colinear FRAME per cluster from the real mm9 terminal genes
(Hox1 anterior <-> Hox13 posterior, UCSC refGene), then read each LIMB-determinant
Hox gene's position not from the gene body but from the hypomethylation-weighted
CENTROID of its archived enhancers in the adult gut fossil record -- the real,
measured cis-regulatory footprint. Deeper hypomethylation = more strongly archived
= higher weight.

Determinant paralogs (the genomic CODE; positions read from the archive):
    forelimb / pectoral  <- Hox9 anterior boundary  (Hoxa9/Hoxd9; Xu & Wellik 2011)
    hindlimb / pelvic     <- Hox10 lumbosacral level (Hoxa10/Hoxd10; Wellik & Capecchi
                             2003; Cohn & Tickle 1997)
HoxA and HoxD are the limb clusters (Hox9/Hox10 read there); HoxB/HoxC contribute
Hox9 only (no Hox10 member / not in the posterior anchor set).

HONEST SCOPE. The fossil record is intestinal epithelium (Jadhav 2019), which does
not itself feature Hox -- but chromosomal colinearity is a property of the CLUSTER
architecture, tissue-independent, so reading the cluster's enhancer geometry from
gut cells to place the limb Hox levels is legitimate. mm9 coords are UCSC refGene
(Hox1/Hox9/Hox13 verified; Hoxa10/a11, Hoxd10/11 interpolated between verified
flanking anchors and marked). The two body anchors (anterior Hox limit, tail) are
the same already-derived landmarks the previous colinear map used.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.genome.hox_ap_address
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

DATA_DIR = "data/jadhav_mouse"
ADULT_LMRS = os.path.join(DATA_DIR, "Adult_LMRs.bed")
LMR_MAX = 50.0
PAD = 10_000

FORE_PARALOG = 9
HIND_PARALOG = 10
# The paired appendage clusters (Hoxa/Hoxd 9-13 build the limb; Hoxb/Hoxc play
# lesser limb roles) -- the limb AP address is read from these two.
LIMB_CLUSTERS = ("HoxA", "HoxD")

# Real mm9 (NCBI37) colinear FRAME per cluster: ant = anterior terminal gene mid
# (Hox1 / lowest paralog), post = posterior terminal gene mid (Hox13). genes = the
# posterior determinant paralogs with mm9 gene spans (interpolated ones marked).
_G = lambda s, e: (s + e) // 2
CLUSTER_FRAME = {
    "HoxA": dict(chrom="chr6",  ant=_G(52_105_366, 52_108_316), post=_G(52_208_852, 52_210_874),
                 genes={9: (52_174_053, 52_177_369), 10: (52_182_000, 52_186_000),
                        11: (52_193_000, 52_196_000), 13: (52_208_852, 52_210_874)},
                 interp={10, 11}),
    "HoxB": dict(chrom="chr11", ant=_G(96_227_072, 96_229_567), post=_G(96_055_630, 96_057_924),
                 genes={9: (96_132_644, 96_137_907), 13: (96_055_630, 96_057_924)}, interp=set()),
    "HoxC": dict(chrom="chr15", ant=_G(102_864_826, 102_867_283), post=_G(102_751_562, 102_759_245),
                 genes={9: (102_807_463, 102_814_875), 13: (102_751_562, 102_759_245)}, interp=set()),
    "HoxD": dict(chrom="chr2",  ant=_G(74_601_037, 74_603_199), post=_G(74_506_282, 74_509_660),
                 genes={9: (74_535_820, 74_538_265), 10: (74_529_000, 74_532_000),
                        11: (74_520_000, 74_527_000), 13: (74_506_282, 74_509_660)},
                 interp={10, 11}),
}


def _load_adult(chrom: str, lo: int, hi: int) -> List[Tuple[int, float]]:
    """(midpoint, methylation%) of adult LMRs on chrom overlapping [lo,hi]."""
    out = []
    with open(ADULT_LMRS) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if c[0] != chrom:
                continue
            s, e = int(c[1]), int(c[2])
            if s < hi and e > lo:
                try:
                    out.append(((s + e) // 2, float(c[4])))
                except (ValueError, IndexError):
                    pass
    return out


def _colinear(p: int, ant: int, post: int) -> float:
    """Anterior(0) -> posterior(1) fraction; handles either cluster orientation."""
    h = (p - ant) / (post - ant)
    return max(0.0, min(1.0, h))


def cluster_paralog_positions(name: str) -> Dict[int, Dict]:
    """For one cluster, the enhancer-weighted colinear position of each posterior
    determinant paralog, read from its adult fossil-record enhancer archive."""
    spec = CLUSTER_FRAME[name]
    chrom, ant, post = spec["chrom"], spec["ant"], spec["post"]
    lo, hi = min(ant, post) - PAD, max(ant, post) + PAD
    lmrs = _load_adult(chrom, lo, hi)
    gene_mid = {k: _G(*v) for k, v in spec["genes"].items()}
    buckets: Dict[int, List[Tuple[int, float]]] = {k: [] for k in gene_mid}
    for mid, meth in lmrs:                         # Voronoi: assign each enhancer to nearest paralog
        k = min(gene_mid, key=lambda g: abs(mid - gene_mid[g]))
        buckets[k].append((mid, meth))
    out = {}
    for k, hits in buckets.items():
        if not hits:
            continue
        w = [(LMR_MAX - m) / LMR_MAX for _, m in hits]   # deeper hypomethylation = stronger archive
        wsum = sum(w) or 1.0
        centroid = sum(mid * wi for (mid, _), wi in zip(hits, w)) / wsum
        out[k] = dict(h=_colinear(int(centroid), ant, post), n=len(hits),
                      centroid=int(centroid), interp=(k in spec["interp"]))
    return out


def colinear_curve():
    """The genome's colinear map paralog-number -> colinear fraction h in [0,1], read from the mm9 Hox
    cluster GENE spacing (3' anterior -> 5' posterior). Anchor points: Hox1 (the anterior terminal) = 0,
    Hox13 (posterior terminal) = 1, and Hox9/10/11 at their mean gene-mid colinear fraction across the
    clusters that carry them. This is the genome's own antero-posterior code -- the SHAPE of the Hox->AP
    relation, no fit to any anatomy. Returns sorted [(paralog, h), ...]."""
    per = {1: [0.0], 13: [1.0]}
    for name, spec in CLUSTER_FRAME.items():
        for p, span in spec["genes"].items():
            per.setdefault(p, []).append(_colinear(_G(*span), spec["ant"], spec["post"]))
    pts = sorted((p, sum(v) / len(v)) for p, v in per.items())
    # enforce monotonicity (colinearity is monotone by construction; guard tiny cross-cluster noise)
    out = [list(pts[0])]
    for p, h in pts[1:]:
        out.append([p, max(h, out[-1][1])])
    return [(p, h) for p, h in out]


def colinear_fraction(paralog_number: float) -> float:
    """Genome colinear fraction h in [0,1] for a (possibly fractional, activity-weighted) paralog number,
    linearly interpolated on the mm9 colinear_curve. h(paralog)=body AP under the raw Hox-domain frame."""
    import numpy as np
    cur = colinear_curve()
    ps = [p for p, _ in cur]; hs = [h for _, h in cur]
    return float(np.interp(paralog_number, ps, hs))


def limb_ap_from_fossil(anterior_anchor: float = 0.0,
                        posterior_anchor: float = 1.0, verbose: bool = False):
    """Forelimb / hindlimb AP levels read from the posterior-Hox enhancer archive.

    Returns dict(fore_ap, hind_ap, fore_h, hind_h, detail). fore_ap/hind_ap are
    mapped onto the body between anterior_anchor (anterior Hox limit) and
    posterior_anchor (tail); with the defaults (0,1) they ARE the raw colinear
    fractions -- the drop-in replacement for hox_limb_levels(0.0, 1.0)."""
    detail = {name: cluster_paralog_positions(name) for name in CLUSTER_FRAME}

    def _read(name, paralog):
        """Colinear h of a determinant paralog in one cluster: the ENHANCER-archive
        centroid when the fossil record has one, else the real mm9 GENE position
        within the same colinear frame (the gut archive is silent at the Hox10 limb
        enhancer, so hindlimb falls back to the gene coordinate). Returns (h, source)."""
        spec = CLUSTER_FRAME[name]
        if paralog in detail[name]:
            return detail[name][paralog]["h"], "enhancer"
        if paralog in spec["genes"]:
            gmid = _G(*spec["genes"][paralog])
            return _colinear(gmid, spec["ant"], spec["post"]), "gene"
        return None, None

    fore, hind = [], []
    for name in LIMB_CLUSTERS:
        hf, sf = _read(name, FORE_PARALOG)
        hh, sh = _read(name, HIND_PARALOG)
        if hf is not None:
            fore.append((name, hf, sf))
        if hh is not None:
            hind.append((name, hh, sh))
    fore_h = sum(h for _, h, _ in fore) / len(fore) if fore else None
    hind_h = sum(h for _, h, _ in hind) / len(hind) if hind else None

    def _map(h):
        return None if h is None else anterior_anchor + h * (posterior_anchor - anterior_anchor)

    res = dict(fore_ap=_map(fore_h), hind_ap=_map(hind_h), fore_h=fore_h, hind_h=hind_h,
               fore_src=fore, hind_src=hind, detail=detail)
    if verbose:
        print(__doc__.split("Run:")[0].rstrip())
        print("=" * 74)
        for name, pos in detail.items():
            print(f"\n{name}  (frame {CLUSTER_FRAME[name]['chrom']}: "
                  f"ant={CLUSTER_FRAME[name]['ant']:,} -> post={CLUSTER_FRAME[name]['post']:,})")
            for k in sorted(pos):
                d = pos[k]
                tag = "~" if d["interp"] else " "
                role = " <- FORELIMB" if k == FORE_PARALOG else (" <- HINDLIMB" if k == HIND_PARALOG else "")
                print(f"   Hox{k:<2d}{tag} colinear h={d['h']:.3f}  "
                      f"(enh archive n={d['n']}, centroid {d['centroid']:,}){role}")
        print("\n" + "=" * 74)
        fsrc = ", ".join(f"{n}:{h:.3f}({s})" for n, h, s in fore)
        hsrc = ", ".join(f"{n}:{h:.3f}({s})" for n, h, s in hind)
        print(f"forelimb Hox{FORE_PARALOG}: colinear h = {fore_h:.3f}   [{fsrc}]")
        print(f"hindlimb Hox{HIND_PARALOG}: colinear h = {hind_h:.3f}   [{hsrc}]")
        print(f"\n-> AP address on the body [anterior={anterior_anchor}, tail={posterior_anchor}]:")
        print(f"     fore_ap = {res['fore_ap']:.4f}     hind_ap = {res['hind_ap']:.4f}")
        print("   (previous idealised-human colinearity gave fore=0.456, hind=0.753)")
    return res


def main():
    limb_ap_from_fossil(0.0, 1.0, verbose=True)


if __name__ == "__main__":
    main()
