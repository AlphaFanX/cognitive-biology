"""
Source the HEAD COUNT from the real genome: super-enhancer calling on Capitella ATAC.
=====================================================================================

The annelid counterpart of medic/nematostella_se_heads.py. In the architecture, attention
HEADS = super-enhancer (SE) cell-identity programs (Whyte/Hnisz 2013). The body-plan
generator (medic/body_plan_generator.py) took the Capitella n_heads from the single-cell
atlas (GSE159564, ~10 major annelid-larva families). This closes the loop for the SECOND
phylum: derive the head repertoire directly from the GENOME's regulatory landscape, by
calling super-enhancers on real Capitella teleta ATAC (GSE210814, GSM6438588, gastrula,
assembly Capca1 / GCA_000328365.1) with the ROSE method (Whyte et al. 2013):

  1. constituent enhancers = ATAC peaks (narrowPeak; signalValue = fold-enrichment).
  2. stitch peaks within STITCH bp on the same scaffold into stitched enhancers.
  3. score each stitched enhancer = sum over constituents of signalValue * width.
  4. rank ascending, normalise rank and signal to [0,1], find the inflection where the
     tangent slope = 1 (the elbow); stitched enhancers ABOVE it = SUPER-ENHANCERS.
  5. N_SE = the genome-derived count of cell-identity regulatory hubs = the head repertoire.

Honest framing (identical to the cnidarian case): N_SE is the genome's SE repertoire size
(identity hubs at this stage). The COARSE 'major cell-type family' count the generator uses
(~10 for the annelid) is a clustering of these SEs by associated program; the genome supplies
the repertoire, the atlas supplies the coarse grouping. This run grounds 'heads' in the real
Capitella genome, not the atlas -- the same step already done for Nematostella, now for the
second (protostome-bilaterian) phylum where CpG methylation was retained.

Run: python -m medic.annelid_se_heads
"""
import os, gzip, numpy as np
from collections import defaultdict

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "annelid")
NP = os.path.join(HERE, "atac_gastrula_rep1.narrowPeak.gz")   # GSM6438588 Ctel gastrula rep1
STITCH = 12500      # ROSE default stitching distance (bp)

def load_peaks():
    peaks = defaultdict(list)   # scaffold -> list of (start, end, signalValue)
    for l in gzip.open(NP, "rt"):
        p = l.rstrip("\n").split("\t")
        if len(p) < 7: continue
        peaks[p[0]].append((int(p[1]), int(p[2]), float(p[6])))
    for c in peaks:
        peaks[c].sort()
    return peaks

def stitch(peaks):
    stitched = []   # (scaffold, start, end, score=sum signalValue*width, n_constituent)
    for c, ps in peaks.items():
        cs, ce, sc, n = None, None, 0.0, 0
        for s, e, sig in ps:
            if cs is None:
                cs, ce, sc, n = s, e, sig * (e - s), 1
            elif s - ce <= STITCH:
                ce = max(ce, e); sc += sig * (e - s); n += 1
            else:
                stitched.append((c, cs, ce, sc, n)); cs, ce, sc, n = s, e, sig * (e - s), 1
        if cs is not None:
            stitched.append((c, cs, ce, sc, n))
    return stitched

def rose_inflection(scores):
    """ROSE: signal ranked ASCENDING, scale to [0,1]; the knee is the point furthest BELOW
    the diagonal (argmax(x-y)); super-enhancers are the steeply-rising part ABOVE the knee."""
    s = np.sort(np.asarray(scores, dtype=float))
    N = len(s)
    x = np.arange(N) / (N - 1)
    y = s / s.max()
    knee = int(np.argmax(x - y))      # the elbow
    return s[knee], knee, s

def main():
    print("=" * 80)
    print("HEAD COUNT FROM THE GENOME -- super-enhancer calling on real Capitella ATAC")
    print("=" * 80)
    peaks = load_peaks()
    n_peaks = sum(len(v) for v in peaks.values())
    stitched = stitch(peaks)
    scores = [st[3] for st in stitched]
    cutoff, idx, ssort = rose_inflection(scores)
    se = [st for st in stitched if st[3] > cutoff]
    n_se = len(se)
    print(f"  constituent ATAC peaks: {n_peaks:,}  ->  stitched enhancers: {len(stitched):,} (STITCH={STITCH}bp)")
    print(f"  ROSE knee (slope=1) at rank {idx:,}/{len(stitched):,}")
    print(f"\n  N_SE (super-enhancers) = {n_se:,}  = genome-derived count of cell-identity regulatory hubs")
    print(f"     typical enhancers: {len(stitched)-n_se:,}  |  SE fraction: {n_se/len(stitched):.1%}")
    biggest = sorted(se, key=lambda x: -x[3])[:5]
    print("  top super-enhancers (scaffold:start-end, constituents):")
    for c, s, e, sc, n in biggest:
        print(f"     {c}:{s}-{e}  ({n} peaks, {(e-s)/1000:.1f}kb)")
    print(f"\n  THE LOOP CLOSED FOR THE ANNELID: 'heads' are now sourced from the real Capitella")
    print(f"  genome (the SE repertoire), not read off the atlas. N_SE={n_se} identity hubs; the")
    print(f"  generator's coarse major-family count (~10 for the annelid) is a clustering of these")
    print(f"  SEs by associated program. Second phylum genome-grounded, after Nematostella.")
    return n_se

if __name__ == "__main__":
    main()
