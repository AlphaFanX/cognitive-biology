"""
Kernel readers: decoupling the KERNEL from its SUBSTRATE READER
===============================================================

The vertebrate zygote kernel (zygote_kernel.py) is read from one substrate -- CpG
methylation depth (Jadhav fossil record). But that substrate was independently LOST in
nematodes, flatworms, and dipteran insects, so a methylation-bound kernel cannot port
across the animal tree. The fix (kernel/reader decoupling): the KERNEL is a frozen,
heritable regulatory GROUND STATE -- concretely, an archive of chromatin OPENNESS at each
developmental stage -- and a READER is whatever assay recovers that openness in a given
lineage.

The unifying observation, made explicit here in code:

    hypomethylation depth  and  ATAC accessibility  are TWO READERS OF THE SAME LATENT.

Both estimate per-region chromatin openness. Methylation reads it as a PROXY (a low-
methylated region is open); ATAC reads it DIRECTLY (an accessible region is open). The
zygote-kernel weight formula -- weight = normalized openness, in [0,1] -- is therefore
substrate-agnostic; only the assay that fills it in changes.

  - MethylationReader : openness = (LMR_MAX - methyl%) / LMR_MAX        [vertebrates, annelids, cnidaria, bees]
  - AccessibilityReader: openness = ATAC_signal / ATAC_MAX             [UNIVERSAL -- every eukaryote]

Stratum (embryonic / fetal / adult) is assigned identically by either reader: the EARLIEST
developmental stage at which the region crosses the 'open' threshold (open early = embryonic;
opens only later = fetal/adult). AlphaGenome predicts accessibility from sequence, so the
AccessibilityReader can be SUPPLIED for any genome with no methylation data at all.

Run: python -m medic.genome.kernel_reader
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence
from medic.genome.zygote_kernel import KernelStratum, LMR_MAX

# Accessibility scale: ATAC fold-enrichment at which a region is fully "open".
ATAC_MAX = 8.0
ATAC_OPEN = 2.0          # fold-enrichment threshold to call a region open
METH_OPEN = LMR_MAX      # below this %-methylation a region is open (= LMR ceiling)

# Developmental stage order used for stratum assignment.
STAGE_STRATUM = [KernelStratum.EMBRYONIC, KernelStratum.FETAL, KernelStratum.ADULT]


@dataclass
class RegionSignal:
    """A regulatory region's substrate measurement across developmental stages.

    Provide EITHER a methylation trajectory OR an accessibility trajectory (or both,
    for the concordance demo). Each is a 3-tuple over (early, mid, adult); None = unmeasured.
    """
    chrom: str
    start: int
    end: int
    methylation: Optional[Sequence[Optional[float]]] = None   # %-methylation per stage
    accessibility: Optional[Sequence[Optional[float]]] = None # ATAC fold-enrich per stage


class KernelReader(ABC):
    """Maps a region's substrate signal to (openness weight in [0,1], stratum)."""

    @abstractmethod
    def openness(self, value: float) -> float: ...
    @abstractmethod
    def is_open(self, value: float) -> bool: ...
    @abstractmethod
    def _trajectory(self, sig: RegionSignal) -> Sequence[Optional[float]]: ...

    def read_weight(self, sig: RegionSignal) -> Optional[float]:
        """Openness at the region's STAGE OF USE = the first stage it crosses open.

        Mirrors zygote_kernel.derive_weight: an embryonic enhancer's weight is its
        openness when embryonic (earliest open stage), not its (closed) state earlier.
        Returns None if the region never opens (outside the recoverable kernel).
        """
        traj = self._trajectory(sig)
        for v in traj:
            if v is not None and self.is_open(v):
                return self.openness(v)
        return None

    def read_stratum(self, sig: RegionSignal) -> Optional[KernelStratum]:
        """Earliest developmental stage at which the region crosses 'open'."""
        traj = self._trajectory(sig)
        for i, v in enumerate(traj):
            if v is not None and self.is_open(v):
                return STAGE_STRATUM[min(i, len(STAGE_STRATUM) - 1)]
        return None   # never opens -> outside the recoverable kernel


class MethylationReader(KernelReader):
    """Vertebrate-native reader: openness inferred from hypomethylation depth."""
    name = "methylation"
    def openness(self, meth: float) -> float:
        return max(METH_OPEN - meth, 0.0) / METH_OPEN          # more hypomethylated = more open
    def is_open(self, meth: float) -> bool:
        return meth < METH_OPEN
    def _trajectory(self, sig: RegionSignal):
        if sig.methylation is None:
            raise ValueError("MethylationReader requires a methylation trajectory")
        return sig.methylation


class AccessibilityReader(KernelReader):
    """UNIVERSAL reader: openness measured directly as chromatin accessibility (ATAC).

    Works for any eukaryote regardless of methylation status; AlphaGenome can predict
    the accessibility trajectory from sequence alone, supplying this reader for any genome.
    """
    name = "accessibility"
    def openness(self, atac: float) -> float:
        return max(min(atac, ATAC_MAX), 0.0) / ATAC_MAX        # more accessible = more open
    def is_open(self, atac: float) -> bool:
        return atac >= ATAC_OPEN
    def _trajectory(self, sig: RegionSignal):
        if sig.accessibility is None:
            raise ValueError("AccessibilityReader requires an accessibility trajectory")
        return sig.accessibility


def _demo():
    # Three regions, each measured by BOTH substrates. Methylation and accessibility
    # trajectories are constructed to encode the SAME openness story per region, so the
    # two readers should agree on stratum and give correlated weights -- the equivalence.
    regions = [
        # opens EARLY (embryonic): hypomethylated at E12.5 / accessible at stage 0
        RegionSignal("chrA", 100, 600, methylation=[20.0, 15.0, 12.0], accessibility=[6.5, 7.0, 7.2]),
        # opens MID (fetal): methylated early, hypomethylated by E16.5 / accessible mid
        RegionSignal("chrB", 800, 1300, methylation=[80.0, 30.0, 25.0], accessibility=[0.8, 4.5, 5.0]),
        # opens LATE (adult): open only in adult
        RegionSignal("chrC", 1500, 2000, methylation=[85.0, 78.0, 35.0], accessibility=[0.5, 0.9, 3.5]),
        # never opens: outside the recoverable kernel under either reader
        RegionSignal("chrD", 2200, 2700, methylation=[90.0, 88.0, 86.0], accessibility=[0.3, 0.4, 0.5]),
    ]
    mr, ar = MethylationReader(), AccessibilityReader()
    print("=" * 84)
    print("KERNEL/READER DECOUPLING -- methylation vs accessibility read the SAME openness latent")
    print("=" * 84)
    print(f"  {'region':7s} | {'methylation reader':28s} | {'accessibility reader':28s} | concordant?")
    print("-" * 84)
    agree = 0
    for r in regions:
        ms, mw = mr.read_stratum(r), mr.read_weight(r)
        as_, aw = ar.read_stratum(r), ar.read_weight(r)
        same = (ms == as_)
        agree += same
        msn = ms.name if ms is not None else "none"
        asn = as_.name if as_ is not None else "none"
        mw = mw if mw is not None else 0.0
        aw = aw if aw is not None else 0.0
        print(f"  {r.chrom:7s} | stratum={msn:9s} w={mw:.2f}        | stratum={asn:9s} w={aw:.2f}        | {'YES' if same else 'no'}")
    print("-" * 84)
    print(f"stratum concordance: {agree}/{len(regions)} regions  --  the two readers recover the same kernel.")
    print("Methylation is the vertebrate-convenient proxy; accessibility (ATAC) is the universal reader,")
    print("and AlphaGenome predicts it from sequence -> a kernel for any genome, methylation or not.")


if __name__ == "__main__":
    _demo()
