#!/usr/bin/env python3
"""
Zygote Kernel: the stratified, tagged DNA fossil record (Jacobs hypothesis, 2026-06-02).

The zygote kernel x_0 is NOT a flat genome but a *stratified record* of every
regulatory element the lineage will ever use -- a developmental fossil record
written in DNA methylation.

Grounded in:
    Jadhav, U. et al. (2019) "Extensive Recovery of Embryonic Enhancer and Gene
    Memory Stored in Hypomethylated Enhancer DNA." Molecular Cell 74(3):542-554.e5.
    DOI 10.1016/j.molcel.2019.02.024  (mouse intestinal epithelium)
    Data: GEO GSE111024 and GSE115541 (WGBS, ATAC-seq, RNA-seq, ChIP-seq).

Core finding: ~90% of enhancers used in development stay HYPOMETHYLATED in adult
cells long after every other chromatin mark is erased. The kernel resolves into
three superimposed strata, distinguished by which marks survive decommissioning:

    EMBRYONIC : LMR only                  -- hypomethylation is the SOLE trace
    FETAL     : LMR + H3K4me1             -- primed monomethyl, transient
    ADULT     : LMR + H3K4me1 + H3K27ac   -- active now (Acetylation Engine)

THE CLOCK READS THE KERNEL; THEY COUPLE AT PRC2.
PRC2/H3K27me3 silences the *promoters* of the bivalent TF genes (not the
enhancers themselves). As the TERRA-gated PRC2 mask withdraws (telomeres shorten
-> TERRA reach drops -> PRC2 retreats), those TFs (e.g. the FOX family)
de-repress and reactivate the hypomethylated enhancers in REVERSE developmental
order: fetal strata first, embryonic strata second. The telomere clock therefore
does not generate the temporal coordinate -- it INDEXES INTO one already
inscribed in the kernel.

Orthogonality: this kernel is the *where-in-developmental-history* axis carried
by the chromosome, fully independent of the bioelectric V_m map that supplies
*where-in-the-body*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


class KernelStratum(IntEnum):
    """Developmental strata in chronological order of first use.

    Reverse of this order is the PRC2-withdrawal *reactivation* order
    (fetal reactivates before embryonic; adult is never silenced).
    """
    ADULT = 0       # active now: LMR + H3K4me1 + H3K27ac
    FETAL = 1       # LMR + H3K4me1 (reactivates first on PRC2 loss)
    EMBRYONIC = 2   # LMR only (reactivates last on PRC2 loss)


# Methylation thresholds (percent), consistent with JadhavEnhancerParser.
UMR_MAX = 10.0   # below: unmethylated (promoter)
LMR_MAX = 50.0   # below: low-methylated region (enhancer archive)

# PRC2 occupancy (1.0 = full silencing, 0.0 = full withdrawal) at which each
# stratum reactivates. Higher threshold = reactivates earlier in PRC2 retreat,
# i.e. reverse developmental order (fetal before embryonic).
FETAL_REACTIVATION_PRC2 = 0.50
EMBRYONIC_REACTIVATION_PRC2 = 0.20

# Telomere length (bp) -> PRC2 coupling. Full length keeps PRC2 fully engaged.
ZYGOTE_TELOMERE_BP = 10000.0


@dataclass
class TaggedEnhancer:
    """One regulatory element tagged in the methylation fossil record."""
    chrom: str
    start: int
    end: int
    target_gene: Optional[str] = None

    # Methylation trajectory (% methylation; None = not measured at that stage)
    methylation_e12: Optional[float] = None
    methylation_e16: Optional[float] = None
    methylation_adult: Optional[float] = None

    # Surviving histone marks in the adult cell
    has_h3k4me1: bool = False
    has_h3k27ac: bool = False

    # Continuous MEASURED mark intensities (ChIP fold-enrichment; None = not scored).
    # These are the quantitative signal that the boolean has_* flags binarize away.
    h3k4me1_signal: Optional[float] = None
    h3k27ac_signal: Optional[float] = None

    # Derived
    stratum: Optional[KernelStratum] = None
    active: bool = False  # transcriptionally live under current PRC2 level
    # Regulatory weight DERIVED from measured data (not fitted). See derive_weight().
    weight: Optional[float] = None

    def derive_weight(self) -> Optional[float]:
        """Derive a regulatory weight from MEASURED METHYLATION DEPTH only.

        The frozen, inherited tag in the fossil record is the *hypomethylation*
        itself (Jadhav 2019): it persists for life after every histone mark is
        erased. Histone marks (H3K4me1 / H3K27ac) are dynamic, ADULT-cell
        activity signals -- they belong to the EXTERNAL ABC/SEdb LLM that emits
        the LoRA, NOT to this frozen zygote base (two-MLP separation, 2026-06-04).
        So the weight is purely the depth BELOW the LMR ceiling at the
        developmental stage the stratum was in use:

            EMBRYONIC : E12.5 depth  (earliest in use), then E16.5, then adult
            FETAL     : E16.5 depth  (mid stage),       then E12.5, then adult
            ADULT     : adult depth,                    then E16.5, then E12.5

        weight = max(LMR_MAX - methylation%, 0) / LMR_MAX  in [0, 1]
        (more hypomethylated = more open = higher weight).

        h3k4me1_signal / h3k27ac_signal are retained on the dataclass for
        stratum CLASSIFICATION and for the adult-overlap validation against
        ABC/SEdb -- but they are deliberately NOT read here. Returns the raw
        (un-normalized) weight; real_kernel normalizes within each stratum.
        None only when no methylation of any stage is available.
        """
        def depth(meth: Optional[float]) -> Optional[float]:
            return max(LMR_MAX - meth, 0.0) / LMR_MAX if meth is not None else None

        # Methylation preference order per stratum (stage-of-use first).
        order = {
            KernelStratum.EMBRYONIC: (self.methylation_e12, self.methylation_e16,
                                      self.methylation_adult),
            KernelStratum.FETAL:     (self.methylation_e16, self.methylation_e12,
                                      self.methylation_adult),
            KernelStratum.ADULT:     (self.methylation_adult, self.methylation_e16,
                                      self.methylation_e12),
        }.get(self.stratum, (self.methylation_adult, self.methylation_e16,
                             self.methylation_e12))

        for m in order:
            w = depth(m)
            if w is not None:
                self.weight = w
                return self.weight
        self.weight = None
        return self.weight

    def is_lmr_adult(self) -> bool:
        """Hypomethylated (archived) in the adult -> the surviving tag."""
        return (
            self.methylation_adult is not None
            and UMR_MAX <= self.methylation_adult < LMR_MAX
        )

    def assign_stratum(self) -> Optional[KernelStratum]:
        """Classify into a stratum by which marks survive (Jadhav scheme).

        Primary signal = surviving histone marks. When marks are unavailable
        (no ChIP-seq loaded), fall back to the methylation *trajectory*: an
        enhancer already hypomethylated at E12.5 is embryonic; one that becomes
        hypomethylated only by E16.5 is fetal; one hypomethylated only in the
        adult is adult.
        """
        archived = self.is_lmr_adult()

        # --- Mark-based (preferred) ---
        if self.has_h3k27ac and archived:
            self.stratum = KernelStratum.ADULT
        elif self.has_h3k4me1 and archived:
            self.stratum = KernelStratum.FETAL
        elif archived:
            self.stratum = KernelStratum.EMBRYONIC
        else:
            # Fully methylated (HMR) or never archived -> not in recoverable kernel
            self.stratum = None
            return None

        # --- Methylation-trajectory refinement, ONLY when marks are absent AND
        # trajectory data is actually present. Without marks or trajectory data
        # an archived LMR stays EMBRYONIC (set by the elif above) -- it must not
        # default to ADULT. ---
        have_traj = self.methylation_e12 is not None or self.methylation_e16 is not None
        if not (self.has_h3k4me1 or self.has_h3k27ac) and have_traj:
            e12_low = self.methylation_e12 is not None and self.methylation_e12 < LMR_MAX
            e16_low = self.methylation_e16 is not None and self.methylation_e16 < LMR_MAX
            if e12_low:
                self.stratum = KernelStratum.EMBRYONIC
            elif e16_low:
                self.stratum = KernelStratum.FETAL
            else:
                self.stratum = KernelStratum.ADULT

        return self.stratum


class ZygoteKernel:
    """The stratified, tagged kernel x_0 read by the TERRA-PRC2 clock."""

    # The existing DevelopmentalEnhancerDB 4-way classes map onto the strata:
    #   active   -> ADULT      (LMR + H3K4me1 + H3K27ac, transcribing now)
    #   poised   -> FETAL      (primed; H3K4me1 without H3K27ac)
    #   archived -> EMBRYONIC  (LMR-only developmental memory)
    #   silenced -> None       (HMR; outside the recoverable kernel)
    # NB: cleanly separating FETAL from EMBRYONIC needs H3K4me1 ChIP-seq
    # (GEO GSE115541); without it the DB's "poised" only approximates fetal.
    CLASS_TO_STRATUM = {
        "active": KernelStratum.ADULT,
        "poised": KernelStratum.FETAL,
        "archived": KernelStratum.EMBRYONIC,
        "silenced": None,
    }

    def __init__(self, enhancers: Optional[List[TaggedEnhancer]] = None,
                 restratify: bool = True):
        self.enhancers: List[TaggedEnhancer] = enhancers or []
        self.prc2_level: float = 1.0  # zygote: full PRC2 silencing
        if restratify:
            self.stratify()
        self.set_prc2_level(1.0)

    # ---- construction -------------------------------------------------
    def stratify(self) -> Dict[KernelStratum, int]:
        """Assign every enhancer to a stratum; return per-stratum counts."""
        counts: Dict[KernelStratum, int] = {s: 0 for s in KernelStratum}
        for e in self.enhancers:
            s = e.assign_stratum()
            if s is not None:
                counts[s] += 1
        return counts

    @classmethod
    def from_developmental_db(cls, db) -> "ZygoteKernel":
        """Build from a populated DevelopmentalEnhancerDB (integrate_data run).

        Strata are taken from the DB's own enhancer_class via CLASS_TO_STRATUM,
        so the 4-way scheme (active/poised/archived/silenced) and the 3 tiers
        stay consistent: active->adult, poised->fetal, archived->embryonic,
        silenced->outside the kernel. assign_stratum() (mark / methylation-
        trajectory based) remains the standalone path used by demo() and raw
        WGBS parsing when no enhancer_class is present.
        """
        tagged: List[TaggedEnhancer] = []
        for chrom_enhancers in db.integrated.values():
            for e in chrom_enhancers:
                te = TaggedEnhancer(
                    chrom=e.chrom,
                    start=e.start,
                    end=e.end,
                    target_gene=e.target_gene,
                    methylation_e12=e.methylation_e12,
                    methylation_e16=e.methylation_e16,
                    methylation_adult=e.methylation_adult,
                    has_h3k4me1=bool(getattr(e, "has_h3k4me1", False)),
                    has_h3k27ac=(e.h3k27ac_activity > 0.1),
                )
                te.stratum = cls.CLASS_TO_STRATUM.get(
                    getattr(e, "enhancer_class", None), None
                )
                tagged.append(te)
        return cls(tagged, restratify=False)

    # ---- the clock reads the kernel (couple at PRC2) ------------------
    def set_prc2_level(self, level: float) -> None:
        """Set PRC2 occupancy in [0,1] and update which strata are live.

        Reverse developmental order: ADULT always live; FETAL reactivates as PRC2
        drops below FETAL_REACTIVATION_PRC2; EMBRYONIC reactivates only once PRC2
        drops below EMBRYONIC_REACTIVATION_PRC2.
        """
        self.prc2_level = max(0.0, min(1.0, level))
        for e in self.enhancers:
            if e.stratum == KernelStratum.ADULT:
                e.active = True
            elif e.stratum == KernelStratum.FETAL:
                e.active = self.prc2_level < FETAL_REACTIVATION_PRC2
            elif e.stratum == KernelStratum.EMBRYONIC:
                e.active = self.prc2_level < EMBRYONIC_REACTIVATION_PRC2
            else:
                e.active = False

    def set_from_telomere_length(self, telomere_bp: float) -> None:
        """Couple to the telomere clock: longer telomere -> more TERRA -> more PRC2.

        prc2_level = telomere_bp / ZYGOTE_TELOMERE_BP (clamped). As telomeres
        shorten with division, PRC2 retreats and strata reactivate in reverse
        developmental order -- the kernel is indexed by the clock.
        """
        self.set_prc2_level(telomere_bp / ZYGOTE_TELOMERE_BP)

    # ---- introspection ------------------------------------------------
    def active_enhancers(self) -> List[TaggedEnhancer]:
        return [e for e in self.enhancers if e.active]

    def stratum_counts(self) -> Dict[KernelStratum, int]:
        counts: Dict[KernelStratum, int] = {s: 0 for s in KernelStratum}
        for e in self.enhancers:
            if e.stratum is not None:
                counts[e.stratum] += 1
        return counts

    def reactivation_order(self) -> List[KernelStratum]:
        """Order in which strata come online as PRC2 withdraws."""
        return [KernelStratum.ADULT, KernelStratum.FETAL, KernelStratum.EMBRYONIC]

    def summary(self) -> str:
        c = self.stratum_counts()
        n_active = len(self.active_enhancers())
        return (
            f"ZygoteKernel: {len(self.enhancers)} tagged enhancers | "
            f"embryonic={c[KernelStratum.EMBRYONIC]} fetal={c[KernelStratum.FETAL]} "
            f"adult={c[KernelStratum.ADULT]} | PRC2={self.prc2_level:.2f} "
            f"active={n_active}"
        )

    @classmethod
    def demo(cls) -> "ZygoteKernel":
        """Small synthetic kernel for testing without the GEO download.

        One enhancer per stratum, classified purely by methylation trajectory.
        """
        return cls([
            # embryonic: hypomethylated already at E12.5, archived in adult
            TaggedEnhancer("chr1", 1000, 1400, "FOXA2",
                           methylation_e12=22.0, methylation_e16=28.0, methylation_adult=32.0),
            # fetal: becomes hypomethylated by E16.5, archived in adult
            TaggedEnhancer("chr2", 5000, 5400, "HNF1B",
                           methylation_e12=70.0, methylation_e16=30.0, methylation_adult=35.0),
            # adult: hypomethylated + active H3K27ac
            TaggedEnhancer("chr3", 9000, 9400, "HNF4A",
                           methylation_e12=75.0, methylation_e16=60.0, methylation_adult=18.0,
                           has_h3k4me1=True, has_h3k27ac=True),
            # silenced: fully methylated, not in recoverable kernel
            TaggedEnhancer("chr4", 2000, 2400, "MYOD1",
                           methylation_e12=85.0, methylation_e16=88.0, methylation_adult=92.0),
        ])


def main():
    """Demonstrate reverse-order reactivation as the telomere clock runs down."""
    kernel = ZygoteKernel.demo()
    print(kernel.summary())
    print("Strata:", {s.name: n for s, n in kernel.stratum_counts().items()})
    print("\nReactivation as telomeres shorten (PRC2 withdraws):")
    for telomere in [10000, 6000, 4000, 1000]:
        kernel.set_from_telomere_length(telomere)
        live = sorted({e.stratum.name for e in kernel.active_enhancers()})
        print(f"  telomere={telomere:>6} bp | PRC2={kernel.prc2_level:.2f} | live strata: {live}")


if __name__ == "__main__":
    main()
