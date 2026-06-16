"""
Unified Enhancer-Promoter (E-P) Query Interface
================================================

Combines ABC model predictions and UCSC regulatory tracks
to compute attention weights from real genomic data.

This replaces learned attention with COMPUTED attention based on:
- ABC scores (Activity × Contact)
- Hi-C contact frequencies
- H3K27ac chromatin marks
- Genomic distance
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from medic.genome.abc_client import ABCClient, ABCPrediction
from medic.genome.ucsc_client import UCSCClient, UCSCEnhancer

logger = logging.getLogger(__name__)


@dataclass
class EnhancerGeneLink:
    """Unified enhancer-gene interaction with computed attention weight."""
    enhancer_chr: str
    enhancer_start: int
    enhancer_end: int
    gene_name: str
    gene_tss: int

    # Computed attention weight (0-1)
    attention_weight: float

    # Contributing factors
    abc_score: float          # ABC model prediction (0-1)
    contact_frequency: float  # Hi-C contact (0-1)
    activity_signal: float    # H3K27ac activity (0-1)
    distance: int             # Genomic distance (bp)

    # Metadata
    cell_type: str
    source: str  # "ABC" or "UCSC" or "combined"

    def __repr__(self):
        return (
            f"EnhancerGeneLink({self.enhancer_chr}:{self.enhancer_start}-{self.enhancer_end} "
            f"→ {self.gene_name}, attention={self.attention_weight:.4f})"
        )


class EPInterface:
    """Unified interface for enhancer-promoter interactions."""

    def __init__(
        self,
        abc_client: Optional[ABCClient] = None,
        ucsc_client: Optional[UCSCClient] = None,
        genome: str = "hg38"
    ):
        """
        Initialize E-P interface.

        Args:
            abc_client: Optional pre-loaded ABC client
            ucsc_client: Optional UCSC client
            genome: Genome build (default: hg38)
        """
        # Initialize ABC client
        if abc_client is None:
            logger.info("Initializing ABC client...")
            self.abc = ABCClient()
            self.abc.load()  # Load full dataset
        else:
            self.abc = abc_client

        # Initialize UCSC client
        if ucsc_client is None:
            self.ucsc = UCSCClient(genome=genome)
        else:
            self.ucsc = ucsc_client

        self.genome = genome

        logger.info("E-P Interface initialized")

    def compute_attention_weight(
        self,
        abc_score: float,
        contact_freq: float,
        activity: float,
        distance: int,
        max_distance: int = 1000000
    ) -> float:
        """
        Compute attention weight from genomic features.

        This is the KEY INNOVATION: attention weights computed from
        real genomic data instead of learned from training.

        Formula:
            attention = (
                0.4 * abc_score +
                0.3 * contact_freq +
                0.2 * activity +
                0.1 * distance_decay
            )

        Args:
            abc_score: ABC model score (0-1)
            contact_freq: Hi-C contact frequency (normalized 0-1)
            activity: H3K27ac activity signal (normalized 0-1)
            distance: Enhancer-TSS distance (bp)
            max_distance: Maximum distance for normalization

        Returns:
            Attention weight (0-1)
        """
        # Normalize distance (closer = higher weight)
        distance_decay = max(0.0, 1.0 - (distance / max_distance))

        # Weighted combination
        attention = (
            0.4 * abc_score +
            0.3 * contact_freq +
            0.2 * activity +
            0.1 * distance_decay
        )

        # Clip to [0, 1]
        return max(0.0, min(1.0, attention))

    def get_gene_enhancers(
        self,
        gene_name: str,
        gene_region: Tuple[str, int, int],
        cell_types: Optional[List[str]] = None,
        min_abc_score: float = 0.02,
        window: int = 1000000,
        use_ucsc: bool = True
    ) -> List[EnhancerGeneLink]:
        """
        Get all enhancers regulating a gene with computed attention weights.

        Args:
            gene_name: Gene symbol (e.g., "NKX2-5")
            gene_region: (chr, start, end) of gene TSS
            cell_types: Cell types to query (e.g., ["cardiac_muscle_cell-ENCODE"])
            min_abc_score: Minimum ABC score threshold
            window: Search window around gene (bp)
            use_ucsc: Whether to supplement with UCSC cCREs

        Returns:
            List of EnhancerGeneLink objects with computed attention weights
        """
        chr, tss_start, tss_end = gene_region
        tss_center = (tss_start + tss_end) // 2

        links = []

        # 1. Get ABC predictions
        abc_predictions = self.abc.get_gene_enhancers(
            gene_name=gene_name,
            cell_types=cell_types,
            min_abc_score=min_abc_score
        )

        logger.info(f"Found {len(abc_predictions)} ABC predictions for {gene_name}")

        for pred in abc_predictions:
            # Normalize activity signal (assume max ~50 based on NKX2-5 example)
            activity_norm = min(1.0, pred.activity / 50.0)

            # Normalize contact frequency (assume max ~0.5)
            contact_norm = min(1.0, pred.contact / 0.5)

            # Compute attention weight
            attention = self.compute_attention_weight(
                abc_score=pred.abc_score,
                contact_freq=contact_norm,
                activity=activity_norm,
                distance=abs(pred.distance),
                max_distance=window
            )

            link = EnhancerGeneLink(
                enhancer_chr=pred.enhancer_chr,
                enhancer_start=pred.enhancer_start,
                enhancer_end=pred.enhancer_end,
                gene_name=pred.gene_name,
                gene_tss=pred.gene_tss,
                attention_weight=attention,
                abc_score=pred.abc_score,
                contact_frequency=contact_norm,
                activity_signal=activity_norm,
                distance=abs(pred.distance),
                cell_type=pred.cell_type,
                source="ABC"
            )
            links.append(link)

        # 2. Optionally supplement with UCSC cCREs
        if use_ucsc and len(links) < 10:  # Only if ABC found few enhancers
            search_start = max(0, tss_center - window)
            search_end = tss_center + window

            ccres = self.ucsc.query_region(
                track="encodeCcreCombined",
                chrom=chr,
                start=search_start,
                end=search_end
            )

            logger.info(f"Found {len(ccres)} UCSC cCREs for {gene_name}")

            # Convert cCREs to links (with lower confidence)
            for ccre in ccres:
                # Only use distal enhancer-like signatures (dELS)
                if "dELS" not in ccre.enhancer_type:
                    continue

                ccre_center = (ccre.start + ccre.end) // 2
                distance = abs(ccre_center - tss_center)

                # Estimate activity from z-score (normalize to 0-1)
                activity_norm = min(1.0, ccre.z_score / 5.0)

                # No ABC score or contact data, so use lower weight
                attention = self.compute_attention_weight(
                    abc_score=0.0,  # No ABC prediction
                    contact_freq=0.0,  # No Hi-C data
                    activity=activity_norm,
                    distance=distance,
                    max_distance=window
                )

                # Only add if not already covered by ABC
                already_covered = any(
                    link.enhancer_chr == ccre.chrom and
                    abs(link.enhancer_start - ccre.start) < 5000
                    for link in links
                )

                if not already_covered and attention > 0.05:
                    link = EnhancerGeneLink(
                        enhancer_chr=ccre.chrom,
                        enhancer_start=ccre.start,
                        enhancer_end=ccre.end,
                        gene_name=gene_name,
                        gene_tss=tss_center,
                        attention_weight=attention,
                        abc_score=0.0,
                        contact_frequency=0.0,
                        activity_signal=activity_norm,
                        distance=distance,
                        cell_type="ENCODE_cCRE",
                        source="UCSC"
                    )
                    links.append(link)

        # Sort by attention weight (highest first)
        links.sort(key=lambda x: x.attention_weight, reverse=True)

        logger.info(
            f"Total E-G links for {gene_name}: {len(links)} "
            f"(ABC: {sum(1 for l in links if l.source == 'ABC')}, "
            f"UCSC: {sum(1 for l in links if l.source == 'UCSC')})"
        )

        return links

    def get_super_enhancer_genes(
        self,
        se_region: Tuple[str, int, int],
        cell_types: Optional[List[str]] = None,
        min_attention: float = 0.02
    ) -> Dict[str, List[EnhancerGeneLink]]:
        """
        Get all genes regulated by a super-enhancer region.

        Args:
            se_region: (chr, start, end) of super-enhancer
            cell_types: Cell types to query
            min_attention: Minimum attention weight threshold

        Returns:
            Dict mapping gene_name → list of EnhancerGeneLink objects
        """
        chr, start, end = se_region

        # Query ABC for all E-G links in this region
        abc_predictions = self.abc.get_enhancer_genes(
            enhancer_region=se_region,
            cell_types=cell_types,
            min_abc_score=min_attention / 2.5,  # Lower threshold since we'll filter by attention
            window=5000
        )

        # Group by gene and compute attention weights
        genes = {}
        for pred in abc_predictions:
            activity_norm = min(1.0, pred.activity / 50.0)
            contact_norm = min(1.0, pred.contact / 0.5)

            attention = self.compute_attention_weight(
                abc_score=pred.abc_score,
                contact_freq=contact_norm,
                activity=activity_norm,
                distance=abs(pred.distance),
                max_distance=1000000
            )

            if attention < min_attention:
                continue

            link = EnhancerGeneLink(
                enhancer_chr=pred.enhancer_chr,
                enhancer_start=pred.enhancer_start,
                enhancer_end=pred.enhancer_end,
                gene_name=pred.gene_name,
                gene_tss=pred.gene_tss,
                attention_weight=attention,
                abc_score=pred.abc_score,
                contact_frequency=contact_norm,
                activity_signal=activity_norm,
                distance=abs(pred.distance),
                cell_type=pred.cell_type,
                source="ABC"
            )

            if pred.gene_name not in genes:
                genes[pred.gene_name] = []
            genes[pred.gene_name].append(link)

        # Sort each gene's links by attention
        for gene_name in genes:
            genes[gene_name].sort(key=lambda x: x.attention_weight, reverse=True)

        logger.info(
            f"SE {chr}:{start}-{end} regulates {len(genes)} genes "
            f"(total links: {sum(len(links) for links in genes.values())})"
        )

        return genes


if __name__ == "__main__":
    # Test the unified E-P interface
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("UNIFIED E-P INTERFACE TEST")
    print("="*70)

    # Initialize interface
    print("\nInitializing E-P interface (loading ABC data)...")
    ep = EPInterface()

    # Test: Get enhancers for NKX2-5
    print("\n" + "-"*70)
    print("TEST 1: Get enhancers for NKX2-5 (cardiac gene)")
    print("-"*70)

    nkx25_links = ep.get_gene_enhancers(
        gene_name="NKX2-5",
        gene_region=("chr5", 172662315, 172662315),  # TSS
        cell_types=["cardiac_muscle_cell-ENCODE", "heart_ventricle-ENCODE"],
        min_abc_score=0.02,
        use_ucsc=True
    )

    print(f"\nFound {len(nkx25_links)} E-G links for NKX2-5:")
    for i, link in enumerate(nkx25_links[:5], 1):
        print(f"\n{i}. {link.enhancer_chr}:{link.enhancer_start}-{link.enhancer_end}")
        print(f"   Attention weight: {link.attention_weight:.4f}")
        print(f"   - ABC score:   {link.abc_score:.4f} (×0.4)")
        print(f"   - Contact:     {link.contact_frequency:.4f} (×0.3)")
        print(f"   - Activity:    {link.activity_signal:.4f} (×0.2)")
        print(f"   - Distance:    {link.distance/1000:.1f}kb (×0.1)")
        print(f"   Cell type: {link.cell_type}")
        print(f"   Source: {link.source}")

    # Test: Get genes regulated by a super-enhancer
    print("\n" + "-"*70)
    print("TEST 2: Get genes regulated by NKX2-5 super-enhancer region")
    print("-"*70)

    se_genes = ep.get_super_enhancer_genes(
        se_region=("chr5", 172655000, 172670000),  # ~15kb SE region near NKX2-5
        cell_types=["heart_ventricle-ENCODE"],
        min_attention=0.02
    )

    print(f"\nSuper-enhancer regulates {len(se_genes)} genes:")
    for gene_name, links in list(se_genes.items())[:5]:
        print(f"\n{gene_name}:")
        for link in links[:2]:  # Show top 2 links per gene
            print(f"  - Attention: {link.attention_weight:.4f}, Distance: {link.distance/1000:.1f}kb")

    print("\n" + "="*70)
