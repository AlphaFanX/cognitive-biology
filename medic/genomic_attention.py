"""
Genomic Attention Mechanism
============================

Computes attention weights from real genomic data (ABC scores, Hi-C contacts, H3K27ac)
instead of learning them from training data.

This replaces the traditional learned attention mechanism with a COMPUTED one
based on enhancer-promoter interactions and chromatin state.

Key Innovation:
    Traditional:  attention = MLP(query, key)  # Learned, black box
    Our approach: attention = f(ABC, contact, H3K27ac, distance)  # Computed, interpretable
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenomicAttentionWeights:
    """Attention weights computed from genomic data."""
    gene_name: str
    super_enhancer_region: Tuple[str, int, int]

    # Attention weight (0-1)
    attention_weight: float

    # Contributing factors (for interpretability)
    abc_score: float
    contact_frequency: float
    activity_signal: float
    distance: int

    # Cell type specificity
    cell_type: str

    def __repr__(self):
        return (
            f"GenomicAttention({self.gene_name} ← SE, "
            f"weight={self.attention_weight:.4f})"
        )


class GenomicAttentionComputer:
    """
    Computes attention weights from genomic data.

    This replaces learned attention mechanisms with computed weights based on:
    - ABC model predictions (Activity × Contact)
    - Hi-C contact frequencies (3D genome architecture)
    - H3K27ac chromatin marks (enhancer activity)
    - Genomic distance (linear proximity)
    """

    def __init__(self, ep_interface):
        """
        Initialize genomic attention computer.

        Args:
            ep_interface: EPInterface for querying E-P interactions
        """
        self.ep = ep_interface
        logger.info("GenomicAttentionComputer initialized")

    def compute_gene_se_attention(
        self,
        gene_name: str,
        gene_region: Tuple[str, int, int],
        se_regions: List[Tuple[str, int, int]],
        cell_types: List[str],
        min_attention: float = 0.01
    ) -> Dict[Tuple[str, int, int], GenomicAttentionWeights]:
        """
        Compute attention weights between a gene and multiple super-enhancers.

        This is the KEY function for SE-guided differentiation!

        Args:
            gene_name: Gene symbol (e.g., "NKX2-5")
            gene_region: (chr, start, end) of gene TSS
            se_regions: List of (chr, start, end) super-enhancer regions
            cell_types: Cell types to query (e.g., ["cardiac_muscle_cell-ENCODE"])
            min_attention: Minimum attention threshold

        Returns:
            Dict mapping se_region → GenomicAttentionWeights
        """
        logger.info(
            f"Computing attention: {gene_name} ← {len(se_regions)} SEs "
            f"in cell types: {cell_types}"
        )

        # Get all E-G links for this gene
        gene_links = self.ep.get_gene_enhancers(
            gene_name=gene_name,
            gene_region=gene_region,
            cell_types=cell_types,
            min_abc_score=min_attention / 2.5,  # Lower threshold, we'll filter by attention
            use_ucsc=False  # Use ABC only for high-quality predictions
        )

        # Map each SE region to its attention weight
        se_attention = {}

        for se_region in se_regions:
            se_chr, se_start, se_end = se_region

            # Find links that overlap this SE region
            overlapping_links = [
                link for link in gene_links
                if (link.enhancer_chr == se_chr and
                    not (link.enhancer_end < se_start or link.enhancer_start > se_end))
            ]

            if not overlapping_links:
                # No ABC predictions for this SE, check if SE is tissue-appropriate
                # Default to low weight
                logger.debug(f"No ABC predictions for SE {se_chr}:{se_start}-{se_end}")
                continue

            # Aggregate attention across all overlapping enhancers in this SE
            # Use max (strongest enhancer drives regulation)
            max_link = max(overlapping_links, key=lambda x: x.attention_weight)

            if max_link.attention_weight >= min_attention:
                se_attention[se_region] = GenomicAttentionWeights(
                    gene_name=gene_name,
                    super_enhancer_region=se_region,
                    attention_weight=max_link.attention_weight,
                    abc_score=max_link.abc_score,
                    contact_frequency=max_link.contact_frequency,
                    activity_signal=max_link.activity_signal,
                    distance=max_link.distance,
                    cell_type=max_link.cell_type
                )

        logger.info(
            f"Found {len(se_attention)}/{len(se_regions)} SEs with attention >= {min_attention} "
            f"for {gene_name}"
        )

        return se_attention

    def compute_se_gene_attention(
        self,
        se_region: Tuple[str, int, int],
        marker_genes: List[str],
        cell_types: List[str],
        min_attention: float = 0.02
    ) -> Dict[str, GenomicAttentionWeights]:
        """
        Compute attention from a super-enhancer to multiple genes.

        Used for SE-guided differentiation: which genes does this SE activate?

        Args:
            se_region: (chr, start, end) of super-enhancer
            marker_genes: List of potential target genes
            cell_types: Cell types to query
            min_attention: Minimum attention threshold

        Returns:
            Dict mapping gene_name → GenomicAttentionWeights
        """
        chr, start, end = se_region
        logger.info(
            f"Computing SE → genes attention for {chr}:{start}-{end} "
            f"({len(marker_genes)} candidate genes)"
        )

        # Get all genes regulated by this SE
        se_genes = self.ep.get_super_enhancer_genes(
            se_region=se_region,
            cell_types=cell_types,
            min_attention=min_attention
        )

        # Filter to marker genes only
        gene_attention = {}
        for gene_name in marker_genes:
            if gene_name in se_genes:
                links = se_genes[gene_name]
                # Use strongest link
                max_link = max(links, key=lambda x: x.attention_weight)

                gene_attention[gene_name] = GenomicAttentionWeights(
                    gene_name=gene_name,
                    super_enhancer_region=se_region,
                    attention_weight=max_link.attention_weight,
                    abc_score=max_link.abc_score,
                    contact_frequency=max_link.contact_frequency,
                    activity_signal=max_link.activity_signal,
                    distance=max_link.distance,
                    cell_type=max_link.cell_type
                )

        logger.info(
            f"SE regulates {len(gene_attention)}/{len(marker_genes)} marker genes "
            f"with attention >= {min_attention}"
        )

        return gene_attention

    def compute_fate_scores(
        self,
        active_ses: List[Tuple[str, int, int]],
        fate_marker_genes: Dict[str, List[str]],
        cell_types_by_fate: Dict[str, List[str]],
        aggregation: str = "mean"
    ) -> Dict[str, float]:
        """
        Compute cell fate scores based on SE → marker gene attention.

        This is used for SE-GUIDED DIFFERENTIATION:
        Given active SEs, which cell fate has highest marker gene activation?

        Args:
            active_ses: List of active super-enhancer regions
            fate_marker_genes: Dict mapping fate → list of marker genes
                              e.g., {"heart": ["NKX2-5", "GATA4"], "brain": ["SOX2", "PAX6"]}
            cell_types_by_fate: Dict mapping fate → cell types
                               e.g., {"heart": ["cardiac_muscle_cell-ENCODE"]}
            aggregation: How to aggregate attention across genes ("mean", "max", "sum")

        Returns:
            Dict mapping fate → fate_score (0-1)
        """
        logger.info(
            f"Computing fate scores for {len(fate_marker_genes)} fates "
            f"based on {len(active_ses)} active SEs"
        )

        fate_scores = {}

        for fate, marker_genes in fate_marker_genes.items():
            cell_types = cell_types_by_fate.get(fate, [])

            if not cell_types:
                logger.warning(f"No cell types for fate {fate}, skipping")
                fate_scores[fate] = 0.0
                continue

            # Collect attention weights for all marker genes from all SEs
            all_attention = []

            for se_region in active_ses:
                gene_attention = self.compute_se_gene_attention(
                    se_region=se_region,
                    marker_genes=marker_genes,
                    cell_types=cell_types,
                    min_attention=0.01  # Low threshold to capture all signals
                )

                # Collect attention weights
                for gene_name, attn in gene_attention.items():
                    all_attention.append(attn.attention_weight)

            # Aggregate attention weights
            if not all_attention:
                fate_scores[fate] = 0.0
            elif aggregation == "mean":
                fate_scores[fate] = np.mean(all_attention)
            elif aggregation == "max":
                fate_scores[fate] = np.max(all_attention)
            elif aggregation == "sum":
                fate_scores[fate] = min(1.0, np.sum(all_attention))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        # Normalize to sum to 1 (softmax-like)
        total = sum(fate_scores.values())
        if total > 0:
            fate_scores = {fate: score / total for fate, score in fate_scores.items()}

        logger.info(f"Fate scores: {fate_scores}")

        return fate_scores


if __name__ == "__main__":
    # Test genomic attention computation
    import sys
    sys.path.insert(0, 'C:/Users/jacobsme/cognimed')

    from medic.genome.ep_interface import EPInterface

    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("GENOMIC ATTENTION MECHANISM TEST")
    print("="*70)

    # Initialize
    print("\nInitializing E-P interface...")
    ep = EPInterface()

    print("Initializing genomic attention computer...")
    attention_computer = GenomicAttentionComputer(ep)

    # Test: Compute attention for NKX2-5 from cardiac SEs
    print("\n" + "-"*70)
    print("TEST: NKX2-5 attention from cardiac super-enhancers")
    print("-"*70)

    # Define cardiac super-enhancer regions (from literature/SEdb)
    cardiac_ses = [
        ("chr5", 172655000, 172670000),  # Near NKX2-5
        ("chr5", 172232000, 172250000),  # Distal cardiac SE
    ]

    nkx25_attention = attention_computer.compute_gene_se_attention(
        gene_name="NKX2-5",
        gene_region=("chr5", 172662315, 172662315),
        se_regions=cardiac_ses,
        cell_types=["cardiac_muscle_cell-ENCODE", "heart_ventricle-ENCODE"]
    )

    print(f"\nNKX2-5 attention from {len(cardiac_ses)} SEs:")
    for se_region, attn in nkx25_attention.items():
        chr, start, end = se_region
        print(f"\nSE {chr}:{start}-{end}:")
        print(f"  Attention: {attn.attention_weight:.4f}")
        print(f"  ABC score: {attn.abc_score:.4f}")
        print(f"  Contact:   {attn.contact_frequency:.4f}")
        print(f"  Activity:  {attn.activity_signal:.4f}")
        print(f"  Distance:  {attn.distance/1000:.1f}kb")
        print(f"  Cell type: {attn.cell_type}")

    # Test: Compute fate scores
    print("\n" + "-"*70)
    print("TEST: Fate scores from active cardiac SEs")
    print("-"*70)

    fate_markers = {
        "heart": ["NKX2-5", "GATA4", "HAND2"],
        "brain": ["SOX2", "PAX6", "NEUROD1"],
    }

    cell_types_by_fate = {
        "heart": ["cardiac_muscle_cell-ENCODE", "heart_ventricle-ENCODE"],
        "brain": ["brain-ENCODE"],
    }

    fate_scores = attention_computer.compute_fate_scores(
        active_ses=cardiac_ses,
        fate_marker_genes=fate_markers,
        cell_types_by_fate=cell_types_by_fate,
        aggregation="mean"
    )

    print("\nFate scores (active cardiac SEs):")
    for fate, score in sorted(fate_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {fate}: {score:.4f}")

    print("\n" + "="*70)
