"""
Super-Enhancer Guided Cell Differentiation
===========================================

Uses real genomic data (ABC model, Hi-C, H3K27ac) to guide cell fate decisions
from zygote to terminal organs.

Key Innovation:
    Traditional approach: Random initialization → Learn cell fate transitions
    Our approach:         Super-enhancer activity → Computed fate decisions

The super-enhancers ACT AS ATTENTION HEADS that determine which
differentiation path to take based on REAL genomic regulatory data.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DevelopmentalStage(Enum):
    """Developmental stages in human embryogenesis."""
    ZYGOTE = "zygote"              # 0-24 hours
    MORULA = "morula"              # 3-4 days
    BLASTOCYST = "blastocyst"      # 5-6 days
    GASTRULA = "gastrula"          # 2-3 weeks
    ORGANOGENESIS = "organogenesis"  # 3-8 weeks
    FETAL = "fetal"                # 9+ weeks
    TERMINAL = "terminal"          # Fully differentiated


@dataclass
class DifferentiationNode:
    """Node in the differentiation tree."""
    name: str
    uberon_term: str
    parent: Optional[str]
    children: List[str]

    # Marker genes for this cell type
    marker_genes: List[str]

    # Super-enhancer regions active in this cell type
    active_ses: List[Tuple[str, int, int]]

    # Cell types for ABC queries
    abc_cell_types: List[str]

    # Developmental stage
    stage: DevelopmentalStage

    # Terminal organ?
    is_terminal: bool = False


# Human Differentiation Tree (simplified)
# =========================================
#
#                    zygote
#                      |
#                  morula
#                      |
#                 blastocyst
#                  /   |   \
#          trophoblast ICM  primitive_endoderm
#                      |
#                  epiblast
#                   /  |  \
#            ectoderm  mesoderm  endoderm
#             /    \     |    \      /    \
#        neural  epidermis  cardiac  liver  pancreas
#         |                  |
#       brain             heart


DIFFERENTIATION_TREE = {
    "zygote": DifferentiationNode(
        name="zygote",
        uberon_term="UBERON:0000088",
        parent=None,
        children=["morula"],
        marker_genes=["POU5F1", "SOX2", "NANOG"],  # Pluripotency
        active_ses=[],  # No tissue-specific SEs yet
        abc_cell_types=["H1-hESC-ENCODE"],  # Use ESC as proxy
        stage=DevelopmentalStage.ZYGOTE,
        is_terminal=False
    ),

    "morula": DifferentiationNode(
        name="morula",
        uberon_term="UBERON:0000085",
        parent="zygote",
        children=["blastocyst"],
        marker_genes=["POU5F1", "SOX2", "NANOG"],
        active_ses=[],
        abc_cell_types=["H1-hESC-ENCODE"],
        stage=DevelopmentalStage.MORULA,
        is_terminal=False
    ),

    "blastocyst": DifferentiationNode(
        name="blastocyst",
        uberon_term="UBERON:0000358",
        parent="morula",
        children=["epiblast"],  # Simplified - skip trophoblast/ICM
        marker_genes=["POU5F1", "NANOG"],
        active_ses=[],
        abc_cell_types=["H1-hESC-ENCODE"],
        stage=DevelopmentalStage.BLASTOCYST,
        is_terminal=False
    ),

    "epiblast": DifferentiationNode(
        name="epiblast",
        uberon_term="UBERON:0004086",
        parent="blastocyst",
        children=["ectoderm", "mesoderm", "endoderm"],  # Three germ layers
        marker_genes=["POU5F1", "SOX2"],
        # GUESSED: Germ layer specification SEs (real biology uses morphogen gradients)
        # These are ABC-derived enhancers for germ layer marker genes
        active_ses=[
            ("chr6", 166584000, 166590000),   # T/Brachyury SE - MESODERM (ABC=0.285)
            ("chr18", 19745000, 19752000),    # GATA6 SE - ENDODERM (ABC=0.377)
            ("chr3", 181425000, 181435000),   # SOX2 SE - ECTODERM (ABC=0.442)
        ],
        abc_cell_types=["H1-hESC-ENCODE", "H1_BMP4_Derived_Mesendoderm_Cultured_Cells-Roadmap"],
        stage=DevelopmentalStage.GASTRULA,
        is_terminal=False
    ),

    # ECTODERM lineage
    "ectoderm": DifferentiationNode(
        name="ectoderm",
        uberon_term="UBERON:0000924",
        parent="epiblast",
        children=["neural_progenitor"],
        marker_genes=["SOX2", "PAX6", "NEUROD1"],
        active_ses=[
            ("chr3", 181425000, 181435000),  # SOX2 SE (corrected from ABC)
        ],
        abc_cell_types=["brain-ENCODE", "H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap"],
        stage=DevelopmentalStage.GASTRULA,
        is_terminal=False
    ),

    "neural_progenitor": DifferentiationNode(
        name="neural_progenitor",
        uberon_term="UBERON:0005068",
        parent="ectoderm",
        children=["brain"],
        marker_genes=["SOX2", "PAX6", "NEUROD1", "NEUROG2"],
        active_ses=[
            ("chr3", 181425000, 181435000),  # SOX2 SE (corrected from ABC)
            ("chr11", 31806340, 31840000),   # PAX6 SE (expanded range)
        ],
        abc_cell_types=["brain-ENCODE", "H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap"],
        stage=DevelopmentalStage.ORGANOGENESIS,
        is_terminal=False
    ),

    "brain": DifferentiationNode(
        name="brain",
        uberon_term="UBERON:0000955",
        parent="neural_progenitor",
        children=[],
        marker_genes=["MAP2", "TUBB3", "SYN1", "GFAP"],
        active_ses=[
            ("chr3", 181425000, 181435000),  # SOX2 SE (corrected from ABC)
            ("chr11", 31806340, 31840000),   # PAX6 SE (expanded range)
        ],
        abc_cell_types=["brain-ENCODE", "H1_Derived_Neuronal_Progenitor_Cultured_Cells-Roadmap"],
        stage=DevelopmentalStage.TERMINAL,
        is_terminal=True
    ),

    # MESODERM lineage
    "mesoderm": DifferentiationNode(
        name="mesoderm",
        uberon_term="UBERON:0000926",
        parent="epiblast",
        children=["cardiac_progenitor"],
        # T/TBXT (Brachyury) is the key mesoderm marker - matches epiblast SE
        marker_genes=["T", "MESP1", "MIXL1", "EOMES"],
        active_ses=[
            ("chr6", 166584000, 166590000),   # T/Brachyury SE (ABC=0.285)
        ],
        abc_cell_types=["H1_BMP4_Derived_Mesendoderm_Cultured_Cells-Roadmap", "cardiac_muscle_cell-ENCODE"],
        stage=DevelopmentalStage.GASTRULA,
        is_terminal=False
    ),

    "cardiac_progenitor": DifferentiationNode(
        name="cardiac_progenitor",
        uberon_term="UBERON:0004376",
        parent="mesoderm",
        children=["heart"],
        marker_genes=["NKX2-5", "GATA4", "HAND2", "TBX5"],
        active_ses=[
            ("chr5", 172655000, 172670000),  # NKX2-5 SE
        ],
        abc_cell_types=["cardiac_muscle_cell-ENCODE", "heart_ventricle-ENCODE"],
        stage=DevelopmentalStage.ORGANOGENESIS,
        is_terminal=False
    ),

    "heart": DifferentiationNode(
        name="heart",
        uberon_term="UBERON:0000948",
        parent="cardiac_progenitor",
        children=[],
        marker_genes=["NKX2-5", "GATA4", "HAND2", "MYH6", "TNNT2"],
        active_ses=[
            ("chr5", 172655000, 172670000),  # NKX2-5 SE
            ("chr8", 11561000, 11579000),    # GATA4 SE
        ],
        abc_cell_types=["cardiac_muscle_cell-ENCODE", "heart_ventricle-ENCODE"],
        stage=DevelopmentalStage.TERMINAL,
        is_terminal=True
    ),

    # ENDODERM lineage
    "endoderm": DifferentiationNode(
        name="endoderm",
        uberon_term="UBERON:0000925",
        parent="epiblast",
        children=["hepatic_progenitor"],
        # GATA6 is key endoderm marker - matches epiblast SE
        marker_genes=["GATA6", "SOX17", "FOXA2", "CXCR4"],
        active_ses=[
            ("chr18", 19745000, 19752000),    # GATA6 SE (ABC=0.377)
        ],
        abc_cell_types=["H1_BMP4_Derived_Mesendoderm_Cultured_Cells-Roadmap", "liver-ENCODE"],
        stage=DevelopmentalStage.GASTRULA,
        is_terminal=False
    ),

    "hepatic_progenitor": DifferentiationNode(
        name="hepatic_progenitor",
        uberon_term="UBERON:0005928",
        parent="endoderm",
        children=["liver"],
        marker_genes=["HNF4A", "ALB", "AFP"],
        # Liver progenitor SEs from ABC database
        active_ses=[
            ("chr4", 74295000, 74300000),    # AFP SE (ABC=0.231 in HepG2)
            ("chr20", 42978000, 42983000),   # HNF4A SE (ABC=0.149 in HepG2)
        ],
        abc_cell_types=["liver-ENCODE", "HepG2-Roadmap"],
        stage=DevelopmentalStage.ORGANOGENESIS,
        is_terminal=False
    ),

    "liver": DifferentiationNode(
        name="liver",
        uberon_term="UBERON:0002107",
        parent="hepatic_progenitor",
        children=[],
        marker_genes=["ALB", "HNF4A", "CYP3A4", "TTR"],
        # Liver terminal SEs from ABC database
        active_ses=[
            ("chr4", 74260000, 74265000),    # ALB SE (ABC=0.115 in liver-ENCODE)
            ("chr4", 74295000, 74300000),    # AFP SE (ABC=0.231 in HepG2)
        ],
        abc_cell_types=["liver-ENCODE", "HepG2-Roadmap"],
        stage=DevelopmentalStage.TERMINAL,
        is_terminal=True
    ),
}


class SEGuidedDifferentiation:
    """
    Super-enhancer guided cell differentiation.

    Uses genomic attention to guide differentiation from zygote to organs.
    """

    def __init__(self, attention_computer):
        """
        Initialize SE-guided differentiation.

        Args:
            attention_computer: GenomicAttentionComputer instance
        """
        self.attention = attention_computer
        self.tree = DIFFERENTIATION_TREE
        logger.info("SE-guided differentiation initialized")

    def get_available_fates(self, current_node: str) -> List[str]:
        """Get possible differentiation fates from current node."""
        node = self.tree[current_node]
        return node.children

    def compute_differentiation_probabilities(
        self,
        current_node: str,
        active_ses: Optional[List[Tuple[str, int, int]]] = None
    ) -> Dict[str, float]:
        """
        Compute probabilities for each differentiation fate.

        Uses SE → marker gene attention to determine which fate is most likely.

        Args:
            current_node: Current position in differentiation tree
            active_ses: Optional list of active SE regions
                       If None, uses node's default SEs

        Returns:
            Dict mapping fate_name → probability (0-1, sum to 1)
        """
        node = self.tree[current_node]
        children = node.children

        if not children:
            logger.info(f"{current_node} is terminal, no differentiation")
            return {}

        if active_ses is None:
            active_ses = node.active_ses

        if not active_ses:
            # No SE data, use uniform probabilities
            logger.warning(f"No SE data for {current_node}, using uniform probs")
            uniform_prob = 1.0 / len(children)
            return {child: uniform_prob for child in children}

        # Collect marker genes for each fate
        fate_markers = {}
        cell_types_by_fate = {}

        for child_name in children:
            child_node = self.tree[child_name]
            fate_markers[child_name] = child_node.marker_genes
            cell_types_by_fate[child_name] = child_node.abc_cell_types

        # Compute fate scores using genomic attention
        fate_scores = self.attention.compute_fate_scores(
            active_ses=active_ses,
            fate_marker_genes=fate_markers,
            cell_types_by_fate=cell_types_by_fate,
            aggregation="mean"
        )

        logger.info(
            f"Differentiation from {current_node}: {fate_scores}"
        )

        return fate_scores

    def differentiate(
        self,
        current_node: str,
        active_ses: Optional[List[Tuple[str, int, int]]] = None,
        deterministic: bool = False
    ) -> str:
        """
        Execute one differentiation step.

        Args:
            current_node: Current cell type
            active_ses: Active super-enhancer regions
            deterministic: If True, always pick highest probability fate

        Returns:
            Name of next cell type (fate chosen)
        """
        probs = self.compute_differentiation_probabilities(current_node, active_ses)

        if not probs:
            logger.info(f"{current_node} is terminal")
            return current_node

        if deterministic:
            # Pick highest probability
            next_node = max(probs.items(), key=lambda x: x[1])[0]
        else:
            # Sample from probability distribution
            fates = list(probs.keys())
            probabilities = list(probs.values())

            # Ensure probabilities sum to 1 (handle floating point issues)
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            else:
                # All zeros - use uniform distribution
                probabilities = [1.0 / len(fates)] * len(fates)

            next_node = np.random.choice(fates, p=probabilities)

        logger.info(
            f"Differentiation: {current_node} → {next_node} "
            f"(p={probs[next_node]:.4f})"
        )

        return next_node

    def trace_lineage(
        self,
        start_node: str = "zygote",
        target_organ: Optional[str] = None,
        max_steps: int = 20,
        deterministic: bool = True
    ) -> List[str]:
        """
        Trace a full differentiation lineage.

        Args:
            start_node: Starting cell type (default: zygote)
            target_organ: Optional target organ to reach
            max_steps: Maximum differentiation steps
            deterministic: Use deterministic (highest prob) differentiation

        Returns:
            List of cell types in differentiation path
        """
        lineage = [start_node]
        current = start_node

        for step in range(max_steps):
            if self.tree[current].is_terminal:
                logger.info(f"Reached terminal cell type: {current}")
                break

            if target_organ and current == target_organ:
                logger.info(f"Reached target organ: {target_organ}")
                break

            # Differentiate
            next_node = self.differentiate(current, deterministic=deterministic)

            if next_node == current:  # No children
                break

            lineage.append(next_node)
            current = next_node

        return lineage


if __name__ == "__main__":
    # Test SE-guided differentiation
    import sys
    sys.path.insert(0, 'C:/Users/jacobsme/cognimed')

    from medic.genome.ep_interface import EPInterface
    from medic.genomic_attention import GenomicAttentionComputer

    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("SE-GUIDED DIFFERENTIATION TEST")
    print("="*70)

    # Initialize
    print("\nInitializing E-P interface...")
    ep = EPInterface()

    print("Initializing genomic attention computer...")
    attention = GenomicAttentionComputer(ep)

    print("Initializing SE-guided differentiation...")
    differ = SEGuidedDifferentiation(attention)

    # Test 1: Trace lineage to heart
    print("\n" + "-"*70)
    print("TEST 1: Differentiation path zygote -> heart")
    print("-"*70)

    heart_lineage = differ.trace_lineage(
        start_node="zygote",
        target_organ="heart",
        deterministic=True
    )

    print(f"\nDifferentiation path to heart:")
    for i, cell_type in enumerate(heart_lineage, 1):
        node = DIFFERENTIATION_TREE[cell_type]
        print(f"{i}. {cell_type} ({node.stage.value})")

    # Test 2: Trace lineage to brain
    print("\n" + "-"*70)
    print("TEST 2: Differentiation path zygote -> brain")
    print("-"*70)

    brain_lineage = differ.trace_lineage(
        start_node="zygote",
        target_organ="brain",
        deterministic=True
    )

    print(f"\nDifferentiation path to brain:")
    for i, cell_type in enumerate(brain_lineage, 1):
        node = DIFFERENTIATION_TREE[cell_type]
        print(f"{i}. {cell_type} ({node.stage.value})")

    print("\n" + "="*70)
