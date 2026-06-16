"""
Telomere Clock: TERRA-PRC2-Hox Developmental Timing Mechanism
=============================================================

This module implements Miles Jacobs' hypothesis (2026-01-19) that telomere
length serves as a developmental clock through TERRA-mediated PRC2 recruitment.

Key Mechanism:
1. Telomeres shorten with each cell division
2. TERRA (Telomeric Repeat-containing RNA) is transcribed from telomeres
3. TERRA recruits PRC2 (Polycomb Repressive Complex 2) to target genes
4. Hox genes are arranged colinearly: 3' (telomeric) to 5' (centromeric)
5. As telomeres shorten, TERRA reach decreases
6. PRC2 silencing withdraws progressively from 5' Hox genes
7. Hox genes activate in sequence: 3' (anterior) -> 5' (posterior)

The chromosome itself becomes the developmental clock.

Literature Support:
- TERRA correlates r=0.8 with pluripotency (Nature Cell Research 2011)
- Chromosome looping reaches 10Mb when telomeres long (PMC4233240)
- TERRA recruits PRC2 to pluripotency genes (eLife 2019)
- Telomeric proximity = precocity of Hox activation (Dev 2024)

References:
- elifesciences.org/articles/44656 (TERRA-TRF1-PRC2)
- pmc.ncbi.nlm.nih.gov/articles/PMC4233240/ (TPE over long distances)
- journals.biologists.com/dev/article/151/16/dev202508 (Hox colinearity 2024)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# Human Hox gene clusters with chromosomal positions
# Arranged 3' (telomeric, anterior) to 5' (centromeric, posterior)
HOX_CLUSTERS = {
    'HOXA': {
        'chromosome': 'chr7',
        'start': 27090000,  # 3' end (telomeric)
        'end': 27210000,    # 5' end (centromeric)
        'genes': [
            # 3' genes (anterior, activate FIRST)
            {'name': 'HOXA1', 'position': 27092000, 'body_region': 'hindbrain'},
            {'name': 'HOXA2', 'position': 27100000, 'body_region': 'hindbrain'},
            {'name': 'HOXA3', 'position': 27120000, 'body_region': 'pharynx'},
            {'name': 'HOXA4', 'position': 27130000, 'body_region': 'cervical'},
            {'name': 'HOXA5', 'position': 27140000, 'body_region': 'cervical'},
            {'name': 'HOXA6', 'position': 27150000, 'body_region': 'thoracic'},
            {'name': 'HOXA7', 'position': 27160000, 'body_region': 'thoracic'},
            {'name': 'HOXA9', 'position': 27170000, 'body_region': 'lumbar'},
            {'name': 'HOXA10', 'position': 27180000, 'body_region': 'lumbar'},
            {'name': 'HOXA11', 'position': 27190000, 'body_region': 'sacral'},
            # 5' genes (posterior, activate LAST)
            {'name': 'HOXA13', 'position': 27200000, 'body_region': 'caudal'},
        ]
    },
    'HOXB': {
        'chromosome': 'chr17',
        'start': 48510000,
        'end': 48720000,
        'genes': [
            {'name': 'HOXB1', 'position': 48520000, 'body_region': 'hindbrain'},
            {'name': 'HOXB2', 'position': 48550000, 'body_region': 'hindbrain'},
            {'name': 'HOXB3', 'position': 48580000, 'body_region': 'pharynx'},
            {'name': 'HOXB4', 'position': 48600000, 'body_region': 'cervical'},
            {'name': 'HOXB5', 'position': 48620000, 'body_region': 'cervical'},
            {'name': 'HOXB6', 'position': 48640000, 'body_region': 'thoracic'},
            {'name': 'HOXB7', 'position': 48660000, 'body_region': 'thoracic'},
            {'name': 'HOXB8', 'position': 48680000, 'body_region': 'thoracic'},
            {'name': 'HOXB9', 'position': 48700000, 'body_region': 'lumbar'},
            {'name': 'HOXB13', 'position': 48710000, 'body_region': 'caudal'},
        ]
    },
    'HOXC': {
        'chromosome': 'chr12',
        'start': 53950000,
        'end': 54150000,
        'genes': [
            {'name': 'HOXC4', 'position': 53960000, 'body_region': 'cervical'},
            {'name': 'HOXC5', 'position': 53980000, 'body_region': 'cervical'},
            {'name': 'HOXC6', 'position': 54000000, 'body_region': 'thoracic'},
            {'name': 'HOXC8', 'position': 54050000, 'body_region': 'thoracic'},
            {'name': 'HOXC9', 'position': 54080000, 'body_region': 'lumbar'},
            {'name': 'HOXC10', 'position': 54100000, 'body_region': 'lumbar'},
            {'name': 'HOXC11', 'position': 54120000, 'body_region': 'sacral'},
            {'name': 'HOXC12', 'position': 54130000, 'body_region': 'sacral'},
            {'name': 'HOXC13', 'position': 54140000, 'body_region': 'caudal'},
        ]
    },
    'HOXD': {
        'chromosome': 'chr2',
        'start': 176150000,
        'end': 176400000,
        'genes': [
            {'name': 'HOXD1', 'position': 176160000, 'body_region': 'hindbrain'},
            {'name': 'HOXD3', 'position': 176200000, 'body_region': 'pharynx'},
            {'name': 'HOXD4', 'position': 176230000, 'body_region': 'cervical'},
            {'name': 'HOXD8', 'position': 176280000, 'body_region': 'thoracic'},
            {'name': 'HOXD9', 'position': 176310000, 'body_region': 'lumbar'},
            {'name': 'HOXD10', 'position': 176340000, 'body_region': 'lumbar'},
            {'name': 'HOXD11', 'position': 176360000, 'body_region': 'sacral'},
            {'name': 'HOXD12', 'position': 176380000, 'body_region': 'sacral'},
            {'name': 'HOXD13', 'position': 176390000, 'body_region': 'caudal'},
        ]
    }
}

# Body regions in anterior-posterior order
BODY_REGIONS = [
    'hindbrain',   # Most anterior (Hox1-2)
    'pharynx',     # Hox3
    'cervical',    # Hox4-5
    'thoracic',    # Hox6-8
    'lumbar',      # Hox9-10
    'sacral',      # Hox11-12
    'caudal',      # Most posterior (Hox13)
]


# Hox cluster regulatory regions (super-enhancers and TADs)
# These are PRE-WIRED in the genome but require chromatin decompaction to form loops
# Literature: PNAS 2020 (doi:10.1073/pnas.2015083117), Development 2012
HOX_REGULATORY_REGIONS = {
    'HOXD': {
        'chromosome': 'chr2',
        'cluster_start': 176150000,
        'cluster_end': 176400000,
        'regulatory_regions': [
            {
                'name': 'GCR',  # Global Control Region
                'type': 'super_enhancer',
                'start': 176570000,  # 180kb 5' of Hoxd13
                'end': 176610000,    # ~40kb region
                'size_kb': 40,
                'distance_from_hoxd13_kb': 180,
                'target_genes': ['HOXD13', 'HOXD12', 'HOXD11', 'HOXD10'],
                'function': 'Late phase digit development',
                'active_in': ['distal_limb', 'digits'],
            },
            {
                'name': 'ELCR',  # Early Limb Control Region
                'type': 'super_enhancer',
                'start': 176100000,  # 3' of cluster (T-DOM)
                'end': 176120000,
                'size_kb': 20,
                'distance_from_hoxd13_kb': 50,
                'target_genes': ['HOXD9', 'HOXD10', 'HOXD11'],
                'function': 'Early colinear activation in limb bud',
                'active_in': ['early_limb_bud', 'proximal_limb'],
            },
            {
                'name': 'T-DOM',  # Telomeric TAD
                'type': 'tad',
                'start': 175650000,  # 3' of cluster
                'end': 176150000,
                'size_kb': 500,
                'contains': ['ELCR', 'forearm_enhancers'],
                'function': 'Proximal limb/forearm regulation',
            },
            {
                'name': 'C-DOM',  # Centromeric TAD
                'type': 'tad',
                'start': 176400000,  # 5' of cluster
                'end': 176900000,
                'size_kb': 500,
                'contains': ['GCR', 'digit_enhancers'],
                'function': 'Distal limb/digit regulation',
            },
        ]
    },
    'HOXA': {
        'chromosome': 'chr7',
        'cluster_start': 27090000,
        'cluster_end': 27210000,
        'regulatory_regions': [
            {
                'name': 'HOXA_SE',  # HOXA super-enhancer region
                'type': 'super_enhancer',
                'start': 27220000,
                'end': 27280000,
                'size_kb': 60,
                'target_genes': ['HOXA13', 'HOXA11', 'HOXA10', 'HOXA9'],
                'function': 'Posterior body patterning',
                'active_in': ['posterior_mesoderm', 'limb'],
            },
        ]
    },
    'HOXB': {
        'chromosome': 'chr17',
        'cluster_start': 48510000,
        'cluster_end': 48720000,
        'regulatory_regions': [
            {
                'name': 'HOXB_SE',
                'type': 'super_enhancer',
                'start': 48730000,
                'end': 48780000,
                'size_kb': 50,
                'target_genes': ['HOXB13', 'HOXB9', 'HOXB8'],
                'function': 'Axial patterning',
                'active_in': ['spinal_cord', 'somites'],
            },
        ]
    },
    'HOXC': {
        'chromosome': 'chr12',
        'cluster_start': 53950000,
        'cluster_end': 54150000,
        'regulatory_regions': [
            {
                'name': 'HOXC_SE',
                'type': 'super_enhancer',
                'start': 54160000,
                'end': 54200000,
                'size_kb': 40,
                'target_genes': ['HOXC13', 'HOXC12', 'HOXC11'],
                'function': 'Posterior body and hair follicle',
                'active_in': ['skin', 'posterior_mesoderm'],
            },
        ]
    },
}


@dataclass
class TelomereState:
    """State of a cell's telomeres."""

    # Average telomere length in base pairs
    # Human newborn: ~10,000 bp, Adult: ~5,000-8,000 bp
    length_bp: float = 10000.0

    # Loss per division (typically 50-200 bp)
    loss_per_division: float = 100.0

    # Telomerase activity (0 = none, 1 = fully active)
    # Stem cells have ~0.3, somatic cells ~0.0
    telomerase_activity: float = 0.0

    # Division count
    divisions: int = 0

    # Hayflick limit (divisions before senescence)
    hayflick_limit: int = 50

    def divide(self):
        """Simulate one cell division."""
        # Telomere shortening (reduced by telomerase)
        shortening = self.loss_per_division * (1.0 - self.telomerase_activity)
        self.length_bp = max(0, self.length_bp - shortening)
        self.divisions += 1

    def get_relative_length(self, max_length: float = 15000.0) -> float:
        """Get telomere length normalized to 0-1."""
        return min(1.0, self.length_bp / max_length)

    def is_senescent(self) -> bool:
        """Check if cell has reached Hayflick limit."""
        return self.divisions >= self.hayflick_limit or self.length_bp < 2000


@dataclass
class TERRAState:
    """State of TERRA lncRNA transcription and reach."""

    # TERRA expression level (correlates with telomere length)
    expression_level: float = 1.0

    # Maximum chromosomal reach in bp (up to 10Mb when telomeres long)
    max_reach_bp: int = 10_000_000

    # Current effective reach
    current_reach_bp: int = 10_000_000

    def update_from_telomere(self, telomere: TelomereState):
        """Update TERRA state based on telomere length."""
        # TERRA expression scales with telomere length
        rel_length = telomere.get_relative_length()
        self.expression_level = rel_length

        # Chromosomal reach decreases as telomeres shorten
        # When telomeres at 100%: reach = 10Mb
        # When telomeres at 50%: reach = 5Mb
        # When telomeres at 0%: reach = 0
        self.current_reach_bp = int(self.max_reach_bp * rel_length)


@dataclass
class PRC2State:
    """State of Polycomb Repressive Complex 2.

    PRC2 has TWO functions:
    1. Gene silencing via H3K27me3 marks
    2. Chromatin compaction (prevents enhancer-gene loops)
    """

    # H3K27me3 mark intensity (silencing strength)
    silencing_strength: float = 1.0

    # Genes currently silenced by PRC2
    silenced_genes: List[str] = field(default_factory=list)

    # Silencing reach (determined by TERRA)
    silencing_reach_bp: int = 10_000_000

    # Chromatin compaction level (0 = open, 1 = fully compacted)
    # Compacted chromatin prevents enhancer-gene loop formation
    compaction_level: float = 1.0


@dataclass
class ChromatinState:
    """
    Chromatin compaction state for a Hox cluster region.

    PRC2 compacts chromatin, preventing super-enhancer loops from forming.
    When PRC2 withdraws, chromatin decompacts and loops can form.

    Literature: Development 2012 - "loss of H3K27me3 and chromatin decompaction
    over HoxD in the distal posterior limb"
    """

    # Cluster name (e.g., 'HOXD')
    cluster: str = ''

    # Compaction level per region (0 = open/accessible, 1 = compacted/closed)
    # Keys are gene names or region names
    compaction_map: Dict[str, float] = field(default_factory=dict)

    # Loop formation status (which enhancer-gene loops have formed)
    active_loops: List[Tuple[str, str]] = field(default_factory=list)  # (enhancer, gene)

    def get_compaction(self, gene_name: str) -> float:
        """Get compaction level for a gene region."""
        return self.compaction_map.get(gene_name, 1.0)

    def is_loop_formed(self, enhancer: str, gene: str) -> bool:
        """Check if an enhancer-gene loop has formed."""
        return (enhancer, gene) in self.active_loops


@dataclass
class SuperEnhancerState:
    """
    State of a super-enhancer and its target gene loops.

    Super-enhancers are PRE-WIRED in the genome but require:
    1. PRC2 withdrawal (chromatin decompaction)
    2. Loop formation to contact target genes

    Literature: PNAS 2020 - "GCR spatially co-localizes with 5' HoxD
    specifically in the distal posterior region"
    """

    # Enhancer name (e.g., 'GCR', 'ELCR')
    name: str = ''

    # Whether the enhancer region is accessible (PRC2 removed)
    is_accessible: bool = False

    # Target genes this enhancer can activate
    target_genes: List[str] = field(default_factory=list)

    # Which loops have formed (subset of target_genes)
    active_loops: List[str] = field(default_factory=list)

    # Expression boost when loop is active (multiplicative)
    expression_boost: float = 10.0

    def can_form_loop(self, gene: str, chromatin: ChromatinState) -> bool:
        """
        Check if a loop can form to a target gene.
        Requires both enhancer accessibility AND gene region decompaction.
        """
        if not self.is_accessible:
            return False
        if gene not in self.target_genes:
            return False
        # Gene region must be decompacted (< 0.5 compaction)
        return chromatin.get_compaction(gene) < 0.5

    def get_activation_boost(self, gene: str) -> float:
        """Get expression boost for a gene if loop is active."""
        if gene in self.active_loops:
            return self.expression_boost
        return 1.0


class TelomereClock:
    """
    Implements the telomere-TERRA-PRC2-Hox developmental clock.

    The clock mechanism:
    1. Cell divides -> telomere shortens
    2. TERRA reach decreases proportionally
    3. PRC2 silencing withdraws from distal (5') Hox genes
    4. Chromatin decompacts where PRC2 withdraws
    5. Pre-wired super-enhancers form loops to target genes
    6. Hox genes activate with super-enhancer boost
    7. Body axis patterns anterior -> posterior

    Key insight: Super-enhancers are ALWAYS present but locked by PRC2.
    The telomere is the key that unlocks sequential access.
    """

    def __init__(self, initial_telomere_length: float = 10000.0):
        """
        Initialize the telomere clock.

        Args:
            initial_telomere_length: Starting telomere length in bp
        """
        self.telomere = TelomereState(length_bp=initial_telomere_length)
        self.terra = TERRAState()
        self.prc2 = PRC2State()

        # Track Hox gene activation state
        self.hox_activation = {}  # gene_name -> activation_level (0-1)
        self._initialize_hox_genes()

        # Track chromatin state per cluster
        self.chromatin_states = {}  # cluster_name -> ChromatinState
        self._initialize_chromatin_states()

        # Track super-enhancer states
        self.super_enhancers = {}  # (cluster, se_name) -> SuperEnhancerState
        self._initialize_super_enhancers()

        logger.info(f"TelomereClock initialized: {initial_telomere_length} bp")

    def _initialize_chromatin_states(self):
        """Initialize chromatin as fully compacted for all clusters."""
        for cluster_name, cluster in HOX_CLUSTERS.items():
            compaction_map = {}
            for gene in cluster['genes']:
                compaction_map[gene['name']] = 1.0  # Fully compacted
            self.chromatin_states[cluster_name] = ChromatinState(
                cluster=cluster_name,
                compaction_map=compaction_map
            )

    def _initialize_super_enhancers(self):
        """Initialize super-enhancers as inaccessible (locked by PRC2)."""
        for cluster_name, reg_info in HOX_REGULATORY_REGIONS.items():
            for region in reg_info['regulatory_regions']:
                if region['type'] == 'super_enhancer':
                    se_key = (cluster_name, region['name'])
                    self.super_enhancers[se_key] = SuperEnhancerState(
                        name=region['name'],
                        is_accessible=False,
                        target_genes=region.get('target_genes', []),
                        expression_boost=10.0  # 10x boost when loop forms
                    )

    def _initialize_hox_genes(self):
        """Initialize all Hox genes as silenced (0)."""
        for cluster_name, cluster in HOX_CLUSTERS.items():
            for gene in cluster['genes']:
                self.hox_activation[gene['name']] = 0.0

    def divide(self) -> Dict[str, float]:
        """
        Simulate one cell division and return updated Hox activation.

        The complete mechanism:
        1. Telomere shortens
        2. TERRA reach decreases
        3. PRC2 silencing reach decreases
        4. Chromatin decompacts where PRC2 withdraws
        5. Super-enhancers become accessible
        6. Enhancer-gene loops form
        7. Hox genes activate with SE boost

        Returns:
            Dict mapping gene names to activation levels
        """
        # 1. Telomere shortens
        self.telomere.divide()

        # 2. TERRA reach decreases
        self.terra.update_from_telomere(self.telomere)

        # 3. PRC2 silencing reach decreases
        self.prc2.silencing_reach_bp = self.terra.current_reach_bp

        # 4. Update chromatin compaction (decompact where PRC2 withdraws)
        self._update_chromatin_compaction()

        # 5. Update super-enhancer accessibility
        self._update_super_enhancer_accessibility()

        # 6. Form enhancer-gene loops where possible
        self._form_enhancer_gene_loops()

        # 7. Update Hox gene activation (with SE boost)
        self._update_hox_activation()

        logger.debug(
            f"Division {self.telomere.divisions}: "
            f"telomere={self.telomere.length_bp:.0f}bp, "
            f"TERRA reach={self.terra.current_reach_bp/1e6:.1f}Mb"
        )

        return self.hox_activation.copy()

    def _update_chromatin_compaction(self):
        """
        Update chromatin compaction based on PRC2 silencing reach.

        Chromatin decompacts where PRC2 withdraws, allowing enhancer-gene loops.
        Literature: "loss of H3K27me3 and chromatin decompaction over HoxD"
        """
        for cluster_name, cluster in HOX_CLUSTERS.items():
            cluster_start = cluster['start']
            chromatin = self.chromatin_states[cluster_name]

            for gene in cluster['genes']:
                distance_from_3prime = gene['position'] - cluster_start

                # Compaction decreases as gene escapes PRC2 reach
                if distance_from_3prime > self.prc2.silencing_reach_bp / 100:
                    # Gene is beyond PRC2 reach - chromatin decompacts
                    excess = distance_from_3prime - (self.prc2.silencing_reach_bp / 100)
                    # Gradual decompaction (1.0 -> 0.0)
                    decompaction = min(1.0, excess / 50000)
                    chromatin.compaction_map[gene['name']] = 1.0 - decompaction
                else:
                    # Gene still under PRC2 - remains compacted
                    chromatin.compaction_map[gene['name']] = 1.0

    def _update_super_enhancer_accessibility(self):
        """
        Update super-enhancer accessibility based on PRC2 withdrawal.

        Super-enhancers become accessible when their region escapes PRC2.
        """
        for (cluster_name, se_name), se_state in self.super_enhancers.items():
            if cluster_name not in HOX_REGULATORY_REGIONS:
                continue

            reg_info = HOX_REGULATORY_REGIONS[cluster_name]
            cluster_start = HOX_CLUSTERS[cluster_name]['start']

            # Find this SE's position
            for region in reg_info['regulatory_regions']:
                if region['name'] == se_name and region['type'] == 'super_enhancer':
                    # Distance from 3' end to SE
                    se_center = (region['start'] + region['end']) // 2
                    distance_from_3prime = abs(se_center - cluster_start)

                    # SE becomes accessible when beyond PRC2 reach
                    se_state.is_accessible = distance_from_3prime > self.prc2.silencing_reach_bp / 100

    def _form_enhancer_gene_loops(self):
        """
        Form enhancer-gene loops where both enhancer and gene are decompacted.

        Loops require:
        1. Enhancer is accessible (escaped PRC2)
        2. Target gene region is decompacted (< 0.5 compaction)
        """
        for (cluster_name, se_name), se_state in self.super_enhancers.items():
            if not se_state.is_accessible:
                continue

            chromatin = self.chromatin_states.get(cluster_name)
            if chromatin is None:
                continue

            # Try to form loops to each target gene
            for target_gene in se_state.target_genes:
                if target_gene in chromatin.compaction_map:
                    compaction = chromatin.compaction_map[target_gene]
                    # Loop forms when compaction < 0.5
                    if compaction < 0.5 and target_gene not in se_state.active_loops:
                        se_state.active_loops.append(target_gene)
                        chromatin.active_loops.append((se_name, target_gene))
                        logger.debug(f"Loop formed: {se_name} -> {target_gene}")

    def _update_hox_activation(self):
        """
        Update Hox gene activation based on PRC2 silencing reach AND super-enhancer loops.

        Genes beyond the PRC2 reach become activated.
        Genes with active SE loops get a 10x expression boost.
        """
        for cluster_name, cluster in HOX_CLUSTERS.items():
            cluster_start = cluster['start']  # 3' end (telomeric)

            for gene in cluster['genes']:
                gene_name = gene['name']
                # Distance from telomeric (3') end of cluster
                distance_from_3prime = gene['position'] - cluster_start

                # If gene is beyond PRC2 reach, it activates
                if distance_from_3prime > self.prc2.silencing_reach_bp / 100:
                    # Base activation from escaping PRC2
                    excess_distance = distance_from_3prime - (self.prc2.silencing_reach_bp / 100)
                    base_activation = min(1.0, excess_distance / 50000)

                    # Check for super-enhancer boost
                    se_boost = self._get_se_boost_for_gene(cluster_name, gene_name)

                    # Final activation = base * boost (capped at 1.0)
                    # SE boost increases the RATE of activation, not the cap
                    activation = min(1.0, base_activation * se_boost)
                    self.hox_activation[gene_name] = activation
                else:
                    # Gene still within silencing reach - remains repressed
                    self.hox_activation[gene_name] *= 0.95  # Slight decay

    def _get_se_boost_for_gene(self, cluster_name: str, gene_name: str) -> float:
        """
        Get the super-enhancer expression boost for a gene.

        Returns boost factor (1.0 if no active loop, up to 10.0 with active SE).
        """
        total_boost = 1.0

        for (c_name, se_name), se_state in self.super_enhancers.items():
            if c_name == cluster_name and gene_name in se_state.active_loops:
                total_boost *= se_state.expression_boost

        return total_boost

    def get_active_loops(self) -> List[Dict]:
        """
        Get list of all active enhancer-gene loops.

        Returns:
            List of dicts with loop information
        """
        loops = []
        for (cluster_name, se_name), se_state in self.super_enhancers.items():
            for target_gene in se_state.active_loops:
                loops.append({
                    'cluster': cluster_name,
                    'enhancer': se_name,
                    'gene': target_gene,
                    'boost': se_state.expression_boost
                })
        return loops

    def get_chromatin_state_summary(self) -> Dict[str, Dict]:
        """
        Get summary of chromatin compaction states.

        Returns:
            Dict mapping cluster -> {gene: compaction_level}
        """
        summary = {}
        for cluster_name, chromatin in self.chromatin_states.items():
            summary[cluster_name] = {
                'compaction_map': chromatin.compaction_map.copy(),
                'active_loops': chromatin.active_loops.copy(),
                'avg_compaction': np.mean(list(chromatin.compaction_map.values()))
            }
        return summary

    def get_active_body_region(self) -> str:
        """
        Determine the most posterior active body region.

        Returns:
            Body region name (e.g., 'cervical', 'thoracic')
        """
        # Find most posterior (5') activated Hox gene
        most_posterior_region = 'hindbrain'
        max_posterior_idx = 0

        for gene_name, activation in self.hox_activation.items():
            if activation > 0.5:  # Considered active
                # Find the gene's body region
                for cluster in HOX_CLUSTERS.values():
                    for gene in cluster['genes']:
                        if gene['name'] == gene_name:
                            region = gene['body_region']
                            region_idx = BODY_REGIONS.index(region)
                            if region_idx > max_posterior_idx:
                                max_posterior_idx = region_idx
                                most_posterior_region = region

        return most_posterior_region

    def get_hox_expression_pattern(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get the current Hox expression pattern organized by cluster.

        Returns:
            Dict mapping cluster name to list of (gene_name, activation) tuples
        """
        pattern = {}
        for cluster_name, cluster in HOX_CLUSTERS.items():
            genes = []
            for gene in cluster['genes']:
                activation = self.hox_activation.get(gene['name'], 0.0)
                genes.append((gene['name'], activation))
            pattern[cluster_name] = genes
        return pattern

    def get_developmental_stage(self) -> str:
        """
        Infer developmental stage from Hox activation pattern.

        Returns:
            Stage name (e.g., 'gastrulation', 'somitogenesis')
        """
        active_region = self.get_active_body_region()
        region_idx = BODY_REGIONS.index(active_region)

        if region_idx == 0:
            return 'early_gastrulation'
        elif region_idx <= 2:
            return 'late_gastrulation'
        elif region_idx <= 4:
            return 'early_somitogenesis'
        elif region_idx <= 5:
            return 'late_somitogenesis'
        else:
            return 'tail_bud'

    def simulate_development(self, n_divisions: int) -> List[Dict]:
        """
        Simulate multiple divisions and track Hox activation over time.

        Args:
            n_divisions: Number of cell divisions to simulate

        Returns:
            List of state snapshots at each division
        """
        history = []

        for i in range(n_divisions):
            activation = self.divide()

            snapshot = {
                'division': i + 1,
                'telomere_bp': self.telomere.length_bp,
                'terra_reach_mb': self.terra.current_reach_bp / 1e6,
                'active_region': self.get_active_body_region(),
                'stage': self.get_developmental_stage(),
                'hox_activation': activation.copy(),
                'active_loops': self.get_active_loops(),
                'chromatin_summary': {
                    cluster: data['avg_compaction']
                    for cluster, data in self.get_chromatin_state_summary().items()
                }
            }
            history.append(snapshot)

            if self.telomere.is_senescent():
                logger.warning(f"Cell reached senescence at division {i + 1}")
                break

        return history

    def get_anterior_posterior_gradient(self) -> np.ndarray:
        """
        Compute the anterior-posterior Hox activation gradient.

        Returns:
            Array of activation levels from anterior to posterior
        """
        gradient = np.zeros(len(BODY_REGIONS))

        for gene_name, activation in self.hox_activation.items():
            # Find body region for this gene
            for cluster in HOX_CLUSTERS.values():
                for gene in cluster['genes']:
                    if gene['name'] == gene_name:
                        region = gene['body_region']
                        region_idx = BODY_REGIONS.index(region)
                        # Average activation for each region
                        gradient[region_idx] = max(gradient[region_idx], activation)

        return gradient


def compute_differentiation_from_telomere(
    telomere_length: float,
    initial_length: float = 10000.0
) -> Dict[str, float]:
    """
    Compute cell fate probabilities based on telomere length.

    This provides a direct mapping from the "ruler" (telomere)
    to differentiation program probabilities.

    Args:
        telomere_length: Current telomere length in bp
        initial_length: Original telomere length

    Returns:
        Dict mapping body regions to differentiation probability
    """
    # Normalized telomere position (0 = fully shortened, 1 = original)
    position = telomere_length / initial_length

    # Map to body regions (anterior regions need longer telomeres)
    probabilities = {}

    for i, region in enumerate(BODY_REGIONS):
        # Each region has a "window" of telomere length
        # Anterior regions: high telomere = high probability
        # Posterior regions: low telomere = high probability
        region_center = 1.0 - (i / (len(BODY_REGIONS) - 1))
        region_width = 0.2

        # Gaussian-like probability
        distance = abs(position - region_center)
        probability = np.exp(-distance**2 / (2 * region_width**2))
        probabilities[region] = probability

    # Normalize to sum to 1
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v/total for k, v in probabilities.items()}

    return probabilities


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("TELOMERE CLOCK: TERRA-PRC2-Hox Developmental Timing")
    print("=" * 70)
    print("\nMiles Jacobs Hypothesis (2026-01-19):")
    print("Telomere length serves as a developmental clock through")
    print("TERRA-mediated PRC2 recruitment to Hox gene clusters.")
    print("=" * 70)

    # Initialize clock
    clock = TelomereClock(initial_telomere_length=10000.0)

    print(f"\nInitial state:")
    print(f"  Telomere: {clock.telomere.length_bp:.0f} bp")
    print(f"  TERRA reach: {clock.terra.current_reach_bp/1e6:.1f} Mb")
    print(f"  Active region: {clock.get_active_body_region()}")

    # Simulate development
    print("\n" + "-" * 70)
    print("Simulating 30 cell divisions...")
    print("-" * 70)

    history = clock.simulate_development(30)

    # Show key timepoints
    for snapshot in history[::5]:  # Every 5th division
        print(f"\nDivision {snapshot['division']:2d}:")
        print(f"  Telomere: {snapshot['telomere_bp']:.0f} bp")
        print(f"  TERRA reach: {snapshot['terra_reach_mb']:.1f} Mb")
        print(f"  Active region: {snapshot['active_region']}")
        print(f"  Stage: {snapshot['stage']}")

        # Show most activated Hox genes
        active_genes = [
            (name, act) for name, act in snapshot['hox_activation'].items()
            if act > 0.3
        ]
        if active_genes:
            active_genes.sort(key=lambda x: x[1], reverse=True)
            print(f"  Active Hox: {', '.join(f'{n}({a:.2f})' for n, a in active_genes[:4])}")

    # Show final gradient
    print("\n" + "-" * 70)
    print("Final Anterior-Posterior Gradient:")
    print("-" * 70)

    gradient = clock.get_anterior_posterior_gradient()
    for i, region in enumerate(BODY_REGIONS):
        bar = "#" * int(gradient[i] * 30)
        print(f"  {region:12s} [{gradient[i]:.2f}] {bar}")

    # Show super-enhancer loops
    print("\n" + "-" * 70)
    print("Super-Enhancer Loops (Chromatin Decompaction Mechanism):")
    print("-" * 70)

    active_loops = clock.get_active_loops()
    if active_loops:
        print(f"\n  Active SE-Gene Loops ({len(active_loops)} total):")
        for loop in active_loops:
            print(f"    {loop['cluster']}: {loop['enhancer']} -> {loop['gene']} (boost: {loop['boost']}x)")
    else:
        print("\n  No SE-gene loops formed yet (chromatin still compacted)")

    # Show chromatin state
    print("\n  Chromatin Compaction by Cluster:")
    chromatin_summary = clock.get_chromatin_state_summary()
    for cluster, data in chromatin_summary.items():
        avg = data['avg_compaction']
        status = "OPEN" if avg < 0.3 else "PARTIAL" if avg < 0.7 else "COMPACTED"
        print(f"    {cluster}: {avg:.2f} ({status})")

    # Show differentiation probability at various telomere lengths
    print("\n" + "-" * 70)
    print("Telomere -> Differentiation Probability Mapping:")
    print("-" * 70)

    for telomere_pct in [100, 75, 50, 25]:
        telomere_bp = 10000 * telomere_pct / 100
        probs = compute_differentiation_from_telomere(telomere_bp)
        top_region = max(probs, key=probs.get)
        print(f"\n  Telomere {telomere_pct}% ({telomere_bp:.0f} bp):")
        print(f"    Most likely: {top_region} ({probs[top_region]:.2f})")

    print("\n" + "=" * 70)
    print("The chromosome is the developmental clock.")
    print("TERRA is the hand. Hox genes are the hours.")
    print("Super-enhancers are pre-wired. PRC2 is the lock.")
    print("=" * 70)
