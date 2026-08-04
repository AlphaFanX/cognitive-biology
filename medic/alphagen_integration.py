"""
AlphaGenome → Human Digital Twin Integration
=============================================

Maps AlphaGenome predictions to organ-specific gene expression and
bioelectric states for the minimal human digital twin.

Pipeline:
1. AlphaGenome: genomic interval → predictions (expression, chromatin, etc.)
2. Gene Expression Mapper: predictions → ion channel expression
3. BETSE Adapter: ion channels → bioelectric state (Vm, Ca2+, etc.)
4. TRM Controller: bioelectric state → optimized parameters
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from .genome.client import AlphaGenomeClient, GenomicInterval
    from .human_topology import HUMAN_ORGANS, OrganSpec
except ImportError:
    from genome.client import AlphaGenomeClient, GenomicInterval
    from human_topology import HUMAN_ORGANS, OrganSpec

logger = logging.getLogger(__name__)


@dataclass
class OrganGeneExpression:
    """Gene expression profile for an organ."""
    organ_name: str
    gene_expression: np.ndarray  # (n_genes,) expression levels
    chromatin_state: np.ndarray  # (n_genes,) accessibility
    ion_channel_expression: Dict[str, float]  # Channel name -> expression level


class AlphaGenomeOrganMapper:
    """Maps AlphaGenome predictions to organ-specific states."""

    def __init__(self, api_key: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize mapper.

        Args:
            api_key: AlphaGenome API key (or use env var)
            mock_mode: Use mock predictions (False by default - will use real API if key available)
        """
        self.client = AlphaGenomeClient(api_key=api_key, mock_mode=mock_mode)
        self.gene_names = self._initialize_gene_panel()

        logger.info(f"AlphaGenome mapper initialized (mock={mock_mode})")

    def _initialize_gene_panel(self) -> List[str]:
        """
        Define the gene panel for ion channels and key regulatory genes.

        These are the genes we'll extract from AlphaGenome predictions.
        """
        return [
            # Sodium channels
            "SCN5A",   # Cardiac Na+ channel
            "SCN1A",   # Neuronal Na+ channel
            "SCN4A",   # Muscle Na+ channel

            # Potassium channels
            "KCNQ1",   # Cardiac K+ channel
            "KCNH2",   # hERG K+ channel
            "KCNJ2",   # Inward rectifier K+

            # Calcium channels
            "CACNA1C", # L-type Ca2+ channel
            "CACNA1A", # P/Q-type Ca2+ channel

            # Gap junctions
            "GJA1",    # Connexin43 (heart, brain)
            "GJB2",    # Connexin26 (cochlea, skin)

            # Chloride channels
            "CFTR",    # Cl- channel (lung, gut)
            "CLCN1",   # Muscle Cl- channel

            # Metabolic genes
            "INS",     # Insulin (pancreas)
            "GCK",     # Glucokinase (liver, pancreas)
            "CYP2D6",  # Liver enzyme

            # Developmental/structural
            "MAPT",    # Tau protein (brain)
            "APOE",    # Apolipoprotein E (brain)
            "MYH2",    # Myosin (muscle)
        ]

    def fetch_organ_data(self, organ_name: str) -> OrganGeneExpression:
        """
        Fetch AlphaGenome data for a specific organ.

        Args:
            organ_name: Name of organ (must be in HUMAN_ORGANS)

        Returns:
            OrganGeneExpression with tissue-specific gene data
        """
        if organ_name not in HUMAN_ORGANS:
            raise ValueError(f"Unknown organ: {organ_name}")

        organ_spec = HUMAN_ORGANS[organ_name]

        logger.info(f"Fetching AlphaGenome data for {organ_spec.name} ({organ_spec.uberon_term})")

        # Get predictions from AlphaGenome for each genomic locus
        all_predictions = []
        for chr, start, end in organ_spec.genomic_loci:
            interval = GenomicInterval(chromosome=chr, start=start, end=end)

            predictions = self.client.predict_interval(
                interval=interval,
                ontology_terms=[organ_spec.uberon_term],
                output_types=["RNA_SEQ", "CHROMATIN"]
            )

            all_predictions.append(predictions)

        # Aggregate predictions across loci
        gene_expression = self._aggregate_expression(all_predictions)
        chromatin_state = self._aggregate_chromatin(all_predictions)

        # Map to ion channel expression levels
        ion_channels = self._map_to_ion_channels(gene_expression, organ_spec)

        return OrganGeneExpression(
            organ_name=organ_name,
            gene_expression=gene_expression,
            chromatin_state=chromatin_state,
            ion_channel_expression=ion_channels
        )

    def _aggregate_expression(self, predictions: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Aggregate RNA-seq predictions across genomic loci.

        Args:
            predictions: List of prediction dicts from AlphaGenome

        Returns:
            (n_genes,) array of expression levels
        """
        # Extract RNA-seq tracks
        rna_tracks = []
        for pred in predictions:
            if "rna_seq" in pred:
                rna_tracks.append(pred["rna_seq"])
            elif "gene_expression" in pred:
                rna_tracks.append(pred["gene_expression"])

        if not rna_tracks:
            # No RNA data - return uniform baseline
            logger.warning("No RNA-seq data in predictions - using baseline")
            return np.ones(len(self.gene_names)) * 0.5

        # Average across loci
        # If tracks are different lengths, take mean of each track separately
        expression_levels = []
        for track in rna_tracks:
            expression_levels.append(np.mean(track))

        # Normalize to [0, 1] range
        expr_array = np.array(expression_levels)
        expr_array = (expr_array - expr_array.min()) / (expr_array.max() - expr_array.min() + 1e-8)

        # Expand to gene panel size (duplicate if needed)
        n_genes = len(self.gene_names)
        if len(expr_array) < n_genes:
            expr_array = np.tile(expr_array, (n_genes // len(expr_array)) + 1)[:n_genes]
        else:
            expr_array = expr_array[:n_genes]

        return expr_array

    def _aggregate_chromatin(self, predictions: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """Aggregate chromatin accessibility across loci."""
        chrom_tracks = []
        for pred in predictions:
            if "dnase" in pred:
                chrom_tracks.append(pred["dnase"])
            elif "h3k27ac" in pred:
                chrom_tracks.append(pred["h3k27ac"])

        if not chrom_tracks:
            return np.ones(len(self.gene_names)) * 0.5

        # Average and normalize
        chrom_levels = [np.mean(track) for track in chrom_tracks]
        chrom_array = np.array(chrom_levels)
        chrom_array = (chrom_array - chrom_array.min()) / (chrom_array.max() - chrom_array.min() + 1e-8)

        # Expand to gene panel size
        n_genes = len(self.gene_names)
        if len(chrom_array) < n_genes:
            chrom_array = np.tile(chrom_array, (n_genes // len(chrom_array)) + 1)[:n_genes]
        else:
            chrom_array = chrom_array[:n_genes]

        return chrom_array

    def _map_to_ion_channels(
        self,
        gene_expression: np.ndarray,
        organ_spec: OrganSpec
    ) -> Dict[str, float]:
        """
        Map gene expression to ion channel conductances.

        Different organs express different channels at different levels.

        Args:
            gene_expression: (n_genes,) expression levels
            organ_spec: Organ specification with properties

        Returns:
            Dict of channel_name -> conductance (mS/cm2)
        """
        # Gene indices (based on gene_names list)
        gene_map = {name: i for i, name in enumerate(self.gene_names)}

        # Base conductances (scaled by expression)
        channels = {}

        # Sodium channels
        if organ_spec.excitable:
            if "SCN5A" in gene_map:
                scn5a_expr = gene_expression[gene_map["SCN5A"]]
                channels["g_Na"] = 120.0 * scn5a_expr  # High for excitable tissue
            else:
                channels["g_Na"] = 120.0 * np.mean(gene_expression[:3])
        else:
            channels["g_Na"] = 10.0 * np.mean(gene_expression[:3])  # Low baseline

        # Potassium channels
        if "KCNQ1" in gene_map:
            kcnq1_expr = gene_expression[gene_map["KCNQ1"]]
            channels["g_K"] = 36.0 * kcnq1_expr
        else:
            channels["g_K"] = 36.0 * np.mean(gene_expression[3:6])

        # Calcium channels
        if "CACNA1C" in gene_map:
            cacna1c_expr = gene_expression[gene_map["CACNA1C"]]
            channels["g_Ca"] = 1.0 * cacna1c_expr
        else:
            channels["g_Ca"] = 1.0 * np.mean(gene_expression[6:8])

        # Gap junctions
        if "GJA1" in gene_map:
            gja1_expr = gene_expression[gene_map["GJA1"]]
            channels["g_gj"] = organ_spec.gap_junction_strength * gja1_expr
        else:
            channels["g_gj"] = organ_spec.gap_junction_strength

        # Chloride channels
        if "CFTR" in gene_map:
            cftr_expr = gene_expression[gene_map["CFTR"]]
            channels["g_Cl"] = 0.3 * cftr_expr
        else:
            channels["g_Cl"] = 0.3 * np.mean(gene_expression[10:12])

        # Apply organ-specific modulation
        if organ_spec.name == "Heart":
            channels["g_Na"] *= 1.5  # Cardiac action potential needs high Na+
            channels["g_K"] *= 0.8   # Lower K+ for plateau phase
            channels["g_Ca"] *= 3.0  # High Ca2+ for contraction coupling

        elif organ_spec.name == "Brain":
            channels["g_Na"] *= 1.2  # Fast spiking
            channels["g_K"] *= 1.5   # Fast repolarization
            channels["g_Ca"] *= 1.5  # Synaptic transmission

        elif organ_spec.name == "Skeletal Muscle":
            channels["g_Na"] *= 1.3
            channels["g_Ca"] *= 5.0  # Very high for contraction

        elif organ_spec.metabolic:
            channels["g_Na"] *= 0.5  # Metabolic organs less excitable
            channels["g_K"] *= 0.7
            channels["g_Ca"] *= 0.5

        return channels


def fetch_all_organs(mock_mode: bool = False) -> Dict[str, OrganGeneExpression]:
    """
    Fetch AlphaGenome data for all organs in the digital twin.

    Args:
        mock_mode: Use mock predictions (default False - will use real API if available)

    Returns:
        Dict mapping organ_name -> OrganGeneExpression
    """
    mapper = AlphaGenomeOrganMapper(mock_mode=mock_mode)

    organ_data = {}
    for organ_name in HUMAN_ORGANS.keys():
        try:
            data = mapper.fetch_organ_data(organ_name)
            organ_data[organ_name] = data
            logger.info(f"✓ Loaded {organ_name}")
        except Exception as e:
            logger.error(f"✗ Failed to load {organ_name}: {e}")

    return organ_data


def print_organ_expression_summary(organ_data: Dict[str, OrganGeneExpression]):
    """Print summary of organ gene expression."""
    print("=" * 70)
    print("ORGAN GENE EXPRESSION SUMMARY")
    print("=" * 70)

    for organ_name, data in organ_data.items():
        print(f"\n{organ_name.upper()}:")
        print(f"  Expression mean: {data.gene_expression.mean():.3f} ± {data.gene_expression.std():.3f}")
        print(f"  Chromatin mean:  {data.chromatin_state.mean():.3f} ± {data.chromatin_state.std():.3f}")
        print(f"  Ion channels:")
        for ch_name, ch_val in data.ion_channel_expression.items():
            print(f"    {ch_name:6s}: {ch_val:6.2f} mS/cm2")

    print("=" * 70)


if __name__ == "__main__":
    # Test the integration
    logging.basicConfig(level=logging.INFO)

    print("Fetching AlphaGenome data for all organs...")
    organ_data = fetch_all_organs(mock_mode=False)  # Use real API by default

    print_organ_expression_summary(organ_data)
