"""
ABC (Activity-by-Contact) Model Client
=======================================

Client for querying enhancer-gene predictions from the ABC model.

Data source: Nasser et al. Nature 2021
131 cell types and tissues
ABC score = Activity × Contact × other factors

ABC score >= 0.02 is typically considered a functional E-G connection.
"""

import os
import gzip
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ABCPrediction:
    """Enhancer-gene prediction from ABC model."""
    enhancer_chr: str
    enhancer_start: int
    enhancer_end: int
    gene_name: str
    gene_tss: int
    abc_score: float
    activity: float  # H3K27ac + ATAC signal (activity_base)
    contact: float   # Hi-C contact frequency
    cell_type: str
    distance: int    # Enhancer-TSS distance (already provided)


class ABCClient:
    """Client for ABC enhancer-gene predictions."""

    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize ABC client.

        Args:
            data_file: Path to AllPredictions.ABC.txt.gz
                      If None, looks in default location
        """
        if data_file is None:
            data_file = os.path.join(
                os.path.dirname(__file__),
                "../../data/enhancer_promoter/AllPredictions.ABC.txt.gz"
            )

        self.data_file = Path(data_file)
        self.data = None
        self._loaded = False

        logger.info(f"ABC client initialized with file: {self.data_file}")

    def load(self, nrows: Optional[int] = None):
        """
        Load ABC predictions into memory.

        Args:
            nrows: Optional limit on rows to load (for testing)
        """
        if self._loaded:
            logger.info("ABC data already loaded")
            return

        if not self.data_file.exists():
            raise FileNotFoundError(
                f"ABC data file not found: {self.data_file}\n"
                f"Download from: https://mitra.stanford.edu/engreitz/oak/public/Nasser2021/"
            )

        logger.info(f"Loading ABC predictions from {self.data_file}...")

        # Read gzipped file
        with gzip.open(self.data_file, 'rt') as f:
            self.data = pd.read_csv(f, sep='\t', nrows=nrows)

        logger.info(f"Loaded {len(self.data)} ABC predictions")
        logger.info(f"Columns: {list(self.data.columns)}")

        # Print sample
        logger.info(f"Sample data:\n{self.data.head()}")

        self._loaded = True

    def get_gene_enhancers(
        self,
        gene_name: str,
        cell_types: Optional[List[str]] = None,
        min_abc_score: float = 0.02
    ) -> List[ABCPrediction]:
        """
        Get enhancers predicted to regulate a gene.

        Args:
            gene_name: Gene symbol (e.g., "NKX2-5", "SOX2")
            cell_types: Optional list of cell types to filter
            min_abc_score: Minimum ABC score threshold

        Returns:
            List of ABCPrediction objects
        """
        if not self._loaded:
            self.load()

        # Filter by gene
        gene_data = self.data[self.data['TargetGene'] == gene_name]

        # Filter by cell type if specified
        if cell_types:
            gene_data = gene_data[gene_data['CellType'].isin(cell_types)]

        # Filter by ABC score
        gene_data = gene_data[gene_data['ABC.Score'] >= min_abc_score]

        # Convert to ABCPrediction objects
        predictions = []
        for _, row in gene_data.iterrows():
            pred = ABCPrediction(
                enhancer_chr=row['chr'],
                enhancer_start=int(row['start']),
                enhancer_end=int(row['end']),
                gene_name=row['TargetGene'],
                gene_tss=int(row['TargetGeneTSS']),
                abc_score=float(row['ABC.Score']),
                activity=float(row.get('activity_base', 0.0)),
                contact=float(row.get('hic_contact', 0.0)),
                cell_type=row['CellType'],
                distance=int(row['distance'])
            )
            predictions.append(pred)

        return predictions

    def get_enhancer_genes(
        self,
        enhancer_region: Tuple[str, int, int],
        cell_types: Optional[List[str]] = None,
        min_abc_score: float = 0.02,
        window: int = 5000
    ) -> List[ABCPrediction]:
        """
        Get genes predicted to be regulated by an enhancer region.

        Args:
            enhancer_region: (chr, start, end) of enhancer
            cell_types: Optional list of cell types
            min_abc_score: Minimum ABC score
            window: Search window around enhancer (bp)

        Returns:
            List of ABCPrediction objects
        """
        if not self._loaded:
            self.load()

        chr, start, end = enhancer_region

        # Find overlapping enhancers
        enhancer_data = self.data[
            (self.data['chr'] == chr) &
            (self.data['start'] >= start - window) &
            (self.data['end'] <= end + window) &
            (self.data['ABC.Score'] >= min_abc_score)
        ]

        # Filter by cell type if specified
        if cell_types:
            enhancer_data = enhancer_data[enhancer_data['CellType'].isin(cell_types)]

        # Convert to predictions
        predictions = []
        for _, row in enhancer_data.iterrows():
            pred = ABCPrediction(
                enhancer_chr=row['chr'],
                enhancer_start=int(row['start']),
                enhancer_end=int(row['end']),
                gene_name=row['TargetGene'],
                gene_tss=int(row['TargetGeneTSS']),
                abc_score=float(row['ABC.Score']),
                activity=float(row.get('activity_base', 0.0)),
                contact=float(row.get('hic_contact', 0.0)),
                cell_type=row['CellType'],
                distance=int(row['distance'])
            )
            predictions.append(pred)

        return predictions

    def get_available_cell_types(self) -> List[str]:
        """Get list of available cell types in dataset."""
        if not self._loaded:
            self.load()

        return sorted(self.data['CellType'].unique())

    def get_contact_strength(
        self,
        enhancer_region: Tuple[str, int, int],
        gene_name: str,
        cell_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Get contact strength between enhancer and gene.

        This is the KEY function for computing attention weights!

        Args:
            enhancer_region: (chr, start, end) of SE
            gene_name: Target gene name
            cell_types: Optional cell types to query

        Returns:
            Dict mapping cell_type → ABC_score (contact strength)
        """
        if not self._loaded:
            self.load()

        chr, start, end = enhancer_region

        # Find overlapping enhancer-gene pairs
        matches = self.data[
            (self.data['chr'] == chr) &
            (self.data['start'] >= start - 5000) &
            (self.data['end'] <= end + 5000) &
            (self.data['TargetGene'] == gene_name)
        ]

        # Filter by cell type
        if cell_types:
            matches = matches[matches['CellType'].isin(cell_types)]

        # Return ABC scores per cell type
        contact_strengths = {}
        for _, row in matches.iterrows():
            cell_type = row['CellType']
            abc_score = float(row['ABC.Score'])

            # Keep highest score if multiple matches
            if cell_type not in contact_strengths:
                contact_strengths[cell_type] = abc_score
            else:
                contact_strengths[cell_type] = max(
                    contact_strengths[cell_type],
                    abc_score
                )

        return contact_strengths


# ============================================================================
# Utility Functions
# ============================================================================

def find_tissue_cell_types(abc_client: ABCClient, tissue_keyword: str) -> List[str]:
    """
    Find cell types matching a tissue keyword.

    Args:
        abc_client: Loaded ABC client
        tissue_keyword: Keyword like "heart", "brain", "liver"

    Returns:
        List of matching cell type names
    """
    all_cell_types = abc_client.get_available_cell_types()

    matches = [
        ct for ct in all_cell_types
        if tissue_keyword.lower() in ct.lower()
    ]

    return matches


if __name__ == "__main__":
    # Test the ABC client
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("ABC CLIENT TEST")
    print("="*70)

    # Initialize client
    client = ABCClient()

    # Load data (first 100K rows for testing)
    client.load(nrows=100000)

    # Get available cell types
    cell_types = client.get_available_cell_types()
    print(f"\nAvailable cell types: {len(cell_types)}")
    print(f"Sample: {cell_types[:10]}")

    # Find heart-related cell types
    heart_types = find_tissue_cell_types(client, "heart")
    print(f"\nHeart-related cell types: {heart_types}")

    # Test: Get enhancers for NKX2-5 (cardiac gene)
    if heart_types:
        nkx25_enhancers = client.get_gene_enhancers(
            gene_name="NKX2-5",
            cell_types=heart_types,
            min_abc_score=0.02
        )

        print(f"\nNKX2-5 enhancers (ABC >= 0.02): {len(nkx25_enhancers)}")
        for pred in nkx25_enhancers[:5]:
            print(f"  {pred.enhancer_chr}:{pred.enhancer_start}-{pred.enhancer_end}")
            print(f"    ABC score: {pred.abc_score:.3f}")
            print(f"    Cell type: {pred.cell_type}")
            print(f"    Distance: {pred.distance/1000:.1f}kb")

    print("\n" + "="*70)
