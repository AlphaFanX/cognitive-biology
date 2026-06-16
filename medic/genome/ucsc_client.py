"""
UCSC Genome Browser REST API Client
====================================

Client for querying enhancer and regulatory element tracks from UCSC.

Available tracks:
- encodeCcreCombined: ENCODE Candidate Cis-Regulatory Elements
- geneHancer: GeneHancer Regulatory Elements
- encRegTfbsClustered: TF ChIP-seq Clusters (340 factors, 129 cell types)
- ReMap: ReMap Atlas of Regulatory Regions

API docs: https://genome.ucsc.edu/goldenPath/help/api.html
"""

import requests
import logging
import urllib3
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Disable SSL warnings for UCSC API
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


@dataclass
class UCSCEnhancer:
    """Enhancer element from UCSC Genome Browser."""
    chrom: str
    start: int
    end: int
    name: str
    score: float
    enhancer_type: str  # e.g., "dELS" (distal enhancer-like signature)
    z_score: float
    description: str
    track: str  # Source track name


class UCSCClient:
    """Client for UCSC Genome Browser REST API."""

    def __init__(self, genome: str = "hg38"):
        """
        Initialize UCSC API client.

        Args:
            genome: Genome build (default: hg38)
        """
        self.base_url = "https://api.genome.ucsc.edu"
        self.genome = genome
        logger.info(f"UCSC client initialized for {genome}")

    def list_enhancer_tracks(self) -> Dict[str, str]:
        """
        Get available enhancer/regulatory tracks.

        Returns:
            Dict mapping track_name → description
        """
        # Query all tracks
        response = requests.get(
            f"{self.base_url}/list/tracks?genome={self.genome}",
            verify=False,
            timeout=30
        )
        response.raise_for_status()

        all_tracks = response.json().get(self.genome, {})

        # Filter for enhancer/regulatory tracks
        keywords = ['enhancer', 'regulatory', 'ccre', 'genehancer',
                    'fantom', 'vista', 'chip', 'h3k27ac', 'h3k4me1']

        enhancer_tracks = {}
        for track_name, track_info in all_tracks.items():
            if isinstance(track_info, dict):
                track_lower = track_name.lower()
                label_lower = track_info.get('longLabel', '').lower()

                if any(kw in track_lower or kw in label_lower for kw in keywords):
                    enhancer_tracks[track_name] = track_info.get('longLabel', '')

        logger.info(f"Found {len(enhancer_tracks)} enhancer/regulatory tracks")
        return enhancer_tracks

    def query_region(
        self,
        track: str,
        chrom: str,
        start: int,
        end: int
    ) -> List[UCSCEnhancer]:
        """
        Query enhancers in a genomic region.

        Args:
            track: Track name (e.g., "encodeCcreCombined")
            chrom: Chromosome (e.g., "chr5")
            start: Start position (0-based)
            end: End position

        Returns:
            List of UCSCEnhancer objects
        """
        query_url = (
            f"{self.base_url}/getData/track?"
            f"genome={self.genome};"
            f"track={track};"
            f"chrom={chrom};"
            f"start={start};"
            f"end={end}"
        )

        try:
            response = requests.get(query_url, timeout=30, verify=False)
            response.raise_for_status()
            data = response.json()

            # Extract enhancers from response
            enhancers = []
            if track in data:
                track_data = data[track]
                if isinstance(track_data, list):
                    for record in track_data:
                        enhancer = self._parse_enhancer(record, track)
                        if enhancer:
                            enhancers.append(enhancer)

            logger.info(f"Found {len(enhancers)} enhancers in {chrom}:{start}-{end}")
            return enhancers

        except Exception as e:
            logger.error(f"Failed to query {track}: {e}")
            return []

    def _parse_enhancer(self, record: dict, track: str) -> Optional[UCSCEnhancer]:
        """Parse UCSC track record into UCSCEnhancer object."""
        try:
            # Handle encodeCcreCombined format
            if track == "encodeCcreCombined":
                return UCSCEnhancer(
                    chrom=record.get('chrom', ''),
                    start=int(record.get('chromStart', 0)),
                    end=int(record.get('chromEnd', 0)),
                    name=record.get('name', ''),
                    score=float(record.get('score', 0.0)),
                    enhancer_type=record.get('ccre', ''),
                    z_score=float(record.get('zScore', 0.0)),
                    description=record.get('description', ''),
                    track=track
                )

            # Handle geneHancer format (if available)
            elif track == "geneHancer":
                return UCSCEnhancer(
                    chrom=record.get('chrom', ''),
                    start=int(record.get('chromStart', 0)),
                    end=int(record.get('chromEnd', 0)),
                    name=record.get('name', ''),
                    score=float(record.get('score', 0.0)),
                    enhancer_type="genehancer",
                    z_score=0.0,
                    description=record.get('geneSymbol', ''),
                    track=track
                )

            # Generic format
            else:
                return UCSCEnhancer(
                    chrom=record.get('chrom', ''),
                    start=int(record.get('chromStart', 0)),
                    end=int(record.get('chromEnd', 0)),
                    name=record.get('name', ''),
                    score=float(record.get('score', 0.0)),
                    enhancer_type="unknown",
                    z_score=0.0,
                    description=str(record),
                    track=track
                )

        except Exception as e:
            logger.error(f"Failed to parse record: {e}")
            return None

    def query_gene_enhancers(
        self,
        gene_name: str,
        gene_region: Tuple[str, int, int],
        track: str = "encodeCcreCombined",
        window: int = 1000000
    ) -> List[UCSCEnhancer]:
        """
        Query enhancers near a gene.

        Args:
            gene_name: Gene symbol (for logging)
            gene_region: (chr, tss_start, tss_end) of gene
            track: UCSC track to query
            window: Search window around gene (bp)

        Returns:
            List of UCSCEnhancer objects
        """
        chrom, tss_start, tss_end = gene_region

        # Expand search window
        search_start = max(0, tss_start - window)
        search_end = tss_end + window

        logger.info(
            f"Querying {track} for {gene_name} enhancers "
            f"in {chrom}:{search_start}-{search_end}"
        )

        return self.query_region(track, chrom, search_start, search_end)


# ============================================================================
# Utility Functions
# ============================================================================

def get_encode_ccres(
    ucsc_client: UCSCClient,
    region: Tuple[str, int, int]
) -> List[UCSCEnhancer]:
    """
    Get ENCODE cCREs (Candidate Cis-Regulatory Elements) in a region.

    cCRE types:
    - dELS: distal enhancer-like signature
    - pELS: proximal enhancer-like signature
    - PLS: promoter-like signature
    - DNase-H3K4me3: open chromatin + promoter mark
    - CTCF-only: insulator elements

    Args:
        ucsc_client: Initialized UCSC client
        region: (chr, start, end)

    Returns:
        List of enhancer elements
    """
    chrom, start, end = region
    return ucsc_client.query_region("encodeCcreCombined", chrom, start, end)


if __name__ == "__main__":
    # Test the UCSC client
    logging.basicConfig(level=logging.INFO)

    print("="*70)
    print("UCSC CLIENT TEST")
    print("="*70)

    # Initialize client
    client = UCSCClient(genome="hg38")

    # List available enhancer tracks
    tracks = client.list_enhancer_tracks()
    print(f"\nAvailable enhancer tracks: {len(tracks)}")
    for track_name in list(tracks.keys())[:10]:
        print(f"  - {track_name}")

    # Test: Query NKX2-5 region for ENCODE cCREs
    print("\nQuerying ENCODE cCREs for NKX2-5 region...")
    nkx25_enhancers = client.query_region(
        track="encodeCcreCombined",
        chrom="chr5",
        start=172232000,
        end=172250000
    )

    print(f"Found {len(nkx25_enhancers)} cCREs:")
    for enh in nkx25_enhancers[:5]:
        print(f"\n  {enh.name}")
        print(f"    Location: {enh.chrom}:{enh.start}-{enh.end}")
        print(f"    Type: {enh.enhancer_type}")
        print(f"    Z-score: {enh.z_score:.2f}")

    print("\n" + "="*70)
