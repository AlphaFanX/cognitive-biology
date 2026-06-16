"""
Minimal Human Digital Twin Topology
====================================

Defines the organ structure for a minimal viable digital human.
Each organ has just enough cells to capture essential mech/elec/chem behavior.

Design Philosophy:
- Functional representation, not anatomical accuracy
- Minimal cell counts while preserving key dynamics
- Cross-organ communication via bioelectric coupling
- Human-specific gene expression from AlphaGenome

Cell Count Strategy:
- Excitable tissues (heart, brain, muscle): ~16-64 cells (capture oscillations)
- Metabolic organs (liver, kidney): ~16-25 cells (capture gradients)
- Endocrine (pancreas, thyroid): ~9-16 cells (capture hormone pulses)
- Circulatory: network topology (not individual cells)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class OrganSpec:
    """Specification for a minimal organ representation."""

    name: str
    cell_count: int  # Minimal viable cell count
    grid_shape: Tuple[int, int]  # 2D grid topology
    position_3d: Tuple[float, float, float]  # 3D position in body (normalized 0-1)

    # AlphaGenome tissue identifiers
    uberon_term: str  # UBERON ontology term
    genomic_loci: List[Tuple[str, int, int]]  # Key genomic regions for this tissue

    # Functional properties
    excitable: bool = False  # Can generate action potentials
    metabolic: bool = False  # Metabolic activity important
    endocrine: bool = False  # Hormone secretion

    # Bioelectric baseline
    resting_voltage: float = -70.0  # mV
    gap_junction_strength: float = 0.5  # Coupling strength

    # Mechanical properties (for future integration)
    contractile: bool = False
    elastic_modulus: float = 1.0  # kPa (placeholder)


# ============================================================================
# Human Organ Definitions (Minimal Digital Twin)
# ============================================================================

HUMAN_ORGANS = {
    # ========== BRAIN (Central Nervous System) ==========
    "brain": OrganSpec(
        name="Brain",
        cell_count=64,  # 8x8 grid - represents cortical column
        grid_shape=(8, 8),
        position_3d=(0.5, 0.9, 0.5),  # Top center
        uberon_term="UBERON:0000955",  # brain
        genomic_loci=[
            ("chr17", 44000000, 44100000),  # MAPT (tau protein, neuronal)
            ("chr19", 45000000, 45100000),  # APOE (synaptic function)
        ],
        excitable=True,
        resting_voltage=-70.0,
        gap_junction_strength=0.3,  # Lower coupling (specific synapses)
    ),

    # ========== HEART (Cardiovascular) ==========
    "heart": OrganSpec(
        name="Heart",
        cell_count=64,  # 8x8 grid - represents cardiac syncytium
        grid_shape=(8, 8),
        position_3d=(0.5, 0.65, 0.45),  # Upper chest, slightly left
        uberon_term="UBERON:0000948",  # heart
        genomic_loci=[
            ("chr12", 110000000, 110100000),  # SCN5A (cardiac Na+ channel)
            ("chr7", 150000000, 150100000),   # KCNH2 (cardiac K+ channel)
        ],
        excitable=True,
        contractile=True,
        resting_voltage=-90.0,  # Cardiac resting potential
        gap_junction_strength=0.9,  # High coupling (syncytium)
    ),

    # ========== LIVER (Metabolic) ==========
    "liver": OrganSpec(
        name="Liver",
        cell_count=25,  # 5x5 grid - hepatic lobule abstraction
        grid_shape=(5, 5),
        position_3d=(0.55, 0.45, 0.5),  # Upper right abdomen
        uberon_term="UBERON:0002107",  # liver
        genomic_loci=[
            ("chr2", 169000000, 169100000),   # CYP2D6 (drug metabolism)
            ("chr19", 41000000, 41100000),    # CYP2A6 (xenobiotic metabolism)
        ],
        metabolic=True,
        resting_voltage=-40.0,  # Hepatocytes are less polarized
        gap_junction_strength=0.6,
    ),

    # ========== KIDNEY (Filtration/Metabolic) ==========
    "kidney_left": OrganSpec(
        name="Kidney (L)",
        cell_count=16,  # 4x4 grid - nephron unit
        grid_shape=(4, 4),
        position_3d=(0.3, 0.4, 0.5),  # Left mid-back
        uberon_term="UBERON:0002113",  # kidney
        genomic_loci=[
            ("chr12", 121000000, 121100000),  # HNF1A (kidney development)
            ("chr17", 37000000, 37100000),    # SLC6A19 (amino acid transport)
        ],
        metabolic=True,
        resting_voltage=-50.0,
        gap_junction_strength=0.4,
    ),

    "kidney_right": OrganSpec(
        name="Kidney (R)",
        cell_count=16,
        grid_shape=(4, 4),
        position_3d=(0.7, 0.4, 0.5),  # Right mid-back
        uberon_term="UBERON:0002113",
        genomic_loci=[
            ("chr12", 121000000, 121100000),
            ("chr17", 37000000, 37100000),
        ],
        metabolic=True,
        resting_voltage=-50.0,
        gap_junction_strength=0.4,
    ),

    # ========== PANCREAS (Endocrine/Metabolic) ==========
    "pancreas": OrganSpec(
        name="Pancreas",
        cell_count=16,  # 4x4 grid - islet of Langerhans
        grid_shape=(4, 4),
        position_3d=(0.45, 0.45, 0.55),  # Upper left abdomen
        uberon_term="UBERON:0001264",  # pancreas
        genomic_loci=[
            ("chr11", 2160000, 2170000),    # INS (insulin gene)
            ("chr20", 63000000, 63010000),  # GCK (glucose sensing)
        ],
        endocrine=True,
        excitable=True,  # Beta cells have electrical activity
        metabolic=True,
        resting_voltage=-70.0,
        gap_junction_strength=0.7,  # Beta cells are coupled
    ),

    # ========== LUNGS (Respiratory) ==========
    "lung_left": OrganSpec(
        name="Lung (L)",
        cell_count=25,  # 5x5 grid - alveolar sheet
        grid_shape=(5, 5),
        position_3d=(0.4, 0.6, 0.45),  # Upper left chest
        uberon_term="UBERON:0002048",  # lung
        genomic_loci=[
            ("chr4", 154000000, 154100000),  # SFTPB (surfactant)
            ("chr11", 102000000, 102100000), # ADIPOQ (epithelial)
        ],
        resting_voltage=-50.0,
        gap_junction_strength=0.3,
    ),

    "lung_right": OrganSpec(
        name="Lung (R)",
        cell_count=25,
        grid_shape=(5, 5),
        position_3d=(0.6, 0.6, 0.45),  # Upper right chest
        uberon_term="UBERON:0002048",
        genomic_loci=[
            ("chr4", 154000000, 154100000),
            ("chr11", 102000000, 102100000),
        ],
        resting_voltage=-50.0,
        gap_junction_strength=0.3,
    ),

    # ========== SKELETAL MUSCLE (Locomotion) ==========
    "muscle": OrganSpec(
        name="Skeletal Muscle",
        cell_count=36,  # 6x6 grid - muscle fiber bundle
        grid_shape=(6, 6),
        position_3d=(0.5, 0.3, 0.6),  # Lower body (representative)
        uberon_term="UBERON:0001134",  # skeletal muscle
        genomic_loci=[
            ("chr19", 55000000, 55100000),  # ACTN4 (muscle contraction)
            ("chr11", 64000000, 64100000),  # MYH2 (myosin)
        ],
        excitable=True,
        contractile=True,
        resting_voltage=-90.0,
        gap_junction_strength=0.2,  # Low (individual fibers)
    ),

    # ========== GUT (Digestive/Enteric) ==========
    "gut": OrganSpec(
        name="Gut",
        cell_count=25,  # 5x5 grid - intestinal segment
        grid_shape=(5, 5),
        position_3d=(0.5, 0.35, 0.6),  # Lower abdomen
        uberon_term="UBERON:0001155",  # colon (more specific tissue type)
        genomic_loci=[
            ("chr7", 100000000, 100100000),  # CFTR (ion transport)
            ("chr2", 234000000, 234100000),  # ENO2 (enteric neurons)
        ],
        excitable=True,  # Enteric nervous system
        metabolic=True,
        resting_voltage=-60.0,
        gap_junction_strength=0.5,
    ),

    # ========== THYROID (Endocrine) ==========
    "thyroid": OrganSpec(
        name="Thyroid",
        cell_count=9,  # 3x3 grid - follicle cluster
        grid_shape=(3, 3),
        position_3d=(0.5, 0.75, 0.4),  # Neck
        uberon_term="UBERON:0002046",  # thyroid
        genomic_loci=[
            ("chr8", 133000000, 133100000),  # TG (thyroglobulin)
            ("chr10", 43000000, 43100000),   # RET (thyroid development)
        ],
        endocrine=True,
        resting_voltage=-50.0,
        gap_junction_strength=0.5,
    ),
}


# ============================================================================
# Inter-Organ Connections (Network Topology)
# ============================================================================

@dataclass
class OrganConnection:
    """Connection between two organs."""
    source: str
    target: str
    connection_type: str  # "circulatory", "neural", "hormonal"
    strength: float  # 0-1


ORGAN_CONNECTIONS = [
    # Brain connections
    OrganConnection("brain", "heart", "neural", 0.8),  # Autonomic control
    OrganConnection("brain", "gut", "neural", 0.6),    # Vagal innervation
    OrganConnection("brain", "muscle", "neural", 0.9), # Motor control

    # Heart connections (circulatory)
    OrganConnection("heart", "brain", "circulatory", 1.0),
    OrganConnection("heart", "liver", "circulatory", 1.0),
    OrganConnection("heart", "kidney_left", "circulatory", 1.0),
    OrganConnection("heart", "kidney_right", "circulatory", 1.0),
    OrganConnection("heart", "lung_left", "circulatory", 1.0),
    OrganConnection("heart", "lung_right", "circulatory", 1.0),

    # Endocrine connections (hormonal)
    OrganConnection("pancreas", "liver", "hormonal", 0.9),  # Insulin → glucose
    OrganConnection("pancreas", "muscle", "hormonal", 0.8), # Insulin → uptake
    OrganConnection("thyroid", "brain", "hormonal", 0.7),   # T3/T4 → metabolism
    OrganConnection("thyroid", "heart", "hormonal", 0.8),   # T3/T4 → rate

    # Metabolic connections
    OrganConnection("liver", "kidney_left", "circulatory", 0.7),  # Waste products
    OrganConnection("liver", "kidney_right", "circulatory", 0.7),
    OrganConnection("gut", "liver", "circulatory", 0.9),  # Portal circulation
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_total_cell_count() -> int:
    """Calculate total cells in minimal human."""
    return sum(spec.cell_count for spec in HUMAN_ORGANS.values())


def get_organ_positions_3d() -> Dict[str, Tuple[float, float, float]]:
    """Get 3D positions of all organs for visualization."""
    return {name: spec.position_3d for name, spec in HUMAN_ORGANS.items()}


def get_excitable_organs() -> List[str]:
    """Get list of excitable organs (can generate action potentials)."""
    return [name for name, spec in HUMAN_ORGANS.items() if spec.excitable]


def get_metabolic_organs() -> List[str]:
    """Get list of metabolically active organs."""
    return [name for name, spec in HUMAN_ORGANS.items() if spec.metabolic]


def get_endocrine_organs() -> List[str]:
    """Get list of endocrine organs."""
    return [name for name, spec in HUMAN_ORGANS.items() if spec.endocrine]


def get_connection_matrix() -> np.ndarray:
    """Build adjacency matrix for organ connections."""
    organ_names = list(HUMAN_ORGANS.keys())
    n = len(organ_names)
    matrix = np.zeros((n, n))

    for conn in ORGAN_CONNECTIONS:
        if conn.source in organ_names and conn.target in organ_names:
            i = organ_names.index(conn.source)
            j = organ_names.index(conn.target)
            matrix[i, j] = conn.strength

    return matrix


def print_topology_summary():
    """Print a summary of the minimal human topology."""
    print("=" * 70)
    print("MINIMAL HUMAN DIGITAL TWIN TOPOLOGY")
    print("=" * 70)
    print(f"Total organs: {len(HUMAN_ORGANS)}")
    print(f"Total cells: {get_total_cell_count()}")
    print(f"Total connections: {len(ORGAN_CONNECTIONS)}")
    print()

    print("ORGANS:")
    print("-" * 70)
    for name, spec in HUMAN_ORGANS.items():
        flags = []
        if spec.excitable: flags.append("EXCIT")
        if spec.metabolic: flags.append("METAB")
        if spec.endocrine: flags.append("ENDO")
        if spec.contractile: flags.append("CONTRACT")

        flags_str = ", ".join(flags) if flags else "---"
        print(f"  {spec.name:20s} {spec.cell_count:3d} cells  {spec.grid_shape}  [{flags_str}]")

    print()
    print("CONNECTIONS:")
    print("-" * 70)
    for conn in ORGAN_CONNECTIONS:
        print(f"  {conn.source:15s} -> {conn.target:15s}  [{conn.connection_type:12s}] {conn.strength:.1f}")

    print("=" * 70)


if __name__ == "__main__":
    print_topology_summary()
