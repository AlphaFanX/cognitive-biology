"""
Extended Human Digital Twin Topology
=====================================

Higher-resolution organ models for MakeHuman visualization.
Increased cell counts to match realistic organ complexity.

Cell Count Philosophy:
- Original: 321 cells total (minimal functional model)
- Extended: 3,000-5,000 cells (realistic anatomical resolution)
- Organs scale based on size and functional complexity
- Matches MakeHuman mesh vertex density

Benefits:
- Better spatial resolution for morphogen gradients
- More accurate bioelectric field patterns
- Realistic organ-organ coupling dynamics
- Suitable for detailed regulatory overlays
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class OrganSpec:
    """Extended specification for high-resolution organ."""

    name: str
    cell_count: int  # Increased cell count
    grid_shape: Tuple[int, int, int]  # 3D grid topology
    position_3d: Tuple[float, float, float]  # 3D position (meters, anatomical coords)
    bounding_box: Tuple[Tuple[float, float, float], Tuple[float, float, float]]  # (min, max) corners

    # AlphaGenome tissue identifiers
    uberon_term: str
    genomic_loci: List[Tuple[str, int, int]]

    # Functional properties
    excitable: bool = False
    metabolic: bool = False
    endocrine: bool = False
    contractile: bool = False

    # Bioelectric baseline
    resting_voltage: float = -70.0  # mV
    gap_junction_strength: float = 0.5

    # Visual properties
    color_rgb: Tuple[float, float, float] = (0.7, 0.7, 0.9)  # Default blue-ish


# ============================================================================
# Extended Human Organs (High Resolution)
# ============================================================================

# Anatomical coordinate system:
# X: Left (-) to Right (+)
# Y: Inferior (feet, 0) to Superior (head, 1.75m)
# Z: Posterior (back, -) to Anterior (front, +)

HUMAN_ORGANS_EXTENDED = {
    # ========== BRAIN ==========
    "brain": OrganSpec(
        name="Brain",
        cell_count=500,  # Up from 64
        grid_shape=(10, 10, 5),  # 3D grid
        position_3d=(0.0, 1.65, 0.0),  # Head center
        bounding_box=((-0.08, 1.55, -0.08), (0.08, 1.75, 0.08)),  # 16cm x 20cm x 16cm
        uberon_term="UBERON:0000955",
        genomic_loci=[
            ("chr17", 44000000, 44100000),  # MAPT
            ("chr19", 45000000, 45100000),  # APOE
            ("chr1", 155000000, 155100000),  # NTRK1 (neuronal)
        ],
        excitable=True,
        resting_voltage=-70.0,
        gap_junction_strength=0.3,
        color_rgb=(1.0, 0.4, 0.4),  # Pink-red
    ),

    # ========== HEART ==========
    "heart": OrganSpec(
        name="Heart",
        cell_count=400,  # Up from 64
        grid_shape=(10, 10, 4),
        position_3d=(-0.02, 1.25, 0.08),  # Left-center chest
        bounding_box=((-0.08, 1.15, 0.05), (0.04, 1.35, 0.12)),  # ~12cm x 20cm x 7cm
        uberon_term="UBERON:0000948",
        genomic_loci=[
            ("chr12", 110000000, 110100000),  # SCN5A
            ("chr7", 150000000, 150100000),   # KCNH2
            ("chr5", 172660000, 172670000),   # NKX2-5
        ],
        excitable=True,
        contractile=True,
        resting_voltage=-90.0,
        gap_junction_strength=0.9,
        color_rgb=(1.0, 0.0, 0.0),  # Red
    ),

    # ========== LIVER ==========
    "liver": OrganSpec(
        name="Liver",
        cell_count=300,  # Up from 25
        grid_shape=(10, 10, 3),
        position_3d=(0.08, 1.05, 0.05),  # Right upper abdomen
        bounding_box=((0.02, 0.95, 0.02), (0.15, 1.15, 0.10)),  # ~13cm x 20cm x 8cm
        uberon_term="UBERON:0002107",
        genomic_loci=[
            ("chr2", 169000000, 169100000),  # CYP2D6
            ("chr19", 41000000, 41100000),   # CYP2A6
            ("chr10", 94000000, 94100000),   # CYP2C19
        ],
        metabolic=True,
        resting_voltage=-40.0,
        gap_junction_strength=0.6,
        color_rgb=(0.55, 0.27, 0.07),  # Brown
    ),

    # ========== KIDNEYS ==========
    "kidney_left": OrganSpec(
        name="Kidney (L)",
        cell_count=200,  # Up from 16
        grid_shape=(8, 8, 4),
        position_3d=(-0.10, 1.00, -0.02),  # Left mid-back
        bounding_box=((-0.14, 0.90, -0.05), (-0.06, 1.10, 0.02)),  # ~8cm x 20cm x 7cm
        uberon_term="UBERON:0002113",
        genomic_loci=[
            ("chr12", 121000000, 121100000),  # HNF1A
            ("chr17", 37000000, 37100000),    # SLC6A19
        ],
        metabolic=True,
        resting_voltage=-50.0,
        gap_junction_strength=0.4,
        color_rgb=(0.26, 0.41, 0.88),  # Blue
    ),

    "kidney_right": OrganSpec(
        name="Kidney (R)",
        cell_count=200,
        grid_shape=(8, 8, 4),
        position_3d=(0.10, 1.00, -0.02),  # Right mid-back
        bounding_box=((0.06, 0.90, -0.05), (0.14, 1.10, 0.02)),
        uberon_term="UBERON:0002113",
        genomic_loci=[
            ("chr12", 121000000, 121100000),
            ("chr17", 37000000, 37100000),
        ],
        metabolic=True,
        resting_voltage=-50.0,
        gap_junction_strength=0.4,
        color_rgb=(0.26, 0.41, 0.88),
    ),

    # ========== LUNGS ==========
    "lung_left": OrganSpec(
        name="Lung (L)",
        cell_count=250,  # Up from 25
        grid_shape=(10, 10, 3),
        position_3d=(-0.08, 1.30, 0.00),  # Left upper chest
        bounding_box=((-0.14, 1.15, -0.05), (-0.02, 1.45, 0.08)),  # ~12cm x 30cm x 13cm
        uberon_term="UBERON:0002048",
        genomic_loci=[
            ("chr4", 154000000, 154100000),  # SFTPB
            ("chr11", 102000000, 102100000), # ADIPOQ
        ],
        resting_voltage=-50.0,
        gap_junction_strength=0.3,
        color_rgb=(0.53, 0.81, 0.92),  # Light blue
    ),

    "lung_right": OrganSpec(
        name="Lung (R)",
        cell_count=250,
        grid_shape=(10, 10, 3),
        position_3d=(0.08, 1.30, 0.00),  # Right upper chest
        bounding_box=((0.02, 1.15, -0.05), (0.14, 1.45, 0.08)),
        uberon_term="UBERON:0002048",
        genomic_loci=[
            ("chr4", 154000000, 154100000),
            ("chr11", 102000000, 102100000),
        ],
        resting_voltage=-50.0,
        gap_junction_strength=0.3,
        color_rgb=(0.53, 0.81, 0.92),
    ),

    # ========== PANCREAS ==========
    "pancreas": OrganSpec(
        name="Pancreas",
        cell_count=150,  # Up from 16
        grid_shape=(10, 5, 3),
        position_3d=(-0.05, 0.98, 0.03),  # Upper left abdomen
        bounding_box=((-0.10, 0.93, 0.00), (0.00, 1.03, 0.06)),  # ~10cm x 10cm x 6cm
        uberon_term="UBERON:0001264",
        genomic_loci=[
            ("chr11", 2160000, 2170000),    # INS
            ("chr20", 63000000, 63010000),  # GCK
            ("chr6", 132000000, 132100000), # PDX1
        ],
        endocrine=True,
        excitable=True,
        metabolic=True,
        resting_voltage=-70.0,
        gap_junction_strength=0.7,
        color_rgb=(1.0, 0.65, 0.0),  # Orange
    ),

    # ========== SKELETAL MUSCLE ==========
    "muscle": OrganSpec(
        name="Skeletal Muscle",
        cell_count=200,  # Reduced to make room for bone
        grid_shape=(8, 10, 3),
        position_3d=(0.0, 0.50, 0.05),  # Thigh muscles
        bounding_box=((-0.12, 0.30, 0.02), (0.12, 0.70, 0.10)),  # ~24cm x 40cm x 8cm
        uberon_term="UBERON:0001134",
        genomic_loci=[
            ("chr19", 55000000, 55100000),  # ACTN4
            ("chr11", 64000000, 64100000),  # MYH2
            ("chr17", 10000000, 10100000),  # MYOD1
        ],
        excitable=True,
        contractile=True,
        resting_voltage=-90.0,
        gap_junction_strength=0.2,
        color_rgb=(1.0, 0.08, 0.58),  # Pink
    ),

    # ========== BONE (FEMUR/TIBIA) ==========
    "bone": OrganSpec(
        name="Bone",
        cell_count=150,  # New
        grid_shape=(6, 10, 3),
        position_3d=(0.0, 0.40, 0.0),  # Legs (femur/tibia)
        bounding_box=((-0.08, 0.05, -0.05), (0.08, 0.75, 0.05)),  # Legs from feet to hips
        uberon_term="UBERON:0002481",
        genomic_loci=[
            ("chr6", 45420000, 45470000),   # RUNX2
            ("chr12", 54000000, 54100000),  # SP7 (osterix)
        ],
        metabolic=True,
        resting_voltage=-60.0,
        gap_junction_strength=0.1,
        color_rgb=(0.9, 0.9, 0.8),  # Bone white
    ),

    # ========== GUT ==========
    "gut": OrganSpec(
        name="Gut",
        cell_count=300,  # Up from 25
        grid_shape=(10, 10, 3),
        position_3d=(0.0, 0.85, 0.08),  # Lower abdomen
        bounding_box=((-0.08, 0.75, 0.05), (0.08, 0.95, 0.12)),  # ~16cm x 20cm x 7cm
        uberon_term="UBERON:0001155",
        genomic_loci=[
            ("chr7", 100000000, 100100000),  # CFTR
            ("chr2", 234000000, 234100000),  # ENO2
        ],
        excitable=True,
        metabolic=True,
        resting_voltage=-60.0,
        gap_junction_strength=0.5,
        color_rgb=(0.60, 0.80, 0.20),  # Yellow-green
    ),

    # ========== THYROID ==========
    "thyroid": OrganSpec(
        name="Thyroid",
        cell_count=100,  # Up from 9
        grid_shape=(10, 5, 2),
        position_3d=(0.0, 1.48, 0.03),  # Neck
        bounding_box=((-0.03, 1.46, 0.02), (0.03, 1.50, 0.05)),  # ~6cm x 4cm x 3cm
        uberon_term="UBERON:0002046",
        genomic_loci=[
            ("chr8", 133000000, 133100000),  # TG
            ("chr10", 43000000, 43100000),   # RET
        ],
        endocrine=True,
        resting_voltage=-50.0,
        gap_junction_strength=0.5,
        color_rgb=(0.87, 0.63, 0.87),  # Plum
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_total_cell_count() -> int:
    """Calculate total cells in extended human."""
    return sum(spec.cell_count for spec in HUMAN_ORGANS_EXTENDED.values())


def get_organ_bounding_boxes() -> Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Get bounding boxes of all organs for vertex mapping."""
    return {name: spec.bounding_box for name, spec in HUMAN_ORGANS_EXTENDED.items()}


def point_in_bounding_box(
    point: Tuple[float, float, float],
    bbox: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
) -> bool:
    """Check if a 3D point is inside a bounding box."""
    (x, y, z) = point
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bbox
    return (min_x <= x <= max_x and
            min_y <= y <= max_y and
            min_z <= z <= max_z)


def assign_vertex_to_organ(vertex_pos: Tuple[float, float, float]) -> Optional[str]:
    """
    Assign a mesh vertex to an organ based on its 3D position.

    Args:
        vertex_pos: (x, y, z) position in meters

    Returns:
        Organ name or None if vertex is not in any organ
    """
    for organ_name, spec in HUMAN_ORGANS_EXTENDED.items():
        if point_in_bounding_box(vertex_pos, spec.bounding_box):
            return organ_name
    return None


def distribute_cells_in_organ(organ_name: str) -> np.ndarray:
    """
    Generate cell positions within an organ's bounding box.

    Args:
        organ_name: Name of organ

    Returns:
        Array of shape (n_cells, 3) with cell positions
    """
    spec = HUMAN_ORGANS_EXTENDED[organ_name]
    (min_x, min_y, min_z), (max_x, max_y, max_z) = spec.bounding_box
    n_cells = spec.cell_count

    # Generate positions using grid + jitter for natural distribution
    nx, ny, nz = spec.grid_shape

    cell_positions = []
    cells_per_grid = n_cells // (nx * ny * nz)
    remainder = n_cells % (nx * ny * nz)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                # Grid position
                x = min_x + (ix + 0.5) * (max_x - min_x) / nx
                y = min_y + (iy + 0.5) * (max_y - min_y) / ny
                z = min_z + (iz + 0.5) * (max_z - min_z) / nz

                # Add jitter
                jitter_scale = 0.3  # 30% of grid cell size
                x += np.random.uniform(-jitter_scale * (max_x - min_x) / nx, jitter_scale * (max_x - min_x) / nx)
                y += np.random.uniform(-jitter_scale * (max_y - min_y) / ny, jitter_scale * (max_y - min_y) / ny)
                z += np.random.uniform(-jitter_scale * (max_z - min_z) / nz, jitter_scale * (max_z - min_z) / nz)

                # Clip to bounding box
                x = np.clip(x, min_x, max_x)
                y = np.clip(y, min_y, max_y)
                z = np.clip(z, min_z, max_z)

                cell_positions.append([x, y, z])

                # Add extra cells if needed
                if remainder > 0:
                    cell_positions.append([x, y, z])
                    remainder -= 1

    return np.array(cell_positions[:n_cells])


def print_topology_summary():
    """Print summary of extended human topology."""
    print("=" * 70)
    print("EXTENDED HUMAN DIGITAL TWIN TOPOLOGY")
    print("=" * 70)
    print(f"Total organs: {len(HUMAN_ORGANS_EXTENDED)}")
    print(f"Total cells: {get_total_cell_count()}")
    print()

    print("ORGANS (High Resolution):")
    print("-" * 70)
    for name, spec in HUMAN_ORGANS_EXTENDED.items():
        flags = []
        if spec.excitable: flags.append("EXCIT")
        if spec.metabolic: flags.append("METAB")
        if spec.endocrine: flags.append("ENDO")
        if spec.contractile: flags.append("CONTRACT")

        flags_str = ", ".join(flags) if flags else "---"
        print(f"  {spec.name:20s} {spec.cell_count:4d} cells  {spec.grid_shape}  [{flags_str}]")

    print()
    print("BOUNDING BOXES (Anatomical Coordinates):")
    print("-" * 70)
    for name, spec in HUMAN_ORGANS_EXTENDED.items():
        (min_x, min_y, min_z), (max_x, max_y, max_z) = spec.bounding_box
        print(f"  {spec.name:20s} X:[{min_x:5.2f}, {max_x:5.2f}]  "
              f"Y:[{min_y:5.2f}, {max_y:5.2f}]  Z:[{min_z:5.2f}, {max_z:5.2f}]")

    print("=" * 70)


if __name__ == "__main__":
    print_topology_summary()

    # Test cell distribution
    print("\nGenerating cell distributions for organs...")
    for organ_name in ["brain", "heart", "liver"]:
        cells = distribute_cells_in_organ(organ_name)
        spec = HUMAN_ORGANS_EXTENDED[organ_name]
        print(f"\n{organ_name.upper()}:")
        print(f"  Generated {len(cells)} cells")
        print(f"  Position range: X=[{cells[:, 0].min():.3f}, {cells[:, 0].max():.3f}], "
              f"Y=[{cells[:, 1].min():.3f}, {cells[:, 1].max():.3f}], "
              f"Z=[{cells[:, 2].min():.3f}, {cells[:, 2].max():.3f}]")
