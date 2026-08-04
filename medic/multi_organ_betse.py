"""
Multi-Organ BETSE Adapter for Digital Human Twin
=================================================

Manages bioelectric simulation across all organs with inter-organ coupling.

Architecture:
- Each organ has its own tissue grid with BETSE-style dynamics
- Cross-organ connections via circulatory/neural/hormonal pathways
- Tissue-specific ion channel expression from AlphaGenome
- TRM optimization for homeostasis

Total: 321 cells across 11 organs
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from .human_topology import HUMAN_ORGANS, ORGAN_CONNECTIONS, OrganSpec
    from .alphagen_integration import OrganGeneExpression
except ImportError:
    from human_topology import HUMAN_ORGANS, ORGAN_CONNECTIONS, OrganSpec
    from alphagen_integration import OrganGeneExpression

logger = logging.getLogger(__name__)


@dataclass
class OrganBioelectricState:
    """Bioelectric state for a single organ."""
    organ_name: str
    n_cells: int
    grid_shape: Tuple[int, int]

    # Electrical state
    voltage: np.ndarray  # (n_cells,) membrane potential (mV)
    calcium: np.ndarray  # (n_cells,) intracellular Ca2+ (μM)

    # Ion channel conductances (mS/cm2)
    g_Na: np.ndarray  # (n_cells,) Sodium
    g_K: np.ndarray   # (n_cells,) Potassium
    g_Ca: np.ndarray  # (n_cells,) Calcium
    g_Cl: np.ndarray  # (n_cells,) Chloride
    g_gj: float       # Gap junction coupling strength

    # Spatial topology
    positions_2d: np.ndarray  # (n_cells, 2) grid positions
    adjacency: np.ndarray     # (n_cells, n_cells) connectivity matrix

    # Metabolic state (optional)
    atp: np.ndarray = field(default=None)  # (n_cells,) ATP concentration
    glucose: np.ndarray = field(default=None)  # (n_cells,) Glucose level

    def __post_init__(self):
        """Initialize optional fields if None."""
        if self.atp is None:
            self.atp = np.ones(self.n_cells) * 5.0  # 5 mM ATP baseline
        if self.glucose is None:
            self.glucose = np.ones(self.n_cells) * 5.0  # 5 mM glucose baseline


class MultiOrganBETSE:
    """Multi-organ bioelectric tissue simulator."""

    def __init__(self, dt: float = 0.1):
        """
        Initialize multi-organ simulator.

        Args:
            dt: Time step (milliseconds)
        """
        self.dt = dt
        self.time = 0.0
        self.step_count = 0

        # Storage for organ states
        self.organs: Dict[str, OrganBioelectricState] = {}

        # Inter-organ connection matrix
        self.connection_matrix = self._build_connection_matrix()

        # Physical constants
        self.C_m = 1.0  # Membrane capacitance (μF/cm2)
        self.F = 96485.0  # Faraday constant (C/mol)
        self.R = 8314.0  # Gas constant (J/(mol*K))
        self.T = 310.0  # Temperature (K, ~37°C)

        # Reversal potentials (mV) - Nernst equilibrium
        self.E_Na = 55.0
        self.E_K = -77.0
        self.E_Ca = 120.0
        self.E_Cl = -40.0

        logger.info(f"MultiOrganBETSE initialized with {len(HUMAN_ORGANS)} organs")

    def _build_connection_matrix(self) -> np.ndarray:
        """Build adjacency matrix for organ-organ connections."""
        organ_names = list(HUMAN_ORGANS.keys())
        n = len(organ_names)
        matrix = np.zeros((n, n))

        for conn in ORGAN_CONNECTIONS:
            if conn.source in organ_names and conn.target in organ_names:
                i = organ_names.index(conn.source)
                j = organ_names.index(conn.target)
                matrix[i, j] = conn.strength

        return matrix

    def initialize_organ(
        self,
        organ_name: str,
        gene_expression: OrganGeneExpression
    ) -> OrganBioelectricState:
        """
        Initialize bioelectric state for an organ from gene expression data.

        Args:
            organ_name: Name of organ
            gene_expression: AlphaGenome-derived gene expression

        Returns:
            Initialized organ state
        """
        if organ_name not in HUMAN_ORGANS:
            raise ValueError(f"Unknown organ: {organ_name}")

        spec = HUMAN_ORGANS[organ_name]

        # Create 2D grid positions
        h, w = spec.grid_shape
        positions = np.array([(i, j) for i in range(h) for j in range(w)])

        # Create adjacency matrix (4-neighbor connectivity)
        n_cells = spec.cell_count
        adjacency = np.zeros((n_cells, n_cells))

        for idx, (i, j) in enumerate(positions):
            # Connect to 4-neighbors
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    neighbor_idx = ni * w + nj
                    adjacency[idx, neighbor_idx] = 1.0

        # Initialize voltages at resting potential
        voltage = np.ones(n_cells) * spec.resting_voltage

        # Small random perturbation for interesting dynamics
        voltage += np.random.normal(0, 2.0, n_cells)

        # Initialize calcium
        calcium = np.ones(n_cells) * 0.1  # 0.1 μM resting Ca2+

        # Set ion channel conductances from gene expression
        channels = gene_expression.ion_channel_expression

        g_Na = np.ones(n_cells) * channels.get("g_Na", 10.0)
        g_K = np.ones(n_cells) * channels.get("g_K", 36.0)
        g_Ca = np.ones(n_cells) * channels.get("g_Ca", 1.0)
        g_Cl = np.ones(n_cells) * channels.get("g_Cl", 0.3)
        g_gj = channels.get("g_gj", 0.5)

        # Add spatial gradients for interesting patterns
        for i in range(n_cells):
            row, col = positions[i]
            gradient_factor = (row / h) * 0.2 + 1.0  # 20% variation
            g_Na[i] *= gradient_factor
            g_K[i] *= gradient_factor

        state = OrganBioelectricState(
            organ_name=organ_name,
            n_cells=n_cells,
            grid_shape=spec.grid_shape,
            voltage=voltage,
            calcium=calcium,
            g_Na=g_Na,
            g_K=g_K,
            g_Ca=g_Ca,
            g_Cl=g_Cl,
            g_gj=g_gj,
            positions_2d=positions,
            adjacency=adjacency
        )

        self.organs[organ_name] = state

        logger.info(f"✓ Initialized {organ_name}: {n_cells} cells at {spec.resting_voltage} mV")

        return state

    def step_organ(self, organ_name: str):
        """
        Advance bioelectric dynamics for one organ by dt.

        Uses simplified Hodgkin-Huxley-style dynamics:
        C_m * dV/dt = -Σ[g_ion * (V - E_rev)] + I_gap + I_external
        """
        if organ_name not in self.organs:
            raise ValueError(f"Organ {organ_name} not initialized")

        state = self.organs[organ_name]
        n = state.n_cells

        # Copy current voltage for gap junction calculation
        V = state.voltage.copy()

        # Calculate ionic currents for each cell
        I_Na = state.g_Na * (V - self.E_Na)
        I_K = state.g_K * (V - self.E_K)
        I_Ca = state.g_Ca * (V - self.E_Ca)
        I_Cl = state.g_Cl * (V - self.E_Cl)

        # Total ionic current
        I_ion = I_Na + I_K + I_Ca + I_Cl

        # Gap junction coupling
        I_gap = np.zeros(n)
        for i in range(n):
            neighbors = np.where(state.adjacency[i] > 0)[0]
            if len(neighbors) > 0:
                # Current flows from neighbors to cell i
                V_neighbors = V[neighbors]
                I_gap[i] = state.g_gj * np.sum(V_neighbors - V[i])

        # Voltage update (Forward Euler)
        dV_dt = (-I_ion + I_gap) / self.C_m
        state.voltage += self.dt * dV_dt

        # Calcium dynamics (simplified)
        # Ca2+ increases with Ca2+ current, decays to resting level
        Ca_rest = 0.1  # μM
        tau_Ca = 100.0  # ms
        alpha_Ca = 0.001  # Conversion factor from current to concentration

        dCa_dt = -alpha_Ca * I_Ca - (state.calcium - Ca_rest) / tau_Ca
        state.calcium += self.dt * dCa_dt

        # Clip to physiological ranges
        state.voltage = np.clip(state.voltage, -150.0, 50.0)
        state.calcium = np.clip(state.calcium, 0.05, 10.0)

    def apply_inter_organ_coupling(self):
        """
        Apply coupling between organs via circulatory/neural/hormonal pathways.

        This implements the connection matrix where organs influence each other's
        electrical state based on connection type and strength.
        """
        organ_names = list(self.organs.keys())

        for i, source_name in enumerate(organ_names):
            if source_name not in self.organs:
                continue

            source = self.organs[source_name]

            for j, target_name in enumerate(organ_names):
                if target_name not in self.organs or i == j:
                    continue

                coupling_strength = self.connection_matrix[i, j]

                if coupling_strength > 0:
                    target = self.organs[target_name]

                    # Calculate average voltage of source organ
                    V_source_avg = np.mean(source.voltage)

                    # Apply weak voltage influence (long-range coupling)
                    # This represents systemic effects (hormones, neural, etc.)
                    influence_factor = 0.01 * coupling_strength  # Weak coupling

                    # Add influence to target voltage
                    target.voltage += influence_factor * (V_source_avg - np.mean(target.voltage))

    def step_all(self):
        """Advance all organs by one time step with inter-organ coupling."""
        # Step each organ independently
        for organ_name in self.organs.keys():
            self.step_organ(organ_name)

        # Apply inter-organ coupling
        self.apply_inter_organ_coupling()

        # Update time
        self.time += self.dt
        self.step_count += 1

    def get_summary_stats(self) -> Dict:
        """Get summary statistics across all organs."""
        stats = {
            "time": self.time,
            "step": self.step_count,
            "total_cells": sum(org.n_cells for org in self.organs.values()),
            "organs": {}
        }

        for organ_name, state in self.organs.items():
            stats["organs"][organ_name] = {
                "voltage_mean": float(np.mean(state.voltage)),
                "voltage_std": float(np.std(state.voltage)),
                "voltage_min": float(np.min(state.voltage)),
                "voltage_max": float(np.max(state.voltage)),
                "calcium_mean": float(np.mean(state.calcium)),
                "n_cells": state.n_cells
            }

        return stats

    def print_status(self):
        """Print current system status."""
        print(f"=" * 70)
        print(f"Time: {self.time:.1f} ms | Step: {self.step_count}")
        print(f"-" * 70)

        for organ_name, state in self.organs.items():
            V_mean = np.mean(state.voltage)
            V_std = np.std(state.voltage)
            Ca_mean = np.mean(state.calcium)

            print(f"{organ_name:15s}  Vm: {V_mean:7.2f} +/- {V_std:5.2f} mV  |  Ca: {Ca_mean:5.3f} uM")

        print(f"=" * 70)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_digital_human(organ_data: Dict[str, OrganGeneExpression], dt: float = 0.1) -> MultiOrganBETSE:
    """
    Create a complete digital human from AlphaGenome data.

    Args:
        organ_data: Dict from fetch_all_organs()
        dt: Time step (ms)

    Returns:
        Initialized MultiOrganBETSE simulator
    """
    sim = MultiOrganBETSE(dt=dt)

    for organ_name, gene_expr in organ_data.items():
        sim.initialize_organ(organ_name, gene_expr)

    logger.info(f"Digital human created: {sim.get_summary_stats()['total_cells']} cells")

    return sim


if __name__ == "__main__":
    # Test multi-organ BETSE
    logging.basicConfig(level=logging.INFO)

    from alphagen_integration import fetch_all_organs

    print("Creating digital human from AlphaGenome data...")
    organ_data = fetch_all_organs(mock_mode=True)

    print("\nInitializing multi-organ bioelectric simulator...")
    human = create_digital_human(organ_data, dt=0.1)

    print("\nInitial state:")
    human.print_status()

    print("\nRunning simulation for 100 ms...")
    for _ in range(1000):  # 100 ms / 0.1 ms per step
        human.step_all()

        if human.step_count % 200 == 0:
            print(f"\nStep {human.step_count}:")
            human.print_status()

    print("\nSimulation complete!")
