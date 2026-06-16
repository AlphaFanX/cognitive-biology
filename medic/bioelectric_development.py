"""
Bioelectric Development Layer for Cognimed
============================================

Connects BETSE bioelectric physics to the developmental simulator so that
voltage patterns (Vmem gradients) guide morphogenesis -- the missing spatial
pattern controller above genetics.

Key insight (Levin): Vmem gradients define organ territories BEFORE genetics
commits cell fates. Without this layer, SE-guided differentiation produces
teratomas (right cell types, wrong spatial organization).

Bioelectric tissue patterning data hardcoded from Levin's published
Xenopus/planaria Vmem maps.

Classes:
    EmbryoBioelectricState   -- voltage/calcium/conductances for growing embryo
    DevelopmentalBioelectricField -- HH dynamics for embryonic tissue
    VmemGradientComputer     -- spatial gradient from irregular cell mesh
    ElectrotaxisController   -- cathodal cell migration
    BioelectricFateModulator -- voltage modulates SE differentiation probs
    VoltagePatternMemory     -- organ boundary gap junction modulation
    ApoptosisController      -- depolarization-triggered cell removal
    BioelectricDevelopmentOrchestrator -- wires all components, single step()
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    from .human_topology import HUMAN_ORGANS, OrganSpec
except ImportError:
    from human_topology import HUMAN_ORGANS, OrganSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Constants (from Levin's Xenopus data)
# =============================================================================

# Resting potentials by germ layer (mV)
LEVIN_ECTODERM_VMEM = -70.0
LEVIN_MESODERM_VMEM = -30.0
LEVIN_ENDODERM_VMEM = -20.0

# Gap junction threshold distance (normalized coordinates)
GAP_JUNCTION_THRESHOLD = 0.15

# Bioelectric modulation strength (30% bioelectric, 70% genomic)
BIOELECTRIC_MODULATION_STRENGTH = 0.3

# Apoptosis parameters
APOPTOSIS_MISMATCH_THRESHOLD = 30.0  # mV
DEPOLARIZATION_THRESHOLD = -20.0  # mV

# Substeps per developmental step (numerical stability)
BIOELECTRIC_SUBSTEPS = 10

# Reversal potentials (mV) -- Nernst equilibrium
E_NA = 55.0
E_K = -77.0
E_CA = 120.0
E_CL = -40.0

# Default conductances (mS/cm2) -- embryonic non-excitable tissue
# Much lower than neural HH (120/36) to avoid voltage rail saturation
DEFAULT_G_NA = 0.5
DEFAULT_G_K = 1.5
DEFAULT_G_CA = 0.1
DEFAULT_G_CL = 0.1
DEFAULT_G_GJ = 0.05

# Membrane capacitance
C_M = 1.0  # uF/cm2

# Calcium dynamics
CA_REST = 0.1  # uM
CA_TAU = 100.0  # ms
CA_ALPHA = 0.001  # current-to-concentration conversion

# Organ-to-germ-layer mapping (for fate preference voltages)
ORGAN_GERM_LAYER = {
    "brain": "ectoderm",
    "heart": "mesoderm",
    "liver": "endoderm",
    "kidney_left": "mesoderm",
    "kidney_right": "mesoderm",
    "pancreas": "endoderm",
    "lung_left": "endoderm",
    "lung_right": "endoderm",
    "muscle": "mesoderm",
    "gut": "endoderm",
    "thyroid": "endoderm",
}

# Preferred voltage per organ (mV) -- from germ layer + organ-specific tuning
ORGAN_PREFERRED_VOLTAGE = {
    "brain": -70.0,       # Ectoderm: strongly hyperpolarized
    "heart": -30.0,       # Mesoderm: moderately depolarized
    "liver": -20.0,       # Endoderm: depolarized
    "kidney_left": -35.0, # Mesoderm variant
    "kidney_right": -35.0,
    "pancreas": -25.0,    # Endoderm, excitable beta cells
    "lung_left": -25.0,   # Endoderm
    "lung_right": -25.0,
    "muscle": -40.0,      # Mesoderm, excitable
    "gut": -25.0,         # Endoderm
    "thyroid": -30.0,     # Endoderm
}

# Gaussian sigma for voltage preference (mV)
VOLTAGE_PREFERENCE_SIGMA = 20.0

# Temporal phasing thresholds (cell count boundaries)
PHASE_1_END = 32    # Voltage pattern establishes (zygote -> blastula)
PHASE_2_END = 128   # Signals accumulate (gastrulation -> neurulation)
# Phase 3: 128+ cells -- fate reads from cached signaling landscape


# =============================================================================
# SE-Derived Ion Channel Conductance Profiles (ABC Database, Nasser 2021)
# =============================================================================
#
# Queried AllPredictions.ABC.txt.gz for 45 ion channel genes across 131 cell
# types matched to our 11 organs. Total activity_base scores per channel type
# normalized to conductance ratios.
#
# These replace the simple linear A-P gradient with organ-specific voltage
# territories. The conductance profile IS the cymatic generator -- the SE
# landscape for ion channel genes that creates the standing wave pattern.
#
# Source: ABC database, Nasser et al. Nature 2021
# Ion channel families: SCN (Na+), KCNJ/KCNH/KCNQ/KCNK/KCNN (K+),
#   CACNA/TRPC/TRPV (Ca2+), CLCN/CFTR (Cl-), GJA/GJB/GJC (gap junctions)
#

# Raw total activity scores from ABC database (sum of activity_base across
# all enhancer-gene predictions per channel family per organ cell type)
_ABC_ION_CHANNEL_ACTIVITY = {
    # organ:      (Na_total,  K_total,   Ca_total,  Cl_total,  GJ_total)
    "brain":      (207.27,    470.72,    305.45,    111.69,    75.26),
    "heart":      (125.64,    521.73,    154.02,    144.09,    123.83),
    "liver":      (116.54,    458.29,    180.40,    154.09,    110.45),
    "kidney":     (133.95,    305.77,     76.75,    101.89,     59.43),
    "pancreas":   (297.90,    556.86,     86.37,    167.81,     36.34),
    "lung":       ( 51.43,    134.77,     53.95,     37.54,     35.69),
    "muscle":     (359.06,    745.18,    180.72,    233.97,    145.81),
    "gut":        (372.88,    710.31,    228.28,    240.45,    151.28),
    "thyroid":    (157.02,    237.15,     53.64,     77.18,     71.51),
}

# Normalize activity to conductance scale (mS/cm2)
# Strategy: use ABC ratios for RELATIVE channel balance within each organ,
# then anchor absolute conductances to achieve the known developmental
# resting voltages from Levin data (ORGAN_PREFERRED_VOLTAGE).
#
# ABC data = adult tissue expression → tells us WHICH channels dominate
# Levin data = embryonic voltage map → tells us WHAT voltage each territory wants
# Combined: ABC ratios + Levin anchor = SE-derived cymatic conductances
def _compute_organ_conductances():
    """Convert ABC activity scores to conductance profiles per organ.

    Uses ABC ratios for relative channel balance, anchored to achieve
    Levin developmental voltages. This ensures brain territory is
    hyperpolarised (-70mV, ectoderm) and liver is depolarised (-20mV,
    endoderm) while maintaining organ-specific channel signatures.

    Returns dict mapping organ_name -> (g_Na, g_K, g_Ca, g_Cl, g_gj)
    """
    import numpy as np

    profiles = {}

    for organ, acts in _ABC_ION_CHANNEL_ACTIVITY.items():
        na_raw, k_raw, ca_raw, cl_raw, gj_raw = acts

        # Normalize ABC activity to relative fractions per organ
        total = na_raw + k_raw + ca_raw + cl_raw
        f_Na = na_raw / total
        f_K = k_raw / total
        f_Ca = ca_raw / total
        f_Cl = cl_raw / total

        # Target resting voltage from Levin data
        v_target = ORGAN_PREFERRED_VOLTAGE.get(organ, -50.0)

        # Total conductance budget -- scale based on how active the organ is
        # More active organs (muscle, gut) get higher total conductance
        median_total = 1100.0  # Rough median of all organ totals
        g_budget = 2.5 * (total / median_total)  # 2.5 mS/cm2 at median
        g_budget = np.clip(g_budget, 1.0, 5.0)

        # Distribute budget according to ABC fractions
        g_Na = g_budget * f_Na
        g_K = g_budget * f_K
        g_Ca = g_budget * f_Ca
        g_Cl = g_budget * f_Cl

        # For resting potential calculation, only a fraction of Ca2+ channels
        # contribute to steady-state V_rest. Most Ca2+ channels are voltage-gated
        # and closed at rest -- their activity score reflects signaling capacity,
        # not resting current. Use 10% of g_Ca for resting potential computation.
        g_Ca_rest = g_Ca * 0.10

        # Solve for g_K to hit target voltage:
        # V_target = (g_Na*E_Na + g_K*E_K + g_Ca_rest*E_Ca + g_Cl*E_Cl) /
        #            (g_Na + g_K + g_Ca_rest + g_Cl)
        numerator = g_Na * (E_NA - v_target) + g_Ca_rest * (E_CA - v_target) + g_Cl * (E_CL - v_target)
        denominator = v_target - E_K  # v_target - (-77)

        if abs(denominator) > 1e-6:
            g_K_adjusted = numerator / denominator
            # Keep K+ adjustment within range of ABC-derived value
            # Wider range (10x) for hyperpolarised organs (ectoderm, brain)
            # where K+ dominance is the genomic signature
            max_factor = 10.0 if v_target < -50 else 5.0
            g_K_adjusted = np.clip(g_K_adjusted, g_K * 0.2, g_K * max_factor)
            g_K = g_K_adjusted

        # Clamp all to physiological range
        # g_K upper limit raised to 45.0 for neural dominance
        g_Na = np.clip(g_Na, 0.05, 3.0)
        g_K = np.clip(g_K, 0.1, 45.0)
        g_Ca = np.clip(g_Ca, 0.01, 1.0)
        g_Cl = np.clip(g_Cl, 0.01, 0.5)

        # Gap junction conductance from ABC GJ data
        gj_median = 75.0  # Rough median across organs
        g_gj = DEFAULT_G_GJ * (gj_raw / gj_median)
        g_gj = np.clip(g_gj, 0.01, 0.15)

        profiles[organ] = (float(g_Na), float(g_K), float(g_Ca), float(g_Cl), float(g_gj))

    # Paired organs: recompute with their specific target voltages
    # (kidney and lung base profiles used -50 mV default, but
    #  kidney_left/right target -35 mV and lung_left/right target -25 mV)
    for paired, base in [("kidney_left", "kidney"), ("kidney_right", "kidney"),
                         ("lung_left", "lung"), ("lung_right", "lung")]:
        v_target = ORGAN_PREFERRED_VOLTAGE.get(paired, -50.0)
        base_profile = profiles[base]
        g_Na_p, g_K_p, g_Ca_p, g_Cl_p, g_gj_p = base_profile
        g_Ca_rest_p = g_Ca_p * 0.10

        # Re-solve g_K for the paired organ's specific target voltage
        numerator = g_Na_p * (E_NA - v_target) + g_Ca_rest_p * (E_CA - v_target) + g_Cl_p * (E_CL - v_target)
        denominator = v_target - E_K
        if abs(denominator) > 1e-6:
            g_K_adj = numerator / denominator
            g_K_adj = np.clip(g_K_adj, g_K_p * 0.2, g_K_p * 5.0)
        else:
            g_K_adj = g_K_p
        g_K_adj = np.clip(g_K_adj, 0.1, 6.0)

        profiles[paired] = (g_Na_p, float(g_K_adj), g_Ca_p, g_Cl_p, g_gj_p)

    return profiles


# Precompute at import time (no I/O, just arithmetic)
ORGAN_CONDUCTANCE_PROFILES = _compute_organ_conductances()

# Serotonin-like signaling parameters
SEROTONIN_TRANSPORT_RATE = 0.02    # Voltage-dependent transport rate
SEROTONIN_BASELINE = 0.5           # Baseline concentration (arbitrary units)
SEROTONIN_DECAY = 0.01             # Decay rate per step

# Morphogen electrophoresis parameters
ELECTROPHORESIS_RATE = 0.01        # Voltage-driven morphogen transport rate
MORPHOGEN_DIFFUSION = 0.005        # Passive diffusion rate

# Signal accumulation parameters
SIGNAL_ACCUMULATION_RATE = 0.1     # How fast signals build in the landscape
SIGNAL_MEMORY_DECAY = 0.98         # Slow decay (long memory)


# =============================================================================
# SignalingLandscape (accumulated voltage-driven signals)
# =============================================================================

@dataclass
class SignalingLandscape:
    """Accumulated signaling landscape driven by the voltage pattern.

    This is the 'KV cache' -- voltage drives signal redistribution over
    many steps, and the accumulated landscape is what fate decisions read from.

    Three signal channels:
        calcium_integral: Accumulated Ca2+ influx (voltage-gated channels)
        serotonin: Voltage-dependent transporter redistribution (Levin 5-HT)
        morphogen_field: Electrophoresed morphogen gradients through gap junctions
    """
    n_cells: int
    calcium_integral: np.ndarray    # (n_cells,) accumulated Ca2+ signal
    serotonin: np.ndarray           # (n_cells,) serotonin-like concentration
    morphogen_field: np.ndarray     # (n_cells, 3) directional morphogen gradients
    steps_accumulated: int = 0      # How many steps of signal have built up

    def maturity(self) -> float:
        """How mature is the signaling landscape (0-1)?

        Returns a value indicating how much signal has accumulated.
        Used to gate when fate decisions should start reading from the cache.
        """
        # Sigmoid on steps -- ramps up around 20 steps, saturates by 40
        return 1.0 / (1.0 + np.exp(-(self.steps_accumulated - 20) / 5.0))


def create_signaling_landscape(n_cells: int) -> SignalingLandscape:
    """Create empty signaling landscape."""
    return SignalingLandscape(
        n_cells=n_cells,
        calcium_integral=np.zeros(n_cells),
        serotonin=np.ones(n_cells) * SEROTONIN_BASELINE,
        morphogen_field=np.zeros((n_cells, 3)),
        steps_accumulated=0,
    )


# =============================================================================
# EmbryoBioelectricState
# =============================================================================

@dataclass
class EmbryoBioelectricState:
    """Bioelectric state for the growing embryo (dynamic cell count)."""

    n_cells: int

    # Electrical state (per cell)
    voltage: np.ndarray       # (n_cells,) membrane potential (mV)
    calcium: np.ndarray       # (n_cells,) intracellular Ca2+ (uM)

    # Ion channel conductances (per cell, mS/cm2)
    g_Na: np.ndarray
    g_K: np.ndarray
    g_Ca: np.ndarray
    g_Cl: np.ndarray

    # Gap junction conductance (scalar, uniform)
    g_gj: float = DEFAULT_G_GJ

    # Topology -- adjacency is a sparse CSR matrix (k-NN gap-junction graph).
    # Built by cKDTree.query_pairs in DevelopmentalBioelectricField.rebuild_adjacency.
    adjacency: sp.csr_matrix = field(default=None)   # (n_cells, n_cells), sparse
    positions: np.ndarray = field(default=None)      # (n_cells, 3)

    def __post_init__(self):
        if self.adjacency is None:
            self.adjacency = sp.csr_matrix((self.n_cells, self.n_cells), dtype=np.float64)
        if self.positions is None:
            self.positions = np.zeros((self.n_cells, 3))


def create_initial_state(n_cells: int, positions: np.ndarray) -> EmbryoBioelectricState:
    """Create initial bioelectric state with SE-derived organ voltage territories.

    Replaces the simple linear A-P gradient with organ-specific conductance
    profiles derived from the ABC database (Nasser 2021). Each cell's
    conductances are set based on proximity to organ target positions,
    creating complex voltage domains that act as the cymatic standing wave.

    The SE landscape for ion channel genes IS the cymatic generator:
      SE landscape → ion channel gene expression → spatial conductances
        → voltage pattern (standing wave) → signaling landscape → fate

    For early embryo (few cells), blends nearby organ profiles based on
    distance. As cells multiply and cluster near organ positions, they
    acquire organ-specific conductance signatures.
    """
    g_Na = np.zeros(n_cells)
    g_K = np.zeros(n_cells)
    g_Ca = np.zeros(n_cells)
    g_Cl = np.zeros(n_cells)
    g_gj_per_cell = np.zeros(n_cells)

    organ_names = list(HUMAN_ORGANS.keys())
    organ_positions = np.array([HUMAN_ORGANS[o].position_3d for o in organ_names])

    # Adaptive sigma: sharp for many cells (cells resolve organ territories),
    # broader for few cells (uniform-ish embryo). Range: 0.08 (sharp) to 0.3 (broad)
    sigma = max(0.06, 0.20 / (1.0 + n_cells / 20.0))

    for i in range(n_cells):
        pos = positions[i]

        # Compute squared distance to each organ center
        dists = np.linalg.norm(organ_positions - pos, axis=1)

        # Gaussian weighting (sharper falloff than exponential)
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        weights /= (weights.sum() + 1e-10)

        # Blend organ conductance profiles by proximity weights
        for j, organ_name in enumerate(organ_names):
            if organ_name in ORGAN_CONDUCTANCE_PROFILES:
                profile = ORGAN_CONDUCTANCE_PROFILES[organ_name]
                g_Na[i] += weights[j] * profile[0]
                g_K[i] += weights[j] * profile[1]
                g_Ca[i] += weights[j] * profile[2]
                g_Cl[i] += weights[j] * profile[3]
                g_gj_per_cell[i] += weights[j] * profile[4]

    # Initial voltage from conductance-determined resting potential
    # Use 10% of g_Ca for resting potential (most Ca2+ channels closed at rest)
    # This matches the profile computation which also uses g_Ca*0.10
    g_Ca_rest = g_Ca * 0.10
    g_total = g_Na + g_K + g_Ca_rest + g_Cl
    voltage = (g_Na * E_NA + g_K * E_K + g_Ca_rest * E_CA + g_Cl * E_CL) / (g_total + 1e-10)

    # Average gap junction conductance
    mean_g_gj = float(g_gj_per_cell.mean()) if n_cells > 0 else DEFAULT_G_GJ

    return EmbryoBioelectricState(
        n_cells=n_cells,
        voltage=voltage,
        calcium=np.ones(n_cells) * CA_REST,
        g_Na=g_Na,
        g_K=g_K,
        g_Ca=g_Ca,
        g_Cl=g_Cl,
        g_gj=mean_g_gj,
        adjacency=sp.csr_matrix((n_cells, n_cells), dtype=np.float64),
        positions=positions.copy(),
    )


# =============================================================================
# DevelopmentalBioelectricField
# =============================================================================

class DevelopmentalBioelectricField:
    """BETSE-style Hodgkin-Huxley dynamics for embryonic tissue.

    Ported from multi_organ_betse.py (ion current math) and
    neural_betse_adapter.py (gap junction Laplacian).

    Supports dynamic adjacency rebuild when cells divide.
    """

    def __init__(self, dt: float = 0.1):
        """
        Args:
            dt: Integration time step (ms) for a single substep.
        """
        self.dt = dt

    def rebuild_adjacency(self, state: EmbryoBioelectricState):
        """Rebuild adjacency matrix from cell positions via cKDTree.

        Cells within GAP_JUNCTION_THRESHOLD of each other are connected.
        Uses scipy.spatial.cKDTree.query_pairs for O(N log N) construction
        instead of the O(N^2) double-loop, and stores the result as a
        scipy.sparse.csr_matrix so downstream Laplacian and neighbor
        lookups remain cheap as n_cells grows toward Silic-atlas scale (~3k).
        """
        n = state.n_cells
        if n == 0:
            state.adjacency = sp.csr_matrix((0, 0), dtype=np.float64)
            return

        pos = state.positions
        # query_pairs returns the set of (i, j) with i < j and dist < r
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=GAP_JUNCTION_THRESHOLD, output_type='ndarray')

        if len(pairs) == 0:
            state.adjacency = sp.csr_matrix((n, n), dtype=np.float64)
            return

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(rows), dtype=np.float64)
        state.adjacency = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    def substep(self, state: EmbryoBioelectricState):
        """Run one integration substep of bioelectric dynamics.

        C_m * dV/dt = -sum(I_ion) + I_gap
        """
        n = state.n_cells
        if n == 0:
            return

        V = state.voltage.copy()

        # Ionic currents (simplified HH without gating variables)
        I_Na = state.g_Na * (V - E_NA)
        I_K = state.g_K * (V - E_K)
        I_Ca = state.g_Ca * (V - E_CA)
        I_Cl = state.g_Cl * (V - E_CL)
        I_ion = I_Na + I_K + I_Ca + I_Cl

        # Gap junction current via Laplacian: I_gap = g_gj * (D - A) @ V
        # Sparse: D is the diagonal of row-sums, L @ V = D*V - A @ V.
        adj = state.adjacency
        degree = np.asarray(adj.sum(axis=1)).ravel()
        I_gap = state.g_gj * (degree * V - adj.dot(V))

        # Voltage update (Forward Euler)
        dV_dt = (-I_ion - I_gap) / C_M
        state.voltage += self.dt * dV_dt

        # Calcium dynamics
        dCa_dt = -CA_ALPHA * I_Ca - (state.calcium - CA_REST) / CA_TAU
        state.calcium += self.dt * dCa_dt

        # Clip to physiological ranges
        state.voltage = np.clip(state.voltage, -150.0, 50.0)
        state.calcium = np.clip(state.calcium, 0.05, 10.0)

    def step(self, state: EmbryoBioelectricState, n_substeps: int = BIOELECTRIC_SUBSTEPS):
        """Run multiple substeps for numerical stability."""
        for _ in range(n_substeps):
            self.substep(state)


# =============================================================================
# VmemGradientComputer
# =============================================================================

class VmemGradientComputer:
    """Computes spatial voltage gradient from irregular cell mesh.

    Uses weighted finite difference:
        dV/dx[i] = sum_j(adj[i,j] * (V[j]-V[i]) * (x[j]-x[i]))
                  / sum_j(adj[i,j] * (x[j]-x[i])^2)

    Outputs gradient vectors (n_cells, 3) and positional codes [AP, DV, LR].
    """

    def compute_gradient(self, state: EmbryoBioelectricState) -> np.ndarray:
        """Compute voltage gradient at each cell.

        Returns:
            (n_cells, 3) gradient vectors [dV/dx, dV/dy, dV/dz]
        """
        n = state.n_cells
        gradient = np.zeros((n, 3))

        # CSR row slices give per-cell neighbor indices in O(deg(i)) -- no
        # whole-row scan needed.
        adj = state.adjacency
        indptr = adj.indptr
        indices = adj.indices
        data = adj.data

        for i in range(n):
            row_start, row_end = indptr[i], indptr[i + 1]
            neighbors = indices[row_start:row_end]
            if len(neighbors) == 0:
                continue
            weights = data[row_start:row_end]

            for axis in range(3):
                dx = state.positions[neighbors, axis] - state.positions[i, axis]
                dV = state.voltage[neighbors] - state.voltage[i]

                numerator = np.sum(weights * dV * dx)
                denominator = np.sum(weights * dx * dx)

                if abs(denominator) > 1e-10:
                    gradient[i, axis] = numerator / denominator

        return gradient

    def compute_positional_codes(
        self, state: EmbryoBioelectricState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute positional codes from voltage gradient.

        Returns:
            (AP, DV, LR) arrays each of shape (n_cells,)
            AP: anterior-posterior (from x-gradient)
            DV: dorsal-ventral (from y-gradient)
            LR: left-right (from z-gradient)
        """
        gradient = self.compute_gradient(state)

        # Normalize gradient magnitude to 0-1 range per axis
        ap = self._normalize_signal(gradient[:, 0])
        dv = self._normalize_signal(gradient[:, 1])
        lr = self._normalize_signal(gradient[:, 2])

        return ap, dv, lr

    @staticmethod
    def _normalize_signal(signal: np.ndarray) -> np.ndarray:
        """Normalize signal to 0-1 range."""
        vmin, vmax = signal.min(), signal.max()
        if vmax - vmin < 1e-10:
            return np.full_like(signal, 0.5)
        return (signal - vmin) / (vmax - vmin)


# =============================================================================
# ElectrotaxisController
# =============================================================================

class ElectrotaxisController:
    """Cathodal cell migration: cells move toward lower voltage.

    migration_bias[i] = -gain * gradient[i]

    This bias is added to the MLP migration output in the developmental
    simulator.
    """

    def __init__(self, gain: float = 0.05):
        self.gain = gain

    def compute_migration_bias(
        self, gradient: np.ndarray
    ) -> np.ndarray:
        """Compute electrotaxis migration bias.

        Args:
            gradient: (n_cells, 3) voltage gradient from VmemGradientComputer

        Returns:
            (n_cells, 3) migration bias vectors
        """
        return -self.gain * gradient


# =============================================================================
# BioelectricFateModulator
# =============================================================================

class BioelectricFateModulator:
    """Voltage modulates SE differentiation probabilities.

    Levin data (hardcoded): ectoderm prefers -70mV, mesoderm -30mV,
    endoderm -20mV. Uses Gaussian preference model.

    Blended: 70% genomic + 30% bioelectric.
    Backward-compatible: no voltage = pure genomic behavior.
    """

    def __init__(
        self,
        modulation_strength: float = BIOELECTRIC_MODULATION_STRENGTH,
        sigma: float = VOLTAGE_PREFERENCE_SIGMA,
    ):
        self.modulation_strength = modulation_strength
        self.sigma = sigma

        # Build organ name -> index mapping
        self.organ_names = list(HUMAN_ORGANS.keys())

    def compute_voltage_preference(
        self, cell_voltage: float
    ) -> np.ndarray:
        """Compute how much each organ fate is favored by the current voltage.

        Returns:
            (n_organs,) preference scores (0-1), higher = more preferred
        """
        n_organs = len(self.organ_names)
        preferences = np.zeros(n_organs)

        for i, organ_name in enumerate(self.organ_names):
            preferred_v = ORGAN_PREFERRED_VOLTAGE.get(organ_name, -50.0)
            diff = cell_voltage - preferred_v
            preferences[i] = np.exp(-0.5 * (diff / self.sigma) ** 2)

        # Normalize to sum to 1
        total = preferences.sum()
        if total > 1e-10:
            preferences /= total

        return preferences

    def modulate_fate_probabilities(
        self,
        genomic_probs: np.ndarray,
        cell_voltage: float,
        cell_calcium: Optional[float] = None,
        override_strength: Optional[float] = None,
        landscape_signals: Optional[Dict] = None,
    ) -> np.ndarray:
        """Blend genomic and bioelectric fate probabilities.

        When landscape_signals are provided, fate is driven by the
        accumulated signaling landscape (the KV cache) rather than
        instantaneous voltage. This is the temporally-phased mode.

        Args:
            genomic_probs: (n_organs,) from MLP/SE differentiation
            cell_voltage: Current membrane potential (mV)
            cell_calcium: Current calcium (uM, unused for now, future use)
            override_strength: If set, use this instead of default modulation_strength
            landscape_signals: Optional dict with 'calcium_integral', 'serotonin',
                             'morphogen_field' for landscape-driven fate modulation

        Returns:
            (n_organs,) blended probabilities
        """
        alpha = override_strength if override_strength is not None else self.modulation_strength

        if alpha < 1e-6:
            return genomic_probs.copy()

        if landscape_signals is not None:
            # Landscape-driven: use accumulated signals, not instantaneous voltage
            # Calcium integral biases toward organs that want high Ca2+ (endoderm)
            # Serotonin concentration biases toward organs sensitive to 5-HT
            voltage_prefs = self.compute_voltage_preference(cell_voltage)

            # Modulate voltage prefs by landscape signals
            ca_int = landscape_signals.get('calcium_integral', 0.0)
            serotonin = landscape_signals.get('serotonin', SEROTONIN_BASELINE)

            # High calcium -> favour depolarised fates (endoderm/mesoderm)
            # High serotonin -> favour anterior fates (ectoderm, Levin's 5-HT data)
            ca_factor = 1.0 + np.clip(ca_int, 0, 2.0)  # 1x to 3x boost
            sero_factor = serotonin / SEROTONIN_BASELINE  # relative to baseline

            for i, organ_name in enumerate(self.organ_names):
                preferred_v = ORGAN_PREFERRED_VOLTAGE.get(organ_name, -50.0)
                if preferred_v > -35.0:  # Depolarised fates (endoderm)
                    voltage_prefs[i] *= ca_factor
                if preferred_v < -55.0:  # Hyperpolarised fates (ectoderm)
                    voltage_prefs[i] *= sero_factor

            # Renormalise voltage prefs
            vp_total = voltage_prefs.sum()
            if vp_total > 1e-10:
                voltage_prefs /= vp_total
        else:
            voltage_prefs = self.compute_voltage_preference(cell_voltage)

        # Blend: (1 - alpha) * genomic + alpha * bioelectric
        blended = (1.0 - alpha) * genomic_probs + alpha * voltage_prefs

        # Renormalize
        total = blended.sum()
        if total > 1e-10:
            blended /= total

        return blended


# =============================================================================
# VoltagePatternMemory
# =============================================================================

class VoltagePatternMemory:
    """Organ boundary definition via gap junction modulation.

    Compartments defined from HUMAN_ORGANS positions + resting voltages.
    Gap junction conductance reduced at organ boundaries: cells with
    different committed fates get 10% coupling.
    """

    BOUNDARY_COUPLING_FACTOR = 0.1  # 10% coupling at organ boundaries

    def modulate_gap_junctions(
        self,
        state: EmbryoBioelectricState,
        cell_fates: List[Optional[str]],
    ):
        """Reduce gap junction conductance at organ boundaries.

        Cells with different committed fates get reduced coupling.
        This creates voltage compartments that reinforce organ territories.

        Args:
            state: Current bioelectric state (adjacency modified in place)
            cell_fates: Per-cell organ identity (None if uncommitted)
        """
        n = state.n_cells
        if n == 0 or state.adjacency.nnz == 0:
            return

        # Vectorized boundary modulation over existing edges only.
        # Map each fate (or None) to an int so we can compare with numpy.
        # -1 sentinel marks uncommitted cells (always pass through).
        fate_to_id: Dict[str, int] = {}
        fate_ids = np.empty(n, dtype=np.int64)
        for i, f in enumerate(cell_fates[:n]):
            if f is None:
                fate_ids[i] = -1
            else:
                if f not in fate_to_id:
                    fate_to_id[f] = len(fate_to_id)
                fate_ids[i] = fate_to_id[f]

        coo = state.adjacency.tocoo()
        fi = fate_ids[coo.row]
        fj = fate_ids[coo.col]
        boundary = (fi != -1) & (fj != -1) & (fi != fj)
        if boundary.any():
            coo.data = coo.data.copy()
            coo.data[boundary] *= self.BOUNDARY_COUPLING_FACTOR
            state.adjacency = coo.tocsr()


# =============================================================================
# ApoptosisController
# =============================================================================

class ApoptosisController:
    """Depolarization-triggered programmed cell death for sculpting.

    Cells that are:
    1. Committed to a fate
    2. Depolarized (V > DEPOLARIZATION_THRESHOLD)
    3. More than APOPTOSIS_MISMATCH_THRESHOLD away from their preferred voltage

    ...are marked for removal (apoptosis). This sculpts organ boundaries
    by eliminating cells that ended up in the wrong voltage territory.
    """

    def __init__(
        self,
        depolarization_threshold: float = DEPOLARIZATION_THRESHOLD,
        mismatch_threshold: float = APOPTOSIS_MISMATCH_THRESHOLD,
    ):
        self.depolarization_threshold = depolarization_threshold
        self.mismatch_threshold = mismatch_threshold

    def find_apoptotic_cells(
        self,
        state: EmbryoBioelectricState,
        cell_fates: List[Optional[str]],
    ) -> List[int]:
        """Identify cells that should undergo apoptosis.

        Returns:
            List of cell indices to remove.
        """
        apoptotic = []
        for i in range(state.n_cells):
            fate = cell_fates[i]
            if fate is None:
                continue  # Uncommitted cells don't undergo fate-mismatch apoptosis

            v = state.voltage[i]

            # Must be depolarized
            if v < self.depolarization_threshold:
                continue

            # Check mismatch with preferred voltage
            preferred = ORGAN_PREFERRED_VOLTAGE.get(fate, -50.0)
            mismatch = abs(v - preferred)

            if mismatch > self.mismatch_threshold:
                apoptotic.append(i)

        return apoptotic


# =============================================================================
# ECMLayer (Phase 4: Pattern Hardening)
# =============================================================================

# ECM parameters -- tuned to allow clustering before locking
# At 0.005/step + 30 step delay: locking starts ~step 130 (100 steps after phase 3)
# Previous: 0.01/step + 10 delay = locked at step 80 (still too fast)
# Key: cells need 30+ steps of free migration in phase 3 to cluster by fate
ECM_DEPOSITION_RATE = 0.005       # Very slow ECM accumulation
ECM_LOCK_THRESHOLD = 0.5          # ECM density at which cells become locked
ECM_MIGRATION_DAMPING = 0.85      # How much ECM damps migration (was 0.9)
ECM_DC_FIELD_STRENGTH = 0.02      # Persistent DC field from fixed charges
ECM_DEPOSITION_DELAY = 30         # Steps after phase 3 starts before ECM deposits


@dataclass
class ECMState:
    """Extracellular matrix state for the developing embryo.

    ECM provides:
    1. Mechanical scaffolding (locks cell positions)
    2. DC electric field (Donnan potential from GAG fixed charges)
    3. Organ-specific matrix composition
    """
    n_cells: int
    density: np.ndarray             # (n_cells,) ECM density 0-1 (0=none, 1=fully deposited)
    composition: List[Optional[str]]  # Per-cell ECM type (organ-specific, None if no ECM)
    dc_field: np.ndarray            # (n_cells, 3) persistent DC field from fixed charges
    locked: np.ndarray              # (n_cells,) bool -- whether cell is mechanically locked

    def fraction_locked(self) -> float:
        """Fraction of cells that are mechanically locked by ECM."""
        if self.n_cells == 0:
            return 0.0
        return float(self.locked.sum()) / self.n_cells


def create_ecm_state(n_cells: int) -> ECMState:
    """Create empty ECM state."""
    return ECMState(
        n_cells=n_cells,
        density=np.zeros(n_cells),
        composition=[None] * n_cells,
        dc_field=np.zeros((n_cells, 3)),
        locked=np.zeros(n_cells, dtype=bool),
    )


class ECMController:
    """Controls ECM deposition and mechanical locking.

    Phase 4 of development: committed cells secrete ECM proteins,
    which progressively lock the tissue pattern into permanent structure.

    The ECM also generates a persistent DC electric field from the
    Donnan potential of fixed charges in the glycosaminoglycans.
    This replaces the transient developmental voltage pattern.
    """

    def __init__(
        self,
        deposition_rate: float = ECM_DEPOSITION_RATE,
        lock_threshold: float = ECM_LOCK_THRESHOLD,
        migration_damping: float = ECM_MIGRATION_DAMPING,
    ):
        self.deposition_rate = deposition_rate
        self.lock_threshold = lock_threshold
        self.migration_damping = migration_damping

    def deposit(
        self,
        ecm: ECMState,
        cell_fates: List[Optional[str]],
        positions: np.ndarray,
    ):
        """Deposit ECM around committed cells.

        Only committed cells (with organ identity) secrete ECM.
        ECM density increases over time up to 1.0.

        Args:
            ecm: ECM state (modified in place)
            cell_fates: Per-cell organ identity
            positions: Current cell positions
        """
        n = min(ecm.n_cells, len(cell_fates))

        for i in range(n):
            if cell_fates[i] is not None:
                # Committed cell deposits ECM
                ecm.density[i] = min(1.0, ecm.density[i] + self.deposition_rate)
                ecm.composition[i] = cell_fates[i]

                # Lock cell if ECM density exceeds threshold
                if ecm.density[i] >= self.lock_threshold:
                    ecm.locked[i] = True

    def compute_dc_field(
        self,
        ecm: ECMState,
        positions: np.ndarray,
        voltage_gradient: np.ndarray,
    ):
        """Compute persistent DC field from ECM fixed charges.

        The DC field is derived from the developmental voltage gradient
        but is maintained by ECM charge density (Donnan potential), not
        by ion channel activity. It persists after the developmental
        voltage pattern fades.

        Args:
            ecm: ECM state (dc_field modified in place)
            positions: Cell positions
            voltage_gradient: Current voltage gradient from bioelectric layer
        """
        n = min(ecm.n_cells, len(voltage_gradient))

        for i in range(n):
            if ecm.density[i] > 0:
                # DC field is proportional to ECM density and derived
                # from the developmental voltage gradient (frozen in)
                ecm.dc_field[i] = (
                    ecm.dc_field[i] * 0.95  # Slow decay of existing field
                    + ECM_DC_FIELD_STRENGTH * ecm.density[i] * voltage_gradient[i]
                )

    def compute_migration_damping(self, ecm: ECMState) -> np.ndarray:
        """Compute per-cell migration damping factor from ECM.

        Locked cells cannot migrate. Partially deposited ECM slows migration.

        Returns:
            (n_cells,) damping factors (1.0 = free, 0.0 = fully locked)
        """
        damping = np.ones(ecm.n_cells)
        for i in range(ecm.n_cells):
            if ecm.locked[i]:
                damping[i] = 1.0 - self.migration_damping  # Nearly zero
            elif ecm.density[i] > 0:
                # Progressive damping proportional to ECM density
                damping[i] = 1.0 - self.migration_damping * ecm.density[i]
        return damping

    def grow(self, ecm: ECMState, n_new: int):
        """Extend ECM state when cells divide."""
        n_added = n_new - ecm.n_cells
        if n_added <= 0:
            return

        ecm.density = np.concatenate([ecm.density, np.zeros(n_added)])
        ecm.composition.extend([None] * n_added)
        ecm.dc_field = np.concatenate([ecm.dc_field, np.zeros((n_added, 3))])
        ecm.locked = np.concatenate([ecm.locked, np.zeros(n_added, dtype=bool)])
        ecm.n_cells = n_new

    def remove_cells(self, ecm: ECMState, indices: List[int]):
        """Remove cells from ECM state."""
        if not indices:
            return
        mask = np.ones(ecm.n_cells, dtype=bool)
        mask[indices] = False

        ecm.density = ecm.density[mask]
        ecm.composition = [ecm.composition[i] for i in range(ecm.n_cells) if mask[i]]
        ecm.dc_field = ecm.dc_field[mask]
        ecm.locked = ecm.locked[mask]
        ecm.n_cells = int(mask.sum())


# =============================================================================
# SignalingAccumulator
# =============================================================================

class SignalingAccumulator:
    """Accumulates voltage-driven signals into the signaling landscape.

    This is the bridge between the cymatic attention head (voltage pattern)
    and the channel MLP (fate decisions). The voltage pattern drives:

    1. Ca2+ influx through voltage-gated channels -> calcium_integral
    2. Serotonin redistribution via voltage-dependent transporters -> serotonin
    3. Morphogen electrophoresis through gap junctions -> morphogen_field

    These signals accumulate over many developmental steps, building the
    KV cache that fate decisions will eventually read from.
    """

    def accumulate(
        self,
        state: EmbryoBioelectricState,
        landscape: SignalingLandscape,
        gradient: np.ndarray,
    ):
        """Accumulate one step of voltage-driven signals.

        Args:
            state: Current bioelectric state (voltage, calcium, adjacency)
            landscape: Signaling landscape to update (modified in place)
            gradient: (n_cells, 3) voltage gradient from VmemGradientComputer
        """
        n = state.n_cells

        # Grow landscape if cells divided
        if landscape.n_cells < n:
            n_added = n - landscape.n_cells
            landscape.calcium_integral = np.concatenate([
                landscape.calcium_integral, np.zeros(n_added)
            ])
            landscape.serotonin = np.concatenate([
                landscape.serotonin, np.ones(n_added) * SEROTONIN_BASELINE
            ])
            landscape.morphogen_field = np.concatenate([
                landscape.morphogen_field, np.zeros((n_added, 3))
            ])
            landscape.n_cells = n

        # 1. Calcium integral: accumulate Ca2+ influx from voltage-gated channels
        #    Ca2+ influx is proportional to how far V is from E_Ca and g_Ca
        #    More depolarized -> more Ca2+ influx (channels open at ~-40mV)
        ca_activation = 1.0 / (1.0 + np.exp(-(state.voltage[:n] + 40.0) / 10.0))
        ca_influx = state.g_Ca[:n] * ca_activation * SIGNAL_ACCUMULATION_RATE
        landscape.calcium_integral[:n] = (
            SIGNAL_MEMORY_DECAY * landscape.calcium_integral[:n] + ca_influx
        )

        # 2. Serotonin redistribution via voltage-dependent transporters
        #    Serotonin moves DOWN the voltage gradient through gap junctions
        #    (Levin: 5-HT transported by voltage-dependent GJC1 channels)
        if n > 1 and state.adjacency.shape[0] >= n:
            adj = state.adjacency
            indptr = adj.indptr
            indices = adj.indices
            data = adj.data
            for i in range(n):
                row_start, row_end = indptr[i], indptr[i + 1]
                neighbors = indices[row_start:row_end]
                if len(neighbors) == 0:
                    continue
                weights = data[row_start:row_end]
                for k, j in enumerate(neighbors):
                    if j >= n:
                        continue
                    # Transport rate proportional to voltage difference
                    dV = state.voltage[i] - state.voltage[j]
                    transport = SEROTONIN_TRANSPORT_RATE * dV * weights[k]
                    # Move serotonin from high-V to low-V cell
                    amount = np.clip(transport * landscape.serotonin[i], -0.1, 0.1)
                    landscape.serotonin[i] -= amount
                    landscape.serotonin[j] += amount

            # Decay toward baseline
            landscape.serotonin[:n] = (
                SIGNAL_MEMORY_DECAY * landscape.serotonin[:n]
                + (1.0 - SIGNAL_MEMORY_DECAY) * SEROTONIN_BASELINE
            )
            landscape.serotonin[:n] = np.clip(landscape.serotonin[:n], 0.05, 5.0)

        # 3. Morphogen electrophoresis: voltage gradient drives charged morphogens
        #    through gap junctions, creating directional morphogen fields
        landscape.morphogen_field[:n] = (
            SIGNAL_MEMORY_DECAY * landscape.morphogen_field[:n]
            + ELECTROPHORESIS_RATE * gradient[:n]
        )

        landscape.steps_accumulated += 1


# =============================================================================
# BioelectricDevelopmentOrchestrator
# =============================================================================

class BioelectricDevelopmentOrchestrator:
    """Wires all bioelectric components. Single step() call.

    Temporally-phased orchestration (Levin mechanism):
        Phase 1 (<=32 cells): Voltage pattern ESTABLISHES.
            - Ion channel dynamics run, standing wave forms
            - Signaling landscape begins accumulating
            - NO fate modulation yet (attention map still computing)

        Phase 2 (32-128 cells): Signals ACCUMULATE.
            - Voltage-driven Ca2+, serotonin, morphogen redistribution
            - Signaling landscape builds up (KV cache filling)
            - Weak fate modulation begins (landscape maturity gates it)

        Phase 3 (128+ cells): Fate READS from cache.
            - Full fate modulation from accumulated signaling landscape
            - Apoptosis active (sculpting organ boundaries)
            - Pattern memory reinforces compartments
    """

    def __init__(self, dt: float = 0.1, electrotaxis_gain: float = 0.05,
                 use_temporal_phasing: bool = True):
        self.field = DevelopmentalBioelectricField(dt=dt)
        self.gradient_computer = VmemGradientComputer()
        self.electrotaxis = ElectrotaxisController(gain=electrotaxis_gain)
        self.fate_modulator = BioelectricFateModulator()
        self.pattern_memory = VoltagePatternMemory()
        self.apoptosis = ApoptosisController()
        self.accumulator = SignalingAccumulator()
        self.ecm_controller = ECMController()

        self.state: Optional[EmbryoBioelectricState] = None
        self.landscape: Optional[SignalingLandscape] = None
        self.ecm: Optional[ECMState] = None
        self._last_gradient: Optional[np.ndarray] = None
        self.use_temporal_phasing = use_temporal_phasing
        self._steps_in_phase3 = 0  # Track steps since entering phase 3

    def initialize(self, n_cells: int, positions: np.ndarray):
        """Initialize bioelectric state for the embryo.

        Sets up A-P voltage gradient (anterior hyperpolarized, posterior
        depolarized) and builds initial adjacency.
        """
        self.state = create_initial_state(n_cells, positions)
        self.landscape = create_signaling_landscape(n_cells)
        self.ecm = create_ecm_state(n_cells)
        self.field.rebuild_adjacency(self.state)
        logger.info(
            f"Bioelectric layer initialized: {n_cells} cells, "
            f"V range [{self.state.voltage.min():.1f}, {self.state.voltage.max():.1f}] mV"
        )

    def sync_positions(self, positions: np.ndarray):
        """Update cell positions (called after migration/division)."""
        if self.state is None:
            return
        n_new = len(positions)
        n_old = self.state.n_cells

        if n_new != n_old:
            # Cells were added (division) -- extend state arrays
            self._grow_state(n_new, positions)
        else:
            self.state.positions = positions.copy()

    def _grow_state(self, n_new: int, positions: np.ndarray):
        """Extend bioelectric state when new cells are added via division.

        New cells get SE-derived conductances based on their position
        (proximity to organ territories), not flat defaults.
        """
        n_old = self.state.n_cells
        n_added = n_new - n_old

        if n_added <= 0:
            return

        # Compute SE-derived conductances for new cells
        new_positions = positions[n_old:]
        organ_names = list(HUMAN_ORGANS.keys())
        organ_positions = np.array([HUMAN_ORGANS[o].position_3d for o in organ_names])

        new_g_Na = np.zeros(n_added)
        new_g_K = np.zeros(n_added)
        new_g_Ca = np.zeros(n_added)
        new_g_Cl = np.zeros(n_added)

        sigma = max(0.06, 0.20 / (1.0 + n_new / 20.0))

        for i in range(n_added):
            pos = new_positions[i]
            dists = np.linalg.norm(organ_positions - pos, axis=1)
            weights = np.exp(-0.5 * (dists / sigma) ** 2)
            weights /= (weights.sum() + 1e-10)

            for j, organ_name in enumerate(organ_names):
                if organ_name in ORGAN_CONDUCTANCE_PROFILES:
                    profile = ORGAN_CONDUCTANCE_PROFILES[organ_name]
                    new_g_Na[i] += weights[j] * profile[0]
                    new_g_K[i] += weights[j] * profile[1]
                    new_g_Ca[i] += weights[j] * profile[2]
                    new_g_Cl[i] += weights[j] * profile[3]

        # Initial voltage from conductance-determined resting potential
        # Use 10% of g_Ca for resting potential (most Ca2+ channels closed at rest)
        new_g_Ca_rest = new_g_Ca * 0.10
        g_total = new_g_Na + new_g_K + new_g_Ca_rest + new_g_Cl
        new_voltage = (
            new_g_Na * E_NA + new_g_K * E_K + new_g_Ca_rest * E_CA + new_g_Cl * E_CL
        ) / (g_total + 1e-10)

        # Extend state arrays
        self.state.voltage = np.concatenate([self.state.voltage, new_voltage])
        self.state.calcium = np.concatenate(
            [self.state.calcium, np.ones(n_added) * CA_REST]
        )
        self.state.g_Na = np.concatenate([self.state.g_Na, new_g_Na])
        self.state.g_K = np.concatenate([self.state.g_K, new_g_K])
        self.state.g_Ca = np.concatenate([self.state.g_Ca, new_g_Ca])
        self.state.g_Cl = np.concatenate([self.state.g_Cl, new_g_Cl])

        # Update count and positions
        self.state.n_cells = n_new
        self.state.positions = positions.copy()

        # Rebuild adjacency for new topology
        self.state.adjacency = sp.csr_matrix((n_new, n_new), dtype=np.float64)

        # Grow ECM state
        if self.ecm is not None:
            self.ecm_controller.grow(self.ecm, n_new)

    def _get_phase(self) -> int:
        """Determine current developmental phase from cell count."""
        if self.state is None:
            return 1
        n = self.state.n_cells
        if n <= PHASE_1_END:
            return 1
        elif n <= PHASE_2_END:
            return 2
        else:
            return 3

    def step(
        self,
        cell_fates: List[Optional[str]],
        rebuild_adjacency: bool = True,
    ) -> Dict:
        """Run one full bioelectric development step with temporal phasing.

        Phase 1 (<=32 cells): Voltage establishes. No fate modulation.
        Phase 2 (32-128 cells): Signals accumulate. Weak fate modulation.
        Phase 3 (128+ cells): Full fate modulation from signaling landscape.

        Args:
            cell_fates: Per-cell organ identity (None if uncommitted)
            rebuild_adjacency: Whether to rebuild adjacency (True after division)

        Returns:
            Dict with:
                'positional_codes': (AP, DV, LR) arrays
                'migration_bias': (n_cells, 3) electrotaxis vectors
                'apoptotic_indices': list of cell indices to remove
                'voltage': (n_cells,) current voltages
                'calcium': (n_cells,) current calcium
                'phase': current developmental phase (1, 2, or 3)
                'landscape_maturity': how mature the signaling landscape is (0-1)
                'fate_modulation_strength': effective fate modulation strength
        """
        if self.state is None:
            return {
                'positional_codes': (np.array([]), np.array([]), np.array([])),
                'migration_bias': np.zeros((0, 3)),
                'apoptotic_indices': [],
                'voltage': np.array([]),
                'calcium': np.array([]),
                'phase': 1,
                'landscape_maturity': 0.0,
                'fate_modulation_strength': 0.0,
            }

        phase = self._get_phase() if self.use_temporal_phasing else 3

        # 1. Rebuild adjacency if needed
        if rebuild_adjacency:
            self.field.rebuild_adjacency(self.state)

        # 2. Modulate gap junctions at organ boundaries
        #    (active in all phases -- helps stabilise voltage compartments)
        if phase >= 2:
            self.pattern_memory.modulate_gap_junctions(self.state, cell_fates)

        # 3. Run bioelectric dynamics (the cymatic pattern)
        self.field.step(self.state)

        # 4. Compute gradients and positional codes
        gradient = self.gradient_computer.compute_gradient(self.state)
        self._last_gradient = gradient
        ap, dv, lr = self.gradient_computer.compute_positional_codes(self.state)

        # 5. Accumulate signals into landscape (KV cache filling)
        if self.landscape is not None:
            self.accumulator.accumulate(self.state, self.landscape, gradient)

        # 6. Compute electrotaxis bias
        migration_bias = self.electrotaxis.compute_migration_bias(gradient)

        # 7. Find apoptotic cells (only in phase 3 -- need mature landscape)
        apoptotic = []
        if phase >= 3:
            apoptotic = self.apoptosis.find_apoptotic_cells(self.state, cell_fates)

        # 8. ECM deposition (phase 3+: committed cells secrete matrix)
        #    Delayed start: cells need time to cluster by fate before ECM locks
        migration_damping = np.ones(self.state.n_cells)
        if phase >= 3:
            self._steps_in_phase3 += 1

        if self.ecm is not None and phase >= 3 and self._steps_in_phase3 > ECM_DEPOSITION_DELAY:
            self.ecm_controller.deposit(self.ecm, cell_fates, self.state.positions)
            self.ecm_controller.compute_dc_field(self.ecm, self.state.positions, gradient)
            migration_damping = self.ecm_controller.compute_migration_damping(self.ecm)

        # Compute effective fate modulation strength
        if self.use_temporal_phasing and self.landscape is not None:
            maturity = self.landscape.maturity()
            if phase == 1:
                fate_strength = 0.0        # No fate modulation yet
            elif phase == 2:
                fate_strength = maturity * BIOELECTRIC_MODULATION_STRENGTH * 0.5
            else:
                fate_strength = maturity * BIOELECTRIC_MODULATION_STRENGTH
        else:
            fate_strength = BIOELECTRIC_MODULATION_STRENGTH
            maturity = 1.0

        return {
            'positional_codes': (ap, dv, lr),
            'migration_bias': migration_bias,
            'migration_damping': migration_damping,
            'apoptotic_indices': apoptotic,
            'voltage': self.state.voltage.copy(),
            'calcium': self.state.calcium.copy(),
            'phase': phase,
            'landscape_maturity': maturity if self.landscape else 0.0,
            'fate_modulation_strength': fate_strength,
            'ecm_locked_fraction': self.ecm.fraction_locked() if self.ecm else 0.0,
        }

    def remove_cells(self, indices_to_remove: List[int]):
        """Remove apoptotic cells from bioelectric state.

        Args:
            indices_to_remove: Cell indices to delete
        """
        if not indices_to_remove or self.state is None:
            return

        mask = np.ones(self.state.n_cells, dtype=bool)
        mask[indices_to_remove] = False

        self.state.voltage = self.state.voltage[mask]
        self.state.calcium = self.state.calcium[mask]
        self.state.g_Na = self.state.g_Na[mask]
        self.state.g_K = self.state.g_K[mask]
        self.state.g_Ca = self.state.g_Ca[mask]
        self.state.g_Cl = self.state.g_Cl[mask]
        self.state.positions = self.state.positions[mask]

        n_new = mask.sum()
        self.state.n_cells = n_new

        # Rebuild adjacency for remaining cells
        self.state.adjacency = sp.csr_matrix((n_new, n_new), dtype=np.float64)
        self.field.rebuild_adjacency(self.state)

        # Remove from ECM
        if self.ecm is not None:
            self.ecm_controller.remove_cells(self.ecm, indices_to_remove)

    def get_cell_voltage(self, idx: int) -> float:
        """Get voltage for a single cell."""
        if self.state is None or idx >= self.state.n_cells:
            return -70.0
        return float(self.state.voltage[idx])

    def get_cell_calcium(self, idx: int) -> float:
        """Get calcium for a single cell."""
        if self.state is None or idx >= self.state.n_cells:
            return CA_REST
        return float(self.state.calcium[idx])


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("BIOELECTRIC DEVELOPMENT LAYER - STANDALONE TEST")
    print("=" * 70)

    # Test 1: DevelopmentalBioelectricField with 50 cells
    print("\n--- Test 1: Bioelectric Field (50 cells) ---")
    rng = np.random.default_rng(42)
    n = 50
    positions = rng.uniform(0, 1, (n, 3))

    orchestrator = BioelectricDevelopmentOrchestrator()
    orchestrator.initialize(n, positions)

    print(f"Initial voltage range: [{orchestrator.state.voltage.min():.1f}, "
          f"{orchestrator.state.voltage.max():.1f}] mV")
    print(f"Adjacency connections: {int(orchestrator.state.adjacency.sum()) // 2}")

    # Run 5 steps with no committed fates
    fates = [None] * n
    for step in range(5):
        result = orchestrator.step(fates)
        v = result['voltage']
        print(f"  Step {step}: V=[{v.min():.1f}, {v.max():.1f}] mV, "
              f"bias_mag={np.linalg.norm(result['migration_bias'], axis=1).mean():.4f}")

    # Test 2: VmemGradientComputer
    print("\n--- Test 2: Gradient Computation ---")
    gradient = orchestrator._last_gradient
    print(f"Gradient shape: {gradient.shape}")
    print(f"Mean gradient magnitude: {np.linalg.norm(gradient, axis=1).mean():.4f}")
    ap, dv, lr = result['positional_codes']
    print(f"AP range: [{ap.min():.3f}, {ap.max():.3f}]")
    print(f"DV range: [{dv.min():.3f}, {dv.max():.3f}]")

    # Test 3: BioelectricFateModulator
    print("\n--- Test 3: Fate Modulation ---")
    modulator = BioelectricFateModulator()
    organ_names = list(HUMAN_ORGANS.keys())

    # Test at -70 mV (should favor ectoderm/brain)
    prefs_hyper = modulator.compute_voltage_preference(-70.0)
    brain_idx = organ_names.index("brain")
    heart_idx = organ_names.index("heart")
    print(f"At -70 mV: brain={prefs_hyper[brain_idx]:.3f}, heart={prefs_hyper[heart_idx]:.3f}")

    # Test at -30 mV (should favor mesoderm/heart)
    prefs_depol = modulator.compute_voltage_preference(-30.0)
    print(f"At -30 mV: brain={prefs_depol[brain_idx]:.3f}, heart={prefs_depol[heart_idx]:.3f}")

    # Verify brain favored at -70, heart favored at -30
    assert prefs_hyper[brain_idx] > prefs_hyper[heart_idx], "Brain should be favored at -70 mV"
    assert prefs_depol[heart_idx] > prefs_depol[brain_idx], "Heart should be favored at -30 mV"
    print("Fate modulation assertions passed")

    # Test 4: Apoptosis
    print("\n--- Test 4: Apoptosis Controller ---")
    apoptosis = ApoptosisController()
    # Create a scenario: cell committed to brain but at -10 mV (way off from -70)
    test_state = create_initial_state(3, np.array([[0.1, 0.5, 0.5],
                                                     [0.5, 0.5, 0.5],
                                                     [0.9, 0.5, 0.5]]))
    test_state.voltage = np.array([-10.0, -70.0, -15.0])
    test_fates = ["brain", "brain", "gut"]
    apoptotic = apoptosis.find_apoptotic_cells(test_state, test_fates)
    print(f"Apoptotic cells: {apoptotic}")
    # Cell 0: brain at -10 mV, depolarized AND mismatch = 60 mV > 30 -> apoptotic
    # Cell 1: brain at -70 mV, not depolarized -> safe
    # Cell 2: gut at -15 mV, depolarized, mismatch = |-15 - (-25)| = 10 < 30 -> safe
    assert 0 in apoptotic, "Cell 0 (brain at -10mV) should be apoptotic"
    assert 1 not in apoptotic, "Cell 1 (brain at -70mV) should be safe"
    print("Apoptosis assertions passed")

    print("\n" + "=" * 70)
    print("All bioelectric development tests passed")
    print("=" * 70)
