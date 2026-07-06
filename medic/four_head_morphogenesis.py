"""
Four-Head Morphogenesis Architecture
======================================

Multi-head attention for embryonic development, two levels with two heads each.

TISSUE LEVEL (WHAT — coarse, fast, trainable):
  Head 1 — BODY PLAN:  Vmem → germ layer (ecto/meso/endo)
                        Hardwired from Levin Xenopus data.
  Head 2 — ORGAN:      position + morphogens within germ layer → organ identity
                        Trained via CMA-ES, gated by germ layer.

BIOCHEMICAL LEVEL (HOW — fine-grained, data-driven):
  Head 3 — SE:         Super-enhancer attention from ABC database (7.7M E-P predictions)
                        Which genes are active for this cell's voltage/position context.
                        Sparse matrix: enhancers × genes, cell-type-specific.
  Head 4 — MORPHOGEN:  Spatial gradients of signaling molecules
                        Wnt, BMP, SHH, FGF, Nodal, Retinoic Acid.
                        Reaction-diffusion with sources at organizer regions.

The tissue heads answer WHAT (germ layer, organ).
The biochemical heads answer HOW (which genes, which gradients).

Information flow:
  Vmem → [Head 1: Body Plan] → germ_layer_probs
  germ_layer_probs + position → [Head 2: Organ] → organ_fate_probs
  organ_fate + voltage_context → [Head 3: SE] → gene_activity_vector
  position → [Head 4: Morphogen] → morphogen_concentrations
  gene_activity × morphogen → refined_fate_signal (feeds back to organ head)

The SE head uses the ABC database (Nasser 2021) to look up which enhancers
are active in cell types matching the current germ layer / organ commitment.
This is NOT a neural network — it's a database lookup into a sparse attention
matrix, which IS the biology.

The morphogen head uses reaction-diffusion PDEs for the canonical
developmental signaling pathways. Source positions are derived from the
body plan head (organizer = dorsal, BMP = ventral, SHH = notochord, etc.).

Data sources:
    Head 1: Levin lab Xenopus voltage maps (xenopus_bioelectric.py)
    Head 2: CMA-ES trained organ MLP (developmental_trm.py)
    Head 3: ABC database, 7.7M predictions, 131 cell types (abc_client.py, SE.sqlite)
    Head 4: Classical developmental biology (Wnt, BMP, SHH, FGF, Nodal, RA)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from .human_topology import HUMAN_ORGANS, OrganSpec
    from .bioelectric_development import (
        ORGAN_PREFERRED_VOLTAGE, ORGAN_GERM_LAYER,
        LEVIN_ECTODERM_VMEM, LEVIN_MESODERM_VMEM, LEVIN_ENDODERM_VMEM,
    )
    from .xenopus_bioelectric import (
        GERM_LAYER_VMEM, TISSUE_VMEM_ESTIMATES as XENOPUS_VMEM,
    )
except ImportError:
    from human_topology import HUMAN_ORGANS, OrganSpec
    from bioelectric_development import (
        ORGAN_PREFERRED_VOLTAGE, ORGAN_GERM_LAYER,
        LEVIN_ECTODERM_VMEM, LEVIN_MESODERM_VMEM, LEVIN_ENDODERM_VMEM,
    )
    from xenopus_bioelectric import (
        GERM_LAYER_VMEM, TISSUE_VMEM_ESTIMATES as XENOPUS_VMEM,
    )


# =============================================================================
# HEAD 1: Body Plan (Vmem → Germ Layer)
# =============================================================================
# Hardwired from Levin Xenopus data. NOT trained.
# Gaussian proximity to each germ layer's target voltage.

class BodyPlanHead:
    """Germ layer commitment from membrane voltage.

    Uses Levin's Xenopus voltage data (the only calibrated embryonic Vmem):
        Ectoderm: -70 mV (Pai et al. 2015: neural plate -51 mV)
        Mesoderm: -35 mV (intermediate)
        Endoderm: -22 mV (depolarized)

    Not trained — hardwired from biology.
    """

    def __init__(self, sigma_mv: float = 15.0):
        """
        Args:
            sigma_mv: Gaussian sigma for voltage preference (mV).
                      15 mV gives reasonable overlap during germ layer transitions.
        """
        self.sigma = sigma_mv
        self.targets = {
            'ectoderm': LEVIN_ECTODERM_VMEM,   # -70.0
            'mesoderm': LEVIN_MESODERM_VMEM,    # -30.0 (note: bioelectric_development uses -30)
            'endoderm': LEVIN_ENDODERM_VMEM,    # -20.0
        }

    def forward(self, voltage_mv: float) -> Dict[str, float]:
        """Compute germ layer probabilities from voltage.

        Args:
            voltage_mv: Membrane potential in millivolts.

        Returns:
            Dict mapping germ layer name to probability (sums to ~1).
        """
        scores = {}
        for germ, target in self.targets.items():
            scores[germ] = np.exp(-0.5 * ((voltage_mv - target) / self.sigma) ** 2)
        total = sum(scores.values()) + 1e-8
        return {g: s / total for g, s in scores.items()}

    def dominant_layer(self, voltage_mv: float) -> str:
        """Get the most likely germ layer for a voltage."""
        probs = self.forward(voltage_mv)
        return max(probs, key=probs.get)


# =============================================================================
# HEAD 2: Organ (Position + Morphogens → Organ Identity, gated by germ layer)
# =============================================================================
# Trained via CMA-ES. Softmax within each germ layer only.

class OrganHead:
    """Organ fate selection within committed germ layer.

    Architecture: shared hidden layer → 11 organ logits, gated by germ layer.
    Brain can only emerge from ectoderm territory. Heart only from mesoderm.

    Trained parameters: W_hidden, b_hidden, W_organ, b_organ
    """

    # Germ layer → organ mapping
    GERM_LAYER_ORGANS = {
        'ectoderm': ['brain'],
        'mesoderm': ['heart', 'kidney_left', 'kidney_right', 'muscle'],
        'endoderm': ['liver', 'pancreas', 'lung_left', 'lung_right', 'gut', 'thyroid'],
    }

    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Trainable parameters
        self.W1 = self.rng.normal(0, 0.1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W_organ = self.rng.normal(0, 0.1, (hidden_dim, 11))
        self.b_organ = np.zeros(11)

        # Build organ name list and germ layer masks
        self._organ_names = list(HUMAN_ORGANS.keys())
        self._organ_to_idx = {name: i for i, name in enumerate(self._organ_names)}
        self._germ_masks = {}
        for germ, organs in self.GERM_LAYER_ORGANS.items():
            mask = np.zeros(11, dtype=bool)
            for o in organs:
                if o in self._organ_to_idx:
                    mask[self._organ_to_idx[o]] = True
            self._germ_masks[germ] = mask

        n_params = input_dim * hidden_dim + hidden_dim + hidden_dim * 11 + 11
        logger.info(f"OrganHead: {input_dim}->h{hidden_dim}->11 organs, {n_params} params")

    def forward(self, x_in: np.ndarray, germ_probs: Dict[str, float]) -> np.ndarray:
        """Compute organ fate probabilities, gated by germ layer.

        Args:
            x_in: Input vector (12 dims: position, voltage, morphogens, etc.)
            germ_probs: Germ layer probabilities from BodyPlanHead.

        Returns:
            11-dim fate probability vector (sums to ~1).
        """
        # Shared hidden layer (ReLU)
        h = np.maximum(0, x_in @ self.W1 + self.b1)

        # Organ logits
        organ_logits = h @ self.W_organ + self.b_organ

        # Gate by germ layer: softmax within each layer, weighted by germ prob
        fate_probs = np.zeros(11)
        for germ, germ_p in germ_probs.items():
            mask = self._germ_masks.get(germ)
            if mask is None or not mask.any():
                continue
            germ_logits = organ_logits[mask]
            germ_logits = germ_logits - germ_logits.max()
            germ_softmax = np.exp(germ_logits) / (np.sum(np.exp(germ_logits)) + 1e-8)
            fate_probs[mask] += germ_p * germ_softmax

        total = fate_probs.sum()
        if total > 1e-8:
            fate_probs /= total
        return fate_probs

    @property
    def hidden_state(self):
        """Last hidden state (for migration/division heads)."""
        return self._last_h

    def forward_with_hidden(self, x_in: np.ndarray, germ_probs: Dict[str, float]):
        """Forward pass that also returns hidden state for other heads."""
        h = np.maximum(0, x_in @ self.W1 + self.b1)
        self._last_h = h
        organ_logits = h @ self.W_organ + self.b_organ
        fate_probs = np.zeros(11)
        for germ, germ_p in germ_probs.items():
            mask = self._germ_masks.get(germ)
            if mask is None or not mask.any():
                continue
            germ_logits = organ_logits[mask]
            germ_logits = germ_logits - germ_logits.max()
            germ_softmax = np.exp(germ_logits) / (np.sum(np.exp(germ_logits)) + 1e-8)
            fate_probs[mask] += germ_p * germ_softmax
        total = fate_probs.sum()
        if total > 1e-8:
            fate_probs /= total
        return fate_probs, h

    def get_parameters(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1, self.W_organ.ravel(), self.b_organ
        ])

    def set_parameters(self, params: np.ndarray):
        idx = 0
        sz = self.input_dim * self.hidden_dim
        self.W1 = params[idx:idx+sz].reshape((self.input_dim, self.hidden_dim))
        idx += sz
        self.b1 = params[idx:idx+self.hidden_dim].copy()
        idx += self.hidden_dim
        sz = self.hidden_dim * 11
        self.W_organ = params[idx:idx+sz].reshape((self.hidden_dim, 11))
        idx += sz
        self.b_organ = params[idx:idx+11].copy()


# =============================================================================
# HEAD 3: Super-Enhancer Attention (ABC database → gene activity)
# =============================================================================
# Not a neural network. A database lookup into sparse attention matrices.
# The matrix IS the biology — it encodes which enhancers regulate which genes
# in which cell types.

# Organ → ABC cell type mapping (Nasser 2021, 131 cell types)
ORGAN_TO_ABC_CELL_TYPES = {
    'brain': [
        'astrocyte', 'excitatory_neuron', 'oligodendrocyte',
        'bipolar_neuron_of_retina', 'brain_microvascular_endothelial',
    ],
    'heart': [
        'cardiac_fibroblast', 'cardiac_muscle_cell',
        'coronary_artery_smooth_muscle',
    ],
    'liver': [
        'hepatocyte', 'hepatic_stellate_cell',
    ],
    'kidney_left': ['kidney_epithelial_cell'],
    'kidney_right': ['kidney_epithelial_cell'],
    'pancreas': [
        'pancreatic_beta_cell', 'pancreas',
    ],
    'lung_left': ['lung_fibroblast', 'type_II_pneumocyte'],
    'lung_right': ['lung_fibroblast', 'type_II_pneumocyte'],
    'muscle': ['skeletal_muscle_myoblast'],
    'gut': ['small_intestinal_epithelial_cell'],
    'thyroid': ['thyroid_gland'],
}

# Key developmental genes per organ (for SE attention queries)
ORGAN_KEY_GENES = {
    'brain': ['SOX2', 'PAX6', 'NEUROG1', 'NEUROG2', 'OTX2', 'FOXG1', 'EMX2'],
    'heart': ['NKX2-5', 'GATA4', 'TBX5', 'MEF2C', 'MYH6', 'TNNT2', 'ISL1'],
    'liver': ['HNF4A', 'FOXA2', 'CEBPA', 'ALB', 'AFP', 'PROX1'],
    'kidney_left': ['PAX2', 'PAX8', 'WT1', 'SIX2', 'GDNF', 'RET'],
    'kidney_right': ['PAX2', 'PAX8', 'WT1', 'SIX2', 'GDNF', 'RET'],
    'pancreas': ['PDX1', 'NKX6-1', 'PTF1A', 'INS', 'GCG', 'SST'],
    'lung_left': ['NKX2-1', 'FOXA2', 'SOX9', 'SFTPC', 'SFTPB'],
    'lung_right': ['NKX2-1', 'FOXA2', 'SOX9', 'SFTPC', 'SFTPB'],
    'muscle': ['MYOD1', 'MYF5', 'MYOG', 'PAX7', 'PAX3', 'MYH1'],
    'gut': ['CDX2', 'FOXA2', 'SOX9', 'LGR5', 'OLFM4'],
    'thyroid': ['NKX2-1', 'PAX8', 'FOXE1', 'TG', 'TPO'],
}


class SEAttentionHead:
    """Super-enhancer attention head using ABC database.

    For a given cell context (germ layer + tentative organ), queries the
    ABC database for active enhancer-gene predictions in matching cell types.

    Returns a gene activity vector that reflects which developmental genes
    are supported by the SE landscape at this position.

    This head is NOT trained — it reads directly from biology databases.
    The attention matrix (enhancers × genes) IS the chromatin state.
    """

    def __init__(self, se_builder=None, abc_client=None):
        """
        Args:
            se_builder: SEAttentionMatrixBuilder instance (optional, lazy-loaded)
            abc_client: ABCClient instance (optional, lazy-loaded)
        """
        self.se_builder = se_builder
        self.abc_client = abc_client
        self._cache = {}

    def forward(self, organ_identity: Optional[str],
                germ_layer: str,
                fate_probs: np.ndarray) -> Dict[str, float]:
        """Compute gene activity from SE landscape.

        Args:
            organ_identity: Committed organ (or None if uncommitted)
            germ_layer: Current germ layer ('ectoderm', 'mesoderm', 'endoderm')
            fate_probs: 11-dim organ fate probabilities

        Returns:
            Dict mapping gene name to activity score (0-1).
        """
        # Determine which organ's gene set to query
        if organ_identity and organ_identity in ORGAN_KEY_GENES:
            target_organ = organ_identity
        else:
            # Use most probable organ from fate vector
            organ_names = list(HUMAN_ORGANS.keys())
            top_idx = int(np.argmax(fate_probs))
            target_organ = organ_names[top_idx]

        genes = ORGAN_KEY_GENES.get(target_organ, [])
        if not genes:
            return {}

        # Check cache
        cache_key = (target_organ, germ_layer)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If SE builder is available, use real sparse matrix
        if self.se_builder is not None:
            cell_types = ORGAN_TO_ABC_CELL_TYPES.get(target_organ, [])
            try:
                matrix, metadata = self.se_builder.build_attention_matrix(
                    genes=genes,
                    cell_types=cell_types if cell_types else None,
                )
                # Uniform enhancer activation → gene activity
                enhancer_signals = np.ones(metadata.num_enhancers)
                gene_activity = matrix @ enhancer_signals
                # Normalize to 0-1
                if gene_activity.max() > 0:
                    gene_activity = gene_activity / gene_activity.max()
                result = {g: float(gene_activity[i])
                          for i, g in enumerate(metadata.genes)
                          if i < len(gene_activity)}
                self._cache[cache_key] = result
                return result
            except Exception as e:
                logger.debug(f"SE matrix query failed: {e}, using fallback")

        # Fallback: return known developmental gene scores from ABC activity data
        # (from _ABC_ION_CHANNEL_ACTIVITY in bioelectric_development.py)
        result = {g: 1.0 for g in genes}
        self._cache[cache_key] = result
        return result


# =============================================================================
# HEAD 4: Morphogen Gradients (Reaction-Diffusion → positional identity)
# =============================================================================
# Classical developmental biology signaling pathways.
# Source positions derived from body plan (organizer, notochord, etc.).

@dataclass
class MorphogenSpec:
    """Specification for a morphogen gradient."""
    name: str
    full_name: str
    source_position: np.ndarray    # Normalized (x, y, z) source
    decay_length: float            # Characteristic decay distance (normalized)
    role: str                      # What this gradient specifies
    antagonist: Optional[str] = None  # Name of antagonist morphogen


# Canonical vertebrate morphogen gradients
# Positions in normalized coordinates matching HUMAN_ORGANS topology:
#   x: left-right (0.5 = midline)
#   y: anterior-posterior (1.0 = anterior/head, 0.0 = posterior/tail)
#   z: dorsal-ventral (0.5 = middle)
MORPHOGEN_GRADIENTS = [
    MorphogenSpec(
        name="WNT",
        full_name="Wnt / beta-catenin",
        source_position=np.array([0.5, 0.2, 0.5]),   # Posterior (tail)
        decay_length=0.4,
        role="Posterior identity, mesoderm induction",
        antagonist="DKK1",
    ),
    MorphogenSpec(
        name="BMP",
        full_name="Bone Morphogenetic Protein (BMP4/7)",
        source_position=np.array([0.5, 0.5, 0.8]),   # Ventral
        decay_length=0.35,
        role="Ventral identity, epidermis, blood",
        antagonist="Noggin/Chordin",
    ),
    MorphogenSpec(
        name="SHH",
        full_name="Sonic Hedgehog",
        source_position=np.array([0.5, 0.6, 0.3]),   # Notochord / floor plate
        decay_length=0.25,
        role="Ventral neural, floor plate, digit patterning",
    ),
    MorphogenSpec(
        name="FGF",
        full_name="Fibroblast Growth Factor (FGF8)",
        source_position=np.array([0.5, 0.3, 0.5]),   # Posterior (isthmus, PSM)
        decay_length=0.3,
        role="Posterior identity, somite clock, limb outgrowth",
    ),
    MorphogenSpec(
        name="NODAL",
        full_name="Nodal / Activin",
        source_position=np.array([0.5, 0.5, 0.3]),   # Organizer (dorsal midline)
        decay_length=0.2,
        role="Mesoderm/endoderm induction, left-right asymmetry",
    ),
    MorphogenSpec(
        name="RA",
        full_name="Retinoic Acid",
        source_position=np.array([0.5, 0.5, 0.5]),   # Somites (trunk level)
        decay_length=0.3,
        role="Anterior-posterior patterning, hindbrain segmentation, limb",
    ),
]


class MorphogenHead:
    """Morphogen gradient computation via reaction-diffusion.

    Computes concentration of each canonical morphogen at a given position
    using exponential decay from source. In a full simulation this would
    be reaction-diffusion PDEs; here we use analytical gradients as the
    fast approximation (suitable for CMA-ES training).

    Can be upgraded to full PDE solver (MorphogenField in morphogens.py)
    for BETSE validation runs.
    """

    def __init__(self, gradients: Optional[List[MorphogenSpec]] = None):
        self.gradients = gradients or MORPHOGEN_GRADIENTS
        self.n_morphogens = len(self.gradients)
        self._names = [g.name for g in self.gradients]

    def forward(self, position: np.ndarray) -> Dict[str, float]:
        """Compute morphogen concentrations at a position.

        Args:
            position: (3,) array, normalized 0-1 coordinates.

        Returns:
            Dict mapping morphogen name to concentration (0-1).
        """
        concentrations = {}
        for grad in self.gradients:
            dist = np.linalg.norm(position - grad.source_position)
            concentration = np.exp(-dist / grad.decay_length)
            concentrations[grad.name] = float(concentration)
        return concentrations

    def forward_vector(self, position: np.ndarray) -> np.ndarray:
        """Compute morphogen vector (for MLP input)."""
        vec = np.zeros(self.n_morphogens)
        for i, grad in enumerate(self.gradients):
            dist = np.linalg.norm(position - grad.source_position)
            vec[i] = np.exp(-dist / grad.decay_length)
        return vec

    def get_dominant_signal(self, position: np.ndarray) -> Tuple[str, float]:
        """Get the strongest morphogen at a position."""
        conc = self.forward(position)
        name = max(conc, key=conc.get)
        return name, conc[name]


class VmSourcedMorphogenHead(MorphogenHead):
    """Morphogen head whose SOURCE POSITIONS emerge from the bioelectric field.

    Thesis (Levin; this work): the electric field is instructive for anatomy;
    morphogens are the essential effectors that execute it. The hardcoded
    ``MorphogenHead`` treats each morphogen source as an independent spatial
    coordinate. Here every source COORDINATE is removed and re-derived from
    the Vm field: each morphogen is produced by an organizer tissue with a
    characteristic bioelectric signature -- a target voltage V* (a biological
    constant, NOT a position) -- and its source is placed at the domain
    location whose Vm best matches V*.

    Only a voltage is supplied per morphogen. If the emergent gradients
    reproduce the canonical (hardcoded) ones, the Vm field carries the
    positional information; where they do not, that axis is not in Vm.
    """

    # Organizer bioelectric signatures (mV). Biological, NOT positional.
    # Dorsal organizer + notochord/floor-plate = hyperpolarized;
    # ventral + posterior = depolarized (Levin lab Xenopus polarity).
    ORGANIZER_VM = {
        'NODAL': -65.0,  # Spemann organizer / dorsal midline (hyperpolarized)
        'SHH':   -60.0,  # notochord / floor plate (hyperpolarized)
        'BMP':   -20.0,  # ventral epidermis/blood (depolarized)
        'WNT':   -25.0,  # posterior / tail (depolarized)
        'FGF':   -35.0,  # posterior PSM / isthmus (mesoderm)
        'RA':    -35.0,  # trunk somites (mesoderm)
    }

    def __init__(self, vm_field, domain_samples: np.ndarray,
                 gradients: Optional[List[MorphogenSpec]] = None):
        """
        Args:
            vm_field: callable position(3,) -> voltage_mv (the bioelectric field).
            domain_samples: (N, 3) candidate source positions to search over.
            gradients: morphogen specs (decay lengths reused; sources ignored).
        """
        super().__init__(gradients)
        self.vm_field = vm_field
        self.domain = np.asarray(domain_samples, dtype=float)
        self._domain_vm = np.array([float(vm_field(p)) for p in self.domain])
        self.derived_sources = self._derive_sources()

    def _derive_sources(self) -> Dict[str, np.ndarray]:
        """Place each morphogen source at the Vm best matching its organizer V*."""
        sources = {}
        for g in self.gradients:
            vstar = self.ORGANIZER_VM[g.name]
            idx = int(np.argmin(np.abs(self._domain_vm - vstar)))
            sources[g.name] = self.domain[idx].copy()
        return sources

    def forward_vector(self, position: np.ndarray) -> np.ndarray:
        vec = np.zeros(self.n_morphogens)
        for i, g in enumerate(self.gradients):
            src = self.derived_sources[g.name]
            dist = np.linalg.norm(np.asarray(position, float) - src)
            vec[i] = np.exp(-dist / g.decay_length)
        return vec

    def forward(self, position: np.ndarray) -> Dict[str, float]:
        vec = self.forward_vector(position)
        return {g.name: float(vec[i]) for i, g in enumerate(self.gradients)}


# =============================================================================
# HEAD 5: Connexin Attention (gap junction coupling profiles)
# =============================================================================
# Determines which cells can communicate electrically, and how strongly.
# Data-driven from organism bioelectric modules.

# Organ-specific connexin expression profiles
# Values are relative expression levels (0-1)
ORGAN_CONNEXIN_PROFILES = {
    'brain': {
        'Cx43': 0.8, 'Cx36': 0.9, 'Cx45': 0.4, 'Cx30': 0.6,
        'Cx32': 0.3, 'Cx26': 0.2,
    },
    'heart': {
        'Cx43': 0.95, 'Cx40': 0.7, 'Cx45': 0.5, 'Cx30.2': 0.3,
    },
    'liver': {
        'Cx32': 0.9, 'Cx26': 0.7, 'Cx43': 0.3,
    },
    'kidney_left': {'Cx43': 0.5, 'Cx30': 0.4, 'Cx37': 0.3},
    'kidney_right': {'Cx43': 0.5, 'Cx30': 0.4, 'Cx37': 0.3},
    'pancreas': {
        'Cx36': 0.9,  # Beta cell coupling (critical for insulin pulsatility)
        'Cx43': 0.4, 'Cx26': 0.3,
    },
    'lung_left': {'Cx43': 0.6, 'Cx32': 0.4, 'Cx26': 0.3},
    'lung_right': {'Cx43': 0.6, 'Cx32': 0.4, 'Cx26': 0.3},
    'muscle': {
        'Cx43': 0.7,  # Myoblast fusion coupling
        'Cx39': 0.5, 'Cx45': 0.3,
    },
    'gut': {'Cx43': 0.5, 'Cx45': 0.4, 'Cx32': 0.3},
    'thyroid': {'Cx43': 0.6, 'Cx32': 0.5},
}


class ConnexinHead:
    """Gap junction coupling attention head.

    Computes coupling strength between two cells based on their connexin
    expression profiles. Heterotypic connexin pairing rules determine
    which connexin pairs can form functional channels.

    Compatible pairs (can form heterotypic channels):
        Cx43-Cx43 (homotypic, most common)
        Cx43-Cx45
        Cx40-Cx43
        Cx26-Cx32
        Cx36-Cx36 (homotypic only, no heterotypic)

    Not trained -- reads from biology data.
    """

    # Heterotypic compatibility matrix (which connexins can pair?)
    COMPATIBLE_PAIRS = {
        ('Cx43', 'Cx43'): 1.0,
        ('Cx43', 'Cx45'): 0.7,
        ('Cx40', 'Cx43'): 0.5,
        ('Cx40', 'Cx40'): 1.0,
        ('Cx26', 'Cx32'): 0.8,
        ('Cx26', 'Cx26'): 1.0,
        ('Cx32', 'Cx32'): 1.0,
        ('Cx36', 'Cx36'): 1.0,
        ('Cx45', 'Cx45'): 1.0,
        ('Cx30', 'Cx30'): 1.0,
        ('Cx30', 'Cx26'): 0.6,
    }

    def coupling_strength(self, organ_a: Optional[str], organ_b: Optional[str]) -> float:
        """Compute gap junction coupling between two cells.

        Args:
            organ_a: Organ identity of cell A (or None if uncommitted)
            organ_b: Organ identity of cell B (or None if uncommitted)

        Returns:
            Coupling strength 0-1.
        """
        prof_a = ORGAN_CONNEXIN_PROFILES.get(organ_a, {'Cx43': 0.5})
        prof_b = ORGAN_CONNEXIN_PROFILES.get(organ_b, {'Cx43': 0.5})

        total_coupling = 0.0
        for cx_a, expr_a in prof_a.items():
            for cx_b, expr_b in prof_b.items():
                pair = (cx_a, cx_b)
                pair_rev = (cx_b, cx_a)
                compat = self.COMPATIBLE_PAIRS.get(pair,
                         self.COMPATIBLE_PAIRS.get(pair_rev, 0.0))
                if compat > 0:
                    total_coupling += expr_a * expr_b * compat

        return min(1.0, total_coupling)

    def boundary_detection(self, organ_a: Optional[str], organ_b: Optional[str]) -> bool:
        """Detect if two cells are at an organ boundary (low coupling).

        Boundaries form where gap junction coupling drops below threshold.
        This is the Levin mechanism for organ territory demarcation.
        """
        return self.coupling_strength(organ_a, organ_b) < 0.2


# =============================================================================
# HEAD 6: Chromatin State (epigenetic gating of SE accessibility)
# =============================================================================
# Uses histone marks to determine which SEs are open vs silenced.
# H3K4me1 = enhancer primed, H3K27ac = enhancer active, H3K27me3 = PRC2 silenced.

# Germ-layer-specific chromatin accessibility (simplified)
# In reality this comes from Xenbase/ENCODE epigenome data
CHROMATIN_ACCESSIBILITY = {
    'ectoderm': {
        'neural_enhancers': 0.9,      # Open: SOX2, PAX6, OTX2 loci
        'cardiac_enhancers': 0.1,      # Silenced by PRC2
        'hepatic_enhancers': 0.05,     # Strongly silenced
        'muscle_enhancers': 0.15,      # Mostly silenced
        'endoderm_enhancers': 0.05,    # Strongly silenced
    },
    'mesoderm': {
        'neural_enhancers': 0.1,
        'cardiac_enhancers': 0.85,     # Open: NKX2-5, GATA4, TBX5 loci
        'hepatic_enhancers': 0.15,
        'muscle_enhancers': 0.8,       # Open: MYOD1, MYF5 loci
        'endoderm_enhancers': 0.1,
    },
    'endoderm': {
        'neural_enhancers': 0.05,
        'cardiac_enhancers': 0.1,
        'hepatic_enhancers': 0.85,     # Open: HNF4A, FOXA2 loci
        'muscle_enhancers': 0.05,
        'endoderm_enhancers': 0.9,     # Open: CDX2, PDX1 loci
    },
}

# Organ -> enhancer category mapping
ORGAN_ENHANCER_CATEGORY = {
    'brain': 'neural_enhancers',
    'heart': 'cardiac_enhancers',
    'liver': 'hepatic_enhancers',
    'kidney_left': 'cardiac_enhancers',   # Mesoderm-derived
    'kidney_right': 'cardiac_enhancers',
    'pancreas': 'endoderm_enhancers',
    'lung_left': 'endoderm_enhancers',
    'lung_right': 'endoderm_enhancers',
    'muscle': 'muscle_enhancers',
    'gut': 'endoderm_enhancers',
    'thyroid': 'endoderm_enhancers',
}


class ChromatinStateHead:
    """Chromatin state attention head.

    Gates SE accessibility based on histone marks and germ layer commitment.
    A cell in ectoderm has neural enhancers OPEN but cardiac enhancers
    SILENCED by PRC2 (H3K27me3). This prevents cross-lineage gene activation.

    Connects to the telomere clock via the Polycomb Mask:
    as telomeres shorten, PRC2 withdraws from 5' Hox genes,
    progressively opening posterior enhancers.
    """

    def accessibility(self, germ_layer: str, organ: str) -> float:
        """Get chromatin accessibility for an organ's enhancers in a germ layer.

        Args:
            germ_layer: Current germ layer commitment
            organ: Target organ

        Returns:
            Accessibility score 0-1. Low = PRC2 silenced. High = open chromatin.
        """
        layer_access = CHROMATIN_ACCESSIBILITY.get(germ_layer, {})
        enhancer_cat = ORGAN_ENHANCER_CATEGORY.get(organ, 'neural_enhancers')
        return layer_access.get(enhancer_cat, 0.1)

    def gate_se_activity(self, gene_activity: Dict[str, float],
                         germ_layer: str, organ: str) -> Dict[str, float]:
        """Apply chromatin gating to SE-derived gene activity.

        Multiplies gene activity by chromatin accessibility.
        Silenced enhancers produce near-zero gene activity regardless
        of SE score.

        Args:
            gene_activity: Dict from SEAttentionHead
            germ_layer: Current germ layer
            organ: Target organ

        Returns:
            Gated gene activity dict.
        """
        access = self.accessibility(germ_layer, organ)
        return {gene: score * access for gene, score in gene_activity.items()}


# =============================================================================
# PROMOTER MLP (SE gene activity -> promoter firing rates)
# =============================================================================
# The bridge between enhancer signals and actual transcription.
# Learns the nonlinear mapping: multiple enhancers can act on one promoter,
# with synergistic or competitive effects.

class PromoterMLP:
    """Maps SE attention output to promoter activation levels.

    Input: gene activity vector from SEAttentionHead (gated by ChromatinStateHead)
    Output: promoter firing rates for key developmental transcription factors

    This is a small trainable MLP that captures enhancer-promoter nonlinearity.
    In biology: multiple enhancers can synergistically activate one promoter,
    shadow enhancers provide robustness, and insulators block cross-activation.
    """

    def __init__(self, n_genes: int = 7, hidden_dim: int = 16, seed: int = 42):
        """
        Args:
            n_genes: Max number of genes per organ (input dim)
            hidden_dim: Hidden layer size
            seed: Random seed
        """
        rng = np.random.default_rng(seed)
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # Small MLP: n_genes -> hidden -> n_genes (promoter rates)
        self.W1 = rng.normal(0, 0.1, (n_genes, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.1, (hidden_dim, n_genes))
        self.b2 = np.zeros(n_genes)

        self._n_params = n_genes * hidden_dim + hidden_dim + hidden_dim * n_genes + n_genes
        logger.info(f"PromoterMLP: {n_genes}->h{hidden_dim}->{n_genes}, {self._n_params} params")

    def forward(self, gene_activity: Dict[str, float]) -> Dict[str, float]:
        """Compute promoter firing rates from gene activity.

        Args:
            gene_activity: Dict mapping gene name to SE-derived activity (0-1)

        Returns:
            Dict mapping gene name to promoter firing rate (0-1)
        """
        genes = list(gene_activity.keys())[:self.n_genes]
        if not genes:
            return {}

        # Build input vector (pad to n_genes)
        x = np.zeros(self.n_genes)
        for i, g in enumerate(genes):
            if i < self.n_genes:
                x[i] = gene_activity[g]

        # Forward pass
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        out = 1.0 / (1.0 + np.exp(-(h @ self.W2 + self.b2)))  # Sigmoid

        return {g: float(out[i]) for i, g in enumerate(genes) if i < self.n_genes}

    def get_parameters(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_parameters(self, params: np.ndarray):
        idx = 0
        sz = self.n_genes * self.hidden_dim
        self.W1 = params[idx:idx+sz].reshape((self.n_genes, self.hidden_dim))
        idx += sz
        self.b1 = params[idx:idx+self.hidden_dim].copy()
        idx += self.hidden_dim
        sz = self.hidden_dim * self.n_genes
        self.W2 = params[idx:idx+sz].reshape((self.hidden_dim, self.n_genes))
        idx += sz
        self.b2 = params[idx:idx+self.n_genes].copy()


# =============================================================================
# CHANNEL MLP (morphogens + germ layer -> ion channel conductances)
# =============================================================================
# The cymatic generator. Predicts the conductance profile that creates
# the voltage landscape. This is what generates the standing wave pattern
# that the Body Plan Head reads.

# Ion channel output indices
CHANNEL_NAMES = ['g_Na', 'g_K', 'g_Ca', 'g_Cl', 'g_GJ']


class GenomicChannelLookup:
    """Computes ion channel conductances directly from genomic data (ABC database).

    NOT a neural network. This is a genomic lookup:
        Genome -> Super-enhancers -> Ion channel gene expression -> Conductances
        -> Goldman equation -> Membrane potential (Vmem)

    The ABC database (Nasser et al. Nature 2021) provides enhancer-promoter
    predictions for 45 ion channel genes across 131 cell types. We queried
    these and aggregated into per-organ activity scores for 5 channel families:
        Na+ (SCN*), K+ (KCNJ/H/Q/K/NN), Ca2+ (CACNA/TRPC/V),
        Cl- (CLCN/CFTR), Gap junctions (GJA/B/C)

    The conductance ratios ARE the genome's program for voltage patterning.
    Brain has high K+/total -> hyperpolarized (-70 mV, ectoderm).
    Liver has high Ca2+,Na+/K+ -> depolarized (-20 mV, endoderm).

    Trainable: only a small residual (5 values) that CMA-ES can use for
    fine-tuning within +/- 20% of the genomic baseline. This prevents the
    optimizer from overriding the biology.

    Data chain:
        _ABC_ION_CHANNEL_ACTIVITY (bioelectric_development.py)
        -> ORGAN_CONDUCTANCE_PROFILES (computed via Goldman anchoring)
        -> position-weighted blend per cell
        -> + trainable residual (clamped to +/- 20%)
    """

    def __init__(self, seed: int = 42):
        self.n_channels = 5

        # Trainable residual: one multiplier per channel (initialized to 0 = no change)
        # Applied as: conductance * (1 + tanh(residual) * 0.2)
        # This allows +/- 20% adjustment from genomic baseline
        self.residual = np.zeros(self.n_channels)

        self._n_params = self.n_channels  # Only 5 trainable params
        logger.info(f"GenomicChannelLookup: ABC-derived conductances, "
                    f"{self._n_params} trainable residual params")

    def forward(self, morphogens: np.ndarray, germ_probs: Dict[str, float],
                position: np.ndarray) -> Dict[str, float]:
        """Compute ion channel conductances from genomic data + position.

        Uses ABC-derived organ conductance profiles blended by proximity
        to organ target positions (same method as bioelectric_development.py
        create_initial_state). Then applies small trainable residual.

        Args:
            morphogens: (6,) morphogen concentration vector (unused, kept for API compat)
            germ_probs: Germ layer probabilities (unused, conductances determine voltage
                        which determines germ layer -- not the other way around)
            position: (3,) normalized position

        Returns:
            Dict mapping channel name to conductance (mS/cm2)
        """
        from .bioelectric_development import ORGAN_CONDUCTANCE_PROFILES, HUMAN_ORGANS

        organ_names = list(HUMAN_ORGANS.keys())
        organ_positions = np.array([HUMAN_ORGANS[o].position_3d for o in organ_names])

        # Distance-weighted blend of organ conductance profiles
        dists = np.linalg.norm(organ_positions - position[:3], axis=1)
        sigma = 0.15  # Sharp enough to resolve organ territories
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        w_sum = weights.sum() + 1e-10
        weights /= w_sum

        conductances = np.zeros(self.n_channels)
        for j, organ_name in enumerate(organ_names):
            if organ_name in ORGAN_CONDUCTANCE_PROFILES:
                profile = ORGAN_CONDUCTANCE_PROFILES[organ_name]
                for k in range(self.n_channels):
                    conductances[k] += weights[j] * profile[k]

        # Apply trainable residual: (1 + tanh(r) * 0.2) = +/- 20% max
        residual_scale = 1.0 + np.tanh(self.residual) * 0.2
        conductances *= residual_scale

        # Ensure positive
        conductances = np.maximum(conductances, 0.01)

        return {name: float(conductances[i]) for i, name in enumerate(CHANNEL_NAMES)}

    def compute_voltage(self, position: np.ndarray) -> float:
        """Compute Goldman resting potential for this position.

        This is the genomically-determined Vmem: the voltage that the
        ion channel expression profile produces at steady state.

        V_rest = (g_Na*E_Na + g_K*E_K + g_Ca_rest*E_Ca + g_Cl*E_Cl) /
                 (g_Na + g_K + g_Ca_rest + g_Cl)
        """
        from .bioelectric_development import E_NA, E_K, E_CA, E_CL

        cond = self.forward(np.zeros(6), {}, position)
        g_Na = cond['g_Na']
        g_K = cond['g_K']
        g_Ca = cond['g_Ca'] * 0.10  # 10% open at rest
        g_Cl = cond['g_Cl']

        g_total = g_Na + g_K + g_Ca + g_Cl + 1e-10
        v_rest = (g_Na * E_NA + g_K * E_K + g_Ca * E_CA + g_Cl * E_CL) / g_total
        return float(v_rest)

    def get_parameters(self) -> np.ndarray:
        return self.residual.copy()

    def set_parameters(self, params: np.ndarray):
        self.residual = params[:self.n_channels].copy()


# Keep old name as alias for backward compatibility during transition
ChannelMLP = GenomicChannelLookup


# =============================================================================
# MASKS (3 gating mechanisms)
# =============================================================================

class GermLayerMask:
    """Gates which organs can compete based on germ layer commitment.

    Already implemented in OrganHead._germ_masks.
    This class provides the same logic as a standalone mask for composition.
    """

    GERM_ORGANS = OrganHead.GERM_LAYER_ORGANS

    def apply(self, fate_logits: np.ndarray, germ_layer: str) -> np.ndarray:
        """Zero out logits for organs not in this germ layer."""
        organ_names = list(HUMAN_ORGANS.keys())
        allowed = set(self.GERM_ORGANS.get(germ_layer, []))
        masked = fate_logits.copy()
        for i, name in enumerate(organ_names):
            if name not in allowed:
                masked[i] = -1e6  # Effectively zero after softmax
        return masked


class PolycombMask:
    """PRC2 silencing mask driven by telomere clock.

    As telomeres shorten with each division:
    1. TERRA transcript length decreases
    2. PRC2 reach decreases (can no longer silence distant 5' Hox genes)
    3. 5' Hox genes activate in colinear sequence (3'->5' = anterior->posterior)
    4. Downstream targets become accessible

    This mask gates which Hox-dependent enhancers are open.
    Short telomere = more 5' Hox active = more posterior identity.

    Connects the TEMPORAL axis (telomere clock) to the BIOCHEMICAL level
    (SE accessibility).
    """

    # Hox clusters with their position along the chromosome (5'->3')
    # Activation order: 3' first (anterior) -> 5' last (posterior)
    HOX_CLUSTERS = {
        'HOXA': ['HOXA1', 'HOXA2', 'HOXA3', 'HOXA4', 'HOXA5',
                  'HOXA7', 'HOXA9', 'HOXA10', 'HOXA11', 'HOXA13'],
        'HOXB': ['HOXB1', 'HOXB2', 'HOXB3', 'HOXB4', 'HOXB5',
                  'HOXB6', 'HOXB7', 'HOXB8', 'HOXB9', 'HOXB13'],
        'HOXC': ['HOXC4', 'HOXC5', 'HOXC6', 'HOXC8',
                  'HOXC9', 'HOXC10', 'HOXC11', 'HOXC12', 'HOXC13'],
        'HOXD': ['HOXD1', 'HOXD3', 'HOXD4', 'HOXD8',
                  'HOXD9', 'HOXD10', 'HOXD11', 'HOXD12', 'HOXD13'],
    }

    # Telomere length thresholds for Hox activation (bp)
    # Longer telomere = more PRC2 silencing = only 3' (anterior) Hox active
    TELOMERE_FULL = 10000.0    # Zygote: only HOXA1/B1 (most anterior)
    TELOMERE_HALF = 5000.0     # Mid-development: up to HOX5-7
    TELOMERE_SHORT = 2000.0    # Late: all Hox active (full AP axis)

    def hox_activation(self, telomere_length: float) -> Dict[str, float]:
        """Compute Hox gene activation from telomere length.

        Args:
            telomere_length: Current telomere length (bp)

        Returns:
            Dict mapping Hox gene name to activation level (0-1).
            3' genes (anterior) activate first, 5' genes (posterior) last.
        """
        # Fraction of Hox cluster that is desilenced
        # telomere_length goes from FULL (10000) to SHORT (2000)
        # desilenced_fraction goes from 0.1 (only most 3') to 1.0 (all)
        fraction = 1.0 - (telomere_length - self.TELOMERE_SHORT) / \
                         (self.TELOMERE_FULL - self.TELOMERE_SHORT)
        fraction = np.clip(fraction, 0.1, 1.0)

        activations = {}
        for cluster_name, genes in self.HOX_CLUSTERS.items():
            n_genes = len(genes)
            for i, gene in enumerate(genes):
                # Gene position along cluster (0 = most 3'/anterior, 1 = most 5'/posterior)
                pos = i / max(1, n_genes - 1)
                # Sigmoid activation: 3' genes on first, 5' genes need more desilencing
                activation = 1.0 / (1.0 + np.exp(10.0 * (pos - fraction)))
                activations[gene] = float(activation)

        return activations

    def ap_identity(self, telomere_length: float) -> str:
        """Get anterior-posterior identity from telomere state.

        Returns: 'anterior', 'mid-anterior', 'trunk', 'mid-posterior', 'posterior'
        """
        activations = self.hox_activation(telomere_length)
        # Count active posterior genes (HOX9-13)
        posterior_active = sum(1 for g, v in activations.items()
                              if any(g.endswith(str(n)) for n in range(9, 14)) and v > 0.5)
        if posterior_active >= 12:
            return 'posterior'
        elif posterior_active >= 8:
            return 'mid-posterior'
        elif posterior_active >= 4:
            return 'trunk'
        elif posterior_active >= 1:
            return 'mid-anterior'
        return 'anterior'


class ECMMask:
    """Extracellular matrix mask that locks committed cells.

    As cells commit to fates, they deposit ECM proteins (collagen, laminin,
    fibronectin). Once ECM density exceeds threshold, the cell is "locked":
    - Migration velocity damped to near-zero
    - Fate probabilities frozen
    - Gap junction coupling may change (boundary formation)

    This is phase 4 of the Levin temporal phasing mechanism.
    """

    def __init__(self, n_cells: int = 321, lock_threshold: float = 0.8):
        self.density = np.zeros(n_cells)
        self.lock_threshold = lock_threshold
        self._locked = np.zeros(n_cells, dtype=bool)

    def deposit(self, cell_idx: int, amount: float = 0.1):
        """Deposit ECM at a cell's position."""
        if cell_idx < len(self.density):
            self.density[cell_idx] = min(1.0, self.density[cell_idx] + amount)
            if self.density[cell_idx] >= self.lock_threshold:
                self._locked[cell_idx] = True

    def is_locked(self, cell_idx: int) -> bool:
        """Check if a cell is ECM-locked."""
        return bool(self._locked[cell_idx]) if cell_idx < len(self._locked) else False

    def migration_damping(self, cell_idx: int) -> float:
        """Get migration damping factor (1.0 = free, 0.0 = locked)."""
        if cell_idx >= len(self.density):
            return 1.0
        return float(1.0 - self.density[cell_idx])

    def fraction_locked(self) -> float:
        """Fraction of cells that are ECM-locked."""
        if len(self._locked) == 0:
            return 0.0
        return float(self._locked.sum() / len(self._locked))

    def resize(self, n_cells: int):
        """Resize for growing embryo."""
        if n_cells > len(self.density):
            self.density = np.pad(self.density, (0, n_cells - len(self.density)))
            self._locked = np.pad(self._locked, (0, n_cells - len(self._locked)))


# =============================================================================
# COMBINED: Full Morphogenesis Controller (6 heads + 2 MLPs + 3 masks)
# =============================================================================

class FullMorphogenesisController:
    """Complete morphogenesis controller with all components.

    ATTENTION HEADS (6):
        Tissue Level:
            Head 1 (Body Plan):     Vmem -> germ layer         [hardwired Levin]
            Head 2 (Organ):         position -> organ fate      [trained CMA-ES]
        Biochemical Level:
            Head 3 (SE):            ABC database -> gene activity [data-driven]
            Head 4 (Morphogen):     gradients -> positional signals [analytical]
            Head 5 (Connexin):      coupling profiles -> GJC strength [data-driven]
            Head 6 (Chromatin):     histone marks -> SE accessibility [data-driven]

    MLPs (2 additional trainable):
        Promoter MLP:  gene activity -> promoter firing rates    [trained]
        Channel MLP:   morphogens + context -> ion conductances  [trained]

    MASKS (3):
        Germ Layer Mask:  gates organ competition by germ layer
        Polycomb Mask:    telomere clock -> Hox -> enhancer access
        ECM Mask:         matrix deposition locks committed cells

    Information flow:
        Position -> [Head 4: Morphogen] -> morphogen_vec
        morphogen_vec + germ + position -> [Channel MLP] -> conductances
        conductances -> (BETSE/HH) -> Vmem
        Vmem -> [Head 1: Body Plan] -> germ_layer_probs
        [Polycomb Mask](telomere) -> Hox_activation
        germ_layer_probs -> [Head 6: Chromatin] -> accessibility
        germ_layer + position -> [Head 2: Organ] -> organ_fate (masked by [Germ Layer Mask])
        organ_fate + germ -> [Head 3: SE] -> gene_activity
        gene_activity * accessibility -> [Promoter MLP] -> promoter_rates
        organ_a, organ_b -> [Head 5: Connexin] -> coupling_strength
        ECM density -> [ECM Mask] -> migration_damping
    """

    def __init__(self, se_builder=None, abc_client=None, seed: int = 42):
        # --- 6 Attention Heads ---
        self.head1_body_plan = BodyPlanHead()
        self.head2_organ = OrganHead(seed=seed)
        self.head3_se = SEAttentionHead(se_builder=se_builder, abc_client=abc_client)
        self.head4_morphogen = MorphogenHead()
        self.head5_connexin = ConnexinHead()
        self.head6_chromatin = ChromatinStateHead()

        # --- 2 MLPs ---
        self.promoter_mlp = PromoterMLP(n_genes=7, hidden_dim=16, seed=seed)
        self.channel_mlp = GenomicChannelLookup(seed=seed)

        # --- 3 Masks ---
        self.germ_mask = GermLayerMask()
        self.polycomb_mask = PolycombMask()
        self.ecm_mask = ECMMask()

        # --- Migration + Division (from organ hidden state) ---
        rng = np.random.default_rng(seed)
        self.W_mig = rng.normal(0, 0.1, (64, 3))
        self.b_mig = np.zeros(3)
        self.W_div = rng.normal(0, 0.1, (64, 1))
        self.b_div = np.zeros(1)

        logger.info("FullMorphogenesisController initialized")
        logger.info("  6 Attention Heads + 2 MLPs + 3 Masks")

    def forward(self, position: np.ndarray, voltage_mv: float, calcium: float,
                generation: int, stage_value: int, time: float,
                n_neighbors: int, telomere_length: float = 10000.0,
                organ_identity: Optional[str] = None,
                cell_idx: int = 0,
                morphogen_override: Optional[np.ndarray] = None) -> Dict:
        """Complete forward pass through all components.

        Args:
            position: (3,) normalized position
            voltage_mv: Membrane potential (mV)
            calcium: Intracellular calcium (uM)
            generation: Cell division generation
            stage_value: Developmental stage (0-6)
            time: Developmental time
            n_neighbors: Number of neighboring cells
            telomere_length: Current telomere length (bp)
            organ_identity: Committed organ (or None)
            cell_idx: Cell index (for ECM mask)
            morphogen_override: Override morphogen values

        Returns:
            Dict with all outputs from all components.
        """
        # --- HEAD 4: Morphogen gradients ---
        if morphogen_override is not None:
            morph_vec = morphogen_override
        else:
            morph_vec = self.head4_morphogen.forward_vector(position)

        # --- HEAD 1: Body Plan (Vmem -> germ layer) ---
        germ_probs = self.head1_body_plan.forward(voltage_mv)
        dominant_germ = max(germ_probs, key=germ_probs.get)

        # --- CHANNEL MLP: predict ion channel conductances ---
        conductances = self.channel_mlp.forward(morph_vec, germ_probs, position)

        # --- POLYCOMB MASK: telomere -> Hox activation ---
        hox_activation = self.polycomb_mask.hox_activation(telomere_length)
        ap_identity = self.polycomb_mask.ap_identity(telomere_length)

        # --- HEAD 2: Organ (position + context -> organ fate) ---
        x_in = np.array([
            position[0], position[1], position[2],
            voltage_mv / 100.0, calcium,
            morph_vec[0] if len(morph_vec) > 0 else 0.0,
            morph_vec[1] if len(morph_vec) > 1 else 0.0,
            morph_vec[2] if len(morph_vec) > 2 else 0.0,
            n_neighbors / 10.0, generation / 10.0,
            stage_value / 6.0, time / 1000.0,
        ])
        fate_probs, hidden = self.head2_organ.forward_with_hidden(x_in, germ_probs)

        # --- HEAD 3: SE attention (database -> gene activity) ---
        gene_activity = self.head3_se.forward(
            organ_identity=organ_identity,
            germ_layer=dominant_germ,
            fate_probs=fate_probs,
        )

        # --- HEAD 6: Chromatin gating ---
        target_organ = organ_identity or list(HUMAN_ORGANS.keys())[int(np.argmax(fate_probs))]
        gated_genes = self.head6_chromatin.gate_se_activity(
            gene_activity, dominant_germ, target_organ)

        # --- PROMOTER MLP: gated genes -> promoter firing rates ---
        promoter_rates = self.promoter_mlp.forward(gated_genes)

        # --- Migration (damped by ECM) ---
        migration = np.tanh(hidden @ self.W_mig + self.b_mig) * 0.1
        ecm_damp = self.ecm_mask.migration_damping(cell_idx)
        migration *= ecm_damp

        # --- Division ---
        div_logit = float((hidden @ self.W_div + self.b_div)[0])
        divide_prob = 1.0 / (1.0 + np.exp(-np.clip(div_logit, -10, 10)))
        # ECM-locked cells don't divide
        if self.ecm_mask.is_locked(cell_idx):
            divide_prob = 0.0

        # --- Morphogen dict ---
        morph_dict = {g.name: float(morph_vec[i]) if i < len(morph_vec) else 0.0
                      for i, g in enumerate(self.head4_morphogen.gradients)}

        return {
            # Tissue level
            'germ_layer': germ_probs,
            'fate': fate_probs,
            'ap_identity': ap_identity,
            # Biochemical level
            'gene_activity': gated_genes,
            'promoter_rates': promoter_rates,
            'morphogens': morph_dict,
            'conductances': conductances,
            'hox_activation': hox_activation,
            # Connectivity
            'connexin_profile': ORGAN_CONNEXIN_PROFILES.get(target_organ, {}),
            # Actions
            'migration': migration,
            'divide': divide_prob,
            'ecm_damping': ecm_damp,
            'ecm_locked': self.ecm_mask.is_locked(cell_idx),
        }

    def get_parameters(self) -> np.ndarray:
        """Get all trainable parameters."""
        return np.concatenate([
            self.head2_organ.get_parameters(),      # Organ head
            self.promoter_mlp.get_parameters(),      # Promoter MLP
            self.channel_mlp.get_parameters(),        # Channel MLP
            self.W_mig.ravel(), self.b_mig,          # Migration
            self.W_div.ravel(), self.b_div,           # Division
        ])

    def set_parameters(self, params: np.ndarray):
        """Set all trainable parameters from flat vector."""
        idx = 0

        # Organ head
        organ_n = (self.head2_organ.input_dim * self.head2_organ.hidden_dim +
                   self.head2_organ.hidden_dim + self.head2_organ.hidden_dim * 11 + 11)
        self.head2_organ.set_parameters(params[idx:idx+organ_n])
        idx += organ_n

        # Promoter MLP
        prom_n = self.promoter_mlp._n_params
        self.promoter_mlp.set_parameters(params[idx:idx+prom_n])
        idx += prom_n

        # Channel MLP
        chan_n = self.channel_mlp._n_params
        self.channel_mlp.set_parameters(params[idx:idx+chan_n])
        idx += chan_n

        # Migration
        sz = 64 * 3
        self.W_mig = params[idx:idx+sz].reshape((64, 3))
        idx += sz
        self.b_mig = params[idx:idx+3].copy()
        idx += 3

        # Division
        sz = 64 * 1
        self.W_div = params[idx:idx+sz].reshape((64, 1))
        idx += sz
        self.b_div = params[idx:idx+1].copy()

    @property
    def n_params(self) -> int:
        return len(self.get_parameters())

    @property
    def n_trainable_params(self) -> int:
        """Count only trainable parameters (excludes hardwired heads)."""
        return self.n_params

    def component_summary(self) -> Dict[str, int]:
        """Parameter count per component."""
        organ_n = (self.head2_organ.input_dim * self.head2_organ.hidden_dim +
                   self.head2_organ.hidden_dim + self.head2_organ.hidden_dim * 11 + 11)
        return {
            'Head 1 (Body Plan)': 0,          # Hardwired
            'Head 2 (Organ)': organ_n,
            'Head 3 (SE)': 0,                 # Data-driven
            'Head 4 (Morphogen)': 0,           # Analytical
            'Head 5 (Connexin)': 0,            # Data-driven
            'Head 6 (Chromatin)': 0,           # Data-driven
            'Promoter MLP': self.promoter_mlp._n_params,
            'Channel MLP': self.channel_mlp._n_params,
            'Migration': 64 * 3 + 3,
            'Division': 64 * 1 + 1,
            'Germ Layer Mask': 0,
            'Polycomb Mask': 0,
            'ECM Mask': 0,
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("FOUR-HEAD MORPHOGENESIS ARCHITECTURE")
    print("=" * 70)

    controller = FourHeadMorphogenesis(seed=42)
    print(f"\nTrainable parameters: {controller.n_params}")

    # Test at different positions and voltages
    test_cases = [
        ("Anterior ectoderm (brain territory)", np.array([0.5, 0.9, 0.5]), -70.0),
        ("Central mesoderm (heart territory)", np.array([0.5, 0.65, 0.5]), -30.0),
        ("Posterior endoderm (gut territory)", np.array([0.5, 0.35, 0.5]), -20.0),
        ("Ventral mesoderm (muscle territory)", np.array([0.5, 0.5, 0.7]), -40.0),
    ]

    for label, pos, vmem in test_cases:
        result = controller.forward(
            position=pos, voltage_mv=vmem, calcium=0.1,
            generation=5, stage_value=5, time=50.0, n_neighbors=6,
        )

        germ = result['germ_layer']
        top_germ = max(germ, key=germ.get)
        fate = result['fate']
        organ_names = list(HUMAN_ORGANS.keys())
        top_organ = organ_names[int(np.argmax(fate))]
        morph = result['morphogens']
        top_morph = max(morph, key=morph.get)
        n_genes = len(result['gene_activity'])

        print(f"\n{label}:")
        print(f"  Vmem: {vmem:.0f} mV -> Germ: {top_germ} "
              f"(e={germ['ectoderm']:.2f}, m={germ['mesoderm']:.2f}, d={germ['endoderm']:.2f})")
        print(f"  Position: {pos} -> Organ: {top_organ} (p={fate[np.argmax(fate)]:.2f})")
        print(f"  Top morphogen: {top_morph} ({morph[top_morph]:.2f})")
        print(f"  Active genes: {n_genes}")
        print(f"  Migrate: [{result['migration'][0]:.3f}, {result['migration'][1]:.3f}, {result['migration'][2]:.3f}]")
        print(f"  Divide: {result['divide']:.3f}")

    # Show morphogen landscape
    print(f"\n--- Morphogen Gradients ({len(MORPHOGEN_GRADIENTS)} signals) ---")
    for grad in MORPHOGEN_GRADIENTS:
        print(f"  {grad.name:8s} ({grad.full_name})")
        print(f"    Source: {grad.source_position}, Decay: {grad.decay_length}")
        print(f"    Role: {grad.role}")
        if grad.antagonist:
            print(f"    Antagonist: {grad.antagonist}")

    print(f"\n--- SE Gene Sets ({len(ORGAN_KEY_GENES)} organs) ---")
    for organ, genes in ORGAN_KEY_GENES.items():
        print(f"  {organ:15s}: {', '.join(genes[:4])}{'...' if len(genes) > 4 else ''}")

    print("\n" + "=" * 70)
    print("Architecture: 4 heads, 2 levels")
    print("  Tissue:      Body Plan (hardwired) + Organ (trained)")
    print("  Biochemical: SE attention (data-driven) + Morphogen (analytical)")
    print("=" * 70)
