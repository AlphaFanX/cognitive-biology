"""
The menagerie genome: a low-rank set of knobs on the basic vertebrate.

Each CONTINUOUS knob is a scalar LoRA coefficient on the conserved tetrapod Bauplan,
annotated with the real developmental mechanism and (where known) the gene it stands for.
The DISCRETE genes are qualitative body-plan features gained or lost as macro-evolutionary
events (a horn, tusks, a trunk, a posture, a coat type) -- these are not continuously
interpolated but switched.

The base genome (all continuous knobs = 1.0, discrete = the generic tetrapod state) is the
"basic vertebrate" from which every species is evolved.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace
import numpy as np

# ----------------------------------------------------------------------------------
# CONTINUOUS knobs: name -> (min, max, mechanism, gene/pathway)
# All are multiplicative factors on the base Bauplan (base value = 1.0).
# ----------------------------------------------------------------------------------
KNOBS = {
    "body_size":     (0.4, 3.0, "overall allometric scale",                 "GH/IGF1, HMGA2"),
    "cervical_elong":(0.6, 6.0, "per-cervical vertebra length (count FROZEN at 7)", "FGFRL1, Hox(C)"),
    "trunk_len":     (0.6, 2.0, "thoraco-lumbar segment length",            "Hox(T/L), Gdf11"),
    "trunk_girth":   (0.5, 2.4, "body-wall radius over the trunk",          "myostatin/allometry"),
    "tail_len":      (0.1, 3.0, "per-caudal vertebra length",               "Hox(caudal), Wnt3a"),
    "tail_count":    (0.3, 3.0, "number of caudal vertebrae (axis clock)",  "Lfng/Hes7 segmentation"),
    "limb_len":      (0.5, 2.2, "limb-segment length (cursorial<->long)",   "Shh/Fgf8 AER, Hox(A/D)"),
    "limb_gracility":(0.4, 2.6, "limb thickness (slender<->columnar/graviportal)", "Runx2, Bmp"),
    "skull_size":    (0.6, 2.2, "cranial vault + jaw size",                 "cranial neural crest"),
    "snout_len":     (0.5, 3.2, "rostrum / snout elongation",               "Bmp4, Wnt (FaceBase axis)"),
    "neck_raise":    (0.0, 1.0, "habitual neck carriage angle (0 flat->1 vertical)", "postural (not developmental)"),
    "horn_size":     (0.0, 2.5, "cranial appendage size",                   "keratin/os cornu program"),
    "tusk_len":      (0.0, 3.0, "elongated incisor/canine length",          "tooth EDA/BMP, ever-growth"),
    "proboscis_len": (0.0, 3.5, "muscular-hydrostat nose+lip (trunk)",      "proboscis facial program"),
    "coat_wavelength":(0.3, 3.0,"reaction-diffusion pattern wavelength",    "RD head (Turing)"),
    "convergent_ext": (0.5, 1.6,"axial convergent extension: body narrowing (fish) vs widening (tetrapod); high narrows the body -> no left-right eigenmode -> limbless; low keeps width -> limbs", "Wnt-PCP: Vangl2, Wnt5a/Wnt11"),
}
CONT = list(KNOBS.keys())

# DISCRETE genes: name -> allowed states (first = generic tetrapod default)
DISCRETE = {
    "horn_mode": ["none", "single_median", "paired"],   # rhino median (LR-node) / buffalo paired (lat. inhib.)
    "coat_type": ["plain", "spot", "rosette", "patch", "stripe", "armor"],
    "posture":   ["erect", "sprawling", "biped"],        # quadruped / crocodilian sprawl / upright biped
    "dentition": ["normal", "tusks"],
    "nose":      ["normal", "trunk"],
    "sex":       ["none", "male", "female"],             # sexual dimorphism (girdle widths, build)
    "body_plan": ["tetrapod", "finned"],                 # deep homology: the SAME paired-appendage module -> legs or fins
}

# The conserved axial formula -- FROZEN counts (mammalian). Elongation is per-segment scale.
CERVICAL_COUNT = 7          # <-- the giraffe invariant: same 7 as every mammal
THORACIC_COUNT = 13
LUMBAR_COUNT   = 6
SACRAL_COUNT   = 4
BASE_CAUDAL    = 12         # scaled by the tail_count knob


@dataclass
class Genome:
    # continuous knobs (base vertebrate = all 1.0, except knobs whose base is 0)
    body_size: float = 1.0
    cervical_elong: float = 1.0
    trunk_len: float = 1.0
    trunk_girth: float = 1.0
    tail_len: float = 1.0
    tail_count: float = 1.0
    limb_len: float = 1.0
    limb_gracility: float = 1.0
    skull_size: float = 1.0
    snout_len: float = 1.0
    neck_raise: float = 0.0
    horn_size: float = 0.0
    tusk_len: float = 0.0
    proboscis_len: float = 0.0
    coat_wavelength: float = 1.0
    convergent_ext: float = 1.0          # Wnt-PCP convergent extension: width (fish<->tetrapod); see KNOBS
    # discrete genes
    horn_mode: str = "none"
    coat_type: str = "plain"
    posture: str = "erect"
    dentition: str = "normal"
    nose: str = "normal"
    sex: str = "none"
    body_plan: str = "tetrapod"

    # ---- vector interface for the evolution strategy (continuous knobs only) ----
    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in CONT], dtype=float)

    @staticmethod
    def from_vector(v, template: "Genome | None" = None) -> "Genome":
        base = replace(template) if template is not None else Genome()
        for k, val in zip(CONT, v):
            lo, hi, *_ = KNOBS[k]
            setattr(base, k, float(np.clip(val, lo, hi)))
        return base

    def clipped(self) -> "Genome":
        g = replace(self)
        for k in CONT:
            lo, hi, *_ = KNOBS[k]
            setattr(g, k, float(np.clip(getattr(g, k), lo, hi)))
        return g

    def n_cervical(self) -> int:
        return CERVICAL_COUNT            # never changes -- the frozen count

    def as_dict(self):
        return asdict(self)


def base_vertebrate() -> Genome:
    """The generic tetrapod all species evolve from."""
    return Genome()


def lora_rank(g: Genome, base: Genome | None = None) -> int:
    """How many knobs differ from the base vertebrate -- the 'rank' of the species adapter."""
    base = base or base_vertebrate()
    n = 0
    for k in CONT:
        if abs(getattr(g, k) - getattr(base, k)) > 1e-3:
            n += 1
    for k in DISCRETE:
        if getattr(g, k) != getattr(base, k):
            n += 1
    return n
