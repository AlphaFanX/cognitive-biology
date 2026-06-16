"""
Zebrafish (Danio rerio) Bioelectric Tissue Characterization
=============================================================

Hardcoded bioelectric data for zebrafish embryonic development,
from Silic et al. 2022 (the only whole-embryo developmental voltage
atlas for any of our LSM organisms) and functional genetics studies.

Data sources:
    Whole-embryo voltage imaging:
        Silic, Dong, Chen, Kimbrough & Zhang 2022 (Cells 11:3586)
        - Tg(ubiquitin:ASAP1) transgenic, light-sheet microscopy
        - Stage-by-stage from cleavage through segmentation
        - RELATIVE fluorescence (not calibrated to absolute mV)

    Fin bioelectricity:
        Perathoner et al. 2014 (PLoS Genet) - kcnk5b gain-of-function
        Silic & Zhang 2020 (Genetics) - dermomyotome K+ channels
        Bhatt et al. 2018 (Sci Rep) - calcineurin-NFAT pathway

    Review:
        Bhatt & Adams 2023 (Cells 12:1148) - zebrafish bioelectric tools

    Ion channels & electrophysiology:
        Bhatt et al. 2021 (Bioelectricity) - ion channel mutant phenotypes
        Tseng & Bhatt 2014 (Biomolecules) - bioelectric signaling review

Key limitation: All Silic et al. 2022 data is relative fluorescence intensity
(ASAP1 deltaF/F), NOT calibrated to absolute millivolts. We encode the
qualitative patterns (hyper/depolarized relative to surroundings) and use
vertebrate electrophysiology literature for absolute estimates.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Developmental stages (Kimmel et al. 1995) with bioelectric annotations
# =============================================================================

@dataclass
class ZebrafishBioelectricStage:
    """Bioelectric signature at a Kimmel developmental stage."""
    stage_name: str
    hpf: float                        # Hours post-fertilization
    kimmel_period: str                 # Cleavage, Blastula, Gastrula, etc.
    voltage_pattern: str              # Qualitative description
    key_features: List[str]           # Bioelectric events at this stage
    tissue_polarity: Dict[str, str]   # tissue -> "hyper"/"depol"/"neutral"
    source: str                       # Data source


DEVELOPMENTAL_VOLTAGE_ATLAS = [
    # ========== CLEAVAGE PERIOD (0.75-2.25 hpf) ==========
    ZebrafishBioelectricStage(
        stage_name="2-cell",
        hpf=0.75,
        kimmel_period="Cleavage",
        voltage_pattern="Cleavage furrow hyperpolarization",
        key_features=[
            "Local membrane HYPERPOLARIZATION at cleavage furrow",
            "Signal appears BEFORE cytokinesis completes",
            "Fluctuates during cleavage progression",
        ],
        tissue_polarity={"cleavage_furrow": "hyper", "blastomere_body": "neutral"},
        source="Silic et al. 2022"
    ),
    ZebrafishBioelectricStage(
        stage_name="8-cell",
        hpf=1.25,
        kimmel_period="Cleavage",
        voltage_pattern="Furrow hyperpolarization persists at all cleavage planes",
        key_features=[
            "Each new cleavage furrow shows transient hyperpolarization",
            "Original center furrow (from 1st cleavage) REMAINS hyperpolarized",
            "Creates a grid-like voltage pattern on the blastoderm surface",
        ],
        tissue_polarity={"cleavage_furrows": "hyper", "cell_bodies": "neutral"},
        source="Silic et al. 2022"
    ),
    ZebrafishBioelectricStage(
        stage_name="64-cell",
        hpf=2.0,
        kimmel_period="Cleavage",
        voltage_pattern="Dense furrow network, less distinct individual signals",
        key_features=[
            "Many overlapping furrow signals",
            "Superficial cells show more ASAP1 signal than deep cells",
        ],
        tissue_polarity={"surface_blastomeres": "hyper", "deep_cells": "neutral"},
        source="Silic et al. 2022"
    ),

    # ========== BLASTULA PERIOD (2.25-5.25 hpf) ==========
    ZebrafishBioelectricStage(
        stage_name="256-cell",
        hpf=2.5,
        kimmel_period="Blastula",
        voltage_pattern="Transition from furrow to whole-cell transient events",
        key_features=[
            "Whole-cell transient hyperpolarization in superficial blastomeres",
            "NOT detected in deeper cells until gastrulation",
            "Enveloping layer (EVL) cells show distinct bioelectric behavior",
        ],
        tissue_polarity={"EVL": "hyper_transient", "deep_cells": "neutral"},
        source="Silic et al. 2022"
    ),
    ZebrafishBioelectricStage(
        stage_name="sphere",
        hpf=4.0,
        kimmel_period="Blastula",
        voltage_pattern="Sporadic transient hyperpolarization events in EVL",
        key_features=[
            "YSL (yolk syncytial layer) begins to form",
            "Maternal-zygotic transition occurring",
            "Bioelectric signaling becomes cell-autonomous",
            "YSL shows transient HYPERPOLARIZATION (Silic Fig 4), not depolarization; "
            "becomes prominent during gastrula epiboly",
        ],
        tissue_polarity={"EVL": "hyper_transient", "YSL": "hyper_transient", "deep_cells": "neutral"},
        source="Silic et al. 2022"
    ),

    # ========== GASTRULA PERIOD (5.25-10 hpf) ==========
    ZebrafishBioelectricStage(
        stage_name="shield",
        hpf=6.0,
        kimmel_period="Gastrula",
        voltage_pattern="Whole-cell transient hyperpolarization spreads from superficial layers into deep cells during epiboly",
        key_features=[
            "Silic Fig 4: transient hyperpolarization in EVL and YSL continues",
            "Headline gastrula finding -- transients begin appearing in DEEP cells at ~30% epiboly",
            "Deep cell movements begin (involution)",
            "Organizer/ventral resting pattern below is INFERRED (morphogen data), not Silic-imaged",
        ],
        tissue_polarity={
            # Silic et al. 2022 (Fig 4): transient hyperpolarization, superficial -> deep during epiboly
            "EVL": "hyper_transient",
            "YSL": "hyper_transient",
            "deep_cells_epiboly": "hyper_transient",
            # Inferred from morphogen data (NOT Silic imaging):
            "shield_organizer": "hyper",
            "ventral_margin": "depol",
        },
        source="Silic et al. 2022 (Fig 4: EVL/YSL/deep-cell transients during epiboly); organizer/ventral pattern inferred from morphogen data"
    ),
    ZebrafishBioelectricStage(
        stage_name="bud",
        hpf=10.0,
        kimmel_period="Gastrula",
        voltage_pattern="Tail bud and neural plate visible",
        key_features=[
            "Neural plate begins to show bioelectric identity",
            "Somite precursors (PSM) becoming organized",
            "Notochord specified",
        ],
        tissue_polarity={
            "neural_plate": "hyper",
            "notochord": "hyper",
            "epidermis": "depol",
            "PSM": "neutral",
        },
        source="Inferred from vertebrate data + Silic et al. 2022"
    ),

    # ========== SEGMENTATION PERIOD (10-24 hpf) ==========
    ZebrafishBioelectricStage(
        stage_name="6-somite",
        hpf=12.0,
        kimmel_period="Segmentation",
        voltage_pattern="Somites and notochord hyperpolarized relative to surroundings",
        key_features=[
            "SOMITES are HYPERPOLARIZED vs adjacent tissue",
            "NOTOCHORD is HYPERPOLARIZED",
            "This is the first clear tissue-level voltage compartmentalization",
            "Neural tube forming with distinct bioelectric signature",
        ],
        tissue_polarity={
            "somites": "hyper",
            "notochord": "hyper",
            "neural_tube": "hyper",
            "epidermis": "depol",
            "lateral_plate_mesoderm": "neutral",
        },
        source="Silic et al. 2022"
    ),
    ZebrafishBioelectricStage(
        stage_name="12-somite",
        hpf=15.0,
        kimmel_period="Segmentation",
        voltage_pattern="Dynamic Vm fluctuations in middle-aged somites",
        key_features=[
            "Middle-aged somites show DYNAMIC voltage fluctuations",
            "Fluctuations start at ~12-somite stage",
            "Newest (posterior) somites are more uniformly hyperpolarized",
            "Oldest (anterior) somites stabilize",
            "This may correlate with myotome differentiation waves",
        ],
        tissue_polarity={
            "new_somites_posterior": "hyper_stable",
            "mid_somites": "hyper_oscillating",
            "old_somites_anterior": "hyper_stable",
            "notochord": "hyper",
            "neural_tube": "hyper",
            "epidermis": "depol",
        },
        source="Silic et al. 2022"
    ),
    ZebrafishBioelectricStage(
        stage_name="18-somite",
        hpf=18.0,
        kimmel_period="Segmentation",
        voltage_pattern="Organ primordia emerging with distinct voltage",
        key_features=[
            "Optic vesicles forming (forebrain, anterior hyperpolarized)",
            "Heart primordia forming (lateral plate mesoderm)",
            "Pronephros forming (intermediate mesoderm)",
            "Otic vesicle forming",
        ],
        tissue_polarity={
            "brain_forebrain": "hyper",
            "brain_midbrain": "hyper",
            "brain_hindbrain": "hyper",
            "optic_vesicle": "hyper",
            "heart_primordium": "depol",  # Cardiac cells are depolarized
            "pronephros": "neutral",
            "somites": "hyper",
            "notochord": "hyper",
        },
        source="Somites/notochord: validated by Silic et al. 2022 (segmentation). Organ-primordia Vmem "
               "(optic vesicle, heart primordium, pronephros, brain regions) are MODEL PREDICTIONS extrapolated "
               "beyond the Silic atlas (which ends at somitogenesis), consistent with vertebrate electrophysiology "
               "-- falsifiable, awaiting later-stage GEVI imaging"
    ),

    # ========== PHARYNGULA PERIOD (24-48 hpf) ==========
    ZebrafishBioelectricStage(
        stage_name="prim-5",
        hpf=24.0,
        kimmel_period="Pharyngula",
        voltage_pattern="Organogenesis with tissue-specific voltage compartments",
        key_features=[
            "Heart beating begins (rhythmic depolarization)",
            "Brain regionalized (forebrain/midbrain/hindbrain)",
            "Pigmentation beginning (melanophores)",
            "Spontaneous muscle contractions (Ca2+ driven)",
        ],
        tissue_polarity={
            "brain": "hyper",
            "spinal_cord": "hyper",
            "heart": "depol_oscillating",
            "somite_muscle": "hyper_excitable",
            "notochord": "hyper",
            "skin": "depol",
            "gut_endoderm": "depol",
            "liver_bud": "depol",
        },
        source="Model predictions (vertebrate electrophysiology); heart Vm anchored to Hou et al. 2014 "
               "(Front Physiol 5:344, in vivo zebrafish membrane-voltage/Ca mapping). Beyond Silic atlas range."
    ),
    ZebrafishBioelectricStage(
        stage_name="long-pec",
        hpf=48.0,
        kimmel_period="Pharyngula",
        voltage_pattern="Functional organs with mature bioelectric profiles",
        key_features=[
            "Hatching occurs (48-72 hpf)",
            "Pectoral fin buds growing",
            "Swim bladder inflating",
            "Functional neural circuits (escape response)",
        ],
        tissue_polarity={
            "brain": "hyper",
            "retina": "hyper",
            "heart": "depol_oscillating",
            "skeletal_muscle": "hyper_excitable",
            "skin": "depol",
            "gut": "depol",
            "liver": "depol",
            "pancreas": "depol",
            "fin_bud": "hyper",  # Fin growth requires hyperpolarization (kcnk5b)
        },
        source="Model predictions (vertebrate electrophysiology); fin: Perathoner 2014; heart Vm anchored to "
               "Hou et al. 2014 (Front Physiol 5:344, in vivo zebrafish membrane-voltage/Ca mapping). Beyond Silic atlas range."
    ),
]


# =============================================================================
# Tissue-level resting potentials (estimated, mV)
# =============================================================================
# No calibrated mV values exist from zebrafish embryo imaging (Silic data
# is relative deltaF/F). These are estimates from vertebrate electrophysiology
# literature and the qualitative Silic et al. patterns.

TISSUE_VMEM_ESTIMATES = {
    # Neural tissues -- hyperpolarized (like all vertebrates)
    "brain_neuron": -65.0,           # Vertebrate neuron consensus
    "spinal_cord_neuron": -65.0,
    "retinal_ganglion": -60.0,
    "sensory_neuron_DRG": -60.0,
    "rohon_beard": -55.0,           # Zebrafish-specific primary sensory neurons

    # Muscle -- variable
    "skeletal_muscle": -85.0,        # Vertebrate skeletal muscle consensus
    "cardiac_muscle": -80.0,         # Resting (between action potentials)
    "smooth_muscle": -55.0,

    # Somites (pre-differentiation) -- hyperpolarized per Silic 2022
    "somite_new": -60.0,             # Posterior, just segmented
    "somite_maturing": -55.0,        # Mid-body, oscillating
    "somite_mature": -65.0,          # Anterior, stabilized

    # Notochord -- hyperpolarized per Silic 2022
    "notochord": -60.0,              # Consistently hyperpolarized

    # Neural crest derivatives
    "melanophore": -45.0,            # Pigment cell
    "craniofacial_cartilage": -40.0,

    # Epithelia -- moderately depolarized
    "epidermis": -35.0,
    "EVL": -40.0,                    # Enveloping layer (early)
    "ionocyte": -30.0,               # Ion-transporting skin cells

    # Endoderm -- depolarized
    "gut_endoderm": -30.0,
    "liver": -35.0,
    "pancreas": -35.0,
    "swim_bladder": -40.0,

    # Mesoderm derivatives
    "blood_progenitor": -25.0,       # Hematopoietic
    "endothelium": -40.0,            # Vascular
    "pronephros": -45.0,             # Embryonic kidney
    "lateral_plate_mesoderm": -35.0,
    "heart_primordium": -40.0,       # Before differentiation

    # Fin (critical bioelectric tissue)
    "fin_mesenchyme": -55.0,         # Hyperpolarized (growth signal)
    "dermomyotome": -55.0,           # K+ channels set fin patterning

    # Germ cells
    "primordial_germ_cell": -30.0,   # Typically depolarized

    # Yolk
    "YSL": -25.0,                    # Yolk syncytial layer, depolarized
}


# =============================================================================
# Ion channel mutant phenotypes (functional bioelectric genetics)
# =============================================================================

@dataclass
class IonChannelMutant:
    """Zebrafish ion channel mutant with bioelectric phenotype."""
    gene: str
    channel_type: str          # K+, Na+, Ca2+, Cl-
    mutation_type: str         # LOF, GOF, dominant-negative
    phenotype: str
    affected_tissue: str
    bioelectric_effect: str    # "hyperpolarization", "depolarization"
    source: str


ION_CHANNEL_MUTANTS = [
    # Fin size regulation (the best-characterized zebrafish bioelectric phenotype)
    IonChannelMutant(
        gene="kcnk5b", channel_type="K+ (TWIK/TREK)",
        mutation_type="GOF",
        phenotype="Long fin (another longfin / alf)",
        affected_tissue="fin mesenchyme",
        bioelectric_effect="hyperpolarization -> excess growth",
        source="Perathoner et al. 2014 (PLoS Genet)"
    ),
    IonChannelMutant(
        gene="kcnj13", channel_type="K+ (Kir)",
        mutation_type="GOF",
        phenotype="Long fin (longfin / lof)",
        affected_tissue="fin mesenchyme",
        bioelectric_effect="hyperpolarization -> excess growth",
        source="Iovine & Johnson 2000; Silic & Zhang 2020"
    ),
    IonChannelMutant(
        gene="kcnj1b", channel_type="K+ (ROMK)",
        mutation_type="GOF",
        phenotype="Long fin when overexpressed in dermomyotome",
        affected_tissue="dermomyotome",
        bioelectric_effect="hyperpolarization",
        source="Silic & Zhang 2020 (Genetics)"
    ),
    IonChannelMutant(
        gene="kcnj10a", channel_type="K+ (Kir4.1)",
        mutation_type="GOF",
        phenotype="Long fin when overexpressed",
        affected_tissue="dermomyotome",
        bioelectric_effect="hyperpolarization",
        source="Silic & Zhang 2020 (Genetics)"
    ),
    IonChannelMutant(
        gene="kcnk9", channel_type="K+ (TASK3)",
        mutation_type="GOF",
        phenotype="Long fin when overexpressed",
        affected_tissue="dermomyotome",
        bioelectric_effect="hyperpolarization",
        source="Silic & Zhang 2020 (Genetics)"
    ),

    # Cardiac
    IonChannelMutant(
        gene="kcnh2a", channel_type="K+ (hERG/Kv11.1)",
        mutation_type="LOF",
        phenotype="Long QT, cardiac arrhythmia (breakdance)",
        affected_tissue="heart",
        bioelectric_effect="delayed repolarization",
        source="Langheinrich et al. 2003"
    ),
    IonChannelMutant(
        gene="scn5lab", channel_type="Na+ (NaV1.5)",
        mutation_type="LOF",
        phenotype="Cardiac conduction defect",
        affected_tissue="heart",
        bioelectric_effect="reduced depolarization",
        source="Chopra et al. 2010"
    ),
    IonChannelMutant(
        gene="cacna1c", channel_type="Ca2+ (L-type CaV1.2)",
        mutation_type="LOF",
        phenotype="No heartbeat (island beat)",
        affected_tissue="heart ventricular cardiomyocytes",
        bioelectric_effect="loss of Ca2+ current -> no contraction",
        source="Rottbauer et al. 2001"
    ),

    # Pigmentation (bioelectric control of melanophore fate)
    IonChannelMutant(
        gene="kcnj13", channel_type="K+ (Kir7.1)",
        mutation_type="LOF",
        phenotype="Jaguar/obelix -- disrupted stripe pattern",
        affected_tissue="melanophores",
        bioelectric_effect="depolarization -> altered cell-cell signaling",
        source="Iwashita et al. 2006"
    ),

    # Neural
    IonChannelMutant(
        gene="kcna1a", channel_type="K+ (Kv1.1)",
        mutation_type="LOF",
        phenotype="Seizure susceptibility",
        affected_tissue="brain",
        bioelectric_effect="hyperexcitability",
        source="Teng et al. 2010"
    ),

    # Muscle
    IonChannelMutant(
        gene="scn1lab", channel_type="Na+ (NaV1.1)",
        mutation_type="LOF",
        phenotype="Dravet syndrome model, seizures",
        affected_tissue="brain + muscle",
        bioelectric_effect="reduced excitability",
        source="Baraban et al. 2013"
    ),

    # Craniofacial
    IonChannelMutant(
        gene="kcnk5b", channel_type="K+ (TWIK/TREK)",
        mutation_type="GOF",
        phenotype="Jaw malformation alongside long fins",
        affected_tissue="craniofacial cartilage",
        bioelectric_effect="hyperpolarization alters chondrocyte behavior",
        source="Perathoner et al. 2014"
    ),
]


# =============================================================================
# Gap junction (connexin) genes in zebrafish
# =============================================================================
# Zebrafish have connexins (like all vertebrates), NOT innexins (like C. elegans).
# The zebrafish genome has ~40 connexin genes due to teleost genome duplication.

CONNEXIN_TISSUE_MAP = {
    # Neural gap junctions
    "cx35/cx35.5": {"tissues": ["retina", "Mauthner neurons", "hindbrain"],
                     "function": "Electrical synapses, fast escape response"},
    "cx34.1": {"tissues": ["retina", "brain"],
                "function": "Photoreceptor coupling"},
    "cx34.7": {"tissues": ["retina"],
                "function": "Horizontal cell coupling"},
    "cx55.5": {"tissues": ["retina horizontal cells"],
                "function": "Color opponency"},

    # Cardiac gap junctions
    "cx43 (gja1)": {"tissues": ["heart ventricle", "brain", "osteoblasts"],
                     "function": "Cardiac conduction, bone growth"},
    "cx45.6": {"tissues": ["heart atrium"],
                "function": "Atrial conduction"},
    "cx36.7": {"tissues": ["heart"],
                "function": "Cardiac pacemaker"},

    # Skin / epithelial
    "cx41.8 (gja5b)": {"tissues": ["skin", "fins"],
                         "function": "Melanophore-xanthophore coupling, stripe patterning"},
    "cx39.4": {"tissues": ["skin"],
                "function": "Pigment pattern (leopard)"},

    # Muscle
    "cx39.9": {"tissues": ["skeletal muscle"],
                "function": "Myocyte coupling"},

    # Lens
    "cx44.1": {"tissues": ["lens"],
                "function": "Lens fiber coupling, transparency"},
    "cx48.5": {"tissues": ["lens"],
                "function": "Lens homeostasis"},
}


# =============================================================================
# Tissue coupling strength (estimated from connexin expression)
# =============================================================================

TISSUE_COUPLING = {
    ("somite", "somite"): 0.7,                  # Strong within segments
    ("notochord", "notochord"): 0.8,             # Continuous rod
    ("neural_tube", "neural_tube"): 0.6,
    ("epidermis", "epidermis"): 0.5,
    ("heart", "heart"): 0.9,                     # Cardiac syncytium
    ("retina_horizontal", "retina_horizontal"): 0.8,
    ("melanophore", "xanthophore"): 0.4,         # cx41.8 (leopard/jaguar)
    ("skeletal_muscle", "skeletal_muscle"): 0.3,  # Myocyte coupling
    ("somite", "notochord"): 0.2,                # Weak boundary coupling
    ("neural_tube", "somite"): 0.2,
    ("epidermis", "somite"): 0.1,                # Tissue boundary, low coupling
}


# =============================================================================
# Convenience functions
# =============================================================================

def get_tissue_vmem(tissue: str) -> float:
    """Get estimated resting potential for a tissue type."""
    return TISSUE_VMEM_ESTIMATES.get(tissue, -50.0)


def get_stage_bioelectric(stage_name: str) -> Optional[ZebrafishBioelectricStage]:
    """Get bioelectric data for a Kimmel stage."""
    for stage in DEVELOPMENTAL_VOLTAGE_ATLAS:
        if stage.stage_name == stage_name:
            return stage
    return None


def get_stage_by_hpf(hpf: float) -> Optional[ZebrafishBioelectricStage]:
    """Get closest bioelectric stage for a given time (hours post-fertilization)."""
    closest = None
    min_diff = float('inf')
    for stage in DEVELOPMENTAL_VOLTAGE_ATLAS:
        diff = abs(stage.hpf - hpf)
        if diff < min_diff:
            min_diff = diff
            closest = stage
    return closest


def get_coupling_strength(tissue_a: str, tissue_b: str) -> float:
    """Get gap junction coupling strength between two tissues."""
    key = (tissue_a, tissue_b)
    if key in TISSUE_COUPLING:
        return TISSUE_COUPLING[key]
    key_rev = (tissue_b, tissue_a)
    if key_rev in TISSUE_COUPLING:
        return TISSUE_COUPLING[key_rev]
    return 0.0


def get_channel_mutants_for_tissue(tissue: str) -> List[IonChannelMutant]:
    """Get all ion channel mutants affecting a given tissue."""
    return [m for m in ION_CHANNEL_MUTANTS if tissue in m.affected_tissue]


def print_bioelectric_summary():
    """Print summary of all zebrafish bioelectric data."""
    print("=" * 70)
    print("ZEBRAFISH (Danio rerio) BIOELECTRIC TISSUE CHARACTERIZATION")
    print("=" * 70)

    print(f"\n--- Developmental Voltage Atlas ({len(DEVELOPMENTAL_VOLTAGE_ATLAS)} stages) ---")
    print(f"    Source: Silic et al. 2022 (Tg(ubiquitin:ASAP1) + light-sheet)")
    print(f"    WARNING: Relative fluorescence, NOT calibrated mV")
    for stage in DEVELOPMENTAL_VOLTAGE_ATLAS:
        n_tissues = len(stage.tissue_polarity)
        print(f"\n  {stage.stage_name:15s} ({stage.hpf:5.1f} hpf, {stage.kimmel_period})")
        print(f"    Pattern: {stage.voltage_pattern}")
        for tissue, polarity in stage.tissue_polarity.items():
            print(f"      {tissue:30s}: {polarity}")

    print(f"\n--- Tissue Vmem Estimates ({len(TISSUE_VMEM_ESTIMATES)} tissues) ---")
    for tissue, vmem in sorted(TISSUE_VMEM_ESTIMATES.items(), key=lambda x: x[1]):
        print(f"  {tissue:30s}: {vmem:6.1f} mV")

    print(f"\n--- Ion Channel Mutants ({len(ION_CHANNEL_MUTANTS)} entries) ---")
    for m in ION_CHANNEL_MUTANTS:
        print(f"  {m.gene:12s} ({m.channel_type:20s}): {m.phenotype}")
        print(f"    {'':12s}  Effect: {m.bioelectric_effect}")

    print(f"\n--- Connexin Map ({len(CONNEXIN_TISSUE_MAP)} connexins) ---")
    for cx, info in CONNEXIN_TISSUE_MAP.items():
        tissues = ", ".join(info["tissues"])
        print(f"  {cx:20s}: {tissues}")

    print(f"\n--- Key Bioelectric Insights ---")
    print("  Zebrafish is the ONLY organism with a whole-embryo voltage atlas")
    print("  Somites and notochord are hyperpolarized (Silic et al. 2022)")
    print("  Fin growth is K+ channel dependent (kcnk5b, kcnj13)")
    print("  Pigment patterns use connexin-mediated bioelectric signaling")
    print("  All data is relative fluorescence, NOT calibrated mV")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_bioelectric_summary()
