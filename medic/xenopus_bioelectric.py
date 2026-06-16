"""
Xenopus laevis Bioelectric Tissue Characterization
====================================================

Hardcoded bioelectric data for Xenopus embryonic development,
from Michael Levin's lab (Tufts University) -- the richest bioelectric
dataset for any vertebrate embryo.

Xenopus is THE model organism for developmental bioelectricity because:
1. Large embryos (easy to image/inject)
2. External development (accessible at all stages)
3. Well-characterized staging (Nieuwkoop-Faber, 66 stages)
4. DiBAC4(3) / CC2-DMPE voltage-sensitive dye imaging
5. Levin lab has 20+ years of Vmem mapping data

Data sources:
    Voltage imaging (DiBAC4(3) / CC2-DMPE):
        Vandenberg, Morrie & Adams 2011 (Development 139:313-323)
        - Bilateral hyperpolarized spots mark future eye positions (NF 17/18)
        Adams & Levin 2012 (Cold Spring Harb Protoc)
        - Protocol for voltage reporter pair imaging

    Craniofacial bioelectricity ("The Electric Face"):
        Adams et al. 2016 (J Physiol 594:3245-3270, PMC4908029)
        - KCNJ2 (Kir2.1) Andersen-Tawil mutations cause face defects
        - Voltage prepattern at NF 19 defines face territories

    Neural plate calibrated Vmem:
        Pai et al. 2015 (J Neurosci 35:4366, PMC4355204)
        - Neural plate cells: -51 mV (patch-clamp, NF 16-17)
        - Non-neural neighbors: ~40 mV more depolarized

    Left-right asymmetry:
        Levin et al. 2002 (Cell 111:77-89)
        - H+/K+-ATPase asymmetric at 4-cell stage
        Fukumoto, Kema & Levin 2005 (Curr Biol 15:794-803)
        - Serotonin redistributes through gap junctions
        Morokuma, Blackiston & Levin 2008 (Cell Physiol Biochem 21:357-372)
        - KCNQ1/KCNE1 asymmetrically localized

    Tail regeneration:
        Adams, Masi & Bhatt 2007 (Development 134:1323-1335)
        - V-ATPase required in first 24h post-amputation
        Tseng et al. 2010 (J Neurosci 30:13192-13200)
        - NaV1.2 sodium current required for regeneration

    Eye induction:
        Pai et al. 2012 (Development 139:313-323)
        - DNKir6.1p (dominant-negative KATP) induces ectopic eyes

    Connexin atlas:
        Landesman et al. 2003 (Dev Biol 263:303-317)
        - 7 connexins characterized across development

    Blastomere electrophysiology:
        Slack & Warner 1973 (J Physiol 232:313-330, PMC1192408)
        - K+-dependent membrane potential, stage 7-9

Key advantage over zebrafish: Levin lab has calibrated mV measurements
(Pai et al. 2015) plus extensive functional perturbation data (optogenetics,
dominant-negative channels, pharmacology). Zebrafish Silic data is relative
fluorescence only.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Developmental stages with bioelectric annotations (Nieuwkoop-Faber)
# =============================================================================

@dataclass
class XenopusBioelectricStage:
    """Bioelectric signature at a Nieuwkoop-Faber developmental stage."""
    stage_name: str
    nf_stage: float               # NF stage number
    hpf: float                    # Hours post-fertilization (~22-23C)
    period: str                   # Cleavage, Blastula, Gastrula, etc.
    voltage_pattern: str          # Qualitative description
    key_features: List[str]       # Bioelectric events at this stage
    tissue_polarity: Dict[str, str]   # tissue -> "hyper"/"depol"/"neutral"
    source: str                   # Data source


DEVELOPMENTAL_VOLTAGE_ATLAS = [
    # ========== CLEAVAGE PERIOD (0-3.5h, NF1-6) ==========
    XenopusBioelectricStage(
        stage_name="1-cell",
        nf_stage=1,
        hpf=0.0,
        period="Cleavage",
        voltage_pattern="Uniform maternal voltage, animal-vegetal polarity",
        key_features=[
            "Maternal H+/K+-ATPase mRNA symmetrically distributed",
            "K+-dependent membrane potential (~-30 mV from Slack & Warner 1973)",
            "Animal-vegetal axis established (cortical rotation)",
        ],
        tissue_polarity={"whole_egg": "neutral"},
        source="Slack & Warner 1973; Levin et al. 2002"
    ),
    XenopusBioelectricStage(
        stage_name="4-cell",
        nf_stage=3,
        hpf=2.0,
        period="Cleavage",
        voltage_pattern="CRITICAL: H+/K+-ATPase becomes asymmetric",
        key_features=[
            "H+/K+-ATPase mRNA localizes to VENTRAL RIGHT blastomere",
            "Ventral right becomes more HYPERPOLARIZED than left",
            "KCNQ1/KCNE1 also asymmetrically distributed",
            "This is the earliest known left-right symmetry breaking event",
            "Omeprazole (H+/K+-ATPase inhibitor) randomizes laterality",
        ],
        tissue_polarity={
            "ventral_right": "hyper",
            "ventral_left": "depol",
            "dorsal_right": "neutral",
            "dorsal_left": "neutral",
        },
        source="Levin et al. 2002 (Cell); Morokuma et al. 2008"
    ),
    XenopusBioelectricStage(
        stage_name="32-cell",
        nf_stage=6,
        hpf=3.5,
        period="Cleavage",
        voltage_pattern="Serotonin redistribution through gap junctions begins",
        key_features=[
            "5-HT (serotonin) initially homogeneous",
            "Begins redistribution through Cx43 circumferential dorsal path",
            "Gap junction communication (GJC) is REQUIRED for LR transfer",
            "Blocking GJC randomizes laterality",
        ],
        tissue_polarity={
            "right_blastomeres": "hyper",
            "left_blastomeres": "neutral",
        },
        source="Fukumoto et al. 2005 (Curr Biol)"
    ),

    # ========== BLASTULA PERIOD (4.5-7h, NF7-9) ==========
    XenopusBioelectricStage(
        stage_name="mid-blastula",
        nf_stage=8,
        hpf=5.0,
        period="Blastula",
        voltage_pattern="K+ selectivity increasing, Na-K pump declining",
        key_features=[
            "Membrane becomes more K+-selective (Slack & Warner 1973)",
            "Na-K pump activity: ~0.19 uA/uF at NF7 -> ~0.04 uA/uF at NF9",
            "Specific membrane resistance: 100-300 kohm.cm2",
            "Serotonin accumulates on RIGHT side (completes LR signal)",
        ],
        tissue_polarity={
            "animal_cap": "depol",
            "vegetal_mass": "neutral",
            "marginal_zone": "neutral",
        },
        source="Slack & Warner 1973 (J Physiol)"
    ),
    XenopusBioelectricStage(
        stage_name="late-blastula",
        nf_stage=9,
        hpf=7.0,
        period="Blastula",
        voltage_pattern="Mid-blastula transition (MBT), zygotic transcription begins",
        key_features=[
            "Zygotic genome activation",
            "LR asymmetric serotonin signal COMPLETE by this stage",
            "Right-side 5-HT represses Xnr-1 via HDAC (epigenetic)",
            "This LR mechanism is CILIA-INDEPENDENT",
        ],
        tissue_polarity={
            "animal_cap": "depol",
            "marginal_zone": "neutral",
            "vegetal_mass": "neutral",
        },
        source="Fukumoto et al. 2005; Levin et al. 2002"
    ),

    # ========== GASTRULA PERIOD (9-14h, NF10-12.5) ==========
    XenopusBioelectricStage(
        stage_name="early-gastrula",
        nf_stage=10,
        hpf=9.0,
        period="Gastrula",
        voltage_pattern="Organizer region bioelectric identity forming",
        key_features=[
            "Dorsal lip (Spemann organizer) forms",
            "Ectoderm ~-30 mV (Warner 1973, axolotl -- closest available)",
            "Presumptive neural ectoderm ~-27 mV (not yet differentiated)",
            "Mesoderm involuting through blastopore",
        ],
        tissue_polarity={
            "dorsal_ectoderm": "neutral",  # ~-30 mV
            "ventral_ectoderm": "neutral",
            "organizer": "hyper",  # Inferred from BMP antagonism
            "endoderm": "depol",
        },
        source="Warner 1973 (J Physiol, axolotl); inferred"
    ),

    # ========== NEURULA PERIOD (15-21h, NF13-19) ==========
    XenopusBioelectricStage(
        stage_name="early-neurula",
        nf_stage=13,
        hpf=15.0,
        period="Neurula",
        voltage_pattern="Neural plate begins to hyperpolarize",
        key_features=[
            "First detectable neural hyperpolarization (Pai et al. 2015)",
            "Neural plate separating bioelectrically from epidermis",
            "CRITICAL WINDOW for craniofacial patterning begins (NF 11-14)",
        ],
        tissue_polarity={
            "neural_plate": "hyper",
            "epidermis": "depol",
            "notochord": "hyper",
        },
        source="Pai et al. 2015 (J Neurosci)"
    ),
    XenopusBioelectricStage(
        stage_name="mid-neurula",
        nf_stage=16,
        hpf=18.0,
        period="Neurula",
        voltage_pattern="Neural plate CALIBRATED: -51 mV (patch-clamp)",
        key_features=[
            "Neural plate cells: -51 mV (Pai et al. 2015, whole-cell patch-clamp)",
            "Non-neural ectoderm: ~-11 mV (40 mV more depolarized)",
            "THIS IS THE ONLY CALIBRATED Vmem IN XENOPUS EMBRYOS",
            "Neural folds elevating",
        ],
        tissue_polarity={
            "neural_plate": "hyper",     # -51 mV CALIBRATED
            "non_neural_ectoderm": "depol",  # ~-11 mV CALIBRATED
            "neural_folds": "hyper",
            "notochord": "hyper",
        },
        source="Pai et al. 2015 (J Neurosci, PMC4355204)"
    ),
    XenopusBioelectricStage(
        stage_name="late-neurula-eye",
        nf_stage=17,
        hpf=19.0,
        period="Neurula",
        voltage_pattern="Bilateral hyperpolarized spots mark future EYE positions",
        key_features=[
            "CC2-DMPE/DiBAC voltage dye pair reveals bilateral spots",
            "Spots are ~10 mV more negative than surrounding tissue",
            "These mark PROSPECTIVE EYE positions (confirmed by fate mapping)",
            "Optogenetic depolarization at this stage causes eye defects",
        ],
        tissue_polarity={
            "eye_primordia_L": "hyper",
            "eye_primordia_R": "hyper",
            "neural_plate": "hyper",
            "epidermis": "depol",
        },
        source="Vandenberg et al. 2011 (Development)"
    ),
    XenopusBioelectricStage(
        stage_name="electric-face",
        nf_stage=19,
        hpf=21.0,
        period="Neurula",
        voltage_pattern="THE ELECTRIC FACE: Vmem prepattern defines face territories",
        key_features=[
            "Broad HYPERPOLARIZED region in anterior central ectoderm",
            "This region later thins and splits -> nose and mouth positions",
            "Bilateral HYPERPOLARIZED spots -> eye positions (from NF 17)",
            "Surrounding non-face ectoderm is DEPOLARIZED",
            "KCNJ2 mutations disrupt this pattern -> craniofacial anomalies",
            "Downstream genes affected: Otx2, Six1, Sox3, FoxE4",
            "Critical window ENDS here -- perturbation at NF 20-24 has NO effect",
        ],
        tissue_polarity={
            "face_midline": "hyper",
            "prospective_nose": "hyper",
            "prospective_mouth": "hyper",
            "eye_fields": "hyper",
            "surrounding_ectoderm": "depol",
        },
        source="Adams et al. 2016 (J Physiol, PMC4908029)"
    ),

    # ========== TAILBUD PERIOD (22-50h, NF20-33) ==========
    XenopusBioelectricStage(
        stage_name="early-tailbud",
        nf_stage=22,
        hpf=24.0,
        period="Tailbud",
        voltage_pattern="Organ primordia with distinct bioelectric compartments",
        key_features=[
            "Eye primordia visible (correlates with voltage spots)",
            "Neural tube closed, brain regionalized",
            "Craniofacial critical window CLOSED (perturbation no longer affects face)",
            "Warner 1973 (axolotl): nerve cells ~-44 mV, ectoderm ~-31 mV",
        ],
        tissue_polarity={
            "brain": "hyper",
            "eye": "hyper",
            "spinal_cord": "hyper",
            "somites": "hyper",
            "notochord": "hyper",
            "epidermis": "depol",
            "gut_endoderm": "depol",
        },
        source="Warner 1973; Vandenberg et al. 2011"
    ),
    XenopusBioelectricStage(
        stage_name="heartbeat",
        nf_stage=26,
        hpf=30.0,
        period="Tailbud",
        voltage_pattern="Heart beating, functional voltage-gated channels in heart",
        key_features=[
            "Heart begins beating (rhythmic depolarization-repolarization)",
            "Cardiac ion channels functional (Ca2+, K+, Na+)",
            "Tail elongating",
        ],
        tissue_polarity={
            "brain": "hyper",
            "heart": "depol_oscillating",
            "somites": "hyper",
            "tail": "hyper",
            "epidermis": "depol",
        },
        source="Vertebrate cardiac physiology"
    ),

    # ========== TADPOLE PERIOD (50h+, NF35-46) ==========
    XenopusBioelectricStage(
        stage_name="regeneration-competent",
        nf_stage=40,
        hpf=66.0,
        period="Tadpole",
        voltage_pattern="Tail regeneration-competent, V-ATPase responsive",
        key_features=[
            "Tadpole feeding begins",
            "Tail amputation -> V-ATPase upregulated at wound (6 hpa)",
            "Wound repolarizes by 24 hpa (V-ATPase dependent)",
            "NaV1.2 appears at 18 hpa in regeneration bud mesenchyme",
            "Full tail regeneration (spinal cord, muscle, notochord)",
            "Rohon-Beard neurons: -88 mV (Bhatt, J Physiol)",
        ],
        tissue_polarity={
            "brain": "hyper",
            "spinal_cord": "hyper",
            "muscle": "hyper",
            "tail_wound": "depol",  # Initially depolarized, then repolarizes
            "skin": "depol",
        },
        source="Adams et al. 2007; Tseng et al. 2010"
    ),
    XenopusBioelectricStage(
        stage_name="refractory-period",
        nf_stage=46,
        hpf=192.0,  # ~8 days
        period="Tadpole",
        voltage_pattern="Refractory period: tails FAIL to repolarize after amputation",
        key_features=[
            "NF 45-47 is the 'refractory period' for tail regeneration",
            "Amputated tails remain HIGHLY DEPOLARIZED (fail to repolarize)",
            "V-ATPase NOT upregulated at wound",
            "NaV1.2 NOT expressed in wound",
            "Forced NaV1.5 expression or monensin (Na+ ionophore) RESCUES regen",
            "Hindlimb bud visible",
        ],
        tissue_polarity={
            "tail_wound": "depol",  # Stays depolarized -> no regeneration
            "hindlimb_bud": "hyper",
        },
        source="Adams et al. 2007; Tseng et al. 2010"
    ),
]


# =============================================================================
# Tissue-level resting potentials (mV)
# =============================================================================
# Xenopus has the ONLY calibrated embryonic voltage measurement among our
# organisms (Pai et al. 2015: neural plate -51 mV at NF 16-17).
# Other values from Warner 1973 (axolotl, closest amphibian data),
# vertebrate consensus, and Levin lab functional studies.

TISSUE_VMEM_ESTIMATES = {
    # === CALIBRATED (Xenopus patch-clamp) ===
    "neural_plate": -51.0,           # Pai et al. 2015, NF 16-17, CALIBRATED
    "non_neural_ectoderm": -11.0,    # Pai et al. 2015, 40 mV more depolarized

    # === From axolotl electrophysiology (Warner 1973) ===
    # Closest available amphibian data, commonly cited for Xenopus comparison
    "gastrula_ectoderm": -30.0,      # Warner 1973 (axolotl, +/- 1.5 mV)
    "gastrula_presumptive_neural": -27.0,  # Warner 1973 (axolotl, +/- 1.6 mV)
    "late_neurula_nerve": -44.0,     # Warner 1973 (axolotl, +/- 1.7 mV)
    "late_neurula_ectoderm": -31.0,  # Warner 1973 (axolotl, +/- 1.5 mV)

    # === Xenopus-specific measurements ===
    "rohon_beard_neuron": -88.0,     # Bhatt, J Physiol (Xenopus tadpole)

    # === Neural tissues (vertebrate consensus + Xenopus context) ===
    "brain_neuron": -65.0,           # Vertebrate consensus
    "spinal_cord_neuron": -65.0,
    "retinal_ganglion": -60.0,
    "eye_primordia": -61.0,          # ~10 mV more negative than neural plate

    # === Muscle ===
    "skeletal_muscle": -85.0,        # Vertebrate skeletal muscle consensus
    "cardiac_muscle": -80.0,         # Resting (between action potentials)
    "smooth_muscle": -55.0,

    # === Somites ===
    "somite_new": -55.0,
    "somite_mature": -60.0,

    # === Notochord ===
    "notochord": -55.0,              # Hyperpolarized relative to ectoderm

    # === Epithelia ===
    "epidermis": -35.0,
    "ionocyte": -30.0,

    # === Endoderm ===
    "gut_endoderm": -30.0,
    "liver_bud": -35.0,
    "pancreas_bud": -35.0,

    # === Mesoderm derivatives ===
    "heart_primordium": -40.0,
    "pronephros": -45.0,
    "blood_island": -25.0,
    "lateral_plate_mesoderm": -35.0,

    # === Neural crest ===
    "cranial_neural_crest": -50.0,   # Hyperpolarized (migratory)
    "trunk_neural_crest": -45.0,

    # === Craniofacial (from Adams et al. 2016 functional data) ===
    "face_midline_hyper": -55.0,     # Hyperpolarized face territory
    "face_surround_depol": -15.0,    # Depolarized surrounding ectoderm
    "meckel_cartilage": -40.0,
    "branchial_arch": -40.0,

    # === Regeneration ===
    "tail_wound_regen_competent": -20.0,   # Initially depol, repolarizes
    "tail_wound_refractory": -10.0,        # Stays depolarized -> no regen

    # === Germ cells ===
    "primordial_germ_cell": -30.0,

    # === Early blastomere ===
    "blastomere_nf7": -30.0,        # Slack & Warner 1973 (K+-dependent)
    "blastomere_nf9": -35.0,        # More K+-selective at this stage
}


# =============================================================================
# Ion channel functional characterization
# =============================================================================

@dataclass
class IonChannelPerturbation:
    """Functionally characterized ion channel in Xenopus development."""
    gene: str
    channel_type: str          # K+, Na+, Ca2+, H+, etc.
    perturbation: str          # GOF, LOF, dominant-negative, pharmacological
    phenotype: str
    affected_system: str       # LR, craniofacial, regeneration, eye, brain
    bioelectric_effect: str
    quantitative_data: str     # Defect percentages, regeneration index, etc.
    source: str


ION_CHANNEL_PERTURBATIONS = [
    # === LEFT-RIGHT ASYMMETRY ===
    IonChannelPerturbation(
        gene="ATP4a (H+/K+-ATPase)",
        channel_type="H+/K+ exchanger",
        perturbation="Pharmacological (omeprazole, SCH28080) or dominant-negative",
        phenotype="Heterotaxia (randomized organ laterality)",
        affected_system="LR",
        bioelectric_effect="Loss of right-side hyperpolarization at 4-cell stage",
        quantitative_data="Omeprazole: significant heterotaxia vs controls",
        source="Levin et al. 2002 (Cell 111:77-89)"
    ),
    IonChannelPerturbation(
        gene="KCNQ1",
        channel_type="Voltage-gated K+",
        perturbation="Dominant-negative",
        phenotype="Heterotaxia",
        affected_system="LR",
        bioelectric_effect="Loss of asymmetric K+ current at 4-cell stage",
        quantitative_data="DN-KCNQ1: significant laterality defects",
        source="Morokuma et al. 2008 (Cell Physiol Biochem 21:357-372)"
    ),
    IonChannelPerturbation(
        gene="KCNE1",
        channel_type="K+ channel accessory subunit",
        perturbation="Dominant-negative",
        phenotype="Heterotaxia",
        affected_system="LR",
        bioelectric_effect="Disrupts KCNQ1 function",
        quantitative_data="Localization depends on microtubule + actin cytoskeleton",
        source="Morokuma et al. 2008"
    ),

    # === CRANIOFACIAL ("The Electric Face") ===
    IonChannelPerturbation(
        gene="KCNJ2 (Kir2.1) WT",
        channel_type="Inward-rectifier K+",
        perturbation="Overexpression (hyperpolarizes)",
        phenotype="Craniofacial anomalies (misshapen eyes, bent cartilage)",
        affected_system="craniofacial",
        bioelectric_effect="Excessive hyperpolarization disrupts face voltage pattern",
        quantitative_data="68% craniofacial defects vs 14% uninjected controls",
        source="Adams et al. 2016 (J Physiol 594:3245-3270)"
    ),
    IonChannelPerturbation(
        gene="KCNJ2 D71V (Andersen-Tawil)",
        channel_type="Inward-rectifier K+ (LOF)",
        perturbation="LOF mutation (depolarizes)",
        phenotype="Craniofacial anomalies",
        affected_system="craniofacial",
        bioelectric_effect="Depolarization disrupts face voltage pattern",
        quantitative_data="60% craniofacial defects",
        source="Adams et al. 2016"
    ),
    IonChannelPerturbation(
        gene="KCNJ2 T75R",
        channel_type="Inward-rectifier K+",
        perturbation="Andersen-Tawil mutation",
        phenotype="Craniofacial anomalies",
        affected_system="craniofacial",
        bioelectric_effect="Disrupts face voltage pattern",
        quantitative_data="59% craniofacial defects",
        source="Adams et al. 2016"
    ),
    IonChannelPerturbation(
        gene="KCNJ2 R218W",
        channel_type="Inward-rectifier K+ (LOF)",
        perturbation="LOF mutation (depolarizes)",
        phenotype="Craniofacial anomalies (mild)",
        affected_system="craniofacial",
        bioelectric_effect="Depolarization",
        quantitative_data="31% craniofacial defects",
        source="Adams et al. 2016"
    ),

    # === TAIL REGENERATION ===
    IonChannelPerturbation(
        gene="V-ATPase (ATP6V subunits)",
        channel_type="H+ pump",
        perturbation="Concanamycin (inhibitor) or dominant-negative subunit",
        phenotype="Complete loss of tail regeneration",
        affected_system="regeneration",
        bioelectric_effect="Wound fails to repolarize, stays depolarized",
        quantitative_data=(
            "V-ATPase upregulated at wound by 6 hpa. "
            "Required during first 24h. "
            "Inhibition abolishes regen but NOT wound healing."
        ),
        source="Adams et al. 2007 (Development 134:1323-1335)"
    ),
    IonChannelPerturbation(
        gene="SCN2A (NaV1.2)",
        channel_type="Voltage-gated Na+",
        perturbation="MS222 (tricaine, 250 uM) or RNAi",
        phenotype="Reduced tail regeneration",
        affected_system="regeneration",
        bioelectric_effect="Loss of Na+ influx in regeneration bud",
        quantitative_data=(
            "NaV1.2 appears 18 hpa in mesenchyme. "
            "MS222: RI 265->44. "
            "NaV1.2 RNAi: RI 261->198."
        ),
        source="Tseng et al. 2010 (J Neurosci 30:13192-13200)"
    ),
    IonChannelPerturbation(
        gene="SCN5A (hNaV1.5)",
        channel_type="Voltage-gated Na+",
        perturbation="mRNA injection (rescue of refractory period)",
        phenotype="RESCUES tail regeneration during refractory period",
        affected_system="regeneration",
        bioelectric_effect="Restores Na+ current -> triggers regeneration",
        quantitative_data="RI 10->39 (hNaV1.5 mRNA). Monensin (Na+ ionophore): RI 16.8->48.5",
        source="Tseng et al. 2010"
    ),

    # === EYE INDUCTION ===
    IonChannelPerturbation(
        gene="Kir6.1p (KATP dominant-negative)",
        channel_type="ATP-sensitive K+ (dominant-negative)",
        perturbation="Dominant-negative (depolarizes)",
        phenotype="Endogenous eye defects (25%) AND ectopic eye induction (~20%)",
        affected_system="eye",
        bioelectric_effect="Depolarization at eye field disrupts voltage pattern",
        quantitative_data="25% endogenous eye defects, ~20% ectopic eyes",
        source="Pai et al. 2012 (Development 139:313-323)"
    ),
    IonChannelPerturbation(
        gene="KCNA5 (Kv1.5)",
        channel_type="Voltage-gated K+",
        perturbation="Overexpression (hyperpolarizes)",
        phenotype="Rescues brain patterning defects, rescues ectopic eye defects",
        affected_system="brain/eye",
        bioelectric_effect="Restores hyperpolarization",
        quantitative_data="Rescues Notch misexpression brain defects",
        source="Pai et al. 2012; Pai et al. 2015"
    ),
    IonChannelPerturbation(
        gene="GlyR-alpha1",
        channel_type="Glycine-gated Cl-",
        perturbation="Ivermectin (agonist, depolarizes Cl- flux)",
        phenotype="Eye defects",
        affected_system="eye",
        bioelectric_effect="Depolarization via Cl- conductance increase",
        quantitative_data="48% eye defects with ivermectin",
        source="Pai et al. 2012"
    ),
    IonChannelPerturbation(
        gene="EXP1",
        channel_type="Cation channel",
        perturbation="Overexpression (depolarizes)",
        phenotype="Malformed eyes",
        affected_system="eye",
        bioelectric_effect="Depolarization",
        quantitative_data="46% malformed eyes",
        source="Pai et al. 2012"
    ),

    # === BRAIN PATTERNING ===
    IonChannelPerturbation(
        gene="Bir10",
        channel_type="Hyperpolarizing channel",
        perturbation="Overexpression",
        phenotype="Rescues Notch-induced brain patterning defects",
        affected_system="brain",
        bioelectric_effect="Restores hyperpolarization to neural tissue",
        quantitative_data="Brain defects: 19% with Bir10 rescue vs 46% Notch alone",
        source="Pai et al. 2015 (J Neurosci 35:4366)"
    ),
]


# =============================================================================
# Connexin (gap junction) atlas -- Landesman et al. 2003
# =============================================================================
# Xenopus has ~7 characterized connexins (fewer than zebrafish ~40,
# due to lack of teleost genome duplication).

CONNEXIN_ATLAS = {
    "Cx38": {
        "expression": "Maternal (dominant early connexin)",
        "tissues": ["ubiquitous_early"],
        "function": (
            "Redundant -- Cx38 depletion does NOT block GJC at 32-128 cell stages. "
            "Other connexins compensate."
        ),
        "lr_role": False,
        "source": "Landesman et al. 2003"
    },
    "Cx31": {
        "expression": "Maternal + zygotic",
        "tissues": ["broadly_expressed"],
        "function": "Redundancy with Cx38 for early coupling",
        "lr_role": False,
        "source": "Landesman et al. 2003"
    },
    "Cx43 (GJA1)": {
        "expression": "Maternal + zygotic",
        "tissues": ["neural", "cardiac", "ubiquitous"],
        "function": (
            "CRITICAL for LR asymmetry: forms circumferential dorsal path "
            "through which serotonin redistributes. Also neural crest chemotaxis."
        ),
        "lr_role": True,
        "source": "Landesman et al. 2003; Levin & Mercola 1998"
    },
    "Cx43.4": {
        "expression": "Maternal + zygotic (most abundant zygotic connexin)",
        "tissues": ["brain", "eyes", "spinal_cord"],
        "function": "Neural development, most abundant after MBT",
        "lr_role": False,
        "source": "Landesman et al. 2003"
    },
    "Cx26": {
        "expression": "Zygotic",
        "tissues": ["midline", "left_LPM"],
        "function": (
            "Required for LR cue transfer from MIDLINE to LEFT lateral plate "
            "mesoderm. Cx32 has no effect on laterality."
        ),
        "lr_role": True,
        "source": "Beyer et al. 2012 (PMC3507211)"
    },
    "Cx28.6": {
        "expression": "Transient (NF 22-26 only)",
        "tissues": ["unknown"],
        "function": "Unknown, only expressed transiently during tailbud stages",
        "lr_role": False,
        "source": "Landesman et al. 2003"
    },
    "Cx29": {
        "expression": "Throughout development",
        "tissues": ["endoderm", "lateral_mesoderm", "liver_anlage", "pronephros", "proctodeum"],
        "function": "Homologous to mouse Cx26/Cx30, broad endodermal expression",
        "lr_role": False,
        "source": "Landesman et al. 2003"
    },
}


# =============================================================================
# Tissue coupling strength (estimated from connexin expression)
# =============================================================================

TISSUE_COUPLING = {
    # Neural
    ("neural_plate", "neural_plate"): 0.7,
    ("brain", "brain"): 0.8,
    ("spinal_cord", "spinal_cord"): 0.7,
    ("retina", "retina"): 0.6,

    # Cardiac
    ("heart", "heart"): 0.9,           # Cardiac syncytium

    # Muscle
    ("skeletal_muscle", "skeletal_muscle"): 0.4,
    ("somite", "somite"): 0.6,

    # Notochord
    ("notochord", "notochord"): 0.8,

    # Epithelial
    ("epidermis", "epidermis"): 0.5,

    # Endoderm (Cx29)
    ("gut_endoderm", "gut_endoderm"): 0.5,
    ("liver_bud", "gut_endoderm"): 0.4,

    # Cross-tissue (weaker)
    ("neural_plate", "epidermis"): 0.2,
    ("somite", "notochord"): 0.3,
    ("neural_plate", "somite"): 0.2,
    ("epidermis", "somite"): 0.1,

    # LR-critical coupling (Cx43 path)
    ("right_blastomere", "left_blastomere"): 0.6,  # Via circumferential Cx43 path
    ("midline", "left_LPM"): 0.4,                  # Via Cx26
}


# =============================================================================
# Craniofacial bioelectric domains ("The Electric Face")
# =============================================================================

@dataclass
class CraniofacialBioelectricDomain:
    """A bioelectric domain in the developing face."""
    region: str
    polarity: str                # "hyper" or "depol"
    estimated_vmem_mv: float
    prospective_structure: str   # What this domain becomes
    critical_genes: List[str]    # Downstream genes affected by voltage
    critical_window_nf: Tuple[float, float]  # NF stage range


CRANIOFACIAL_DOMAINS = [
    CraniofacialBioelectricDomain(
        region="anterior_midline",
        polarity="hyper",
        estimated_vmem_mv=-55.0,
        prospective_structure="nose + mouth (splits from single domain)",
        critical_genes=["Otx2", "Six1", "Sox3", "FoxE4"],
        critical_window_nf=(11, 14),
    ),
    CraniofacialBioelectricDomain(
        region="bilateral_eye_field_L",
        polarity="hyper",
        estimated_vmem_mv=-61.0,  # ~10 mV more negative than neural plate
        prospective_structure="left eye",
        critical_genes=["Pax6", "Pax2", "Sox3"],
        critical_window_nf=(11, 14),
    ),
    CraniofacialBioelectricDomain(
        region="bilateral_eye_field_R",
        polarity="hyper",
        estimated_vmem_mv=-61.0,
        prospective_structure="right eye",
        critical_genes=["Pax6", "Pax2", "Sox3"],
        critical_window_nf=(11, 14),
    ),
    CraniofacialBioelectricDomain(
        region="surrounding_ectoderm",
        polarity="depol",
        estimated_vmem_mv=-15.0,
        prospective_structure="non-face epidermis",
        critical_genes=[],
        critical_window_nf=(11, 14),
    ),
    CraniofacialBioelectricDomain(
        region="branchial_arches",
        polarity="neutral",
        estimated_vmem_mv=-35.0,
        prospective_structure="jaw cartilage (Meckel's), branchial arches",
        critical_genes=["Slug", "Sox10", "Fz3", "FGF8"],
        critical_window_nf=(11, 14),
    ),
]

# KCNJ2 Andersen-Tawil syndrome mutation defect rates (Adams et al. 2016)
KCNJ2_DEFECT_RATES = {
    "uninjected_control": 0.14,
    "KCNJ2_WT_overexpression": 0.68,  # Hyperpolarizes
    "D71V_LOF": 0.60,                 # Depolarizes
    "T75R": 0.59,
    "T192A": 0.29,
    "R218W_LOF": 0.31,                # Depolarizes
    "Y242F_GOF": 0.43,                # Hyperpolarizes
}


# =============================================================================
# Left-right patterning mechanism (unique Levin contribution)
# =============================================================================

LR_MECHANISM = {
    "step_1_ion_asymmetry": {
        "stage": "NF 3 (4-cell)",
        "event": "H+/K+-ATPase and KCNQ1/KCNE1 localize to ventral right blastomere",
        "mechanism": "Maternal mRNA + cytoskeletal transport (microtubule + actin dependent)",
        "result": "Ventral right cell becomes hyperpolarized relative to left",
        "source": "Levin et al. 2002; Morokuma et al. 2008",
    },
    "step_2_serotonin_redistribution": {
        "stage": "NF 3-9 (4-cell to late blastula)",
        "event": "Serotonin (5-HT) redistributes through Cx43 gap junction path",
        "mechanism": "Circumferential dorsal path, electrophoresis through GJC",
        "result": "5-HT accumulates on RIGHT side",
        "source": "Fukumoto et al. 2005",
    },
    "step_3_epigenetic_repression": {
        "stage": "NF 9-10 (late blastula to early gastrula)",
        "event": "Right-side 5-HT represses Xnr-1 (Xenopus Nodal) via HDAC",
        "mechanism": "Serotonin -> HDAC -> epigenetic repression of Nodal on RIGHT",
        "result": "Xnr-1 expressed only on LEFT side",
        "source": "Fukumoto et al. 2005",
    },
    "step_4_signal_transfer": {
        "stage": "NF 10-12 (gastrula)",
        "event": "LR signal transfers from midline to left lateral plate mesoderm",
        "mechanism": "Cx26 gap junctions required (Cx32 has no effect)",
        "result": "Left LPM expresses Nodal -> Pitx2 -> organ laterality",
        "source": "Beyer et al. 2012",
    },
    "key_insight": (
        "This LR mechanism is CILIA-INDEPENDENT. It occurs BEFORE cilia form "
        "(NF 3-9). The ion channel / gap junction / serotonin pathway is the "
        "earliest known symmetry-breaking mechanism in any vertebrate."
    ),
}


# =============================================================================
# Tail regeneration bioelectric cascade
# =============================================================================

REGENERATION_CASCADE = {
    "0_hpa": {
        "event": "Tail amputation",
        "voltage": "Wound depolarizes (loss of epithelial barrier)",
        "channels": [],
    },
    "6_hpa": {
        "event": "V-ATPase upregulated at wound surface",
        "voltage": "Wound begins to repolarize",
        "channels": ["V-ATPase (ATP6V subunits)"],
    },
    "18_hpa": {
        "event": "NaV1.2 (SCN2A) expressed in regeneration bud mesenchyme",
        "voltage": "Na+ influx supports regeneration bud growth",
        "channels": ["V-ATPase", "NaV1.2"],
    },
    "24_hpa": {
        "event": "Regeneration bud repolarized, V-ATPase no longer required",
        "voltage": "Bud has restored normal voltage",
        "channels": ["NaV1.2", "various"],
    },
    "48_hpa_plus": {
        "event": "Regeneration bud patterning and outgrowth",
        "voltage": "Normal tissue voltage profiles restoring",
        "channels": ["NaV1.2", "various developmental channels"],
    },
    "rescue_refractory": {
        "event": "hNaV1.5 mRNA or monensin rescues NF 45-47 refractory tails",
        "voltage": "Forced Na+ current restores regeneration competence",
        "channels": ["hNaV1.5 (injected)", "monensin (Na+ ionophore)"],
        "quantitative": "hNaV1.5: RI 10->39. Monensin 20uM: RI 16.8->48.5",
    },
}


# =============================================================================
# Germ layer voltage assignments (used by bioelectric_development.py)
# =============================================================================

GERM_LAYER_VMEM = {
    "ectoderm": -70.0,    # Levin: strongly hyperpolarized (neural fate)
    "mesoderm": -30.0,    # Levin: moderately depolarized
    "endoderm": -20.0,    # Levin: depolarized
}

# Maps to bioelectric_development.py constants:
# LEVIN_ECTODERM_VMEM = -70.0
# LEVIN_MESODERM_VMEM = -30.0
# LEVIN_ENDODERM_VMEM = -20.0


# =============================================================================
# Convenience functions
# =============================================================================

def get_tissue_vmem(tissue: str) -> float:
    """Get estimated resting potential for a tissue type."""
    return TISSUE_VMEM_ESTIMATES.get(tissue, -40.0)


def get_stage_bioelectric(nf_stage: float) -> Optional[XenopusBioelectricStage]:
    """Get bioelectric data for a Nieuwkoop-Faber stage."""
    closest = None
    min_diff = float('inf')
    for stage in DEVELOPMENTAL_VOLTAGE_ATLAS:
        diff = abs(stage.nf_stage - nf_stage)
        if diff < min_diff:
            min_diff = diff
            closest = stage
    return closest


def get_stage_by_hpf(hpf: float) -> Optional[XenopusBioelectricStage]:
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


def get_perturbations_for_system(system: str) -> List[IonChannelPerturbation]:
    """Get all ion channel perturbations affecting a given system.

    Args:
        system: One of 'LR', 'craniofacial', 'regeneration', 'eye', 'brain'
    """
    return [p for p in ION_CHANNEL_PERTURBATIONS
            if system.lower() in p.affected_system.lower()]


def get_craniofacial_domains() -> List[CraniofacialBioelectricDomain]:
    """Get all craniofacial bioelectric domains."""
    return CRANIOFACIAL_DOMAINS


def get_lr_mechanism() -> Dict:
    """Get the left-right patterning mechanism."""
    return LR_MECHANISM


def get_regeneration_cascade() -> Dict:
    """Get the tail regeneration bioelectric cascade."""
    return REGENERATION_CASCADE


def print_bioelectric_summary():
    """Print summary of all Xenopus bioelectric data."""
    print("=" * 70)
    print("XENOPUS LAEVIS BIOELECTRIC TISSUE CHARACTERIZATION")
    print("=" * 70)

    print(f"\n--- Developmental Voltage Atlas ({len(DEVELOPMENTAL_VOLTAGE_ATLAS)} stages) ---")
    print("    Source: Levin lab (Tufts), multiple papers 2002-2016")
    print("    CALIBRATED: Neural plate -51 mV (Pai et al. 2015, patch-clamp)")
    for stage in DEVELOPMENTAL_VOLTAGE_ATLAS:
        n_tissues = len(stage.tissue_polarity)
        print(f"\n  {stage.stage_name:25s} (NF{stage.nf_stage:5.1f}, {stage.hpf:5.1f} hpf, {stage.period})")
        print(f"    Pattern: {stage.voltage_pattern}")
        for tissue, polarity in stage.tissue_polarity.items():
            print(f"      {tissue:30s}: {polarity}")

    print(f"\n--- Tissue Vmem Estimates ({len(TISSUE_VMEM_ESTIMATES)} tissues) ---")
    for tissue, vmem in sorted(TISSUE_VMEM_ESTIMATES.items(), key=lambda x: x[1]):
        print(f"  {tissue:35s}: {vmem:6.1f} mV")

    print(f"\n--- Ion Channel Perturbations ({len(ION_CHANNEL_PERTURBATIONS)} entries) ---")
    for p in ION_CHANNEL_PERTURBATIONS:
        print(f"  {p.gene:30s} ({p.channel_type})")
        print(f"    System: {p.affected_system}, Effect: {p.bioelectric_effect[:60]}")
        print(f"    Data: {p.quantitative_data[:60]}")

    print(f"\n--- Connexin Atlas ({len(CONNEXIN_ATLAS)} connexins) ---")
    for cx, info in CONNEXIN_ATLAS.items():
        tissues = ", ".join(info["tissues"])
        lr = " [LR CRITICAL]" if info["lr_role"] else ""
        print(f"  {cx:15s}: {tissues}{lr}")

    print(f"\n--- Craniofacial Domains ({len(CRANIOFACIAL_DOMAINS)} regions) ---")
    for domain in CRANIOFACIAL_DOMAINS:
        print(f"  {domain.region:30s}: {domain.polarity:5s} ({domain.estimated_vmem_mv:+.0f} mV)")
        print(f"    -> {domain.prospective_structure}")

    print(f"\n--- Left-Right Mechanism (4 steps, cilia-independent) ---")
    for step_key in ["step_1_ion_asymmetry", "step_2_serotonin_redistribution",
                     "step_3_epigenetic_repression", "step_4_signal_transfer"]:
        step = LR_MECHANISM[step_key]
        print(f"  {step_key}: {step['event'][:60]}")

    print(f"\n--- Tail Regeneration Cascade ---")
    for time_key in ["0_hpa", "6_hpa", "18_hpa", "24_hpa"]:
        entry = REGENERATION_CASCADE[time_key]
        print(f"  {time_key:10s}: {entry['event'][:60]}")

    print(f"\n--- Key Bioelectric Insights ---")
    print("  Xenopus has the ONLY calibrated embryonic Vmem (-51 mV, Pai 2015)")
    print("  LR asymmetry is CILIA-INDEPENDENT (ion channels + gap junctions)")
    print("  'The Electric Face' prepattern at NF 19 defines face territories")
    print("  Tail regeneration requires V-ATPase -> NaV1.2 cascade")
    print("  7 connexins characterized (vs 40+ in zebrafish, 25 innexins in C. elegans)")
    print("  Craniofacial critical window: NF 11-14 ONLY")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_bioelectric_summary()
