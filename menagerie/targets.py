"""
Reference genomes + display colours for the African Big 5 / 7, and the true phylogeny.

Each reference genome is curated so its decoded proportions match the real animal (the knob
values ARE the species' low-rank adapter on the base vertebrate). Evolution (evolve.py) then
searches from the base vertebrate to recover a genome whose phenotype matches the reference's,
demonstrating that each species is reachable by a low-rank move -- and how many knobs it takes.

Morphometric anchors (approximate, from standard references): giraffe neck ~2.0-2.2 m over the
SAME 7 cervical vertebrae; elephant graviportal columnar limbs + tusks + trunk; rhino single
median horn (on the midline = the left-right eigenmode node); buffalo paired lateral horns
(a frame + lateral-inhibition pair); crocodile long snout + long tail + sprawling posture (a
deeper, non-mammalian adapter).
"""
from __future__ import annotations
from .genome import Genome

# species -> (reference Genome, display base-colour rgb)
REFERENCE = {
    "base_vertebrate": (Genome(), (0.74, 0.62, 0.47)),

    "lion": (Genome(
        body_size=1.30, trunk_girth=1.20, trunk_len=1.05, tail_len=1.7, tail_count=1.1,
        limb_len=1.05, limb_gracility=1.05, skull_size=1.20, snout_len=0.85,
        coat_type="plain", coat_wavelength=1.0), (0.80, 0.66, 0.42)),

    "leopard": (Genome(
        body_size=0.95, trunk_girth=1.00, tail_len=1.9, tail_count=1.15,
        limb_len=1.00, limb_gracility=0.85, skull_size=1.00, snout_len=0.80,
        coat_type="rosette", coat_wavelength=0.75), (0.86, 0.72, 0.40)),

    "buffalo": (Genome(
        body_size=1.60, trunk_girth=1.60, trunk_len=1.10, tail_len=1.1,
        limb_len=0.95, limb_gracility=1.55, skull_size=1.30, snout_len=0.95,
        horn_mode="paired", horn_size=1.6, coat_type="plain",
        coat_wavelength=1.0), (0.22, 0.20, 0.19)),

    "elephant": (Genome(
        body_size=2.60, cervical_elong=0.90, trunk_girth=1.95, trunk_len=1.10,
        tail_len=1.2, limb_len=1.20, limb_gracility=2.35, skull_size=1.85, snout_len=0.85,
        dentition="tusks", tusk_len=1.8, nose="trunk", proboscis_len=2.3,
        coat_type="plain"), (0.55, 0.55, 0.57)),

    "rhino": (Genome(
        body_size=2.15, cervical_elong=0.85, trunk_girth=1.80, trunk_len=1.05,
        tail_len=0.9, limb_len=0.90, limb_gracility=2.00, skull_size=1.50, snout_len=1.20,
        horn_mode="single_median", horn_size=1.7, coat_type="plain"), (0.55, 0.53, 0.50)),

    "giraffe": (Genome(
        body_size=1.30, cervical_elong=2.70, trunk_len=1.00, trunk_girth=1.05,
        tail_len=1.4, limb_len=2.05, limb_gracility=0.95, skull_size=1.05, snout_len=1.10,
        neck_raise=0.92, horn_mode="paired", horn_size=0.30,
        coat_type="patch", coat_wavelength=1.0), (0.80, 0.60, 0.32)),

    "crocodile": (Genome(
        body_size=1.55, cervical_elong=0.80, trunk_len=1.30, trunk_girth=1.05,
        tail_len=2.60, tail_count=1.80, limb_len=0.50, limb_gracility=1.10,
        skull_size=1.35, snout_len=2.60, neck_raise=0.0, posture="sprawling",
        coat_type="armor", coat_wavelength=1.1), (0.30, 0.36, 0.27)),

    # --- Homo sapiens: bipedal, big brain, flat face, long legs, vestigial tail ---
    "human_male": (Genome(
        body_size=1.15, cervical_elong=1.05, trunk_len=0.95, trunk_girth=1.05,
        tail_len=0.1, tail_count=0.3, limb_len=1.75, limb_gracility=1.20,
        skull_size=1.55, snout_len=0.45, neck_raise=1.0,
        posture="biped", sex="male", coat_type="plain"), (0.78, 0.58, 0.45)),

    "human_female": (Genome(
        body_size=1.02, cervical_elong=1.05, trunk_len=0.95, trunk_girth=0.98,
        tail_len=0.1, tail_count=0.3, limb_len=1.68, limb_gracility=0.92,
        skull_size=1.50, snout_len=0.45, neck_raise=1.0,
        posture="biped", sex="female", coat_type="plain"), (0.86, 0.66, 0.54)),

    # --- the two organisms the NCA+LGM model is TRAINED on (MOSTA / ZESTA atlases) ---
    "mouse": (Genome(               # small quadruped rodent (Mus musculus)
        body_size=0.30, cervical_elong=0.70, trunk_len=1.10, trunk_girth=0.85,
        tail_len=2.20, tail_count=1.80, limb_len=0.75, limb_gracility=0.60,
        skull_size=1.00, snout_len=1.25, neck_raise=0.20,
        coat_type="plain"), (0.58, 0.52, 0.48)),

    "zebrafish": (Genome(           # finned fish (Danio rerio): fins = the SAME limb module
        body_size=0.50, cervical_elong=0.60, trunk_len=1.40, trunk_girth=0.75,
        tail_len=1.60, tail_count=1.30, limb_len=0.60, limb_gracility=0.45,
        skull_size=1.15, snout_len=0.85, neck_raise=0.0,
        convergent_ext=1.45,        # HIGH Wnt-PCP: strong CE -> narrow body -> no LR eigenmode -> finned/limbless
        body_plan="finned", coat_type="stripe"), (0.55, 0.63, 0.76)),
}

BIG_FIVE = ["lion", "leopard", "buffalo", "elephant", "rhino"]
BIG_SEVEN = BIG_FIVE + ["giraffe", "crocodile"]
HUMANS = ["human_male", "human_female"]

# true clades, for validating the genome-space morphospace
CLADE = {
    "lion": "Felidae", "leopard": "Felidae",
    "buffalo": "Bovidae", "giraffe": "Giraffidae",
    "elephant": "Proboscidea", "rhino": "Perissodactyla",
    "crocodile": "Crocodylia",   # the non-mammal outgroup
    "mouse": "Rodentia", "zebrafish": "Actinopterygii",   # the training organisms
}


def reference_genome(species: str) -> Genome:
    return REFERENCE[species][0]


def display_color(species: str):
    return REFERENCE[species][1]
