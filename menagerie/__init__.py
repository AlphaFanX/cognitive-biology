"""
menagerie -- evolve the African Big 5/7 from the basic vertebrate.

An artificial-life morphology system grounded in Cognitive Biology: ONE developmental
engine (a conserved tetrapod Bauplan) decoded from a shared genome space, where each
species is a LOW-RANK set of knobs (a LoRA) on the base vertebrate. Every knob is tied to
a real developmental mechanism / gene, not an arbitrary encoding.

  genome.py    the LoRA knobs (each annotated with mechanism + gene)
  decoder.py   the conserved Bauplan the knobs deform -> parcels + morphometrics
               (7 cervical vertebrae are FROZEN; elongation is per-segment scale)
  targets.py   real morphometrics + reference genomes for lion/leopard/buffalo/
               elephant/rhino/giraffe/crocodile
  evolve.py    a (mu,lambda) evolution strategy: base vertebrate -> each species
  figure_menagerie.py   decode all 7, render the montage + morphospace + evolution
"""
