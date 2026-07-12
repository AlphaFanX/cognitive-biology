# Cognitive Biology

Code accompanying the paper:

> **Cognitive Biology: From Systems to Somatic Networks, Tissue Cognition,
> Cognitive Morphogenesis, and Tissue Emotions**
> Miles B. Jacobs (genetec.io, Cape Town, South Africa)
> Zenodo, 2026. DOI: [10.5281/zenodo.20722139](https://doi.org/10.5281/zenodo.20722139)
> Manuscript PDF: [`paper/cognitive_biology_paper_v2.pdf`](paper/cognitive_biology_paper_v2.pdf)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20722139.svg)](https://doi.org/10.5281/zenodo.20722139)

This repository contains a curated, reproducible subset of the computational
framework described in the paper. It is **not** the full working tree — only
the modules that implement the mechanisms and generate the results discussed in
the manuscript, with no API keys, credentials, or large private data.

It also contains the code for six companion papers: the design paper,
*Cognitive Biology: Perceptrons and Morphogen Primordia* (see
[Perceptrons and morphogen primordia](#perceptrons-and-morphogen-primordia-companion-paper));
the cross-phylum paper, *Cognitive Biology Across Phyla: One Bioelectric
Operator, Many Body Plans, and the Substrate-Reader That Sets the Boundary* (see
[Cross-phylum validation](#cross-phylum-validation-companion-paper)); the
organ-formation paper, *Cognitive Biology and Organ Formation: Organs as
Master-Transcription-Factor Heads Read From an Accessibility Code* (see
[Organ formation](#organ-formation-companion-paper)); the two-perceptron paper,
*Cognitive Biology: The Inner Perceptron versus the Outer Perceptron* (see
[Inner and outer perceptron](#the-inner-and-outer-perceptron-companion-paper));
the embryo-computation paper, *Cognitive Biology: Computing the Embryo* (see
[Computing the embryo](#computing-the-embryo-companion-paper)); and the
differentiation-clock paper, *Cognitive Biology: Differentiation Clocks, Organ
Formation and the MLP* (see
[Differentiation clocks, organ formation and the MLP](#differentiation-clocks-organ-formation-and-the-mlp-companion-paper)).

## The idea in one paragraph

Living matter is modeled as an ensemble of adaptive automata that compute,
represent, and pursue goals. Each cell is a **Cellutron** — a hybrid
Mealy–Moore state machine driven by transcriptional attention (genetic
Query/Key/Value), ion-channel admittance, and signalling. Development is
initialized by a **telomere-gated chromosomal clock** that reads a stratified,
methylation-tagged **zygote kernel** (a hypomethylated developmental-enhancer
fossil record), and morphogenesis is realized as an expanded **Neural Cellular
Automaton** with a language model over transcriptional attention, bioelectric
(BETSE) guidance, and low-rank adaptation. The headline demonstration: a
recognizable vertebrate **phylotypic body plan computes from the inherited
zygote kernel alone**, with adult/species/individual identity entering only as
a low-rank adaptation on top.

## Repository map (code → paper)

| Module | Role | Paper |
|--------|------|-------|
| `medic/genome/zygote_kernel.py` | Zygote kernel = stratified methylation fossil record (embryonic/fetal/adult), PRC2 reverse-order reader | §2.4 |
| `medic/genome/telomere_clock.py` | Telomere-gated chromosomal clock; Hox temporal colinearity | §2.2 |
| `medic/genome/real_kernel.py`, `embryonic_methylation.py` | Build the real kernel from Jadhav methylation tracks | §2.4 |
| `medic/four_head_morphogenesis.py` | `GenomicChannelLookup` (compact genomic hardcoding) + three kernel heads (Migration/Fate/Division) | §3.1 |
| `medic/tissue/genomic_nca.py` | Genome-conditioned NCA; NCA ≡ TRM(LLM) correspondence | §3.1 |
| `medic/tissue/phylotypic_form.py` | **Headline result:** vertebrate phylotypic form from the zygote kernel alone | §3.4 |
| `medic/body_plan_morphogenesis.py` | Bauplan scaffold (axes, primordia) | §3.4 |
| `medic/genomic_attention.py` | Transcriptional Query/Key/Value attention computer | §2.3 |
| `medic/se_guided_differentiation.py` | Super-enhancer-guided cell-fate (softmax over SE attention) | §2.3 |
| `medic/zebrafish_somitogenesis.py` | her1 segmentation clock + wavefront → somites (S = v·T) | §3.4 |
| `medic/zebrafish_embryo.py`, `zebrafish_movie.py` | Genome→embryo growth; open-ended power-law proliferation | §3.4 |
| `medic/embryo_montage.py` | Multi-stage embryo montage figure | §3.4, Fig. |
| `medic/silic_validation.py` | Validation against the 12-stage Silic developmental atlas | §3.4 |
| `medic/zebrafish_bioelectric.py`, `xenopus_bioelectric.py` | Developmental voltage atlases (Levin-style Vmem maps) | §2.3 |
| `medic/genome/abc_client.py`, `ep_interface.py`, `ucsc_client.py` | Activity-by-Contact / enhancer–promoter conditioning interfaces | §3.6 |
| `backend/app_developmental.py` | Developmental simulation service (FastAPI) | — |

## Perceptrons and morphogen primordia (companion paper)

These modules implement the design companion paper:

> **Cognitive Biology: Perceptrons and Morphogen Primordia**
> Miles B. Jacobs (genetec.io, Cape Town, South Africa)
> Zenodo, 2026. DOI: [10.5281/zenodo.21143761](https://doi.org/10.5281/zenodo.21143761)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21143761.svg)](https://doi.org/10.5281/zenodo.21143761)

The layer-resolved "glass-box" kernel, the GWAS magnitude (cis-LoRA) layer, and
morphogen **primordia** placed as operators on the bioelectric field — the
electric face and trunk from a symmetric positional frame.

| Module | Role |
|--------|------|
| `medic/glass_box_kernel.py`, `glass_box_kernel_3d.py` | Layer-resolved glass-box kernel (every layer has an assay) |
| `medic/magnitude_layer.py`, `magnitude_layer_singlecell.py` | GWAS magnitude / cis-LoRA adapter layer |
| `medic/primordium_operator.py` | Morphogen-primordium operator on the gap-junction field |
| `medic/trained_kernel_head.py` | Trained kernel head |
| `medic/heart_primordium_3d.py`, `gut_primordium_3d.py` | 3-D organ primordia |
| `medic/alphagenome_morphogen_embryonic.py`, `alphagenome_morphogen_rd.py` | AlphaGenome-conditioned morphogen fronts |
| `medic/conductance_gwas_test.py`, `gtex_forward_test.py` | GWAS conductance test; GTEx forward validation |
| `face_demo/face_morphogenesis.py`, `face_nca_growth.py` | Electric-face morphogenesis and NCA growth |
| `face_demo/morphogen_orientation.py`, `primordium_placement.py`, `trunk_placement.py` | Symmetric morphogen orientation and primordium/trunk placement |

## Cross-phylum validation (companion paper)

These modules implement the cross-phylum companion paper:

> **Cognitive Biology Across Phyla: One Bioelectric Operator, Many Body Plans,
> and the Substrate-Reader That Sets the Boundary**
> Miles B. Jacobs (genetec.io, Cape Town, South Africa)
> Zenodo, 2026. DOI: [10.5281/zenodo.20746637](https://doi.org/10.5281/zenodo.20746637)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20746637.svg)](https://doi.org/10.5281/zenodo.20746637)

One Goldman/gap-junction operator validated across phyla on distinct
morphogenetic readouts, the kernel/reader decoupling, and the body-plan generator.

| Module | Role |
|--------|------|
| `medic/crossphylum_validation.py` | Combined scorecard runner (one operator, many readouts) |
| `medic/planaria_bioelectric.py` | Planarian axial polarity (8/8 regeneration outcomes; no genomic kernel) |
| `medic/zebrafish_vm_validation.py` | Zebrafish fin growth (7/7 Vm-direction) |
| `medic/xenopus_vm_validation.py` | Xenopus craniofacial "electric face" (bidirectional, ρ=+0.83) |
| `medic/insect_bioelectric.py` | Holometaboly + the Drosophila wing-disc Vm handle |
| `medic/genome/kernel_reader.py` | Shared reader interface (methylation and ATAC accessibility) |
| `medic/body_plan_generator.py` | Body-plan topology from (heads, masks, softmaxes) + Vm axis |
| `medic/annelid_kernel.py` | Annelid (Capitella) dual-reader concordance across the bilaterian split |
| `medic/clade_ladder.py` | Clade-ladder breadth program |
| `medic/nematostella_embryo.py`, `capitella_embryo.py`, `planaria_embryo.py` | Development-from-the-genome montages |
| `medic/nematostella_se_heads.py`, `annelid_se_heads.py` | Genome-derived head repertoires via super-enhancer calling on real ATAC (N_SE = 643 / 949) |

## Organ formation (companion paper)

These modules implement the organ-formation companion paper:

> **Cognitive Biology and Organ Formation: Organs as Master-Transcription-Factor
> Heads Read From an Accessibility Code**
> Miles B. Jacobs (genetec.io, Cape Town, South Africa)
> Zenodo, 2026. DOI: [10.5281/zenodo.20925727](https://doi.org/10.5281/zenodo.20925727)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20925727.svg)](https://doi.org/10.5281/zenodo.20925727)

Organs as separable, conserved master-transcription-factor **heads** read from
the super-enhancer/accessibility landscape (not from enhancer sequence), and the
bioelectric substrate's **field–form correspondence** (the gap-junction
operator's cymatic modes coincide with facial, cardiac and gut geometry).

| Module | Role |
|--------|------|
| `medic/organ_cascade.py` | **Stage 1:** organ Fate heads — ABC super-enhancers recover each organ's master-TF kernel (recall@25 23.9% vs 4.3% null, *p* = 0.0005) |
| `medic/organ_cascade_wiring.py`, `_v2.py`, `_v3.py` | **Stage 2:** TF→super-enhancer motif wiring (presence / enrichment / cluster-density) — the negative: organ identity is not in enhancer sequence |
| `medic/organ_cascade_combinatorial.py` | "TF words" bag-of-motifs organ classifier (≈ chance) |
| `medic/organ_cascade_kmer.py` | 6-mer sequence-grammar classifier (faint, borderline) |
| `medic/organ_cascade_process_heads.py` | Migration + Division heads by cell-state contrast (SNAI2 first, *p* = 0.0007; Division MYC-only) |
| `medic/kernel_nca_stress.py` | Bioelectric substrate: settling wave, ablation recovery, screening length, cymatic eigenmodes, stability bound |
| `medic/nca_abc_modes.py` | NCA cymatic modes from the ABC gap-junction operator; low-rank target; NCA↔FaceBase mode match |
| `medic/organ_modes.py` | Field–form correspondence for heart (LV) and gut (tube) electrical syncytia |
| `medic/bolster_electric_tests.py` | Tests 1–3: forward un-anchored Vm, shuffle null, channel-class ablation |
| `medic/craniofacial_stability.py` | Test 4: CS13–CS17 craniofacial enhancer stability across morphogenesis |
| `face_demo/face_eigenmodes.py` | FaceBase mesh cleaning + cotangent Laplace–Beltrami geometry modes |
| `face_demo/electric_face_correspondence.py` | Gap-junction field vs facial-geometry shared eigenbasis (ρ = 1.00) |
| `face_demo/mesh_morph.py` | FaceBase mean-mesh loader (genome→face morph helper) |

## The inner and outer perceptron (companion paper)

These modules implement the two-perceptron companion paper:

> **Cognitive Biology: The Inner Perceptron versus the Outer Perceptron**
> Miles B. Jacobs (genetec.io, Cape Town, South Africa)
> Zenodo, 2026. DOI: [10.5281/zenodo.21143016](https://doi.org/10.5281/zenodo.21143016)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21143016.svg)](https://doi.org/10.5281/zenodo.21143016)

The inner (NCA) versus outer (Large Genomic Model) perceptron split, with every
coefficient traced to a genome × literature equation, plus surgery-free
bioelectric interventions.

| Module | Role |
|--------|------|
| `medic/perceptron_trace.py` | Two-perceptron (inner NCA / outer LGM) traceability + the three tables |
| `medic/ectopic_eye.py` | Bioelectric ectopic-eye demonstration (Pai/Levin 2012) |
| `medic/two_headed_planaria.py` | Two-headed planarian from a gap-junction perturbation |
| `medic/limb_inverse_design.py` | Limb inverse-design demonstration |
| `medic/morphogen_rd.py` | Morphogen reaction–diffusion organs |
| `medic/nca_vertebrate_3d.py` | Forward Bauplan + 3-D ectopic body (`--ectopic`) |
| `medic/motility_clock.py` | FGF/ERK motility clock |
| `medic/zebrafish_somitogenesis.py` | Segmentation clock (shared with the foundational paper) |
| `medic/bioelectric_development.py` | Goldman/conductance read (shared) |

## Computing the embryo (companion paper)

These modules implement the embryo-computation companion paper:

> **Cognitive Biology: Computing the Embryo**
> Miles B. Jacobs (genetec.io, Cape Town, South Africa)
> Zenodo, 2026. DOI: [10.5281/zenodo.21221930](https://doi.org/10.5281/zenodo.21221930)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21221930.svg)](https://doi.org/10.5281/zenodo.21221930)

The four morphogenetic heads (differentiation, division, migration, shape)
learned on real spatiotemporal atlases and driven forward as a single
grow-from-one-cell embryo, plus the "electric body" positional frame and the
lateral-inhibition mechanism behind the symmetric eye pair and cyclopia.

| Module | Role |
|--------|------|
| `medic/silic_train.py`, `zesta_train.py` | Spatial-NCA training on the Silic and ZESTA (Stereo-seq) atlases |
| `medic/zesta_temporal_4d.py` | Temporal 4-D fit: the field condenses out of the blastula (Moran 0→0.94) |
| `medic/gj_operator_train.py` | Genome-derived gap-junction operator (the connexin-domain null) |
| `medic/shareseq_diff_head.py`, `shareseq_div_head.py`, `shareseq_mig_head.py`, `shareseq_shape_head.py` | The four heads fit as hypernetwork weights on SHARE-seq skin |
| `medic/differentiation_clock.py` | Telomere→PRC2-withdrawal + Hox clock: WHEN each fate unlocks (ρ = 0.95) |
| `medic/division_head.py` | Proliferation program → division→telomere→PRC2 clock (ρ = 0.94) |
| `medic/migration_head.py` | Motility program + convergent extension computed forward in 3-D (×3.2 elongation) |
| `medic/shape_head.py` | Constriction/adhesion program → neural-tube closure and lip fusion; knockdown → NTD/harelip |
| `medic/mechanical_fusion.py` | Folding as a buckling eigenmode; cleft ⟺ Fiedler λ₂ = 0 |
| `medic/growing_domain.py` | Grow-from-one-cell domain substrate |
| `medic/unified_embryo.py`, `unified_embryo_figure.py` | All four heads intercalated per timestep in one forward pass |
| `medic/symmetric_embryo_figure.py` | Bilaterally symmetric embryo hero figure |
| `medic/electric_body_frame.py` | Low gap-junction eigenmodes ARE the body axes (AP 0.96 / DV 0.77 / LR = midline) |
| `medic/electric_face_feedback.py`, `field_driven_eye.py` | Frame + lateral inhibition → symmetric eye pair; weak frame → cyclopia |
| `medic/morphogenesis_failures.py`, `gill_covering.py` | Failure-mode and covering demonstrations |

## Differentiation clocks, organ formation and the MLP (companion paper)

These modules implement the differentiation-clock companion paper:

> **Cognitive Biology: Differentiation Clocks, Organ Formation and the MLP**
> Miles B. Jacobs (genetec.io, Cape Town, South Africa)
> Zenodo, 2026. DOI: [10.5281/zenodo.21322049](https://doi.org/10.5281/zenodo.21322049)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21322049.svg)](https://doi.org/10.5281/zenodo.21322049)

Identity (what a cell becomes) separated from timing (when) and coupled by one
clock shared by two MLP heads: the division head drives it (division→telomere→
PRC2 withdrawal) and the differentiation head reads it. A mechanism clock and a
measured clock (ENCODE mouse fetal atlas) time cell types across organs, and the
organ head's four effectors (connexin frame, morphogen reaction-diffusion,
Delta–Notch lateral inhibition, cadherin cohesion) assemble the organ, closing
in a genome-derived vertebrate grown from a single cell.

| Module | Role |
|--------|------|
| `medic/differentiation_clock.py` | The mechanism clock (telomere→PRC2 + Hox); adds the guarded measured multi-organ clock |
| `medic/encode_opening_time_clock.py`, `encode_temporal_index.py` | Measured clock: each cell type timed by its marker-enhancer opening (ENCODE fetal atlas), 38 cell types across 11 organs |
| `medic/organ_execution.py` | Integrated per-organ forward pass: connexin frame → morphogen spacing → lateral-inhibition discretization → cadherin cohesion |
| `medic/hox_boundary.py` | Anterior-posterior address read from the genome (single-cell Hox posterior boundary, ρ = 0.81) |
| `medic/placement_3d.py` | Organ placement on the 3-D electric-body frame (gap-junction eigenmodes = body axes) |
| `medic/vertebrate_growth.py` | The assembled vertebrate (whole + cutaway of the derived organs) |
| `medic/vertebrate_grand.py` | The grand single-panel vertebrate render (~20k cells, limbs, organs) |

## Install

```bash
python -m pip install -r requirements.txt   # Python 3.11+
```

## Quick start (no external data needed)

The pure-dynamics demonstrations run from public constants in the source:

```bash
python -m medic.zebrafish_somitogenesis      # segmentation clock + somite definition
python -m medic.silic_validation             # Silic 12-stage validation summary
python -m medic.embryo_montage               # genome->embryo montage figure
python -m medic.crossphylum_validation       # cross-phylum operator scorecard (companion paper)
```

Genome-conditioned modules (kernel construction, SE/ABC attention) additionally
require the public datasets listed in [`data/README.md`](data/README.md)
(GEO GSE111024, super-enhancer atlas, ABC predictions). No credentials are
required for anything in this repository.

## Data availability

See [`data/README.md`](data/README.md). Large conditioning datasets are public
but not redistributed here; obtain them from the cited sources.

## Citing

See [`CITATION.cff`](CITATION.cff).

> Jacobs, M. B. (2026). *Cognitive Biology: From Systems to Somatic Networks,
> Tissue Cognition, Cognitive Morphogenesis, and Tissue Emotions.* Zenodo.
> https://doi.org/10.5281/zenodo.20722139

> Jacobs, M. B. (2026). *Cognitive Biology: Perceptrons and Morphogen Primordia.*
> Zenodo. https://doi.org/10.5281/zenodo.21143761

> Jacobs, M. B. (2026). *Cognitive Biology Across Phyla: One Bioelectric Operator,
> Many Body Plans, and the Substrate-Reader That Sets the Boundary.* Zenodo.
> https://doi.org/10.5281/zenodo.20746637

> Jacobs, M. B. (2026). *Cognitive Biology and Organ Formation: Organs as
> Master-Transcription-Factor Heads Read From an Accessibility Code.* Zenodo.
> https://doi.org/10.5281/zenodo.20925727

> Jacobs, M. B. (2026). *Cognitive Biology: The Inner Perceptron versus the
> Outer Perceptron.* Zenodo. https://doi.org/10.5281/zenodo.21143016

> Jacobs, M. B. (2026). *Cognitive Biology: Computing the Embryo.* Zenodo.
> https://doi.org/10.5281/zenodo.21221930

> Jacobs, M. B. (2026). *Cognitive Biology: Differentiation Clocks, Organ
> Formation and the MLP.* Zenodo. https://doi.org/10.5281/zenodo.21322049

## License

Code: **MIT** (see [`LICENSE`](LICENSE)). Manuscript text and figures
(`paper/`): **CC-BY 4.0**.
