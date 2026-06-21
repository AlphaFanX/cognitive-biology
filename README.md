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

It also contains the code for the cross-phylum companion paper, *Cognitive
Biology Across Phyla: One Bioelectric Operator, Many Body Plans, and the
Substrate-Reader That Sets the Boundary* (see the
[Cross-phylum validation](#cross-phylum-validation-companion-paper) section).

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

> Jacobs, M. B. (2026). *Cognitive Biology Across Phyla: One Bioelectric Operator,
> Many Body Plans, and the Substrate-Reader That Sets the Boundary.* Zenodo.
> https://doi.org/10.5281/zenodo.20746637

## License

Code: **MIT** (see [`LICENSE`](LICENSE)). Manuscript text and figures
(`paper/`): **CC-BY 4.0**.
