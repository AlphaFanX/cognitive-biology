# Data availability

The simulation and figure code in this repository is provided in full. The
*conditioning data* it consumes are large public datasets that are **not
redistributed here** (size and licensing). Obtain them from the original
sources and place them under `data/` as indicated. The pure-dynamics demos
(segmentation clock, embryo growth, phylotypic-form scaffold) run **without**
these datasets; only the genome-conditioned modules require them.

| Dataset | Used by | Source |
|---------|---------|--------|
| **Jadhav developmental-enhancer methylation** (WGBS E12.5/E16.5/Adult + H3K4me1/H3K27ac) | `medic/genome/zygote_kernel.py`, `medic/genome/real_kernel.py`, `medic/genome/embryonic_methylation.py` | GEO **GSE111024** → `data/jadhav_mouse/` |
| **Super-enhancer atlas** (`SE.sqlite`) | `medic/se_guided_differentiation.py`, `medic/genomic_attention.py` | SEdb / dbSUPER (build SQLite from the published SE.bed) |
| **Activity-by-Contact enhancer–promoter predictions** (`AllPredictions.ABC.txt.gz`) | `medic/genome/abc_client.py`, `medic/genome/ep_interface.py` | ABC model predictions (Nasser et al. 2021) → `data/enhancer_promoter/` |
| **Additional GEO tracks** | kernel construction | GEO **GSE115541** |

Notes:
- The bioelectric voltage constants used by the zebrafish/Xenopus modules
  (`zebrafish_bioelectric.py`, `xenopus_bioelectric.py`) are encoded directly
  in source as published estimates — no download required.
- No API keys are needed for the modules in this repository. Front-end
  sequence-model access (AlphaGenome) is **not** part of this paper's code and
  is intentionally excluded.
