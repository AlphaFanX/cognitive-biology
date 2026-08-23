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
| **Activity-by-Contact enhancer–promoter predictions** (`AllPredictions.ABC.txt.gz`) | `medic/genome/abc_client.py`, `medic/genome/ep_interface.py`, `medic/organ_cascade.py` | ABC model predictions (Nasser et al. 2021) → `data/enhancer_promoter/` |
| **Human transcription-factor list** (`human_tfs_lambert2018.txt`) | `medic/organ_cascade*.py` | Lambert et al. 2018 → `data/` |
| **JASPAR 2024 CORE vertebrate PFMs** | `medic/organ_cascade_wiring*.py` | JASPAR 2024 → `data/JASPAR2024_CORE_vertebrates_nr_pfms.txt` |
| **Super-enhancer sequences** (`se_sequences.json`) | `medic/organ_cascade_wiring*.py`, `medic/organ_cascade_{combinatorial,kmer}.py` | Ensembl REST region endpoint (GRCh38), cached → `data/se_sequences.json` |
| **Human embryonic craniofacial H3K27ac** (Carnegie stages CS13–CS17) | `medic/craniofacial_stability.py` | Wilderman et al. 2018 (Cell Reports), human craniofacial epigenome (GEO) |
| **FaceBase mean-face mesh** (`meanface.npz`: 43,071-vertex surface + landmarks) | `face_demo/*.py`, `medic/nca_abc_modes.py`, `medic/organ_modes.py` | FaceBase (`Meanshape.mat`) → `face_demo/data/meanface.npz` (override path with `$FACEBASE_MEANFACE`) |
| **Additional GEO tracks** | kernel construction | GEO **GSE115541** |

Redistributed here (small, permissively licensed):
- `data/grays_plates/` — 20 plates from the 1918 US edition of *Gray's
  Anatomy* (public domain), fetched from Wikimedia Commons by
  `medic/grays_tab_export.py` for the viewer's Gray's tab.
- `data/bodybase/makehuman_decimated.npz` — a decimated MakeHuman base mesh
  (MakeHuman assets are CC0), used by the movie's handoff/maturation phases.
- `data/movie/grays_parts.json`, `data/movie/nca_llm.json` — the viewer's tab
  datasets (regenerable from the exports).
- `data/adapter_table.json`, `data/organ_cascade/*_search.json`,
  `mouse_to_human_builder.json` — fitted knob values consumed by
  `medic/human_movie.py`; committed so the movie regenerates from a clone.
- The movie frames themselves (`data/movie/human_movie_frames*.json`, ~58 MB)
  are generated artifacts and are **not** committed — rebuild with
  `python -m medic.human_movie`.

Notes:
- The bioelectric voltage constants used by the zebrafish/Xenopus modules
  (`zebrafish_bioelectric.py`, `xenopus_bioelectric.py`) are encoded directly
  in source as published estimates — no download required.
- No API keys are needed for the modules in this repository. Front-end
  sequence-model access (AlphaGenome) is **not** part of this paper's code and
  is intentionally excluded.
