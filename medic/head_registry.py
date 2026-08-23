"""
The genome-grounded head registry: every head/subhead tied to SEdb and AlphaGenome.
====================================================================================
A head is a master-TF super-enhancer cluster read out in a cell type. This registry grounds each head,
subhead, and predicted new head "in terms of SEdb and AlphaGenome" (Miles): its master TF is the SEdb
super-enhancer-cluster master, and its identity is a tissue/cell-type context AlphaGenome-mouse resolves
(a literal biosample in its RNA/DNase/ATAC tracks -- see medic.head_resolution_compare). Nothing here is
asserted without a genome-side anchor: the master TFs are canonical SE-cluster regulators and the tissue
column is an AlphaGenome mouse track.

Fields per head: (parent organ, master TF, AlphaGenome mouse tissue track, kind).
  kind: 'organ'    -- a MOSTA-annotated organ head (already in the model);
        'subhead'  -- an organ subtype, NOT a separate MOSTA label, validated by master TF;
        'new'      -- resolved by AlphaGenome/SEdb, NOT in MOSTA's 35 (a predicted new head).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.head_registry
"""
from __future__ import annotations
import json
from pathlib import Path

# name -> dict(parent, tf, ag_tissue, kind, implemented)
REGISTRY = {
    # ---- organ heads already in the model (MOSTA-annotated) ----
    "Heart":       dict(parent=None,   tf="Nkx2-5",   ag="heart",              kind="organ",   impl=True),
    "Liver":       dict(parent=None,   tf="Foxa3",    ag="liver",              kind="organ",   impl=True),
    "Lung":        dict(parent=None,   tf="Elf5",     ag="lung",               kind="organ",   impl=True),
    "Kidney":      dict(parent=None,   tf="Pax2",     ag="kidney",             kind="organ",   impl=True),
    "Gut":         dict(parent=None,   tf="Cdx2",     ag="intestine",          kind="organ",   impl=True),
    "Eye":         dict(parent=None,   tf="Pax6",     ag="retina",             kind="organ",   impl=True),
    "Gonad":       dict(parent=None,   tf="Nr5a1",    ag="gonadal fat pad",    kind="organ",   impl=True),
    # ---- ORGAN SUBHEADS (organ subtypes; not separate MOSTA labels; master-TF validated) ----
    "Atrium":      dict(parent="Heart",  tf="Tbx5",   ag="heart",              kind="subhead", impl=False),
    "Ventricle":   dict(parent="Heart",  tf="Irx4/Hand2", ag="heart",          kind="subhead", impl=False),
    "Outflow":     dict(parent="Heart",  tf="Isl1",   ag="heart",              kind="subhead", impl=False),
    "LiverHaem":   dict(parent="Liver",  tf="Gata1",  ag="liver",              kind="subhead", impl=False),
    "Foregut":     dict(parent="Gut",    tf="Sox2/Nkx2-1", ag="stomach",       kind="subhead", impl=False),
    "Hindgut":     dict(parent="Gut",    tf="Cdx2/Hoxb", ag="large intestine", kind="subhead", impl=False),
    "Nephron":     dict(parent="Kidney", tf="Six2",   ag="kidney",             kind="subhead", impl=False),
    "Retina":      dict(parent="Eye",    tf="Crx/Rax", ag="retina",            kind="subhead", impl=False),  # also a new head
    "Forebrain":   dict(parent="Brain",  tf="Arx",    ag="forebrain",          kind="subhead", impl=True),
    "Midbrain":    dict(parent="Brain",  tf="Otx2",   ag="midbrain",           kind="subhead", impl=True),
    "Hindbrain":   dict(parent="Brain",  tf="Gbx2",   ag="hindbrain",          kind="subhead", impl=True),
    "Cerebellum":  dict(parent="Brain",  tf="Atoh1",  ag="cerebellum",         kind="subhead", impl=True),
    # ---- PREDICTED NEW HEADS (AlphaGenome/SEdb resolve; not in MOSTA's 35) ----
    "Adrenal":     dict(parent=None,   tf="Nr5a1",    ag="adrenal gland",      kind="new",     impl=False),
    "Thymus":      dict(parent=None,   tf="Foxn1",    ag="thymus",             kind="new",     impl=False),
    "Spleen":      dict(parent=None,   tf="Tlx1/Ikzf1", ag="spleen",           kind="new",     impl=False),
    "Bladder":     dict(parent=None,   tf="Foxa1",    ag="urinary bladder",    kind="new",     impl=False),
    "Adipose":     dict(parent=None,   tf="Pparg",    ag="subcutaneous adipose tissue", kind="new", impl=False),
    "OlfactoryBulb": dict(parent="Brain", tf="Tbr1/Sox2", ag="olfactory bulb", kind="new",     impl=False),
}


def implemented_names():
    from medic.unified_embryo import FATES
    return set(FATES)


def main():
    impl = implemented_names()
    for k, v in REGISTRY.items():                                    # sync impl flag to the live model
        v["impl"] = k in impl
    print("GENOME-GROUNDED HEAD REGISTRY  (master TF = SEdb SE-cluster master; tissue = AlphaGenome track)\n"
          + "=" * 90)
    for kind in ("organ", "subhead", "new"):
        rows = [(k, v) for k, v in REGISTRY.items() if v["kind"] == kind]
        print(f"\n{kind.upper()} ({sum(v['impl'] for _, v in rows)}/{len(rows)} implemented):")
        for k, v in rows:
            p = f"  <-{v['parent']}" if v["parent"] else ""
            print(f"   {'OK ' if v['impl'] else '   '} {k:14s} master {v['tf']:14s} AlphaGenome:{v['ag']:22s}{p}")
    Path("data/organ_cascade").mkdir(parents=True, exist_ok=True)
    json.dump(REGISTRY, open("data/organ_cascade/head_registry.json", "w"), indent=1)
    n_impl = sum(v["impl"] for v in REGISTRY.values())
    print(f"\n{n_impl}/{len(REGISTRY)} heads implemented in the model. Grounded in SEdb (SE clusters) + "
          f"AlphaGenome (mouse tissue tracks).")


if __name__ == "__main__":
    main()
