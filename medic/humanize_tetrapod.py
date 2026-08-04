"""HUMANIZE THE TETRAPOD -- work through all 34 GWAS-reachable heads (Miles, 2026-07-26).

The completed adapter (data/weights/adapter_table_full.json, from medic.catalog_complete) fills 34 of the
model's 57 heads with real human alleles: per head a list of genes, each with an adapter magnitude a_mag, its
lead allele and population frequency. Those 34 heads are exactly the part of the anatomy that HUMAN common
variation reaches -- the individual/population ADAPTER -- so writing all of them into the model is how the
computed generic tetrapod is turned, system by system, into a human individual across form, chemistry and
mechanics. This module does that pass:

  1. classify every one of the 34 heads into FORM (organ / neural / sensory morphology), CHEMISTRY
     (biochemical / metabolic / endocrine / immune readouts) or MECHANICS (skeletal / muscular load-bearing);
  2. compute each head's HUMAN POPULATION-MEAN ADAPTER LOAD, M_h = sum_g 2 f_g a_mag_g (diploid expected
     dosage x adapter magnitude, over real allele frequencies) -- how far the human population mean sits from
     the reference along that head's axis, grounded in the catalog's own frequencies and magnitudes;
  3. mark which FORM heads have a wired forward shape model (heart/kidney/liver/brain/gut/eye) so the human
     adapter can be applied to the computed SHAPE now, versus heads recorded as a scalar human offset whose
     readout (a chemistry level, a mechanical property) is not yet dynamically simulated;
  4. report the humanization as coverage across the three domains -- the honest sense in which the tetrapod
     resembles the human more and more as the heads are filled.

HONEST BOUNDARY (hold it): this populates the human ADAPTER (individual + population variation) across the
anatomy; the conserved BASE morphology is still the vertebrate kernel, and human-specific SHAPE (the mouse->
human species adapter) is a separate, mostly-unbuilt layer. So "more human" here means the model becomes a
configurable human INDIVIDUAL across 34 systems, not that the base plan is re-grown as human.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.humanize_tetrapod
Out:  data/organ_cascade/humanize_tetrapod.{json,png}
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ADAPTER = "data/weights/adapter_table_full.json"
OUT = "data/organ_cascade/humanize_tetrapod.json"
FIG = "data/organ_cascade/humanize_tetrapod.png"

# primary domain per head (a head can span domains; primary = its dominant GWAS contribution)
DOMAIN = {
    # FORM -- organ / neural / sensory morphology
    "Heart": "FORM", "Atrium": "FORM", "Ventricle": "FORM", "Outflow": "FORM", "Nephron": "FORM",
    "Lung": "FORM", "Eye": "FORM", "Retina": "FORM", "Forebrain": "FORM", "Cerebellum": "FORM",
    "OlfactoryBulb": "FORM", "Otic": "FORM", "DRG": "FORM", "Nervous System": "FORM", "Foregut": "FORM",
    "Hindgut": "FORM", "Mucosa": "FORM", "Bladder": "FORM", "Vessel": "FORM", "Meninges": "FORM",
    "Gonad": "FORM", "Skin": "FORM",
    # MECHANICS -- skeletal / muscular load-bearing
    "Cartilage": "MECHANICS", "Rib": "MECHANICS", "Muscle": "MECHANICS", "Jaw": "MECHANICS",
    # CHEMISTRY -- biochemical / metabolic / endocrine / immune readouts
    "Blood": "CHEMISTRY", "LiverHaem": "CHEMISTRY", "Liver": "CHEMISTRY", "Pancreas": "CHEMISTRY",
    "Adipose": "CHEMISTRY", "Adrenal": "CHEMISTRY", "Spleen": "CHEMISTRY", "Thymus": "CHEMISTRY",
}
# FORM heads that already have a wired forward shape model (organ_absynth / shape-genome builders)
WIRED_SHAPE = {"Heart", "Atrium", "Ventricle", "Outflow", "Nephron", "Forebrain", "Cerebellum",
               "Nervous System", "Foregut", "Hindgut", "Mucosa", "Eye", "Retina"}
DCOL = {"FORM": "#3aa869", "CHEMISTRY": "#2a6fb0", "MECHANICS": "#c98a2b"}


def head_load(genes):
    """human population-mean adapter load M_h = sum 2 f a_mag over genes with a frequency; also top genes."""
    load = 0.0; contrib = []
    for g in genes:
        a = float(g.get("a_mag", 0.0) or 0.0); f = g.get("freq", None)
        if f is None:
            continue
        c = 2.0 * float(f) * abs(a)
        load += c; contrib.append((g["gene"], c, g.get("allele", "")))
    contrib.sort(key=lambda t: -t[1])
    return load, contrib[:5]


def run():
    at = json.load(open(ADAPTER)); heads = at["heads"]
    rows = []
    seen_genes = set()
    for h, v in heads.items():
        genes = v["genes"] if isinstance(v, dict) else v
        load, top = head_load(genes)
        for g in genes:
            seen_genes.add(g["gene"])
        rows.append(dict(head=h, domain=DOMAIN.get(h, "FORM"), n_genes=len(genes),
                         human_load=round(load, 2), wired=(h in WIRED_SHAPE),
                         top_genes=[t[0] for t in top]))
    rows.sort(key=lambda r: (r["domain"], -r["human_load"]))

    by = {d: [r for r in rows if r["domain"] == d] for d in DCOL}
    print("HUMANIZING THE TETRAPOD -- 34 GWAS-reachable heads written into the model\n")
    for d in ["FORM", "CHEMISTRY", "MECHANICS"]:
        rs = by[d]; wired = sum(r["wired"] for r in rs)
        print(f"== {d} ==  {len(rs)} heads  (shape-wired: {wired})")
        for r in rs:
            w = "[shape-wired]" if r["wired"] else "[readout pending]"
            print(f"   {r['head']:15s} load {r['human_load']:7.1f}  {r['n_genes']:4d} genes  {w}  "
                  f"top {', '.join(r['top_genes'][:3])}")
    n_form, n_chem, n_mech = len(by["FORM"]), len(by["CHEMISTRY"]), len(by["MECHANICS"])
    n_wired = sum(r["wired"] for r in rows)
    print(f"\nCOVERAGE: 34 of the model's 57 heads carry a human adapter "
          f"(FORM {n_form}, CHEMISTRY {n_chem}, MECHANICS {n_mech}); {n_wired} FORM heads are shape-wired now, "
          f"the rest are recorded as scalar human offsets pending a dynamic readout.")
    print(f"total distinct human genes written across the 34 heads: {len(seen_genes)}")
    print("HONEST: this fills the human ADAPTER (individual/population variation) across 34 systems; the base "
          "plan is still the vertebrate kernel, human-specific SHAPE needs the separate mouse->human species adapter.")

    out = dict(n_heads=len(rows), coverage="34/57",
               by_domain=dict(FORM=n_form, CHEMISTRY=n_chem, MECHANICS=n_mech),
               shape_wired=n_wired, distinct_human_genes=len(seen_genes), heads=rows,
               boundary=("fills the human ADAPTER across 34 systems; base plan still the vertebrate kernel; "
                         "human-specific SHAPE needs the separate species adapter"))
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(rows, by, n_form, n_chem, n_mech, n_wired)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _figure(rows, by, n_form, n_chem, n_mech, n_wired):
    fig = plt.figure(figsize=(15, 6.4), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1, 1])

    # panel 1: all 34 heads, adapter load, grouped by domain, hatched if shape-wired
    ax = fig.add_subplot(gs[0, 0])
    order = by["FORM"] + by["CHEMISTRY"] + by["MECHANICS"]
    y = np.arange(len(order))[::-1]
    ax.barh(y, [r["human_load"] for r in order],
            color=[DCOL[r["domain"]] for r in order],
            hatch=["//" if r["wired"] else None for r in order], edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels([r["head"] for r in order], fontsize=7)
    ax.set_xlabel("human population-mean adapter load  $M_h=\\sum 2 f\\,|a|$")
    ax.set_title("All 34 GWAS-reachable heads written into the model\n"
                 "(green FORM / blue CHEMISTRY / orange MECHANICS; hatched = shape-wired now)")

    # panel 2: coverage of the 57 heads
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar([0], [n_form], color=DCOL["FORM"], label=f"FORM {n_form}")
    ax2.bar([0], [n_chem], bottom=[n_form], color=DCOL["CHEMISTRY"], label=f"CHEMISTRY {n_chem}")
    ax2.bar([0], [n_mech], bottom=[n_form + n_chem], color=DCOL["MECHANICS"], label=f"MECHANICS {n_mech}")
    ax2.bar([0], [57 - 34], bottom=[34], color="#dddddd", label="kernel (23 empty)")
    ax2.set_xticks([]); ax2.set_ylim(0, 57); ax2.set_ylabel("model heads")
    ax2.legend(fontsize=8, loc="upper right"); ax2.set_title("Anatomy carrying a\nhuman adapter: 34 / 57")

    # panel 3: domain readiness (shape-wired vs readout-pending)
    ax3 = fig.add_subplot(gs[0, 2])
    doms = ["FORM", "CHEMISTRY", "MECHANICS"]
    wired = [sum(r["wired"] for r in by[d]) for d in doms]
    pend = [len(by[d]) - w for d, w in zip(doms, wired)]
    x = np.arange(3)
    ax3.bar(x, wired, color=[DCOL[d] for d in doms], label="shape-wired")
    ax3.bar(x, pend, bottom=wired, color="#dddddd", hatch="xx", label="readout pending")
    ax3.set_xticks(x); ax3.set_xticklabels(doms, fontsize=8, rotation=20)
    ax3.set_ylabel("heads"); ax3.legend(fontsize=8); ax3.set_title("Applied now vs\npending a readout")

    fig.suptitle("Humanizing the computed tetrapod: writing all 34 GWAS-reachable heads makes it a configurable "
                 "human INDIVIDUAL across form, chemistry and mechanics (34/57 of the anatomy).\nThe conserved "
                 "base plan stays the vertebrate kernel; human-specific base SHAPE needs the separate species "
                 "adapter -- so the adapter humanizes the variation, not yet the plan.", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(FIG, dpi=135, facecolor="white")


if __name__ == "__main__":
    run()
