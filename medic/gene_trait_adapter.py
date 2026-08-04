"""The catalog -> model loader: assemble the individual adapter from gene->trait->allele databases.

This generalises the face proof-of-concept (medic/ground_face_betas.py -> real_effects.json -> the morph
deformers) into ONE table that wires real per-allele effect sizes into every model knob that exists.

    GWAS Catalog / Open Targets / study summary stats   -> beta_k   (magnitude)
    Open Targets L2G / ENCODE / the head structure       -> e_k     (which knob)
                     |
                     v   adapter table {knob: (gene, snp, effect_allele, beta, direction, source)}
                     v
    model individual-adapter layer:  Delta W = sum_k beta_k * e_k

A KNOB REGISTRY names, for each parameter the model currently has (face, body, cardiac, eye, kidney),
the gene/lead SNP and where its beta comes from. A puller reads the beta from:
  - 'xiong'      : the local Xiong C-GWAS extraction (data/xiong_cgwas/real_effects.json)
  - 'catalog'    : a live GWAS Catalog REST query on the lead rsID
  - 'literature' : an effect size quoted in a published study (unit noted)
  - 'polygenic'  : a trait too polygenic for one allele (height); flagged, represented by a PGS direction
and emits data/adapter_table.json, the single source of truth the model reads (mesh_morph loads the
face rows). Every new head the model grows auto-inherits its real alleles by adding one registry row.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.gene_trait_adapter
"""
import os, json, subprocess

HERE = os.path.dirname(__file__)
REAL_EFFECTS = os.path.join(HERE, "..", "data", "xiong_cgwas", "real_effects.json")
OUT = os.path.join(HERE, "..", "data", "adapter_table.json")

# ---------------------------------------------------------------- the knob registry
# knob -> dict(gene, snp, source, trait, direction[, beta, unit, note])
REGISTRY = {
    # FACE (real Xiong C-GWAS betas, extracted by ground_face_betas.py)
    "face.chin":        dict(gene="EDAR",  snp="rs3827760",  source="xiong", trait="chin protrusion", direction="+z chin",
                             note="canonical EDAR lead; Xiong beta from EDAR-region proxy rs11676729"),
    "face.nasion":      dict(gene="PAX3",  snp="rs7559271",  source="xiong", trait="nasion depth",     direction="+z nasion"),
    "face.nasal_tip":   dict(gene="DCHS2", snp="rs2045323",  source="xiong", trait="nasal tip",        direction="+y/z tip",
                             note="L2G assigns neighbouring SFRP2 (causal-gene ambiguity); Adhikari 2016 = DCHS2"),
    "face.nasal_bridge":dict(gene="RUNX2", snp="rs227833",   source="xiong", trait="bridge breadth",   direction="+/-x bridge",
                             note="RUNX2/SUPT3H locus; rs227833 = SUPT3H (causal gene debated)"),
    "face.alae":        dict(gene="GLI3",  snp="rs17640804", source="xiong", trait="alar breadth",     direction="+/-x alae",
                             note="corrected lead (L2G->GLI3 0.86); beta pending re-extraction with this SNP"),
    "face.alae2":       dict(gene="PAX1",  snp="rs6047635",  source="xiong", trait="alar breadth",     direction="+/-x alae",
                             note="corrected lead (L2G->PAX1); beta pending re-extraction"),
    # refined: the real alare-alare (max nose WIDTH) distance, landmarks 26-27, extracted from Xiong Lowernose
    "face.nose_width":  dict(gene="GLI3",  snp="rs17640804", source="xiong_distance", trait="alare-alare nose width (lm 26-27)",
                             direction="alar widening",
                             note="distance-specific; co-acting RUNX2 +0.019, DCHS2 -0.025, PAX1 +0.005 (all tiny -> polygenic)"),
    # CARDIAC (Pirruccello UKB cardiac-MRI; already in Paper #7)
    "heart.la_volume":  dict(gene="PITX2", snp="rs2129977",  source="literature", trait="left-atrial volume",
                             beta=0.057, unit="SD/allele", direction="D-loop laterality knob"),
    "heart.lv_cavity":  dict(gene="TTN",   snp="rs2042995",  source="literature", trait="LV end-diastolic volume",
                             beta=0.060, unit="SD/allele", direction="tube-length knob"),
    "heart.wall":       dict(gene="TBX5",  snp=None,          source="developmental", trait="chamber wall",
                             direction="wall knob", note="developmental anchor; no morphology association"),
    # EYE (Cheng 2013 axial length, mm/allele; in Paper #7)
    "eye.axial_length": dict(gene="RSPO1", snp="rs2777044",  source="literature", trait="ocular axial length",
                             beta=0.073, unit="mm/allele", direction="eye-field scale"),
    # KIDNEY (Liu 2021 UKB abdominal MRI cortex volume; in Paper #7)
    "kidney.cortex":    dict(gene="MYCN",  snp="rs2168101",  source="literature", trait="kidney cortex volume",
                             beta=0.06,  unit="SD/allele", direction="ureteric-branching knob"),
    # LIMB (GDF5 = the cleanest single limb-length/skeletal locus; Chondrogenesis GDF5-bmp; height/limb effect)
    "limb.length":      dict(gene="GDF5",  snp="rs143383", source="literature", trait="limb long-bone length",
                             beta=0.055, unit="SD/allele", direction="PD (AER/FGF) length knob",
                             note="GDF5 growth-plate; height/limb-length lead SNP (Yengo/OA cross-trait)"),
    # BODY (too polygenic for one allele -> PGS direction, flagged)
    "body.height":      dict(gene="(polygenic ~12k loci)", snp=None, source="polygenic", trait="standing height",
                             direction="overall maturation size", note="Yengo 2022; represent by a PGS, not one allele"),
    "body.proportion":  dict(gene="(polygenic)", snp=None, source="polygenic", trait="sitting-height ratio",
                             direction="limb:trunk (embryonic)", note="represent by a PGS"),
}


def _get(url):
    r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "-m", "25", url], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {}


def _ot_graphql(query):
    r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "-m", "30", "-X", "POST",
                        "https://api.platform.opentargets.org/api/v4/graphql",
                        "-H", "Content-Type: application/json",
                        "-d", json.dumps({"query": query})], capture_output=True, text=True)
    try:
        return json.loads(r.stdout).get("data", {})
    except Exception:
        return {}


def opentargets_l2g(rsid):
    """The e_k 'which gene' side: Open Targets Locus-to-Gene. rsID -> variant -> top L2G causal gene.
    This is the automatable version of the hand-curated gene assignment (variant->gene = eNCODE/L2G)."""
    d = _ot_graphql('{search(queryString:"%s",entityNames:["variant"]){hits{id}}}' % rsid)
    hits = d.get("search", {}).get("hits", [])
    if not hits:
        return None
    vid = hits[0]["id"]
    d = _ot_graphql('{variant(variantId:"%s"){credibleSets{rows{l2GPredictions{rows{target{approvedSymbol} score}}}}}}' % vid)
    best = {}
    for cs in d.get("variant", {}).get("credibleSets", {}).get("rows", []):
        for p in cs.get("l2GPredictions", {}).get("rows", []):
            g, sc = p["target"]["approvedSymbol"], p["score"]
            if g not in best or sc > best[g]:
                best[g] = sc
    if not best:
        return None
    g, sc = max(best.items(), key=lambda x: x[1])
    return dict(gene=g, l2g_score=round(sc, 3))


def catalog_beta(rsid):
    """Live GWAS Catalog: first association with a numeric beta on a morphology trait."""
    d = _get(f"https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/{rsid}/associations?projection=associationBySnp")
    for a in d.get("_embedded", {}).get("associations", []):
        b = a.get("betaNum")
        if b is None:
            continue
        allele = ""
        for l in a.get("loci", []):
            for r in l.get("strongestRiskAlleles", []):
                allele = r.get("riskAlleleName", "")
        traits = [t.get("trait", "") for t in a.get("efoTraits", [])]
        return dict(beta=b, unit=a.get("betaUnit") or "NR", direction=a.get("betaDirection"),
                    allele=allele, trait=traits)
    return None


def build(live_catalog=True, l2g=False):
    real = {}
    if os.path.exists(REAL_EFFECTS):
        real = json.load(open(REAL_EFFECTS))
    table = {}
    for knob, r in REGISTRY.items():
        row = dict(gene=r["gene"], snp=r.get("snp"), trait=r["trait"], direction=r["direction"],
                   source=r["source"])
        src = r["source"]
        if src == "xiong":
            e = real.get(r["gene"], {})
            row.update(beta=e.get("beta"), unit="C-GWAS std", effect_allele=e.get("a1"),
                       provenance="Xiong 2025 C-GWAS (Zenodo 13730680)")
        elif src == "literature":
            row.update(beta=r.get("beta"), unit=r.get("unit"), provenance="published effect size (Paper #7 refs)")
        elif src == "catalog" and live_catalog and r.get("snp"):
            c = catalog_beta(r["snp"])
            if c:
                row.update(beta=c["beta"], unit=c["unit"], effect_allele=c["allele"],
                           provenance=f"GWAS Catalog live: {c['trait']}")
        elif src == "xiong_distance":
            nw = {}
            p = os.path.join(HERE, "..", "data", "xiong_cgwas", "nose_width_betas.json")
            if os.path.exists(p):
                nw = json.load(open(p))
            row.update(beta=nw.get("betas", {}).get(r["gene"]), unit="C-GWAS std",
                       width_betas=nw.get("betas", {}),   # co-acting bridge/alae betas (mesh_morph reads these)
                       provenance="Xiong C-GWAS alare-alare distance 26_27 (nose_width_betas.json)")
        elif src == "developmental":
            row.update(beta=None, unit=None, provenance="developmental anchor (no morphology GWAS)")
        elif src == "polygenic":
            row.update(beta=None, unit="PGS", provenance=r.get("note"))
        if r.get("note"):
            row["note"] = r["note"]
        if l2g and r.get("snp") and r.get("snp", "").startswith("rs"):   # validate e_k via Open Targets L2G
            ot = opentargets_l2g(r["snp"])
            if ot:
                row["l2g"] = ot
                row["l2g_confirms"] = (ot["gene"] == r["gene"])
        table[knob] = row

    json.dump(table, open(OUT, "w"), indent=1)
    print(f"assembled {len(table)} knobs -> {OUT}\n")
    for k, v in table.items():
        b = v.get("beta")
        bs = f"{b:+.4f} {v.get('unit','')}" if isinstance(b, (int, float)) else f"[{v['source']}]"
        print(f"  {k:20s} {v['gene']:10s} {str(v.get('snp')):12s} {bs:22s} {v['trait']}")
    return table


def complete_from_catalog():
    """Scale the 14 curated knobs to the WHOLE GWAS Catalog (medic.catalog_complete): route every
    association to a model head, partition every gene kernel-vs-adapter, and emit the completed LGM
    adapter (data/weights/adapter_table_full.json) + the kernel/adapter partition
    (data/weights/gene_partition.json). The 14 hand rows here remain the VALIDATED e_k anchors (their
    directions are curated); the catalog fills the rest of every head. See catalog_complete for the report."""
    from medic.catalog_complete import build as _cat
    return _cat()


if __name__ == "__main__":
    import sys
    if "--full" in sys.argv:
        complete_from_catalog()          # complete every head from the whole catalog
    else:
        build()                          # the 14 curated, validated knobs
