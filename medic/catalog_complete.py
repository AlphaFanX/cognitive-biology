"""Complete the model's weights from the WHOLE GWAS Catalog (the parts list), not 14 hand knobs.

This is the scaled-up engine behind medic/gene_trait_adapter.py ("the true view"). gene_trait_adapter
places ~14 curated alleles by hand; this streams the full GWAS Catalog associations file
(data/gwas/gwas-catalog-download-associations-alt-full.tsv, ~1.18M rows) and routes EVERY association
into the generative model, producing two weight artefacts:

  B1  data/weights/gene_partition.json    -- every mapped gene split KERNEL vs ADAPTER by
      fixation x pleiotropy. High-pleiotropy / broadly-constrained genes are the conserved KERNEL
      (they belong in the NCA MLP base W0); single-head common-variant genes are the individual
      ADAPTER (they belong in the LGM's Delta W = sum_k beta_k e_k). This IS the falsifiable
      allele-freq x effect x pleiotropy axis of Paper #2, computed on real data.

  B2  data/weights/adapter_table_full.json -- the completed LGM adapter: for every model HEAD
      (the ~60 fates of medic.unified_embryo), the genes that route to it with their effect
      magnitudes (a_m) and direction (e_k = the head). Generalises adapter_table.json's 14 knobs
      to the whole catalog. Traits that route to NO current head are reported as CANDIDATE NEW HEADS
      (the "atlas is a floor" rule).

The router (trait -> head) is a transparent keyword ontology over the model's own head names, honest
that it is v1 (keyword, not EFO-graph). Coverage is reported plainly. Nothing is asserted end-to-end;
each association keeps its trait provenance.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.catalog_complete
"""
import os, csv, json, math, sys, re
from collections import defaultdict

# non-coding / annotation-artifact symbols that inflate raw GWAS pleiotropy but are NOT developmental
# master regulators (spuriously mapped ncRNA, antisense/divergent transcripts, clone IDs). Excluded from
# the KERNEL (they are not W0 modules); still parsed so their associations are not silently dropped.
_ARTIFACT = re.compile(
    r"^(Y_RNA|Metazoa_SRP|7SK|7SL|RMRP|RN7S|RNU\d|U\d+$|SNOR|SCARNA|RNVU|VTRNA|RNA5S|RNA45S)"
    r"|^MIR\d|^MIRLET|^LINC\d|^LOC\d|^(RP\d{1,2}|AC\d|AL\d|AP\d|CTD|CTC|CTB|RP11|XXbac)"
    r"|(-AS\d?$)|(-DT$)|(-IT\d$)", re.IGNORECASE)


def is_real_gene(sym):
    return bool(sym) and not _ARTIFACT.search(sym)

HERE = os.path.dirname(__file__)
CATALOG = os.path.join(HERE, "..", "data", "gwas", "gwas-catalog-download-associations-alt-full.tsv")
WDIR = os.path.join(HERE, "..", "data", "weights")
os.makedirs(WDIR, exist_ok=True)

# ---- the model's heads (kept in sync with medic.unified_embryo.FATES) -----------------------------
try:
    from medic.unified_embryo import FATES as MODEL_HEADS
except Exception:                                                    # decoupled fallback (same list)
    MODEL_HEADS = ["Forebrain", "Eye", "Nervous System", "Spinal Cord", "Neural Crest", "Mesoderm",
        "Somite", "Epidermal", "Hypoblast", "Yolk Syncytial Layer", "Blastodisc",
        "Proliferative Like Cell", "Limb Bud", "Heart", "Otic", "Liver", "Lung", "Pancreas", "Gut",
        "Rib", "Kidney", "Muscle", "Notochord", "Skin", "Cartilage", "DRG", "Sympathetic", "Vessel",
        "Meninges", "Connective", "Jaw", "Choroid", "Gonad", "Midbrain", "Hindbrain", "Cerebellum",
        "Mesothelium", "Mesentery", "Mucosa", "HeadMes", "Branchial", "Blood", "Atrium", "Ventricle",
        "Outflow", "LiverHaem", "Foregut", "Hindgut", "Nephron", "Retina", "Adrenal", "Thymus",
        "Spleen", "Bladder", "Adipose", "OlfactoryBulb", "Cavity"]
MODEL_HEADS = set(MODEL_HEADS)

# ---- the trait -> head router (v1 keyword ontology; each value MUST be a real model head) ----------
# ordered specific -> generic; the FIRST head whose any keyword is a substring of the (lowercased)
# trait wins. Grouped by organ system. A head that is not in MODEL_HEADS is dropped with a warning.
ROUTES = [
    # --- eye / ear ---
    ("Retina",    ["retina", "macular", "macular degeneration", "diabetic retinopathy"]),
    ("Eye",       ["myopia", "refractive error", "intraocular pressure", "glaucoma", "corneal",
                   "astigmatism", "cataract", "visual acuity", "eye colour", "eye color", "optic disc"]),
    ("Otic",      ["hearing", "auditory", "tinnitus", "cochlea", "age-related hearing"]),
    # --- heart / vessels / blood ---
    ("Atrium",    ["atrial fibrillation", "left atrial", "p wave", "pr interval"]),
    ("Ventricle", ["left ventric", "qrs", "qt interval", "ejection fraction", "ventricular"]),
    ("Outflow",   ["aortic", "aorta", "aortic root", "ascending aorta"]),
    ("Heart",     ["cardiac", "heart rate", "heart failure", "myocard", "coronary artery disease",
                   "cardiomyopathy", "coronary", "heart"]),
    ("Vessel",    ["blood pressure", "hypertension", "pulse pressure", "carotid", "vascular",
                   "intima media", "varicose", "aneurysm"]),
    ("Blood",     ["platelet", "erythrocyte", "red blood cell", "white blood cell", "haemoglobin",
                   "hemoglobin", "leukocyte", "lymphocyte", "monocyte", "neutrophil", "eosinophil",
                   "basophil", "reticulocyte", "mean corpuscular", "haematocrit", "hematocrit",
                   "mean platelet", "red cell", "blood cell"]),
    # --- respiratory ---
    ("Lung",      ["fev1", "fvc", "lung function", "pulmonary", "copd", "chronic obstructive",
                   "asthma", "respiratory", "lung", "interstitial lung"]),
    # --- renal / bladder ---
    ("Nephron",   ["glomerular filtration", "egfr", "creatinine", "cystatin", "chronic kidney",
                   "nephro", "renal function", "kidney"]),
    ("Bladder",   ["bladder", "urinary incontinence", "urinary tract"]),
    # --- liver / pancreas / gut ---
    ("LiverHaem", ["alanine aminotransferase", "aspartate aminotransferase", "gamma glutamyl",
                   "bilirubin", "alkaline phosphatase", "liver enzyme"]),
    ("Liver",     ["liver", "hepatic", "nafld", "cirrhosis", "non-alcoholic fatty"]),
    ("Pancreas",  ["pancrea", "amylase", "lipase", "type 2 diabetes", "fasting glucose",
                   "fasting insulin", "glycated", "hba1c", "2 hour glucose"]),
    ("Foregut",   ["gastric", "gastro-oesophageal", "peptic ulcer", "barrett"]),
    ("Hindgut",   ["colorectal", "colon", "rectal", "hindgut"]),
    ("Mucosa",    ["inflammatory bowel", "crohn", "ulcerative colitis", "intestinal", "coeliac",
                   "celiac", "bowel", "irritable bowel"]),
    # --- endocrine / reproductive ---
    ("Adrenal",   ["cortisol", "adrenal", "aldosterone"]),
    ("Gonad",     ["testosterone", "menarche", "menopause", "reproductive", "testicular", "ovarian",
                   "polycystic ovary", "endometriosis", "sperm", "age at first birth", "oestradiol"]),
    ("Thymus",    ["thymus", "t cell", "t-cell"]),
    ("Spleen",    ["spleen", "splenic"]),
    # --- musculoskeletal ---
    ("Muscle",    ["grip strength", "lean body mass", "lean mass", "appendicular", "muscle",
                   "sarcopenia", "handgrip"]),
    ("Cartilage", ["bone mineral density", "osteoporosis", "osteoarthritis", "bone density",
                   "heel bone", "fracture", "cartilage", "hip osteoarthritis", "knee osteoarthritis",
                   "height", "standing height", "skeletal"]),
    ("Rib",       ["rib", "thoracic", "scoliosis"]),
    ("Adipose",   ["body mass index", "obesity", "body fat", "waist", "adiponectin", "adiposity",
                   "waist-hip", "waist circumference", "leptin", "visceral adipose"]),
    # --- skin / craniofacial ---
    ("Skin",      ["eczema", "atopic dermatitis", "psoriasis", "acne", "pigmentation", "melanoma",
                   "nevi", "hair", "male-pattern baldness", "vitiligo", "skin", "sunburn",
                   "basal cell carcinoma", "freckl", "tanning"]),
    ("Jaw",       ["facial", "craniofacial", "nonsyndromic cleft", "cleft lip", "cleft palate",
                   "mandible", "chin", "nose shape", "dental", "tooth", "malocclusion"]),
    # --- neural ---
    ("Retina",    ["optic nerve"]),
    ("Forebrain", ["cerebral cortex", "cortical surface", "cortical thickness", "prefrontal"]),
    ("Cerebellum",["cerebellar", "cerebellum"]),
    ("OlfactoryBulb", ["olfactory", "smell"]),
    ("DRG",       ["chronic pain", "neuropathic pain", "back pain", "pain"]),
    ("Nervous System", ["brain", "cognitive", "intelligence", "educational attainment", "neuroticism",
                   "depression", "schizophrenia", "bipolar", "alzheimer", "parkinson",
                   "multiple sclerosis", "epilepsy", "white matter", "intracranial", "autism",
                   "migraine", "stroke", "neurodevelopment", "reaction time", "insomnia",
                   "mood", "anxiety", "adhd", "dementia", "subcortical", "hippocamp"]),
    ("Meninges",  ["meningioma", "meningi"]),
]


def route_trait(trait):
    """trait string -> model head name, or None (candidate new head / not morphological)."""
    t = trait.lower()
    for head, keys in ROUTES:
        if head not in MODEL_HEADS:
            continue
        for k in keys:
            if k in t:
                return head
    return None


def split_genes(mapped_gene):
    """MAPPED_GENE field may hold 'A', 'A - B' (intergenic flanks) or 'A, B'. Yield clean symbols."""
    if not mapped_gene or mapped_gene in ("NR", "", "intergenic"):
        return []
    parts = mapped_gene.replace(" - ", ", ").replace(";", ",").replace(" x ", ", ").split(",")
    out = []
    for p in parts:
        g = p.strip()
        if g and g not in ("NR", "intergenic") and not g.startswith("LOC") and len(g) <= 20:
            out.append(g)
    return out


def _f(x):
    try:
        return float(str(x).split()[0])
    except Exception:
        return None


def stream_catalog(limit=None):
    """Yield parsed rows: (genes[list], trait, head_or_None, effect, mlog, freq, allele, snp, pmid)."""
    with open(CATALOG, encoding="utf-8", errors="replace", newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        col = {h: i for i, h in enumerate(header)}
        gi, ti, mti = col["MAPPED_GENE"], col["DISEASE/TRAIT"], col.get("MAPPED_TRAIT", col["DISEASE/TRAIT"])
        ei = col["OR or BETA"]; li = col["PVALUE_MLOG"]; fi = col["RISK ALLELE FREQUENCY"]
        ai = col["STRONGEST SNP-RISK ALLELE"]; si = col["SNPS"]; pi = col["PUBMEDID"]
        for n, row in enumerate(rd):
            if limit and n >= limit:
                break
            if len(row) <= mti:
                continue
            trait = (row[mti] or row[ti] or "").strip()
            head = route_trait(trait) or route_trait((row[ti] or "").strip())
            yield (split_genes(row[gi]), trait, head, _f(row[ei]), _f(row[li]),
                   _f(row[fi]), row[ai].strip(), row[si].strip(), row[pi].strip())


def build(limit=None):
    # per-gene and per-head aggregates -------------------------------------------------------------
    gene_heads = defaultdict(set)      # gene -> set(head)
    gene_traits = defaultdict(set)     # gene -> set(trait)  (pleiotropy)
    gene_nassoc = defaultdict(int)
    gene_bestmlog = defaultdict(float)
    gene_freqs = defaultdict(list)
    # head -> gene -> aggregate
    head_gene = defaultdict(lambda: defaultdict(lambda: dict(n=0, mlog=0.0, effects=[], traits=set(),
                                                             allele="", freq=None, snp="")))
    unrouted = defaultdict(int)        # trait -> count (candidate new heads)
    n_rows = n_routed = n_geneassoc = 0

    for genes, trait, head, eff, mlog, freq, allele, snp, pmid in stream_catalog(limit):
        n_rows += 1
        mlog = mlog or 0.0
        for g in genes:
            gene_traits[g].add(trait)
            gene_nassoc[g] += 1
            gene_bestmlog[g] = max(gene_bestmlog[g], mlog)
            if freq is not None and 0 < freq <= 1:
                gene_freqs[g].append(freq)
            if head:
                gene_heads[g].add(head)
        if head is None:
            if trait:
                unrouted[trait] += 1
            continue
        n_routed += 1
        for g in genes:
            n_geneassoc += 1
            hg = head_gene[head][g]
            hg["n"] += 1
            hg["mlog"] = max(hg["mlog"], mlog)
            if eff is not None:
                hg["effects"].append(eff)
            hg["traits"].add(trait)
            if not hg["allele"]:
                hg["allele"], hg["snp"] = allele, snp
            if hg["freq"] is None and freq is not None:
                hg["freq"] = freq

    # ---- B1: kernel vs adapter partition (fixation x pleiotropy) ----------------------------------
    partition = {}
    for g in gene_traits:
        n_heads = len(gene_heads[g])
        n_traits = len(gene_traits[g])
        freqs = gene_freqs[g]
        med_freq = sorted(freqs)[len(freqs) // 2] if freqs else None
        # kernel score: broad pleiotropy across heads (conserved master regulators) is the signature.
        # continuous in [0,1]: saturating in #heads, with a #traits pleiotropy boost.
        ks = 1.0 - math.exp(-(0.55 * n_heads + 0.12 * n_traits))
        if not is_real_gene(g):
            cls = "artifact"                              # ncRNA / clone-ID annotation noise
        elif g.startswith("HLA-") or g in ("C4A", "C4B", "TAP2", "MICA", "MICB"):
            cls = "hla_ld"                                # MHC LD block: pleiotropic by LD, not development
        elif n_heads >= 4:
            cls = "kernel"                                # conserved master regulator -> NCA W0 module
        elif n_heads <= 1:
            cls = "adapter"                               # individual variation -> LGM dW
        else:
            cls = "mixed"
        partition[g] = dict(n_heads=n_heads, heads=sorted(gene_heads[g]), n_traits=n_traits,
                            n_assoc=gene_nassoc[g], best_mlog=round(gene_bestmlog[g], 1),
                            median_freq=med_freq, kernel_score=round(ks, 3), klass=cls)

    # ---- B2: the completed LGM adapter table ------------------------------------------------------
    TOPN = 200
    adapter = {}
    for head, genes in head_gene.items():
        rows = []
        for g, a in genes.items():
            effects = a["effects"]
            a_mag = (sum(abs(e) for e in effects) / len(effects)) if effects else None
            rows.append(dict(gene=g, a_mag=(round(a_mag, 4) if a_mag is not None else None),
                             strength_mlog=round(a["mlog"], 1), n_assoc=a["n"],
                             snp=a["snp"], allele=a["allele"], freq=a["freq"],
                             klass=partition.get(g, {}).get("klass", "adapter"),
                             sample_traits=sorted(a["traits"])[:3]))
        rows.sort(key=lambda r: r["strength_mlog"], reverse=True)
        adapter[head] = dict(direction=head, n_genes=len(rows), source="gwas_catalog_full",
                             genes=rows[:TOPN], truncated=(len(rows) > TOPN))

    # ---- coverage accounting (honest) -------------------------------------------------------------
    kernel_n = sum(1 for v in partition.values() if v["klass"] == "kernel")
    mixed_n = sum(1 for v in partition.values() if v["klass"] == "mixed")
    adapter_n = sum(1 for v in partition.values() if v["klass"] == "adapter")
    artifact_n = sum(1 for v in partition.values() if v["klass"] == "artifact")
    hla_n = sum(1 for v in partition.values() if v["klass"] == "hla_ld")
    heads_covered = sorted(adapter.keys())
    heads_empty = sorted(MODEL_HEADS - set(heads_covered))
    top_unrouted = sorted(unrouted.items(), key=lambda x: -x[1])[:40]

    meta = dict(catalog_rows=n_rows, routed_associations=n_routed,
                route_rate=round(n_routed / max(1, n_rows), 3), genes_total=len(partition),
                kernel_genes=kernel_n, mixed_genes=mixed_n, adapter_genes=adapter_n,
                artifact_genes=artifact_n, hla_ld_genes=hla_n,
                heads_covered=len(heads_covered), heads_total=len(MODEL_HEADS),
                heads_empty=heads_empty,
                candidate_new_heads=[dict(trait=t, n=c) for t, c in top_unrouted])

    json.dump(dict(meta=meta, gene=partition), open(os.path.join(WDIR, "gene_partition.json"), "w"))
    json.dump(dict(meta=meta, heads=adapter), open(os.path.join(WDIR, "adapter_table_full.json"), "w"))

    # ---- report -----------------------------------------------------------------------------------
    print(f"\n=== GWAS Catalog -> model weight completion ===")
    print(f"catalog rows parsed        {n_rows:,}")
    print(f"associations routed to a head {n_routed:,}  ({100*meta['route_rate']:.1f}%)")
    print(f"mapped genes               {len(partition):,}")
    print(f"  KERNEL  (>=4 heads, -> NCA W0):   {kernel_n:,}")
    print(f"  mixed   (2-3 heads):              {mixed_n:,}")
    print(f"  ADAPTER (<=1 head,  -> LGM dW):   {adapter_n:,}")
    print(f"  (artifact ncRNA/clone dropped:    {artifact_n:,};  HLA-LD flagged: {hla_n})")
    print(f"heads covered              {len(heads_covered)} / {len(MODEL_HEADS)}")
    print(f"heads with no adult trait  {heads_empty}")
    print(f"\ntop KERNEL genes (real developmental masters, highest pleiotropy across heads):")
    kern = {g: v for g, v in partition.items() if v["klass"] == "kernel"}
    for g, v in sorted(kern.items(), key=lambda kv: -kv[1]["n_heads"])[:25]:
        print(f"  {g:12s} heads={v['n_heads']:2d} traits={v['n_traits']:4d} mlog={v['best_mlog']:5.0f}  {','.join(v['heads'][:6])}")
    print(f"\nper-head adapter gene counts:")
    for h in sorted(adapter, key=lambda h: -adapter[h]["n_genes"]):
        print(f"  {h:16s} {adapter[h]['n_genes']:5d} genes")
    print(f"\ncandidate NEW heads (top unrouted traits):")
    for t, c in top_unrouted[:15]:
        print(f"  {c:6d}  {t[:70]}")
    print(f"\nwrote {WDIR}\\gene_partition.json  and  adapter_table_full.json")
    return meta


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build(limit=lim)
