"""
Temporal-clock upgrade: a MEASURED per-gene opening time from the ENCODE fetal atlas,
to replace the 3-tier (embryonic/fetal/adult) methylation strata with a continuous "when".

Pipeline:
  1. mm10 gene TSS from UCSC refGene (mm10-native, matches the ENCODE atlas + Jadhav).
  2. forebrain ATAC peaks across E11.5..E15.5 (reused from encode_temporal_index).
  3. per-peak opening time (first stage open); assign peaks to genes within +/-WIN.
  4. per-gene opening time = latest-opening DYNAMIC enhancer near the TSS (the enhancer
     that switches on late is what times the gene's differentiation).
  5. VALIDATION: early neural-progenitor markers should time EARLIER than late neuronal-
     differentiation markers. This is the measured analogue of the clock's rank-order test.

Provides gene_opening_times() as the interface the differentiation clock can consume
(guarded, like the zygote-kernel wiring), without modifying the live clock here.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.encode_opening_time_clock
Out: data/encode_opening_time_clock.{json,png}
"""
from __future__ import annotations
import os, gzip, json
from pathlib import Path
import numpy as np
import requests

from medic.encode_temporal_index import (
    discover_one_bed_per_stage, fetch_bed, openness_matrix, opening_time,
    STAGE_T, CHROMS)

HOME = Path(os.path.expanduser("~"))
_ca = HOME / ".ca_combined.pem"
if _ca.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_ca))

TISSUE = "forebrain"          # all 5 stages present; matched to neural markers
NEAR = 50_000                 # peak-to-TSS association window (bp)
CACHE = Path("data/encode_fetal"); CACHE.mkdir(parents=True, exist_ok=True)

# forebrain developmental markers (mm10), by known timing
EARLY = ["Sox2", "Pax6", "Nes", "Hes5", "Fabp7", "Notch1", "Vim", "Hes1"]
LATE = ["Neurod2", "Neurod6", "Tbr1", "Bcl11b", "Satb2", "Mef2c", "Rbfox3", "Grin1"]


def load_refgene_tss(symbols):
    """mm10 refGene TSS: symbol -> (chrom, tss), from the UCSC public MySQL mirror.
    One representative per symbol (the most upstream TSS)."""
    import pymysql
    syms = sorted(set(symbols))
    fmt = ",".join(["%s"] * len(syms))
    con = pymysql.connect(host="genome-mysql.soe.ucsc.edu", user="genome",
                          database="mm10", connect_timeout=30)
    try:
        cur = con.cursor()
        cur.execute(f"SELECT name2,chrom,strand,txStart,txEnd FROM refGene "
                    f"WHERE name2 IN ({fmt})", syms)
        rows = cur.fetchall()
    finally:
        con.close()
    tss = {}
    for sym, chrom, strand, txStart, txEnd in rows:
        if chrom not in CHROMS:
            continue
        t = int(txStart) if strand == "+" else int(txEnd)
        if sym not in tss:
            tss[sym] = (chrom, t)
        else:
            # keep the most upstream TSS
            c0, t0 = tss[sym]
            tss[sym] = (chrom, min(t0, t) if strand == "+" else max(t0, t))
    return tss


def peaks_with_opening_times(beds):
    """Union of peaks across stages -> array of (chrom, start, end, opening_time)."""
    allp = [(chr_, int(s), int(e)) for b in beds if b is not None
            for chr_, arr in b.items() for s, e in arr]
    # dedup near-identical peaks by (chrom, start//200)
    seen, uniq = set(), []
    for c, s, e in allp:
        k = (c, s // 200)
        if k in seen:
            continue
        seen.add(k); uniq.append((c, s, e))
    M = openness_matrix(uniq, beds)
    ot = opening_time(M)
    return uniq, ot


def gene_opening_time(sym, tss, peaks, ot, by_chrom):
    if sym not in tss:
        return None
    chrom, t = tss[sym]
    idx = by_chrom.get(chrom)
    if not idx:
        return None
    near = [(peaks[i], ot[i]) for i in idx
            if abs((peaks[i][1] + peaks[i][2]) // 2 - t) <= NEAR and not np.isnan(ot[i])]
    if not near:
        return None
    times = [o for _, o in near]
    dynamic = [o for o in times if o > STAGE_T[0]]  # opens after the first stage
    return dict(n_peaks=len(times), earliest=float(min(times)),
                latest=float(max(times)), median=float(np.median(times)),
                latest_dynamic=float(max(dynamic)) if dynamic else float(STAGE_T[0]),
                frac_dynamic=float(np.mean([o > STAGE_T[0] for o in times])))


def main():
    print("loading mm10 refGene TSS ...")
    tss = load_refgene_tss(EARLY + LATE)
    print(f"  {len(tss)} gene symbols")
    accs = discover_one_bed_per_stage(TISSUE)
    print(f"{TISSUE} stage beds: {dict(zip([s[:4] for s in ['11.5','12.5','13.5','14.5','15.5']], accs))}")
    beds = [fetch_bed(a) if a else None for a in accs]
    peaks, ot = peaks_with_opening_times(beds)
    print(f"  {len(peaks)} union peaks; dynamic (open after E{STAGE_T[0]}): "
          f"{np.mean(ot > STAGE_T[0]):.2f}")

    # index peaks by chrom for association
    by_chrom = {}
    for i, (c, s, e) in enumerate(peaks):
        by_chrom.setdefault(c, []).append(i)

    result = {"early": {}, "late": {}}
    for grp, genes in (("early", EARLY), ("late", LATE)):
        for sym in genes:
            g = gene_opening_time(sym, tss, peaks, ot, by_chrom)
            if g:
                result[grp][sym] = g

    def stat(grp, key):
        return np.array([v[key] for v in result[grp].values()])
    print(f"\n{'gene':10s}{'class':7s}{'n':>4s}{'earliest':>10s}{'median':>9s}{'latest_dyn':>12s}{'frac_dyn':>10s}")
    for grp in ("early", "late"):
        for s, v in result[grp].items():
            print(f"{s:10s}{grp:7s}{v['n_peaks']:>4d}{v['earliest']:>10.1f}{v['median']:>9.1f}"
                  f"{v['latest_dynamic']:>12.1f}{v['frac_dynamic']:>10.2f}")

    print("\n--- measured opening time: early vs late neural markers ---")
    for key in ("median", "latest_dynamic", "frac_dynamic"):
        e, l = stat("early", key), stat("late", key)
        print(f"  {key:14s}: early {np.mean(e):.2f}  vs late {np.mean(l):.2f}"
              f"   (late later: {np.mean(l) > np.mean(e)})")

    ct_rows, ct_val = run_cell_type_organ_timing()
    result["_cell_type_organ_timing"] = ct_rows
    result["_within_organ_ordering"] = {"ok": ct_val[0], "total": ct_val[1]}

    Path("data").mkdir(exist_ok=True)
    Path("data/encode_opening_time_clock.json").write_text(json.dumps(result, indent=2))
    print("saved data/encode_opening_time_clock.json")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        for j, key in enumerate(("latest_dynamic", "frac_dynamic")):
            names, vals, cols = [], [], []
            for grp, col in (("early", "#1f77b4"), ("late", "#d62728")):
                for s, v in result[grp].items():
                    names.append(s); vals.append(v[key]); cols.append(col)
            ax[j].bar(range(len(vals)), vals, color=cols)
            ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(names, rotation=60, ha="right", fontsize=8)
            ax[j].set_title(f"{key}\n(blue=early progenitor, red=late neuronal)")
            ax[j].set_ylabel(key)
        plt.tight_layout(); plt.savefig("data/encode_opening_time_clock.png", dpi=140)
        print("saved data/encode_opening_time_clock.png")
    except Exception as ex:
        print("figure skipped:", repr(ex))


# ------------------------------------------------------------------
# Multi-organ extension: cell type -> (organ, ENCODE tissue, markers) -> opening time.
# This is the mechanism that scales to the ~200 human cell types and, by aggregation,
# to the organs they build: each cell type is timed by the opening of its marker enhancers
# in its own tissue, and an organ's timing is the span of its constituent cell types.
CELL_TYPES = [
    # (cell type, organ, ENCODE tissue, [markers], phase: early=progenitor / late=differentiated)
    # --- forebrain ---
    ("neural progenitor",     "forebrain", "forebrain", ["Sox2", "Pax6", "Nes", "Hes5"],        "early"),
    ("cortical neuron",       "forebrain", "forebrain", ["Neurod2", "Tbr1", "Bcl11b", "Satb2"], "late"),
    ("GABA interneuron",      "forebrain", "forebrain", ["Gad1", "Gad2", "Dlx2"],               "late"),
    ("astrocyte",             "forebrain", "forebrain", ["Gfap", "Aqp4", "Slc1a3"],             "late"),
    ("oligodendrocyte",       "forebrain", "forebrain", ["Olig2", "Sox10", "Mbp"],              "late"),
    # --- hindbrain ---
    ("cerebellar progenitor", "hindbrain", "hindbrain", ["Atoh1", "Barhl1", "Sox2"],            "early"),
    ("Purkinje neuron",       "hindbrain", "hindbrain", ["Calb1", "Pcp2", "Foxp2"],             "late"),
    # --- neural tube ---
    ("neural tube progenitor","neural tube","neural tube",["Sox2", "Nes", "Pax6"],              "early"),
    ("motor neuron",          "neural tube","neural tube",["Mnx1", "Isl1", "Chat"],             "late"),
    ("floor plate",           "neural tube","neural tube",["Foxa2", "Shh", "Arx"],              "late"),
    # --- facial prominence (cranial neural crest) ---
    ("cranial neural crest",  "craniofacial","embryonic facial prominence",["Sox10", "Twist1", "Foxd3"],  "early"),
    ("craniofacial chondrocyte","craniofacial","embryonic facial prominence",["Sox9", "Col2a1", "Acan"],   "late"),
    ("craniofacial osteoblast","craniofacial","embryonic facial prominence",["Runx2", "Sp7", "Bglap"],     "late"),
    # --- heart ---
    ("cardiac progenitor",    "heart",     "heart",     ["Isl1", "Gata4", "Tbx5", "Mef2c"],     "early"),
    ("cardiomyocyte",         "heart",     "heart",     ["Tnnt2", "Myh6", "Actc1", "Myl7"],     "late"),
    ("endocardial cell",      "heart",     "heart",     ["Nfatc1", "Npr3", "Pecam1"],           "late"),
    # --- limb ---
    ("limb mesenchyme",       "limb",      "limb",      ["Prrx1", "Msx1", "Tbx5"],              "early"),
    ("limb chondrocyte",      "limb",      "limb",      ["Sox9", "Col2a1", "Acan"],             "late"),
    ("myoblast",              "limb",      "limb",      ["Myf5", "Myod1", "Myog"],              "late"),
    ("tenocyte",              "limb",      "limb",      ["Scx", "Tnmd", "Mkx"],                 "late"),
    # --- liver ---
    ("hepatoblast",           "liver",     "liver",     ["Afp", "Dlk1", "Hnf4a"],               "early"),
    ("hepatocyte",            "liver",     "liver",     ["Alb", "Apoa1", "Ttr", "Fga"],         "late"),
    ("cholangiocyte",         "liver",     "liver",     ["Krt19", "Sox9", "Hnf1b"],             "late"),
    ("fetal erythroblast",    "liver",     "liver",     ["Gata1", "Klf1", "Hba-a1"],            "late"),
    # --- lung ---
    ("lung bud progenitor",   "lung",      "lung",      ["Sox9", "Id2", "Sox2"],                "early"),
    ("alveolar type II",      "lung",      "lung",      ["Sftpc", "Sftpb", "Ager"],             "late"),
    ("airway club cell",      "lung",      "lung",      ["Scgb1a1", "Foxj1", "Cyp2f2"],         "late"),
    ("pulmonary endothelium", "lung",      "lung",      ["Pecam1", "Cdh5", "Kdr"],              "late"),
    # --- kidney ---
    ("nephron progenitor",    "kidney",    "kidney",    ["Six2", "Cited1", "Eya1"],             "early"),
    ("proximal tubule",       "kidney",    "kidney",    ["Slc12a1", "Slc34a1", "Lrp2"],         "late"),
    ("podocyte",              "kidney",    "kidney",    ["Nphs1", "Nphs2", "Wt1"],              "late"),
    ("ureteric bud",          "kidney",    "kidney",    ["Ret", "Gata3", "Calb1"],              "late"),
    # --- intestine ---
    ("intestinal progenitor", "intestine", "intestine", ["Lgr5", "Cdx2", "Olfm4"],              "early"),
    ("enterocyte",            "intestine", "intestine", ["Vil1", "Fabp2", "Apoa4"],             "late"),
    ("goblet cell",           "intestine", "intestine", ["Muc2", "Tff3", "Agr2"],              "late"),
    ("enteroendocrine cell",  "intestine", "intestine", ["Chga", "Neurog3", "Neurod1"],         "late"),
    # --- stomach ---
    ("gastric progenitor",    "stomach",   "stomach",   ["Sox2", "Barx1", "Sox9"],              "early"),
    ("gastric pit/chief cell","stomach",   "stomach",   ["Muc5ac", "Pgc", "Gif"],               "late"),
]

_CTX = {}


def _tissue_context(tissue):
    if tissue not in _CTX:
        beds = [fetch_bed(a) if a else None for a in discover_one_bed_per_stage(tissue)]
        peaks, ot = peaks_with_opening_times(beds)
        bc = {}
        for i, (c, s, e) in enumerate(peaks):
            bc.setdefault(c, []).append(i)
        _CTX[tissue] = (peaks, ot, bc)
    return _CTX[tissue]


def cell_type_organ_timing():
    """Measured opening time per cell type and per organ, across the ENCODE fetal tissues."""
    markers = sorted({m for _, _, _, ms, _ in CELL_TYPES for m in ms})
    tss = load_refgene_tss(markers)
    rows = []
    for ct, organ, tissue, ms, phase in CELL_TYPES:
        peaks, ot, bc = _tissue_context(tissue)
        vals = []
        for m in ms:
            g = gene_opening_time(m, tss, peaks, ot, bc)
            if g:
                vals.append(g["latest_dynamic"])
        if not vals:
            continue
        rows.append(dict(cell_type=ct, organ=organ, tissue=tissue, phase=phase,
                         n_markers=len(vals), opening=float(np.median(vals))))
    return rows


def _organ_figure(rows):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from collections import defaultdict
    by_o = defaultdict(list)
    for r in rows:
        by_o[r["organ"]].append(r)
    organs = sorted(by_o, key=lambda o: min(r["opening"] for r in by_o[o]))
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(organs) + 2))
    for yi, o in enumerate(organs):
        cs = sorted(by_o[o], key=lambda r: r["opening"])
        xs = [r["opening"] for r in cs]
        ax.plot([min(xs), max(xs)], [yi, yi], color="0.82", lw=6, solid_capstyle="round", zorder=1)
        n = len(cs)
        for k, r in enumerate(cs):
            off = (k - (n - 1) / 2) * 0.17
            col = "#1f77b4" if r["phase"] == "early" else "#d62728"
            ax.scatter(r["opening"], yi + off, color=col, s=55, edgecolor="w", lw=0.6, zorder=3)
            ax.annotate(r["cell_type"], (r["opening"], yi + off), xytext=(5, 0),
                        textcoords="offset points", fontsize=6, va="center")
    ax.set_yticks(range(len(organs))); ax.set_yticklabels(organs)
    ax.set_xlabel("measured opening time (developmental stage, days)")
    ax.set_xticks(STAGE_T); ax.set_xlim(STAGE_T[0] - 0.4, STAGE_T[-1] + 1.4)
    ax.set_title("Measured cell-type differentiation clock across organs\n"
                 "(blue = progenitor, red = differentiated; grey bar = organ developmental span)")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", label="progenitor", markersize=8),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", label="differentiated", markersize=8)],
              loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig("data/encode_cell_type_organ_clock.png", dpi=150)
    plt.close(fig); print("saved data/encode_cell_type_organ_clock.png")


def run_cell_type_organ_timing():
    rows = cell_type_organ_timing()
    print("\n(5) MEASURED MULTI-ORGAN CLOCK -- cell-type opening time across ENCODE fetal tissues")
    print(f"    {'cell type':22s}{'organ':11s}{'phase':7s}{'nmark':>6s}{'opening(E)':>11s}")
    for r in sorted(rows, key=lambda x: (x["organ"], x["opening"])):
        print(f"    {r['cell_type']:22s}{r['organ']:11s}{r['phase']:7s}{r['n_markers']:>6d}{r['opening']:>11.1f}")
    # within-organ validation: differentiated cell type opens no earlier than its progenitor
    by_organ = {}
    for r in rows:
        by_organ.setdefault(r["organ"], {})[r["phase"]] = r["opening"]
    ok = tot = 0
    for organ, ph in by_organ.items():
        if "early" in ph and "late" in ph:
            tot += 1
            ok += ph["late"] >= ph["early"]
    print(f"    within-organ ordering (late >= early opening): {ok}/{tot} organs")
    try:
        _organ_figure(rows)
    except Exception as ex:
        print("organ figure skipped:", repr(ex))
    return rows, (ok, tot)


def calibrate_unlock_schedule(rows=None, div_max=40):
    """Wire the measured opening time into the two MLP heads.

    The division head drives one clock (division -> telomere -> PRC2); the differentiation
    head reads it (a fate unlocks when PRC2 < its threshold). The measured opening time
    calibrates that gate: per cell type it sets (a) the DIFFERENTIATION head's PRC2 unlock
    threshold, and (b), on the same telomere axis, the DIVISION head's division count at
    unlock. Both come from one monotone map of the measured opening stage; the map is a
    modeling choice, the opening stage is the datum.
    """
    rows = rows or cell_type_organ_timing()
    s0, s1 = STAGE_T[0], STAGE_T[-1]
    out = {}
    for r in rows:
        u = (r["opening"] - s0) / (s1 - s0)          # 0 (earliest) .. 1 (latest)
        prc2_threshold = float(0.5 ** (1.0 + 5.0 * u))  # earlier open -> higher threshold -> unlocks sooner
        divisions = int(round(div_max * u))             # later open -> more divisions on the shared telomere axis
        out[r["cell_type"]] = dict(organ=r["organ"], opening=r["opening"],
                                   prc2_threshold=prc2_threshold, divisions_to_unlock=divisions)
    return out


def gene_opening_times(genes=None):
    """Interface for the differentiation clock: {symbol: latest_dynamic_opening_time}.
    Guarded consumer -- returns measured opening times where the atlas has them, else {}.
    """
    genes = genes or (EARLY + LATE)
    tss = load_refgene_tss(genes)
    beds = [fetch_bed(a) if a else None for a in discover_one_bed_per_stage(TISSUE)]
    peaks, ot = peaks_with_opening_times(beds)
    by_chrom = {}
    for i, (c, s, e) in enumerate(peaks):
        by_chrom.setdefault(c, []).append(i)
    out = {}
    for sym in (genes or list(tss.keys())):
        g = gene_opening_time(sym, tss, peaks, ot, by_chrom)
        if g:
            out[sym] = g["latest_dynamic"]
    return out


if __name__ == "__main__":
    main()
