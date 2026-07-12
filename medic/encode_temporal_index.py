"""
Temporal accessibility index PoC: a per-enhancer OPENING TIME from the ENCODE
mouse fetal-development atlas, read at the Jadhav fossil loci.

Motivation: AlphaGenome collapses developmental time to a 2-value embryonic-vs-adult
contrast (see medic/alphagenome_embryonic_panel.py). The FINE temporal index lives in
the ENCODE fetal atlas: stage-resolved ATAC peaks (E11.5..E15.5) per tissue, mm10 --
the SAME assembly as the Jadhav LMRs, so no cross-assembly lift is needed.

For each Jadhav enhancer we ask, at each stage, does it overlap an open-chromatin peak
in embryonic facial prominence (and limb)? That trajectory's first-open stage is a
measured OPENING TIME -- a continuous "when" that replaces the 3-tier methylation strata.
Random genomic intervals are the negative control.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.encode_temporal_index
Out:  data/encode_fetal/*.bed.gz (cached), data/encode_temporal_index.{json,png}
"""
from __future__ import annotations
import os, io, gzip, json
from pathlib import Path
import numpy as np
import requests

HOME = Path(os.path.expanduser("~"))
_ca = HOME / ".ca_combined.pem"
if _ca.exists():
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_ca))

S = "https://www.encodeproject.org"
STAGES = ["11.5 days", "12.5 days", "13.5 days", "14.5 days", "15.5 days"]
STAGE_T = [11.5, 12.5, 13.5, 14.5, 15.5]
TISSUES = ["intestine", "forebrain"]  # intestine = tissue-matched to Jadhav; forebrain = mismatched control
# housekeeping gene windows (mm10) that must be open in every tissue -> assembly + overlap sanity check
POS_CTRL = [("Actb", "chr5", 142903000, 142907000),
            ("Gapdh", "chr6", 125161000, 125165000),
            ("Actg1", "chr11", 120347000, 120351000),
            ("Tbp", "chr17", 15497000, 15521000)]
CACHE = Path("data/encode_fetal"); CACHE.mkdir(parents=True, exist_ok=True)
CHROMS = {f"chr{c}" for c in list(range(1, 20)) + ["X"]}
WIDTH_PAD = 0  # enhancer intervals used as-is


def discover_one_bed_per_stage(tissue):
    """One ATAC IDR-peak bed (mm10) per developmental stage for a tissue."""
    j = requests.get(S + "/search/", params={
        "type": "File", "assay_title": "ATAC-seq",
        "biosample_ontology.term_name": tissue,
        "output_type": "IDR thresholded peaks", "file_format": "bed",
        "assembly": "mm10", "status": "released", "format": "json", "limit": "80",
        "field": ["accession", "file_size", "dataset"]}, timeout=60).json()
    files = j.get("@graph", [])
    # map dataset -> stage
    stage_of = {}
    for d in sorted({g["dataset"] for g in files}):
        e = requests.get(S + d, params={"format": "json"}, timeout=60).json()
        age = ""
        for rep in e.get("replicates", []):
            bs = rep.get("library", {}).get("biosample", {})
            age = bs.get("age_display") or age
            if age:
                break
        stage_of[d] = age
    picked = {}
    for g in files:
        st = stage_of.get(g["dataset"], "")
        if st in STAGES and st not in picked:
            picked[st] = g["accession"]
    return [picked.get(st) for st in STAGES]


def fetch_bed(acc):
    p = CACHE / f"{acc}.bed.gz"
    if not p.exists():
        url = f"{S}/files/{acc}/@@download/{acc}.bed.gz"
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        p.write_bytes(r.content)
    # parse chrom,start,end
    by = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            c = line.split("\t", 3)
            if len(c) < 3 or c[0] not in CHROMS:
                continue
            by.setdefault(c[0], []).append((int(c[1]), int(c[2])))
    for c in by:
        a = np.array(sorted(by[c]))
        by[c] = a
    return by


def overlaps(peaks_by_chrom, chrom, s, e):
    a = peaks_by_chrom.get(chrom)
    if a is None or len(a) == 0:
        return False
    # peak.start < e and peak.end > s
    starts = a[:, 0]
    idx = np.searchsorted(starts, e)  # peaks with start < e are [:idx]
    if idx == 0:
        return False
    ends = a[:idx, 1]
    return bool(np.any(ends > s))


def load_jadhav(n=80, seed=7):
    """Sample developmental Jadhav enhancers (mm10) + matched random controls."""
    rows = []
    for fn in ("E12.5_LMRs.bed", "E16.5_LMRs.bed"):
        p = Path("data/jadhav_mouse") / fn
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                c = line.split("\t")
                if len(c) < 3 or c[0] not in CHROMS:
                    continue
                rows.append((c[0], int(c[1]), int(c[2])))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    enh = [rows[i] for i in idx]
    # controls: same chrom + width, shifted by a large random offset
    ctrl = []
    for (c, s, e) in enh:
        w = e - s
        shift = int(rng.integers(2_000_000, 40_000_000)) * (1 if rng.random() < .5 else -1)
        ns = max(1, s + shift)
        ctrl.append((c, ns, ns + w))
    return enh, ctrl


def openness_matrix(intervals, beds):
    """rows=intervals, cols=stages -> 1 if the interval overlaps a peak at that stage."""
    M = np.zeros((len(intervals), len(beds)), dtype=int)
    for j, bed in enumerate(beds):
        if bed is None:
            continue
        for i, (c, s, e) in enumerate(intervals):
            M[i, j] = overlaps(bed, c, s, e)
    return M


def opening_time(M):
    """First stage index that is open; NaN if never open."""
    out = np.full(M.shape[0], np.nan)
    for i in range(M.shape[0]):
        w = np.where(M[i] == 1)[0]
        if len(w):
            out[i] = STAGE_T[w[0]]
    return out


def main():
    result = {}
    for tissue in TISSUES:
        print("=" * 70); print(tissue)
        accs = discover_one_bed_per_stage(tissue)
        print("  stage beds:", dict(zip([s[:4] for s in STAGES], accs)))
        beds = [fetch_bed(a) if a else None for a in accs]

        # ASSEMBLY / overlap sanity: housekeeping windows must be open at all stages
        pos_iv = [(c, s, e) for (_, c, s, e) in POS_CTRL]
        Mp = openness_matrix(pos_iv, beds)
        pos_frac = float((Mp.sum(axis=1) > 0).mean())
        print(f"  [sanity] housekeeping windows ever-open: {pos_frac:.2f} "
              f"({'assembly+overlap OK' if pos_frac >= 0.75 else 'CHECK ASSEMBLY'})")

        enh, ctrl = load_jadhav()
        Me = openness_matrix(enh, beds)
        Mc = openness_matrix(ctrl, beds)

        frac_e = Me.mean(axis=0)          # fraction of enhancers open per stage
        frac_c = Mc.mean(axis=0)
        ever_e = (Me.sum(axis=1) > 0).mean()
        ever_c = (Mc.sum(axis=1) > 0).mean()
        ot_e = opening_time(Me)
        n_temporal = int(np.sum(~np.isnan(ot_e)))
        spread = float(np.nanstd(ot_e)) if n_temporal else 0.0

        print(f"  Jadhav enhancers open at each stage:  " +
              "  ".join(f"E{t}:{f:.2f}" for t, f in zip(STAGE_T, frac_e)))
        print(f"  random controls open at each stage:   " +
              "  ".join(f"E{t}:{f:.2f}" for t, f in zip(STAGE_T, frac_c)))
        print(f"  ever-open: Jadhav {ever_e:.2f} vs control {ever_c:.2f}"
              f"  ({ever_e/max(ever_c,1e-9):.1f}x)")
        print(f"  temporal-resolved enhancers: {n_temporal}/{len(enh)},"
              f" opening-time spread {spread:.2f} days")

        result[tissue] = dict(
            stages=STAGE_T, accessions=accs,
            frac_open_enh=frac_e.tolist(), frac_open_ctrl=frac_c.tolist(),
            ever_open_enh=float(ever_e), ever_open_ctrl=float(ever_c),
            n_temporal=n_temporal, opening_time_spread=spread,
            opening_times=[None if np.isnan(x) else float(x) for x in ot_e])

    # --- THE POSITIVE PoC: temporal resolution of the atlas's OWN accessible enhancers
    # (forebrain has all 5 stages). Each accessible peak gets a measured opening time.
    print("=" * 70)
    print("TEMPORAL INDEX on the atlas's own dynamic peaks (forebrain, E11.5-E15.5)")
    accs = discover_one_bed_per_stage("forebrain")
    beds = [fetch_bed(a) if a else None for a in accs]
    rng = np.random.default_rng(3)
    allp = [(c, int(s), int(e)) for b in beds if b is not None
            for c, arr in b.items() for s, e in arr]
    samp_idx = rng.choice(len(allp), size=min(4000, len(allp)), replace=False)
    sample = [allp[i] for i in samp_idx]
    Mm = openness_matrix(sample, beds)
    present = Mm.sum(axis=1)
    constitutive = float((present == 5).mean())
    dynamic = float(((present >= 1) & (present < 5)).mean())
    ot = opening_time(Mm)
    first_open_hist = {f"E{t}": int(np.sum(ot == t)) for t in STAGE_T}
    print(f"  sampled accessible peaks: {len(sample)}")
    print(f"  constitutive (open all 5 stages): {constitutive:.2f}")
    print(f"  DYNAMIC (stage-specific timing):  {dynamic:.2f}  <- the temporal resolution")
    print(f"  first-open stage distribution: {first_open_hist}")
    result["_atlas_temporal_resolution_forebrain"] = dict(
        n=len(sample), constitutive=constitutive, dynamic=dynamic,
        first_open_hist=first_open_hist)

    Path("data").mkdir(exist_ok=True)
    Path("data/encode_temporal_index.json").write_text(json.dumps(result, indent=2))
    print("\nsaved data/encode_temporal_index.json")

    # --- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        t = TISSUES[0]; r = result[t]
        ax[0].plot(STAGE_T, r["frac_open_enh"], "-o", color="#2ca02c", label="Jadhav dev. enhancers")
        ax[0].plot(STAGE_T, r["frac_open_ctrl"], "-o", color="#999999", label="random controls")
        ax[0].set_xlabel("developmental stage (days)"); ax[0].set_ylabel("fraction open")
        ax[0].set_title(f"{t}\naccessibility vs stage"); ax[0].legend(); ax[0].set_ylim(0, 1)

        ots = [x for x in r["opening_times"] if x is not None]
        ax[1].hist(ots, bins=STAGE_T + [16.0], color="#2ca02c", edgecolor="w", align="left")
        ax[1].set_xlabel("opening time (first open stage, days)")
        ax[1].set_ylabel("# enhancers"); ax[1].set_title("per-enhancer OPENING TIME\n(continuous 'when')")

        # heatmap: enhancers x stages, sorted by opening time
        enh, _ = load_jadhav()
        accs = r["accessions"]; beds = [fetch_bed(a) if a else None for a in accs]
        M = openness_matrix(enh, beds)
        ot = opening_time(M)
        order = np.argsort(np.where(np.isnan(ot), 99, ot))
        ax[2].imshow(M[order], aspect="auto", cmap="Greens", interpolation="nearest")
        ax[2].set_xticks(range(len(STAGE_T))); ax[2].set_xticklabels([f"E{t}" for t in STAGE_T])
        ax[2].set_xlabel("stage"); ax[2].set_ylabel("enhancer (sorted by opening time)")
        ax[2].set_title("temporal accessibility index\n(the measured clock)")
        plt.tight_layout()
        plt.savefig("data/encode_temporal_index.png", dpi=140)
        print("saved data/encode_temporal_index.png")
    except Exception as ex:
        print("figure skipped:", repr(ex))


if __name__ == "__main__":
    main()
