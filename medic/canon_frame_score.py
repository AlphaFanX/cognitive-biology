"""canon_frame_score.py -- FRAME-TO-FRAME comparison of the movie against its staged canonical reference
(Miles 2026-08-30: "then compare frame to frame, please").

For EVERY movie frame, pick the same reference the viewer's canon toggle shows (Carnegie stage across the
cloud phase, parametric fetal/child stages across the mesh phase, adult at the end) and score the model's
cloud against it: whole-body D2 shape match + per-organ-group D2 (the same 8 groups / colors). The output
curve is the stage-to-stage compare-and-correct instrument: dips = where the model departs from the
reference at that age.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.canon_frame_score
Out: data/organ_cascade/canon_frame_score.{json,png}
"""
from __future__ import annotations
import json, os
import numpy as np

from zlib import crc32

from medic.canonical_atlas import d2_signature, _sub, surface_pts
from medic.unified_embryo import FIDX
from medic.subhead_program import expand_names

SCORER = "v4-cnsparity"  # v4 (2026-09-01, the embryo loop): for CARNEGIE-stage frames the "brain" trace
# scores the model's whole CNS (brain families + Spinal Cord/Nervous System/DRG) -- the reference "brain"
# group substrings (neural/spinal_gangl/...) span the WHOLE neural tube+canal at embryonic stages, so the
# old trace compared a forebrain cap against the full tube (the know-what-object-you-measure class, again).
# Parametric/adult refs keep the brain-organ families (their reference IS the brain organ).
# v2 (2026-08-30): surface-normalised clouds + per-(stage,organ) seeded rng.
# v3 (2026-08-31, the scorer audit): the "skin" trace now scores the frame's actual SKIN SURFACE
# verts (fr["skin"], the closed marching-cubes surface the eyes judge and the form work shapes)
# against the reference body ghost -- v2 scored the Skin-FATE cells (the patchy epidermal shell
# subset, ~5k display cells), a different object entirely: the fast cache-path read 88 where the
# v2 trace read 70, and the SURFACE was the honest one. Fate-cell fallback for cloud frames
# (no surface exists yet). Traces are NOT comparable across scorer versions.
#   v1 defects (measured): one shared rng stream coupled every trace (untouched fates swung +-3 when another
#   organ's cells moved -- the sampling draws shifted downstream); solid model clouds vs surface references
#   mismatched interior mass (preferred the fused kidney over the measured pair). Traces are NOT comparable
#   across scorer versions.


def _rng(*key):
    return np.random.default_rng(crc32("|".join(map(str, key)).encode()))


def _sig(P, n, *key):
    r = _rng(*key)
    return d2_signature(_sub(surface_pts(P), n, r), r)

# model fate parents per reference organ group (expand_names folds in the sub-head children)
MODEL_GROUPS = {
    "brain":   ["Forebrain", "Telencephalon", "Midbrain", "Hindbrain", "Cerebellum", "OlfactoryBulb"],
    "lung":    ["Lung"],
    "heart":   ["Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow"],
    "liver":   ["Liver", "LiverHaem"],
    "stomach": ["Stomach", "Duodenum", "Foregut"],
    "spleen":  ["Spleen"],
    "kidney":  ["Kidney", "Nephron"],
    "gut":     ["Gut", "Hindgut"],
    "skin":    ["Skin"],
}


def _pick(frames, i, emb, by_label):
    fr = frames[i]
    if fr.get("phase") == "cloud" and emb:
        cloud_idx = [k for k, f in enumerate(frames) if f.get("phase") == "cloud"]
        k = cloud_idx.index(i) if i in cloud_idx else 0
        return emb[min(len(emb) - 1, int(k / len(cloud_idx) * len(emb)))]
    st = (fr.get("stage") or "").lower()
    for w, lab in [("fetus", "FETUS·param"), ("newborn", "NEWBORN·param"), ("infant", "INFANT·param"),
                   ("adolescent", "ADOL·param"), ("child", "CHILD·param")]:
        if w in st and lab in by_label:
            return by_label[lab]
    return by_label["ADULT"]


def run():
    frames = json.load(open("data/movie/human_movie_frames.json"))["frames"]
    st = json.load(open("data/movie/canon_stages.json"))["stages"]
    ad = json.load(open("data/movie/canon_reference.json"))
    ad2 = dict(cs=-1, label="ADULT", param=False, skin=ad["skin"], organs=ad["organs"])
    emb = [s for s in st if not s.get("param")]
    by_label = {s["label"]: s for s in st}
    by_label["ADULT"] = ad2
    gids = {g: [FIDX[n] for n in expand_names(names) if n in FIDX] for g, names in MODEL_GROUPS.items()}
    cns_ids = sorted(set(gids["brain"]) | {FIDX[n] for n in expand_names(
        ["Spinal Cord", "Nervous System", "DRG"]) if n in FIDX})      # v4: embryonic brain-ref parity

    # cache reference clouds + signatures per stage label
    ref_cache = {}

    def ref_sigs(s):
        if s["label"] in ref_cache:
            return ref_cache[s["label"]]
        ghost = np.asarray(s["skin"], float).reshape(-1, 3)
        allp, org = [ghost], {}
        for o in s["organs"]:
            P = np.asarray(o["xyz"], float).reshape(-1, 3)
            allp.append(P)
            if len(P) >= 30:
                org[o["label"]] = _sig(P, 3000, s["label"], o["label"])
        org["skin"] = _sig(ghost, 3000, s["label"], "skin")   # the body ghost = the skin/outer-form target
        whole = _sig(np.vstack(allp), 4000, s["label"], "_whole")
        ref_cache[s["label"]] = (whole, org)
        return ref_cache[s["label"]]

    rows = []
    for i, fr in enumerate(frames):
        if not fr.get("xyz"):
            continue
        P = np.asarray(fr["xyz"], float).reshape(-1, 3)
        F = np.asarray(fr["fate"], int) if fr.get("fate") else None
        s = _pick(frames, i, emb, by_label)
        whole_ref, org_ref = ref_sigs(s)
        d2 = float(1 - 0.5 * np.abs(whole_ref - _sig(P, 4000, s["label"], "_whole")).sum())
        row = dict(frame=i, phase=fr.get("phase"), ref=s["label"], scorer=SCORER,
                   whole=round(100 * d2, 1), organs={})
        if F is not None:
            for g, ids in gids.items():
                if g not in org_ref:
                    continue
                if g == "skin" and fr.get("skin"):
                    Pg = np.asarray(fr["skin"], float).reshape(-1, 3)   # the real skin SURFACE (v3)
                elif g == "skin" and fr.get("phase") == "anatomy":
                    continue    # skin dissolved in the reveal: nothing to score (the fate-cell
                    #             fallback here polluted the adult-end trace with a different object)
                else:
                    use = cns_ids if (g == "brain" and not s.get("param") and s.get("cs", -1) > 0) else ids
                    m = np.isin(F, use)
                    if m.sum() < 30:
                        continue
                    Pg = P[m]
                dg = float(1 - 0.5 * np.abs(org_ref[g] - _sig(Pg, 3000, s["label"], g)).sum())
                row["organs"][g] = round(100 * dg, 1)
        rows.append(row)
        if i % 20 == 0:
            print(f"frame {i:3d} vs {row['ref']:14s} whole {row['whole']:5.1f}%  "
                  + " ".join(f"{g}:{v:.0f}" for g, v in row["organs"].items()))

    # ---- plot ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 7), facecolor="#0d1017")
    ax.set_facecolor("#0d1017")
    x = [r["frame"] for r in rows]
    ax.plot(x, [r["whole"] for r in rows], color="#ffd23a", lw=2.4, label="whole body")
    COLS = {"brain": "#c084fc", "lung": "#7dd3fc", "heart": "#f87171", "liver": "#b45309",
            "stomach": "#fbbf24", "spleen": "#9f1239", "kidney": "#34d399", "gut": "#d6a86b",
            "skin": "#e2e8f0"}
    for g, col in COLS.items():
        xs = [r["frame"] for r in rows if g in r["organs"]]
        ys = [r["organs"][g] for r in rows if g in r["organs"]]
        if len(xs) > 3:
            ax.plot(xs, ys, color=col, lw=1.0, alpha=0.75, label=g)
    seen = set()
    for r in rows:                                          # stage-change markers
        if r["ref"] not in seen:
            seen.add(r["ref"])
            ax.axvline(r["frame"], color="#2a3547", lw=0.7)
            ax.text(r["frame"] + 0.3, 101.5, r["ref"].replace("·param", "*"), rotation=90,
                    color="#8091a8", fontsize=7, va="bottom")
    ax.set_xlabel("movie frame", color="#cbd5e1"); ax.set_ylabel("D2 shape match %", color="#cbd5e1")
    ax.set_ylim(30, 108); ax.tick_params(colors="#8091a8")
    for sp in ax.spines.values():
        sp.set_color("#2a3547")
    ax.legend(loc="lower right", facecolor="#141a26", labelcolor="#e2e8f0", edgecolor="#2a3547", ncol=3)
    ax.set_title("MODEL vs the STAGED CANONICAL REFERENCE — frame to frame (* = parametric stage)",
                 color="#e8eef4")
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/canon_frame_score.png", dpi=130, facecolor="#0d1017")
    json.dump(rows, open("data/organ_cascade/canon_frame_score.json", "w"), indent=0)
    wl = [r["whole"] for r in rows]
    print(f"\nwhole-body D2 vs reference: mean {np.mean(wl):.1f}%  min {min(wl):.1f}% "
          f"(frame {rows[int(np.argmin(wl))]['frame']}, ref {rows[int(np.argmin(wl))]['ref']})")
    print("saved data/organ_cascade/canon_frame_score.{json,png}")


if __name__ == "__main__":
    run()
