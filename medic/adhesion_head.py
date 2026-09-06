"""
adhesion_head.py -- THE FIFTH BEHAVIOUR (cycle 45; Miles 2026-09-05: "an adhesion head from the
genome"). Division makes mass, differentiation makes identity, motility moves cells, ADHESION
connects them, death prunes them.

Every hand-written structural law in the compiler (SOX9 condensation, tube recruitment, capsule
closures, girdle attachment, the muscle O-I line, anoikis-on-adhesion-loss) is differential adhesion
under an assumed name (Steinberg 1963; cadherin code, Eph-ephrin boundaries, integrin-ECM anchorage).
This module promotes it to a class, INSTRUMENT FIRST:

  canon_adjacency()   -- the TARGET adjacency matrix, computed from the canon itself: the bp3d organ
                         meshes sit in-situ in one mm frame, so which-parts-touch is measurable from
                         the reference with no new data. Contact threshold chosen from the measured
                         gap histogram (touching pairs sit at ~0-3 mm, separated pairs 10+ mm).
  model_adjacency()   -- the model's REALISED adjacency on the matured standing cloud, per grays
                         organ family; dimensionless contact = cross-family 2nd-percentile gap under
                         1.5x the pair's own median cell spacing.
  trace()             -- the relational trace: canon vs model contacts -> missing contacts (relations
                         to earn) and false contacts (separations to earn). THE instrument on which
                         structural resemblance is measured, not hoped.
  apply()             -- v1 mechanism: per-family affinity FITTED FROM THE CANON ADJACENCY (the
                         declared-anchor pattern -- genome derivation from cadherin-family expression
                         through ABC/SEdb/AlphaGenome is the successor); bounded family-level
                         attraction closes missing contacts on the matured cloud (families move as
                         families; displacement capped; guarded by the frozen benchmark).

Run:  venv_win_new/Scripts/python.exe -m medic.adhesion_head            (canon target + report)
      venv_win_new/Scripts/python.exe -m medic.adhesion_head --trace    (+ build the model, trace it)
Out:  data/canon/adjacency_canon.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.spatial import cKDTree

OUT = "data/canon/adjacency_canon.json"
CONTACT_MM = 4.0            # canon contact threshold; see the gap histogram printed by canon_adjacency
MODEL_K = 1.5               # model contact = cross-family p2 gap < MODEL_K x pair median cell spacing

# the organ pairs are read from the wired map's organ rows (bp3d, in-situ). Sides matter for the
# adjacency truth (liver touches the RIGHT kidney), but the model's grays composites are per-organ,
# so the canon matrix is folded to the same granularity the model exposes.
_ORGAN_KEYS = None


def _organ_specs():
    wired = json.load(open("data/canonical_map.json"))
    out = {}
    for key, spec in wired.items():
        if key.startswith("organ__") and spec[0] == "bp3d":
            out[key.split("__", 1)[1]] = spec
    return out


def canon_adjacency(verbose=True):
    """Pairwise surface gaps between the in-situ canon organ meshes -> the TARGET contact matrix."""
    from medic.canonical_scorecard import load_spec
    specs = _organ_specs()
    clouds = {}
    for name, spec in specs.items():
        P = load_spec(spec)
        if len(P) >= 50:
            clouds[name] = P
    names = sorted(clouds)
    gaps = {}
    trees = {n: cKDTree(clouds[n]) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d, _ = trees[b].query(clouds[a][:: max(1, len(clouds[a]) // 4000)], k=1)
            gaps[f"{a}|{b}"] = round(float(np.percentile(d, 2)), 2)
    contacts = {k: g <= CONTACT_MM for k, g in gaps.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(contact_mm=CONTACT_MM, gaps_mm=gaps, contacts=contacts), open(OUT, "w"), indent=1)
    if verbose:
        touch = sorted((g, k) for k, g in gaps.items() if contacts[k])
        apart = sorted((g, k) for k, g in gaps.items() if not contacts[k])
        print(f"CANON ADJACENCY ({len(names)} organs, {len(gaps)} pairs) -> {OUT}")
        print(f"  contacts (gap <= {CONTACT_MM} mm): {len(touch)}")
        for g, k in touch:
            print(f"    {k:34s} {g:6.1f} mm")
        print(f"  separated: {len(apart)} (nearest miss {apart[0][1]} at {apart[0][0]:.1f} mm)" if apart else "")
    return gaps, contacts


def _model_organ_clouds(Q, F):
    """The grays organ composites on the given cloud (same folding the scorecard sees)."""
    from medic.grays_scorecard import collect_all_parts
    R = {"base": Q, "F": F}
    return {name.split(":", 1)[1]: P for name, cat, P, meta in collect_all_parts(R) if cat == "organ"}


def model_adjacency(Q, F, verbose=True):
    orgs = _model_organ_clouds(Q, F)
    names = sorted(orgs)
    trees = {n: cKDTree(orgs[n]) for n in names}
    spacing = {}
    for n in names:
        d, _ = trees[n].query(orgs[n][:: max(1, len(orgs[n]) // 2000)], k=2)
        spacing[n] = float(np.median(d[:, 1]))
    gaps, contacts = {}, {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d, _ = trees[b].query(orgs[a][:: max(1, len(orgs[a]) // 4000)], k=1)
            g = float(np.percentile(d, 2))
            thr = MODEL_K * 0.5 * (spacing[a] + spacing[b])
            gaps[f"{a}|{b}"] = round(g, 4)
            contacts[f"{a}|{b}"] = g <= thr
    if verbose:
        print(f"model adjacency: {len(names)} organ families, "
              f"{sum(contacts.values())} contacts of {len(contacts)} pairs")
    return gaps, contacts


# CONTAINED subfamilies are NOT organs of the adjacency tier: LiverHaem lives INSIDE the liver and
# Nephron inside the kidney -- their canon rows are SYNONYM MESHES (gap 0.0 = the same mesh), i.e.
# containment statements, not contacts. Treating them as independent organs made the v1 mechanism
# pull LiverHaem toward its synonym "contacts" while the containment step pulled it home (cycle 46).
# Adjacency folds them into their containers; containment is the containment step's job, and its
# instrument is the family-dispersion / anoikis ledger.
_CONTAINED = {"LiverHaem": "Liver", "Nephron": "Kidney"}


def _fold_canon_to_model(canon_contacts, model_names):
    """Fold sided/renamed canon rows onto the model's organ naming (Left Lung+Right... -> Lung etc.).
    A folded pair is in contact if ANY of its member pairs touch."""
    def fold(n):
        n2 = n.replace("Left ", "").replace("Right ", "").strip()
        n2 = _CONTAINED.get(n2, n2)
        return {"Gut": "Gut", "Heart": "Heart"}.get(n2, n2)
    out = {}
    for k, c in canon_contacts.items():
        a, b = k.split("|")
        fa, fb = fold(a), fold(b)
        if fa == fb or fa not in model_names or fb not in model_names:
            continue
        kk = "|".join(sorted((fa, fb)))
        out[kk] = out.get(kk, False) or c
    return out


def trace(Q=None, F=None, verbose=True):
    """The relational trace: canon contacts vs model contacts on the standing specimen."""
    if not os.path.exists(OUT):
        canon_adjacency()
    cj = json.load(open(OUT))
    if Q is None:
        from medic.integrated_body import assemble, mature_parts
        M = mature_parts(assemble())
        Q, F = M["base"], M["F"]
    mg, mc = model_adjacency(Q, F, verbose=verbose)
    cc = _fold_canon_to_model(cj["contacts"], {k.split("|")[0] for k in mc} | {k.split("|")[1] for k in mc})
    both = sorted(set(cc) & set(mc))
    hits = [k for k in both if cc[k] and mc[k]]
    missing = [k for k in both if cc[k] and not mc[k]]
    false = [k for k in both if not cc[k] and mc[k]]
    seps = [k for k in both if not cc[k] and not mc[k]]
    score = (len(hits) + len(seps)) / max(1, len(both))
    if verbose:
        print(f"\nRELATIONAL TRACE ({len(both)} shared pairs): agreement {100 * score:.1f}%")
        print(f"  contacts held: {len(hits)}   separations held: {len(seps)}")
        print(f"  MISSING contacts (relations to earn, {len(missing)}): {missing}")
        print(f"  FALSE contacts (separations to earn, {len(false)}): {false}")
    return dict(agreement=round(score, 4), hits=hits, missing=missing, false=false)


# ------------------------------------------------------------------ v1 mechanism
# VISCERA ONLY: the brain's internal contacts belong to the cephalic assembly (moving vesicles by
# adhesion would fight the vault), and organ-vs-Rib contacts are wall relations the packing plant
# owns. The affinity here is FITTED FROM THE CANON ADJACENCY (declared anchor; the successor reads
# cadherin-family expression per fate through ABC/SEdb/AlphaGenome).
_VISCERA = {"Heart", "Lung", "Liver", "LiverHaem", "Kidney", "Nephron", "Spleen", "Stomach",
            "Gut", "Bladder", "Pancreas", "Adrenal", "Thymus"}
_CAP_FRAC = 0.020           # max total displacement per family, as a fraction of stature
_TARGET_GAP = 0.5           # close a missing contact to this multiple of the pair's cell spacing
# ANCHORED families (cycle 82): the bladder is held in the pelvis by the pubic symphysis, the pelvic
# floor and the urachus -- a MIDLINE organ at a REGISTERED height. Once it had real cells (the cycle-82
# allocation) the "small organ moves toward large" rule dragged it 0.05 stature LEFT and 0.05 UP toward
# the gut coil, undoing the standing register. Per-family free-axis mask (AP, DV, ML): the bladder may
# settle only in DV; its partners come to it.
_FREE_AXES = {"Bladder": np.array([0.0, 1.0, 0.0])}


def _family_masks(F):
    """Model organ-family masks at the same folding the trace uses (grays composites)."""
    from medic.unified_embryo import FIDX
    from medic.subhead_program import composites as _sub_comp
    comps = {"Heart": ("Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow"),
             "Gut": ("Gut", "Foregut", "Hindgut")}
    comps.update(_sub_comp())
    children = {c for parts in comps.values() for c in parts[1:]}
    masks = {}
    for organ in _VISCERA:
        if organ in _CONTAINED:
            continue                                      # contained subfamilies fold into their container
        fates = list(comps.get(organ, (organ,)))
        fates += [sub for sub, container in _CONTAINED.items() if container == organ]
        ids = [FIDX[c] for c in fates if c in FIDX]
        if organ in children and organ not in comps:
            continue
        m = np.isin(F, ids)
        if m.sum() >= 40:
            masks[organ] = m
    return masks


def apply(Q, F, iters=3, verbose=True):
    """Close the trace's MISSING visceral contacts and open its FALSE ones by bounded family-level
    translation (families move as families; per-family displacement capped at _CAP_FRAC of stature
    PER PASS; small organ moves toward large, not the reverse -- the adrenal rides to the kidney).
    Iterative RELAXATION: one pass cancels multi-pair pulls under the cap (the liver is pulled toward
    kidney, lung and adrenal at once), so the settle runs a few passes with the trees recomputed --
    the Steinberg sorting dynamic at the family level."""
    if not os.path.exists(OUT):
        canon_adjacency(verbose=False)
    cj = json.load(open(OUT))
    Q = np.asarray(Q, float).copy()
    masks = _family_masks(F)
    cc = _fold_canon_to_model(cj["contacts"], set(masks))
    stat = float(np.ptp(Q[:, 0]))
    cap = _CAP_FRAC * stat
    for it in range(iters):
        moves = {n: np.zeros(3) for n in masks}
        trees = {n: cKDTree(Q[m]) for n, m in masks.items()}
        spacing = {}
        for n, m in masks.items():
            d, _ = trees[n].query(Q[m][:: max(1, m.sum() // 1500)], k=2)
            spacing[n] = float(np.median(d[:, 1]))
        report = []
        for pair, want in sorted(cc.items()):
            a, b = pair.split("|")
            if a not in masks or b not in masks:
                continue
            Pa = Q[masks[a]]
            sub = Pa[:: max(1, len(Pa) // 3000)]
            d, idx = trees[b].query(sub, k=1)
            j = int(np.argmin(d))
            gap = float(d[j])
            thr = MODEL_K * 0.5 * (spacing[a] + spacing[b])
            have = gap <= thr
            if want == have:
                continue
            direction = Q[masks[b]][idx[j]] - sub[j]
            u = direction / (np.linalg.norm(direction) + 1e-12)
            if want:                                         # MISSING contact: close to _TARGET_GAP
                move = max(0.0, gap - _TARGET_GAP * 0.5 * (spacing[a] + spacing[b]))
            else:                                            # FALSE contact: open past the threshold
                move = -(thr * 1.2 - gap)
            wa = masks[b].sum() / (masks[a].sum() + masks[b].sum())   # small family moves more
            moves[a] += u * move * wa
            moves[b] -= u * move * (1 - wa)
            report.append((pair, "close" if want else "open", round(gap / stat, 4)))
        moved = 0
        for n, v in moves.items():
            if n in _FREE_AXES:
                v = v * _FREE_AXES[n]                          # anchored family: only its free axes move
            nv = np.linalg.norm(v)
            if nv > cap:
                v = v * (cap / nv)
            if nv > 1e-9:
                Q[masks[n]] += v
                moved += 1
        if verbose:
            print(f"  [adhesion] pass {it + 1}: {len(report)} pair corrections, {moved} families moved")
        if not report:
            break
    return Q


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true")
    a = ap.parse_args()
    canon_adjacency()
    if a.trace:
        trace()
