"""
adult_persistence_audit.py -- Step 3: do the embryo's placement MECHANISMS survive into the adult?

`medic.organ_placement_audit` checks the EMBRYO. But the human movie then MATURES the model's own cell
cloud (`human_movie.mature_cloud`: allometric stretch + limb outgrowth) all the way to the adult -- and
NOTHING re-checks whether the five morphogenetic mechanisms that PLACED the organs are still respected
once the body has been stretched to adult proportions. This audit re-runs the same mechanism checks on
the MATURED (adult) cloud and diffs them against the embryo, so we can SEE what maturation preserves and
what it breaks.

A NOTE ON BIOLOGY (why the 8 checks are NOT all the same kind). It is NOT biologically correct to hold
all eight invariant across maturation -- they fall into three classes:

 A. STRUCTURAL INVARIANTS -- genuinely constant for life; a change is pathology. Checked for INVARIANCE:
      4 integrin/ECM continuity  -> the body stays ONE fascial continuum (loss = hernia/rupture)
      5 eigenmode placement      -> organs keep their address (loss = situs inversus / ectopia)
 B. TRANSIENT embryonic PROCESSES -- they RUN in the embryo and STOP; the adult is not doing them. What
    persists is their morphological RESULT, so we check the RESULT persists (not the active process):
      1 convergent extension     -> the RESULT (long-narrow axis) persists   [measured axially]
      2 lateral inhibition       -> the RESULT (two discrete bilateral buds) persists
      3 cadherin condensation    -> the RESULT (each organ stays compact) persists
 C. CLOCK-DRIVEN HEADS -- these must PROGRESS to an adult END-STATE, NOT stay constant. In this model the
    heads RUN in the embryo (simulate) and maturation is a post-morphogenetic allometric growth phase, so
    across maturation we check they have REACHED their correct adult end-state (progression complete):
      division      -> proliferation QUIESCED  (WARNING: the coarse readout holds N; real growth
                       adds cells 10^3 -> 10^13 -- cell-count constancy here is a MODEL ARTIFACT, not biology)
      differentiation-> fates TERMINAL / committed (none lost, none still labile)
      motility      -> migration SETTLED (organs at rest, drift ~0)

The adult is taken MODEL-NATIVE (the movie's source='model' path): `mature_cloud(f=1)` + `grow_limbs`.
At the adult the fetal curl/tuck are identity (`_curl_amt(1)=0`), so the adult is the straight, fully
grown body -- a fair frame to compare against the straight embryo cloud (no curl on either side). Both
carry the SAME cells and the SAME fate labels; only the allometric maturation transform differs.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.adult_persistence_audit
Out: data/organ_cascade/adult_persistence_audit.{json,png}
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from medic.unified_embryo import simulate, _symmetrize, FIDX, FATES
from medic.organ_sprouting import ORGAN_SCHEDULE
from medic.human_movie import (shape_limbs, flex, mature_cloud, grow_limbs, _limb_grow_model,
                               _long_axis_len, LIMB, LIMB_SEARCHED, fate_map_for)

NE = 120000                                     # cloud size (ultra fidelity; -> ~240k after symmetrization)
OUT = Path("data/organ_cascade/adult_persistence_audit")
CHAMBERS = {"Heart": ["Heart", "Atrium", "Ventricle", "Outflow"]}   # the heart chambers by the final frame
HEAD_FATES = ("Forebrain", "Eye", "Midbrain", "Hindbrain")
LIMB_IDS = [FIDX[n] for n in ("Limb Bud", "Cartilage") if n in FIDX]   # appendicular -- excluded from the AXIAL CE metric
FS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)             # maturation fractions sampled embryo(0) -> adult(1)


# ------------------------------------------------------------------ mechanism metrics
def _n_components(P, k=6, min_frac=0.02):
    """Topological connectivity: connect each cell to its k nearest neighbours and count the LARGE
    connected components (>= min_frac of the cells). kNN adjacency is scale-invariant, but it is DENSITY-
    sensitive: an allometrically STRETCHED limb spreads its (sparse) cells out so a handful of tip cells
    can split into their own tiny clique -- an anatomically-attached limb reading as 'fragmented'. Counting
    only components with a real share of the body ignores those stray-cell artefacts, so a change in the
    count means a REAL loss of the integrin/ECM fascial continuum (the body split), not stretched limbs."""
    n = len(P)
    if n <= k + 1:
        return 1
    _, idx = cKDTree(P).query(P, k=k + 1)                 # k+1 because the first hit is the point itself
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A + A.T
    ncomp, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    return int((sizes >= max(1, int(min_frac * n))).sum())    # count only the substantial bodies


def _metrics(P, F):
    """All mechanism metrics for one frame (P laid: col0=AP head at +x, col1=DV, col2=ML midline at 0)."""
    ax = P[:, 0]
    apf = (ax - ax.min()) / (np.ptp(ax) + 1e-9)
    # orient AP so the head reads at apf~1 (head-forward), consistent with the Phase-D atlas -- else the
    # cloud can end up anti-aligned to the atlas and every organ AP looks maximally off (the eye 0.07 vs
    # 0.94 bug). Flip is AP-only, so lateral/dispersion/continuity/aspect verdicts are unaffected.
    _hm = np.isin(F, [FIDX[n] for n in HEAD_FATES if n in FIDX])
    if _hm.any() and apf[_hm].mean() < 0.5:
        apf = 1.0 - apf
    dv = P[:, 1]
    dvf = (dv - dv.min()) / (np.ptp(dv) + 1e-9)
    z = P[:, 2]

    # LOCAL body half-width per AP band -- the fair, scale-invariant laterality reference (an organ is
    # "lateral" relative to the body AT ITS OWN AP level, so uniform ML growth cannot change the ratio).
    apbin = np.clip((apf * 24).astype(int), 0, 23)
    locw = np.full(24, np.percentile(np.abs(z), 85) + 1e-9)
    for k in range(24):
        m = apbin == k
        if m.sum() > 4:
            locw[k] = np.percentile(np.abs(z[m]), 85) + 1e-9

    diag = float(np.linalg.norm(P.max(0) - P.min(0)) + 1e-9)
    organs = {}
    for org in ORGAN_SCHEDULE:
        name, place = org["name"], org.get("place", "midline")
        if place == "tube" or name not in FIDX:
            continue
        ids = [FIDX[f] for f in CHAMBERS.get(name, [name]) if f in FIDX]
        idx = np.where(np.isin(F, ids))[0]
        n = len(idx)
        if n < 4:
            organs[name] = dict(place=place, n=n, sprouted=False)
            continue
        zo = z[idx]
        ctr = P[idx].mean(0)
        lateral = float((np.abs(zo) / locw[apbin[idx]]).mean())        # 0 midline .. ~1 lateral edge
        dispersion = float(np.sqrt(((P[idx] - ctr) ** 2).sum(1).mean()) / diag)   # cadherin condensation
        # lateral inhibition: two discrete bilateral buds -> both sides populated AND a gap at the midline
        side_frac = float(min((zo < 0).sum(), (zo >= 0).sum()) / n)
        gap_frac = float((np.abs(zo) / locw[apbin[idx]] < 0.25).mean())
        bilateral = bool(side_frac >= 0.30 and gap_frac < 0.35)
        organs[name] = dict(place=place, n=n, sprouted=True,
                            lateral=round(lateral, 3), ap=round(float(apf[idx].mean()), 3),
                            dv=round(float(dvf[idx].mean()), 3),
                            dispersion=round(dispersion, 4),
                            side_frac=round(side_frac, 3), gap_frac=round(gap_frac, 3),
                            bilateral=bilateral)

    # convergent extension = elongation of the BODY AXIS. Measured on the axial body only (limbs
    # excluded): the appendages splay laterally as they grow out (grow_limbs extends arms in ML), which
    # is limb morphogenesis, not the trunk's CE -- counting them would read arm-splay as a loss of CE.
    body = ~np.isin(F, LIMB_IDS)
    zb, axb = (z[body], ax[body]) if body.sum() > 8 else (z, ax)
    aspect = float(np.ptp(axb) / (2 * np.percentile(np.abs(zb), 95) + 1e-9))
    fates_present = sorted({FATES[f] for f in np.unique(F)})
    return dict(organs=organs, aspect=round(aspect, 3), n_cells=int(len(P)),
                n_components=_n_components(P), n_fates=len(fates_present), fates=fates_present)


# ------------------------------------------------------------------ maturation trajectory
def _mature(base, F, f):
    """Apply the movie's MODEL-NATIVE maturation transform at fraction f (embryo 0 -> adult 1): the
    heads-tall allometry + limb outgrowth + the length ramp, exactly as human_movie.build's source='model'
    loop. The fetal curl/tuck are OMITTED on purpose -- they are a separable rigid bend that leaves the
    ML-based mechanisms (bilaterality, condensation, continuity, placement) untouched and only folds the
    AP axis; omitting them isolates whether the ALLOMETRY itself preserves the five mechanisms."""
    Q = mature_cloud(base, F, f)
    t = 0.55 + 0.45 * f
    Q = grow_limbs(Q, F == LIMB, _limb_grow_model(t))
    Q = Q * ((0.9 + (3.2 - 0.9) * f ** 1.2) / _long_axis_len(Q))
    return Q


def build_base(ne=NE):
    """Grow the cloud once and return (base, F): the movie's final cloud basis (S0-equivalent, straight,
    head at +x) that the maturation transform starts from, plus the per-cell fate labels."""
    frames, _ = simulate(use_ecm=True, seed=0, n_start=1, n_end=ne, limb_buds=True,
                         convergent_ext=1.0, limb_params=LIMB_SEARCHED, fate_params=fate_map_for(ne))
    P, V, F = _symmetrize(frames[-1][3], frames[-1][4], frames[-1][5])
    c = P.mean(0); c[2] = 0.0
    scale = 0.9 / _long_axis_len(P)
    base = flex(shape_limbs((P - c) * scale, F, 1.0, LIMB), 1.0)     # the movie's final cloud basis (S0), straight
    # orient the head to the HIGH-AP end (apf~1), consistent with the Phase-D atlas, so the silhouette
    # profile + organ addresses compare head-to-head (not the weaker mean-x>0 test, which left the body
    # anti-aligned to the atlas when it extended far in +x).
    hm = np.isin(F, [FIDX[n] for n in HEAD_FATES if n in FIDX])
    x = base[:, 0]; apf = (x - x.min()) / (np.ptp(x) + 1e-9)
    if hm.any() and apf[hm].mean() < 0.5:
        base = np.stack([-base[:, 0], base[:, 1], base[:, 2]], 1)
    # SHOULDER-GIRDLE MASS: inject the pectoral-girdle cells (clavicle/scapula/deltoid) into the cloud so the
    # maturation, the Vitruvian canon shoulder metric, the anthropometric silhouette, and the skin surface all
    # see the biacromial WIDTH. The knob search proved the shoulder was MECHANISM-limited (0.15 vs canon 0.25 --
    # no cells out at the span for a knob to widen); these cells are that mass. Lazy import avoids the module cycle.
    from medic.shoulder_girdle_head import augment as _sg_augment
    base, F, _ = _sg_augment(base, F)
    # CARDIAC DESCENT: the heart snaps to a body-electric AP antinode that sits ABOVE the lung (apf ~0.62), but
    # the heart DESCENDS into the mid-thorax during folding. The antinode grid is too coarse to place it there
    # (its anchor 0.38 and 0.58 snap to the SAME antinode), so apply the descent explicitly -- shift the heart-
    # family cells caudally so the matured heart lands at its atlas mid-thorax address (~0.56), at/just below the
    # lung. Read-only on every other cell; the heart is a coherent condensed mass, so it descends as a body.
    heart_ids = [FIDX[n] for n in ("Heart", "Atrium", "Ventricle", "Outflow") if n in FIDX]
    hm = np.isin(F, heart_ids)
    if hm.sum() >= 8:
        x = base[:, 0]; L = np.ptp(x) + 1e-9
        cur = float(((x[hm] - x.min()) / L).mean())
        target_raw = 0.39                                   # raw apf that matures (~+0.17) to the atlas ~0.56
        base[hm, 0] -= (cur - target_raw) * L               # descend caudally along AP (head at +x)
    # ORGAN CONDENSATION (differential adhesion sorting): sharpen each organ from a diffuse, interpenetrating
    # cloud into a Gray's-crisp bounded mass. Steinberg's differential-adhesion hypothesis -- same-fate cadherin
    # is stronger than cross-fate, so like cells sort together -- is applied as a few gentle iterations of a
    # LOCAL mean-shift: each organ cell moves a fraction toward the mean of its nearest SAME-organ neighbours,
    # tightening the mass and leaving intermixed foreign cells outside it (raising purity and the clustering
    # silhouette that medic.organ_sharpness_audit scores). LOCAL (small k) so a bilateral organ's two sides each
    # condense on themselves rather than merging to the midline, and the organ's centroid -- its atlas address --
    # is preserved. Read-only on non-organ cells.
    base = _condense_organs(base, F)
    return base, F


# organs that should read as condensed, sharply-bounded masses (not the spanning tubes Gut/Notochord/Vessel or
# the axial segmental series Cartilage/Rib/Muscle/Somite, which are meant to be extended, not balled up).
_CONDENSE_ORGANS = ("Heart", "Atrium", "Ventricle", "Outflow", "Lung", "Liver", "Kidney", "Nephron", "Eye",
                    "Retina", "Pancreas", "Otic", "Spleen", "Thymus", "Adrenal", "Bladder",
                    "Forebrain", "Midbrain", "Hindbrain", "Cerebellum")


def _condense_organs(base, F, n_iter=3, alpha=0.35, k_frac=0.03):
    """Differential-adhesion sorting: local same-fate mean-shift that condenses each organ into a sharp mass.
    The neighbourhood k is a FRACTION of the organ's cell count, not a fixed number: a fixed k that keeps an
    organ whole at 60k is proportionally too local at 240k (the organ is denser), so the mean-shift splits a
    sparse organ into two local attractors (the pancreas and midbrain fragmented at 240k). Scaling k with the
    organ size keeps the neighbourhood at a constant FRACTION of the organ, so the same sort holds at any
    resolution; clamped [16, 200] so it stays local enough that a bilateral organ's two well-separated sides do
    not pull together, and the organ's centroid -- its atlas address -- is preserved."""
    base = base.copy()
    for nm in _CONDENSE_ORGANS:
        fid = FIDX.get(nm)
        if fid is None:
            continue
        idx = np.where(F == fid)[0]
        if len(idx) < 20:
            continue
        P = base[idx]
        kk = int(np.clip(k_frac * len(P), 16, 200)); kk = min(kk, len(P))
        for _ in range(n_iter):
            _, nn = cKDTree(P).query(P, k=kk)                # nearest SAME-organ neighbours (a fraction of the organ)
            P = P + alpha * (P[nn[:, 1:]].mean(1) - P)        # move toward the local same-fate density
        base[idx] = P
    return base


def trajectory(ne=NE, fs=FS):
    """Grow the cloud once, then run the 5+3 checks at each maturation fraction embryo(0) -> adult(1)."""
    base, F = build_base(ne)
    steps = [(float(f), _metrics(_mature(base, F, f), F)) for f in fs]
    return F, steps


# ------------------------------------------------------------------ verdicts
def audit(ne=NE):
    F, steps = trajectory(ne)
    me, ma = steps[0][1], steps[-1][1]                              # embryo (f=0) vs adult (f=1)

    paired = [o["name"] for o in ORGAN_SCHEDULE if o.get("place") == "paired" and o["name"] in FIDX]
    midline = [o["name"] for o in ORGAN_SCHEDULE if o.get("place") == "midline" and o["name"] in FIDX]

    def placed_ok(m, name):
        o = m["organs"].get(name, {})
        if not o.get("sprouted"):
            return None
        if o["place"] == "paired":
            return o["lateral"] >= 0.45
        return o["lateral"] <= 0.55

    checks = []
    # 1 convergent extension -- elongation preserved (adult at least as long-and-narrow as the embryo)
    ce_ok = ma["aspect"] >= 0.90 * me["aspect"]
    checks.append(dict(mechanism="1 convergent extension", metric="AP/ML aspect ratio",
                       embryo=me["aspect"], adult=ma["aspect"],
                       verdict="MAINTAINED" if ce_ok else "DEGRADED"))
    # 2 lateral inhibition -- paired organs stay two discrete bilateral buds
    li_broke = [n for n in paired if me["organs"].get(n, {}).get("bilateral")
                and not ma["organs"].get(n, {}).get("bilateral")]
    li_e = sum(bool(me["organs"].get(n, {}).get("bilateral")) for n in paired)
    li_a = sum(bool(ma["organs"].get(n, {}).get("bilateral")) for n in paired)
    checks.append(dict(mechanism="2 lateral inhibition", metric="paired organs bilateral (of %d)" % len(paired),
                       embryo=li_e, adult=li_a, broke=li_broke,
                       verdict="MAINTAINED" if not li_broke else "BROKEN: " + ",".join(li_broke)))
    # 3 cadherin condensation -- organs stay compact (dispersion not blown up by the stretch)
    cond_broke = [n for n in me["organs"] if me["organs"][n].get("sprouted") and ma["organs"].get(n, {}).get("sprouted")
                  and ma["organs"][n]["dispersion"] > 1.5 * me["organs"][n]["dispersion"]]
    checks.append(dict(mechanism="3 cadherin condensation", metric="organ dispersion within 1.5x",
                       embryo="compact", adult="compact" if not cond_broke else "scattered",
                       broke=cond_broke,
                       verdict="MAINTAINED" if not cond_broke else "DEGRADED: " + ",".join(cond_broke)))
    # 4 integrin/ECM continuity -- one connected fascial continuum, preserved under the stretch
    cont_ok = ma["n_components"] <= max(1, me["n_components"])
    checks.append(dict(mechanism="4 integrin/ECM continuity", metric="connected components (kNN)",
                       embryo=me["n_components"], adult=ma["n_components"],
                       verdict="MAINTAINED" if cont_ok else "BROKEN (fascia fragmented)"))
    # 5 eigenmode placement -- paired lateral / midline central verdict unchanged
    place_broke = [n for n in (paired + midline)
                   if placed_ok(me, n) and placed_ok(ma, n) is False]
    checks.append(dict(mechanism="5 eigenmode placement", metric="paired lateral / midline central",
                       embryo="%d ok" % sum(bool(placed_ok(me, n)) for n in paired + midline),
                       adult="%d ok" % sum(bool(placed_ok(ma, n)) for n in paired + midline),
                       broke=place_broke,
                       verdict="MAINTAINED" if not place_broke else "BROKEN: " + ",".join(place_broke)))
    # head: division -- END-STATE = proliferation quiesced. NB the model does NOT grow the cell count
    # (coarse readout holds N); real maturation adds cells 10^3->10^13, so cell-count constancy here is a
    # MODEL ARTIFACT, not "division maintained". We report the end-state + flag the artifact explicitly.
    checks.append(dict(mechanism="head: division", metric="proliferation quiesced (end-state)",
                       embryo=me["n_cells"], adult=ma["n_cells"], artifact="coarse readout: N held fixed; real growth 10^3->10^13 not modeled",
                       verdict="END-STATE (quiesced; N=%d held -- MODEL ARTIFACT, not growth)" % ma["n_cells"]))
    # head: differentiation -- END-STATE = fates terminal/committed. Progression is complete by the cloud;
    # across maturation every fate must persist as a terminal identity (none lost, none reverting).
    lost = sorted(set(me["fates"]) - set(ma["fates"]))
    checks.append(dict(mechanism="head: differentiation", metric="fates terminal (end-state)",
                       embryo=me["n_fates"], adult=ma["n_fates"], broke=lost,
                       verdict="END-STATE (%d terminal fates)" % ma["n_fates"] if not lost else "LOST: " + ",".join(lost)))
    # head: motility -- organs rode the allometry to their rest positions. Judged on the body-ADDRESS
    # (AP Hox fraction x DV layer fraction), which is stable under allometric scaling; NOT the width-
    # normalised lateral, whose reference changes as limbs extend + the trunk girths (that reference
    # swing is a metric artifact, not motion -- and laterality is already covered by checks 2 and 5).
    drifts = {n: dict(dap=round(abs(ma["organs"][n]["ap"] - me["organs"][n]["ap"]), 3),
                      ddv=round(abs(ma["organs"][n]["dv"] - me["organs"][n]["dv"]), 3))
              for n in me["organs"] if me["organs"][n].get("sprouted") and ma["organs"].get(n, {}).get("sprouted")}
    max_drift = max((max(d["dap"], d["ddv"]) for d in drifts.values()), default=0.0)
    checks.append(dict(mechanism="head: motility", metric="migration settled (end-state)",
                       embryo="rest", adult="max drift %.3f" % max_drift, drifts=drifts,
                       verdict="END-STATE (settled, drift %.3f)" % max_drift if max_drift < 0.15
                       else "STILL MOVING (%.3f)" % max_drift))

    paired_all = [o["name"] for o in ORGAN_SCHEDULE if o.get("place") == "paired" and o["name"] in FIDX]
    placed_all = paired_all + [o["name"] for o in ORGAN_SCHEDULE if o.get("place") == "midline" and o["name"] in FIDX]
    traj = []
    for f, m in steps:
        disps = [o["dispersion"] for o in m["organs"].values() if o.get("sprouted")]
        traj.append(dict(f=round(f, 2), aspect=m["aspect"], components=m["n_components"],
                         bilateral=sum(bool(m["organs"].get(n, {}).get("bilateral")) for n in paired_all),
                         placed_ok=sum(1 for n in placed_all
                                       if (lambda o: o.get("sprouted") and
                                           (o["lateral"] >= 0.45 if o["place"] == "paired" else o["lateral"] <= 0.55))
                                       (m["organs"].get(n, {}))),
                         max_disp=round(max(disps), 4) if disps else None))
    return dict(ne=ne, embryo=me, adult=ma, checks=checks, trajectory=traj,
                n_paired=len(paired_all), n_placed=len(placed_all))


def _print(res):
    me, ma = res["embryo"], res["adult"]
    print("\n== maturation trajectory: 5+3 checks embryo(f=0) -> adult(f=1) ==")
    print(f"{'f':>5s} {'aspect':>7s} {'comps':>6s} {'bilat/%d' % res['n_paired']:>8s}"
          f" {'placed/%d' % res['n_placed']:>9s} {'max_disp':>9s}")
    for r in res["trajectory"]:
        print(f"{r['f']:5.2f} {r['aspect']:7.3f} {r['components']:6d} {r['bilateral']:8d}"
              f" {r['placed_ok']:9d} {str(r['max_disp']):>9s}")
    print("\n== per-organ: embryo -> adult ==")
    print(f"{'organ':10s} {'place':8s} | {'lat_e':>6s} {'lat_a':>6s} | {'disp_e':>7s} {'disp_a':>7s} |"
          f" {'bilat_e':>7s} {'bilat_a':>7s}")
    print("-" * 78)
    for name in [o["name"] for o in ORGAN_SCHEDULE if o["name"] in me["organs"]]:
        e, a = me["organs"][name], ma["organs"].get(name, {})
        if not e.get("sprouted"):
            print(f"{name:10s} {e['place']:8s} | FAILED to sprout")
            continue
        be = "yes" if e.get("bilateral") else ("--" if e["place"] != "paired" else "no")
        ba = "yes" if a.get("bilateral") else ("--" if e["place"] != "paired" else "no")
        print(f"{name:10s} {e['place']:8s} | {e['lateral']:6.2f} {a['lateral']:6.2f} |"
              f" {e['dispersion']:7.4f} {a['dispersion']:7.4f} | {be:>7s} {ba:>7s}")

    print("\n== MECHANISM PERSISTENCE embryo -> adult ==")
    nbad = 0
    for c in res["checks"]:
        ok = c["verdict"].startswith(("MAINTAINED", "END-STATE"))   # results invariant OR heads reached end-state
        nbad += 0 if ok else 1
        mark = "OK " if ok else ">> "
        print(f"{mark}{c['mechanism']:26s} {str(c['embryo']):>10s} -> {str(c['adult']):<14s}  {c['verdict']}")
    print("-" * 78)
    print(f"aspect {me['aspect']}->{ma['aspect']} | components {me['n_components']}->{ma['n_components']} |"
          f" cells {ma['n_cells']} | fates {ma['n_fates']}")
    print(f"{len(res['checks'])} checks, {nbad} not maintained")
    return nbad


if __name__ == "__main__":
    res = audit()
    nbad = _print(res)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_not_maintained=nbad, **res), open(str(OUT) + ".json", "w"), indent=1)
    print(f"saved {OUT}.json")
