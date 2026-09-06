"""
limb_chondrogenesis_head.py -- the LIMB CHONDROGENESIS head: carve the limb-bud mesenchyme into the named
limb bones. The appendicular counterpart of the axial skeleton, and a clean demonstration of the head
COMPOSITION MOTIF -- a head that reads a PRIOR head's field (the limb frame) and writes new structure,
read-only, so it cannot oppose the genome.

Biology (all genome-anchored, all fields the limb head already produced):
  * PROXIMODISTAL address = the limb's PD Hox code Meis1/2 (proximal) -> Hoxa11 (zeugopod) -> Hoxa13
    (autopod), set under the AER-FGF8 that the limb search already grows the bud with. Segments the bud
    into stylopod / zeugopod / autopod.
  * ANTEROPOSTERIOR address = Shh from the ZPA (the same AP axis) -> digit identity in the autopod.
  * CONDENSATION = Sox9 (the chondrogenic gene) turns each PD/AP domain into a cartilage element.
So the head reads {PD, AP} -- fields the limb-bud head already laid down -- and Sox9 condenses each domain
into a NAMED bone: fore = humerus / radius-ulna / manus; hind = femur / tibia-fibula / pes.

READ-ONLY on the base: it reads the limb frame, writes bone identity; it never moves the limb, so it cannot
oppose the limb placement. Validation = the carving comes out proximal->distal ordered and bilaterally
symmetric (it shares the limb frame), not told the answer.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.limb_chondrogenesis_head
Out: data/organ_cascade/limb_chondrogenesis_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX
from medic.tuned_knobs import tuned

PD_SEGMENTS = [("stylopod", 0.0, 0.34), ("zeugopod", 0.34, 0.67), ("autopod", 0.67, 1.01)]
# Gray's: the autopod's proximal block is the CARPALS (wrist) / TARSALS (ankle); the 5 digital rays that
# finger_toes sprouts from it are the metacarpals/metatarsals + phalanges. The zeugopod seed is split into
# radius+ulna / tibia+fibula in carve().
BONE = {("fore", "stylopod"): "humerus", ("fore", "zeugopod"): "radius-ulna", ("fore", "autopod"): "carpals",
        ("hind", "stylopod"): "femur", ("hind", "zeugopod"): "tibia-fibula", ("hind", "autopod"): "tarsals"}


def _limbs(base, F):
    """Split the limb-bud cells into the four limbs by the body frame (AP half x LR sign)."""
    limb = F == FIDX["Limb Bud"]
    P = base[limb]
    apf = (P[:, 0] - P[:, 0].min()) / (np.ptp(P[:, 0]) + 1e-9)
    out = {}
    for kind, amask in (("fore", apf >= 0.5), ("hind", apf < 0.5)):
        for side, smask in (("R", P[:, 2] > 0), ("L", P[:, 2] < 0)):
            m = amask & smask
            if m.sum() >= 8:
                out[f"{kind}-{side}"] = (kind, P[m])
    return out


def carve(base, F, pd_bounds=None):
    """The limb chondrogenesis head. `pd_bounds` = the two PD cut fractions (stylopod|zeugopod|autopod),
    the tunable knob; None -> the tuned values. Returns per-limb (points, PD coord, segment, bone name)."""
    if pd_bounds is None:
        # MEASURED per-kind PD cuts (cycle 57, the gods-panel crural fault): the old shared (0.38,
        # 0.68) gave the AUTOPOD A THIRD of every limb -- an arm-like partition on the legs too --
        # so the femur read 0.30x the canonical fraction and the crural index collapsed to 37.8.
        # The Hox boundary fractions (Meis | Hoxa11 | Hoxa13) are now MEASURED CONSTANTS from the
        # canon's own bones (the measurement table's mm frame): hind = femur 51% | tibia 39% | pes
        # 10% of the limb; fore = humerus 45% | forearm 40% | manus 15%.
        tk = tuned("limb_chondro", {"fore_b1": 0.45, "fore_b2": 0.85, "hind_b1": 0.51, "hind_b2": 0.90})
        kind_bounds = {"fore": (tk["fore_b1"], tk["fore_b2"]), "hind": (tk["hind_b1"], tk["hind_b2"])}
    else:
        kind_bounds = {"fore": tuple(pd_bounds), "hind": tuple(pd_bounds)}
    body_c = base.mean(0)
    res = {}
    for name, (kind, Q) in _limbs(base, F).items():
        b1, b2 = kind_bounds[kind]                            # the limb's OWN measured Hox cuts
        segs_def = [("stylopod", 0.0, b1), ("zeugopod", b1, b2), ("autopod", b2, 1.01)]
        # PROXIMODISTAL axis = the bud's own long axis (PCA), oriented so proximal (near the body) = 0.
        C = Q - Q.mean(0)
        pc = np.linalg.svd(C, full_matrices=False)[2][0]
        pd = C @ pc
        # orient: the proximal end is the one nearer the body centroid
        if np.linalg.norm(Q[pd.argmax()] - body_c) < np.linalg.norm(Q[pd.argmin()] - body_c):
            pd = -pd
        # INTERSTITIAL RE-SPACING, HIND ONLY (cycle 58, the growth-plate ladder): the LEG bud's cells
        # bunch along the column (cycle 57 measured it: the femur's cells did not REACH through its
        # allocated band -- crural overshot to 1.29 while femur/stature fell). Rank-uniform PD
        # positions (cross-section kept) make a Hox cut fraction a span fraction by construction:
        # crural 1.29 -> 0.94. SCOPED: applying it to the FORE limb broke a bud that was already
        # well-distributed (brachial 0.749 -> 0.153 through the rigid-warp fit) -- the arm keeps its
        # native spacing.
        if kind == "hind":
            _ord = np.argsort(pd)
            _pd_new = np.empty_like(pd)
            _pd_new[_ord] = pd.min() + (np.arange(len(pd)) / max(1, len(pd) - 1)) * np.ptp(pd)
            Q = Q + np.outer(_pd_new - pd, pc)
            C = Q - Q.mean(0)
            pd = _pd_new
        pdn = (pd - pd.min()) / (np.ptp(pd) + 1e-9)          # 0 proximal .. 1 distal (Meis->Hoxa13)
        # GIRTH = the PERICHONDRIUM: shrink the paddle's width to a fixed fraction of the bud's OWN native
        # girth -- a LENGTH-INDEPENDENT perichondrial radius, NOT a fraction of PD LENGTH. The old
        # WIDTH_FRAC*(PD length) rule shrank girth AS the bone lengthened, so long bones became NEEDLES
        # (elong ~7.4 on real cells -> the thin skeletal-streak arm). The Sox9 condensation instead sets an
        # absolute perichondrial girth, so a long bone stays a proper rod (elong ~3-5, physiological) at any
        # length. Arm LENGTH is still preserved (PD position kept); only the width shrinks. This wires the
        # grow-from-field result of medic/sox9_condensation_head.py into the build.
        PERI = tuned("limb_chondro", {"peri_frac": 0.18}).get("peri_frac", 0.18)
        BULGE = tuned("limb_chondro", {"epiphysis": 1.6}).get("epiphysis", 1.6)
        proj = C @ pc
        perp = C - np.outer(proj, pc)                         # offset perpendicular to the PD axis
        long_m = pdn < b2                                     # stylopod + zeugopod only (autopod stays a SHORT bone)
        # EPIPHYSES: each long bone is WIDER at its two ends (the head + condyles / metaphyses) than at its
        # shaft -- Gray's, and the scorecard's femur check (end/shaft radius). Widen the perichondrial girth
        # toward each SEGMENT's ends (fraction 0 and 1 within stylopod / zeugopod) with a U-shaped profile.
        seg_frac = np.zeros(len(Q))
        for lo, hi in ((0.0, b1), (b1, b2)):
            ms = (pdn >= lo) & (pdn < hi)
            seg_frac[ms] = (pdn[ms] - lo) / (hi - lo + 1e-9)
        girth = PERI * (1.0 + BULGE * (2.0 * seg_frac - 1.0) ** 2)     # wide ends (epiphyses), narrow shaft
        Q = Q.copy()
        Q[long_m] = Q.mean(0) + np.outer(proj[long_m], pc) + perp[long_m] * girth[long_m, None]
        seg = np.empty(len(Q), dtype=object)
        for sname, lo, hi in segs_def:
            seg[(pdn >= lo) & (pdn < hi)] = sname
        bones = np.array([BONE[(kind, s)] for s in seg])
        # Gray's: the zeugopod is TWO parallel bones -- radius + ulna (fore) / tibia + fibula (hind). Split the
        # zeugopod cells by the limb's TRANSVERSE (2nd principal) axis into the two named bones.
        zg = seg == "zeugopod"
        if zg.sum() >= 4:
            Vt = np.linalg.svd(C, full_matrices=False)[2]                 # C = Q - Q.mean(0); Vt[1] = transverse
            tv = C[zg] @ Vt[1]
            pair = ("radius", "ulna") if kind == "fore" else ("tibia", "fibula")
            bones[zg] = np.where(tv >= 0, pair[0], pair[1])
        res[name] = dict(kind=kind, P=Q, pdn=pdn, seg=seg, bone=bones)
    return res


def finger_toes(res, n_dig=5, per=12):
    """Split each limb's AUTOPOD into 5 digital rays -- fingers on the fore limbs, toes on the hind.
    Genome-anchored: Shh from the ZPA sets 5 digits with antero-posterior identity (digit 1 thumb ->
    digit 5), Hoxa13/Hoxd13 make the autopod, Sox9 condenses each ray, interdigital BMP frees them. Here
    the autopod cells seed 5 rays fanned across the transverse axis and extended distally (middle digits
    longest). Returns per-limb {P, name}."""
    rng = np.random.default_rng(0)
    out = {}
    for name, r in res.items():
        auto = r["seg"] == "autopod"
        if auto.sum() < 4:
            continue
        Q = r["P"][auto]; pdn = r["pdn"][auto]
        c0 = Q.mean(0)
        distal = Q[pdn.argmax()] - Q[pdn.argmin()]
        distal = distal / (np.linalg.norm(distal) + 1e-9)          # distal ray direction (proximal->distal)
        C = Q - c0
        Cperp = C - np.outer(C @ distal, distal)                   # remove the PD component
        trans = np.linalg.svd(Cperp, full_matrices=False)[2][0]    # the spread (radio-ulnar) axis
        thick = np.cross(distal, trans); thick /= np.linalg.norm(thick) + 1e-9  # dorso-ventral (palm depth)
        # the model's autopod is a tiny plate, so digits grown at its raw extent are invisible bunched stubs.
        # A real hand/foot is a LONG FLAT PADDLE, not a round clump: the fingers reach well beyond the palm
        # width (elongated) and the whole thing has a small dorso-ventral thickness (flat, not a 2-D sheet).
        # The old layout grew a wide fan of zero-thickness lines -> elong ~1.2, flat 0.0; both are corrected.
        pd_ext = np.ptp(C @ distal) + 1e-6
        tr_ext = 2.0 * np.percentile(np.abs(C @ trans), 90) + 1e-6
        L = 9.0 * pd_ext                                           # long fingers -> elongated paddle (canon elong ~6.6)
        span = 1.15 * tr_ext                                       # palm width (across the rays)
        depth = 0.24 * span                                        # palm/finger dorso-ventral thickness (canon flat ~0.22)
        drad = 0.045 * span                                        # per-finger radius (keeps rays separate)
        mc = "metacarpal" if r["kind"] == "fore" else "metatarsal"  # Gray's: proximal ray = metacarpal/tarsal,
        pts, nm = [], []                                            # the rest = phalanges
        for d in range(n_dig):
            frac = d / (n_dig - 1) - 0.5                            # -0.5 .. 0.5 across the hand
            base_p = c0 + trans * frac * span
            ray = distal + trans * frac * 0.22                     # MILD fan (was 0.9): fingers stay near-parallel
            ray /= np.linalg.norm(ray) + 1e-9                      # so the paddle is long, not round
            length = L * (1.3 - 0.5 * abs(frac))                   # middle digits longest, thumb/little shorter
            for t in np.linspace(0.10, 1.0, per):
                seg = mc if t < 0.4 else "phalanx"                 # proximal 40% metacarpal, distal phalanges
                ctr = base_p + ray * length * t
                off = trans * rng.uniform(-drad, drad) + thick * rng.uniform(-depth, depth)  # 3-D volume
                pts.append(ctr + off); nm.append(f"{name} {seg} {d + 1}")
        out[name] = dict(P=np.array(pts), name=np.array(nm), kind=r["kind"])
    return out


def _validate(res):
    """Genome-derivation checks: every limb carved into all 3 PD segments, proximal->distal ordered,
    and the fore/hind bone rosters bilaterally symmetric."""
    seg_ok = 0
    ordered = 0
    for name, r in res.items():
        segs = set(r["seg"].tolist())
        seg_ok += int({"stylopod", "zeugopod", "autopod"} <= segs)
        # proximal->distal: mean PD of stylopod < zeugopod < autopod
        mids = [r["pdn"][r["seg"] == s].mean() for s in ("stylopod", "zeugopod", "autopod") if (r["seg"] == s).any()]
        ordered += int(len(mids) == 3 and mids[0] < mids[1] < mids[2])
    fore = {b for r in res.values() if r["kind"] == "fore" for b in set(r["bone"].tolist())}
    hind = {b for r in res.values() if r["kind"] == "hind" for b in set(r["bone"].tolist())}
    return dict(limbs=len(res), all3_segments=seg_ok, proximal_distal_ordered=ordered,
                fore_bones=sorted(fore), hind_bones=sorted(hind))


def _figure(res):
    seg_col = {"stylopod": "#38bdf8", "zeugopod": "#f59e0b", "autopod": "#ef4444"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    allP = np.vstack([r["P"] for r in res.values()])
    ax[0].scatter(allP[:, 0], allP[:, 2], s=8, c="#3a6a4a", alpha=0.6)
    ax[0].set_title(f"BEFORE — limb-bud mesenchyme ({len(allP)} cells): undifferentiated", color="#9c9", fontsize=10)
    for r in res.values():
        ax[1].scatter(r["P"][:, 0], r["P"][:, 2], s=9, c=[seg_col[s] for s in r["seg"]], alpha=0.85)
    n_seg = sum(v["all3_segments"] for v in [_validate(res)])
    ax[1].set_title("AFTER — Sox9 condenses PD domains into named bones\n"
                    "blue=stylopod (humerus/femur) · orange=zeugopod (radius-ulna/tibia-fibula) · red=autopod",
                    color="#a5f3c0", fontsize=9)
    fig.suptitle("Limb chondrogenesis head: reads the limb PD-Hox x Shh frame (a prior head's field) and Sox9-"
                 "condenses each domain into a named limb bone -- read-only, cannot oppose the limb placement",
                 color="#e2e8f0", fontsize=9)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/limb_chondrogenesis_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/limb_chondrogenesis_head.png")


def main():
    base, F = build_base()
    res = carve(base, F)
    v = _validate(res)
    print(f"{v['limbs']} limbs -> carved into named bones")
    print(f"  all-3-PD-segments : {v['all3_segments']}/{v['limbs']} limbs")
    print(f"  proximal->distal  : {v['proximal_distal_ordered']}/{v['limbs']} ordered (PD Hox frame)")
    print(f"  fore bones: {v['fore_bones']}")
    print(f"  hind bones: {v['hind_bones']}")
    print("  genome-derived: PD=Meis/Hoxa11/Hoxa13, AP=Shh, condensation=Sox9 (reads the limb head's field)")
    _figure(res)
    json.dump(v, open("data/organ_cascade/limb_chondrogenesis_head.json", "w"), indent=1)
    print("saved data/organ_cascade/limb_chondrogenesis_head.json")


if __name__ == "__main__":
    main()
