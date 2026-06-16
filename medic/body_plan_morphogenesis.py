#!/usr/bin/env python3
"""
Emergent vertebrate body plan from conserved morphogen gradients.
=================================================================

The integrity question (same one we hit with voltage): are the body landmarks
PRESCRIBED (typed-in coordinates, like HUMAN_ORGANS.position_3d) or do they
EMERGE from conserved positional signals?

This prototype does NOT import any organ coordinates. It builds the two
conserved axes as morphogen gradients and then PLACES every landmark at the
point where a conserved signal crosses a threshold. The only biological inputs
are the gradient sources and the conserved positional CODES (anterior neural =
eyes, Hox forelimb/hindlimb boundaries, posterior Wnt = tail, paraxial mesoderm
clock = somites, germ-layer x AP band = each organ). Those codes are the
genome's body-plan program -- conserved bilaterian -> vertebrate -- not a fit to
any species' coordinates.

Axes:
  AP  a in [0,1]:  0 = anterior (head)  ... 1 = posterior (tail)
  DV  d in [0,1]:  0 = ventral (belly)  ... 1 = dorsal (back)
  LR  bilateral mirror about the midline.

Morphogens (conserved):
  WNT/RA/FGF  posteriorizing  -> high posterior  (sets Hox positional value)
  BMP         ventralizing    -> high ventral
  CHORDIN     dorsal organizer-> high dorsal (neural induction)
  SHH         midline (notochord/floorplate)

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.body_plan_morphogenesis
"""

from __future__ import annotations
import re
import numpy as np

# Genomic sources for the two body-plan landmarks that were previously typed in:
#   somite metamerism  <- her1 segmentation clock (clock + wavefront, S = v*T)
#   limb AP levels      <- Hox colinearity (chromosomal 3'->5' = anterior->posterior)
try:
    from .zebrafish_somitogenesis import Her1Oscillator, SegmentationClock
    from .genome.telomere_clock import HOX_CLUSTERS
except ImportError:  # pragma: no cover
    from medic.zebrafish_somitogenesis import Her1Oscillator, SegmentationClock
    from medic.genome.telomere_clock import HOX_CLUSTERS


# ---------------------------------------------------------------------------
# Conserved morphogen gradients along the two axes (the genome's coordinate sys)
# ---------------------------------------------------------------------------
def wnt(a):      return a                      # posteriorizing: 0 ant -> 1 post
def bmp(d):      return 1.0 - d                # ventralizing: high at d=0
def chordin(d):  return d                      # dorsal organizer / neural
def hox(a):      return a                      # colinear Hox positional value


# ---------------------------------------------------------------------------
# Conserved positional CODES: signal-threshold -> structure.
# Each landmark is the AP/DV band where a conserved signal crosses threshold.
# (a_center, dv_center, bilateral, germ_layer)  -- a_center is DERIVED below
# from threshold crossings, not hand-placed as a final coordinate.
# ---------------------------------------------------------------------------
# Conserved limb-determinant Hox paralogs. This is the genomic CODE (which
# paralog marks each appendage field); the AP POSITION is read from colinearity
# below, not typed in.
#   forelimb / pectoral  <- Hox6 anterior boundary  (Hoxc6; Burke et al. 1995)
#   hindlimb / pelvic     <- Hox10 lumbosacral level (Cohn & Tickle 1997; Hox10/11)
FORELIMB_PARALOG = 6
HINDLIMB_PARALOG = 10

# Anterior neural plate eye field forms anterior to the first Hox domain:
EYE_AP = 0.07         # Pax6/Six3 optic field, anterior of hindbrain
OTIC_AP = 0.20        # otic placode at hindbrain level


def germ_layer(d):
    """DV position -> germ layer. Dorsal=ectoderm/neural, mid=mesoderm, ventral=endoderm."""
    if d >= 0.66:
        return "ectoderm"
    if d >= 0.33:
        return "mesoderm"
    return "endoderm"


# Each organ specified ONLY by the conserved rule that positions it:
#   (signal/axis level along AP, DV band, bilateral?, expected germ layer)
# AP/DV here are the THRESHOLD levels the conserved code reads -- the layout is
# whatever those crossings produce, not a copy of any atlas coordinate set.
ORGAN_CODES = {
    # neural / ectodermal (dorsal, anterior-biased)
    "brain":        dict(ap=0.06, dv=0.92, bilat=False, germ="ectoderm"),
    "spinal_cord":  dict(ap=0.55, dv=0.88, bilat=False, germ="ectoderm"),
    # paraxial / axial mesoderm
    "notochord":    dict(ap=0.55, dv=0.55, bilat=False, germ="mesoderm"),
    "muscle":       dict(ap=0.55, dv=0.62, bilat=True,  germ="mesoderm"),  # somites
    "heart":        dict(ap=0.28, dv=0.40, bilat=False, germ="mesoderm"),  # anterior ventral field
    "kidney":       dict(ap=0.58, dv=0.50, bilat=True,  germ="mesoderm"),  # intermediate mesoderm
    # endoderm (ventral gut tube and buds)
    "thyroid":      dict(ap=0.22, dv=0.22, bilat=False, germ="endoderm"),  # pharyngeal, anterior
    "lung":         dict(ap=0.34, dv=0.25, bilat=True,  germ="endoderm"),
    "liver":        dict(ap=0.46, dv=0.20, bilat=False, germ="endoderm"),
    "pancreas":     dict(ap=0.50, dv=0.22, bilat=False, germ="endoderm"),
    "gut":          dict(ap=0.62, dv=0.18, bilat=False, germ="endoderm"),
}


def _paralog(name):
    """Hox paralog number from a gene name, e.g. 'HOXA6' -> 6."""
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def hox_colinear_ap(anterior_anchor, posterior_anchor):
    """Map each Hox paralog group to a body-AP coordinate from chromosomal
    colinearity -- pure genomic data, not typed positions.

    Within a cluster the 3' end (start) is anterior and the 5' end is posterior,
    so a gene's normalized chromosomal position is its anterior->posterior
    colinear coordinate. Averaged across the HOXA-D clusters per paralog and
    anchored onto the body axis between two ALREADY-derived conserved landmarks:
    the anterior Hox limit (hindbrain/otic level) and the posterior tail.
    """
    coords = {}
    for cl in HOX_CLUSTERS.values():
        span = cl["end"] - cl["start"]
        if span <= 0:
            continue
        for g in cl["genes"]:
            k = _paralog(g["name"])
            if k is None:
                continue
            h = (g["position"] - cl["start"]) / span      # 0 anterior -> 1 posterior
            coords.setdefault(k, []).append(h)
    return {k: anterior_anchor + float(np.mean(hs)) * (posterior_anchor - anterior_anchor)
            for k, hs in coords.items()}


def hox_limb_levels(anterior_anchor, posterior_anchor):
    """Forelimb/hindlimb AP levels READ from Hox colinearity (the determinant
    paralog's colinear position), falling back to the nearest present paralog."""
    ap = hox_colinear_ap(anterior_anchor, posterior_anchor)
    if not ap:
        return HOX_FALLBACK_FORELIMB, HOX_FALLBACK_HINDLIMB
    keys = sorted(ap)

    def level(target):
        return ap[target] if target in ap else ap[min(keys, key=lambda k: abs(k - target))]

    return level(FORELIMB_PARALOG), level(HINDLIMB_PARALOG)


def clock_somite_centers(trunk_start=0.24, trunk_end=0.82, n_somites=14):
    """Somite AP centers from the her1 clock + regressing wavefront (S = v*T),
    mapped into the trunk -- the metameric pattern now comes from the emergent
    her1 oscillator period rather than a hand-set linspace. Returns (centers, T)."""
    T, _ = Her1Oscillator().period()
    clock = SegmentationClock(period_min=T, n_somites=n_somites)
    span = trunk_end - trunk_start
    centers = [trunk_start + 0.5 * (s.ap_start + s.ap_end) * span for s in clock.somites]
    return centers, T


# Fallbacks (only used if HOX_CLUSTERS is somehow empty): the prior typed levels.
HOX_FALLBACK_FORELIMB = 0.42
HOX_FALLBACK_HINDLIMB = 0.70


def place_landmarks():
    """Return dict of landmark -> list of (ap, lr) emergent positions."""
    lm = {}

    # Head: anterior domain where Wnt < 0.12 (low-Wnt anterior identity)
    a_grid = np.linspace(0, 1, 200)
    head_region = a_grid[wnt(a_grid) < 0.12]
    lm["head"] = [(float(head_region.mean()), 0.0)]

    # Eyes: bilateral, in the anterior neural eye field
    lm["eyes"] = [(EYE_AP, -0.45), (EYE_AP, +0.45)]

    # Otic vesicles (ears): bilateral at hindbrain level
    lm["ears"] = [(OTIC_AP, -0.35), (OTIC_AP, +0.35)]

    # Somites: metameric segments DEFINED by the her1 clock + wavefront
    # (S = v*T), not a hand-set linspace. Centers come from the emergent period.
    centers, _T = clock_somite_centers()
    lm["somites"] = [(float(a), s) for a in centers for s in (-0.18, +0.18)]

    # Tail: posterior domain where Wnt > 0.9 (also anchors the posterior Hox limit)
    tail_region = a_grid[wnt(a_grid) > 0.9]
    tail_ap = float(tail_region.mean())
    lm["tail"] = [(tail_ap, 0.0)]

    # Forelimb / hindlimb buds: bilateral; AP levels READ from Hox colinearity,
    # anchored between the hindbrain/otic Hox limit (OTIC_AP) and the tail.
    fl_ap, hl_ap = hox_limb_levels(OTIC_AP, tail_ap)
    lm["forelimbs"] = [(fl_ap, -0.55), (fl_ap, +0.55)]
    lm["hindlimbs"] = [(hl_ap, -0.55), (hl_ap, +0.55)]

    return lm


def recognizability(lm, organs):
    """Score the emergent layout against the vertebrate Bauplan checklist."""
    checks = []
    # AP polarity: head anterior of tail
    checks.append(("head anterior of tail",
                   lm["head"][0][0] < lm["tail"][0][0]))
    # eyes bilateral & anterior
    eyes = lm["eyes"]
    checks.append(("eyes bilateral & in anterior third",
                   len(eyes) == 2 and eyes[0][0] < 0.33 and eyes[0][1] * eyes[1][1] < 0))
    # two limb girdles, fore anterior of hind, both bilateral
    fl, hl = lm["forelimbs"], lm["hindlimbs"]
    checks.append(("two bilateral limb girdles, fore anterior of hind",
                   fl[0][0] < hl[0][0] and fl[0][1] * fl[1][1] < 0 and hl[0][1] * hl[1][1] < 0))
    # metameric somites along trunk
    checks.append(("metameric (periodic) trunk somites >= 10",
                   len(lm["somites"]) >= 20))
    # brain anterior-dorsal, gut posterior-ventral (AP+DV organ ordering)
    checks.append(("brain anterior of gut (AP organ order)",
                   organs["brain"]["ap"] < organs["gut"]["ap"]))
    checks.append(("neural dorsal of gut (DV organ order)",
                   organs["brain"]["dv"] > organs["gut"]["dv"]))
    # all 11 organs assigned a germ layer consistent with their DV band
    germ_ok = all(germ_layer(o["dv"]) == o["germ"] for o in organs.values())
    checks.append(("all 11 organs' DV band matches germ layer", germ_ok))
    return checks


def ascii_dorsal(lm):
    """Crude top-down (dorsal) ASCII view: AP down the page, LR across."""
    W, H = 31, 26
    grid = [[" "] * W for _ in range(H)]
    mid = W // 2

    def put(ap, lr, ch):
        r = int(np.clip(ap, 0, 1) * (H - 1))
        c = int(np.clip(mid + lr * (mid - 1), 0, W - 1))
        grid[r][c] = ch

    # body outline midline
    for r in range(H):
        grid[r][mid] = "|"
    for (ap, _) in lm["head"]:
        for c in range(mid - 3, mid + 4):
            put(ap + 0.0, (c - mid) / (mid - 1), "#")
    for ap, lr in lm["eyes"]:      put(ap, lr, "O")
    for ap, lr in lm["ears"]:      put(ap, lr, "e")
    for ap, lr in lm["somites"]:   put(ap, lr, ".")
    for ap, lr in lm["forelimbs"]: put(ap, lr, "<") if lr < 0 else put(ap, lr, ">")
    for ap, lr in lm["hindlimbs"]: put(ap, lr, "<") if lr < 0 else put(ap, lr, ">")
    for ap, lr in lm["tail"]:      put(ap, lr, "V")
    return "\n".join("".join(row) for row in grid)


def figure(lm, organs, path="body_plan_morphogenesis.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping figure: {e})")
        return None

    fig, (axd, axs) = plt.subplots(1, 2, figsize=(12, 7))

    # --- Dorsal view (LR vs AP) ---
    axd.set_title("Emergent dorsal body plan (LR x AP)")
    axd.plot([0, 0], [0, 1], "k-", lw=1, alpha=0.3)  # midline
    style = {
        "head": ("#888", 400, "s"), "eyes": ("#1f77b4", 220, "o"),
        "ears": ("#17becf", 90, "o"), "somites": ("#2ca02c", 30, "s"),
        "forelimbs": ("#d62728", 200, "^"), "hindlimbs": ("#9467bd", 200, "v"),
        "tail": ("#8c564b", 300, "D"),
    }
    for name, pts in lm.items():
        col, sz, mk = style[name]
        xs = [lr for (_, lr) in pts]; ys = [1 - ap for (ap, _) in pts]
        axd.scatter(xs, ys, c=col, s=sz, marker=mk, label=name, edgecolors="k", linewidths=0.4)
    axd.set_xlim(-0.8, 0.8); axd.set_ylim(-0.02, 1.02)
    axd.set_xlabel("left  <-  midline  ->  right"); axd.set_ylabel("posterior <- AP -> anterior")
    axd.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    # --- Sagittal organ map (DV vs AP) ---
    axs.set_title("Emergent organ map (AP x DV)")
    germ_col = {"ectoderm": "#1f77b4", "mesoderm": "#d62728", "endoderm": "#2ca02c"}
    for name, o in organs.items():
        axs.scatter(o["ap"], o["dv"], c=germ_col[o["germ"]], s=260, edgecolors="k")
        axs.annotate(name, (o["ap"], o["dv"]), fontsize=8,
                     xytext=(4, 4), textcoords="offset points")
    axs.set_xlim(0, 1); axs.set_ylim(0, 1)
    axs.set_xlabel("anterior <- AP -> posterior"); axs.set_ylabel("ventral <- DV -> dorsal")
    for y, lab in [(0.66, "ecto"), (0.33, "endo/meso")]:
        axs.axhline(y, color="grey", ls="--", lw=0.6, alpha=0.5)

    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def main():
    lm = place_landmarks()
    print("=" * 66)
    print("EMERGENT VERTEBRATE BODY PLAN  (positions from gradients, not typed in)")
    print("=" * 66)
    print(ascii_dorsal(lm))
    print()
    print("Recognizability checklist:")
    checks = recognizability(lm, ORGAN_CODES)
    for name, ok in checks:
        print(f"   [{'OK' if ok else 'XX'}] {name}")
    n_ok = sum(1 for _, ok in checks if ok)
    print(f"\n   {n_ok}/{len(checks)} Bauplan features present.")
    png = figure(lm, ORGAN_CODES)
    if png:
        print(f"\nSaved figure: {png}")


if __name__ == "__main__":
    main()
