"""anoikis_head.py -- the APOPTOSIS head: the missing member of the four fundamental cell behaviours
(divide, differentiate, migrate, DIE -- Miles 2026-08-30: "number 4?" -- yes: our build wired the first
three plus shape; death was never built).

v1 implements the ANOIKIS rule, the death trigger Miles's connexin-skin framing names: a cell survives on
contact with its own organ's gap-junction neighbourhood; a cell whose local neighbourhood carries (almost)
no same-family cells has lost that survival signal and undergoes programmed death. This is the genomic
form of "ablate what should not be there": stragglers die by a rule the genome states (caspase machinery
downstream of lost integrin/cadherin contact), never by comparison to the reference atlas.

Discipline (order matters): a mispatterned POPULATION (e.g. jaw cells seated at the crown) is a placement
fault of its head and must be fixed THERE; anoikis eats only the residual stragglers. Hence the kill is
CAPPED (max_kill_frac) and reported per fate, so a large kill on one fate flags a placement fault instead
of silently hiding it.

Scope: only fates in FAMILIES are subject to death. Migratory and filler fates (muscle, mesenchyme,
cartilage, limb) are exempt -- intermixing is their biology, not a lost survival signal.

Wired: adult_persistence_audit.build_base, LAST (after dv_spread), so it cleans the final placement.
Run standalone for the kill report:  venv_win_new/Scripts/python.exe -m medic.anoikis_head
"""
from __future__ import annotations
import numpy as np

from medic.unified_embryo import FIDX
from medic.subhead_program import expand_names

# family -> parent fates (expand_names folds in sub-head children). Only these fates can die.
FAMILIES = {
    "brain":   ["Forebrain", "Midbrain", "Hindbrain", "Cerebellum", "OlfactoryBulb", "Telencephalon"],
    "cord":    ["Spinal Cord"],           # Notochord EXEMPT: a thin rod (see the sheet/rod note below)
    "heart":   ["Heart", "Atrium", "Ventricle", "Left Ventricle", "Right Ventricle", "Outflow"],
    "lung":    ["Lung"],
    "liver":   ["Liver", "LiverHaem"],
    "kidney":  ["Kidney", "Nephron"],
    "spleen":  ["Spleen"],
    "pancreas": ["Pancreas"],
    "gut":     ["Gut", "Hindgut", "Foregut", "Stomach", "Duodenum", "Oesophagus"],
    "bladder": ["Bladder"],
    "adrenal": ["Adrenal"],
    "thymus":  ["Thymus"],
    "gonad":   ["Gonad"],
    "eye":     ["Eye", "Retina"],
    "otic":    ["Otic"],
    "jaw":     ["Jaw"],
    "thyroid": ["Thyroid"],
}
# SHEET and ROD geometries are EXEMPT: a skin cell's 3D nearest neighbours are legitimately the fat and
# muscle beneath the sheet, and a notochord cell's are the vertebrae around the rod -- the isotropic k-NN
# rule mis-reads both as detached (v1 killed 2360 skin + 486 notochord cells and dented the silhouette).
# Their survival contact runs ALONG the sheet/rod; a dimension-aware neighbourhood is future work.


def _family_of():
    fam = {}
    for name, parents in FAMILIES.items():
        for n in expand_names(parents):
            if n in FIDX:
                fam[FIDX[n]] = name
    return fam


def apply(base, F, k=12, survive=2, max_kill_frac=0.02, crowd_frac=0.30, max_crowd_frac=0.01,
          max_fate_frac=0.20, verbose=True):
    """The two Conway deaths, genomically owned. LONELINESS (anoikis, integrin/cadherin loss): fewer
    than `survive` of a cell's k nearest neighbours share its organ family -> death. OVERCROWDING
    (Hippo/YAP contact inhibition -> live-cell extrusion): a cell packed so tight that its k-th
    neighbour sits closer than crowd_frac of its OWN FAMILY's median spacing (>15x the organ's normal
    density) -> death. Each capped (worst offenders first) and reported per fate, so a large kill
    flags a placement fault to fix at its head. Returns (base, F, report)."""
    from scipy.spatial import cKDTree
    base = np.asarray(base, float)
    F = np.asarray(F)
    fam = _family_of()
    fids = np.array([fam.get(int(f), "") for f in F], dtype=object)
    subject = fids != ""
    idx_subject = np.where(subject)[0]
    tree = cKDTree(base)
    dist, nn = tree.query(base[subject], k=k + 1)
    dist, nn = dist[:, 1:], nn[:, 1:]                       # drop self
    # -- loneliness (anoikis) --
    same = (fids[nn] == fids[subject][:, None]).sum(1)
    lone = same < survive
    lone_idx = idx_subject[lone]
    cap = int(max_kill_frac * len(base))
    if len(lone_idx) > cap:
        lone_idx = lone_idx[np.argsort(same[lone])[:cap]]
    # -- overcrowding (contact inhibition), measured against the cell's own family's spacing --
    dk = dist[:, -1]
    sf = fids[subject]
    crowd = np.zeros(len(dk), bool)
    for f in np.unique(sf):
        m = sf == f
        med = np.median(dk[m])
        if med > 0:
            crowd[m] = dk[m] < crowd_frac * med
    crowd_idx = idx_subject[crowd & ~np.isin(np.arange(subject.sum()), np.where(lone)[0])]
    ccap = int(max_crowd_frac * len(base))
    if len(crowd_idx) > ccap:                               # densest first
        sub_dk = dk[crowd]
        crowd_idx = crowd_idx[np.argsort(sub_dk)[:ccap]] if len(sub_dk) == len(crowd_idx) else crowd_idx[:ccap]
    kill_idx = np.unique(np.concatenate([lone_idx, crowd_idx]))
    # PER-FATE kill cap (2026-08-30 tuning): apoptosis never takes more than max_fate_frac of any single
    # fate -- v1 halved the small thymic lobes and thinned parts below their scoring thresholds (grays
    # 288->286, canonical -2 parts). A capped-out fate still flags its fault in the report.
    if len(kill_idx):
        fate_n = {int(f): int(n) for f, n in zip(*np.unique(F, return_counts=True))}
        kept_kills = []
        for f in np.unique(F[kill_idx]):
            if fate_n.get(int(f), 0) < 40:                  # tiny fates are statistics, not stragglers:
                continue                                    # death on them erases organs (SmallIntestine
            fk = kill_idx[F[kill_idx] == f]                 # fell under the PBD tier's 8-cell floor)
            fcap = int(max_fate_frac * fate_n.get(int(f), 0))
            kept_kills.append(fk[:fcap] if len(fk) > fcap else fk)
        kill_idx = np.concatenate(kept_kills) if kept_kills else kill_idx[:0]
    keep = np.ones(len(base), bool)
    keep[kill_idx] = False
    report = {}
    if len(kill_idx):
        IDX2NAME = {v: n for n, v in FIDX.items()}
        u, c = np.unique(F[kill_idx], return_counts=True)
        report = {IDX2NAME.get(int(f), str(f)): int(n) for f, n in
                  sorted(zip(u, c), key=lambda t: -t[1])}
    if verbose:
        print(f"  [death] loneliness {len(lone_idx)} + overcrowding {len(crowd_idx)} of {len(base)} cells "
              f"({100 * len(kill_idx) / len(base):.2f}%)")
        for name, n in list(report.items())[:10]:
            print(f"      {name:22s} {n}")
    return base[keep], F[keep], report


if __name__ == "__main__":
    from medic.adult_persistence_audit import build_base
    b, F = build_base(30000)
    b2, F2, rep = apply(b, F)
    print(f"body {len(b)} -> {len(b2)} cells; kills by fate:")
    for name, n in rep.items():
        print(f"  {name:24s} {n}")
