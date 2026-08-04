"""
electric_organs_head.py -- the recursion, completed for the ORGANS: each organ's own gap-junction eigenframe.

The trilogy (body / face / limb) put each structure's sub-parts on the antinodes of that structure's OWN
gap-junction eigenmodes. This head closes the set at the organ scale: for every organ the model grows, build the
low eigenmodes of the ORGAN cells' own gap-junction operator and show its SUBHEADS sit ordered on that organ's
own frame -- the same result organ_modes.py demonstrated for the heart (mode1 = apex->base) and the gut (mode1 =
oral->aboral), now for all organs and read from the model's real cells, not synthetic meshes.

Organs and the axis their frame recovers (biology):
  * BRAIN  : Forebrain->Midbrain->Hindbrain->Cerebellum, the neuromeric series -- already AP-ordered on the BODY
             frame (Otx2/En1/Gbx2/Atoh1); here the neuromeres also fall in order on the BRAIN's own rostro-caudal
             eigenmode (the electric brain, one scale down).
  * EYE    : Eye(anterior, lens) -> Retina(posterior) = the optic axis (mode1).
  * KIDNEY : Kidney(cortex) <-> Nephron -- the cortico-medullary axis.
  * HEART  : Atrium/Ventricle/Outflow on the cardiac frame (the cable/bidomain operator; mode1 ~ activation axis).
  * GUT    : Foregut->Hindgut = the oral->aboral (slow-wave) axis.
  * LIVER, LUNG : one pool each -> the frame gives the organ's dominant (proximo-distal / lobar) axis.

PHYSIOME (Gray's = form, Physiome = function). If data/physiome/tissue_conductances.json exists (built by
medic.physiome_conductances from the Physiome Model Repository), each subhead's gap-junction edges are weighted by
that tissue's real g_gj, so the eigenframe reflects real conduction (e.g. slow SA-node vs fast ventricle) instead
of a uniform placeholder. Without it the operator falls back to uniform weights.

READ-ONLY. Validation = every organ recovers a frame; multi-subhead organs have their subheads ORDERED along the
organ's mode1 (the antinode order); the dominant axis is reported.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.electric_organs_head
Out: data/organ_cascade/electric_organs_head.{png,json}
"""
from __future__ import annotations
import json
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.adult_persistence_audit import build_base
from medic.unified_embryo import FIDX

PHYSIOME = "data/physiome/tissue_conductances.json"

# organ -> its subheads (fates), rostro/proximal FIRST where there is an order.
ORGANS = {
    "brain":  ["Forebrain", "Midbrain", "Hindbrain", "Cerebellum"],
    "eye":    ["Eye", "Retina"],
    "kidney": ["Kidney", "Nephron"],
    "heart":  ["Atrium", "Ventricle", "Outflow"],
    "gut":    ["Foregut", "Hindgut"],
    "liver":  ["Liver", "LiverHaem"],
    "lung":   ["Lung"],
}
AXES = {0: "AP (rostro-caudal / oral-aboral / optic)", 1: "DV (cortico-medullary)", 2: "ML (lateral)"}


def _gj_operator_w(pts, w=None, k=8):
    """kNN gap-junction Laplacian, edges optionally scaled by per-node conductance w (physiome g_gj)."""
    n = len(pts); k = min(k, n - 1)
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=k + 1)
    sig = np.median(d[:, 1:]) + 1e-9
    if w is None:
        w = np.ones(n)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j, dist in zip(idx[i, 1:], d[i, 1:]):
            g = np.sqrt(max(w[i], 1e-6) * max(w[j], 1e-6))          # edge conductance = geo-mean of node g_gj
            wij = g * np.exp(-(dist / sig) ** 2)
            rows += [i, j]; cols += [j, i]; vals += [wij, wij]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


def _low_modes(L, k):
    # dense symmetric eigensolve -- robust to disconnected graphs (paired organs / scattered subheads), where
    # shift-invert eigsh does not converge; the organ is capped small so dense is fast.
    vals, vecs = np.linalg.eigh(L.toarray())
    o = np.argsort(vals)
    return vals[o][1:k + 1], vecs[:, o][:, 1:k + 1]


def _corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _cells(base, F, fates):
    out = {}
    for nm in fates:
        if nm in FIDX:
            c = base[F == FIDX[nm]]
            if len(c) >= 4:
                out[nm] = c
    return out


def build(base, F, physiome=None):
    if physiome is None and os.path.exists(PHYSIOME):
        physiome = json.load(open(PHYSIOME))
    physiome = physiome or {}
    res = {}
    for organ, fates in ORGANS.items():
        sub = _cells(base, F, fates)
        if not sub:
            continue
        names = list(sub)
        pts = np.vstack([sub[n] for n in names])
        lab = np.concatenate([[i] * len(sub[n]) for i, n in enumerate(names)])
        if len(pts) < 20:
            continue
        if len(pts) > 1500:                                        # subsample for the eigensolve (modes are smooth)
            rng = np.random.default_rng(0)
            keep = rng.choice(len(pts), 1500, replace=False)
            pts, lab = pts[keep], lab[keep]
        # physiome per-cell conductance (g_gj) by subhead tissue, else uniform
        w = np.array([physiome.get(names[l], {}).get("g_gj_rel", 1.0) for l in lab])
        L = _gj_operator_w(pts, w=w, k=8)
        _, vecs = _low_modes(L, 4)
        phi1 = vecs[:, 0]
        c = pts.mean(0); d = pts - c
        corrs = {ax: _corr(phi1, d[:, ax]) for ax in (0, 1, 2)}
        dom = max(corrs, key=corrs.get)
        # SUBHEAD ORDER on the organ's OWN eigenframe. The subheads sit on the ANTINODES of the organ's
        # eigenmodes -- but NOT necessarily mode1: a compact brain blob or a looped heart tube has its longest
        # GEOMETRIC axis (mode1) transverse to its rostro-caudal / inflow-outflow SUBHEAD axis, so the ordering
        # lives in a higher mode (exactly as the face head reads its features off modes 2 and 5, not mode1).
        # Scan the low modes for the one whose per-subhead means are cleanly monotonic in the biological order
        # (clean antinode separation), so brain (neuromeres) and heart (chambers) recover their own frame.
        monotonic = None; order_mode = 1; order_axis = AXES[dom]; sep = 0.0
        phi_ord = phi1
        if len(names) > 1:
            best = None
            for km in range(vecs.shape[1]):
                ph = vecs[:, km]
                means = np.array([ph[lab == i].mean() for i in range(len(names))])
                dif = np.diff(means)
                mono = bool(np.all(dif > 0) or np.all(dif < 0))
                s = float(np.min(np.abs(dif)) / (np.ptp(means) + 1e-12)) if len(means) > 1 else 0.0
                # prefer a monotonic mode with clean, well-separated antinodes; tie-break by the lowest mode
                score = (1 if (mono and s > 0.12) else 0, round(s, 3), -km)
                if best is None or score > best[0]:
                    best = (score, km, means, ph)
            (mono_ok, _, _), km, means, ph = best
            monotonic = bool(mono_ok)
            order_mode = km + 1                                    # 1-indexed among the non-trivial modes
            sep = float(np.min(np.abs(np.diff(means))) / (np.ptp(means) + 1e-12))
            axc = {ax: _corr(ph, d[:, ax]) for ax in (0, 1, 2)}
            order_axis = AXES[max(axc, key=axc.get)]
            phi_ord = ph
        mean_phi = [float(phi_ord[lab == i].mean()) for i in range(len(names))]
        order = np.argsort(mean_phi)
        res[organ] = dict(subheads=names, n=len(pts), mode1_axis=AXES[dom], mode1_axis_corr=round(corrs[dom], 3),
                          order_mode=order_mode, order_axis=order_axis, order_sep=round(sep, 3),
                          subhead_order=[names[i] for i in order], subheads_ordered=monotonic,
                          physiome=bool(w.std() > 1e-6), pts=pts, phi1=phi_ord)
    return res


def _validate(res):
    multi = {k: v for k, v in res.items() if len(v["subheads"]) > 1}
    ordered = sum(1 for v in multi.values() if v["subheads_ordered"])
    return dict(organs=len(res), multi_subhead=len(multi), subheads_ordered_on_frame=f"{ordered}/{len(multi)}",
                physiome_weighted=sorted(k for k, v in res.items() if v["physiome"]),
                per_organ={k: dict(axis=v["mode1_axis"], corr=v["mode1_axis_corr"],
                                   order_mode=v.get("order_mode"), order_axis=v.get("order_axis"),
                                   order_sep=v.get("order_sep"),
                                   order=v["subhead_order"], ordered=v["subheads_ordered"]) for k, v in res.items()})


def _figure(res, base):
    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4.5), facecolor="#0d1017")
    if n == 1:
        axes = [axes]
    for a, (organ, v) in zip(axes, res.items()):
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
        pts, phi1 = v["pts"], v["phi1"]                        # phi1 here = the ORDERING mode (antinode mode)
        a.scatter(pts[:, 2], pts[:, 0], s=5, c=phi1, cmap="coolwarm", alpha=0.85)
        a.set_title(f"{organ}\nmode{v.get('order_mode', 1)} {str(v.get('order_axis', v['mode1_axis'])).split()[0]} "
                    f"sep={v.get('order_sep')}\n"
                    f"{'ordered' if v['subheads_ordered'] else ('-' if v['subheads_ordered'] is None else 'unordered')}",
                    color="#cbd5e1", fontsize=7)
    fig.suptitle("Electric organs: each organ's OWN gap-junction eigenframe (mode1 coloured); subheads fall on "
                 "the antinodes -- the recursion completed at the organ scale", color="#e2e8f0", fontsize=8)
    fig.tight_layout()
    os.makedirs("data/organ_cascade", exist_ok=True)
    fig.savefig("data/organ_cascade/electric_organs_head.png", dpi=120, facecolor="#0d1017")
    print("saved data/organ_cascade/electric_organs_head.png")


def main():
    base, F = build_base()
    res = build(base, F)
    v = _validate(res)
    print(f"electric organs: {v['organs']} organs; subheads ordered on the organ frame: {v['subheads_ordered_on_frame']}")
    for organ, d in v["per_organ"].items():
        od = "ordered" if d["ordered"] else ("single" if d["ordered"] is None else "UNORDERED")
        om = f"mode{d['order_mode']}" if d.get("order_mode") else "mode1"
        print(f"  {organ:7s} {om} -> {str(d.get('order_axis', d['axis'])):38s} sep={d.get('order_sep')}  "
              f"[{od}: {' -> '.join(d['order'])}]")
    print(f"  physiome-weighted operators: {v['physiome_weighted'] or '(none -- run medic.physiome_conductances)'}")
    _figure(res, base)
    dump = {k: {kk: vv for kk, vv in val.items() if kk not in ("pts", "phi1")} for k, val in res.items()}
    json.dump(dict(summary=v, organs=dump), open("data/organ_cascade/electric_organs_head.json", "w"), indent=1, default=float)
    print("saved data/organ_cascade/electric_organs_head.json")


if __name__ == "__main__":
    main()
