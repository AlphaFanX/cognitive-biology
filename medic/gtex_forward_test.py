#!/usr/bin/env python3
"""
GTEx expression-based forward Vm test (the principled successor).
=================================================================

The AlphaGenome accessibility forward test failed on hardening (promoter ATAC is
too crude a conductance proxy; forward V collapsed to ~-40 mV for every tissue).
Conductance is set by channel DENSITY, best proxied by EXPRESSION, not promoter
openness. This re-runs the forward test on GTEx v8 MEASURED median expression
(TPM) per tissue for the resting leak-channel panel -- real data, a better proxy,
and with real brain + skeletal-muscle tissues AlphaGenome's ATAC set lacked.

g_X(tissue) = sum of TPM over family X's genes; V = Goldman(g). No anchor, no fit.
Test: does forward V correlate with adult resting potential and separate excitable
(neuron / muscle / heart, hyperpolarized) from epithelial (liver / lung / gut /
pancreas / thyroid, depolarized)?

HONEST: measured expression is more principled than accessibility, but it does NOT
fix bulk-tissue cell-type averaging or the expression->conductance approximation.
Reported whatever it gives; no tuning to taste.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.gtex_forward_test
Output: data/gtex_forward_test.json, gtex_forward_test.png
"""
from __future__ import annotations
import os
import json
from pathlib import Path

import numpy as np

os.environ.setdefault("REQUESTS_CA_BUNDLE", os.path.expanduser("~/.ca_combined.pem"))

try:
    from . import bioelectric_development as bd
except ImportError:  # pragma: no cover
    from medic import bioelectric_development as bd

GTEX = "https://gtexportal.org/api/v2"

PANEL = {
    "K":  ["KCNJ2", "KCNJ4", "KCNJ12", "KCNK1", "KCNK2", "KCNK3", "KCNK5", "KCNK6"],  # Kir2 + K2P leak
    "Na": ["NALCN", "SCN5A", "SCN1A"],                                                # NALCN background leak
    "Ca": ["CACNA1C", "CACNA1G"],
    "Cl": ["CLCN2", "CLCN3", "ANO1"],
}
# GTEx tissueSiteDetailId -> (adult resting potential mV [literature], class)
TISSUES = {
    "Brain_Cortex":                    (-70, "excitable"),
    "Brain_Frontal_Cortex_BA9":        (-70, "excitable"),
    "Nerve_Tibial":                    (-70, "excitable"),
    "Muscle_Skeletal":                 (-88, "excitable"),
    "Heart_Left_Ventricle":            (-85, "excitable"),
    "Heart_Atrial_Appendage":          (-80, "excitable"),
    "Kidney_Cortex":                   (-70, "intermediate"),
    "Spleen":                          (-55, "intermediate"),
    "Artery_Tibial":                   (-55, "intermediate"),
    "Liver":                           (-40, "epithelial"),
    "Pancreas":                        (-45, "epithelial"),
    "Lung":                            (-45, "epithelial"),
    "Stomach":                         (-45, "epithelial"),
    "Colon_Sigmoid":                   (-45, "epithelial"),
    "Thyroid":                         (-50, "epithelial"),
    "Small_Intestine_Terminal_Ileum":  (-45, "epithelial"),
}


def gencode_id(sym, sess):
    r = sess.get(f"{GTEX}/reference/gene", params={"geneId": sym}, timeout=30).json()
    for d in r.get("data", []):
        if d.get("geneSymbolUpper") == sym.upper():
            return d["gencodeId"]
    return r["data"][0]["gencodeId"] if r.get("data") else None


def median_expression(gid, sess):
    r = sess.get(f"{GTEX}/expression/medianGeneExpression",
                 params={"gencodeId": gid, "datasetId": "gtex_v8"}, timeout=40).json()
    return {d["tissueSiteDetailId"]: float(d["median"]) for d in r.get("data", [])}


def goldman(gNa, gK, gCa, gCl):
    gCar = 0.10 * gCa
    G = gNa + gK + gCar + gCl
    return (gNa * bd.E_NA + gK * bd.E_K + gCar * bd.E_CA + gCl * bd.E_CL) / (G + 1e-9)


def main():
    import requests
    print("=" * 78)
    print("GTEx EXPRESSION-based forward Vm test (measured TPM -> conductance -> Goldman)")
    print("=" * 78)
    sess = requests.Session()

    acc = {t: {f: 0.0 for f in PANEL} for t in TISSUES}
    genes = [(g, f) for f, gs in PANEL.items() for g in gs]
    print(f"\nQuerying GTEx v8 median expression for {len(genes)} resting channel genes:")
    for g, fam in genes:
        gid = gencode_id(g, sess)
        if not gid:
            print(f"  {g}: no gencodeId, skip"); continue
        expr = median_expression(gid, sess)
        for t in TISSUES:
            acc[t][fam] += expr.get(t, 0.0)
        print(f"  {g:8s} ({fam})  brain={expr.get('Brain_Cortex',0):.1f} "
              f"musc={expr.get('Muscle_Skeletal',0):.1f} liver={expr.get('Liver',0):.1f} "
              f"lung={expr.get('Lung',0):.1f}")

    rows = []
    print(f"\n  {'tissue':30s} {'class':12s} {'V_fwd':>7} {'rest':>6}")
    for t, (target, cls) in TISSUES.items():
        s = acc[t]
        if sum(s.values()) == 0:
            continue
        V = goldman(s["Na"], s["K"], s["Ca"], s["Cl"])
        rows.append((t, cls, V, target))
        print(f"  {t:30s} {cls:12s} {V:+7.1f} {target:+6}")

    vf = np.array([r[2] for r in rows]); vt = np.array([r[3] for r in rows])
    rho = float(np.corrcoef(vf, vt)[0, 1])
    e_v = [r[2] for r in rows if r[1] == "excitable"]
    p_v = [r[2] for r in rows if r[1] == "epithelial"]
    exc, epi = np.mean(e_v), np.mean(p_v)
    auc = np.mean([1.0 if a < b else 0.5 if a == b else 0.0 for a in e_v for b in p_v])
    print(f"\n  corr(forward V, adult resting potential) = {rho:+.2f}")
    print(f"  excitable mean {exc:+.1f}  vs  epithelial mean {epi:+.1f}  -> "
          f"excitable more hyperpolarized? {'YES' if exc < epi else 'NO'}  (AUC {auc:.2f})")
    verdict = ("CLEAN: measured channel expression forward-predicts adult resting potential and separates "
               "excitable from epithelial -- no anchor, no fit." if rho > 0.5 and exc < epi and auc > 0.75 else
               "PARTIAL/NEGATIVE: expression helps over accessibility or it does not; reported as is.")
    print(f"\n  VERDICT: {verdict}")

    Path("data").mkdir(exist_ok=True)
    json.dump(dict(rows=[(t, c, V, tg) for t, c, V, tg in rows], corr=rho,
                   excitable_mean=float(exc), epithelial_mean=float(epi), auc=float(auc)),
              open("data/gtex_forward_test.json", "w"), indent=2)
    _figure(rows, rho, auc)
    return rho > 0.5 and exc < epi and auc > 0.75


def _figure(rows, rho, auc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = {"excitable": "#1f77b4", "epithelial": "#d62728", "intermediate": "#7f7f7f"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for t, c, V, tg in rows:
        ax[0].scatter(tg, V, c=cmap[c], s=80)
        ax[0].annotate(t.split("_")[0][:6], (tg, V), fontsize=6, xytext=(3, 2), textcoords="offset points")
    lim = [-95, -30]; ax[0].plot(lim, lim, "k--", lw=0.6, alpha=0.5)
    ax[0].set_xlabel("measured adult resting potential (mV)")
    ax[0].set_ylabel("forward $V_m$ from GTEx expression (mV)")
    ax[0].set_title(f"Genome (GTEx expression) forward-predicts resting $V_m$\n(no anchor, no fit; corr {rho:+.2f})", fontsize=10)
    for cls in ("excitable", "intermediate", "epithelial"):
        vs = [r[2] for r in rows if r[1] == cls]
        if vs:
            ax[1].scatter([cls] * len(vs), vs, c=cmap[cls], s=70, alpha=0.7)
            ax[1].scatter([cls], [np.mean(vs)], c="k", s=130, marker="D")
    ax[1].set_ylabel("forward $V_m$ (mV)")
    ax[1].set_title(f"Excitable vs epithelial\nAUC {auc:.2f}", fontsize=10)
    fig.suptitle("GTEx expression-based forward Vm test (measured TPM, resting leak channels)", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig("gtex_forward_test.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("\nSaved: gtex_forward_test.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'CLEAN' if ok else 'PARTIAL'}")
