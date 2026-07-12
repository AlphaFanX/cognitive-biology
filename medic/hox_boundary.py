"""
Genome-derived AP address, single-cell + posterior-boundary route.

The bulk / weighted-mean Hox attempt failed (medic/hox_address.py): bulk tissue mixes cell
types, and a weighted mean is meaningless for Hox-negative tissues. This tries the principled
alternative:
  * SINGLE-CELL: use each organ's DEFINING cell type, not the bulk tissue (HPA single-cell nTPM).
  * POSTERIOR BOUNDARY: the AP code is the most-posterior Hox a cell expresses -- the Hox
    colinearity boundary -- not an average. Anterior cell types express no Hox (boundary 0);
    posterior cell types express up to a high paralog group.

Derived AP = the posterior Hox boundary of the organ's defining cell type, normalized.
Validated against the anatomical AP order.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.hox_boundary
Out: data/hox_boundary.{json,png}
"""
from __future__ import annotations
import io, re, json, zipfile
from pathlib import Path
import numpy as np

SC_ZIP = Path("data/hpa_sc.zip")
THRESH = 1.0  # nTPM above which a Hox gene counts as expressed

HOX = ([f"HOXA{i}" for i in list(range(1, 8)) + [9, 10, 11, 13]] +
       [f"HOXB{i}" for i in list(range(1, 10)) + [13]] +
       [f"HOXC{i}" for i in list(range(4, 14))] +
       [f"HOXD{i}" for i in [1, 3, 4, 8, 9, 10, 11, 12, 13]])
PG = {g: int(re.match(r"HOX[A-D](\d+)", g).group(1)) for g in HOX}

# organ -> (defining-cell-type search terms, anatomical AP for validation only)
ORGANS = [
    ("forebrain", ["excitatory neurons", "neurons", "neuronal"], 0.07),
    ("heart",     ["cardiomyocytes"],                              0.30),
    ("lung",      ["alveolar", "pneumocyte", "club cells"],        0.30),
    ("liver",     ["hepatocytes"],                                 0.40),
    ("stomach",   ["gastric", "gland cells"],                      0.44),
    ("limb",      ["chondrocytes", "skeletal myocytes", "fibroblasts"], 0.50),
    ("kidney",    ["proximal tubular", "tubul", "collecting duct", "renal", "podocyte", "kidney"], 0.56),
    ("intestine", ["enterocytes", "intestinal", "colon"],          0.64),
]


def load_hox_sc():
    z = zipfile.ZipFile(io.BytesIO(SC_ZIP.read_bytes()))
    lines = z.read(z.namelist()[0]).decode("utf-8", "replace").splitlines()
    want = set(HOX)
    sc, celltypes = {}, set()
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) < 4:
            continue
        gene, ct, ntpm = p[1], p[2].lower(), p[-1]
        celltypes.add(ct)
        if gene in want:
            try:
                sc.setdefault(gene, {})[ct] = float(ntpm)
            except ValueError:
                pass
    return sc, celltypes


def pick_cell(terms, celltypes):
    for t in terms:
        t = t.lower()
        for ct in sorted(celltypes):
            if t in ct:
                return ct
    return None


def posterior_boundary(sc, ct):
    """The most-posterior Hox paralog group the cell expresses above threshold."""
    pgs = [PG[g] for g, d in sc.items() if d.get(ct, 0.0) >= THRESH]
    n = len(pgs)
    return (max(pgs) if pgs else 0), n


def main():
    Path("data").mkdir(exist_ok=True)
    sc, celltypes = load_hox_sc()
    print(f"HPA single-cell: {len(celltypes)} cell types; {len(sc)} Hox genes detected")
    rows = []
    for organ, terms, anat in ORGANS:
        ct = pick_cell(terms, celltypes)
        if ct is None:
            print(f"  {organ:10s} NO cell type for {terms}"); continue
        bnd, n = posterior_boundary(sc, ct)
        rows.append(dict(organ=organ, cell_type=ct, anatomical_ap=anat,
                         posterior_boundary=bnd, n_hox_expressed=n))
        print(f"  {organ:10s} <- {ct:34s} posterior-Hox boundary {bnd:2d}  ({n} Hox expressed)")

    bnd = np.array([r["posterior_boundary"] for r in rows], float)
    lo, hi = bnd.min(), bnd.max()
    for r in rows:
        r["derived_ap"] = round(float((r["posterior_boundary"] - lo) / (hi - lo + 1e-9)), 3)
    anat = np.array([r["anatomical_ap"] for r in rows]); der = np.array([r["derived_ap"] for r in rows])
    from scipy.stats import spearmanr
    rho, p = spearmanr(der, anat); pear = float(np.corrcoef(der, anat)[0, 1])
    print(f"\nderived AP (single-cell posterior-Hox boundary) vs anatomical AP: "
          f"Spearman {rho:.2f} (p={p:.1e}), Pearson {pear:.2f}")
    for r in sorted(rows, key=lambda x: x["derived_ap"]):
        print(f"    {r['organ']:10s} derived AP {r['derived_ap']:.2f} (anat {r['anatomical_ap']:.2f})  "
              f"boundary Hox{r['posterior_boundary']}")

    verdict = "DERIVED" if rho >= 0.7 else ("PARTIAL" if rho >= 0.4 else "OPEN PROBLEM")
    print(f"\nverdict: {verdict}  (Spearman {rho:.2f})")
    json.dump(dict(spearman=round(float(rho), 3), pearson=round(pear, 3), verdict=verdict, organs=rows),
              open("data/hox_boundary.json", "w"), indent=2)
    print("saved data/hox_boundary.json")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        order = np.argsort(der); names = [rows[i]["organ"] for i in order]
        ax[0].barh(range(len(order)), [rows[i]["posterior_boundary"] for i in order], color="#55a868")
        ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(names)
        ax[0].set_xlabel("posterior Hox boundary (max paralog group expressed)")
        ax[0].set_title("Genome-derived AP code, single-cell posterior boundary\n(HPA single-cell Hox, defining cell type)")
        ax[1].scatter(der, anat, s=60)
        for i in range(len(rows)):
            ax[1].annotate(rows[i]["organ"], (der[i], anat[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5)
        ax[1].set_xlabel("derived AP (posterior-Hox boundary)"); ax[1].set_ylabel("anatomical AP")
        ax[1].set_title(f"derived vs anatomical AP (Spearman {rho:.2f}) -- {verdict}")
        plt.tight_layout(); plt.savefig("data/hox_boundary.png", dpi=140); plt.close(fig)
        print("saved data/hox_boundary.png")
    except Exception as ex:
        print("figure skipped:", repr(ex))


if __name__ == "__main__":
    main()
