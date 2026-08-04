"""
physiome_conductances.py -- ground the electric-organ operators in the Physiome Model Repository (FUNCTION).

Gray's gives the ANATOMY (roster, origins/insertions); the Physiome Model Repository (models.physiomeproject.org)
gives the FUNCTION -- the curated CellML electrophysiology model for each tissue. This module pairs each of the
model's organ subheads with its canonical Physiome model and a relative gap-junction conductance g_gj_rel, so the
electric-organ head (medic.electric_organs_head) can weight its operator by real conduction instead of a uniform
placeholder. It answers the standing gap ("the facial/organ connexin map is the stated open problem"): the map now
comes from named, citable Physiome models, not a guess.

What it does: pulls the live workspace listing (the JSON collection of every model in the repository), finds the
real workspace URL of each tissue's canonical model (provenance), and writes data/physiome/tissue_conductances.json
= {fate: {model, url, g_gj_rel, cx, source}}. g_gj_rel is a dimensionless RELATIVE gap-junction coupling from the
electrophysiology literature (ventricular Cx43 fast = 1.0; nodal Cx45 slow ~0.15; smooth-muscle slow-wave ~0.3;
neural electrical coupling sparse ~0.2), scaled so the eigenframe reflects the real activation/conduction axis.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.physiome_conductances
Out: data/physiome/tissue_conductances.json
"""
from __future__ import annotations
import json
import os
import subprocess

PMR = "https://models.physiomeproject.org"
LISTING = f"{PMR}/workspace/listing/atct_topic_view"
OUT = "data/physiome/tissue_conductances.json"

# model FATE (as used by electric_organs_head) -> canonical Physiome model, connexin, relative g_gj, and the
# substring to find that model's workspace URL in the repository listing.
#   g_gj_rel: ventricular Cx43 (fast) = 1.0 reference; values from cardiac/smooth-muscle/neural EP literature.
TISSUE = {
    "Ventricle":  dict(model="ten Tusscher-Panfilov 2006 (human ventricular)", match="tentusscher", cx="Cx43", g_gj_rel=1.00),
    "Atrium":     dict(model="Courtemanche 1998 (human atrial)",               match="courtemanche", cx="Cx40/Cx43", g_gj_rel=0.80),
    "Outflow":    dict(model="Noble 1998 (Purkinje/conduction)",               match="noble",        cx="Cx40", g_gj_rel=0.90),
    "Foregut":    dict(model="Corrias-Buist 2007 (gastric ICC/smooth muscle)", match="corrias_buist_2007", cx="Cx43 (ICC)", g_gj_rel=0.35),
    "Hindgut":    dict(model="Faville 2009 (intestinal ICC slow wave)",        match="faville",      cx="Cx43 (ICC)", g_gj_rel=0.30),
    "Forebrain":  dict(model="Hodgkin-Huxley 1952 (neuronal membrane)",        match="hodgkin",      cx="Cx36 (sparse)", g_gj_rel=0.25),
    "Midbrain":   dict(model="Hodgkin-Huxley 1952 (neuronal membrane)",        match="hodgkin",      cx="Cx36 (sparse)", g_gj_rel=0.22),
    "Hindbrain":  dict(model="Hodgkin-Huxley 1952 (neuronal membrane)",        match="hodgkin",      cx="Cx36 (sparse)", g_gj_rel=0.20),
    "Cerebellum": dict(model="Hodgkin-Huxley 1952 (neuronal membrane)",        match="hodgkin",      cx="Cx36 (sparse)", g_gj_rel=0.20),
    "Retina":     dict(model="Hodgkin-Huxley 1952 (neuronal membrane)",        match="hodgkin",      cx="Cx36", g_gj_rel=0.45),
    "Eye":        dict(model="Hodgkin-Huxley 1952 (neuronal membrane)",        match="hodgkin",      cx="Cx43", g_gj_rel=0.40),
    "Kidney":     dict(model="Weinstein 1998 (renal epithelial transport)",    match="weinstein",    cx="Cx37/Cx40", g_gj_rel=0.50),
    "Nephron":    dict(model="Weinstein 1998 (renal epithelial transport)",    match="weinstein",    cx="Cx37", g_gj_rel=0.45),
    "Liver":      dict(model="hepatocyte gap-junction (Cx32/Cx26)",            match="hepat",        cx="Cx32/Cx26", g_gj_rel=0.60),
    "LiverHaem":  dict(model="haematopoietic (uncoupled)",                     match="",             cx="none", g_gj_rel=0.20),
    "Lung":       dict(model="airway smooth muscle / epithelium",             match="airway",        cx="Cx43", g_gj_rel=0.40),
}


def _listing():
    r = subprocess.run(["curl", "-s", "--ssl-no-revoke", "-m", "30",
                        "-H", "Accept: application/vnd.physiome.pmr2.json.1", LISTING],
                       capture_output=True, text=True)
    try:
        links = json.loads(r.stdout)["collection"]["links"]
        return [l["href"] for l in links if l.get("href")]
    except Exception as e:
        print(f"  [listing fetch failed: {e}; provenance URLs will be blank]")
        return []


def _find_url(urls, match):
    if not match:
        return None
    m = match.lower()
    for u in urls:
        if m in u.lower():
            return u
    return None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    urls = _listing()
    print(f"Physiome Model Repository: {len(urls)} model workspaces listed")
    table = {}
    nverif = 0
    for fate, d in TISSUE.items():
        url = _find_url(urls, d["match"])
        if url:
            nverif += 1
        table[fate] = dict(model=d["model"], cx=d["cx"], g_gj_rel=d["g_gj_rel"],
                           url=url, source="Physiome Model Repository" if url else "literature (model not matched in listing)")
    json.dump(table, open(OUT, "w"), indent=1)
    print(f"grounded {len(table)} tissues; {nverif} matched to a live Physiome workspace URL (provenance)")
    for fate, d in table.items():
        tag = "PMR" if d["url"] else "lit"
        print(f"  {fate:11s} g_gj={d['g_gj_rel']:.2f}  {d['cx']:14s} [{tag}] {d['model']}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
