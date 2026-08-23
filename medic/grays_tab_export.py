"""grays_tab_export.py -- emit data/grays_parts.json for the human viewer's GRAY'S tab (Miles, 2026-08-07).

The viewer already renders the movie from one JSON. This adds a second, on-demand JSON so a new "Gray's" tab
can show EVERY named part as its own isolable 3D cloud, coloured by pass/fail, with the identity-level checklist
(the same scorecard checks) and -- where one is mapped -- the matching 1918 Gray's plate beside it.

It REUSES medic.grays_scorecard.collect_all_parts + score_part, so the parts, clouds and checklists are byte-for
-byte the scorecard's -- no re-derivation, no drift. Clouds are in the shared assemble() frame
(col0=AP, col1=DV, col2=ML); the viewer applies the SAME AP->up / ML->X / DV->Z remap as the movie.

Plates: best-effort download of curated 1918 Gray's plates (public domain) from Wikimedia Commons. Non-fatal --
a part with no mapped/downloaded plate falls back to checklist-only in the viewer.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.grays_tab_export
"""
from __future__ import annotations
import os, json
import numpy as np

from medic.grays_scorecard import collect_all_parts, score_part, _np

OUT = "data/movie/grays_parts.json"
PLATE_DIR = "data/grays_plates"
MAX_PTS = 340                                                   # per-part cloud subsample cap (keeps JSON light)

# category -> display colour (r,g,b 0..1) for the roster + the isolated cloud
CAT_COLOR = {
    "skull": (0.30, 0.46, 0.95), "vertebrae": (0.46, 0.55, 0.91), "rib_cage": (0.94, 0.94, 0.86),
    "shoulder_girdle": (0.85, 0.62, 0.35), "pelvic_girdle": (0.80, 0.52, 0.30), "limb_bone": (0.55, 0.75, 0.55),
    "autopod": (0.60, 0.80, 0.62), "patella": (0.65, 0.80, 0.65), "hyoid": (0.70, 0.75, 0.70),
    "organ": (0.93, 0.36, 0.34), "muscle": (0.86, 0.42, 0.42),
}

# curated part-key SUBSTRING -> (Gray's 1918 plate file on Commons, caption). First match wins.
PLATE_MAP = [
    ("skull:__whole__",   ("Gray188.png", "Skull, lateral view (Gray's Fig. 188)")),
    (":femur",            ("Gray246.png", "Right femur, anterior surface (Fig. 246)")),
    (":humerus",          ("Gray207.png", "Left humerus, anterior view (Fig. 207)")),
    (":radius",           ("Gray213.png", "Bones of the left forearm (Fig. 213)")),
    (":ulna",             ("Gray213.png", "Bones of the left forearm (Fig. 213)")),
    (":tibia",            ("Gray258.png", "Right tibia, anterior surface (Fig. 258)")),
    (":fibula",           ("Gray262.png", "Right fibula (Fig. 262)")),
    ("scapula",           ("Gray202.png", "Left scapula, posterior (Fig. 202)")),
    ("clavicle",          ("Gray200.png", "Left clavicle (Fig. 200)")),
    ("ilium",             ("Gray235.png", "Right hip bone, external (Fig. 235)")),
    ("ischium",           ("Gray235.png", "Right hip bone, external (Fig. 235)")),
    ("pubis",             ("Gray235.png", "Right hip bone, external (Fig. 235)")),
    ("hand-or-foot",      ("Gray219.png", "Bones of the left hand (Fig. 219)")),
    ("mandible",          ("Gray178.png", "Mandible, outer surface (Fig. 178)")),
    ("patella",           ("Gray255.png", "Right patella (Fig. 255)")),
    ("hyoid",             ("Gray186.png", "Hyoid bone (Fig. 186)")),
    ("vertebra:C1",       ("Gray86.png",  "Atlas (first cervical vertebra, Fig. 86)")),
    ("vertebra:C2",       ("Gray87.png",  "Axis (second cervical vertebra, Fig. 87)")),
    ("vertebra:C",        ("Gray84.png",  "A typical cervical vertebra (Fig. 84)")),
    ("vertebra:T",        ("Gray88.png",  "A thoracic vertebra (Fig. 88)")),
    ("vertebra:L",        ("Gray92.png",  "A lumbar vertebra (Fig. 92)")),
    ("rib",               ("Gray122.png", "A central rib (Fig. 122)")),
    ("organ:Heart",       ("Gray490.png", "Heart, anterior surface (Fig. 490)")),
    ("organ:Lung",        ("Gray973.png", "Lungs, anterior view (Fig. 973)")),
    ("organ:Kidney",      ("Gray1120.png","Right kidney, longitudinal section (Fig. 1120)")),
    ("organ:Liver",       ("Gray1075.png","Liver, anterior surface (Fig. 1075)")),
    ("organ:Gut",         ("Gray1039.png","Abdominal viscera / intestines (Fig. 1039)")),
    ("organ:Spleen",      ("Gray1188.png","Spleen, visceral surface (Fig. 1188)")),
    ("organ:Bladder",     ("Gray1136.png","Urinary bladder (Fig. 1136)")),
    ("organ:Pancreas",    ("Gray1100.png","Pancreas (Fig. 1100)")),
    ("organ:Eye",         ("Gray869.png", "Eyeball, horizontal section (Fig. 869)")),
    ("organ:Spinal Cord", ("Gray665.png", "Spinal cord (Fig. 665)")),
    ("organ:Forebrain",   ("Gray728.png", "Cerebrum, lateral surface (Fig. 728)")),
    ("organ:Cerebellum",  ("Gray702.png", "Cerebellum, upper surface (Fig. 702)")),
    ("organ:Midbrain",    ("Gray719.png", "Brain stem (Fig. 719)")),
    ("organ:Hindbrain",   ("Gray719.png", "Brain stem (Fig. 719)")),
]


def _plate_for(key):
    for sub, pc in PLATE_MAP:
        if sub in key:
            return pc
    return None


def _subsample(P):
    P = np.asarray(P, float)
    if len(P) <= MAX_PTS:
        return P
    rng = np.random.default_rng(len(P))                        # deterministic per part
    return P[rng.choice(len(P), MAX_PTS, replace=False)]


def _download_plates(need):
    """Best-effort pull of the curated plates from Wikimedia Commons. Returns set of files present locally."""
    os.makedirs(PLATE_DIR, exist_ok=True)
    # only files >5KB are real plates; smaller = a Commons "not found" placeholder
    have = {f for f in os.listdir(PLATE_DIR)
            if f.lower().endswith((".png", ".jpg")) and os.path.getsize(os.path.join(PLATE_DIR, f)) > 5000}
    if os.environ.get("COGNIMED_SKIP_PLATE_DL"):
        return have
    todo = sorted(need - have)
    if not todo:
        return have
    try:
        import requests
        try:
            import truststore; truststore.inject_into_ssl()   # use the OS (Netskope) trust store
        except Exception:
            pass
        S = requests.Session(); S.headers["User-Agent"] = "cognimed-grays-viewer/1.0 (research)"
        API = "https://commons.wikimedia.org/w/api.php"
        for fn in todo:
            try:
                r = S.get(API, params=dict(action="query", titles=f"File:{fn}", prop="imageinfo",
                                           iiprop="url", iiurlwidth=520, format="json"), timeout=25)
                pages = r.json()["query"]["pages"]
                info = next(iter(pages.values())).get("imageinfo")
                if not info:
                    print(f"  plate MISS (no such file): {fn}"); continue
                url = info[0].get("thumburl") or info[0]["url"]
                img = S.get(url, timeout=40).content
                open(os.path.join(PLATE_DIR, fn), "wb").write(img)
                have.add(fn); print(f"  plate OK: {fn} ({len(img)//1024} KB)")
            except Exception as e:
                print(f"  plate FAIL {fn}: {e}")
    except Exception as e:
        print(f"  [plates skipped -- {e}; checklist-only fallback]")
    return have


def run():
    from medic.integrated_body import assemble
    print("assembling body ...")
    R = assemble()
    parts = collect_all_parts(R)
    # skull-as-whole composite (mirrors grays_scorecard.run)
    skull_P = np.vstack([v["P"] for v in R.get("skull", {}).values()
                         if isinstance(v, dict) and v.get("P") is not None])
    scored = [(("skull:__whole__", "skull", skull_P, {}),
               score_part("skull:__whole__", "skull", skull_P, {}, R))]
    for (name, cat, P, meta) in parts:
        scored.append(((name, cat, P, meta), score_part(name, cat, P, meta, R)))

    # curated plates we might use
    need = set()
    for (name, *_), _s in scored:
        pc = _plate_for(name)
        if pc:
            need.add(pc[0])
    have = _download_plates(need)

    # MATURE all parts together as ONE body so the tab shows the SAME proportioned Vitruvian figure the movie
    # renders, not the raw un-matured build_base cloud (the "insect"). Score on the RAW parts (the scorecard);
    # display the MATURED cloud.
    from medic.human_movie import mature_for_display
    from medic.unified_embryo import FIDX as _FIDX
    _CAT_FATE = {"skull": "Forebrain", "vertebrae": "Cartilage", "rib_cage": "Cartilage",
                 "shoulder_girdle": "Connective", "pelvic_girdle": "Connective", "limb_bone": "Limb Bud",
                 "autopod": "Limb Bud", "patella": "Cartilage", "hyoid": "Cartilage", "muscle": "Muscle"}
    def _pf(name, cat):
        nm = name.split(":", 1)[1] if (cat == "organ" and ":" in name) else _CAT_FATE.get(cat, "Connective")
        return _FIDX.get(nm, _FIDX.get("Connective", 0))
    _allP = np.vstack([np.asarray(P, float) for (n, c, P, m), s in scored])
    _allf = np.concatenate([np.full(len(P), _pf(n, c), int) for (n, c, P, m), s in scored])
    _mat = mature_for_display(_allP, _allf)
    _matured, _i = [], 0
    for (n, c, P, m), s in scored:
        _matured.append(_mat[_i:_i + len(P)]); _i += len(P)

    out_parts = []
    for pi, ((name, cat, P, meta), s) in enumerate(scored):
        Ps = _subsample(_matured[pi])
        pc = _plate_for(name)
        plate = pc[0] if (pc and pc[0] in have) else None
        out_parts.append(dict(
            key=name, name=name.split(":", 1)[-1] if ":" in name else name, cat=cat,
            score=s["score"], passed=s["passed"], nfeat=s["n_features"], ncells=int(len(P)),
            feats=[dict(f=ff["feature"], ok=bool(ff["passed"]), d=ff["detail"]) for ff in s["features"]],
            xyz=[round(float(v), 4) for v in Ps.reshape(-1)],
            plate=plate, plate_cap=(pc[1] if pc else None),
        ))

    # global centre + radius (shared frame) for the viewer camera / context cloud
    allpts = _mat
    ctr = allpts.mean(0); rad = float(np.percentile(np.linalg.norm(allpts - ctr, axis=1), 99))
    by_cat = {}
    for p in out_parts:
        by_cat.setdefault(p["cat"], []).append(p["score"])
    doc = dict(
        parts=out_parts,
        cat_color={c: [round(x, 3) for x in rgb] for c, rgb in CAT_COLOR.items()},
        cat_order=["skull", "vertebrae", "rib_cage", "shoulder_girdle", "pelvic_girdle",
                   "limb_bone", "autopod", "patella", "hyoid", "organ", "muscle"],
        center=[round(float(v), 4) for v in ctr], radius=round(rad, 4),
        n_parts=len(out_parts),
        mean_score=round(float(np.mean([p["score"] for p in out_parts])), 3),
        fully_passing=int(sum(p["score"] == 1.0 for p in out_parts)),
        by_category={c: round(float(np.mean(v)), 3) for c, v in by_cat.items()},
        n_plates=int(sum(1 for p in out_parts if p["plate"])),
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), default=_np, separators=(",", ":"))
    mb = os.path.getsize(OUT) / 1e6
    print(f"\nGRAY'S TAB DATA -> {OUT}  ({doc['n_parts']} parts, {mb:.1f} MB, {doc['n_plates']} plates)")
    print(f"  mean {doc['mean_score']}  fully-pass {doc['fully_passing']}/{doc['n_parts']}")
    print(f"  by category: {doc['by_category']}")


if __name__ == "__main__":
    run()
