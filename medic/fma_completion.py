"""
fma_completion.py -- THE ORGAN CODE CHECKED AGAINST THE FMA, every organ and assemblage (cycle 64;
Miles 2026-09-05: "build the completion instrument, and for every organ and assemblage in the fma.
this is all canonical... could we use the fma to complete the genomic organ code and the organ
cascade? we can check/complete the organ code against the fma?").

THE RELATION: the FMA is the organ code's OUTPUT SPECIFICATION, and the part-of tree (of the lineage
kind -- bones in a limb, lobes in a gland, chambers in a heart) is the developmental descent tree run
to completion: anatomy is frozen development. So for every FMA node under a head's territory there
must exist a developmental EVENT that produces it, TF-owned and clock-windowed, and the code's
completion criterion becomes FORMAL: complete <=> every node at depth has an owning head + event +
window. Reading it backward, every FMA node WITHOUT a code entry is a PREDICTED MISSING SUBHEAD --
the systematic want-list for the SEdb/AlphaGenome head-discovery pipeline. The recursion cascade
gets its termination spec: recurse until the subtree is covered.

THE LEDGER (v1, two measurable columns + the want-list):
  coded      the node name matches a model fate / subhead (the code entry exists)
  completed  the matched fate carries cells at term (the honest full-cloud census, >= 20)
  want-list  named FMA nodes with NO code entry = the predicted subheads, per organ

Run:  venv_win_new/Scripts/python.exe -m medic.fma_completion
Out:  data/organ_cascade/fma_completion.json
"""
from __future__ import annotations
import json, os
import numpy as np

OUT = "data/organ_cascade/fma_completion.json"

# the organ/assemblage ROOTS (FMA ids from the bp3d graph; sided roots take the right side --
# the code is side-symmetric)
ROOTS = {
    "heart": "FMA7088", "right lung": "FMA7310", "liver": "FMA7197", "right kidney": "FMA7204",
    "brain": "FMA50801", "stomach": "FMA7148", "pancreas": "FMA7198", "spleen": "FMA7196",
    "urinary bladder": "FMA15900", "small intestine": "FMA7200", "large intestine": "FMA7201",
    "right foot": "FMA9664", "right hand": "FMA9712", "right free upper limb": "FMA24878",
    "right free lower limb": "FMA24879", "vertebral column": "FMA13478", "skull": "FMA46565",
}

_STOP = {"of", "the", "and", "right", "left", "bone", "lobe", "part", "zone", "region", "wall",
         "surface", "first", "second", "third", "fourth", "fifth", "proximal", "middle", "distal"}


def _toks(s):
    return {w for w in "".join(c if c.isalnum() else " " for c in str(s).lower()).split()
            if w not in _STOP and len(w) > 2}


def run():
    from medic.canonical_atlas import _bp3d
    from medic.canon_measurements import _fma_names
    from medic.unified_embryo import FIDX
    S = _bp3d()
    names = _fma_names()
    # the CODE side: every fate the model can express, token-indexed
    fate_toks = {f: _toks(f) for f in FIDX}
    # the CELLS side: the honest full-cloud census (no build needed)
    census = {}
    # THE ADULT CENSUS (cycle 79 -- the coded-but-empty conviction): the cloud census's last
    # frame is the EMBRYO (subhead sorting -- Stomach out of Foregut, Bladder, hepatic lobes --
    # happens in the build), so stomach read 43 coded / 3 completed and bladder 24/0 against a
    # census with NO Stomach/Bladder keys at all. Cells-at-term now prefers the adult census
    # (regenerate: python -m medic.fma_completion --census; ~90 s build_base, no movie regen).
    apath = "data/organ_cascade/adult_census.json"
    if os.path.exists(apath):
        census = json.load(open(apath)).get("counts", {})
    else:
        cpath = "data/movie/cloud_census.json"
        if os.path.exists(cpath):
            c = json.load(open(cpath))
            # census-format fix (cycle 68): {"note", "census": [per-frame {counts}]} -- read
            # the LAST frame. (V0 looked for a top-level "counts" key and always read 0.)
            if isinstance(c, dict) and isinstance(c.get("census"), list) and c["census"]:
                census = c["census"][-1].get("counts", {})
            elif isinstance(c, dict):
                census = c.get("counts", {})

    # THE PARTS SIDE (cycle 77 -- Miles: "continue with the increase in the fma"): the coded
    # test saw only the FATE table, but the organ code's OUTPUT includes the part-resolved
    # anatomy -- 29 named vertebrae, limb bones, autopod blocks, ~140 named muscles, 28 teeth,
    # girdles, ribs -- each a BUILT structure with cells (the ledger read vertebral column 0/53
    # while the vertebrae exist as model cells). Parts enter the code table with their cell
    # counts. HONESTY RULES: parts match at a STRICTER Jaccard (0.5 vs the fates' 0.4);
    # ordinals + proximal/middle/distal stay as DISCRIMINATORS on the parts side (they are in
    # the fate stop-list) so one phalanx part cannot code fourteen phalanges; vertebra/rib
    # shorthand expands honestly (C7 <-> seventh cervical vertebra).
    _ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
            "ninth", "tenth", "eleventh", "twelfth"]
    _PSTOP = _STOP - set(_ORD[:5]) - {"proximal", "middle", "distal"}

    def _ptoks(s):
        out = set()
        for w in "".join(c if c.isalnum() else " " for c in str(s).lower()).split():
            if w in _PSTOP or len(w) < 2:
                continue
            m = None
            for pre, full in (("c", "cervical"), ("t", "thoracic"), ("l", "lumbar")):
                if w.startswith(pre) and w[len(pre):].isdigit():
                    m = (full, int(w[len(pre):]))
            if w.startswith("rib") and w[3:].isdigit():
                m = ("rib", int(w[3:]))
            if m and 1 <= m[1] <= 12:
                out |= {m[0], _ORD[m[1] - 1], "vertebra" if m[0] != "rib" else "rib"}
            else:
                out.add(w)
        return out

    part_toks = {}                          # display name -> (tokens, n_cells)
    try:
        cs = json.load(open("data/organ_cascade/canonical_scorecard.json"))
        for p in cs.get("parts", []):
            n_c = int(((p.get("ours") or {}).get("n")) or 0)
            t = _ptoks(str(p.get("part", "")).replace("__", " ")) | _ptoks(p.get("ref", ""))
            if t:
                part_toks[f"part:{p.get('part')}"] = (t, n_c)
    except Exception:
        pass
    try:
        import glob as _g
        for fp in _g.glob("data/grays_scorecard/*.json"):
            g = json.load(open(fp))
            k = f"grays:{g.get('part')}"
            t = _ptoks(str(g.get("part", "")).replace("__", " "))
            if t and not any(k2.endswith(str(g.get("part"))) for k2 in part_toks):
                part_toks[k] = (t, int(g.get("n_cells", 0)))
    except Exception:
        pass

    # DISCRIMINATORS ON THE FATE SIDE TOO (cycle 82g): the fate stop-list drops side, ordinal and
    # proximal/middle/distal, so one sided, ordinal fate ("Proximal Phalanx of Left Third Toe") would
    # code every toe phalanx in the tree. A fate that CARRIES such words may only match a node that
    # carries all of them.
    _DISC = {"left", "right", "proximal", "middle", "distal"} | set(_ORD[:5])
    _raw = lambda s: set("".join(c if c.isalnum() else " " for c in str(s).lower()).split())
    fate_disc = {f: (_raw(f) & _DISC) for f in FIDX}

    def code_match(node_name):
        """-> (matched name, cells-at-term-or-None) | (None, None). Fates first (census carries
        their cells), then the built parts (their own cell counts)."""
        nt = _toks(node_name)
        if not nt:
            return None, None
        nraw = _raw(node_name)
        best, bj = None, 0.0
        for f, ft in fate_toks.items():
            if not ft:
                continue
            if fate_disc[f] and fate_disc[f] != (nraw & _DISC):
                continue                      # a discriminated fate matches only its own side/ordinal/level
                # (subset was not enough: "Middle Phalanx of Right Middle Finger" -- 'middle' twice --
                #  passed for "distal phalanx of right middle finger" and won the tie by FIDX order)
            j = len(nt & ft) / len(nt | ft)
            if j > bj:
                best, bj = f, j
        if bj >= 0.4:
            return best, census.get(best, 0)
        npt = _ptoks(node_name)
        bestp, bjp, cells = None, 0.0, 0
        for k, (t, n_c) in part_toks.items():
            j = len(npt & t) / len(npt | t) if npt | t else 0.0
            if j > bjp:
                bestp, bjp, cells = k, j, n_c
        if bjp >= 0.5:
            return bestp, cells
        return None, None

    ledger, want_all = {}, {}
    print("\nTHE ORGAN CODE vs THE FMA (coded = a fate/subhead exists; completed = cells at term):")
    print(f"{'root':26s} {'nodes':>6s} {'coded':>6s} {'compl':>6s}  first uncoded (the predicted subheads)")
    for label, root in sorted(ROOTS.items()):
        seen, stack, nodes = set(), [root], []
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n != root and n in names:
                nodes.append(n)
            stack.extend(S["edges"].get(n, ()))
        rows, want = [], []
        if not nodes:
            # THE FULL-FMA FALLBACK (cycle 78 -- the acquisition): the bp3d part-of graph is
            # viscera-shallow (liver/kidney/spleen/stomach/bladder have ZERO children in both
            # local tables -- measured). data/fma.sqlite (mhalle/fma-sqlite snapshot, 104,517
            # terms) fills those roots. DECLARED LIMIT: the snapshot carries the IS-A taxonomy,
            # so partonomy is enumerated LEXICALLY (terms containing "of <organ>"; sided "left"
            # variants folded -- the code is side-symmetric, matching the bp3d convention);
            # the true part-of graph awaits the OBO fma.owl subset.
            organ = label.replace("right ", "")
            try:
                import sqlite3
                _db = sqlite3.connect("data/fma.sqlite")
                q = _db.execute("SELECT id, name FROM fma WHERE name LIKE ? COLLATE NOCASE",
                                (f"%of {organ}%",)).fetchall()
                _db.close()
                for _id, _nm in q:
                    if "left " in str(_nm).lower():
                        continue                              # sided fold: generic + right only
                    fid = f"FMAS{_id}"
                    names[fid] = _nm
                    nodes.append(fid)
            except Exception as _e:
                print(f"{label:26s}    -- (sqlite fallback failed: {_e})")
            if not nodes:
                print(f"{label:26s}    -- (root has no named descendants in the graph)")
                continue
        for n in nodes:
            nm = names[n]
            f, cells = code_match(nm)
            comp = bool(f) and (cells or 0) >= 20
            rows.append(dict(fma=n, name=nm, coded=f, completed=comp))
            if not f:
                want.append(nm)
        nc = sum(1 for r in rows if r["coded"])
        ncomp = sum(1 for r in rows if r["completed"])
        ledger[label] = dict(root=root, n_nodes=len(rows), coded=nc, completed=ncomp, rows=rows,
                             want=want)
        want_all[label] = want
        print(f"{label:26s} {len(rows):6d} {nc:6d} {ncomp:6d}  {', '.join(want[:4])[:60]}")
    total_nodes = sum(v["n_nodes"] for v in ledger.values())
    total_coded = sum(v["coded"] for v in ledger.values())
    total_comp = sum(v["completed"] for v in ledger.values())
    print(f"\nTOTALS: {total_nodes} FMA nodes under {len(ledger)} roots -- coded {total_coded} "
          f"({100 * total_coded // max(1, total_nodes)}%), completed {total_comp} "
          f"({100 * total_comp // max(1, total_nodes)}%)")
    print("THE WANT-LIST = the organ code's predicted missing subheads, per organ, hierarchical --")
    print("the systematic target for the SEdb/AlphaGenome head-discovery pipeline.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(ledger, open(OUT, "w"), indent=1)
    return ledger


def write_adult_census():
    """Dump the SCORED-body fate census (build_base, subhead-sorted -- the true cells-at-term
    object) to data/organ_cascade/adult_census.json."""
    import numpy as np
    from medic.integrated_body import assemble, mature_parts
    from medic.unified_embryo import FATES
    # THE MATURED SCORED BODY (cycle 82g): cells-at-term = the body tier A scores = mature_parts, where
    # the autopod landing names the 76 hand/foot bones (build_base alone carries no autopod names).
    F = np.asarray(mature_parts(assemble())["F"])
    names_, counts_ = np.unique(np.asarray(F), return_counts=True)
    out = {FATES[int(i)] if 0 <= int(i) < len(FATES) else str(i): int(c)
           for i, c in zip(names_, counts_)}
    # PARENT ROLL-UP (cycle 80 -- the families law applied to the census): at term the
    # subheads claim ALL the cells (every liver cell is a hepatic lobe), so parent keys read
    # empty and the ledger's completed column collapsed (liver 42->0). A parent's census =
    # its own cells + its children's -- expand_names' composites, rolled up.
    from medic.subhead_program import composites
    for parent, fam in composites().items():
        out[parent] = sum(out.get(n, 0) for n in fam)
    json.dump(dict(note="build_base adult fate census (subhead-sorted; cells at term)",
                   n=int(len(F)), counts=out),
              open("data/organ_cascade/adult_census.json", "w"), indent=1)
    print(f"adult census -> data/organ_cascade/adult_census.json ({len(out)} fates, {len(F):,d} cells)")


if __name__ == "__main__":
    import sys
    write_adult_census() if "--census" in sys.argv else run()
