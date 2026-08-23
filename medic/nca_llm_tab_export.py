"""nca_llm_tab_export.py -- emit data/movie/nca_llm.json for the human viewer's NCA+LLM tab (Miles, 2026-08-07).

Exposes the model's INTERNALS down to actual values: the outer LGM (Large Genomic Model), the HEADS (each a
master-TF SE cluster reading an AlphaGenome tissue track), their MASKS (the cells of each fate, in the shared
assemble() frame so they line up with the Gray's tab), the inner NCA's per-fate scalar PARAMETERS
(Vm set-point, cadherin ADH, integrin ECM, PRC2 unlock, proliferation weight), the GenomicChannelLookup MLP
(5 conductances -> Goldman -> Vmem, with its 5 trainable residual params), and the LGM ADAPTER knobs (real GWAS
beta per gene->trait).

Reuses medic.integrated_body.assemble() (same call the Gray's tab uses) for the masks, and the live registries
(head_registry, unified_embryo, differentiation_clock, gene_trait_adapter) so the values never drift from the model.

Run: cd cognimed && venv_win_new/Scripts/python.exe -m medic.nca_llm_tab_export
"""
from __future__ import annotations
import os, json
import numpy as np

OUT = "data/movie/nca_llm.json"
MAX_PTS = 320

# knob-prefix -> the fate/head it acts on (for the direct-knob tag on a head; face/body stay in the global table)
KNOB_HEAD = {"heart": "Heart", "eye": "Eye", "kidney": "Kidney", "limb": "Limb Bud"}


def _sub(P):
    P = np.asarray(P, float)
    if len(P) <= MAX_PTS:
        return P
    rng = np.random.default_rng(len(P))
    return P[rng.choice(len(P), MAX_PTS, replace=False)]


def _channel_lookup_params():
    """The GenomicChannelLookup MLP: 5 ion channels, its 5 trainable residual params, Goldman machinery."""
    info = dict(channels=["g_Na", "g_K", "g_Ca", "g_Cl", "g_GJ"], residual=[0.0] * 5,
                residual_scale="1 + tanh(residual)*0.2  (+-20%)", blend_sigma=0.15,
                v_zygote_mV=-70.0, equation="Goldman: V = sum(g_i*E_i)/sum(g_i)")
    try:
        from medic.four_head_morphogenesis import GenomicChannelLookup, CHANNEL_NAMES
        gl = GenomicChannelLookup()
        info["channels"] = list(CHANNEL_NAMES)
        info["residual"] = [round(float(x), 4) for x in np.asarray(gl.residual).ravel()]
    except Exception as e:
        info["note"] = f"(defaults; live read failed: {str(e)[:60]})"
    return info


def run():
    from medic.integrated_body import assemble
    from medic.unified_embryo import FIDX, ADH, ECM, VM_OF, ORG_TARGET, PAIRED_ORGANS
    from medic.head_registry import REGISTRY
    from medic.differentiation_clock import FATE_PRC2
    from medic.gene_trait_adapter import REGISTRY as KNOBS
    try:
        from medic.head_tree import MASTER_TF
    except Exception:
        MASTER_TF = {}

    prolif = {}
    pf = "data/organ_cascade/proliferation_weights.json"
    if os.path.exists(pf):
        prolif = json.load(open(pf)).get("fate_weight", {})

    print("assembling body ...")
    R = assemble()
    base, F = R.get("base"), R.get("F")
    if base is None or F is None:
        raise SystemExit("assemble() returned no base/F -- cannot build masks")
    base = np.asarray(base, float); F = np.asarray(F)

    # antinode grid of the whole body (the electric frame the heads are placed on)
    antinode = None
    try:
        from medic.body_electric_antinodes import body_electric_antinodes
        a = body_electric_antinodes(base)
        antinode = dict(ap_levels=[round(float(x), 3) for x in a.get("ap_levels", [])],
                        dv=[round(float(a.get(k, 0)), 3) for k in ("dv_ventral", "dv_mid", "dv_dorsal")],
                        lr_node=round(float(a.get("lr_node", 0.5)), 3), has_lr=bool(a.get("has_lr", False)))
    except Exception as e:
        print(f"  [antinode grid skipped: {str(e)[:60]}]")

    # PRC2 unlock: a subhead inherits its parent's if it has none of its own
    def prc2_of(name):
        if name in FATE_PRC2:
            return FATE_PRC2[name]
        par = REGISTRY.get(name, {}).get("parent")
        return FATE_PRC2.get(par) if par else None

    # direct GWAS knobs per head
    head_knobs = {}
    for kname, kv in KNOBS.items():
        h = KNOB_HEAD.get(kname.split(".")[0])
        if h:
            head_knobs.setdefault(h, []).append(dict(knob=kname, gene=kv.get("gene"), snp=kv.get("snp"),
                                                     beta=kv.get("beta"), unit=kv.get("unit"),
                                                     direction=kv.get("direction"), source=kv.get("source")))

    # organs the build stores SPLIT into subheads (heart chambers, gut fore/hind) -- recombine the whole-organ
    # mask so the organ head appears alongside its subheads (mirrors grays_scorecard).
    COMPOSITE = {"Heart": ("Heart", "Atrium", "Ventricle", "Outflow"), "Gut": ("Gut", "Foregut", "Hindgut")}

    from medic.human_movie import mature_for_display
    dbase = mature_for_display(base, F)      # matured (proportioned Vitruvian) cloud for DISPLAY -- matches the movie

    def cells_for(fate):
        if fate in COMPOSITE:
            ids = [FIDX[c] for c in COMPOSITE[fate] if c in FIDX]
            return dbase[np.isin(F, ids)]
        return dbase[F == FIDX[fate]]

    heads = []
    fate_names = list(FIDX.keys()) + [c for c in COMPOSITE if c not in FIDX]
    for fate in fate_names:
        cells = cells_for(fate)
        if len(cells) < 6:
            continue
        reg = REGISTRY.get(fate, {})
        ctr = cells.mean(0)                                    # (AP, DV, ML)
        heads.append(dict(
            name=fate, kind=reg.get("kind", "structural"), parent=reg.get("parent"),
            master_tf=reg.get("tf") or MASTER_TF.get(fate) or "-",
            ag_tissue=reg.get("ag"), implemented=bool(reg.get("impl", fate in FIDX)),
            n_cells=int(len(cells)),
            # ---- the inner NCA's ACTUAL per-fate parameters ----
            vm_setpoint_mV=VM_OF.get(fate), adhesion=ADH.get(fate), ecm=ECM.get(fate),
            prc2_unlock=prc2_of(fate), prolif_weight=round(float(prolif.get(fate)), 3) if fate in prolif else None,
            org_target=ORG_TARGET.get(fate), paired=fate in PAIRED_ORGANS,
            # ---- placement on the electric frame (measured centroid) ----
            ap=round(float(ctr[0]), 3), dv=round(float(ctr[1]), 3), lr=round(float(ctr[2]), 3),
            knobs=head_knobs.get(fate, []),
            xyz=[round(float(v), 4) for v in _sub(cells).reshape(-1)],
        ))
    heads.sort(key=lambda h: {"organ": 0, "subhead": 1, "new": 2, "structural": 3}[h["kind"]])

    # the architecture stack, with the real global values at each layer
    n_impl = sum(1 for h in heads)
    arch = [
        dict(layer="LGM (outer)", role="frozen Large Genomic Model -- the promoter-MLP 'LLM'",
             detail="genes are literal A/T/G/C; AlphaGenome tissue tracks + SEdb SE-clusters -> head identity. "
                    "Predicts the bioelectric Vm LATENT (a JEPA), not molecular pixels.",
             values=dict(heads_registered=len(REGISTRY), adapter_knobs=len(KNOBS))),
        dict(layer="Heads", role="each head = a master-TF super-enhancer cluster reading one tissue track",
             detail="organ / subhead / new. The head is the identity bottleneck (nearest-centroid on 499 "
                    "regulons recovers tissue at 0.74 = 18x chance).",
             values=dict(total=len(heads), organ=sum(h["kind"] == "organ" for h in heads),
                         subhead=sum(h["kind"] == "subhead" for h in heads),
                         new=sum(h["kind"] == "new" for h in heads),
                         structural=sum(h["kind"] == "structural" for h in heads))),
        dict(layer="Masks", role="a head's cells = a fate-map index + its electric antinode",
             detail="fate index FIDX per cell; each head fills the antinodes of the body's gap-junction "
                    "eigenmodes in PRC2-clock order.",
             values=dict(fates=len(FIDX), antinode_grid=antinode)),
        dict(layer="MLP (GenomicChannelLookup)", role="genome -> 5 conductances -> Goldman -> Vmem",
             detail="the inner-loop channel MLP. W = W0(Jadhav zygote kernel) + LoRA dW (ABC/SEdb), clock-gated. "
                    "Per-fate Vm set-points are the supervision target.",
             values=_channel_lookup_params()),
        dict(layer="NCA (inner)", role="Neural Cellular Automaton -- local Sobel perception + update, weight-shared",
             detail="TRM(LLM): recursive deep-supervision on a locally gap-junction-coupled field. Grows the body "
                    "from one cell; heads intercalate (differentiate/divide/migrate/shape).",
             values=dict(perception="3x3 Sobel gradients", update="shared kernel, clock-gated")),
        dict(layer="Body", role="the cell cloud -- the read-out", detail="~{} cells assembled".format(len(base)),
             values=dict(cells=int(len(base)))),
    ]

    knob_table = [dict(knob=k, gene=v.get("gene"), snp=v.get("snp"), trait=v.get("trait"),
                       beta=v.get("beta"), unit=v.get("unit"), direction=v.get("direction"),
                       source=v.get("source"), note=v.get("note")) for k, v in KNOBS.items()]

    # cascade edges (parent -> derivative) from the registry
    edges = [dict(parent=v["parent"], child=k) for k, v in REGISTRY.items() if v.get("parent")]

    allpts = dbase
    ctr = allpts.mean(0); rad = float(np.percentile(np.linalg.norm(allpts - ctr, axis=1), 99))
    fate_color = R.get("anat") or {}                           # if assemble exposes fate colours
    doc = dict(architecture=arch, heads=heads, adapter_knobs=knob_table, cascade_edges=edges,
               kind_order=["organ", "subhead", "new", "structural"],
               kind_color={"organ": [0.93, 0.36, 0.34], "subhead": [0.95, 0.72, 0.30],
                           "new": [0.47, 0.83, 0.98], "structural": [0.55, 0.62, 0.72]},
               center=[round(float(v), 4) for v in ctr], radius=round(rad, 4),
               n_heads=len(heads), n_knobs=len(knob_table))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), separators=(",", ":"))
    mb = os.path.getsize(OUT) / 1e6
    print(f"\nNCA+LLM TAB DATA -> {OUT}  ({doc['n_heads']} heads, {doc['n_knobs']} knobs, {mb:.1f} MB)")
    by = {}
    for h in heads:
        by[h["kind"]] = by.get(h["kind"], 0) + 1
    print(f"  heads by kind: {by}")
    print(f"  channel MLP residual: {arch[3]['values'].get('residual')}")
    print(f"  antinode AP levels: {antinode['ap_levels'] if antinode else 'n/a'}")


if __name__ == "__main__":
    run()
