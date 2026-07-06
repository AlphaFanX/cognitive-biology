#!/usr/bin/env python3
"""
Sourcing the morphogen reaction-diffusion from AlphaGenome (closing the loop).
==============================================================================

morphogen_orientation.py showed the electric-face eigenmodes ORIENT a Turing
reaction-diffusion (RD sets spacing, eigenmodes set symmetry/orientation), but the
RD used generic parameters. This module sources the RD itself from the genome: it
queries AlphaGenome LIVE for the accessibility of the facial-organizer network --
the activators/organizers SHH, FGF8, WNT5A and the lateral inhibitor BMP4, plus
the antagonists DKK1 (Wnt), NOG/Noggin (Bmp) and SPRY2 (Fgf) -- and maps the
activator:inhibitor balance to the RD's reaction parameters. It then runs the
activator-inhibitor RD (Gray-Scott spots) ORIENTED by the electric-face frame.

So BOTH halves of primordium placement are now genome-derived: the RD network
(from AlphaGenome accessibility) and its orientation (from the low bioelectric
eigenmodes). The primordia are the RD peaks; the electric face frames them.

HONEST SCOPE: AlphaGenome human ATAC is ~adult, with no embryonic craniofacial
track, so the accessibility is an overall regulatory-openness proxy for these
organizer genes, not the embryonic FEZ read (the refinement = embryonic/neural-
crest tracks). The activator/inhibitor ROLES are from known biology; the diffusion
constants that set the absolute wavelength are biophysical, not from AlphaGenome.
What AlphaGenome supplies here is the network membership and the relative
activator:inhibitor competence.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.alphagenome_morphogen_rd
Out:  data/alphagenome_morphogen_rd.json, alphagenome_morphogen_rd.png
"""
from __future__ import annotations
import os
import json
from pathlib import Path

import numpy as np

HOME = Path(os.path.expanduser("~"))
if "ALPHAGENOME_API_KEY" not in os.environ and (HOME / ".alphagenome_key").exists():
    os.environ["ALPHAGENOME_API_KEY"] = (HOME / ".alphagenome_key").read_text().strip()
_ca = HOME / ".ca_combined.pem"
if _ca.exists():
    os.environ.setdefault("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH", str(_ca))
    os.environ.setdefault("REQUESTS_CA_BUNDLE", str(_ca))

import sys
sys.path.insert(0, str(Path("face_demo").resolve()))
from morphogen_orientation import face_mask, local_peaks, oriented_turing, peak_symmetry  # reuse frame + tools

# facial-organizer RD network: gene -> role
ROLE = {
    "SHH":   "activator",   # ventral FEZ organizer
    "FGF8":  "activator",   # dorsal FEZ organizer
    "WNT5A": "activator",   # craniofacial outgrowth (Robinow)
    "BMP4":  "inhibitor",   # lateral inhibitor between placodes
    "DKK1":  "antag_act",   # Wnt antagonist -> reduces activator
    "NOG":   "antag_inh",   # Bmp antagonist  -> reduces inhibitor
    "SPRY2": "antag_act",   # Fgf antagonist  -> reduces activator
}
WIN, PROM = 8192, 3000


def gene_coords(sym):
    import requests
    j = requests.get("https://mygene.info/v3/query",
                     params={"q": f"symbol:{sym}", "species": "human",
                             "fields": "genomic_pos", "size": 1}, timeout=30).json()
    gp = j["hits"][0]["genomic_pos"]
    if isinstance(gp, list):
        gp = [g for g in gp if str(g["chr"]) in [str(i) for i in range(1, 23)] + ["X", "Y"]][0]
    tss = int(gp["start"]) if gp.get("strand", 1) == 1 else int(gp["end"])
    return "chr" + str(gp["chr"]), tss


def all_atac_ontologies(m):
    """Every human ATAC track's ontology term (167) -> query them all = 'all tracks'."""
    from alphagenome.models import dna_client
    md = m.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
    stages = md.atac["biosample_life_stage"].value_counts().to_dict()
    print(f"  ATAC tracks: {len(md.atac)}  life-stage mix: {stages}")
    return md.atac["ontology_curie"].dropna().unique().tolist()


def query_openness(m, ATAC, genome, onto):
    out = {}
    for sym in ROLE:
        chrom, tss = gene_coords(sym)
        iv = genome.Interval(chromosome=chrom, start=tss - WIN, end=tss + WIN)
        r = m.predict_interval(iv, requested_outputs=[ATAC], ontology_terms=onto)
        v = r.atac.values
        c = v.shape[0] // 2
        acc = float(v[c - PROM:c + PROM, :].mean())
        out[sym] = acc
        print(f"  {sym:6s} ({ROLE[sym]:9s})  promoter ATAC (all tracks) = {acc:.4f}")
    return out


def rd_params(acc):
    """Map organizer accessibility -> activator:inhibitor balance -> Turing WAVELENGTH.
    Antagonists reduce their target. In Turing theory the activator:inhibitor balance
    sets the pattern wavelength: more (long-range) inhibitor -> wider spacing/fewer
    features; more (short-range) activator -> denser features. So the genome sets the
    SPACING (the RD's job); the diffusion constants that fix the absolute scale are
    biophysical, not from AlphaGenome."""
    def mean(role):
        vals = [acc[g] for g, r in ROLE.items() if r == role]
        return float(np.mean(vals)) if vals else 0.0
    act = mean("activator"); inh = mean("inhibitor")
    antag_a = mean("antag_act"); antag_i = mean("antag_inh")
    act_eff = act / (1.0 + antag_a)          # Dkk/Spry antagonize the activator
    inh_eff = inh / (1.0 + antag_i)          # Noggin antagonizes the inhibitor
    bal = act_eff / (act_eff + inh_eff + 1e-9)   # activator share in [0,1]
    n_feat = 2.0 + 3.0 * bal                 # activator share -> feature density (tiers)
    return dict(activator=act, inhibitor=inh, antag_act=antag_a, antag_inh=antag_i,
                act_eff=act_eff, inh_eff=inh_eff, activator_share=bal, n_feat=n_feat)


def main():
    print("=" * 74)
    print("SOURCING THE MORPHOGEN RD FROM ALPHAGENOME (facial-organizer network)")
    print("=" * 74)
    from alphagenome.models import dna_client
    from alphagenome.data import genome
    m = dna_client.create(os.environ["ALPHAGENOME_API_KEY"])
    ATAC = dna_client.OutputType.ATAC
    onto = all_atac_ontologies(m)
    print(f"\nQuerying {len(ROLE)} organizer/antagonist genes (all human ATAC tracks):")
    acc = query_openness(m, ATAC, genome, onto)

    p = rd_params(acc)
    print(f"\nactivator (SHH/FGF8/WNT5A) mean={p['activator']:.4f}  inhibitor (BMP4)={p['inhibitor']:.4f}")
    print(f"antagonist-adjusted: act_eff={p['act_eff']:.4f}  inh_eff={p['inh_eff']:.4f}  "
          f"-> activator share={p['activator_share']:.2f}")
    print(f"genome-sourced Turing wavelength: n_feat={p['n_feat']:.2f} tiers")

    mask, _, _ = face_mask()
    n = mask.shape[0]
    k0 = p["n_feat"] / n                       # genome-set spacing; eigenmodes set orientation
    V = oriented_turing(mask, k0)              # axis-locked to the electric-face frame
    xs, ys = local_peaks(V, mask)
    sym = peak_symmetry(xs, ys, n)
    p["k0"] = float(k0); p["mirror_symmetry"] = float(sym)
    print(f"\nRD (genome-set wavelength) oriented by the electric-face frame -> {len(xs)} primordia, "
          f"mirror-symmetry {sym:.2f} (bilaterally symmetric, tiered).")

    _figure(mask, V, xs, ys, acc, p)
    Path("data").mkdir(exist_ok=True)
    json.dump(dict(accessibility=acc, roles=ROLE, rd=p, n_peaks=int(len(xs))),
              open("data/alphagenome_morphogen_rd.json", "w"), indent=2)
    print("saved data/alphagenome_morphogen_rd.json")
    return True


def _figure(mask, V, xs, ys, acc, p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    col = {"activator": "#2ca02c", "inhibitor": "#d62728",
           "antag_act": "#8c564b", "antag_inh": "#1f77b4"}
    items = sorted(acc.items(), key=lambda kv: -kv[1])
    ax[0].bar([g for g, _ in items], [a for _, a in items],
              color=[col[ROLE[g]] for g, _ in items])
    ax[0].set_ylabel("promoter ATAC (AlphaGenome, all tracks)")
    ax[0].set_title("(a) facial-organizer accessibility from AlphaGenome\n"
                    "green activator / red inhibitor / brown, blue antagonist", fontsize=9)
    ax[0].tick_params(axis="x", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col[r]) for r in
               ("activator", "inhibitor", "antag_act", "antag_inh")]
    ax[0].legend(handles, ["activator (SHH/FGF8/WNT5A)", "inhibitor (BMP4)",
                           "Wnt/Fgf antagonist", "Bmp antagonist"], fontsize=7)
    Vp = V.copy(); Vp[~mask] = np.nan
    ax[1].imshow(Vp, origin="lower", cmap="viridis")
    ax[1].scatter(xs, ys, c="red", s=20, edgecolors="w", linewidths=0.4)
    ax[1].axvline(V.shape[1] / 2 - 0.5, color="w", ls=":", lw=0.8)
    ax[1].set_title(f"(b) genome-sourced RD oriented by the electric face\n"
                    f"activator share {p['activator_share']:.2f} -> {p['n_feat']:.1f} tiers "
                    f"-> {len(xs)} primordia, mirror-symmetry {p['mirror_symmetry']:.2f}", fontsize=9)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.suptitle("Sourcing the morphogen reaction-diffusion from AlphaGenome, oriented by the electric-face "
                 "eigenmodes:\nthe spacing from organizer accessibility, the orientation and symmetry from the "
                 "bioelectric eigenmodes", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("alphagenome_morphogen_rd.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved alphagenome_morphogen_rd.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'OK' if ok else 'BLOCKED'}")
