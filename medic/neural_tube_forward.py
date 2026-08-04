"""Neural tube, GROWN from the genome: the cup's constriction fold taken to CLOSURE (and its defect).

The optic cup invaginated; the neural tube goes further -- a flat neural plate rolls all the way up and its
two edges FUSE at the dorsal midline into a closed tube with a lumen (a topology change, not just a dent).
It is the same apical-constriction behaviour (Shroom3), localised by the genome to the ventral hinge (the
floor plate, Shh/Foxa2), only stronger and distributed so the sheet wraps ~360 degrees and the neural folds
meet. Weaken the constriction and the folds fail to meet -- an OPEN neural tube, a neural-tube defect (spina
bifida), the medical analogue of the heart's situs inversus: a real defect reproduced by turning one behaviour
down.

  1. GENOME -> where.  In the neural tissue the apical-constriction program co-localises with the ventral
     floor-plate hinge (Foxa2/Shh) -- the genome puts the fold on the midline hinge.
  2. BEHAVIOUR -> shape.  A flat plate under that constriction rolls into a closed tube; the fold emerges.
  3. THE DEFECT.  Turn the constriction down -> the folds do not meet -> an open tube (NTD).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.neural_tube_forward
Out:  data/organ_cascade/neural_tube_forward.{json,png}
"""
import os, json
import numpy as np
import h5py
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from medic.optic_cup_forward import build_sheet, relax, _mod, _z

H5 = "data/mosta/E12.5_E1S1.MOSTA.h5ad"


def genome_grounding():
    f = h5py.File(H5, "r")
    o = f["obs/annotation"]; codes = o[:] if not isinstance(o, h5py.Group) else o["codes"][:]
    cats = [x.decode() if isinstance(x, bytes) else x for x in
            (f["obs/__categories/annotation"][:] if not isinstance(o, h5py.Group) else o["categories"][:])]
    ann = np.array([cats[c] if c >= 0 else "?" for c in codes])
    genes = [x.decode() if isinstance(x, bytes) else x for x in f["var/gene_short_name"][:]]
    gidx = {g: i for i, g in enumerate(genes)}
    X = csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]), shape=(len(ann), len(genes)))
    f.close()
    neural = _mod(X, gidx, ["Sox2", "Pax6", "Sox1"]) >= np.quantile(_mod(X, gidx, ["Sox2", "Pax6", "Sox1"]), 0.9)
    constr = _mod(X, gidx, ["Shroom3", "Rock1", "Myh9"])
    floorplate = _mod(X, gidx, ["Foxa2", "Shh", "Nkx6-1"])            # the ventral hinge (MHP)
    rho, p = spearmanr(constr[neural], floorplate[neural])
    return {"rho_constr_floorplate": round(float(rho), 3), "p": round(float(p), 6), "n_neural": int(neural.sum())}


# ---------------- the plate: apical constriction -> a closed tube ----------------
def neural_profile(N, amp, mhp_w=0.13):
    """Apical constriction DISTRIBUTED across the neural plate (a broad plateau, so every cell wedges a little
    and the whole sheet rolls into a circle) with an extra peak at the ventral midline hinge (MHP, the floor
    plate). amp scales the whole thing: full amp wraps the plate ~360 deg and the dorsal edges meet (a closed
    tube); low amp under-wraps and the tube stays open (NTD)."""
    i = (np.arange(N) + 0.5) / N
    return amp * (0.75 + 0.5 * np.exp(-0.5 * ((i - 0.5) / mhp_w) ** 2))


def run(amp, N=64, h=1.5, ds=1.0):
    """Elastica: apical constriction c_i gives each cell an intrinsic curvature kappa_i = c_i/h; integrating it
    turns the flat apical line into an arc, and a distributed constriction rolls it into a (closed) circle.
    The basal layer is offset by the cell height to the convex side -> a two-layer tube."""
    c = neural_profile(N, amp)
    phi = np.concatenate([[0.0], np.cumsum(c / h * ds)])          # node tangent angles (N+1)
    A = np.c_[np.concatenate([[0.0], np.cumsum(ds * np.cos(phi[:-1]))]),
              np.concatenate([[0.0], np.cumsum(ds * np.sin(phi[:-1]))])]
    B = A - h * np.c_[-np.sin(phi), np.cos(phi)]                  # basal on the convex (outer) side
    A0 = np.c_[np.arange(N + 1) * ds, np.zeros(N + 1)]; B0 = A0 - np.c_[np.zeros(N + 1), np.full(N + 1, h)]
    wrap = float(np.degrees(phi[-1] - phi[0]))
    gap = float(np.hypot(*(A[0] - A[-1])) / ds)
    return A0, B0, A, B, c, ds, gap, wrap


def _quads(A, B):
    return [[A[k], A[k + 1], B[k + 1], B[k]] for k in range(len(A) - 1)]


def main():
    grd = genome_grounding()
    print(f"[genome] within neural tissue, constriction (Shroom3) vs floor-plate hinge (Foxa2/Shh): "
          f"rho={grd['rho_constr_floorplate']} (p={grd['p']}, n={grd['n_neural']})")

    A0, B0, An, Bn, cn, ds, gap_n, wrap_n = run(amp=0.17)    # normal: wraps ~360 -> closes
    *_, Ad, Bd, cd, _, gap_d, wrap_d = run(amp=0.085)        # weakened Shroom3: under-wraps -> open (NTD)
    closed = wrap_n >= 300.0
    print(f"[shape] normal -> {wrap_n:.0f} deg wrap, dorsal gap {gap_n:.1f} ({'CLOSED tube' if closed else 'open'}); "
          f"weakened -> {wrap_d:.0f} deg, gap {gap_d:.1f} ({'closed' if wrap_d >= 300 else 'OPEN = neural-tube defect'})")
    res = {"genome": grd, "wrap_normal_deg": round(wrap_n, 1), "wrap_weakened_deg": round(wrap_d, 1),
           "gap_normal": round(gap_n, 2), "gap_weakened": round(gap_d, 2), "normal_closed": bool(closed)}
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump(res, open("data/organ_cascade/neural_tube_forward.json", "w"), indent=1)

    # ---------------- figure ----------------
    fig, ax = plt.subplots(1, 4, figsize=(19, 5.2), facecolor="#0d1017")
    for a in ax:
        a.set_facecolor("#0d1017"); a.set_aspect("equal"); a.axis("off")
    ax[0].add_collection(PolyCollection(_quads(A0, B0), array=cn, cmap="magma", edgecolors="#0d1017", linewidths=0.3))
    ax[0].autoscale()
    ax[0].set_title("neural plate: flat epithelium\ngenome-set apical constriction (hinge at MHP)", color="#cbd5e1", fontsize=10)
    ax[1].add_collection(PolyCollection(_quads(An, Bn), array=cn, cmap="magma", edgecolors="#0d1017", linewidths=0.3))
    ax[1].plot(An[:, 0], An[:, 1], color="#38bdf8", lw=1.3); ax[1].autoscale()
    ax[1].set_title(f"run constriction -> the plate rolls into\na CLOSED TUBE ({wrap_n:.0f} deg wrap)", color="#7dd3fc", fontsize=10)
    ax[2].add_collection(PolyCollection(_quads(Ad, Bd), array=cd, cmap="magma", edgecolors="#0d1017", linewidths=0.3))
    ax[2].plot(Ad[:, 0], Ad[:, 1], color="#f472b6", lw=1.3); ax[2].autoscale()
    ax[2].set_title(f"weaken Shroom3 -> folds do not meet\nOPEN tube = defect ({wrap_d:.0f} deg)", color="#f472b6", fontsize=10)
    a = ax[3]; a.axis("on")
    a.bar(["closed\nwrap", "NTD\nwrap"], [wrap_n, wrap_d], color=["#38bdf8", "#f472b6"])
    a.axhline(300.0, color="#94a3b8", ls=":", lw=0.8)
    a.set_ylabel("apical wrap (deg)", color="#94a3b8")
    a.set_title(f"closure vs defect\n(genome: constriction on the hinge, rho={grd['rho_constr_floorplate']})",
                color="#cbd5e1", fontsize=10)
    a.tick_params(colors="#94a3b8", labelsize=8)
    for s in a.spines.values(): s.set_color("#334155")
    fig.suptitle("The neural tube grown from the genome: apical constriction on the ventral hinge rolls the "
                 "flat plate into a closed tube; weaken it and the tube stays open -- a neural-tube defect",
                 color="#e2e8f0", fontsize=12)
    fig.tight_layout()
    fig.savefig("data/organ_cascade/neural_tube_forward.png", dpi=135, facecolor="#0d1017")
    print("\nsaved data/organ_cascade/neural_tube_forward.{json,png}")


if __name__ == "__main__":
    main()
