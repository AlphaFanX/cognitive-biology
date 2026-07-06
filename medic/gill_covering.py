"""
Gill covering: a directed sheet grows over its neighbours and fuses (covering topology).
========================================================================================

field_driven_eye.py drove a MIGRATION; this is the other moving-boundary class Miles
named -- a COVERING. The operculum (fish) or the second pharyngeal arch (human) grows
caudally over the gill arches and fuses to the body wall, enclosing them; incomplete
fusion leaves a persistent opening (a cervical fistula) -- the covering analogue of the
cleft lip. It reuses the two pieces already built: directed field-driven outgrowth (a
KPP growth front advancing under a posterior organizer, as the eye's chemotaxis) and the
Fiedler-eigenvalue fusion criterion of the mechanical layer (\lambda_2>0 / one component
when the sheet reaches and fuses to the body wall; two components while a gap remains).

We grow the opercular sheet caudally over four gill arches, measure how much it covers,
and read the covering's completion off the tissue graph's connectivity: the operculum and
the body wall are two pieces (an open opercular cavity) until the leading edge fuses them
into one. Sweeping the outgrowth gives the covering basin -- sufficient outgrowth covers
and fuses, insufficient outgrowth stalls into a persistent cleft, exactly as the harelip
basin did for fusion by convergence.

HONEST SCOPE: 2D sagittal schematic, a KPP growth front standing for the field-driven
opercular outgrowth, idealised arch/body-wall geometry. It demonstrates the covering
topology -- directed outgrowth over neighbours plus leading-edge fusion -- on the same
machinery as the fold, the cleft, and the migration.

Run: python -m medic.gill_covering
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.mechanical_fusion import tissue_fiedler

OUT = Path("data/organ_cascade")
NX, NY = 180, 64
X = np.linspace(0, 6, NX)[None, :]
Y = np.linspace(0, 1.8, NY)[:, None]
ARCH_X = [1.8, 2.7, 3.6, 4.5]          # four gill arches (to be covered)
WALL_X = 5.4                            # body wall (the fusion target), posterior
BAND = (Y > 0.85) & (Y < 1.7)          # the dorsal band the opercular sheet grows in


def arches():
    A = np.zeros((NY, NX), bool)
    for xa in ARCH_X:
        A |= (np.abs(X - xa) < 0.09) & (Y > 0.12) & (Y < 1.05)
    return A


def body_wall():
    return (X > WALL_X) & np.ones((NY, NX), bool)


def lap(Z):
    Zp = np.pad(Z, 1, mode="edge")
    return (Zp[2:, 1:-1] + Zp[:-2, 1:-1] + Zp[1:-1, 2:] + Zp[1:-1, :-2] - 4 * Zp[1:-1, 1:-1])


def grow_operculum(outgrowth, steps=1700, D=1.0, dt=0.10):
    """KPP growth front: the opercular sheet grows caudally (+x) in the dorsal band,
    driven under a posterior organizer. outgrowth scales the front speed 2*sqrt(D*r)."""
    band = BAND & np.ones((NY, NX), bool)
    phi = np.zeros((NY, NX))
    phi[band & (X < 0.9)] = 1.0                        # anterior-dorsal hinge (the operculum base)
    wall = body_wall()
    r = 0.35 * outgrowth
    frames = []
    for s in range(steps):
        react = r * phi * (1 - phi)                    # KPP proliferation front
        phi = phi + dt * (D * lap(phi) + react) * band
        np.clip(phi, 0, 1, out=phi)
        phi[~band] = 0.0
        if s in (0, steps // 3, 2 * steps // 3, steps - 1):
            frames.append(phi.copy())
    return phi, frames, wall


def measure(phi, wall):
    front = X.ravel()[np.where((phi > 0.5).any(0))[0]].max() if (phi > 0.5).any() else 0.0
    coverage = np.mean([xa < front for xa in ARCH_X])              # fraction of arches covered
    mask = (phi > 0.5) | wall                                       # operculum + body wall
    lam2, ncomp = tissue_fiedler(mask)
    return float(front), float(coverage), int(ncomp), float(lam2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("gill covering: the opercular sheet grows caudally over the gill arches and fuses "
          "to the body wall\n")

    # main run: sufficient outgrowth -> covers + fuses
    phi, frames, wall = grow_operculum(1.0)
    front, cov, ncomp, lam2 = measure(phi, wall)
    print(f"sufficient outgrowth: front reaches x={front:.1f}, coverage {cov:.0%}, "
          f"components={ncomp} ({'FUSED -> gills covered' if ncomp == 1 else 'gap -> open cavity'})")

    # failure mode: insufficient outgrowth -> stalls into a persistent cleft (cervical fistula)
    phi_lo, _, _ = grow_operculum(0.28)
    front_lo, cov_lo, ncomp_lo, _ = measure(phi_lo, wall)
    print(f"insufficient outgrowth: front x={front_lo:.1f}, coverage {cov_lo:.0%}, "
          f"components={ncomp_lo} ({'FUSED' if ncomp_lo == 1 else 'PERSISTENT GAP = cervical fistula'})")

    # covering basin: coverage + fusion vs outgrowth
    ogs = np.linspace(0.15, 1.3, 12)
    cov_curve, fused_curve = [], []
    for og in ogs:
        p, _, w = grow_operculum(og)
        _, c, nc, _ = measure(p, w)
        cov_curve.append(c); fused_curve.append(1 if nc == 1 else 0)
    print(f"\ncovering basin: fused (gills covered) for outgrowth >= "
          f"{ogs[np.argmax(np.array(fused_curve) > 0)] if any(fused_curve) else float('nan'):.2f}")

    _figure(frames, wall, ogs, cov_curve, fused_curve, cov, ncomp, phi_lo)
    json.dump(dict(coverage=cov, components=ncomp, front=front,
                   coverage_lo=cov_lo, components_lo=ncomp_lo,
                   outgrowth=list(map(float, ogs)), coverage_curve=cov_curve,
                   fused_curve=fused_curve),
              open(OUT / "gill_covering.json", "w"), indent=2)
    print("\nsaved", OUT / "gill_covering.json")
    return ncomp == 1 and cov > 0.9 and ncomp_lo == 2


def _figure(frames, wall, ogs, cov_curve, fused_curve, cov, ncomp, phi_lo):
    A = arches()
    fig = plt.figure(figsize=(18, 6.2))
    gs = fig.add_gridspec(2, 4)
    ext = [0, 6, 0, 1.8]

    def scene(ax, phi, title):
        ax.imshow(np.where(wall, 0.35, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        ax.imshow(np.where(A, 0.6, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        ax.imshow(np.where(phi > 0.1, phi, np.nan), origin="lower", extent=ext, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])

    labels = ["hinge (t=0)", "growing over arch 1--2", "over arch 3--4", "reached body wall -> fused"]
    for i, (fr, lab) in enumerate(zip(frames, labels)):
        scene(fig.add_subplot(gs[0, i]), fr, f"({chr(97+i)}) {lab}")

    scene(fig.add_subplot(gs[1, 0]), frames[-1], f"(e) covered + fused\ncoverage {cov:.0%}, components {ncomp}")
    scene(fig.add_subplot(gs[1, 1]), phi_lo, "(f) insufficient outgrowth\npersistent gap = cervical fistula")
    ax = fig.add_subplot(gs[1, 2]); ax.plot(ogs, cov_curve, "o-", ms=3, color="tab:orange")
    ax.set_xlabel("outgrowth"); ax.set_ylabel("gill coverage"); ax.set_ylim(-0.05, 1.05)
    ax.set_title("(g) coverage vs outgrowth", fontsize=9)
    ax = fig.add_subplot(gs[1, 3]); ax.plot(ogs, fused_curve, "s-", ms=4, color="tab:green")
    ax.set_xlabel("outgrowth"); ax.set_ylabel("fused (1) / open (0)"); ax.set_ylim(-0.1, 1.1)
    ax.set_title("(h) the covering basin\n(fused = covered vs persistent fistula)", fontsize=9)
    fig.suptitle("Gill covering: the opercular sheet grows caudally over the gill arches (directed field-driven "
                 "outgrowth) and fuses to the body wall (Fiedler $\\lambda_2$ / one component) -- covering topology on "
                 "the same machinery as the fold, the cleft and the migration", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "gill_covering.png", dpi=130, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "gill_covering.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'COVERED (sheet grows over the gills and fuses; cleft on low outgrowth)' if ok else 'CHECK'}")
