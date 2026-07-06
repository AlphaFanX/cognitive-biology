"""
The failure atlas: each moving-boundary movement, and the malformation it makes when it fails.
==============================================================================================

The four moving-boundary events (medic.mechanical_fusion fold, prominence fusion;
medic.field_driven_eye migration; medic.gill_covering covering) are ONE operation --
a directed field drives a boundary, and a connectivity criterion reads whether it
completed. That framing makes a prediction the normal runs do not: each event has a
NAMED failure, and the same generator produces it by turning its drive down. The
malformation is not a separate model -- it is the same movement, under-run.

  event (normal)            under-run parameter          malformation (failure)
  -----------------------   --------------------------   ---------------------------------
  neural fold closes        apical-constriction drive    neural-tube defect (open plate)
  prominences fuse          adhesion at the seam         cleft lip (unfused prominences)
  eyes medialize            eye motility (chemotaxis)    hypertelorism (eyes stay lateral)
  operculum covers gills    opercular outgrowth          cervical fistula (persistent gap)

Each column is the SAME function called twice: a sufficient drive (top, normal) and an
insufficient drive (bottom, the named malformation). The connectivity discriminator --
gap closed / one component (fused) vs open / two components (cleft) -- classifies them
automatically, so the failure is diagnosed by the model, not asserted. This is the
clinical face of the unification: the compiler that grows the normal form is the compiler
that, mis-parameterised, grows the recognised dysmorphology.

HONEST SCOPE: 2D schematics on idealised geometry, one drive parameter per event. It shows
that each moving-boundary movement and its named malformation are two settings of one
generator, read by one connectivity criterion -- not a quantitative dysmorphology model.

Run: python -m medic.morphogenesis_failures
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.mechanical_fusion import constriction_profile, fold_from_curvature, prominence_fusion
from medic.field_driven_eye import migrate, PHI0, T
from medic.gill_covering import grow_operculum, measure, arches, body_wall, X

OUT = Path("data/organ_cascade")


# ---- the four events, each run normal (sufficient drive) and failed (insufficient) ----

def fold_pair():
    N = 121
    comp = constriction_profile(N)
    healthy = fold_from_curvature(1.0, N=N, competence=comp)   # closes into a tube
    defect = fold_from_curvature(0.45, N=N, competence=comp)   # under-constricts -> stays open
    return healthy, defect


def fusion_pair():
    healthy = prominence_fusion(outgrowth=1.0, adhesion=1.0)   # prominences meet AND adhere
    defect = prominence_fusion(outgrowth=1.0, adhesion=0.0)    # meet but do not adhere -> cleft
    return healthy, defect


def migration_pair():
    lam = 1.0
    with torch.no_grad():
        normal = migrate(torch.tensor(0.13), torch.tensor(lam)).numpy()   # medializes
        failed = migrate(torch.tensor(0.015), torch.tensor(lam)).numpy()  # too little motility
    return normal, failed


def covering_pair():
    phi_hi, _, wall = grow_operculum(1.0)      # covers + fuses
    phi_lo, _, _ = grow_operculum(0.28)        # stalls -> persistent gap
    return (phi_hi, wall), (phi_lo, wall)


# ------------------------------- rendering -------------------------------

def _draw_fold(ax, f, title, color):
    ax.plot(f["A"][:, 0], f["A"][:, 1], color=color, lw=2.3)
    ax.plot(f["B"][:, 0], f["B"][:, 1], color=color, lw=1.2, alpha=0.5)
    ax.scatter(f["A"][[0, -1], 0], f["A"][[0, -1], 1], c="k", s=14, zorder=3)  # the two lip edges
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _draw_mask(ax, res, title, extent):
    m = res["phi"]
    ax.imshow(m, origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.axvline(0, color="w", ls=":", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


def _draw_eyes(ax, traj, title):
    th = np.linspace(0, 2 * np.pi, 200)
    R = 1.0
    ax.plot(R * np.sin(th), R * np.cos(th), "0.6", lw=1)
    ax.scatter([0], [R], c="tab:green", s=70, marker="*", zorder=3)
    phi = traj[-1]
    for sgn in (+1, -1):
        ax.scatter(sgn * R * np.sin(phi), R * np.cos(phi), c="tab:red", s=90, zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def _draw_cover(ax, phi, wall, title):
    A = arches()
    ext = [0, 6, 0, 1.8]
    ax.imshow(np.where(wall, 0.35, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.imshow(np.where(A, 0.6, np.nan), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.imshow(np.where(phi > 0.1, phi, np.nan), origin="lower", extent=ext, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("the failure atlas: each moving-boundary movement and its named malformation\n")

    (f_ok, f_bad) = fold_pair()
    (u_ok, u_bad) = fusion_pair()
    (m_ok, m_bad) = migration_pair()
    (c_ok, c_bad) = covering_pair()

    # classify with each event's own discriminator
    f_ok_closed, f_bad_closed = f_ok["closed"], f_bad["closed"]
    u_ok_nc, u_bad_nc = u_ok["ncomp"], u_bad["ncomp"]
    m_ok_end, m_bad_end = np.degrees(m_ok[-1]), np.degrees(m_bad[-1])
    (phi_ok, wall) = c_ok; (phi_bad, _) = c_bad
    _, c_ok_cov, c_ok_nc, _ = measure(phi_ok, wall)
    _, c_bad_cov, c_bad_nc, _ = measure(phi_bad, wall)

    print(f"  FOLD      normal: neural plate {'CLOSED into a tube' if f_ok_closed else 'open'} "
          f"(gap {f_ok['gap']:.2f}) | failure: plate {'CLOSED' if f_bad_closed else 'stays OPEN'} "
          f"(gap {f_bad['gap']:.2f}) = neural-tube defect")
    print(f"  FUSION    normal: prominences components={u_ok_nc} "
          f"({'FUSED lip' if u_ok_nc == 1 else 'cleft'}) | failure: components={u_bad_nc} "
          f"({'fused' if u_bad_nc == 1 else 'CLEFT LIP'})")
    print(f"  MIGRATION normal: eyes {np.degrees(PHI0):.0f}deg -> {m_ok_end:.0f}deg (medialized) | "
          f"failure: eyes -> {m_bad_end:.0f}deg (stay lateral) = hypertelorism")
    print(f"  COVERING  normal: coverage {c_ok_cov:.0%}, components={c_ok_nc} "
          f"({'covered+FUSED' if c_ok_nc == 1 else 'gap'}) | failure: coverage {c_bad_cov:.0%}, "
          f"components={c_bad_nc} ({'fused' if c_bad_nc == 1 else 'PERSISTENT GAP = cervical fistula'})")

    _figure(f_ok, f_bad, u_ok, u_bad, m_ok, m_bad, c_ok, c_bad,
            f_ok_closed, f_bad_closed, u_ok_nc, u_bad_nc, m_ok_end, m_bad_end,
            c_ok_nc, c_bad_nc, c_ok_cov, c_bad_cov)

    result = dict(
        fold=dict(normal_closed=bool(f_ok_closed), failure_closed=bool(f_bad_closed),
                  malformation="neural-tube defect"),
        fusion=dict(normal_components=int(u_ok_nc), failure_components=int(u_bad_nc),
                    malformation="cleft lip"),
        migration=dict(normal_end_deg=float(m_ok_end), failure_end_deg=float(m_bad_end),
                       malformation="hypertelorism"),
        covering=dict(normal_components=int(c_ok_nc), failure_components=int(c_bad_nc),
                      normal_coverage=float(c_ok_cov), failure_coverage=float(c_bad_cov),
                      malformation="cervical fistula"),
    )
    json.dump(result, open(OUT / "morphogenesis_failures.json", "w"), indent=2)
    print("\nsaved", OUT / "morphogenesis_failures.json")
    # every event: normal completes, failure does not, by its own discriminator
    ok = (f_ok_closed and not f_bad_closed and u_ok_nc == 1 and u_bad_nc == 2
          and m_ok_end < 50 and m_bad_end > 65 and c_ok_nc == 1 and c_bad_nc == 2)
    return ok


def _figure(f_ok, f_bad, u_ok, u_bad, m_ok, m_bad, c_ok, c_bad,
            f_ok_closed, f_bad_closed, u_ok_nc, u_bad_nc, m_ok_end, m_bad_end,
            c_ok_nc, c_bad_nc, c_ok_cov, c_bad_cov):
    fig = plt.figure(figsize=(17, 8.4))
    gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.22)
    uext = [u_ok["xs"].min(), u_ok["xs"].max(), u_ok["ys"].min(), u_ok["ys"].max()]

    # row 0 = NORMAL
    _draw_fold(fig.add_subplot(gs[0, 0]), f_ok, "FOLD -- neural plate closes\n(sufficient apical constriction)", "#1a6fb0")
    _draw_mask(fig.add_subplot(gs[0, 1]), u_ok, "FUSION -- prominences fuse\n(seam adheres, 1 component)", uext)
    _draw_eyes(fig.add_subplot(gs[0, 2]), m_ok, f"MIGRATION -- eyes medialize\n(sufficient motility -> {m_ok_end:.0f}$^\\circ$)")
    _draw_cover(fig.add_subplot(gs[0, 3]), c_ok[0], c_ok[1], f"COVERING -- operculum covers\n(coverage {c_ok_cov:.0%}, fused)")

    # row 1 = FAILURE (named)
    _draw_fold(fig.add_subplot(gs[1, 0]), f_bad, "NEURAL-TUBE DEFECT\n(plate stays open: spina bifida / anencephaly)", "#b02318")
    _draw_mask(fig.add_subplot(gs[1, 1]), u_bad, "CLEFT LIP\n(prominences meet but do not adhere: 2 components)", uext)
    _draw_eyes(fig.add_subplot(gs[1, 2]), m_bad, f"HYPERTELORISM\n(too little motility: eyes stay lateral {m_bad_end:.0f}$^\\circ$)")
    _draw_cover(fig.add_subplot(gs[1, 3]), c_bad[0], c_bad[1], f"CERVICAL FISTULA\n(insufficient outgrowth: coverage {c_bad_cov:.0%}, gap persists)")

    fig.suptitle("The failure atlas: each moving-boundary movement (top, normal) and the named malformation it makes when "
                 "under-run (bottom). Each column is ONE generator called at a sufficient vs an insufficient drive; the "
                 "connectivity discriminator (gap closed / one component vs open / two) diagnoses the failure automatically.",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "morphogenesis_failures.png", dpi=125, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "morphogenesis_failures.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'FAILURE ATLAS (each movement + its named malformation are two settings of one generator)' if ok else 'CHECK'}")
