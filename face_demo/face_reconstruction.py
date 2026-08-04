"""Spectral reconstruction of the FaceBase mean face from the leading modes of the
GEOMETRY operator (cotangent Laplace-Beltrami) and the gap-junction FIELD operator
(graph Laplacian), in frontal and midsagittal-profile views. Generates the two
reconstruction figures used in Paper #4 sec:electric-face:
  data/organ_cascade/face_modes_appear.png   (frontal + profile, 1..12 modes)
  data/organ_cascade/face_profile_20.png      (profile, 12/16/20 modes)

HONEST SCOPE: the modes are derived from the mesh itself (geometry and connectivity),
NOT from the genome. This visualises the operator correspondence and the fact that the
face is low-rank in these modes; 'adding modes' sharpens a static template, it is not a
developmental sequence. Run: python face_demo/face_reconstruction.py  (from cognimed/)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))          # face_demo/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # cognimed/ (for medic.*)
import face_eigenmodes as fe
import mesh_morph as mm
from medic.organ_modes import gapjunction_laplacian, low_modes

OUT = Path("data/organ_cascade")


def ortho(p):
    Q, _ = np.linalg.qr(p)
    return Q


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V0, F, HL = mm.load()
    if F.min() >= 1 and F.max() >= V0.shape[0]:
        F = F - 1
    V, F, keep, s = fe.clean_mesh(V0, F)
    Lc, M, _ = fe.cotangent_laplacian(V, F)
    Lg = gapjunction_laplacian(V, F)
    lam_lb, phi_lb = low_modes(Lc, M, 60, True)
    lam_gj, phi_gj = low_modes(Lg, None, 60, False)
    Qlb, Qgj = ortho(phi_lb), ortho(phi_gj)
    mean = V.mean(0); Vc = V - mean
    mid = np.abs(V[:, 0]) < 0.04

    def recon(Q, k):
        Qk = Q[:, :k]
        return mean + Qk @ (Qk.T @ Vc)

    # ---- figure 1: face appears, frontal + profile, 1..12 ----
    Ks = [1, 3, 5, 8, 10, 12]
    fig, ax = plt.subplots(4, len(Ks), figsize=(2.5 * len(Ks), 11.0))
    rows = [("GEOMETRY\nfrontal", Qlb, "front"), ("FIELD\nfrontal", Qgj, "front"),
            ("GEOMETRY\nprofile", Qlb, "prof"), ("FIELD\nprofile", Qgj, "prof")]
    for r, (lab, Q, view) in enumerate(rows):
        for c, k in enumerate(Ks):
            Vh = recon(Q, k); err = np.linalg.norm(Vh - V) / np.linalg.norm(Vc)
            a = ax[r, c]
            if view == "front":
                a.scatter(Vh[:, 0], Vh[:, 1], c=Vh[:, 2], s=3, cmap="gist_earth", linewidths=0)
            else:
                a.scatter(Vh[mid, 2], Vh[mid, 1], c=Vh[mid, 1], s=3, cmap="viridis", linewidths=0)
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(f"{k} modes", fontsize=11)
            a.text(0.03, 0.03, f"err {err:.2f}", transform=a.transAxes, fontsize=8, color="0.3")
        ax[r, 0].set_ylabel(lab, fontsize=11)
    fig.suptitle("As modes are added the face APPEARS (stop at 12) -- GEOMETRY and gap-junction FIELD modes; "
                 "frontal (colour=depth) and midsagittal profile.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "face_modes_appear.png", dpi=130, bbox_inches="tight")
    print("saved", OUT / "face_modes_appear.png")

    # ---- figure 2: profile at 12/16/20 ----
    fig2, ax2 = plt.subplots(2, 3, figsize=(9, 9))
    for rr, (lab, Q) in enumerate([("GEOMETRY", Qlb), ("FIELD", Qgj)]):
        for cc, k in enumerate([12, 16, 20]):
            Vh = recon(Q, k); err = np.linalg.norm(Vh - V) / np.linalg.norm(Vc)
            a = ax2[rr, cc]
            a.scatter(Vh[mid, 2], Vh[mid, 1], c=Vh[mid, 1], s=4, cmap="viridis", linewidths=0)
            a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
            if rr == 0:
                a.set_title(f"{k} modes", fontsize=12)
            a.text(0.03, 0.03, f"err {err:.2f}", transform=a.transAxes, fontsize=9, color="0.3")
        ax2[rr, 0].set_ylabel(lab, fontsize=12)
    fig2.suptitle("Face PROFILE (midsagittal) at 12/16/20 modes -- geometry (top) vs field (bottom)", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(OUT / "face_profile_20.png", dpi=140, bbox_inches="tight")
    print("saved", OUT / "face_profile_20.png")


if __name__ == "__main__":
    main()
