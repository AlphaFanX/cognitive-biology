"""
Field-driven morphogenesis: the eyes migrate because the field drives them.
============================================================================

growing_domain.py fit a deformation field to a shape trajectory kinematically -- the
shape moved, but nothing caused it. This closes the loop: the FIELD drives the SHAPE.
The concrete event is eye migration (Miles): in the vertebrate head the eyes begin
lateral and rotate frontal as the head grows -- the medialization every human face
undergoes (the flatfish is the extreme). Here the cause is explicit: a frontal
organizer (Shh/Fgf8 at the frontonasal midline, or equally a bioelectric competence
attractor) emits a morphogen; the eye primordium performs chemotaxis UP that gradient,
so it migrates frontally; and the head grows underneath it. No trajectory is imposed --
the eye path EMERGES from the field on the growing domain.

It is also trainable: the migration is a differentiable ODE in the coupling parameters
(the motility and the morphogen length scale), so we fit them to a target medialization
trajectory -- the field->shape coupling trained against a morphogenetic MOVEMENT, the
moving-boundary analogue of the atlas-field training.

HONEST SCOPE: 2D, one morphogen gradient standing for the organizer field, a schematic
target medialization (a real target would come from staged Carnegie/imaging eye
positions); it demonstrates that a field can causally drive, and be trained to drive, a
morphogenetic movement on a growing domain.

Run: python -m medic.field_driven_eye
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/organ_cascade")
torch.manual_seed(0)
T = 40                     # developmental steps
PHI0 = np.radians(85.0)    # eyes start lateral (85 deg from the frontal midline)
R0, GROWTH = 1.0, 0.8      # head radius grows R0 -> R0*(1+GROWTH)


def target_traj():
    """Target medialization: eye angle 85 deg -> 35 deg over development (smooth)."""
    t = np.linspace(0, 1, T + 1)
    return np.radians(85.0 - 50.0 * (1 - np.cos(np.pi * t)) / 2).astype(np.float32)


def migrate(motility, lam):
    """Differentiable ODE: the eye chemotaxes up the morphogen gradient toward the
    frontal organizer (angle 0) while the head grows. Returns the eye-angle trajectory."""
    phi = torch.tensor(PHI0)
    traj = [phi]
    for s in range(T):
        R = R0 * (1.0 + GROWTH * s / T)               # head grows
        arc = R * phi                                 # arc distance to the frontal source
        grad = (1.0 / lam) * torch.exp(-arc / lam)    # morphogen gradient magnitude
        dphi = -motility * grad / R                   # angular migration up-gradient
        phi = torch.clamp(phi + dphi, min=0.0)
        traj.append(phi)
    return torch.stack(traj)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tgt = torch.tensor(target_traj())
    print("field-driven eye migration: the morphogen gradient drives the eye frontally "
          "while the head grows\n")

    # forward (untrained) with plausible constants, then TRAIN the coupling to the target
    p_mot = torch.tensor(0.0, requires_grad=True)     # motility via softplus
    p_lam = torch.tensor(0.0, requires_grad=True)     # length scale via softplus
    opt = torch.optim.Adam([p_mot, p_lam], lr=0.05)
    hist = []
    for ep in range(600):
        opt.zero_grad()
        mot = torch.nn.functional.softplus(p_mot) * 0.5 + 0.01
        lam = torch.nn.functional.softplus(p_lam) * 1.5 + 0.3
        traj = migrate(mot, lam)
        loss = ((traj - tgt) ** 2).mean()
        loss.backward(); opt.step()
        if ep % 30 == 0 or ep == 599:
            hist.append((ep, float(loss.detach())))
    with torch.no_grad():
        mot = float(torch.nn.functional.softplus(p_mot) * 0.5 + 0.01)
        lam = float(torch.nn.functional.softplus(p_lam) * 1.5 + 0.3)
        traj = migrate(torch.tensor(mot), torch.tensor(lam)).numpy()
    start_deg, end_deg = np.degrees(traj[0]), np.degrees(traj[-1])
    err = float(np.degrees(np.sqrt(((traj - tgt.numpy()) ** 2).mean())))
    print(f"trained coupling: motility={mot:.3f}, morphogen length scale={lam:.3f}")
    print(f"eye angle: {start_deg:.0f} deg (lateral) -> {end_deg:.0f} deg (frontal); "
          f"trajectory RMSE vs target = {err:.1f} deg")
    print("  the migration EMERGED from the field on the growing head (not imposed), and the "
          "field->shape coupling was fit to the target medialization")

    _figure(traj, tgt.numpy(), hist, mot, lam, err)
    json.dump(dict(motility=mot, morphogen_lambda=lam, start_deg=float(start_deg),
                   end_deg=float(end_deg), traj_rmse_deg=err),
              open(OUT / "field_driven_eye.json", "w"), indent=2)
    print("\nsaved", OUT / "field_driven_eye.json")
    return err < 6.0


def _figure(traj, tgt, hist, mot, lam, err):
    fig = plt.figure(figsize=(18, 5))
    stages = [0, T // 3, 2 * T // 3, T]
    gs = fig.add_gridspec(1, len(stages) + 2)
    th = np.linspace(0, 2 * np.pi, 200)
    for i, s in enumerate(stages):
        ax = fig.add_subplot(gs[0, i])
        R = R0 * (1.0 + GROWTH * s / T)
        ax.plot(R * np.sin(th), R * np.cos(th), "0.6", lw=1)          # head outline
        ax.scatter([0], [R], c="tab:green", s=80, marker="*", zorder=3)  # frontal organizer
        phi = traj[s]
        for sgn in (+1, -1):                                          # bilateral eyes
            ax.scatter(sgn * R * np.sin(phi), R * np.cos(phi), c="tab:red", s=70, zorder=3)
        ax.set_title(f"stage {s}/{T}  eye {np.degrees(phi):.0f}$^\\circ$", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.text(0, R + 0.15, "organizer", fontsize=6, ha="center", color="tab:green")
    ax = fig.add_subplot(gs[0, len(stages)])
    tt = np.arange(T + 1)
    ax.plot(tt, np.degrees(tgt), "k--", lw=1.5, label="target medialization")
    ax.plot(tt, np.degrees(traj), "o-", ms=3, color="tab:red", label="field-driven (fit)")
    ax.set_xlabel("developmental step"); ax.set_ylabel("eye angle (deg from midline)")
    ax.set_title(f"(e) eye trajectory (RMSE {err:.1f}$^\\circ$)", fontsize=9); ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, len(stages) + 1])
    ax.plot([h[0] for h in hist], [h[1] for h in hist], "o-", ms=3, color="tab:blue")
    ax.set_xlabel("epoch"); ax.set_ylabel("trajectory loss")
    ax.set_title(f"(f) coupling trained\nmotility={mot:.2f}, $\\lambda$={lam:.2f}", fontsize=9)
    fig.suptitle("Field-driven eye migration: the eye chemotaxes up a morphogen gradient toward the frontal "
                 "organizer while the head grows -- the migration EMERGES from the field (not imposed) and the "
                 "field$\\to$shape coupling is TRAINED to the target medialization", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "field_driven_eye.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "field_driven_eye.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'FIELD-DRIVEN (eye migrates by the field, coupling trained)' if ok else 'CHECK'}")
