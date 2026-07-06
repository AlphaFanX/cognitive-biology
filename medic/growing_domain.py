"""
A growing, deforming domain trained to follow a shape trajectory (the Carnegie piece).
======================================================================================

The MOSTA/Silic training (medic.mosta_train, medic.silic_train) fit a FIELD on a
FIXED domain. Morphology atlases -- the Carnegie human embryo series -- supervise the
SHAPE, and the shape GROWS and DEFORMS over stages, which a fixed-domain model cannot
express. This is the moving-boundary piece: a differentiable domain that deforms over
developmental time, trained to follow a staged shape trajectory by a geometric loss.

The domain is a 2D tissue point set; a small deformation field (a neural field of
position and time) displaces every material point, so the domain elongates and flexes.
The target is a staged embryo silhouette trajectory (a straight axis that elongates and
curls into the cephalic-caudal flexure -- the real early-embryo shape change). We match
the deformed points to the staged targets with a Chamfer (bidirectional nearest-point)
loss and back-propagate to the deformation field. This is the same trajectory-inference
loss trained elsewhere against a field, now against a MOVING shape -- the mechanism a
morphology atlas (Carnegie) requires, and where MOSTA (a molecular field) does not reach.

HONEST SCOPE: 2D, a neural deformation field (not yet material division adding mass), and
a schematic parametric target (a real staged shape sequence would come from the Carnegie
3D reconstructions). It demonstrates the differentiable growing/deforming domain following
a shape trajectory -- the capability Carnegie's shape supervision needs.

Run: python -m medic.growing_domain
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/organ_cascade")
torch.manual_seed(0)


def silhouette(t, n=90):
    """Staged embryo silhouette: a tapered axis that elongates (length 1->~2.2) and
    curls into a flexure (bend angle 0 -> ~150 deg) as developmental time t goes 0->1."""
    s = np.linspace(0, 1, n)
    L = 1.0 + 1.3 * t
    bend = np.radians(150.0) * t
    ang = bend * s                                   # accumulating curvature -> C-shape
    dx = np.cos(ang - np.pi / 2) * (L / n)
    dy = np.sin(ang - np.pi / 2) * (L / n)
    cx = np.cumsum(dx); cy = np.cumsum(dy)
    cx -= cx.mean(); cy -= cy.mean()
    w = 0.12 * (1.0 - 0.6 * s)                        # taper: head wider than tail
    tx = np.gradient(cx); ty = np.gradient(cy)
    nn_ = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / nn_, tx / nn_                      # normals
    left = np.stack([cx + w * nx, cy + w * ny], 1)
    right = np.stack([cx - w * nx, cy - w * ny], 1)
    return np.concatenate([left, right], 0).astype(np.float32)


class Deform(nn.Module):
    """Neural deformation field: (x0, y0, t) -> displacement (dx, dy)."""
    def __init__(self, h=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                 nn.Linear(h, 2))

    def forward(self, p0, t):
        tt = torch.full((p0.shape[0], 1), float(t))
        return p0 + self.net(torch.cat([p0, tt], 1))


def chamfer(a, b):
    d = torch.cdist(a, b)                             # (na, nb)
    return d.min(1).values.mean() + d.min(0).values.mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stages = [0.0, 0.33, 0.66, 1.0]
    targets = {t: torch.tensor(silhouette(t)) for t in stages}
    P0 = torch.tensor(silhouette(0.0))               # the initial (straight) domain
    n = P0.shape[0]
    print(f"growing-domain trajectory training: {n} material points, {len(stages)} staged shape keyframes\n")

    net = Deform()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    hist = []
    for ep in range(1200):
        opt.zero_grad()
        loss = sum(chamfer(net(P0, t), targets[t]) for t in stages) / len(stages)
        loss.backward(); opt.step()
        if ep % 60 == 0 or ep == 1199:
            hist.append((ep, float(loss.detach())))
    # final per-stage Chamfer
    with torch.no_grad():
        finals = {t: net(P0, t).numpy() for t in stages}
        cds = {t: float(chamfer(net(P0, t), targets[t])) for t in stages}
    print(f"final mean Chamfer distance = {np.mean(list(cds.values())):.4f} (domain size ~2)")
    for t in stages:
        print(f"  stage t={t:.2f}: Chamfer {cds[t]:.4f}")

    _figure(P0.numpy(), targets, finals, stages, hist, cds)
    json.dump(dict(n_points=n, stages=stages, chamfer=cds,
                   mean_chamfer=float(np.mean(list(cds.values())))),
              open(OUT / "growing_domain.json", "w"), indent=2)
    print("\nsaved", OUT / "growing_domain.json")
    return np.mean(list(cds.values())) < 0.05


def _figure(P0, targets, finals, stages, hist, cds):
    fig = plt.figure(figsize=(18, 5))
    gs = fig.add_gridspec(1, len(stages) + 1)
    for i, t in enumerate(stages):
        ax = fig.add_subplot(gs[0, i])
        tg = targets[t].numpy()
        ax.scatter(tg[:, 0], tg[:, 1], s=8, c="0.7", label="Carnegie-like target")
        ax.scatter(finals[t][:, 0], finals[t][:, 1], s=6, c="tab:red", label="grown domain")
        if i == 0:
            ax.scatter(P0[:, 0], P0[:, 1], s=3, c="tab:blue", alpha=0.4, label="t=0 material")
        ax.set_title(f"stage t={t:.2f}  (Chamfer {cds[t]:.3f})", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")
    ax = fig.add_subplot(gs[0, len(stages)])
    ep = [h[0] for h in hist]; loss = [h[1] for h in hist]
    ax.plot(ep, loss, "o-", ms=3, color="tab:red")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean Chamfer loss")
    ax.set_title("trajectory training\n(deformation field -> staged shapes)", fontsize=9)
    fig.suptitle("A growing, deforming domain trained to follow a shape trajectory: a differentiable neural "
                 "deformation field displaces every material point to match the staged embryo silhouettes "
                 "(the moving-boundary mechanism a morphology atlas -- Carnegie -- requires)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "growing_domain.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "growing_domain.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'GROWN (domain follows the shape trajectory)' if ok else 'CHECK'}")
