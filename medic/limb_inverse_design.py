#!/usr/bin/env python3
"""
Inverse design of a limb: backprop a target shape to its bioelectric set-point.
===============================================================================

This instantiates Requirement 2 of the mammalian-limb outlook in Paper #5: a
mouse keeps the limb cascades (the outer perceptron) but has lost the inner-loop
TARGET FIELD that a planarian reconstructs for free, so the target must be
supplied. The framework supplies it by INVERTING the differentiable forward
model: the inner perceptron grows a tissue field from a bioelectric set-point
field; because that growth is differentiable, we can backpropagate
||grown shape - target shape|| onto the set-point field and recover the field
whose grown attractor IS the target.

The forward model is a small differentiable NCA (the inner perceptron):
    V_0 = 0
    V_{t+1} = V_t + k_relax (theta - V_t) + k_gj * laplacian(V_t)      (T steps)
    rho     = sigmoid((V_T - v0) / tau)         # tissue density (cells present)
where theta is the per-cell bioelectric set-point (the outer perceptron's
output). Gap-junction diffusion makes the recovered field a smooth, coherent
attractor rather than pixel noise.

Two results:
  (1) INVERSE DESIGN. Optimize theta so the grown rho matches a five-digit paw
      silhouette. Recover the set-point field (= the missing inner-loop target).
  (2) "GUESS WHAT TO ADD." At an incomplete limb (palm + three digits), the
      gradient of the full-paw loss with respect to theta -- evaluated without
      taking a step -- localizes the field correction to exactly the two missing
      digits. This is the residual-corrected braces loop (Requirement 4): the
      deficit tells you what to add and where.

Run:
    cd cognimed && venv_win_new/Scripts/python.exe -m medic.limb_inverse_design
Output: limb_inverse_design.png  (+ console metrics)
"""
from __future__ import annotations

import numpy as np
import torch

H, W = 56, 72              # limb field grid (rows = proximo-distal, cols = AP)
T_GROW = 16                # inner-loop growth steps
K_RELAX, K_GJ = 0.5, 0.18  # relaxation + gap-junction coupling (inner perceptron)
V0, TAU = 0.5, 0.08        # tissue threshold + softness for the density readout
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Target silhouettes
# ---------------------------------------------------------------------------
def _ellipse(cr, cc, rr, rc):
    r = np.arange(H)[:, None]; c = np.arange(W)[None, :]
    return ((r - cr) / rr) ** 2 + ((c - cc) / rc) ** 2 <= 1.0


def paw_silhouette(digits=(0, 1, 2, 3, 4)):
    """A paw: a palm ellipse plus up to five distal digits (middle longest)."""
    m = _ellipse(0.30 * H, 0.50 * W, 0.16 * H, 0.30 * W)          # palm
    cols = np.linspace(0.22, 0.78, 5) * W
    lengths = np.array([0.50, 0.60, 0.66, 0.60, 0.50]) * H        # finger tip rows
    for k in digits:
        cc = cols[k]
        cr = 0.40 * H + 0.5 * (lengths[k] - 0.40 * H)
        rr = 0.5 * (lengths[k] - 0.30 * H) + 0.10 * H
        m = m | _ellipse(cr, cc, rr, 0.045 * W)
    return m.astype(np.float32)


# ---------------------------------------------------------------------------
# Forward model: the differentiable inner perceptron (NCA growth)
# ---------------------------------------------------------------------------
def laplacian(V):
    return (torch.roll(V, 1, 0) + torch.roll(V, -1, 0)
            + torch.roll(V, 1, 1) + torch.roll(V, -1, 1) - 4 * V)


def grow(theta):
    """Grow a tissue-density field from the bioelectric set-point field theta."""
    V = torch.zeros_like(theta)
    for _ in range(T_GROW):
        V = V + K_RELAX * (theta - V) + K_GJ * laplacian(V)
    return torch.sigmoid((V - V0) / TAU)


def fit_field(target, steps=400, reg=3e-3, lr=0.15):
    """Backprop the target shape onto the set-point field theta (the inverse)."""
    tgt = torch.tensor(target)
    theta = torch.zeros((H, W), requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        rho = grow(theta)
        loss = ((rho - tgt) ** 2).mean() + reg * (theta ** 2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        rho = grow(theta)
    return theta.detach(), rho.detach(), float(loss.detach())


def iou(rho, target, thr=0.5):
    a = (rho.numpy() > thr); b = (target > 0.5)
    return float((a & b).sum() / ((a | b).sum() + 1e-9))


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("INVERSE DESIGN OF A LIMB  --  backprop a target shape to its set-point")
    print("=" * 74)

    full = paw_silhouette((0, 1, 2, 3, 4))
    partial = paw_silhouette((0, 1, 2))                # palm + 3 digits (2 missing)

    # (1) inverse design: recover the set-point field for the full paw
    theta_full, rho_full, loss_full = fit_field(full)
    print(f"\n(1) Inverse design (5-digit paw): final loss={loss_full:.4f}, "
          f"IoU(grown,target)={iou(rho_full, full):.2f}")
    print("    -> recovered the bioelectric set-point field whose grown attractor is the paw.")

    # (2) "guess what to add": fit the incomplete limb, then read the gradient of
    #     the FULL-paw loss at that field WITHOUT stepping -> where to add tissue.
    theta_part, rho_part, _ = fit_field(partial)
    theta_g = theta_part.clone().requires_grad_(True)
    rho = grow(theta_g)
    deficit_loss = ((rho - torch.tensor(full)) ** 2).mean()
    deficit_loss.backward()
    correction = (-theta_g.grad).detach().numpy()      # ascent dir = what to add
    correction = np.clip(correction, 0, None)          # additive correction only

    # localization: how much of the correction sits in the two missing-digit columns
    missing = (paw_silhouette((3, 4)) - paw_silhouette((0, 1, 2)))
    missing = np.clip(missing, 0, None) > 0.5
    present = full > 0.5
    frac_missing = correction[missing].sum() / (correction.sum() + 1e-9)
    frac_area = missing.sum() / present.sum()
    print(f"\n(2) Guess what to add: {100*frac_missing:.0f}% of the field correction "
          f"falls in the two missing digits,")
    print(f"    which are only {100*frac_area:.0f}% of the paw area "
          f"-> {frac_missing/frac_area:.1f}x enrichment. The deficit localizes the fix.")

    _figure(full, rho_full.numpy(), theta_full.numpy(),
            partial, missing.astype(float), correction)
    return iou(rho_full, full) > 0.7 and frac_missing > 0.5


def _figure(full, grown, theta, partial, missing, correction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    def show(a, Z, title, cmap="viridis"):
        im = a.imshow(Z, origin="lower", aspect="auto", cmap=cmap)
        a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)

    show(ax[0, 0], full, "(1) target paw silhouette", "Greens")
    show(ax[0, 1], grown, "(1) grown from the recovered set-point\n(inner-loop NCA)", "Greens")
    show(ax[0, 2], theta, "(1) recovered bioelectric set-point field\n(the inverse solution)", "magma")
    show(ax[1, 0], partial, "(2) incomplete limb (palm + 3 digits)", "Greens")
    show(ax[1, 1], missing, "(2) deficit vs the 5-digit target\n(the two missing digits)", "Reds")
    show(ax[1, 2], correction, "(2) inverse-gradient field correction\n= where to add tissue", "magma")

    fig.suptitle("Inverse design of a limb: the differentiable inner perceptron lets the outer perceptron "
                 "compute the missing target field.\nTop: recover the set-point whose grown attractor is the "
                 "paw. Bottom: the deficit's gradient localizes the fix to the missing digits (the braces loop).",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig("limb_inverse_design.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: limb_inverse_design.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'PASS' if ok else 'CHECK'}")
