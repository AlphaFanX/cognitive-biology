"""
The trained shared-kernel head: fate/migration/division from a learned MLP.
===========================================================================

In the assembled compiler (medic.glass_box_kernel_3d) the behavioural heads read
the settled field by NEAREST organ setpoint voltage. That is voltage-degenerate:
many body regions share a voltage, so the heart (-30 mV) is placed wherever -30 mV
occurs, not at the anterior-ventral cardiac field. This module replaces the read-off
with a TRAINED shared-kernel MLP -- one trunk (the shared kernel) feeding three
heads (Fate, Migration, Division) -- that is position AND voltage aware.

This is the first learned component, the step from the traced version-1 compiler
toward the trained network. The head learns the genome->fate map exactly as a cell
reads it: local position (the morphogen/AP-DV-LR frame) together with the settled
V_m instruction. Training signal = the anatomical organ atlas (which station is
which organ) -- the developmental ground truth the model is supposed to learn; the
FORWARD field is still not fit, only the readout is learned.

Result: the trained head recovers the anatomy the nearest-voltage head cannot --
in particular it places the heart at the anterior-ventral field rather than at any
-30 mV voxel.

Run: python -m medic.trained_kernel_head
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

from medic.nca_vertebrate_3d import build_static_target, run_clock
from medic.bioelectric_development import ORGAN_PREFERRED_VOLTAGE

OUT = Path("data/organ_cascade")
torch.manual_seed(0)

# anatomical organ atlas: name -> (AP, DV, LR) centre  (AP 0=anterior, DV 0=ventral)
ATLAS = {
    "brain":        (0.10, 0.85, 0.00),
    "thyroid":      (0.20, 0.30, 0.00),
    "heart":        (0.27, 0.40, 0.00),
    "lung_left":    (0.32, 0.30, -0.18),
    "lung_right":   (0.32, 0.30, 0.18),
    "liver":        (0.37, 0.20, 0.12),
    "pancreas":     (0.40, 0.26, -0.05),
    "gut":          (0.58, 0.18, 0.00),
    "kidney_left":  (0.75, 0.38, -0.15),
    "kidney_right": (0.75, 0.38, 0.15),
    "muscle":       (0.55, 0.64, 0.16),
}
ORGANS = list(ATLAS)
CENTRES = np.array([ATLAS[o] for o in ORGANS])
OV = np.array([ORGAN_PREFERRED_VOLTAGE[o] for o in ORGANS])


class SharedKernelHead(nn.Module):
    """One shared trunk (the kernel) -> three behavioural heads."""
    def __init__(self, in_dim=4, hidden=64, n=len(ORGANS)):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.fate = nn.Linear(hidden, n)          # organ logits
        self.migration = nn.Linear(hidden, 3)     # movement cue (AP,DV,LR)
        self.division = nn.Linear(hidden, 1)      # proliferation

    def forward(self, x):
        h = self.trunk(x)
        return self.fate(h), self.migration(h), torch.sigmoid(self.division(h))


def label_voxels(P):
    """Anatomical ground truth: nearest organ centre for every body voxel."""
    d = np.linalg.norm(P[:, None, :] - CENTRES[None, :, :], axis=2)
    return np.argmin(d, axis=1), d.min(axis=1)


def features(P, Vm):
    vmn = (Vm - np.nanmean(Vm)) / (np.nanstd(Vm) + 1e-9)
    return np.concatenate([P, vmn[:, None]], axis=1).astype(np.float32)


def train_head(P, Vm, epochs=600, seed=0):
    """Train the shared-kernel head on the anatomical atlas. Returns (net, hist)."""
    torch.manual_seed(seed)
    labels, dist = label_voxels(P)
    X = features(P, Vm)
    mig = (CENTRES[labels] - P); mig /= (np.linalg.norm(mig, axis=1, keepdims=True) + 1e-9)
    div = np.exp(-4.0 * dist).astype(np.float32)
    n = len(X); idx = np.random.RandomState(0).permutation(n); cut = int(0.8 * n)
    tr, va = idx[:cut], idx[cut:]
    Xt = torch.tensor(X); yt = torch.tensor(labels)
    migt = torch.tensor(mig.astype(np.float32)); divt = torch.tensor(div)
    net = SharedKernelHead(); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss(); mse = nn.MSELoss()
    tri = torch.tensor(tr); vai = torch.tensor(va); hist = []
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        f, m, dv = net(Xt[tri])
        loss = ce(f, yt[tri]) + 0.5 * mse(m, migt[tri]) + 0.5 * mse(dv[:, 0], divt[tri])
        loss.backward(); opt.step()
        if ep % 30 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                acc = (net(Xt[vai])[0].argmax(1) == yt[vai]).float().mean().item()
            hist.append((ep, float(loss.detach()), acc))
    return net, hist, labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("TRAINED SHARED-KERNEL HEAD (fate / migration / division)\n")

    # body + settled field (the assembled compiler's stages 1-4)
    Vstar, mask, AP, DV, LR, _ = build_static_target()
    V = run_clock()["V"]
    P = np.stack([AP[mask], DV[mask], LR[mask]], axis=1)
    Vm = V[mask]
    print(f"body: {len(P)} voxels; {len(ORGANS)} organ atlas classes; features [AP,DV,LR,Vm]")

    net, hist, labels = train_head(P, Vm)
    val_acc = hist[-1][2]
    print(f"trained: val fate accuracy = {val_acc:.3f}")

    # ---- the comparison: trained head vs the nearest-voltage read-off ----
    Xt = torch.tensor(features(P, Vm))
    net.eval()
    with torch.no_grad():
        fate_trained = net(Xt)[0].argmax(1).numpy()
    fate_voltage = np.argmin(np.abs(Vm[:, None] - OV[None, :]), axis=1)
    acc_trained = float((fate_trained == labels).mean())
    acc_voltage = float((fate_voltage == labels).mean())
    print(f"anatomical accuracy over the whole body: trained {acc_trained:.3f}  "
          f"vs nearest-voltage {acc_voltage:.3f}")

    # heart specifically: where does each head put heart fate?
    hi = ORGANS.index("heart")
    ap_true = ATLAS["heart"][0]
    ap_tr = float(P[fate_trained == hi, 0].mean()) if (fate_trained == hi).any() else np.nan
    ap_v = float(P[fate_voltage == hi, 0].mean()) if (fate_voltage == hi).any() else np.nan
    print(f"heart mean AP: atlas {ap_true:.2f} | trained {ap_tr:.2f} | nearest-voltage {ap_v:.2f} "
          f"(nearest-voltage smears it posteriorly)")

    # apply at the primordium loci
    from medic.glass_box_kernel_3d import placement_wave    # lazy import (avoids a cycle)
    prim = placement_wave(mask)[0]
    pv = []
    for (a, b, c) in prim:
        if not np.isfinite(V[a, b, c]):
            continue
        xx = np.array([[AP[a, b, c], DV[a, b, c], LR[a, b, c],
                        (V[a, b, c] - np.nanmean(Vm)) / (np.nanstd(Vm) + 1e-9)]], np.float32)
        with torch.no_grad():
            fo, mo, dvo = net(torch.tensor(xx))
        pv.append(dict(ap=float(AP[a, b, c]), dv=float(DV[a, b, c]), lr=float(LR[a, b, c]),
                       fate=ORGANS[int(fo.argmax())], division=float(dvo.item())))
    placed = sorted(set(d["fate"] for d in pv))
    print(f"trained head at {len(pv)} primordia -> {len(placed)} organ types: {', '.join(placed)}")

    _figure(hist, P, labels, fate_trained, fate_voltage, acc_trained, acc_voltage, val_acc, ap_tr, ap_v, ap_true)
    json.dump(dict(val_fate_accuracy=val_acc, anatomical_acc_trained=acc_trained,
                   anatomical_acc_nearest_voltage=acc_voltage,
                   heart_ap_atlas=ap_true, heart_ap_trained=ap_tr, heart_ap_voltage=ap_v,
                   organs_placed_at_primordia=placed),
              open(OUT / "trained_kernel_head.json", "w"), indent=2)
    print("\nsaved", OUT / "trained_kernel_head.json")
    return acc_trained > acc_voltage + 0.1


def _figure(hist, P, labels, fate_tr, fate_v, acc_tr, acc_v, val_acc, ap_tr, ap_v, ap_true):
    ep = [h[0] for h in hist]; acc = [h[2] for h in hist]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    ax[0].plot(ep, acc, "o-", ms=3)
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("val fate accuracy")
    ax[0].set_title(f"(a) shared-kernel head training\nval accuracy {val_acc:.2f}", fontsize=9)
    ax[0].set_ylim(0, 1)

    def scat(a, fate, title):
        a.scatter(P[:, 0], P[:, 1], c=[cmap(f % 20) for f in fate], s=2, linewidths=0)
        for o, cc in ATLAS.items():
            a.scatter(cc[0], cc[1], c="k", s=12, marker="x")
        a.set_xlabel("AP (anterior->posterior)"); a.set_ylabel("DV (ventral->dorsal)")
        a.set_title(title, fontsize=9)
    scat(ax[1], fate_tr, f"(b) TRAINED head fate (position+V_m)\nanatomical acc {acc_tr:.2f}; heart AP {ap_tr:.2f}")
    scat(ax[2], fate_v, f"(c) nearest-voltage head (V_m only)\nanatomical acc {acc_v:.2f}; heart AP {ap_v:.2f} (smeared)")
    fig.suptitle("The trained shared-kernel head: one trunk feeding fate/migration/division, learned from the anatomical "
                 "atlas.\nPosition+V_m recovers the anatomy the voltage-only read-off cannot -- the heart returns to the "
                 "anterior-ventral field.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "trained_kernel_head.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "trained_kernel_head.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'TRAINED (beats nearest-voltage)' if ok else 'CHECK'}")
