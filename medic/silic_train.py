"""
Training the model to FOLLOW the Silic atlas (the first end-to-end training).
=============================================================================

We already VALIDATE the bioelectric model against the Silic zebrafish atlas at
every stage (medic.silic_validation): a fixed classifier turns each tissue's Vmem
into a polarity and scores it against the atlas label. That per-stage discrepancy
is a loss. This turns the validation into TRAINING: the atlas stages are keyframes,
the model is a differentiable NCA that settles a per-tissue Vm field through
developmental time, and we backprop the multi-stage discrepancy to fit the model's
FREE parameters -- the per-tissue set-points and the two NCA rates (k_relax, k_gj).

This is the trajectory-inference framing made concrete, and it exploits the two
properties the framework has: the dynamics are differentiable, and -- because the
encoder is frozen and every coefficient genome-traced -- there are only a handful
of free parameters, so the sparse atlas keyframes suffice. It is literally the
von Dassow-Odell parameter search turned into gradient descent against the atlas.

Demonstration: start the set-points OFF (all at -50 mV, i.e. every tissue neutral)
and train only against the Silic labels. If the atlas can train the model, the loss
falls, the per-stage polarity accuracy rises to the hand-set baseline, and the
recovered set-points match the biology (neurons hyperpolarized, muscle strongly so,
epithelia depolarized) -- learned from the atlas, not put in by hand.

HONEST SCOPE: a reduced tissue-node NCA (mean-field gap-junction coupling) and the
coarse 3-way polarity labels, not a full spatial field or continuous voltage target;
it demonstrates the training PRINCIPLE. The spatial-field and morphology (Carnegie)
versions need the differentiable spatial NCA and the growing-domain mesh.

Run: python -m medic.silic_train
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.silic_validation import (ALIAS, TRANSIENT_TISSUES, UNMAPPED_TISSUES,
                                    HYPER_MAX, DEPOL_MIN)
from medic.zebrafish_bioelectric import DEVELOPMENTAL_VOLTAGE_ATLAS, TISSUE_VMEM_ESTIMATES

OUT = Path("data/organ_cascade")
V_ZYGOTE = -70.0
torch.manual_seed(0)


def build_targets():
    """From the atlas: ordered stages, the tissue set, and per-stage (tissue_idx, label)."""
    tissues, stage_items = [], []
    idx = {}
    stages = sorted(DEVELOPMENTAL_VOLTAGE_ATLAS, key=lambda s: s.hpf)
    for s in stages:
        items = []
        for tkey, label in s.tissue_polarity.items():
            if tkey in UNMAPPED_TISSUES or tkey in TRANSIENT_TISSUES:
                continue
            vkey = ALIAS.get(tkey, tkey)
            if vkey not in TISSUE_VMEM_ESTIMATES:
                continue
            if vkey not in idx:
                idx[vkey] = len(tissues); tissues.append(vkey)
            lab = "hyper" if "hyper" in label.lower() else ("depol" if "depol" in label.lower() else "neutral")
            items.append((idx[vkey], lab))
        if items:
            stage_items.append((s.stage_name, s.hpf, items))
    return tissues, stage_items


def polarity_loss(v, lab):
    """Differentiable margin loss reproducing the fixed 3-way classifier."""
    if lab == "hyper":            # want v <= -50 (margin -52)
        return torch.relu(v - (-52.0))
    if lab == "depol":            # want v >= -40 (margin -38)
        return torch.relu((-38.0) - v)
    return torch.relu(v - (-41.0)) + torch.relu((-49.0) - v)   # neutral: inside [-49,-41]


def classify(v):
    return "hyper" if v <= HYPER_MAX else ("depol" if v >= DEPOL_MIN else "neutral")


def rollout(theta, k_relax, k_gj, n_tissue, T=60):
    """NCA settle of the per-tissue Vm from the zygote base toward theta with mean-field
    gap-junction coupling; return V at each of the len-checkpoints (one per stage)."""
    V = torch.full((n_tissue,), V_ZYGOTE)
    snaps = []
    for t in range(T):
        V = V + k_relax * (theta - V) + k_gj * (V.mean() - V)
        snaps.append(V)
    return snaps


def accuracy(theta, stage_items):
    v = theta.detach().numpy()
    ok = tot = 0
    for _, _, items in stage_items:
        for ti, lab in items:
            tot += 1; ok += (classify(v[ti]) == lab)
    return ok / max(tot, 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tissues, stage_items = build_targets()
    n = len(tissues)
    print(f"Silic atlas: {len(stage_items)} scored stages, {n} mapped tissues, "
          f"{sum(len(it) for _,_,it in stage_items)} (stage,tissue) keyframes\n")

    # FREE parameters, initialised OFF: every set-point at -50 mV (all-neutral)
    theta = torch.full((n,), -50.0, requires_grad=True)
    p_relax = torch.tensor(0.0, requires_grad=True)   # -> k_relax via sigmoid
    p_gj = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([theta, p_relax, p_gj], lr=0.15)

    # stage k evaluated at rollout checkpoint round((k+1)/K * T)
    K = len(stage_items); T = 60
    ck = [min(T - 1, int(round((k + 1) / K * T)) - 1) for k in range(K)]

    acc0 = accuracy(theta, stage_items)
    hist = []
    for ep in range(400):
        opt.zero_grad()
        k_relax = torch.sigmoid(p_relax) * 0.40 + 0.05
        k_gj = torch.sigmoid(p_gj) * 0.20
        snaps = rollout(theta, k_relax, k_gj, n, T)
        loss = 0.0
        for k, (_, _, items) in enumerate(stage_items):
            Vk = snaps[ck[k]]
            for ti, lab in items:
                loss = loss + polarity_loss(Vk[ti], lab)
        loss = loss / sum(len(it) for _, _, it in stage_items)
        loss.backward(); opt.step()
        if ep % 20 == 0 or ep == 399:
            hist.append((ep, float(loss.detach()), accuracy(theta, stage_items)))
    accF = hist[-1][2]
    with torch.no_grad():
        kr = float(torch.sigmoid(p_relax) * 0.40 + 0.05)
        kg = float(torch.sigmoid(p_gj) * 0.20)

    print(f"Silic polarity accuracy: {acc0:.2f} (off-init, all-neutral) -> {accF:.2f} (trained)")
    print(f"learned NCA rates: k_relax={kr:.3f}, k_gj={kg:.3f}")
    print("\nrecovered set-points vs the biological reference (learned from the atlas):")
    print(f"  {'tissue':22s} {'learned':>8} {'reference':>10}")
    learned = theta.detach().numpy()
    for i, tk in enumerate(tissues):
        print(f"  {tk:22s} {learned[i]:+8.1f} {TISSUE_VMEM_ESTIMATES[tk]:+10.1f}")
    ref = np.array([TISSUE_VMEM_ESTIMATES[t] for t in tissues])
    corr = float(np.corrcoef(learned, ref)[0, 1])
    print(f"\n  corr(learned set-points, biological reference) = {corr:.2f}")

    _figure(hist, tissues, learned, ref, acc0, accF, corr)
    json.dump(dict(n_tissues=n, n_keyframes=sum(len(it) for _, _, it in stage_items),
                   acc_init=acc0, acc_trained=accF, k_relax=kr, k_gj=kg,
                   setpoint_corr_vs_reference=corr,
                   learned={t: float(learned[i]) for i, t in enumerate(tissues)}),
              open(OUT / "silic_train.json", "w"), indent=2)
    print("\nsaved", OUT / "silic_train.json")
    return accF > acc0 + 0.2 and corr > 0.6


def _figure(hist, tissues, learned, ref, acc0, accF, corr):
    ep = [h[0] for h in hist]; loss = [h[1] for h in hist]; acc = [h[2] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    ax[0].plot(ep, loss, "o-", ms=3, color="tab:red")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("multi-stage atlas loss")
    ax[0].set_title("(a) training against the Silic atlas\n(backprop through the NCA rollout)", fontsize=9)
    ax[1].plot(ep, acc, "o-", ms=3, color="tab:green")
    ax[1].axhline(acc0, ls="--", c="0.5", lw=1)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("Silic polarity accuracy"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title(f"(b) atlas accuracy learned\n{acc0:.2f} (off-init) -> {accF:.2f}", fontsize=9)
    ax[2].scatter(ref, learned, s=45)
    lim = [-90, -25]; ax[2].plot(lim, lim, "k--", lw=0.6, alpha=0.5)
    ax[2].set_xlabel("biological reference set-point (mV)")
    ax[2].set_ylabel("set-point learned from the atlas (mV)")
    ax[2].set_title(f"(c) set-points recovered from the atlas\ncorr {corr:.2f}", fontsize=9)
    fig.suptitle("Training the model to follow the Silic atlas: the per-stage polarity validation becomes a loss, "
                 "backpropagated through the NCA rollout to fit the free parameters (set-points, k_relax, k_gj) "
                 "from an off-initialisation", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "silic_train.png", dpi=140, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "silic_train.png")


if __name__ == "__main__":
    ok = main()
    print(f"\nRESULT: {'TRAINED (atlas fits the model)' if ok else 'CHECK'}")
