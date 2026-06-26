"""Bolstering tests for 'the electric template comes from ABC'.
Tests 1-3 (share the forward-Goldman machinery):
  (2) FORWARD UN-ANCHORED: compute organ voltages from ABC WITHOUT solving g_K to
      the Levin targets; test whether the germ-layer ordering (ecto hyperpolarised ->
      endo depolarised) survives without fitting.
  (1) SHUFFLE NULL: permute ABC channel rows across organs; is the real ordering
      significant vs random channel->organ assignment?
  (3) CHANNEL-CLASS NECESSITY: ablate Na/K/Ca/Cl one at a time; which family is
      required for the correct ordering?
Honest: g_GJ does not enter the resting Goldman voltage (it is coupling, not a
resting current), so ablating GJ leaves per-organ V unchanged -- reported, not hidden.
"""
import numpy as np
from scipy.stats import spearmanr
from medic.bioelectric_development import (
    _ABC_ION_CHANNEL_ACTIVITY as ABC, E_NA, E_K, E_CA, E_CL)

GERM = {"brain": 0, "heart": 1, "kidney": 1, "muscle": 1,          # ecto=0, meso=1, endo=2
        "liver": 2, "pancreas": 2, "lung": 2, "gut": 2, "thyroid": 2}
ORGANS = list(ABC.keys())
rng = np.random.default_rng(20260625)


def forward_voltage(acts, drop=None):
    """ABC activity -> forward Goldman V_rest, NO g_K anchoring. drop=channel idx 0..3 zeroes it."""
    na, k, ca, cl, gj = acts
    total = na + k + ca + cl
    f = np.array([na, k, ca, cl]) / total
    g_budget = float(np.clip(2.5 * total / 1100.0, 1.0, 5.0))
    g = g_budget * f                              # g_Na,g_K,g_Ca,g_Cl  (forward, unfitted)
    gNa, gK, gCa, gCl = g
    gCa_rest = gCa * 0.10
    cond = np.array([gNa, gK, gCa_rest, gCl], float)
    E = np.array([E_NA, E_K, E_CA, E_CL], float)
    if drop is not None:
        cond[drop] = 0.0
    return float((cond * E).sum() / (cond.sum() + 1e-12))


def germ_order_score(volts):
    """Spearman(V, germ-rank): positive => ecto hyperpol -> endo depol (correct)."""
    ranks = np.array([GERM[o] for o in ORGANS])
    v = np.array([volts[o] for o in ORGANS])
    return float(spearmanr(v, ranks).correlation)


# ---------- TEST 2: forward un-anchored ----------
volts = {o: forward_voltage(ABC[o]) for o in ORGANS}
real_score = germ_order_score(volts)
print("=== TEST 2: FORWARD UN-ANCHORED organ voltages (no g_K fit) ===")
for o in sorted(ORGANS, key=lambda x: volts[x]):
    lay = ["ecto", "meso", "endo"][GERM[o]]
    print(f"  {o:10s} {lay}  V_forward = {volts[o]:7.1f} mV")
layer_means = {L: np.mean([volts[o] for o in ORGANS if ["ecto","meso","endo"][GERM[o]] == L])
               for L in ["ecto", "meso", "endo"]}
print(f"  layer means: ecto {layer_means['ecto']:.1f}  meso {layer_means['meso']:.1f}  endo {layer_means['endo']:.1f}  (want ecto<meso<endo)")
print(f"  germ-layer ordering Spearman = {real_score:+.3f}  (positive = correct ecto->endo depolarisation)")

# ---------- TEST 1: shuffle null ----------
N = 20000
null = np.empty(N)
acts_arr = [ABC[o] for o in ORGANS]
for i in range(N):
    perm = rng.permutation(len(ORGANS))
    shuff = {ORGANS[j]: forward_voltage(acts_arr[perm[j]]) for j in range(len(ORGANS))}
    null[i] = germ_order_score(shuff)
p = (np.sum(null >= real_score) + 1) / (N + 1)
print("\n=== TEST 1: SHUFFLE NULL (permute ABC rows across organs) ===")
print(f"  real ordering score = {real_score:+.3f}")
print(f"  null mean {null.mean():+.3f}, sd {null.std():.3f}, 95th pct {np.percentile(null,95):+.3f}")
print(f"  p(real >= shuffled channel assignment) = {p:.4f}")

# ---------- TEST 3: channel-class necessity ----------
print("\n=== TEST 3: CHANNEL-CLASS NECESSITY (ablate one family, re-score) ===")
names = ["Na", "K", "Ca", "Cl"]
for d, nm in enumerate(names):
    vd = {o: forward_voltage(ABC[o], drop=d) for o in ORGANS}
    s = germ_order_score(vd)
    print(f"  drop {nm:3s}: ordering Spearman = {s:+.3f}   (delta vs real {s-real_score:+.3f})")
print("  drop GJ : n/a -- gap junctions are coupling, not a resting current; per-organ V unchanged.")
print("\nDONE 1-3.")
