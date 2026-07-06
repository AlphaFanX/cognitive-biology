"""
The electric-face frame + lateral inhibition: a symmetric eye pair, and cyclopia.
=================================================================================

The naive feedback loop (galvanotaxis + aggregation) failed by winner-take-all: pure attraction
collapses to one clump. The fix is the von Dassow--Odell recipe. Their 2000 segment-polarity paper
showed a network STABILIZES a prepattern (robustness from topology); their 2002 neurogenic paper
showed LATERAL INHIBITION does not create a pattern de novo but RESOLVES a prepattern into single,
spaced units. So the division of labour is: the frame (the electric-face low eigenmode) sets the
prepattern -- where, and how many; lateral inhibition resolves it into discrete, spaced organs.

We model it as a neural field on a symmetric head: an eye-competence activator a with SHORT-range
self-activation and LONG-range self-inhibition (the Mexican hat = lateral inhibition), on a
competence set by the electric-face frame. The frame's midline node splits the single eye field
into two lateral competence lobes. Three regimes, each a real outcome:

  (1) frame + lateral inhibition  -> a symmetric PAIR of spaced eye spots at the antinodes.
  (2) frame, NO lateral inhibition -> the field is not resolved: self-activation fills it into one
      merged blob (no discrete, spaced organs) -- lateral inhibition is what resolves.
  (3) weak-midline frame + lateral inhibition -> the field is not split; a single MEDIAN eye forms
      = CYCLOPIA (holoprosencephaly, the real defect of midline/frame failure).

So a normal bilateral pair needs BOTH: the frame (to make two) and lateral inhibition (to resolve
and space them). Missing either gives a single eye.

Run: python -m medic.electric_face_feedback
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("data/organ_cascade")
T_STEPS = 220


def head_domain(n=1000, seed=0):
    rng = np.random.RandomState(seed)
    P = []
    while len(P) < n:
        x, y = rng.uniform(-1, 1, 2)
        if (x / 1.0) ** 2 + (y / 1.05) ** 2 < 1:
            P.append((x, y))
    return np.array(P)


def graph(P, k=10):
    n = len(P)
    nbr = cKDTree(P).query(P, k=k + 1)[1][:, 1:]
    r = np.repeat(np.arange(n), k); c = nbr.ravel()
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    L = diags(np.asarray(A.sum(1)).ravel()) - A
    return nbr, A, L


def lateral_mode(P, L):
    vals, vecs = eigsh(L, k=6, which="SM")
    vecs = vecs[:, np.argsort(vals)]
    ilat = 1 + np.argmax([abs(np.corrcoef(vecs[:, i], P[:, 0])[0, 1]) for i in range(1, 6)])
    phi = np.abs(vecs[:, ilat]); return phi / (phi.max() + 1e-9)      # ~|ML|, node at the midline


def smooth(a, nbr, steps):
    for _ in range(steps):
        a = 0.5 * a + 0.5 * a[nbr].mean(1)
    return a


def competence(P, lat_norm, sigma):
    """Eye-competence prepattern: ONE compact central eye field, split at the midline by the frame's
    nodal line (lat~0 there) with strength sigma. Strong sigma notches it into two lateral lobes;
    weak sigma leaves it a single central field (-> one median eye, cyclopia)."""
    eye_field = np.exp(-((P[:, 0] / 0.62) ** 2 + ((P[:, 1] - 0.30) / 0.30) ** 2))
    midline = np.exp(-((lat_norm / 0.34) ** 2))                    # ridge at the frame's node
    return eye_field * (1.0 - sigma * midline)                     # strong sigma -> deep central notch


def run(nbr, comp, lateral_inhibition, seed=1):
    rng = np.random.RandomState(seed)
    a = 0.02 * rng.rand(len(comp))
    for _ in range(T_STEPS):
        exc = smooth(a, nbr, 2)                                      # short-range self-activation
        inh = smooth(a, nbr, 11) if lateral_inhibition else 0.0      # lateral inhibition (spacing)
        inp = 0.95 * comp + 1.05 * exc - (1.05 if lateral_inhibition else 0.0) * inh
        a = a + 0.4 * (1.0 / (1.0 + np.exp(-9.0 * (inp - 0.35))) - a)
    return a


def spots(a, A, P, thr=0.5):
    on = a > thr
    if on.sum() < 3:
        return 0, []
    sub = A[on][:, on]
    ncomp, lab = connected_components(sub, directed=False)
    idx = np.where(on)[0]
    cents = [P[idx[lab == c]].mean(0) for c in range(ncomp) if (lab == c).sum() >= 8]
    return len(cents), cents


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = head_domain()
    nbr, A, L = graph(P)
    lat = lateral_mode(P, L)

    a_pair = run(nbr, competence(P, lat, 1.2), lateral_inhibition=True)
    a_noLI = run(nbr, competence(P, lat, 1.2), lateral_inhibition=False)
    a_cyc = run(nbr, competence(P, lat, 0.15), lateral_inhibition=True)

    for tag, a in [("frame + lateral inhibition", a_pair),
                   ("frame, NO lateral inhibition", a_noLI),
                   ("weak midline frame + LI", a_cyc)]:
        n, cents = spots(a, A, P)
        xs = sorted(round(c[0], 2) for c in cents)
        on_frac = float((a > 0.5).mean())
        verdict = {2: "symmetric PAIR", 1: "single MEDIAN eye = CYCLOPIA"}.get(n, f"{n} / unresolved")
        if tag.endswith("NO lateral inhibition") and on_frac > 0.10:
            verdict = "UNRESOLVED (one merged blob)"
        print(f"  {tag:34s} spots={n} at ML {xs}  on-fraction {on_frac:.2f}  -> {verdict}")

    _figure(P, lat, competence(P, lat, 1.2), competence(P, lat, 0.15), a_pair, a_noLI, a_cyc, A)
    print("\nsaved", OUT / "electric_face_feedback.png")


def _figure(P, lat, comp_s, comp_w, a_pair, a_noLI, a_cyc, A):
    fig, ax = plt.subplots(1, 4, figsize=(19, 5))

    def dots(ax, spots_cents):
        for c in spots_cents:
            ax.scatter([c[0]], [c[1]], s=140, facecolor="none", edgecolor="k", linewidths=1.4)

    a = ax[0]; a.scatter(P[:, 0], P[:, 1], c=comp_s, cmap="viridis", s=8, linewidths=0)
    a.axvline(0, color="w", ls=":", lw=1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title("(a) electric-face frame: eye competence\nsplit at the midline node", fontsize=9.5)

    a = ax[1]; a.scatter(P[:, 0], P[:, 1], c=a_noLI, cmap="RdPu", vmin=0, vmax=1, s=8, linewidths=0)
    a.axvline(0, color="k", ls=":", lw=1); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title("(b) NO lateral inhibition:\nunresolved (one merged blob)", fontsize=9.5)

    a = ax[2]; a.scatter(P[:, 0], P[:, 1], c=a_pair, cmap="RdPu", vmin=0, vmax=1, s=8, linewidths=0)
    dots(a, spots(a_pair, A, P)[1]); a.axvline(0, color="k", ls=":", lw=1)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title("(c) frame + lateral inhibition:\nsymmetric eye PAIR", fontsize=9.5)

    a = ax[3]; a.scatter(P[:, 0], P[:, 1], c=a_cyc, cmap="RdPu", vmin=0, vmax=1, s=8, linewidths=0)
    dots(a, spots(a_cyc, A, P)[1]); a.axvline(0, color="k", ls=":", lw=1)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title("(d) weak midline frame + LI:\nsingle MEDIAN eye = CYCLOPIA", fontsize=9.5)

    fig.suptitle("The electric-face frame + lateral inhibition (the von Dassow--Odell recipe). The frame sets a split "
                 "eye-competence prepattern (a); lateral inhibition RESOLVES it into a symmetric spaced pair (c), where "
                 "without it the field does not resolve (b); a weak midline frame is not split, so one median eye "
                 "forms---cyclopia (d). A bilateral pair needs BOTH the frame and lateral inhibition.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    for p in (OUT / "electric_face_feedback.png", Path("electric_face_feedback.png")):
        fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
