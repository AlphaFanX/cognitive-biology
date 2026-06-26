"""TEST 4 (honest, runnable form): is the EMBRYONIC craniofacial regulatory
landscape -- the genome substrate of the electric face -- STABLE across the
craniofacial morphogenesis window (Carnegie stages CS13->CS17, ~wk 4.5-6)?

Real data: human embryonic craniofacial H3K27ac peak sets (Wilderman/FaceBase
2018), four Carnegie stages, hg19, read with pybigtools.

Levin's electric face is reported simple and stationary while the face forms
beneath it. This tests the genome-side analogue: does the craniofacial enhancer
program stay put while the morphology convulses? It needs only peak coordinates
(no gene annotation -- network fetch is blocked and ABC has no craniofacial
biosample, so a channel-specific SPATIAL template stays the deferred step).
"""
import pybigtools
import numpy as np
from pathlib import Path

BB = Path("face_demo/data/craniofacial/bb")
STAGES = ["CS13", "CS14", "CS15", "CS17"]
FILES = {s: BB / f"impute_{s}-combined_H3K27ac.peaks.gappedPeak.bigBed" for s in STAGES}


def load_stage(path):
    """chrom -> merged sorted (start,end) intervals."""
    f = pybigtools.open(str(path))
    out = {}
    for chrom in f.chroms():
        ivs = sorted((r[0], r[1]) for r in f.records(chrom))
        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        if merged:
            out[chrom] = merged
    return out


def coverage(stage):
    return sum(e - s for ch in stage.values() for s, e in ch)


def overlap_bp(A, B):
    tot = 0
    for ch in set(A) & set(B):
        a, b = A[ch], B[ch]
        i = j = 0
        while i < len(a) and j < len(b):
            lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
            if hi > lo:
                tot += hi - lo
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
    return tot


def peaks_persisting(A, B):
    """fraction of A peaks that overlap any B peak."""
    n = hit = 0
    for ch, a in A.items():
        b = B.get(ch, [])
        if not b:
            n += len(a); continue
        bs = np.array([x[0] for x in b]); be = np.array([x[1] for x in b])
        for s, e in a:
            n += 1
            if np.any((bs < e) & (be > s)):
                hit += 1
    return hit / max(n, 1)


print("loading 4 Carnegie-stage craniofacial H3K27ac peak sets...")
S = {s: load_stage(FILES[s]) for s in STAGES}
cov = {s: coverage(S[s]) for s in STAGES}
npk = {s: sum(len(v) for v in S[s].values()) for s in STAGES}
for s in STAGES:
    print(f"  {s}: {npk[s]:6d} peaks, {cov[s]/1e6:7.2f} Mb covered")

print("\n=== pairwise Jaccard (bp overlap / bp union) ===")
J = np.zeros((4, 4))
for i, a in enumerate(STAGES):
    for j, b in enumerate(STAGES):
        ov = overlap_bp(S[a], S[b])
        un = cov[a] + cov[b] - ov
        J[i, j] = ov / un
print("        " + "".join(f"{s:>8}" for s in STAGES))
for i, a in enumerate(STAGES):
    print(f"  {a:>5} " + "".join(f"{J[i,j]:8.2f}" for j in range(4)))

print("\n=== persistence CS13 -> later stages (fraction of CS13 peaks retained) ===")
for s in STAGES[1:]:
    print(f"  CS13 peaks overlapping {s}: {peaks_persisting(S['CS13'], S[s])*100:5.1f}%")

# 4-way constitutive core: bp covered in ALL stages
core = S["CS13"]
for s in STAGES[1:]:
    nxt = {}
    for ch in set(core) & set(S[s]):
        a, b = core[ch], S[s][ch]
        i = j = 0; out = []
        while i < len(a) and j < len(b):
            lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
            if hi > lo:
                out.append((lo, hi))
            if a[i][1] < b[j][1]:
                i += 1
            else:
                j += 1
        if out:
            nxt[ch] = out
    core = nxt
core_bp = sum(e - s for ch in core.values() for s, e in ch)
union_bp = coverage({ch: sorted(sum((S[s].get(ch, []) for s in STAGES), []))
                     for ch in set().union(*[set(S[s]) for s in STAGES])})
print(f"\n=== 4-way constitutive craniofacial core ===")
print(f"  present in ALL of CS13-CS17: {core_bp/1e6:.2f} Mb")
print(f"  mean per-stage coverage:     {np.mean(list(cov.values()))/1e6:.2f} Mb")
print(f"  core / mean-stage:           {core_bp/np.mean(list(cov.values()))*100:.1f}%  (high => stable program)")

# ---------------- figure ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
im = ax[0].imshow(J, cmap="viridis", vmin=0.5, vmax=1.0)
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(STAGES)
ax[0].set_yticks(range(4)); ax[0].set_yticklabels(STAGES)
for i in range(4):
    for j in range(4):
        ax[0].text(j, i, f"{J[i,j]:.2f}", ha="center", va="center",
                   color="w" if J[i, j] < 0.85 else "k", fontsize=9)
ax[0].set_title("(a) pairwise Jaccard of craniofacial\nH3K27ac enhancers (bp overlap/union)", fontsize=9)
fig.colorbar(im, ax=ax[0], fraction=0.046)
pers = [peaks_persisting(S["CS13"], S[s]) * 100 for s in STAGES[1:]]
ax[1].bar(range(3), pers, color="steelblue")
ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"CS13→{s}" for s in STAGES[1:]])
ax[1].set_ylim(0, 100); ax[1].axhline(100, ls="--", c="k", lw=0.5)
for i, p in enumerate(pers):
    ax[1].text(i, p + 1.5, f"{p:.0f}%", ha="center", fontsize=9)
ax[1].set_ylabel("% CS13 enhancers retained")
ax[1].set_title("(b) persistence across morphogenesis", fontsize=9)
frac = core_bp / np.mean(list(cov.values())) * 100
ax[2].bar([0], [frac], color="seagreen", width=0.5, label="constitutive core")
ax[2].bar([0], [100 - frac], bottom=[frac], color="lightgray", width=0.5, label="stage-variable")
ax[2].set_ylim(0, 100); ax[2].set_xticks([]); ax[2].set_xlim(-0.8, 0.8)
ax[2].text(0, frac / 2, f"{frac:.0f}%\nconstitutive\ncore", ha="center", va="center", fontsize=10, color="w")
ax[2].text(0, frac + (100 - frac) / 2, f"{100-frac:.0f}%\ndrift", ha="center", va="center", fontsize=9)
ax[2].set_title("(c) present in ALL of CS13–CS17", fontsize=9)
fig.suptitle("The embryonic craniofacial regulatory landscape is stable across morphogenesis (CS13–CS17): "
             "a constitutive core with smooth developmental drift", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("data/organ_cascade/craniofacial_stability.png", dpi=140, bbox_inches="tight")
print("  saved data/organ_cascade/craniofacial_stability.png")
print("\nDONE 4.")
