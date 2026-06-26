"""
Organ Cascade — the SEQUENCE-grammar test (k-mer 'words').

The combinatorial bag-of-90-TF-motifs failed to discriminate organs. But that is
the weakest 'word' (a bag of letters, only 90 motifs). This asks the decisive
generalization of Miles's question: does the raw super-enhancer SEQUENCE encode
its organ AT ALL -- using the full k-mer vocabulary (any 'words', not just our
motifs)? A k-mer spectrum captures local sequence grammar beyond a curated motif
set. If even this fails, organ identity is genuinely NOT in the sequence -> it is
in accessibility (which is what a learned model / AlphaGenome predicts).
"""
from __future__ import annotations
import json
import numpy as np
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score

from medic.organ_cascade_wiring import WIRE_DIR, _load_seq_cache

K = 6
MIN_PER_ORGAN = 12


def canon_index():
    """map each k-mer to a canonical (revcomp-collapsed) index."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A"}
    kmers = ["".join(p) for p in product("ACGT", repeat=K)]
    idx = {}; canon = {}
    nxt = 0
    for km in kmers:
        rc = "".join(comp[b] for b in reversed(km))
        c = min(km, rc)
        if c not in canon:
            canon[c] = nxt; nxt += 1
        idx[km] = canon[c]
    return idx, nxt


def seq_kmer_vec(seq, idx, dim):
    v = np.zeros(dim)
    seq = seq.upper()
    for i in range(len(seq) - K + 1):
        km = seq[i:i + K]
        j = idx.get(km)
        if j is not None:
            v[j] += 1
    s = v.sum()
    return v / s if s else v


def load_seqs_labeled():
    cache = _load_seq_cache()
    ot = json.load(open(WIRE_DIR / "organ_targets.json"))
    reg2org = {}
    for o, d2 in ot.items():
        for tf, regions in d2.items():
            for (c, s, e) in regions:
                reg2org.setdefault((c, int(s), int(e)), set()).add(o)
    items = []
    for (c, s, e), orgs in reg2org.items():
        if len(orgs) != 1:
            continue
        key = f"{c}:{s}-{min(e, s+3000)}"
        seq = cache.get(key)
        if seq and seq.upper().count("N") / max(len(seq), 1) < 0.1:
            items.append((seq, next(iter(orgs))))
    return items


def cv(X, y, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    pred = cross_val_predict(clf, X, y, cv=skf)
    return (balanced_accuracy_score(y, pred), accuracy_score(y, pred),
            f1_score(y, pred, average="macro"))


def main():
    idx, dim = canon_index()
    items = load_seqs_labeled()
    y0 = np.array([o for _, o in items])
    organs, counts = np.unique(y0, return_counts=True)
    keep = set(organs[counts >= MIN_PER_ORGAN])
    items = [(sq, o) for sq, o in items if o in keep]
    X = np.array([seq_kmer_vec(sq, idx, dim) for sq, _ in items])
    y = np.array([o for _, o in items])
    organs = sorted(set(y)); n = len(organs)

    print("=" * 74)
    print(f"ORGAN CASCADE — sequence {K}-mer grammar test")
    print("=" * 74)
    print(f"organ-unique SEs with sequence: {len(y)}  organs: {n}  "
          f"feat dim {dim}  (chance balanced-acc {1/n:.2f})")
    bacc, acc, f1 = cv(X, y)
    print(f"{K}-MER SPECTRUM classifier: balanced-acc {bacc:.3f}  acc {acc:.3f}  "
          f"macroF1 {f1:.3f}")
    rng = np.random.default_rng(0)
    null = np.array([cv(X, rng.permutation(y), seed=1)[0] for _ in range(15)])
    p = (np.sum(null >= bacc) + 1) / (len(null) + 1)
    print(f"permutation null: mean {null.mean():.3f} (max {null.max():.3f})  p={p:.3f}")
    print("-" * 74)
    if bacc > 1.0 / n + 0.08 and p < 0.05:
        v = "raw sequence DOES carry organ signal -> grammar beyond our motif set"
    else:
        v = ("even full k-mer sequence grammar does NOT discriminate organs "
             "-> organ identity is NOT in the sequence; it is in ACCESSIBILITY")
    print("VERDICT:", v)
    json.dump({"k": K, "n_se": int(len(y)), "organs": organs,
               "balanced_acc": float(bacc), "acc": float(acc), "macro_f1": float(f1),
               "null_mean": float(null.mean()), "p_value": float(p), "verdict": v},
              open(WIRE_DIR / "kmer.json", "w"), indent=2)


if __name__ == "__main__":
    main()
